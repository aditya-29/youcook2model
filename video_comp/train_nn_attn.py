import os
import json
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import random
from typing import List, Tuple, Dict
import clip
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm
import argparse
import logging
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoProcessor:
    """Handles video loading and frame extraction"""
    
    def __init__(self, target_fps: int = 5):
        self.target_fps = target_fps
        
    def load_and_downsample_video(self, video_path: str) -> List[np.ndarray]:
        """Load video and downsample to target FPS"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
            
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(original_fps / self.target_fps))
        
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                
            frame_count += 1
            
        cap.release()
        return frames

class VideoTextDataset(Dataset):
    """Dataset for video-text pairs with contrastive learning support"""
    
    def __init__(self, 
                 video_dir: str,
                 caption_map_path: str,
                 clip_model,
                 clip_preprocess,
                 target_fps: int = 5,
                 avg_frame_number: int = None):
        
        self.video_dir = Path(video_dir)
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.target_fps = target_fps
        self.video_processor = VideoProcessor(target_fps)
        
        # Load caption mapping
        with open(caption_map_path, 'r') as f:
            self.caption_map = json.load(f)
            
        # Get all video paths
        self.video_paths = [str(path) for path in self.video_dir.glob('*.mp4')]
        self.video_paths.extend([str(path) for path in self.video_dir.glob('*.avi')])
        self.video_paths.extend([str(path) for path in self.video_dir.glob('*.mov')])
        
        # Filter videos that have captions
        self.video_paths = [path for path in self.video_paths if path in self.caption_map]
        
        logger.info(f"Found {len(self.video_paths)} videos with captions")
        
        # Calculate frame statistics if not provided
        if avg_frame_number is None:
            self.avg_frame_number = self._calculate_avg_frames()
        else:
            self.avg_frame_number = avg_frame_number
            
        logger.info(f"Using average frame number: {self.avg_frame_number}")
        
    def _calculate_avg_frames(self) -> int:
        """Calculate min, max, and average frames across all videos"""
        frame_counts = []
        
        logger.info("Calculating frame statistics...")
        for video_path in tqdm(self.video_paths[:100]):  # Sample first 100 for speed
            try:
                frames = self.video_processor.load_and_downsample_video(video_path)
                frame_counts.append(len(frames))
            except Exception as e:
                logger.warning(f"Error processing {video_path}: {e}")
                continue
                
        if not frame_counts:
            raise ValueError("No valid videos found")
            
        min_frames = min(frame_counts)
        max_frames = max(frame_counts)
        avg_frames = int(np.mean(frame_counts))
        
        logger.info(f"Frame statistics - Min: {min_frames}, Max: {max_frames}, Avg: {avg_frames}")
        return avg_frames
    
    def _extract_frame_group(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Extract sequential frame group of avg_frame_number length"""
        if len(frames) <= self.avg_frame_number:
            # Repeat frames if video is too short
            while len(frames) < self.avg_frame_number:
                frames.extend(frames)
            return frames[:self.avg_frame_number]
        else:
            # Randomly select starting point for sequential frames
            start_idx = random.randint(0, len(frames) - self.avg_frame_number)
            return frames[start_idx:start_idx + self.avg_frame_number]
    
    def _encode_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Encode frames using CLIP image encoder"""
        frame_tensors = []
        
        for frame in frames:
            # Preprocess frame for CLIP
            frame_tensor = self.clip_preprocess(frame).unsqueeze(0)
            frame_tensors.append(frame_tensor)
            
        # Stack all frames
        frame_batch = torch.cat(frame_tensors, dim=0)
        
        # Encode frames
        with torch.no_grad():
            frame_features = self.clip_model.encode_image(frame_batch)
            
        return frame_features
    
    def _encode_text(self, caption: str) -> torch.Tensor:
        """Encode caption using CLIP text encoder"""
        text_tokens = clip.tokenize([caption], truncate=True)
        
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
            
        return text_features.squeeze(0)
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        caption = self.caption_map[video_path]
        
        try:
            # Load and process video
            frames = self.video_processor.load_and_downsample_video(video_path)
            frame_group = self._extract_frame_group(frames)
            
            # Encode frames and text
            frame_features = self._encode_frames(frame_group)
            text_features = self._encode_text(caption)
            
            return {
                'frame_features': frame_features,
                'text_features': text_features,
                'video_path': video_path,
                'caption': caption
            }
            
        except Exception as e:
            logger.warning(f"Error processing {video_path}: {e}")
            # Return a random valid sample instead
            return self.__getitem__(random.randint(0, len(self) - 1))

class ContrastiveDataLoader:
    """DataLoader wrapper that creates positive and negative pairs for contrastive learning"""
    
    def __init__(self, dataset, batch_size: int = 8, num_workers: int = 4):
        self.dataset = dataset
        self.batch_size = batch_size
        self.dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch):
        """Create positive and negative pairs"""
        batch_size = len(batch)
        
        all_frame_features = []
        all_text_features = []
        all_labels = []
        
        # Positive pairs (matching video-text pairs)
        for item in batch:
            all_frame_features.append(item['frame_features'])
            all_text_features.append(item['text_features'])
            all_labels.append(1)  # Positive label
        
        # Negative pairs (mismatched video-text pairs)
        for i in range(batch_size):
            # Randomly select a different text for each video
            neg_idx = random.choice([j for j in range(batch_size) if j != i])
            all_frame_features.append(batch[i]['frame_features'])
            all_text_features.append(batch[neg_idx]['text_features'])
            all_labels.append(0)  # Negative label
            
        return {
            'frame_features': torch.stack(all_frame_features),
            'text_features': torch.stack(all_text_features),
            'labels': torch.tensor(all_labels, dtype=torch.float32)
        }
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)

class AttentionVideoTextMatcher(nn.Module):
    """Attention-based neural network for video-text matching"""
    
    def __init__(self, 
                 frame_feature_dim: int = 512,
                 text_feature_dim: int = 512,
                 hidden_dim: int = 256,
                 num_attention_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.frame_feature_dim = frame_feature_dim
        self.text_feature_dim = text_feature_dim
        self.hidden_dim = hidden_dim
        
        # Frame sequence processing
        self.frame_projection = nn.Linear(frame_feature_dim, hidden_dim)
        self.frame_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Text processing
        self.text_projection = nn.Linear(text_feature_dim, hidden_dim)
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Layer normalization
        self.frame_norm = nn.LayerNorm(hidden_dim)
        self.text_norm = nn.LayerNorm(hidden_dim)
        self.final_norm = nn.LayerNorm(hidden_dim * 2)
        
    def forward(self, frame_features, text_features):
        batch_size, num_frames, frame_dim = frame_features.shape
        
        # Project frame features
        frame_projected = self.frame_projection(frame_features)  # [B, N, H]
        frame_projected = self.frame_norm(frame_projected)
        
        # Self-attention on frame sequence
        frame_attended, _ = self.frame_attention(
            frame_projected, frame_projected, frame_projected
        )  # [B, N, H]
        
        # Project text features
        text_projected = self.text_projection(text_features.unsqueeze(1))  # [B, 1, H]
        text_projected = self.text_norm(text_projected)
        
        # Cross-modal attention: text attends to frame sequence
        cross_attended, attention_weights = self.cross_attention(
            text_projected,  # Query: text
            frame_attended,  # Key: frames
            frame_attended   # Value: frames
        )  # [B, 1, H]
        
        # Global average pooling on frame sequence
        frame_pooled = torch.mean(frame_attended, dim=1)  # [B, H]
        text_pooled = cross_attended.squeeze(1)  # [B, H]
        
        # Concatenate features
        combined_features = torch.cat([frame_pooled, text_pooled], dim=1)  # [B, 2H]
        combined_features = self.final_norm(combined_features)
        
        # Classification
        output = self.classifier(combined_features)  # [B, 1]
        
        return output.squeeze(1), attention_weights

class ContrastiveLoss(nn.Module):
    """Contrastive loss for video-text matching"""
    
    def __init__(self, margin: float = 1.0, temperature: float = 0.1):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
        self.bce_loss = nn.BCELoss()
        
    def forward(self, predictions, labels):
        # Binary cross-entropy loss for matching/non-matching pairs
        return self.bce_loss(predictions, labels)

def create_model_and_optimizer(args):
    """Create model, optimizer, and loss function"""
    
    # Load CLIP model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()  # Keep CLIP frozen
    
    # Create dataset
    dataset = VideoTextDataset(
        video_dir=args.video_dir,
        caption_map_path=args.caption_map,
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        target_fps=args.target_fps
    )
    
    # Create contrastive dataloader
    dataloader = ContrastiveDataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create model
    model = AttentionVideoTextMatcher(
        frame_feature_dim=512,  # CLIP ViT-B/32 feature dim
        text_feature_dim=512,
        hidden_dim=args.hidden_dim,
        num_attention_heads=args.num_attention_heads,
        dropout=args.dropout
    )
    
    # Create optimizer and loss
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    criterion = ContrastiveLoss()
    
    return model, optimizer, criterion, dataloader, clip_model

def train_epoch(model, dataloader, optimizer, criterion, accelerator, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=not accelerator.is_local_main_process)
    
    for batch_idx, batch in enumerate(progress_bar):
        frame_features = batch['frame_features']
        text_features = batch['text_features']
        labels = batch['labels']
        
        # Forward pass
        predictions, attention_weights = model(frame_features, text_features)
        loss = criterion(predictions, labels)
        
        # Backward pass
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        
        # Calculate accuracy
        predicted_labels = (predictions > 0.5).float()
        correct_predictions += (predicted_labels == labels).sum().item()
        total_samples += labels.size(0)
        
        total_loss += loss.item()
        
        # Update progress bar
        if accelerator.is_local_main_process:
            accuracy = correct_predictions / total_samples
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}',
                'accuracy': f'{accuracy:.4f}'
            })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    
    return avg_loss, accuracy

def main():
    parser = argparse.ArgumentParser(description='Video-Text Contrastive Learning')
    parser.add_argument('--video_dir', type=str, default='/mnt/localssd/video_comp/raw_chunks',
                       help='Directory containing videos')
    parser.add_argument('--caption_map', type=str, default='/mnt/localssd/video_comp/caption_map.json',
                       help='Path to caption mapping JSON file')
    parser.add_argument('--target_fps', type=int, default=5,
                       help='Target FPS for video downsampling')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension for the model')
    parser.add_argument('--num_attention_heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='Directory to save model checkpoints')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Initialize accelerator
    accelerator = Accelerator()
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Create model, optimizer, and dataloader
    model, optimizer, criterion, dataloader, clip_model = create_model_and_optimizer(args)
    
    # Prepare everything with accelerator
    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )
    
    if accelerator.is_local_main_process:
        logger.info(f"Starting training with {accelerator.num_processes} processes")
        logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    best_accuracy = 0
    for epoch in range(1, args.num_epochs + 1):
        avg_loss, accuracy = train_epoch(
            model, dataloader, optimizer, criterion, accelerator, epoch
        )
        
        if accelerator.is_local_main_process:
            logger.info(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")
            
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                save_path = os.path.join(args.save_dir, 'best_model.pth')
                accelerator.save(model.state_dict(), save_path)
                logger.info(f"Saved best model with accuracy: {accuracy:.4f}")
    
    if accelerator.is_local_main_process:
        logger.info(f"Training completed! Best accuracy: {best_accuracy:.4f}")

if __name__ == "__main__":
    main()