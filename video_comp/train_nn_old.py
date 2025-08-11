#!/usr/bin/env python
# train_clip_matching_optimized.py

import os
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple
import pickle

# Fix threading issues BEFORE importing heavy libraries
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'

import cv2
import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from transformers import CLIPModel, AutoProcessor
from accelerate import Accelerator
from tqdm import tqdm


# ---------------------------
# Config
# ---------------------------
class Cfg:
    captions_json = Path("/mnt/localssd/video_comp/chunk_videos/caption_map.json")
    embeddings_cache = Path("embeddings_cache.pkl")  # Cache embeddings to disk
    
    fps = 5                          # Target FPS for downsampling
    max_frames_per_video = None      # e.g. 256 to cap, or None for all sampled frames
    batch_size_embed = 128           # Increased batch size for embedding extraction
    batch_size_train = 512           # Increased batch size for training
    num_workers = 4                  # Conservative worker count to avoid threading issues
    num_epochs = 3
    lr = 1e-3
    model_ckpt = "openai/clip-vit-base-patch32"
    seed = 42
    
    # Performance optimizations
    use_fast_processor = False       # Disable fast processor due to compatibility issues
    use_mixed_precision = True       # Enable mixed precision training
    prefetch_factor = 4              # Prefetch batches
    persistent_workers = True        # Keep workers alive between epochs


# ---------------------------
# Utils
# ---------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def sample_frames_by_fps(cap: cv2.VideoCapture, target_fps: float) -> List[np.ndarray]:
    """
    Optimized frame sampling with better memory management.
    """
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(orig_fps / target_fps)))
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if (frame_count % stride) == 0:
            # Don't resize here - let CLIP processor handle it
            frames.append(frame)
            
        frame_count += 1
        
        # Early exit if we have enough frames
        if Cfg.max_frames_per_video and len(frames) >= Cfg.max_frames_per_video:
            break
    
    return frames


# ---------------------------
# CLIP embedding extraction (optimized)
# ---------------------------
@torch.inference_mode()
def compute_clip_image_embeddings(
    frames: List[np.ndarray],
    processor: AutoProcessor,
    clip: CLIPModel,
    device: torch.device,
    batch_size: int = 128,
) -> torch.Tensor:
    """
    Optimized image embedding computation with better batching.
    """
    if len(frames) == 0:
        return torch.empty(0, clip.config.projection_dim)
    
    embs = []
    # Convert all frames to PIL at once to reduce overhead
    pil_frames = [bgr_to_pil(f) for f in frames]
    
    for i in range(0, len(pil_frames), batch_size):
        batch_frames = pil_frames[i:i+batch_size]
        
        try:
            # Process batch with simplified parameters
            inputs = processor(
                images=batch_frames, 
                return_tensors="pt"
            ).to(device, non_blocking=True)
            
            # Use autocast for mixed precision
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                feats = clip.get_image_features(**inputs)
            
            # Normalize
            feats = feats / feats.norm(dim=-1, keepdim=True)
            embs.append(feats.detach().cpu().float())  # Convert back to float32 for storage
            
        except Exception as e:
            print(f"[WARN] Error processing image batch {i}-{i+len(batch_frames)}: {e}")
            # Skip this batch and continue
            continue
    
    if len(embs) == 0:
        return torch.empty(0, clip.config.projection_dim)
    
    return torch.cat(embs, dim=0)


@torch.inference_mode()
def compute_clip_text_embedding(
    caption: str,
    processor: AutoProcessor,
    clip: CLIPModel,
    device: torch.device,
) -> torch.Tensor:
    """
    Optimized text embedding computation.
    """
    inputs = processor(
        text=caption, 
        return_tensors="pt", 
        truncation=True
    ).to(device, non_blocking=True)
    
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        feats = clip.get_text_features(**inputs)[0]
    
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.detach().cpu().float()


def precompute_embeddings(
    caption_map: Dict[str, str],
    cfg: Cfg,
    processor: AutoProcessor,
    clip: CLIPModel,
    device: torch.device,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Precompute embeddings with caching support.
    """
    # Try to load from cache first
    if cfg.embeddings_cache.exists():
        print(f"[INFO] Loading embeddings from cache: {cfg.embeddings_cache}")
        with open(cfg.embeddings_cache, 'rb') as f:
            cached_data = pickle.load(f)
            return cached_data['image_embs'], cached_data['text_embs']
    
    print("[INFO] Computing embeddings (this may take a while, but will be cached)...")
    
    image_embs_list = []
    text_embs_list = []

    videos = list(caption_map.keys())
    
    # Process in batches for better GPU utilization
    for vpath in tqdm(videos, desc="Precomputing CLIP embeddings"):
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            print(f"[WARN] Could not open video: {vpath}")
            continue

        frames = sample_frames_by_fps(cap, cfg.fps)
        cap.release()

        if len(frames) == 0:
            print(f"[WARN] No frames sampled for: {vpath}")
            continue

        # Compute embeddings
        img_embs = compute_clip_image_embeddings(frames, processor, clip, device, cfg.batch_size_embed)
        txt_emb = compute_clip_text_embedding(caption_map[vpath], processor, clip, device).unsqueeze(0)

        image_embs_list.append(img_embs)
        text_embs_list.append(txt_emb)

    # Cache the results
    print(f"[INFO] Saving embeddings to cache: {cfg.embeddings_cache}")
    with open(cfg.embeddings_cache, 'wb') as f:
        pickle.dump({
            'image_embs': image_embs_list,
            'text_embs': text_embs_list
        }, f)

    return image_embs_list, text_embs_list


# ---------------------------
# Optimized Dataset
# ---------------------------
class PairDataset(Dataset):
    """
    Optimized dataset with pre-computed negative samples for faster training.
    """
    def __init__(self, image_embs_list: List[torch.Tensor], text_embs_list: List[torch.Tensor]):
        # Pre-compute positive pairs
        self.pos_pairs = []
        for img_embs, txt_emb in zip(image_embs_list, text_embs_list):
            txt_emb_squeezed = txt_emb.squeeze(0)
            for img_emb in img_embs:
                self.pos_pairs.append((img_emb, txt_emb_squeezed))

        # Build negative sampling pools
        if len(image_embs_list) == 0:
            raise ValueError("No embeddings found.")

        self.neg_img_pool = torch.cat(image_embs_list, dim=0)
        self.neg_txt_pool = torch.cat(text_embs_list, dim=0)

        self.dim = self.neg_img_pool.shape[1]
        self.N_pos = len(self.pos_pairs)
        
        # Pre-generate negative pairs for one epoch
        self._regenerate_negatives()

    def _regenerate_negatives(self):
        """Pre-generate negative pairs to avoid random sampling during training."""
        self.neg_pairs = []
        for _ in range(self.N_pos):
            i = random.randrange(self.neg_img_pool.shape[0])
            j = random.randrange(self.neg_txt_pool.shape[0])
            self.neg_pairs.append((self.neg_img_pool[i], self.neg_txt_pool[j]))

    def __len__(self):
        return self.N_pos * 2  # 50/50 pos/neg

    def __getitem__(self, idx: int):
        if (idx % 2) == 0:
            # Positive pair
            pidx = (idx // 2) % len(self.pos_pairs)
            img, txt = self.pos_pairs[pidx]
            y = 1.0
        else:
            # Negative pair
            nidx = ((idx - 1) // 2) % len(self.neg_pairs)
            img, txt = self.neg_pairs[nidx]
            y = 0.0

        # Concatenate embeddings
        x = torch.cat([img, txt], dim=-1)
        return x, torch.tensor(y, dtype=torch.float32)


# ---------------------------
# Optimized MLP classifier
# ---------------------------
class Matcher(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 1024, pdrop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),  # Added layer norm for stability
            nn.GELU(),
            nn.Dropout(pdrop),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(pdrop),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------
# Main Training Class
# ---------------------------
class TrainNN:
    def __init__(self):
        set_seed(Cfg.seed)
        
        # Use mixed precision if available
        mixed_precision = "fp16" if Cfg.use_mixed_precision and torch.cuda.is_available() else "no"
        self.accelerator = Accelerator(mixed_precision=mixed_precision)
        self.device = self.accelerator.device
        self.caption_map = None

    def _load_model(self):
        """Load CLIP model with optimizations."""
        try:
            # Try fast processor first, fallback to regular if it fails
            if Cfg.use_fast_processor:
                try:
                    processor = AutoProcessor.from_pretrained(
                        Cfg.model_ckpt, 
                        use_fast=True
                    )
                    print("[INFO] Using fast CLIP processor")
                except Exception as e:
                    print(f"[WARN] Fast processor failed ({e}), falling back to regular processor")
                    processor = AutoProcessor.from_pretrained(Cfg.model_ckpt)
            else:
                processor = AutoProcessor.from_pretrained(Cfg.model_ckpt)
                print("[INFO] Using regular CLIP processor")
            
            clip = CLIPModel.from_pretrained(Cfg.model_ckpt)
            clip.to(self.device)
            clip.eval()
            
            # Enable optimizations
            if hasattr(torch.backends.cudnn, 'benchmark'):
                torch.backends.cudnn.benchmark = True

            return clip, processor
        
        except Exception as e:
            print(f"[ERROR] Failed to load CLIP model: {e}")
            raise

    def _load_caption(self):
        """Load and validate caption mapping."""
        assert Cfg.captions_json.exists(), f"Missing captions file: {Cfg.captions_json}"
        self.caption_map: Dict[str, str] = json.loads(Cfg.captions_json.read_text(encoding="utf-8"))
        
        # Validate paths exist
        valid_paths = {p: c for p, c in self.caption_map.items() if Path(p).exists()}
        
        if len(valid_paths) != len(self.caption_map):
            missing = len(self.caption_map) - len(valid_paths)
            print(f"[WARN] {missing} videos not found, using {len(valid_paths)} videos")
            
        self.caption_map = valid_paths
        
        if self.accelerator.is_main_process:
            print(f"[INFO] Found {len(self.caption_map)} videos with captions.")

    def _build_dataset(self, img_list, txt_list):
        """Build optimized dataset and dataloader."""
        dataset = PairDataset(img_list, txt_list)
        in_dim = dataset.dim * 2

        train_loader = DataLoader(
            dataset,
            batch_size=Cfg.batch_size_train,
            shuffle=True,
            num_workers=Cfg.num_workers,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=Cfg.prefetch_factor,
            persistent_workers=Cfg.persistent_workers
        )

        return train_loader, in_dim

    def _train(self, train_loader):
        """Optimized training loop."""
        self.model.train()
        
        for epoch in range(Cfg.num_epochs):
            # Regenerate negative pairs each epoch for variety
            if hasattr(train_loader.dataset, '_regenerate_negatives'):
                train_loader.dataset._regenerate_negatives()
            
            running_loss = 0.0
            num_batches = 0
            
            prog = tqdm(
                train_loader, 
                disable=not self.accelerator.is_local_main_process, 
                desc=f"Epoch {epoch+1}/{Cfg.num_epochs}"
            )
            
            for batch_idx, (x, y) in enumerate(prog):
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)
                
                logits = self.model(x)
                loss = self.criterion(logits, y)

                self.accelerator.backward(loss)
                
                # Gradient clipping for stability
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()

                running_loss += loss.detach().item()
                num_batches += 1
                
                if batch_idx % 100 == 0:
                    avg_loss = running_loss / num_batches
                    prog.set_postfix(loss=f"{avg_loss:.4f}")

            if self.accelerator.is_main_process:
                avg_loss = running_loss / max(1, num_batches)
                print(f"[INFO] Epoch {epoch+1} avg loss: {avg_loss:.4f}")

        # Save model
        if self.accelerator.is_main_process:
            os.makedirs("checkpoints", exist_ok=True)
            unwrapped = self.accelerator.unwrap_model(self.model)
            torch.save({
                'model_state_dict': unwrapped.state_dict(),
                'config': {
                    'in_dim': unwrapped.net[0].in_features,
                    'hidden': 1024,
                    'pdrop': 0.1
                }
            }, "checkpoints/matcher.pt")
            print("[OK] Saved model to checkpoints/matcher.pt")

    def train(self):
        """Main training pipeline."""
        # Load CLIP model
        if self.accelerator.is_main_process:
            print(f"[INFO] Loading CLIP model: {Cfg.model_ckpt}")
        
        clip, processor = self._load_model()

        # Load captions
        self._load_caption()

        # Precompute embeddings (with caching)
        img_list, txt_list = precompute_embeddings(self.caption_map, Cfg, processor, clip, self.device)

        # Free CLIP memory
        del clip, processor
        torch.cuda.empty_cache()

        # Build dataset/dataloader
        train_loader, in_dim = self._build_dataset(img_list=img_list, txt_list=txt_list)

        # Initialize model, optimizer, loss
        self.model = Matcher(in_dim=in_dim, hidden=1024, pdrop=0.1)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=Cfg.lr,
            weight_decay=1e-4,  # Added weight decay
            eps=1e-8
        )
        self.criterion = nn.BCEWithLogitsLoss()

        # Prepare for distributed training
        self.model, self.optimizer, train_loader = self.accelerator.prepare(
            self.model, self.optimizer, train_loader
        )

        # Train
        if self.accelerator.is_main_process:
            print(f"[INFO] Starting training with {len(train_loader.dataset)} samples")
        
        self._train(train_loader=train_loader)


if __name__ == "__main__":
    trainer = TrainNN()
    trainer.train()