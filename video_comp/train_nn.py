#!/usr/bin/env python
# train_clip_matching.py

import os
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

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
    captions_json = Path("/mnt/local_ssd/video_comp/chunk_videos/captions_map.json")
    # If your JSON has a different name, point to it here.

    fps = 5                          # Target FPS for downsampling
    max_frames_per_video = None      # e.g. 256 to cap, or None for all sampled frames
    batch_size_embed = 64            # batch size for embedding extraction
    batch_size_train = 256           # batch size for dataloader (pairs)
    num_workers = 4
    num_epochs = 3
    lr = 1e-3
    model_ckpt = "openai/clip-vit-base-patch32"   # downloads automatically
    seed = 42


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
    Sequentially read frames and pick every Nth frame to approx target_fps.
    """
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(orig_fps / target_fps)))  # simple stride sampler
    frames = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if (i % stride) == 0:
            frames.append(frame)
        i += 1
    return frames


# ---------------------------
# CLIP embedding extraction
# ---------------------------
@torch.inference_mode()
def compute_clip_image_embeddings(
    frames: List[np.ndarray],
    processor: AutoProcessor,
    clip: CLIPModel,
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    """
    Returns tensor [N, D] on CPU with L2-normalized image embeddings.
    """
    embs = []
    for i in range(0, len(frames), batch_size):
        batch = [bgr_to_pil(f) for f in frames[i:i+batch_size]]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        feats = clip.get_image_features(**inputs)  # [B, D]
        feats = feats / feats.norm(dim=-1, keepdim=True)
        embs.append(feats.detach().cpu())
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
    Returns tensor [D] on CPU with L2-normalized text embedding.
    """
    inputs = processor(text=caption, return_tensors="pt", truncation=True).to(device)
    feats = clip.get_text_features(**inputs)[0]  # [D]
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.detach().cpu()


def precompute_embeddings(
    caption_map: Dict[str, str],
    cfg: Cfg,
    processor: AutoProcessor,
    clip: CLIPModel,
    device: torch.device,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    For each video:
      - sample frames at cfg.fps
      - compute image embeddings per frame
      - compute text embedding for its caption
    Returns:
      image_embs_list: list of [Ni, D] tensors (per-video)
      text_embs_list:  list of [1, D] tensors (per-video, one caption per video)
    """
    image_embs_list = []
    text_embs_list = []

    videos = list(caption_map.keys())
    for vpath in tqdm(videos, desc="Precomputing CLIP embeddings"):
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            print(f"[WARN] Could not open video: {vpath}")
            continue

        frames = sample_frames_by_fps(cap, cfg.fps)
        cap.release()

        if cfg.max_frames_per_video and len(frames) > cfg.max_frames_per_video:
            # Uniformly subsample to the cap
            idxs = np.linspace(0, len(frames)-1, cfg.max_frames_per_video).astype(int).tolist()
            frames = [frames[i] for i in idxs]

        if len(frames) == 0:
            print(f"[WARN] No frames sampled for: {vpath}")
            continue

        img_embs = compute_clip_image_embeddings(frames, processor, clip, device, cfg.batch_size_embed)  # [N, D]
        txt_emb = compute_clip_text_embedding(caption_map[vpath], processor, clip, device).unsqueeze(0) # [1, D]

        image_embs_list.append(img_embs)     # [Ni, D]
        text_embs_list.append(txt_emb)       # [1, D]

    return image_embs_list, text_embs_list


# ---------------------------
# Dataset (balanced pos/neg)
# ---------------------------
class PairDataset(Dataset):
    """
    Positives: (image_emb from video i, text_emb from same video i) → y=1
    Negatives: random image_emb from any video j and random text_emb from any (k), j/k random → y=0
    The dataset length is ~2x number of positive pairs (balanced).
    """
    def __init__(self, image_embs_list: List[torch.Tensor], text_embs_list: List[torch.Tensor]):
        # Flatten to per-frame positives
        self.pos_img: List[torch.Tensor] = []
        self.pos_txt: List[torch.Tensor] = []
        for img_embs, txt_emb in zip(image_embs_list, text_embs_list):
            # broadcast caption to all frames in that video
            self.pos_img.extend([e for e in img_embs])                  # list of [D]
            self.pos_txt.extend([txt_emb.squeeze(0) for _ in range(len(img_embs))])  # list of [D]

        # Pre-build pools for negative sampling
        # All image embeddings and all text embeddings (flattened)
        if len(image_embs_list) == 0:
            raise ValueError("No embeddings found. Check your captions.json and videos.")

        self.neg_img_pool = torch.cat(image_embs_list, dim=0)  # [M, D]
        self.neg_txt_pool = torch.cat(text_embs_list, dim=0)   # [V, D] (one per video)

        self.dim = self.neg_img_pool.shape[1]
        self.N_pos = len(self.pos_img)
        self.N_total = self.N_pos * 2  # 50/50 pos/neg

    def __len__(self):
        return self.N_total

    def _sample_negative(self) -> Tuple[torch.Tensor, torch.Tensor]:
        i = random.randrange(self.neg_img_pool.shape[0])
        j = random.randrange(self.neg_txt_pool.shape[0])
        return self.neg_img_pool[i], self.neg_txt_pool[j]

    def __getitem__(self, idx: int):
        if (idx % 2) == 0:
            # positive
            pidx = idx // 2
            img = self.pos_img[pidx]
            txt = self.pos_txt[pidx]
            y = 1.0
        else:
            # negative
            img, txt = self._sample_negative()
            y = 0.0

        # input = concat(image_emb, text_emb)
        x = torch.cat([img, txt], dim=-1)  # [2D]
        return x, torch.tensor(y, dtype=torch.float32)


# ---------------------------
# Simple MLP classifier
# ---------------------------
class Matcher(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 1024, pdrop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(pdrop),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # logits


# ---------------------------
# Main
# ---------------------------
class TrainNN:
    def __init__(self):
        set_seed(Cfg.seed)
        self.accelerator = Accelerator()
        self.__annotations__device = self.accelerator.device
        self.caption_map = None

    def _load_model(self):
        clip = CLIPModel.from_pretrained(Cfg.model_ckpt)
        processor = AutoProcessor.from_pretrained(Cfg.model_ckpt)
        clip.to(self.device)
        clip.eval() 

        return clip, processor
    
    def _load_caption(self):
        assert Cfg.captions_json.exists(), f"Missing captions file: {Cfg.captions_json}"
        self.caption_map: Dict[str, str] = json.loads(Cfg.captions_json.read_text(encoding="utf-8"))
        # Validate paths exist
        self.caption_map = {p: c for p, c in self.caption_map.items() if Path(p).exists()}
        if self.accelerator.is_main_process:
            print(f"[INFO] Found {len(self.caption_map)} videos with captions.")

    def _build_dataset(self, img_list, txt_list):
        dataset = PairDataset(img_list, txt_list)
        in_dim = dataset.dim * 2

        train_loader = DataLoader(
            dataset,
            batch_size=Cfg.batch_size_train,
            shuffle=True,
            num_workers=Cfg.num_workers,
            pin_memory=True,
            drop_last=True
        )

        return train_loader, in_dim
    
    def _train(self, train_loader):
        self.model.train()
        for epoch in range(Cfg.num_epochs):
            running = 0.0
            n = 0
            prog = tqdm(train_loader, disable=not self.accelerator.is_local_main_process, desc=f"Epoch {epoch+1}/{Cfg.num_epochs}")
            for x, y in prog:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                logits = self.model(x)
                loss = self.criterion(logits, y)

                self.optimizer.zero_grad(set_to_none=True)
                self.accelerator.backward(loss)
                self.optimizer.step()

                running += loss.detach().item() * x.size(0)
                n += x.size(0)
                avg = running / max(1, n)
                prog.set_postfix(loss=f"{avg:.4f}")

            # (Optional) sync/print
            if self.accelerator.is_main_process:
                print(f"[INFO] Epoch {epoch+1} avg loss: {avg:.4f}")

        if self.accelerator.is_main_process:
            os.makedirs("checkpoints", exist_ok=True)
            unwrapped = self.accelerator.unwrap_model(self.model)
            torch.save(unwrapped.state_dict(), "checkpoints/matcher.pt")
            print("[OK] Saved model to checkpoints/matcher.pt")

    def train(self):
        
        # 1) Download/load CLIP
        if self.accelerator.is_main_process:
            print(f"[INFO] Loading CLIP model: {Cfg.model_ckpt}")
            clip, processor = self._load_model()

        # 2) Load captions.json
        self._load_caption()
        

        # 3) Precompute embeddings
        img_list, txt_list = precompute_embeddings(self.caption_map, Cfg, processor, clip, self.device)

        # (Optional) free CLIP to save memory during training
        del clip, processor
        torch.cuda.empty_cache()

        # 4) Build dataset/dataloader
        # input = image_emb || text_emb  (concatenation)
        train_loader, in_dim = self._build_dataset(img_list=img_list, txt_list=txt_list)
        
        # 5) Model + optimizer + loss
        self.model = Matcher(in_dim=in_dim, hidden=1024, pdrop=0.1)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=Cfg.lr)
        self.criterion = nn.BCEWithLogitsLoss()

        self.model, self.optimizer, self.train_loader = self.accelerator.prepare(self.model, self.optimizer, train_loader)

        # 6) Training
        self._train(train_loader = train_loader)
        

