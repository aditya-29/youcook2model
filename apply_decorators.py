from __future__ import annotations
import json, os, subprocess, multiprocessing as mp
from pathlib import Path
from tqdm import tqdm

from decorators_ffmpeg import *

TRANSFORMS: dict[str, callable[[Path, Path], None]] = {
    "RR"      : lambda s, d, caption: FFMPEG_REVERSE(s, d, caption),
    "RCS_x0_5": lambda s, d, caption: FFMPEG_SPEED(s, d, 0.5, caption),
    "RCS_x1"  : lambda s, d, caption: FFMPEG_SPEED(s, d, 1.0, caption),
    "RCS_x2"  : lambda s, d, caption: FFMPEG_SPEED(s, d, 2.0, caption),
}

class ApplyDecorators:
    def __init__(self, 
                 raw_annot_root,
                 save_annot_root,
                 caption_file_path,
                 cpu_count):
        self.RAW_ROOT = raw_annot_root
        self.SAVE_ROOT = save_annot_root
        self.CAPTION_FILE = caption_file_path
        self.CPU_COUNT = cpu_count

    def _caption_lookup(self):

        # ------------------------------------------------------------------
        # 1.  Caption look‑up ‒ optional
        # ------------------------------------------------------------------
        self.CAPTION_FILE.touch(exist_ok=True)               # create if absent
        with self.CAPTION_FILE.open() as f:
            try:
                self.CAPTION_MAP: dict[str, str] = json.load(f)
            except json.JSONDecodeError:
                self.CAPTION_MAP = {}

    def _check_if_caption_exists(self, path):
        if path in self.CAPTION_MAP:
            return True
        else:
            return False

    def update_caption_mp(self, path, caption):
        self.CAPTION_MAP[path] = caption

    def save_caption_mp(self):
        with open(self.CAPTION_FILE, 'w') as f:
            json.dump(self.CAPTION_MAP, f)

    def get_caption(self, path: Path) -> str:
            """Return caption text or empty string."""
            return self.get_caption_for_chunk(path)


    def process_clip(self, src_path: Path) -> None:
        # Skip files without a caption (optional; drop this if not needed)
        caption = self.get_caption(str(src_path)).strip()
        if caption == "":
            return

        for iter, (key, transform) in enumerate(self.TRANSFORMS.items()):
            dst = (self.SAVE_ROOT / src_path.relative_to(self.RAW_ROOT)
                    ).with_stem(src_path.stem + f"_tr_{key}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                continue  # already processed
            if self.check_if_caption_exists(str(src_path)):
                continue
            try:
                caption = transform(src_path, dst, caption)
                self.update_caption_mp(str(src_path), caption)
            except subprocess.CalledProcessError as e:
                print(f"[WARN] ffmpeg failed on {src_path} ({key}): {e}")
            self.save_caption_mp()
    
    def run(self):
        all_mp4s = [
            p for p in self.RAW_ROOT.rglob("*.mp4")
            if "_tr_" not in p.stem
        ]

        with mp.Pool(self.CPU_COUNT) as pool:
            for _ in tqdm(
                pool.imap_unordered(self.process_clip, all_mp4s),
                total=len(all_mp4s),
                desc="Processing clips"
            ):
                pass
        