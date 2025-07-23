from __future__ import annotations
import json, os, subprocess, multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
import random
import fcntl           # advisory file‑lock (Linux / macOS)
from typing import Tuple

from decorators_ffmpeg import *
from utils import *

TRANSFORMS: dict[str, callable[[Path, Path], None]] = {
    "RR"      : lambda s, d, caption: FFMPEG_REVERSE(s, d, caption),
    "RCS_x0_5": lambda s, d, caption: FFMPEG_SPEED(s, d, 0.5, caption),
    "RCS_x1"  : lambda s, d, caption: FFMPEG_SPEED(s, d, 1.0, caption),
    "RCS_x2"  : lambda s, d, caption: FFMPEG_SPEED(s, d, 2.0, caption),
}

CLIP_TRANSFORM = {
    "2RP" : lambda s1, c1, s2, c2, d: FFMPEG_RANDOM_PICK(src1, caption1, src2, caption2, d)
}

class ApplyDecorators:
    def __init__(self, 
                 raw_annot_root,
                 save_annot_root,
                 caption_file_path,
                 video_annotations,
                 cpu_count):
        self.RAW_ROOT = raw_annot_root
        self.SAVE_ROOT = save_annot_root
        self.CAPTION_FILE = caption_file_path
        self.video_annotations = video_annotations
        self.CPU_COUNT = cpu_count
        
        self._caption_lookup()
        self._saved_keys: set[str] = set()     # track what’s already flushed

        # self._CAPTIONS_UTILS = Captions(annotations_json_path=self.CAPTION_FILE)

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
        if self.__internal_count == None:
            self.__internal_count = 0

        self.__internal_count += 1
        self.CAPTION_MAP[path] = caption

        if self.__internal_count % 1000 == 0 or self.__internal_count == 1:
            self.save_caption_mp()
            self.CAPTION_MAP = {}

    def save_caption_mp(self) -> None:
        """
        Load existing captions from file, update with new entries from CAPTION_MAP,
        and save the complete updated data back to the file.
        """
        # 1. Collect items not yet persisted
        unsaved = {
            k: v for k, v in self.CAPTION_MAP.items() if k not in self._saved_keys
        }
        if not unsaved:
            return                                  # nothing new
    
        # 2. Load existing captions from file
        existing_captions = {}
        if self.CAPTION_FILE.exists() and self.CAPTION_FILE.stat().st_size > 0:
            with open(self.CAPTION_FILE, "r", encoding="utf-8") as f:
                try:
                    existing_captions = json.load(f)
                except json.JSONDecodeError:
                    existing_captions = {}
        
        # 3. Update with new entries
        existing_captions.update(unsaved)
        
        # 4. Save the complete updated data back to file
        with open(self.CAPTION_FILE, "w", encoding="utf-8") as f:
            # advisory lock so multiple workers won't interleave writes
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                json.dump(existing_captions, f, ensure_ascii=False, indent=2)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    
        # 5. Mark these keys as flushed
        self._saved_keys.update(unsaved.keys())

    def get_caption(self, path: Path) -> str:
            """Return caption text or empty string."""
            return  get_caption_for_chunk(path = path,
                                          video_annotations = self.video_annotations)


    def process_clip(self, src_path: Path) -> None:
        # Skip files without a caption (optional; drop this if not needed)
        caption = self.get_caption(str(src_path)).strip()
        if caption == "":
            return

        for iter, (key, transform) in enumerate(TRANSFORMS.items()):
            dst = (self.SAVE_ROOT / src_path.relative_to(self.RAW_ROOT)
                    ).with_stem(src_path.stem + f"_tr_{key}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                continue  # already processed
            try:
                generated_caption = transform(src_path, dst, caption)
                self.update_caption_mp(str(dst), generated_caption)
            except subprocess.CalledProcessError as e:
                print(f"[WARN] ffmpeg failed on {src_path} ({key}): {e}")
            self.save_caption_mp()

    def process_clip_random_pick(self, pair: Tuple[Path, Path]) -> None:
        """
        Randomly copies *one* of the two input videos to the destination while
        merging their captions:  "caption1 caption2".
    
        Args
        ----
        pair : (src_path1, src_path2)
            Two absolute paths inside `self.RAW_ROOT`.
        """
        src1, src2 = pair
    
        # ── Grab captions ─────────────────────────────────────────
        cap1 = self.get_caption(str(src1)).strip()
        cap2 = self.get_caption(str(src2)).strip()
        if cap1 == "" or cap2 == "":
            return                                    # skip if either missing
    
        # ── Build destination path ───────────────────────────────
        #   • keep the original directory structure (subset/ID/…)
        #   • name:  <stem1>_<stem2>_tr_RP.mp4
        dst = (
            self.SAVE_ROOT
            / src1.relative_to(self.RAW_ROOT).parent
            / f"{src1.stem}_mrg_{src2.stem}_tr_RP.mp4"
        )
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            return                                    # already processed
    
        try:
            merged_caption = FFMPEG_RANDOM_PICK(
                src1=src1, caption1=cap1,
                src2=src2, caption2=cap2,
                dst=dst,
            )
            # Store the new clip‑to‑caption mapping
            self.update_caption_mp(str(dst.relative_to(self.SAVE_ROOT)), merged_caption)
    
        except subprocess.CalledProcessError as e:
            print(f"[WARN] ffmpeg failed on {src1} & {src2} (RP): {e}")
    
        self.save_caption_mp()

    # ──────────────────────────────────────────────────────────────
    #  Dispatcher: sequential random‑pick within '__'‑groups
    # ──────────────────────────────────────────────────────────────
    def run_random_pick(self) -> None:
        """
        For every group of files that share the same prefix before the first
        '__', build sequential pairs and feed them to `process_clip_random_pick`.
        Example stems           → groups
           Qewsnks__1           → ["Qewsnks__1", "Qewsnks__2", "Qewsnks__3"]
           asdas_1              → ["asdas_1"]        (singletons ignored)
        """
        # 1. Gather raw clips (excluding already‑transformed ones)
        all_mp4s = [
            p for p in self.RAW_ROOT.rglob("*.mp4")
            if "_tr_" not in p.stem
        ]
    
        # 2. Bucket them by prefix before the first '__'
        groups: dict[str, list[Path]] = {}
        for p in all_mp4s:
            stem = p.stem
            prefix = stem.split("__", 1)[0]            # "abc__1" → "abc"
            groups.setdefault(prefix, []).append(p)
    
        # 3. Build sequential pairs *within* each group
        pairs: list[tuple[Path, Path]] = []
        for vids in groups.values():
            if len(vids) < 2:
                continue                               # need ≥2 to pair
            vids.sort()                                # deterministic order
            for i in range(0, len(vids) - 1, 2):       # (v0,v1), (v2,v3), …
                pairs.append((vids[i], vids[i + 1]))
    
        # 4. Fan out to the pool
        if not pairs:
            print("[INFO] No eligible pairs for random‑pick.")
            return
    
        with mp.Pool(self.CPU_COUNT) as pool:
            for _ in tqdm(
                pool.imap_unordered(self.process_clip_random_pick, pairs),
                total=len(pairs),
                desc="Random‑pick clips",
            ):
                pass    
        
    def run(self):
        self.run_random_pick()
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

        