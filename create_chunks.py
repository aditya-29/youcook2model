import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from tqdm import tqdm
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
# RAW_VIDEO_ROOT   = Path("./raw_videos")        # original variable from your code
# RAW_ANNOT_ROOT   = Path("./raw_annot_videos")
# EXTENSIONS       = [".mp4", ".mkv", ".mov"]         # whatever you support
# SECONDS_LIMIT    = 50                              # your existing guardrail
# MAX_WORKERS      = os.cpu_count() or 4              # reasonable default
# ------------------------------------------------------------------

class CreateChunk:
    def __init__(self,
                 video_annotations, 
                 raw_video_root, 
                 raw_annot_root, 
                 extensions, 
                 seconds_limit, 
                 max_workers,
                data_folder_path):
        self.video_annotations = video_annotations
        self.RAW_VIDEO_ROOT = raw_video_root
        self.RAW_ANNOT_ROOT = raw_annot_root
        self.EXTENSIONS = extensions
        self.SECONDS_LIMIT = seconds_limit
        self.MAX_WORKERS = max_workers
        self.DATA_FOLDER = data_folder_path

    def find_source(self,video_name: str, any_annot: dict) -> Path | None:
        """Return the first existing file matching the video name + extension."""
        stem_dir = self.RAW_VIDEO_ROOT / any_annot["subset"] / any_annot["recipe_type"]
        for ext in self.EXTENSIONS:
            p = stem_dir / f"{video_name}{ext}"
            if p.exists():
                return p
        return None

    def trim_with_ffmpeg(self, src: Path, dst: Path, start: float, end: float) -> None:
        """ffmpeg stream‑copy: ~100 × faster than decode/re‑encode for long clips."""
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():                                   # skip work already done
            return
        duration = end - start
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start}", "-i", str(src),
            "-t", f"{duration}", "-c", "copy", str(dst)
        ]
        subprocess.run(cmd, check=True)

    def process_one_video(self,video_name: str, annotations: list[dict]) -> None:
        src = self.find_source(video_name, annotations[0])
        if src is None:
            # LOGGER.warning("missing video file for %s", src)
            return

        for ann in annotations:
            if ann["end"] - ann["start"] > self.SECONDS_LIMIT:
                continue

            dst_dir = self.RAW_ANNOT_ROOT / ann["subset"] / ann["recipe_type"]
            dst = dst_dir / f"{video_name}__{ann['id']}.mp4"
            try:
                self.trim_with_ffmpeg(src, dst, ann["start"], ann["end"])
            except subprocess.CalledProcessError as exc:
                LOGGER.error("%s → trim failed (%s)", src.name, exc)

    # ------------------------------------------------------------------
    # MAIN PARALLEL DRIVER
    # ------------------------------------------------------------------
    def run(self, max_videos=None):
        if max_videos is None:
            max_videos = len(self.video_annotations)

        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as pool:
            work = (pool.submit(self.process_one_video, v_name, v_annots)
                    for v_name, v_annots in dict(list(self.video_annotations.items()[:max_videos])[:max_videos]).items())
            list(tqdm(work, total=len(self.video_annotations), desc="clipping"))