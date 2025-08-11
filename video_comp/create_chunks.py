import json
import re
import subprocess
from pathlib import Path
from tqdm import tqdm
import os
from typing import Optional, Tuple
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, wait, as_completed, FIRST_COMPLETED

CAPTION_FILE_NAME = Path("caption_map.json")
MAX_QUEUE = 1000  # max in-flight futures before we throttle

def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)

def _micro_to_sec(x):
    if x is None:
        return None
    x = float(x)
    # Heuristic: dataset uses microseconds (e.g., 289000000 → 289s)
    return x / 1e6 if x > 1e3 else x  # if already seconds, keep

def _find_video_path(video_directory: Path, video_id: str) -> Optional[Path]:
    """
    Find a file named <video_id>.<ext> under:
        video_directory/*/*/<files>
    Prefers .mp4 but falls back to any extension; does a full recursive fallback if needed.
    """
    ext_priority = (".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v")
    candidates = []

    for level1 in video_directory.iterdir():
        if not level1.is_dir():
            continue
        for level2 in level1.iterdir():
            if not level2.is_dir():
                continue

            # 1) Direct file checks for preferred extensions
            for ext in ext_priority:
                p = level2 / f"{video_id}{ext}"
                if p.exists():
                    candidates.append(p)
                    return p

            # 2) Fallback: match by stem
            if not candidates:
                for f in level2.iterdir():
                    if f.is_file() and f.stem == video_id:
                        if f.suffix.lower() in ext_priority:
                            return f
                        candidates.append(f)

    if candidates:
        return candidates[0]

    # Recursive fallback
    for ext in ext_priority:
        p = next(video_directory.rglob(f"{video_id}{ext}"), None)
        if p is not None:
            return p

    for p in video_directory.rglob("*"):
        if p.is_file() and p.stem == video_id:
            return p

    return None

def _extract_clip(in_path: Path, start_s: float, end_s: float, out_path: Path):
    duration = max(0.0, end_s - start_s)
    if duration <= 0:
        raise ValueError(f"Non-positive duration ({duration}) for {in_path.name}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return

    # Improved FFmpeg command with better compatibility
    cmd = [
        "ffmpeg", "-y",
        "-i", str(in_path),
        "-ss", f"{start_s:.3f}",
        "-t", f"{duration:.3f}",
        # Video encoding with better compatibility
        "-c:v", "libx264", 
        "-preset", "medium",  # Changed from veryfast for better quality/compatibility
        "-crf", "23",         # Slightly higher CRF for better compatibility
        "-pix_fmt", "yuv420p", # Explicit pixel format for maximum compatibility
        "-profile:v", "baseline",  # H.264 baseline profile for maximum compatibility
        "-level", "3.0",      # H.264 level 3.0 for broad device support
        # Audio encoding
        "-c:a", "aac", 
        "-b:a", "128k",
        "-ar", "44100",       # Standard audio sample rate
        # Additional compatibility flags
        "-movflags", "+faststart",  # Move moov atom to beginning for web playback
        "-avoid_negative_ts", "make_zero",  # Handle negative timestamps
        str(out_path),
    ]

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start_s:.3f}", "-i", str(in_path),
        "-t", f"{duration:.3f}", "-c", "copy", str(out_path)
    ]
    
    # Run with error handling and capture stderr for debugging
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        # If the clip extraction fails, try a simpler approach
        print(f"Warning: FFmpeg failed for {in_path.name}, trying simpler encoding...")
        
        # Fallback command with stream copy if possible
        fallback_cmd = [
            "ffmpeg", "-y",
            "-i", str(in_path),
            "-ss", f"{start_s:.3f}",
            "-t", f"{duration:.3f}",
            "-c:v", "libx264",
            "-c:a", "aac",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(out_path),
        ]
        
        try:
            subprocess.run(fallback_cmd, check=True, capture_output=True, text=True)
            return True
        except subprocess.CalledProcessError as e2:
            # Last resort: re-encode everything
            last_resort_cmd = [
                "ffmpeg", "-y",
                "-i", str(in_path),
                "-ss", f"{start_s:.3f}",
                "-t", f"{duration:.3f}",
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # Ensure even dimensions
                "-c:v", "libx264",
                "-c:a", "aac",
                "-pix_fmt", "yuv420p",
                str(out_path),
            ]
            subprocess.run(last_resort_cmd, check=True, capture_output=True, text=True)

    return False

def _worker(item: dict, video_directory: str, out_clip_path: str) -> Optional[Tuple[str, str]]:
    """
    Process a single JSON item in a worker:
      - resolve video path by video_id
      - compute start/end
      - write trimmed clip to the provided out_clip_path
      - return (absolute_out_clip_path, caption) or None if skipped/failed
    """
    try:
        video_directory = Path(video_directory)
        out_clip_path = Path(out_clip_path)

        vid = item.get("video_id")
        st = item.get("original_video/start_time")
        et = item.get("original_video/end_time")
        if not vid or st is None or et is None:
            return None

        start_s = _micro_to_sec(st)
        end_s = _micro_to_sec(et)

        in_path = _find_video_path(video_directory, vid)
        if in_path is None:
            return None

        out_bool = _extract_clip(in_path, start_s, end_s, out_clip_path)

        if not out_bool:
            return None

        # Verify the output file was created and has reasonable size
        if not out_clip_path.exists() or out_clip_path.stat().st_size < 1024:
            return None

        caption = (item.get("positive_text")
                   or item.get("caption")
                   or item.get("text")
                   or "").strip()

        return (str(out_clip_path.resolve()), caption)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error for {item.get('video_id', 'unknown')}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error for {item.get('video_id', 'unknown')}: {e}")
        return None


class CreateChunks:
    def __init__(self, video_directory, out_path, json_path, cpu_count=None, max_queue: int = MAX_QUEUE):
        """
        out_path: directory where clips will be written as {video_id}_{iter}.mp4
        cpu_count: number of worker processes (default: os.cpu_count())
        """
        self.json_path = Path(json_path)
        self.video_directory = Path(video_directory)
        self.out_path = Path(out_path)

        self.caption_map_path = self.out_path / CAPTION_FILE_NAME
        self.out_path.mkdir(parents=True, exist_ok=True)

        self.json_data = self.__read_items()

        self.max_workers = cpu_count or os.cpu_count() or 4
        self.max_queue = max_queue

    def __read_items(self):
        """Load either a JSON array/object or JSONL file."""
        txt = self.json_path.read_text(encoding="utf-8")
        try:
            obj = json.loads(txt)
            return obj if isinstance(obj, list) else [obj]
        except json.JSONDecodeError:
            return [json.loads(line) for line in txt.splitlines() if line.strip()]

    def __load_caption_map(self):
        if self.caption_map_path.exists():
            try:
                return json.loads(self.caption_map_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def __save_caption_map_atomic(self, caption_map: dict):
        tmp_path = self.caption_map_path.with_suffix(self.caption_map_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(caption_map, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, self.caption_map_path)

    def __init_iter_counters(self, items):
        """
        Initialize per-video counters by scanning existing files in out_path
        and starting from max(existing) for each video_id.
        """
        counters = defaultdict(int)

        # Seed from existing files like {video_id}_{n}.mp4
        for f in self.out_path.glob("*.mp4"):
            stem = f.stem
            if "_" in stem:
                vid_part, idx_part = stem.rsplit("_", 1)
                if idx_part.isdigit():
                    counters[vid_part] = max(counters[vid_part], int(idx_part))

        # Ensure keys exist for vids in the incoming items
        for it in items:
            vid = it.get("video_id")
            if vid:
                counters[_sanitize(vid)] = counters[_sanitize(vid)]
        return counters

    def extract_clips(self):
        """
        Multiprocess extraction.
        - Pre-assigns output filenames deterministically in the parent process:
              out_dir/{video_id}_{iter}.mp4
        - Uses ProcessPoolExecutor with a live tqdm progress bar.
        """
        items = self.json_data
        caption_map = self.__load_caption_map()
        results = []

        # Pre-assign output paths to avoid cross-process races on counters
        counters = self.__init_iter_counters(items)
        submit_list = []  # list of (item, out_clip_path_str)
        for item in items:
            vid = item.get("video_id")
            if not vid:
                continue
            vid_sanitized = _sanitize(vid)
            counters[vid_sanitized] += 1
            out_name = f"{vid_sanitized}_{counters[vid_sanitized]}.mp4"
            out_clip_path = self.out_path / out_name
            submit_list.append((item, str(out_clip_path)))

        with ProcessPoolExecutor(max_workers=self.max_workers) as ex:
            pending = set()
            with tqdm(total=len(submit_list),
                      desc=f"Extracting video comp clips (mp, {self.max_workers} workers)",
                      unit="clip") as pbar:

                # Submit tasks, throttle when queue size exceeds max_queue
                for item, out_clip_path in submit_list:
                    fut = ex.submit(_worker, item, str(self.video_directory), out_clip_path)
                    pending.add(fut)

                    while len(pending) > self.max_queue:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for f in done:
                            try:
                                res = f.result()
                            except Exception:
                                res = None
                            if res:
                                results.append(res)
                            pbar.update(1)

                # Drain the remainder
                for f in as_completed(pending):
                    try:
                        res = f.result()
                    except Exception:
                        res = None
                    if res:
                        results.append(res)
                    pbar.update(1)

        # Merge results into caption_map and write ONCE (atomic)
        for path_str, caption in results:
            caption_map[path_str] = caption

        self.__save_caption_map_atomic(caption_map)
        print(f"[OK] Wrote caption map: {self.caption_map_path}")
        print(f"[OK] Successfully extracted {len(results)} clips")

    def run(self):
        self.extract_clips()