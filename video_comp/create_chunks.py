import json
import re
import subprocess
from pathlib import Path
from tqdm import tqdm
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Tuple

CAPTION_FILE_NAME = Path("caption_map.json")
MAX_QUEUE = 1000

def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)

def _micro_to_sec(x):
    if x is None:
        return None
    x = float(x)
    # Heuristic: dataset uses microseconds (e.g., 289000000 → 289s)
    return x / 1e6 if x > 1e3 else x  # if already seconds, keep

def _find_video_path(video_directory: Path, video_id: str) -> Optional[Path]:
    # Search recursively for any file named <video_id>.<ext>
    # Use rglob generator to avoid building huge lists when possible
    gen = video_directory.rglob(f"{video_id}.*")
    first = next(gen, None)
    if first is None:
        return None
    if first.suffix.lower() == ".mp4":
        return first
    # Prefer .mp4 if available; fall back to the first match
    for p in gen:
        if p.suffix.lower() == ".mp4":
            return p
    return first

def _extract_clip(in_path: Path, start_s: float, end_s: float, out_path: Path):
    duration = max(0.0, end_s - start_s)
    if duration <= 0:
        raise ValueError(f"Non-positive duration ({duration}) for {in_path.name}")

    # Ensure parent dir exists (each worker handles its own file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # If clip already exists, skip re-encoding (prevents concurrent overwrite)
    if out_path.exists():
        return

    # Accurate trim: -ss after -i (frame-accurate), re-encode video
    cmd = [
        "ffmpeg", "-y",
        "-i", str(in_path),
        "-ss", f"{start_s:.3f}",
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "aac", "-b:a", "128k",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def _worker(item: dict, video_directory: str, out_dir: str) -> Optional[Tuple[str, str]]:
    """
    Process a single JSON item:
      - resolve video path by video_id
      - compute start/end
      - write trimmed clip (skip if exists)
      - return (absolute_out_clip_path, caption) or None if skipped/failed
    """
    try:
        video_directory = Path(video_directory)
        out_root = Path(out_dir)

        vid = item.get("video_id")
        st = item.get("original_video/start_time") or item.get("orignal_video/start_time")
        et = item.get("original_video/end_time") or item.get("orignal_video/end_time")
        if not vid or st is None or et is None:
            return None

        start_s = _micro_to_sec(st)
        end_s = _micro_to_sec(et)

        in_path = _find_video_path(video_directory, vid)
        if in_path is None:
            return None

        key = item.get("key", vid)
        out_name = f"{_sanitize(key)}_{int(start_s*1000)}_{int(end_s*1000)}.mp4"
        out_path = out_root / out_name

        _extract_clip(in_path, start_s, end_s, out_path)

        caption = (item.get("positive_text")
                   or item.get("caption")
                   or item.get("text")
                   or "").strip()

        return (str(out_path.resolve()), caption)
    except subprocess.CalledProcessError:
        return None
    except Exception:
        return None


class CreateChunks:
    def __init__(self, video_directory, out_path, json_path, cpu_count):
        self.json_path = Path(json_path)
        self.video_directory = Path(video_directory)
        self.out_path = Path(out_path)
        self.max_workers = cpu_count
        self.max_queue = MAX_QUEUE

        self.caption_map_path = self.out_path / CAPTION_FILE_NAME
        if not self.out_path.exists():
            os.makedirs(self.out_path, exist_ok=True)

        # Read once in the parent
        self.json_data = self.__read_items()

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
        """
        Atomic write to avoid partial/overlapping writes:
        write to temp file then replace.
        Only the parent calls this once after all workers complete.
        """
        tmp_path = self.caption_map_path.with_suffix(self.caption_map_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(caption_map, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, self.caption_map_path)

    def extract_clips(self):
        """
        num_workers: processes to use (default: os.cpu_count()).
        max_queue:   backpressure for submitting tasks (helps memory on huge datasets).
        """
        items = self.json_data
        caption_map = self.__load_caption_map()

        # Submit all tasks
        results = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as ex:
            futures = []
            for item in items:
                # Backpressure to avoid unbounded futures
                if len(futures) - sum(f.done() for f in futures) > self.max_queue:
                    # Drain completed futures
                    for f in as_completed(futures):
                        res = f.result()
                        if res:
                            results.append(res)
                        # stop draining once queue under threshold
                        if len(futures) - sum(ff.done() for ff in futures) <= self.max_queue:
                            break

                futures.append(ex.submit(_worker, item, str(self.video_directory), str(self.out_path)))

            # Collect remaining
            for f in tqdm(as_completed(futures), total=len(futures), desc="Extracting video comp clips (mp)"):
                res = f.result()
                if res:
                    results.append(res)

        # Merge results into caption_map and write ONCE (atomic)
        for path_str, caption in results:
            caption_map[path_str] = caption

        self.__save_caption_map_atomic(caption_map)
        print(f"[OK] Wrote caption map: {self.caption_map_path}")

    def run(self):
        self.extract_clips()

