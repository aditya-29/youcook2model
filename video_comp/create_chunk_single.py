import json
import re
import subprocess
from pathlib import Path
from tqdm import tqdm
import os
from typing import Optional, Tuple
from collections import defaultdict

CAPTION_FILE_NAME = Path("caption_map.json")

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

    # try:
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
                        
    # except FileNotFoundError:
    #     print("file not found : " + str(video_id))
    #     pass

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

def _worker(item: dict, video_directory: str, out_clip_path: str) -> Optional[Tuple[str, str]]:
    """
    Process a single JSON item (single-threaded):
      - resolve video path by video_id
      - compute start/end
      - write trimmed clip to the provided out_clip_path
      - return (absolute_out_clip_path, caption) or None if skipped/failed
    """
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

    _extract_clip(in_path, start_s, end_s, out_clip_path)

    caption = (item.get("positive_text")
               or item.get("caption")
               or item.get("text")
               or "").strip()

    return (str(out_clip_path.resolve()), caption)


class CreateChunks:
    def __init__(self, video_directory, out_path, json_path, cpu_count=None):
        """
        out_path: directory where clips will be written as {video_id}_{iter}.mp4
        """
        self.json_path = Path(json_path)
        self.video_directory = Path(video_directory)
        self.out_path = Path(out_path)

        self.caption_map_path = self.out_path / CAPTION_FILE_NAME
        self.out_path.mkdir(parents=True, exist_ok=True)

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
            # split from the rightmost underscore; tolerate underscores in video_id
            if "_" in stem:
                vid_part, idx_part = stem.rsplit("_", 1)
                if idx_part.isdigit():
                    counters[vid_part] = max(counters[vid_part], int(idx_part))

        # Ensure keys exist for vids in the incoming items
        for it in items:
            vid = it.get("video_id")
            if vid:
                # store sanitized key because we sanitize filenames
                counters[_sanitize(vid)] = counters[_sanitize(vid)]
        return counters

    def extract_clips(self):
        """
        Single-process extraction. Writes clips as:
            out_dir/{video_id}_{iter}.mp4
        where iter increments per video_id (resuming from existing files if present).
        """
        items = self.json_data
        caption_map = self.__load_caption_map()
        results = []

        counters = self.__init_iter_counters(items)

        for item in tqdm(items, desc="Extracting video comp clips (single process)"):
            vid = item.get("video_id")
            if not vid:
                continue

            vid_sanitized = _sanitize(vid)
            counters[vid_sanitized] += 1
            out_name = f"{vid_sanitized}_{counters[vid_sanitized]}.mp4"
            out_clip_path = self.out_path / out_name

            res = _worker(item, str(self.video_directory), str(out_clip_path))
            if res:
                results.append(res)

        # Merge results into caption_map and write ONCE (atomic)
        for path_str, caption in results:
            caption_map[path_str] = caption

        self.__save_caption_map_atomic(caption_map)
        print(f"[OK] Wrote caption map: {self.caption_map_path}")

    def run(self):
        self.extract_clips()
