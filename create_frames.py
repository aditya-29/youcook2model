#!/usr/bin/env python
"""
Fast GPU‑aware video‑to‑frame converter + caption joiner

* Uses NVIDIA NVDEC/NVJPEG via FFmpeg when available.
* Falls back to CPU FFmpeg on machines without compatible GPUs.
* Parallel over videos with ProcessPoolExecutor.
"""

import os, json, subprocess, shutil, argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

# --------------------------------------------------------------
# 0. GPU probe + FFmpeg helpers
# --------------------------------------------------------------
def _nvdec_available() -> bool:
    """
    Return True only if *both*:
      • `nvidia-smi` lists at least one GPU, and
      • FFmpeg was built with CUVID decoders.
    """
    try:
        # Fast check: does nvidia‑smi see a board?
        smi = subprocess.check_output(["nvidia-smi", "-L"],
                                      text=True, stderr=subprocess.DEVNULL)
        if "GPU" not in smi:
            return False
    except (FileNotFoundError, subprocess.SubprocessError):
        return False

    # Second check: FFmpeg decoders
    try:
        decs = subprocess.check_output(
            ["ffmpeg", "-hide_banner", "-decoders"],
            text=True, stderr=subprocess.DEVNULL)
        # Check for multiple CUVID decoders
        cuvid_decoders = ["h264_cuvid", "hevc_cuvid", "av1_cuvid", "mpeg2_cuvid", "mpeg4_cuvid", "vp8_cuvid", "vp9_cuvid"]
        return any(decoder in decs for decoder in cuvid_decoders)
    except (FileNotFoundError, subprocess.SubprocessError):
        return False

def _get_video_codec(video_path: Path) -> Optional[str]:
    """
    Detect the video codec of the input file.
    Returns codec name or None if detection fails.
    """
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name", "-of", "csv=p=0",
            str(video_path)
        ]
        result = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
        return result.strip().lower()
    except (FileNotFoundError, subprocess.SubprocessError):
        return None

def _get_cuvid_decoder(codec: str) -> Optional[str]:
    """
    Map video codec to corresponding CUVID decoder.
    """
    codec_map = {
        'h264': 'h264_cuvid',
        'h265': 'hevc_cuvid',
        'hevc': 'hevc_cuvid',
        'av1': 'av1_cuvid',
        'mpeg2video': 'mpeg2_cuvid',
        'mpeg4': 'mpeg4_cuvid',
        'vp8': 'vp8_cuvid',
        'vp9': 'vp9_cuvid'
    }
    return codec_map.get(codec)

USE_GPU = _nvdec_available()

def _run_ffmpeg_extract(src: Path, dst_dir: Path, fps: float) -> int:
    """
    Extract <fps> frames/s from *src* into *dst_dir*; return frame count.
    Gracefully falls back to CPU if CUDA decode fails.
    """
    def _try_gpu() -> None:
        """Try GPU-accelerated extraction with codec detection."""
        # Detect video codec
        codec = _get_video_codec(src)
        if not codec:
            raise RuntimeError("Could not detect video codec")
        
        # Get appropriate CUVID decoder
        cuvid_decoder = _get_cuvid_decoder(codec)
        if not cuvid_decoder:
            raise RuntimeError(f"No CUVID decoder available for codec: {codec}")
        
        # Build command with detected codec
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-hwaccel", "cuda",
            "-hwaccel_output_format", "cuda",
            "-c:v", cuvid_decoder,
            "-i", str(src),
            "-vf", f"fps={fps},hwdownload,format=nv12",
            "-start_number", "0",
            "-f", "image2", "-qscale:v", "2",
            str(dst_dir / "frame_%06d.jpg")
        ]
        
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                              stderr=subprocess.PIPE, text=True)
        if proc.returncode:
            raise RuntimeError(f"GPU decode failed: {proc.stderr.strip()}")

    def _try_cpu() -> None:
        """CPU fallback extraction."""
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(src),
            "-vf", f"fps={fps}",
            "-start_number", "0",
            "-f", "image2", "-qscale:v", "2",
            str(dst_dir / "frame_%06d.jpg")
        ]
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                              stderr=subprocess.PIPE, text=True)
        if proc.returncode:
            raise RuntimeError(f"CPU decode failed: {proc.stderr.strip()}")

    dst_dir.mkdir(parents=True, exist_ok=True)

    # Try GPU first if available, then fall back to CPU
    if USE_GPU:
        try:
            _try_gpu()
        except RuntimeError as e:
            error_msg = str(e).lower()
            # More comprehensive error catching for GPU failures
            gpu_error_indicators = [
                "cuda_error_no_device",
                "cuda_error_out_of_memory", 
                "cuda_error_invalid_device",
                "no cuda-capable device",
                "cuda driver version is insufficient",
                "cuvid",
                "nvdec",
                "gpu decode failed"
            ]
            
            if any(indicator in error_msg for indicator in gpu_error_indicators):
                print(f"⚠️  GPU decode failed for {src.name} - retrying with CPU", flush=True)
                _try_cpu()
            else:
                raise
    else:
        _try_cpu()

    return sum(1 for _ in dst_dir.glob("frame_*.jpg"))

    
# --------------------------------------------------------------
# 1. Caption lookup (YouCook2‑specific)
# --------------------------------------------------------------
# Adjust the path to your annotations file if needed
ANNOT_JSON = "./data/youcookii_annotations_trainval.json"

def load_video_annotations() -> Dict[str, List[Dict]]:
    """Load video annotations with error handling."""
    try:
        with open(ANNOT_JSON, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️  Annotations file not found: {ANNOT_JSON}")
        return {}
    except json.JSONDecodeError:
        print(f"⚠️  Invalid JSON in annotations file: {ANNOT_JSON}")
        return {}

video_annotations = load_video_annotations()

def get_caption_for_chunk(path: str,
                          video_annotations: Dict[str, List[Dict]] = video_annotations) -> str | None:
    """Return caption for a transformed or raw clip path."""
    if not video_annotations:
        return None
        
    suffix = ""
    if "_tr_" in path:
        if "RCS" in path:
            suffix = {"x0_5": " played in 0.5x speed",
                      "x1": "", "x2": " played in 2x speed"}.get(
                         path.split("RCS_")[-1][:4], "")
        elif "RR" in path:
            suffix = " played in reverse"
    path_clean = path.split("_tr_")[0] + ".mp4"

    try:
        subset = path_clean.split("raw_annot_videos/")[1].split("/")[0]
        recipe = path_clean.split(subset+"/")[1].split("/")[0]
        video_name = path_clean.split(subset+"/"+recipe+"/")[1].split("__")[0]
        vid_id = path_clean.split("__")[1].split(".mp4")[0]

        for entry in video_annotations.get(video_name, []):
            if entry["id"] == int(vid_id):
                return entry["sentence"] + suffix
    except (IndexError, ValueError):
        pass
    
    return None

# --------------------------------------------------------------
# 2. Utility helpers
# --------------------------------------------------------------
def get_dataset_name_from_path(video_path: Path) -> str:
    path = str(video_path).lower()
    if "ucf" in path:
        return "ucf101"
    if "hmdb" in path:
        return "hmdb51"
    return "unknown_dataset"

# --------------------------------------------------------------
# 3. Worker for multiprocessing
# --------------------------------------------------------------
def _process_one(task: Tuple[int, dict, Path, Path, float]) -> dict | Tuple[Path, str]:
    idx, vinfo, input_root, output_root, fps = task
    video_path: Path = vinfo['path']
    subset     = vinfo['subset']
    rel_path   = vinfo['relative_path']

    video_id   = f"video_{idx:06d}"
    frame_dir  = output_root / "videos" / video_id
    try:
        fcnt = _run_ffmpeg_extract(video_path, frame_dir, fps)
        caption = get_caption_for_chunk("raw_annot_videos/"+rel_path)
        class_id = video_path.parent.name
        split = {'training':'train','validation':'val','testing':'test'}.get(subset, subset)
        return {
            "video_dataset_name": get_dataset_name_from_path(video_path),
            "video_id"          : video_id,
            "video_caption"     : caption,
            "split"             : split,
            "original_action"   : class_id,
            "original_video_folder": video_path.stem,
            "frame_count"       : fcnt
        }
    except Exception as exc:
        return (video_path, str(exc))

# --------------------------------------------------------------
# 4. Main conversion routine
# --------------------------------------------------------------
def convert_videos_to_old_structure(input_root: str,
                                    output_root: str,
                                    fps: float = 2.0,
                                    max_videos: int | None = None):
    input_path  = Path(input_root)
    output_path = Path(output_root)
    (output_path / "videos").mkdir(parents=True, exist_ok=True)

    # Discover .mp4 files
    sets = ['training','validation','testing']
    video_files: List[dict] = []
    for s in sets:
        subset_path = input_path / s
        if subset_path.exists():
            for v in subset_path.glob("*/*.mp4"):
                video_files.append({'path': v,
                                    'subset': s,
                                    'relative_path': str(v.relative_to(input_path))})
    
    if max_videos:
        video_files = video_files[:max_videos]

    print(f"NVDEC available: {'yes' if USE_GPU else 'no — CPU path'}")
    print(f"Found {len(video_files)} videos (processing {len(video_files)})")

    if not video_files:
        print("No video files found! Check your input directory structure.")
        return

    tasks = [(i, v, input_path, output_path, fps)
             for i, v in enumerate(video_files)]
    annotations, errors = [], []
    
    # Reduce workers for GPU processing to avoid memory issues
    workers = min(4, os.cpu_count()) if USE_GPU else max(1, os.cpu_count())
    print(f"Using {workers} workers")

    with ProcessPoolExecutor(max_workers=workers) as ex:
        with tqdm(total=len(tasks), desc="Processing videos") as pbar:
            futures = [ex.submit(_process_one, task) for task in tasks]
            for fut in as_completed(futures):
                res = fut.result()
                if isinstance(res, dict):
                    annotations.append(res)
                else:
                    errors.append(res)
                pbar.update(1)

    # Save annotations
    annot_path = output_path / "annotations.json"
    with open(annot_path, 'w') as f:
        json.dump(annotations, f, indent=2)
    print(f"\nDone! {len(annotations)} videos ok, {len(errors)} failed.")
    print(f"Frames dir : {output_path/'videos'}")
    print(f"Annots JSON: {annot_path}")

    if errors:
        print("\nFailures (first 10):")
        for v, msg in errors[:10]:
            print(f"  {v} → {msg}")

# --------------------------------------------------------------
# 5. Optional verification
# --------------------------------------------------------------
def verify_conversion(output_root: str, num_samples: int = 5):
    out = Path(output_root)
    videos_dir = out / "videos"
    annot_file = out / "annotations.json"
    
    if not annot_file.exists():
        print("No annotations file found!")
        return
        
    with open(annot_file, 'r') as f:
        anns = json.load(f)
    print(f"\nVerify: total {len(anns)} videos")
    for i, ann in enumerate(anns[:num_samples]):
        vdir = videos_dir / ann['video_id']
        if vdir.exists():
            n = len(list(vdir.glob('frame_*.jpg')))
            expected = ann['frame_count']
            ok = "✓" if n == expected else "✗"
            print(f"{i+1}. {ann['video_id']} frames {n}/{expected} {ok}")
        else:
            print(f"{i+1}. {ann['video_id']} directory missing ✗")

# --------------------------------------------------------------
# 6. CLI
# --------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='GPU‑accelerated video conversion')
    parser.add_argument('--input_root',  required=True, help='Input directory with training/validation/testing folders')
    parser.add_argument('--output_root', required=True, help='Output directory for frames and annotations')
    parser.add_argument('--fps', type=float, default=2.0, help='Frames per second to extract')
    parser.add_argument('--max_videos', type=int, default=None, help='Limit number of videos to process')
    parser.add_argument('--verify', action='store_true', help='Verify conversion after completion')
    args = parser.parse_args()

    convert_videos_to_old_structure(args.input_root, args.output_root,
                                    args.fps, args.max_videos)
    if args.verify:
        verify_conversion(args.output_root)

if __name__ == "__main__":
    main()