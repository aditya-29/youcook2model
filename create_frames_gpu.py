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

CAPTION_FILE = "captions.json"

# --------------------------------------------------------------
# 0. GPU probe + FFmpeg helpers
# --------------------------------------------------------------
def _nvdec_available() -> bool:
    """
    Return True only if *both*:
      • `nvidia-smi` lists at least one GPU, and
      • FFmpeg was built with CUDA support.
    """
    try:
        # Fast check: does nvidia‑smi see a board?
        smi = subprocess.check_output(["nvidia-smi", "-L"],
                                      text=True, stderr=subprocess.DEVNULL)
        if "GPU" not in smi:
            return False
    except (FileNotFoundError, subprocess.SubprocessError):
        return False

    # Check FFmpeg CUDA support (more reliable than checking specific decoders)
    try:
        # Test actual CUDA functionality with a simple probe
        test_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc2=duration=0.1:size=320x240:rate=1",
            "-hwaccel", "cuda", "-t", "0.1", "-f", "null", "-"
        ]
        result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.SubprocessError, subprocess.TimeoutExpired):
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
    # Normalize codec name
    codec = codec.lower().strip()
    
    codec_map = {
        # H.264 variants
        'h264': 'h264_cuvid',
        'avc': 'h264_cuvid',
        'avc1': 'h264_cuvid',
        # H.265/HEVC variants  
        'h265': 'hevc_cuvid',
        'hevc': 'hevc_cuvid',
        'hvc1': 'hevc_cuvid',
        'hev1': 'hevc_cuvid',
        # Other codecs
        'av1': 'av1_cuvid',
        'av01': 'av1_cuvid',
        'mpeg2video': 'mpeg2_cuvid',
        'mpeg2': 'mpeg2_cuvid',
        'mpeg4': 'mpeg4_cuvid',
        'xvid': 'mpeg4_cuvid',
        'vp8': 'vp8_cuvid',
        'vp9': 'vp9_cuvid',
        'vp09': 'vp9_cuvid'
    }
    return codec_map.get(codec)

USE_GPU = _nvdec_available()

def _run_ffmpeg_extract(src: Path, dst_dir: Path, fps: float) -> int:
    """
    Extract <fps> frames/s from *src* into *dst_dir*; return frame count.
    Gracefully falls back to CPU if CUDA decode fails.
    """
    def _try_gpu_basic() -> None:
        """Try basic GPU acceleration - most compatible approach."""
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-hwaccel", "cuda",
            "-i", str(src),
            "-vf", f"fps={fps}",
            "-start_number", "0",
            "-f", "image2", "-qscale:v", "2",
            str(dst_dir / "frame_%06d.jpg")
        ]
        
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                              stderr=subprocess.PIPE, text=True)
        if proc.returncode:
            raise RuntimeError(f"Basic GPU decode failed: {proc.stderr.strip()}")

    def _try_gpu_nvenc() -> None:
        """Try GPU with NVENC encoder for better compatibility."""
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-hwaccel", "cuda",
            "-hwaccel_output_format", "cuda",
            "-i", str(src),
            "-vf", f"fps={fps},hwdownload,format=nv12",
            "-start_number", "0", 
            "-f", "image2", "-qscale:v", "2",
            str(dst_dir / "frame_%06d.jpg")
        ]
        
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                              stderr=subprocess.PIPE, text=True)
        if proc.returncode:
            raise RuntimeError(f"NVENC GPU decode failed: {proc.stderr.strip()}")

    def _try_gpu_vaapi_fallback() -> None:
        """Try VAAPI as GPU fallback (works on some systems)."""
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-hwaccel", "vaapi",
            "-i", str(src),
            "-vf", f"fps={fps}",
            "-start_number", "0",
            "-f", "image2", "-qscale:v", "2",
            str(dst_dir / "frame_%06d.jpg")
        ]
        
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                              stderr=subprocess.PIPE, text=True)
        if proc.returncode:
            raise RuntimeError(f"VAAPI decode failed: {proc.stderr.strip()}")

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

    # Try GPU methods in order of compatibility
    if USE_GPU:
        gpu_methods = [
            ("basic CUDA", _try_gpu_basic),
            ("CUDA+NVENC", _try_gpu_nvenc),
            ("VAAPI fallback", _try_gpu_vaapi_fallback)
        ]
        
        for method_name, method_func in gpu_methods:
            try:
                method_func()
                return sum(1 for _ in dst_dir.glob("frame_*.jpg"))
            except RuntimeError as e:
                error_msg = str(e).lower()
                
                # Skip GPU methods if we hit CUVID capability errors
                skip_indicators = [
                    "cuvidgetdecodercaps", "cuvid", "nvdec", "cuda_error",
                    "no device", "out of memory", "invalid device",
                    "driver version", "not supported", "vaapi"
                ]
                
                if any(indicator in error_msg for indicator in skip_indicators):
                    # Don't spam with detailed errors for known GPU issues
                    continue
                else:
                    # Re-raise non-GPU errors
                    raise
        
        # All GPU methods failed silently, use CPU
        print(f"⚠️  GPU acceleration not working for {src.name} - using CPU", flush=True)
    
    _try_cpu()
    return sum(1 for _ in dst_dir.glob("frame_*.jpg"))

    
# --------------------------------------------------------------
# 1. Caption lookup (YouCook2‑specific)
# --------------------------------------------------------------
# Adjust the path to your annotations file if needed

def __modify_name(path):
    video_file = Path(path).name
    video_name = video_file.split("__")[0]
    
    video_chunk_1 = video_file.split(video_name)
    video_chunk_1 = video_name + video_chunk_1[1][:-1]
    
    video_chunk_2 = video_file.split(video_chunk_1)[1][1:].split("_tr")[0]
    
    new_video_file = video_chunk_1 + "_mrg_" + video_chunk_2 + video_file.split(video_chunk_2)[-1]
    new_video_file = Path(path).parent / Path(new_video_file)
    return str(new_video_file)

def get_caption_from_file(path: str, caption_mp) -> str | None:
    """Return caption for a transformed or raw clip path."""
    return caption_mp[path]

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
def _process_one(task: Tuple[int, dict, Path, Path, float, dict]) -> dict | Tuple[Path, str]:
    idx, vinfo, input_root, output_root, fps, caption_mp = task
    video_path: Path = vinfo['path']
    subset     = vinfo['subset']
    rel_path   = vinfo['relative_path']

    video_id   = f"video_{idx:06d}"
    frame_dir  = output_root / "videos" / video_id
    # try:
    fcnt = _run_ffmpeg_extract(video_path, frame_dir, fps)
    caption = get_caption_from_file(str(video_path), caption_mp)
    class_id = video_path.parent.name
    split = {'training':'train','validation':'val','testing':'test'}.get(subset, subset)
    
    return {
        "video_dataset_name": get_dataset_name_from_path(video_path),
        "video_id"          : video_id,
        "video_caption"     : caption,
        "split"             : split,
        "original_action"   : class_id,
        "original_video_folder": str(video_path),
        "frame_count"       : fcnt
    }
    # except Exception as exc:
    #     return (video_path, str(exc))

# --------------------------------------------------------------
# 4. Main conversion routine
# --------------------------------------------------------------
def convert_videos_to_old_structure(input_root: str,
                                    output_root: str,
                                    fps: float = 2.0,
                                    max_videos: int | None = None,
                                    raw_root_dir = None):
    if raw_root_dir is None:
        raise Exception("'raw_root_dir' cannot be None")
    
    input_path  = Path(input_root)
    output_path = Path(output_root)
    (output_path / "videos").mkdir(parents=True, exist_ok=True)
    with open(os.path.join(raw_root_dir, CAPTION_FILE), 'r') as file:
        caption_mp = json.load(file)

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

    tasks = [(i, v, input_path, output_path, fps, caption_mp)
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

def create_frames_gpu(input_root: str,
                      output_root: str, 
                      fps: float = 2.0,
                       max_videos: int | None = None,
                       verify: bool = False):
    convert_videos_to_old_structure(input_root = input_root,
                                    output_root = output_root,
                                    fps = fps,
                                    max_videos = max_videos)

    if verify:
        verify_conversion(output_root)

def main():
    parser = argparse.ArgumentParser(description='GPU‑accelerated video conversion')
    parser.add_argument('--input_root',  required=True, help='Input directory with training/validation/testing folders')
    parser.add_argument('--output_root', required=True, help='Output directory for frames and annotations')
    parser.add_argument('--fps', type=float, default=2.0, help='Frames per second to extract')
    parser.add_argument('--max_videos', type=int, default=None, help='Limit number of videos to process')
    parser.add_argument('--verify', action='store_true', help='Verify conversion after completion')
    args = parser.parse_args()

    create_frames_gpu(input_root = args.input_root,
                      output_root = args.output_root,
                      fps = args.fps,
                      max_videos = args.max_videos,
                      verify = args.verify)

if __name__ == "__main__":
    main()