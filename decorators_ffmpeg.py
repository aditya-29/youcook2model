from pathlib import Path
import subprocess
import random

REVERSE_SUFFIX = " in reverse"

class Captions:
    @staticmethod
    def reverse(caption: str):
        return caption + REVERSE_SUFFIX

    @staticmethod
    def speed(caption: str, factor: float):
        if factor == 1.0:
            return caption
        else:
            return caption + f", played at {factor}x speed"
        

    @staticmethod
    # def random_pick(caption1: str, caption2: str, used_parts: list[int]):
    def random_pick(caption_ls: list[str], used_parts: list[int]):
        used_parts = list(map(lambda x: str(x + 1), used_parts))  # convert to 1-based index
        _prefix = "part {} of ".format(", ".join(used_parts))
        return _prefix + "(" + " ".join(caption_ls) + ")"
    

def _run(cmd: list[str]) -> None:
        """Run ffmpeg quietly; raise if it fails."""
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

def FFMPEG_REVERSE(src: Path, dst: Path, caption: str):
  ffmpeg_reverse(src= src, dst= dst)
  return Captions.reverse(caption)

def FFMPEG_SPEED(src: Path, dst: Path, factor: float, caption: str):
  ffmpeg_speed(src= src, dst= dst, factor= factor)
  return Captions.speed(caption, factor)


def FFMPEG_RANDOM_PICK(src1: str,
                       caption1: str,
                       src2: str,
                       caption2: str, 
                       dst):
    """Randomly choose one of two merging options and execute"""

    def get_video_duration(video_path):
        """Get video duration in seconds using ffprobe"""
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json", 
            "-show_format", str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        import json
        data = json.loads(result.stdout)
        return float(data['format']['duration'])
    
    # Get durations
    vid1_duration = get_video_duration(src1)
    vid2_duration = get_video_duration(src2)
    
    # Randomly choose option
    # option = random.choice([1, 2])
    option = 1
    
    if option == 0:
        # Use filter_complex to cut and concatenate in one command
        start_time = vid1_duration / 2
        cmd = [
            "ffmpeg", "-y", 
            "-i", str(src1), 
            "-i", str(src2),
            "-filter_complex", 
            f"[0:v]trim=start={start_time},setpts=PTS-STARTPTS[v0];"
            f"[0:a]atrim=start={start_time},asetpts=PTS-STARTPTS[a0];"
            f"[v0][a0][1:v][1:a]concat=n=2:v=1:a=1[outv][outa]",
            "-map", "[outv]", "-map", "[outa]",
            str(dst)
        ]
        
    else:        
        # Use filter_complex to cut and concatenate in one command
        duration = vid2_duration / 2
        cmd = [
            "ffmpeg", "-y", 
            "-i", str(src1), 
            "-i", str(src2),
            "-filter_complex", 
            f"[1:v]trim=duration={duration},setpts=PTS-STARTPTS[v1];"
            f"[1:a]atrim=duration={duration},asetpts=PTS-STARTPTS[a1];"
            f"[0:v][0:a][v1][a1]concat=n=2:v=1:a=1[outv][outa]",
            "-map", "[outv]", "-map", "[outa]",
            str(dst)
        ]

    _run(cmd)

    return Captions.random_pick(caption_ls = [caption1, caption2], used_parts=[option])

def ffmpeg_reverse(src: Path, dst: Path) -> None:
    _run(
        [
            "ffmpeg",
            "-y",                        # overwrite
            "-i", str(src),
            "-vf", "reverse",
            "-af", "areverse",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "18",
            str(dst),
        ]
    )

def ffmpeg_speed(src: Path, dst: Path, factor: float) -> None:
    """
    factor < 1 --> slow‑mo  (0.5 => half‑speed)
    factor = 1 --> copy
    factor > 1 --> fast     (2.0 => double‑speed)
    """
    # Video: PTS scaling; Audio: atempo (supports 0.5 – 2.0 per filter)
    if factor == 1.0:
        _run(
            [
                "ffmpeg",
                "-y",
                "-i", str(src),
                "-c", "copy",            # bit‑exact copy
                str(dst),
            ]
        )
        return

    setpts = 1 / factor
    # atempo outside 0.5–2 needs chaining, but our factors are in range
    atempo = factor
    _run(
        [
            "ffmpeg",
            "-y",
            "-i", str(src),
            "-vf", f"setpts={setpts:.6f}*PTS",
            "-af", f"atempo={atempo:.2f}",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "18",
            str(dst),
        ]
    )