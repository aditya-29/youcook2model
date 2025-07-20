from pathlib import Path
import subprocess
import random

REVERSE_SUFFIX = " in reverse"

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
  return caption + REVERSE_SUFFIX

def FFMPEG_SPEED(src: Path, dst: Path, factor: float, caption: str):
  ffmpeg_speed(src= src, dst= dst, factor= factor)
  if factor == 1.0:
    return caption
  else:
    return caption + f" at {factor}x speed"

def FFMPEG_RANDOM_PICK(
    src1: Path, caption1: str,
    src2: Path, caption2: str,
    dst: Path,
) -> str:
    """
    Randomly copies *either* ``src1`` *or* ``src2`` to ``dst`` (bit‑exact),
    but the returned caption concatenates both: ``"caption1 caption2"``.
    """
    pick_src = random.choice([src1, src2])

    _run(
        [
            "ffmpeg",
            "-y",
            "-i", str(pick_src),
            "-c", "copy",          # fast, loss‑less copy
            str(dst),
        ]
    )

    return f"{caption1} {caption2}"


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
    # Video: PTS scaling; Audio: atempo (supports 0.5 – 2.0 per filter)
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