from pathlib import Path
import subprocess
import random

class Captions:
    @staticmethod
    def reverse(caption: str):
        templates = [
            f"{caption} in reverse",
            f"a video of {caption} played in reverse",
            f"{caption} played backwards",
            f"reverse playback of {caption}",
            f"{caption} shown in reverse order"
        ]
        return random.choice(templates)

    @staticmethod
    def speed(caption: str, factor: float):
        if factor == 1.0:
            return caption
        elif factor < 1.0:
            # Slow motion templates
            templates = [
                f"{caption}, played at {factor}x speed",
                f"slow motion version of {caption}",
                f"{caption} in slow motion",
                f"{caption} slowed down to {factor}x speed",
                f"{caption} at reduced speed"
            ]
        else:
            # Fast motion templates
            templates = [
                f"{caption}, played at {factor}x speed",
                f"fast-forwarded {caption}",
                f"{caption} in fast motion",
                f"{caption} sped up to {factor}x speed",
                f"{caption} at accelerated speed"
            ]
        return random.choice(templates)
    
    @staticmethod
    def temporal_order(caption1: str, caption2: str, order: str):
        """Generate caption for temporal ordering"""
        connectors = ["followed by", "then", "and then"]
        connector = random.choice(connectors)
        
        if order == "A_then_B":
            return f"{caption1} {connector} {caption2}"
        else:  # B_then_A
            return f"{caption2} {connector} {caption1}"


def _run(cmd: list[str]) -> None:
    """Run ffmpeg quietly; raise if it fails."""
    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

def FFMPEG_REVERSE(src: Path, dst: Path, caption: str):
    ffmpeg_reverse(src=src, dst=dst)
    return Captions.reverse(caption)

def FFMPEG_SPEED(src: Path, dst: Path, factor: float, caption: str):
    ffmpeg_speed(src=src, dst=dst, factor=factor)
    return Captions.speed(caption, factor)

def FFMPEG_TEMPORAL_ORDER(src1: str, caption1: str, src2: str, caption2: str, dst_a_then_b: str, dst_b_then_a: str):
    """
    Generate both temporal orderings: A then B, and B then A
    Returns a tuple of (caption_a_then_b, caption_b_then_a)
    """
    
    # Create A then B version
    cmd_a_then_b = [
        "ffmpeg", "-y",
        "-i", str(src1),
        "-i", str(src2),
        "-filter_complex",
        "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[outv][outa]",
        "-map", "[outv]", "-map", "[outa]",
        str(dst_a_then_b)
    ]
    _run(cmd_a_then_b)
    
    # Create B then A version
    cmd_b_then_a = [
        "ffmpeg", "-y",
        "-i", str(src2),
        "-i", str(src1),
        "-filter_complex",
        "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[outv][outa]",
        "-map", "[outv]", "-map", "[outa]",
        str(dst_b_then_a)
    ]
    _run(cmd_b_then_a)
    
    # Generate captions for both orderings
    caption_a_then_b = Captions.temporal_order(caption1, caption2, "A_then_B")
    caption_b_then_a = Captions.temporal_order(caption1, caption2, "B_then_A")
    
    return caption_a_then_b, caption_b_then_a

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
    factor < 1 --> slow‑mo  (0.5 => half‑speed)
    factor = 1 --> copy
    factor > 1 --> fast     (2.0 => double‑speed)
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