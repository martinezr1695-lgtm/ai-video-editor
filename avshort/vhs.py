"""Retro VHS / CRT look, built entirely from FFmpeg filters.

Layers, in order:
  - chromatic aberration: red and blue channels pushed apart
  - punchier contrast and saturation
  - soft glow (a blurred copy blended on top with "screen")
  - film grain that changes every frame
  - horizontal scanlines
  - dark vignette around the edges
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def vhs_filter(strength: float = 1.0) -> str:
    """Build the FFmpeg filter graph. strength scales every effect (0.5 = subtle, 1.5 = heavy)."""
    shift = max(1, round(5 * strength))
    grain = round(10 * strength)
    glow = min(0.6, 0.25 * strength)
    lines = min(0.6, 0.3 * strength)

    return (
        f"[0:v]rgbashift=rh=-{shift}:bh={shift},"
        f"eq=contrast=1.08:saturation={1 + 0.1 * strength:.2f},"
        "format=gbrp,split[base][blur];"
        "[blur]gblur=sigma=12[glow];"
        f"[base][glow]blend=all_mode=screen:all_opacity={glow:.2f},"
        f"noise=c0s={grain}:allf=t,"
        f"drawgrid=w=iw:h=4:t=1:c=black@{lines:.2f},"
        "vignette=angle=PI/4.5,"
        "format=yuv420p[v]"
    )


def apply_vhs(video: Path, output: Path, strength: float = 1.0) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video),
        "-filter_complex",
        vhs_filter(strength),
        "-map",
        "[v]",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-c:a",
        "copy",
        str(output),
    ]

    finished = subprocess.run(command, capture_output=True, text=True)

    if finished.returncode != 0:
        raise RuntimeError(f"FFmpeg could not apply the VHS look.\n{finished.stderr.strip()}")
