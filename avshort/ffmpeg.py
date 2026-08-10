from pathlib import Path
import subprocess


SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def validate_video(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported format '{path.suffix}'. "
            f"Use one of: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    return path


def get_duration(video: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def mux_audio(video_only: Path, source: Path, output: Path) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_only),
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-shortest",
        str(output),
    ]

    finished = subprocess.run(command, capture_output=True, text=True)

    if finished.returncode != 0:
        raise RuntimeError(
            "FFmpeg could not attach audio.\n"
            f"{finished.stderr.strip()}"
        )


def center_crop_vertical(video: Path, output: Path, width: int = 1080, height: int = 1920) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video),
        "-vf",
        f"crop=ih*9/16:ih:(iw-ih*9/16)/2:0,scale={width}:{height}",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        str(output),
    ]

    finished = subprocess.run(command, capture_output=True, text=True)

    if finished.returncode != 0:
        raise RuntimeError(
            "FFmpeg could not create the vertical video.\n"
            f"{finished.stderr.strip()}"
        )
