from pathlib import Path
import re
import subprocess


def detect_silences(
    video: Path,
    noise_db: float = -35.0,
    min_duration: float = 2.0,
) -> tuple[list[float], list[float]]:
    command = [
        "ffmpeg",
        "-i",
        str(video),
        "-af",
        f"silencedetect=noise={noise_db}dB:d={min_duration}",
        "-f",
        "null",
        "-",
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    output_text = result.stderr

    starts = [
        float(value)
        for value in re.findall(r"silence_start: ([0-9.]+)", output_text)
    ]
    ends = [
        float(value)
        for value in re.findall(r"silence_end: ([0-9.]+)", output_text)
    ]

    return starts, ends


def build_keep_sections(
    silence_starts: list[float],
    silence_ends: list[float],
    duration: float,
    lead_in: float = 0.15,
    lead_out: float = 0.15,
) -> list[tuple[float, float]]:
    if not silence_starts:
        return [(0.0, duration)]

    keep_sections: list[tuple[float, float]] = []
    current_start = 0.0

    for silence_start, silence_end in zip(silence_starts, silence_ends):
        keep_end = min(duration, silence_start + lead_in)

        if keep_end > current_start:
            keep_sections.append((current_start, keep_end))

        current_start = max(0.0, silence_end - lead_out)

    if current_start < duration:
        keep_sections.append((current_start, duration))

    return keep_sections


def remove_silence(
    video: Path,
    output: Path,
    noise_db: float = -35.0,
    min_duration: float = 2.0,
) -> bool:
    from avshort.ffmpeg import get_duration

    silence_starts, silence_ends = detect_silences(video, noise_db, min_duration)

    if not silence_starts:
        return False

    duration = get_duration(video)
    keep_sections = build_keep_sections(silence_starts, silence_ends, duration)

    filter_parts: list[str] = []

    for index, (start, end) in enumerate(keep_sections):
        filter_parts.append(
            f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[v{index}]"
        )
        filter_parts.append(
            f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{index}]"
        )

    concat_inputs = "".join(f"[v{i}][a{i}]" for i in range(len(keep_sections)))
    filter_parts.append(
        f"{concat_inputs}concat=n={len(keep_sections)}:v=1:a=1[outv][outa]"
    )

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video),
        "-filter_complex",
        ";".join(filter_parts),
        "-map",
        "[outv]",
        "-map",
        "[outa]",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        str(output),
    ]

    finished = subprocess.run(command, capture_output=True, text=True)

    if finished.returncode != 0:
        raise RuntimeError(
            "FFmpeg could not remove silence.\n"
            f"{finished.stderr.strip()}"
        )

    return True
