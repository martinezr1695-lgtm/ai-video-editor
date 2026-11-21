"""Transitions between clips: glitch and slide/push.

Joins clips end to end. Around every cut, frames get progressively
"broken" and then recover, peaking exactly on the cut:

  - whip blur: horizontal motion blur, like a fast camera swipe
  - RGB split: red and blue channels pulled apart
  - slice displacement: horizontal bands torn sideways
  - block corruption: chunks of the picture copied to the wrong place
  - flash: a brightness pop on the cut frame itself

Push: the next clip slides in and shoves the current one off screen,
with an ease-in-out move and motion blur at top speed. The clips
overlap during the move and their audio crossfades.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np


def _fit(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Centre-crop to the target aspect ratio, then resize."""
    h, w = frame.shape[:2]
    target = width / height
    if abs(w / h - target) > 0.01:
        if w / h > target:
            new_w = int(h * target)
            x = (w - new_w) // 2
            frame = frame[:, x : x + new_w]
        else:
            new_h = int(w / target)
            y = (h - new_h) // 2
            frame = frame[y : y + new_h]
    if frame.shape[1] != width or frame.shape[0] != height:
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame


def glitch_frame(frame: np.ndarray, amount: float, rng: np.random.Generator, direction: int = 1) -> np.ndarray:
    """Apply the glitch look. amount: 0 = untouched, 1 = fully broken."""
    if amount <= 0:
        return frame

    height, width = frame.shape[:2]
    out = frame.copy()

    # Whip blur: horizontal motion blur, plus a slide in the swipe direction.
    length = 1 + int(70 * amount)
    if length > 2:
        kernel = np.zeros((1, length), np.float32)
        kernel[0, :] = 1.0 / length
        out = cv2.filter2D(out, -1, kernel)
        slide = int(direction * 0.08 * width * amount)
        out = np.roll(out, slide, axis=1)

    # RGB split: shift red one way and blue the other.
    shift = int(28 * amount)
    if shift:
        blue, green, red = cv2.split(out)
        out = cv2.merge([np.roll(blue, -shift, axis=1), green, np.roll(red, shift, axis=1)])

    # Slice displacement: tear random horizontal bands sideways.
    if amount > 0.25:
        for _ in range(int(4 + 10 * amount)):
            band = int(rng.integers(4, max(6, int(height * 0.08))))
            y = int(rng.integers(0, height - band))
            offset = int(rng.normal(0, 0.12 * width * amount))
            out[y : y + band] = np.roll(out[y : y + band], offset, axis=1)

    # Block corruption: copy a few chunks to the wrong place, with a colour tint.
    if amount > 0.5:
        for _ in range(int(2 + 6 * amount)):
            bw = int(rng.integers(width // 20, width // 5))
            bh = int(rng.integers(height // 30, height // 10))
            sx, sy = int(rng.integers(0, width - bw)), int(rng.integers(0, height - bh))
            dx, dy = int(rng.integers(0, width - bw)), int(rng.integers(0, height - bh))
            block = out[sy : sy + bh, sx : sx + bw].astype(np.int16)
            block[..., int(rng.integers(0, 3))] += 70
            out[dy : dy + bh, dx : dx + bw] = np.clip(block, 0, 255).astype(np.uint8)

    # Flash right on the cut.
    if amount > 0.8:
        flash = int(90 * (amount - 0.8) / 0.2)
        out = cv2.add(out, np.full_like(out, flash))

    return out


def _frame_counts(clips: list[Path]) -> tuple[list[int], float]:
    counts = []
    fps = 0.0
    for clip in clips:
        capture = cv2.VideoCapture(str(clip))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {clip}")
        if not fps:
            fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        counts.append(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        capture.release()
    return counts, fps


def _has_audio(clip: Path) -> bool:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", str(clip)],
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def glitch_join(
    clips: list[Path],
    output: Path,
    transition_seconds: float = 0.4,
    strength: float = 1.0,
    seed: int = 0,
) -> None:
    """Join clips with a glitch transition at every cut. Output matches the first clip's size and fps."""
    if len(clips) < 2:
        raise ValueError("Give at least two clips to join.")

    first = cv2.VideoCapture(str(clips[0]))
    width = int(first.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2 * 2
    height = int(first.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2 * 2
    first.release()

    counts, fps = _frame_counts(clips)
    cuts = np.cumsum(counts)[:-1]  # global frame index where each new clip starts
    half = max(1, int(transition_seconds * fps / 2))
    rng = np.random.default_rng(seed)

    video_only = output.with_name(f"{output.stem}_glitch_frames.mp4")
    encoder = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            str(video_only),
        ],
        stdin=subprocess.PIPE,
    )

    index = 0
    try:
        for clip_number, clip in enumerate(clips):
            capture = cv2.VideoCapture(str(clip))
            direction = 1 if clip_number % 2 == 0 else -1
            while True:
                success, frame = capture.read()
                if not success:
                    break
                frame = _fit(frame, width, height)

                # Distance to the nearest cut decides how broken this frame is.
                distance = int(np.min(np.abs(cuts - index))) if len(cuts) else half + 1
                if distance <= half:
                    amount = (1 - distance / (half + 1)) * strength
                    frame = glitch_frame(frame, min(1.0, amount), rng, direction)

                encoder.stdin.write(frame.tobytes())
                index += 1
            capture.release()
    finally:
        encoder.stdin.close()
        encoder.wait()

    if encoder.returncode != 0:
        raise RuntimeError("FFmpeg could not encode the joined video.")

    _join_audio(clips, counts, fps, video_only, output)
    video_only.unlink(missing_ok=True)


def _join_audio(
    clips: list[Path], counts: list[int], fps: float, video_only: Path, output: Path, overlap_frames: int = 0
) -> None:
    """Join each clip's audio (silence for clips without any) and attach it.

    With overlap_frames > 0 the clips overlap at each cut, so their audio
    crossfades over the same stretch to stay in sync with the picture.
    """
    inputs = ["-i", str(video_only)]
    input_number = 0
    labels = []
    filters = []
    for number, (clip, count) in enumerate(zip(clips, counts)):
        duration = count / fps
        if _has_audio(clip):
            inputs += ["-i", str(clip)]
        else:
            inputs += ["-f", "lavfi", "-t", f"{duration:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
        input_number += 1
        source = f"[{input_number}:a]"
        # Trim or pad every piece to exactly its video length so sound stays in sync.
        filters.append(
            f"{source}aresample=48000,aformat=channel_layouts=stereo,"
            f"apad,atrim=0:{duration:.3f},asetpts=N/SR/TB[a{number}]"
        )
        labels.append(f"[a{number}]")

    if overlap_frames > 0:
        fade = overlap_frames / fps
        chain = labels[0]
        for number, label in enumerate(labels[1:], start=1):
            out_label = "[aout]" if number == len(labels) - 1 else f"[x{number}]"
            filters.append(f"{chain}{label}acrossfade=d={fade:.3f}:c1=tri:c2=tri{out_label}")
            chain = out_label
        graph = ";".join(filters)
    else:
        graph = ";".join(filters) + f";{''.join(labels)}concat=n={len(labels)}:v=0:a=1[aout]"
    command = ["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", graph,
               "-map", "0:v", "-map", "[aout]", "-c:v", "copy", "-c:a", "aac", "-shortest", str(output)]

    finished = subprocess.run(command, capture_output=True, text=True)
    if finished.returncode != 0:
        raise RuntimeError(f"FFmpeg could not join the audio.\n{finished.stderr.strip()}")


def _ease_in_out(t: float) -> float:
    """Cubic ease: slow start, fast middle, slow finish."""
    return 4 * t**3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def push_frame(old: np.ndarray, new: np.ndarray, progress: float, direction: str = "left") -> np.ndarray:
    """New clip shoves the old one off screen. progress 0 = all old, 1 = all new."""
    height, width = old.shape[:2]
    eased = _ease_in_out(progress)
    # Speed of the move right now (slope of the ease curve), used for motion blur.
    step = 0.02
    speed = abs(_ease_in_out(min(1, progress + step)) - _ease_in_out(max(0, progress - step))) / (2 * step)

    horizontal = direction in ("left", "right")
    size = width if horizontal else height
    offset = int(round(eased * size))

    # Lay both frames side by side (or stacked) and take a window that slides across them.
    if direction == "left":
        strip, start = np.hstack([old, new]), offset
    elif direction == "right":
        strip, start = np.hstack([new, old]), size - offset
    elif direction == "up":
        strip, start = np.vstack([old, new]), offset
    elif direction == "down":
        strip, start = np.vstack([new, old]), size - offset
    else:
        raise ValueError("direction must be left, right, up or down")

    frame = strip[:, start : start + width] if horizontal else strip[start : start + height]

    # Motion blur along the direction of travel, strongest at top speed.
    length = int(0.035 * size * speed)
    if length > 2:
        kernel = np.full((1, length) if horizontal else (length, 1), 1.0 / length, np.float32)
        frame = cv2.filter2D(frame, -1, kernel)

    return np.ascontiguousarray(frame)


def push_join(
    clips: list[Path],
    output: Path,
    transition_seconds: float = 0.5,
    direction: str = "left",
) -> None:
    """Join clips with a slide/push at every cut. The clips overlap for transition_seconds."""
    if len(clips) < 2:
        raise ValueError("Give at least two clips to join.")

    first = cv2.VideoCapture(str(clips[0]))
    width = int(first.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2 * 2
    height = int(first.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2 * 2
    first.release()

    counts, fps = _frame_counts(clips)
    # The overlap can't be longer than a third of the shortest clip.
    overlap = max(1, min(int(transition_seconds * fps), min(counts) // 3))

    video_only = output.with_name(f"{output.stem}_push_frames.mp4")
    encoder = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            str(video_only),
        ],
        stdin=subprocess.PIPE,
    )

    held: list[np.ndarray] = []  # last frames of the previous clip, waiting to be pushed out
    try:
        for clip_number, (clip, count) in enumerate(zip(clips, counts)):
            is_last = clip_number == len(clips) - 1
            capture = cv2.VideoCapture(str(clip))
            frames_read = 0
            tail: list[np.ndarray] = []
            while True:
                success, frame = capture.read()
                if not success:
                    break
                frame = _fit(frame, width, height)

                if frames_read < len(held):
                    # Transition: this clip's opening frames push the held frames out.
                    progress = (frames_read + 1) / (len(held) + 1)
                    frame_out = push_frame(held[frames_read], frame, progress, direction)
                elif not is_last and frames_read >= count - overlap:
                    # Hold the ending frames for the next transition.
                    tail.append(frame)
                    frames_read += 1
                    continue
                else:
                    frame_out = frame

                encoder.stdin.write(frame_out.tobytes())
                frames_read += 1
            capture.release()
            held = tail
    finally:
        encoder.stdin.close()
        encoder.wait()

    if encoder.returncode != 0:
        raise RuntimeError("FFmpeg could not encode the joined video.")

    _join_audio(clips, counts, fps, video_only, output, overlap_frames=overlap)
    video_only.unlink(missing_ok=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Join clips with glitch or slide/push transitions.")
    parser.add_argument("clips", nargs="+", help="Two or more video files, in order")
    parser.add_argument("-o", "--output", default="joined.mp4")
    parser.add_argument("--style", choices=("glitch", "push"), default="glitch", help="Transition type (default glitch)")
    parser.add_argument("--length", type=float, default=None,
                        help="Transition length in seconds (default 0.4 glitch, 0.5 push)")
    parser.add_argument("--strength", type=float, default=1.0, help="Glitch only: 0.5 = subtle, 1.0 = full (default)")
    parser.add_argument("--direction", choices=("left", "right", "up", "down"), default="left",
                        help="Push only: which way the old clip leaves (default left)")
    args = parser.parse_args()

    clip_paths = [Path(c) for c in args.clips]
    if args.style == "push":
        push_join(clip_paths, Path(args.output), args.length or 0.5, args.direction)
    else:
        glitch_join(clip_paths, Path(args.output), args.length or 0.4, args.strength)
    print(f"Finished: {args.output}")
