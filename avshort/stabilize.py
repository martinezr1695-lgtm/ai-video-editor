"""Video stabilization for shaky handheld / walking footage.

Two-pass approach:
  1. Track feature points between consecutive frames and estimate the
     camera motion (x shift, y shift, rotation) for every frame.
  2. Smooth that motion path over time, then warp each frame so the
     camera follows the smooth path instead of the shaky one.

A small zoom hides the black borders that appear after warping.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from avshort.ffmpeg import mux_audio


def _estimate_motion(capture: cv2.VideoCapture, frame_count: int) -> np.ndarray:
    """Return an (N-1, 3) array of per-frame motion: dx, dy, d_angle."""
    success, previous = capture.read()
    if not success:
        raise RuntimeError("Could not read the first frame for stabilization.")

    previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    motions = []

    for _ in range(frame_count - 1):
        success, frame = capture.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dx = dy = da = 0.0

        points = cv2.goodFeaturesToTrack(
            previous_gray, maxCorners=300, qualityLevel=0.01, minDistance=20, blockSize=3
        )

        if points is not None and len(points) >= 10:
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, gray, points, None)
            good_old = points[status.flatten() == 1]
            good_new = next_points[status.flatten() == 1]

            if len(good_old) >= 10:
                # Partial affine = translation + rotation + uniform scale.
                # RANSAC ignores points on moving subjects (like your face).
                matrix, _ = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC)
                if matrix is not None:
                    dx = matrix[0, 2]
                    dy = matrix[1, 2]
                    da = np.arctan2(matrix[1, 0], matrix[0, 0])

        motions.append((dx, dy, da))
        previous_gray = gray

    return np.array(motions, dtype=np.float64)


def _smooth(trajectory: np.ndarray, radius: int) -> np.ndarray:
    """Moving-average smoothing of each column, padded at the edges."""
    window = 2 * radius + 1
    kernel = np.ones(window) / window
    smoothed = np.empty_like(trajectory)

    for column in range(trajectory.shape[1]):
        padded = np.pad(trajectory[:, column], (radius, radius), mode="edge")
        smoothed[:, column] = np.convolve(padded, kernel, mode="valid")

    return smoothed


def stabilize_video(
    video: Path,
    output: Path,
    smoothing_seconds: float = 1.0,
    zoom: float = 1.10,
) -> None:
    """Stabilize `video` and write the result (with original audio) to `output`.

    smoothing_seconds: how much of the camera path to average. Higher = smoother
        but less responsive when you intentionally turn. 1.0s suits walking footage.
    zoom: crop-in factor that hides warped edges. Very shaky input may need ~1.15.
    """
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or width <= 0 or height <= 0 or frame_count < 2:
        capture.release()
        raise RuntimeError("Could not read video metadata for stabilization.")

    # Pass 1: measure the shake.
    motions = _estimate_motion(capture, frame_count)
    trajectory = np.cumsum(motions, axis=0)
    radius = max(1, int(fps * smoothing_seconds / 2))
    smoothed = _smooth(trajectory, radius)
    offsets = smoothed - trajectory

    # Pass 2: apply the corrections.
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    temporary_video = output.parent / f"{output.stem}_stab_frames.mp4"
    writer = cv2.VideoWriter(
        str(temporary_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Could not create temporary video writer.")

    zoom_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 0, zoom)

    try:
        frame_index = 0
        while True:
            success, frame = capture.read()
            if not success:
                break

            # How far the shaky path is from the smooth path at this frame.
            if 0 < frame_index <= len(offsets):
                offset = offsets[frame_index - 1]
            else:
                offset = np.zeros(3)

            ox, oy, oa = offset
            transform = np.array(
                [
                    [np.cos(oa), -np.sin(oa), ox],
                    [np.sin(oa), np.cos(oa), oy],
                ]
            )
            stabilized = cv2.warpAffine(frame, transform, (width, height), borderMode=cv2.BORDER_REFLECT)
            stabilized = cv2.warpAffine(stabilized, zoom_matrix, (width, height))

            writer.write(stabilized)
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    mux_audio(temporary_video, video, output)
    temporary_video.unlink(missing_ok=True)
