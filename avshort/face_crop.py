from pathlib import Path

import cv2

from avshort.captions import build_caption_groups, draw_caption, find_font
from avshort.ffmpeg import mux_audio


OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920


def render_face_vertical(
    video: Path,
    output: Path,
    caption_groups: list[dict] | None = None,
    words_per_caption: int = 2,
) -> None:
    capture = cv2.VideoCapture(str(video))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Could not read video metadata.")

    crop_width = int(height * 9 / 16)

    if crop_width > width:
        capture.release()
        raise RuntimeError(
            "Video is too narrow for a 9:16 vertical crop. "
            "Use center crop on a wider source or reframe the shot."
        )

    temporary_video = output.parent / f"{output.stem}_frames.mp4"

    writer = cv2.VideoWriter(
        str(temporary_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (OUTPUT_WIDTH, OUTPUT_HEIGHT),
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Could not create temporary video writer.")

    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    font = None
    if caption_groups:
        font = find_font(max(42, int(OUTPUT_HEIGHT * 0.038)))

    last_center_x = width // 2
    frame_number = 0
    group_index = 0

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(80, 80),
            )

            if len(faces) > 0:
                largest_face = max(faces, key=lambda face: face[2] * face[3])
                x, _, face_width, _ = largest_face
                detected_center_x = x + face_width // 2
                last_center_x = int(0.92 * last_center_x + 0.08 * detected_center_x)

            crop_x = last_center_x - crop_width // 2
            crop_x = max(0, min(crop_x, width - crop_width))

            cropped = frame[0:height, crop_x : crop_x + crop_width]
            vertical_frame = cv2.resize(cropped, (OUTPUT_WIDTH, OUTPUT_HEIGHT))

            if caption_groups and font is not None:
                current_time = frame_number / fps

                while (
                    group_index < len(caption_groups) - 1
                    and current_time > caption_groups[group_index]["end"]
                ):
                    group_index += 1

                current_group = caption_groups[group_index]

                if current_group["start"] <= current_time <= current_group["end"]:
                    vertical_frame = draw_caption(
                        vertical_frame,
                        current_group,
                        current_time,
                        font,
                        OUTPUT_WIDTH,
                        OUTPUT_HEIGHT,
                    )

            writer.write(vertical_frame)
            frame_number += 1
    finally:
        capture.release()
        writer.release()

    mux_audio(temporary_video, video, output)
    temporary_video.unlink(missing_ok=True)


def render_captions_only(
    video: Path,
    output: Path,
    caption_groups: list[dict],
) -> None:
    capture = cv2.VideoCapture(str(video))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Could not read video metadata.")

    temporary_video = output.parent / f"{output.stem}_caption_frames.mp4"

    writer = cv2.VideoWriter(
        str(temporary_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Could not create temporary video writer.")

    font = find_font(max(30, int(height * 0.055)))
    frame_number = 0
    group_index = 0

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            current_time = frame_number / fps

            while (
                group_index < len(caption_groups) - 1
                and current_time > caption_groups[group_index]["end"]
            ):
                group_index += 1

            current_group = caption_groups[group_index]

            if current_group["start"] <= current_time <= current_group["end"]:
                frame = draw_caption(
                    frame,
                    current_group,
                    current_time,
                    font,
                    width,
                    height,
                    y_ratio=0.76,
                    spacing=12,
                )

            writer.write(frame)
            frame_number += 1
    finally:
        capture.release()
        writer.release()

    mux_audio(temporary_video, video, output)
    temporary_video.unlink(missing_ok=True)
