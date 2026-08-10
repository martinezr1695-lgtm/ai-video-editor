#!/usr/bin/env python3
"""CLI for the AI short-form video pipeline."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import whisper

from avshort import __version__
from avshort.captions import build_caption_groups
from avshort.face_crop import render_captions_only, render_face_vertical
from avshort.ffmpeg import center_crop_vertical, validate_video
from avshort.silence import remove_silence


def transcribe(video: Path, model_name: str = "base") -> list[dict]:
    print(f"Loading Whisper ({model_name})...")
    model = whisper.load_model(model_name)

    print("Transcribing with word timestamps...")
    result = model.transcribe(str(video), word_timestamps=True, language="en")

    groups = build_caption_groups(result["segments"])

    if not groups:
        raise RuntimeError("Whisper did not find any spoken words in this video.")

    return groups


def default_output(video: Path, suffix: str) -> Path:
    return video.parent / f"{video.stem}_{suffix}.mp4"


def run_pipeline(args: argparse.Namespace) -> Path:
    video = validate_video(Path(args.input).expanduser().resolve())
    working_video = video
    temp_files: list[Path] = []

    try:
        if args.silence or args.short:
            print("Removing silence...")
            silenced = Path(tempfile.mkstemp(suffix="_no_silence.mp4")[1])
            temp_files.append(silenced)

            if remove_silence(working_video, silenced):
                working_video = silenced
                print("Silence removed.")
            else:
                print("No long silences detected; skipping cut.")
                silenced.unlink(missing_ok=True)
                temp_files.remove(silenced)

        caption_groups = None

        if args.captions or args.short:
            caption_groups = transcribe(working_video, args.model)

        if args.short:
            output = Path(args.output).expanduser().resolve() if args.output else default_output(video, "short")
            print("Rendering face-tracked vertical short with captions...")
            render_face_vertical(
                working_video,
                output,
                caption_groups=caption_groups,
                words_per_caption=args.words,
            )
            return output

        if args.vertical == "face":
            output = Path(args.output).expanduser().resolve() if args.output else default_output(video, "face_vertical")
            print("Rendering face-tracked vertical crop...")
            render_face_vertical(working_video, output)
            return output

        if args.vertical == "center":
            output = Path(args.output).expanduser().resolve() if args.output else default_output(video, "vertical")
            print("Rendering center vertical crop...")
            center_crop_vertical(working_video, output)
            return output

        if args.captions:
            if caption_groups is None:
                caption_groups = transcribe(working_video, args.model)

            output = Path(args.output).expanduser().resolve() if args.output else default_output(video, "captions")
            print("Burning styled word captions...")
            render_captions_only(working_video, output, caption_groups)
            return output

        if args.silence:
            output = Path(args.output).expanduser().resolve() if args.output else default_output(video, "no_silence")
            if working_video == video:
                if not remove_silence(video, output):
                    raise RuntimeError("No long silences detected in this video.")
            else:
                output.write_bytes(working_video.read_bytes())
            return output

        raise RuntimeError("No operation selected. Try --short or pass individual flags.")

    finally:
        for temp_file in temp_files:
            if temp_file.exists() and temp_file != working_video:
                temp_file.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline.py",
        description="Turn talking-head footage into short-form vertical video with AI captions.",
    )
    parser.add_argument("input", help="Path to input video (.mp4, .mov, .mkv, .avi, .webm)")
    parser.add_argument(
        "-o",
        "--output",
        help="Output video path (default: <input>_<operation>.mp4 next to source file)",
    )
    parser.add_argument(
        "--short",
        action="store_true",
        help="Full short-form pipeline: silence removal + face crop + karaoke captions",
    )
    parser.add_argument(
        "--silence",
        action="store_true",
        help="Remove long silent sections with FFmpeg",
    )
    parser.add_argument(
        "--vertical",
        choices=("center", "face"),
        help="Crop to 9:16 — center crop or face-tracked crop",
    )
    parser.add_argument(
        "--captions",
        action="store_true",
        help="Burn in word-level karaoke captions with Whisper",
    )
    parser.add_argument(
        "--model",
        default="base",
        choices=("tiny", "base", "small", "medium", "large"),
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--words",
        type=int,
        default=2,
        help="Words per caption group in short mode (default: 2)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"avshort {__version__}",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not any((args.short, args.silence, args.vertical, args.captions)):
        parser.print_help()
        print("\nTip: start with  python pipeline.py your_video.mp4 --short")
        return 1

    try:
        output = run_pipeline(args)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Finished: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
