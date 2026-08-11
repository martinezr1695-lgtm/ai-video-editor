# AI Video Editor

An AI-powered short-form video editing pipeline built with Python, Whisper, and FFmpeg.

The project takes raw talking-head footage and automatically transforms it into vertical short-form content by removing silence, generating timed captions, and applying face-tracked 9:16 cropping.

## Demo

![AI Video Editor Demo](demo/demo_preview.gif)

## Features

- Automatic silence and dead-space removal
- Whisper transcription with word-level timestamps
- Styled burned-in captions
- Face-tracked vertical 9:16 cropping
- Center-crop vertical mode
- Modular Python architecture
- Command-line interface for running individual editing steps or the full pipeline

## Tech Stack

- Python
- FFmpeg
- OpenAI Whisper
- OpenCV
- NumPy

## Run the Full Pipeline

Activate the virtual environment:

```bash
source .venv/bin/activate

python pipeline.py your_video.mp4 --short