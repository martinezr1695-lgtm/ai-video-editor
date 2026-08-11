# AI Video Editor

An AI-powered short-form video editing pipeline built with Python, Whisper, and FFmpeg.

The project takes raw talking-head footage and automatically transforms it into vertical short-form content by removing silence, generating timed captions, and applying face-tracked 9:16 cropping.

## Demo
![AI Video Editor Demo](demo/demo_preview.gif)

## How It Works

1. FFmpeg detects and removes long silent sections.
2. Whisper transcribes speech with word-level timestamps.
3. OpenCV tracks the speaker and reframes the video to 9:16.
4. Styled captions are generated and burned into the video.
5. FFmpeg exports the finished short-form video.

## Architecture

```text
Raw Video
   ↓
Silence Removal
   ↓
Whisper Transcription
   ↓
Face Tracking
   ↓
9:16 Reframing
   ↓
Caption Rendering
   ↓
Final Short


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
