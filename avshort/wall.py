"""Then-vs-today scrolling comparison wall.

Rows of images or video clips, each with a small caption underneath
("ROME, 80 AD" / "TODAY"), scroll upward like a feed. The row in the
middle of the screen is sharp and bright; rows above and below fade
into blur and darkness so the viewer's eye stays on the centre.

Rows are described in a JSON file:

    {
      "rows": [
        [{"source": "rome.jpg", "label": "Rome, 80 AD"},
         {"source": "today.mp4", "label": "Today"}],
        ...
      ]
    }
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from avshort.captions import find_font


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

BACKGROUND = (11, 11, 11)  # BGR, near-black
TILE_HEIGHT = 230
CAPTION_HEIGHT = 46
ROW_GAP = 44
CORNER_RADIUS = 14


class Tile:
    """One image or looping video clip, pre-resized to tile size."""

    def __init__(self, source: Path, height: int, fps: float, max_seconds: float = 8.0):
        self.frames: list[np.ndarray] = []

        if source.suffix.lower() in IMAGE_EXTENSIONS:
            image = cv2.imread(str(source))
            if image is None:
                raise RuntimeError(f"Could not read image: {source}")
            self.frames.append(self._fit(image, height))
        else:
            capture = cv2.VideoCapture(str(source))
            if not capture.isOpened():
                raise RuntimeError(f"Could not open video: {source}")
            source_fps = capture.get(cv2.CAP_PROP_FPS) or fps
            step = max(1, round(source_fps / fps))
            index = 0
            while len(self.frames) < int(max_seconds * fps):
                success, frame = capture.read()
                if not success:
                    break
                if index % step == 0:
                    self.frames.append(self._fit(frame, height))
                index += 1
            capture.release()
            if not self.frames:
                raise RuntimeError(f"No frames in video: {source}")

        self.width = self.frames[0].shape[1]

    @staticmethod
    def _fit(image: np.ndarray, height: int) -> np.ndarray:
        """Centre-crop to 4:3, then resize to the tile height."""
        h, w = image.shape[:2]
        target_ratio = 4 / 3
        if w / h > target_ratio:
            new_w = int(h * target_ratio)
            x = (w - new_w) // 2
            image = image[:, x : x + new_w]
        else:
            new_h = int(w / target_ratio)
            y = (h - new_h) // 2
            image = image[y : y + new_h]
        return cv2.resize(image, (int(height * target_ratio), height), interpolation=cv2.INTER_AREA)

    def frame_at(self, index: int) -> np.ndarray:
        return self.frames[index % len(self.frames)]


class Row:
    def __init__(self, items: list[dict], base_dir: Path, fps: float):
        self.tiles = [Tile((base_dir / item["source"]).resolve(), TILE_HEIGHT, fps) for item in items]
        self.labels = [str(item.get("label", "")).upper() for item in items]
        self.width = sum(tile.width for tile in self.tiles)
        self.mask = self._rounded_mask(self.width, TILE_HEIGHT, CORNER_RADIUS)
        self.caption = self._render_caption()

    @staticmethod
    def _rounded_mask(width: int, height: int, radius: int) -> np.ndarray:
        mask = np.zeros((height, width), np.uint8)
        cv2.rectangle(mask, (radius, 0), (width - radius - 1, height - 1), 255, -1)
        cv2.rectangle(mask, (0, radius), (width - 1, height - radius - 1), 255, -1)
        for cx, cy in ((radius, radius), (width - radius - 1, radius),
                       (radius, height - radius - 1), (width - radius - 1, height - radius - 1)):
            cv2.circle(mask, (cx, cy), radius, 255, -1, lineType=cv2.LINE_AA)
        return (mask.astype(np.float32) / 255.0)[..., None]

    def _render_caption(self) -> np.ndarray:
        """Labels centred under each tile, white bold small caps."""
        strip = Image.new("RGB", (self.width, CAPTION_HEIGHT), BACKGROUND[::-1])
        draw = ImageDraw.Draw(strip)
        font = find_font(20)
        x = 0
        for tile, label in zip(self.tiles, self.labels):
            box = draw.textbbox((0, 0), label, font=font)
            text_w = box[2] - box[0]
            draw.text((x + (tile.width - text_w) // 2, 12), label, font=font, fill=(235, 235, 235))
            x += tile.width
        return cv2.cvtColor(np.array(strip), cv2.COLOR_RGB2BGR)

    def render(self, frame_index: int) -> np.ndarray:
        """The row's tiles side by side, corners rounded, caption underneath."""
        strip = np.hstack([tile.frame_at(frame_index) for tile in self.tiles]).astype(np.float32)
        background = np.full_like(strip, BACKGROUND, dtype=np.float32)
        strip = strip * self.mask + background * (1 - self.mask)
        return np.vstack([strip.astype(np.uint8), self.caption])


def _paste(canvas: np.ndarray, block: np.ndarray, x: int, y: int) -> None:
    """Paste block onto canvas at (x, y), clipping anything off-screen."""
    h, w = block.shape[:2]
    H, W = canvas.shape[:2]
    top, left = max(0, y), max(0, x)
    bottom, right = min(H, y + h), min(W, x + w)
    if top >= bottom or left >= right:
        return
    canvas[top:bottom, left:right] = block[top - y : bottom - y, left - x : right - x]


def render_wall(
    config_path: Path,
    output: Path,
    duration: float = 10.0,
    width: int = 1920,
    height: int = 1080,
    fps: float = 30.0,
    scroll_speed: float = 120.0,
    watermark: str | None = "CLIPHOUSE",
) -> None:
    """Render the scrolling wall described by config_path to output (no audio)."""
    config = json.loads(config_path.read_text())
    rows = [Row(items, config_path.parent, fps) for items in config["rows"]]
    if not rows:
        raise RuntimeError("The wall config has no rows.")

    block_height = TILE_HEIGHT + CAPTION_HEIGHT + ROW_GAP
    total_frames = int(duration * fps)

    # Focus mask: 0 in the sharp centre band, rising to 1 toward the top and bottom.
    ys = np.arange(height, dtype=np.float32)
    distance = np.abs(ys - height / 2) / (height / 2)
    focus = np.clip((distance - 0.32) / 0.45, 0, 1)[:, None, None]

    # Vignette: darker toward every edge.
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    radial = np.sqrt(((xx - width / 2) / (width / 2)) ** 2 + ((yy - height / 2) / (height / 2)) ** 2)
    vignette = np.clip(1.15 - 0.45 * radial**2, 0.35, 1.0)[..., None]

    watermark_layer = None
    if watermark:
        layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        font = find_font(40)
        text_width = draw.textbbox((0, 0), watermark, font=font)[2]
        draw.text((width - text_width - 48, 34), watermark, font=font, fill=(200, 200, 200, 170))
        watermark_layer = np.array(layer)

    encoder = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            str(output),
        ],
        stdin=subprocess.PIPE,
    )

    try:
        for frame_index in range(total_frames):
            canvas = np.full((height, width, 3), BACKGROUND, np.uint8)
            # First row starts just below the centre and everything moves up.
            scroll = frame_index / fps * scroll_speed
            first_y = height * 0.55 - scroll

            # Draw every row slot that is on screen, repeating the row list as needed.
            slot = int(np.floor((-block_height - first_y) / block_height))
            while True:
                y = int(first_y + slot * block_height)
                if y > height:
                    break
                if y + block_height >= 0:
                    row = rows[slot % len(rows)]
                    _paste(canvas, row.render(frame_index), (width - row.width) // 2, y)
                slot += 1

            # Blur and darken away from the centre band.
            small = cv2.resize(canvas, (width // 4, height // 4), interpolation=cv2.INTER_AREA)
            blurred = cv2.resize(cv2.GaussianBlur(small, (0, 0), 3), (width, height))
            sharp = canvas.astype(np.float32)
            soft = blurred.astype(np.float32) * 0.35
            frame = sharp * (1 - focus) + soft * focus
            frame = frame * vignette

            if watermark_layer is not None:
                alpha = watermark_layer[..., 3:4].astype(np.float32) / 255.0
                frame = frame * (1 - alpha) + watermark_layer[..., 2::-1].astype(np.float32) * alpha

            encoder.stdin.write(np.clip(frame, 0, 255).astype(np.uint8).tobytes())
    finally:
        encoder.stdin.close()
        encoder.wait()

    if encoder.returncode != 0:
        raise RuntimeError("FFmpeg could not encode the wall video.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render a 'then vs today' scrolling comparison wall.")
    parser.add_argument("config", help="JSON file describing the rows")
    parser.add_argument("-o", "--output", default="wall.mp4")
    parser.add_argument("--duration", type=float, default=10.0, help="Length in seconds (default 10)")
    parser.add_argument("--speed", type=float, default=120.0, help="Scroll speed in pixels/second (default 120)")
    parser.add_argument("--watermark", default="CLIPHOUSE", help="Corner text; pass '' for none")
    args = parser.parse_args()

    render_wall(
        Path(args.config),
        Path(args.output),
        duration=args.duration,
        scroll_speed=args.speed,
        watermark=args.watermark or None,
    )
    print(f"Finished: {args.output}")
