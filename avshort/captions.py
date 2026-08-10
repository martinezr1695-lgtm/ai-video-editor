from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def find_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_paths = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]

    for font_path in font_paths:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size)

    return ImageFont.load_default()


def build_caption_groups(segments, words_per_group: int = 3) -> list[dict]:
    words = []

    for segment in segments:
        for word_data in segment.get("words", []):
            word = word_data["word"].strip()

            if word:
                words.append(
                    {
                        "text": word,
                        "start": float(word_data["start"]),
                        "end": float(word_data["end"]),
                    }
                )

    groups = []

    for index in range(0, len(words), words_per_group):
        group_words = words[index : index + words_per_group]

        if not group_words:
            continue

        groups.append(
            {
                "start": group_words[0]["start"],
                "end": group_words[-1]["end"],
                "words": group_words,
            }
        )

    return groups


def draw_caption(
    frame,
    caption_group: dict,
    current_time: float,
    font,
    width: int,
    height: int,
    y_ratio: float = 0.68,
    spacing: int = 18,
) -> np.ndarray:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb_frame)
    draw = ImageDraw.Draw(image)

    words = caption_group["words"]
    word_boxes = []

    for word_data in words:
        text = word_data["text"].upper()
        box = draw.textbbox((0, 0), text, font=font, stroke_width=3)
        word_boxes.append((box[2] - box[0], box[3] - box[1]))

    total_width = sum(box[0] for box in word_boxes) + spacing * max(0, len(words) - 1)
    x = (width - total_width) // 2
    y = int(height * y_ratio)

    for word_data, (word_width, _) in zip(words, word_boxes):
        is_current_word = word_data["start"] <= current_time <= word_data["end"]
        fill_color = "yellow" if is_current_word else "white"

        draw.text(
            (x, y),
            word_data["text"].upper(),
            font=font,
            fill=fill_color,
            stroke_width=4,
            stroke_fill="black",
        )

        x += word_width + spacing

    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
