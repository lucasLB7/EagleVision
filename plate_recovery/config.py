# config.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    # IO
    video_path: Path
    output_dir: Path

    # Frame sampling
    frame_stride: int = 2
    max_frames: int | None = None

    # Plate crop/rectify
    rectify_target_size: tuple[int, int] = (320, 96)  # width, height
    plate_pad_px: int = 8

    # Quality selection
    per_track_top_k: int = 12
    min_quality_threshold: float = 0.25  # tune later

    # Logging
    logging_level: str = "INFO"

    # OCR engine settings (we’ll swap later)
    ocr_engine: str = "easyocr"  # or "paddleocr"
    ocr_langs: tuple[str, ...] = ("en",)
