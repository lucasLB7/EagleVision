# pipeline/models.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Protocol



@dataclass(frozen=True)
class FramePacket:
    index: int
    timestamp_s: float
    image: np.ndarray  # BGR image


@dataclass(frozen=True)
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def clamp(self, w: int, h: int) -> "BBox":
        x1 = max(0, min(self.x1, w - 1))
        y1 = max(0, min(self.y1, h - 1))
        x2 = max(0, min(self.x2, w - 1))
        y2 = max(0, min(self.y2, h - 1))
        if x2 <= x1: x2 = min(w - 1, x1 + 1)
        if y2 <= y1: y2 = min(h - 1, y1 + 1)
        return BBox(x1, y1, x2, y2)

@dataclass(frozen=True)
class Detection:
    """A single plate detection in a frame."""
    plate_bbox: BBox
    conf: float = 1.0
    cls: str = "plate"


class PlateDetector(Protocol):
    """Detector plugin interface (YOLO, heuristic, etc.)."""
    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        ...


class PlateRectifier(Protocol):
    """Rectifier plugin interface (homography, contour warp, etc.)."""
    def rectify(self, image_bgr: np.ndarray, plate_bbox: BBox) -> np.ndarray:
        ...


@dataclass(frozen=True)
class TrackedPlate:
    track_id: int
    plate_bbox: BBox


@dataclass(frozen=True)
class PlateObservation:
    track_id: int
    frame_index: int
    timestamp_s: float
    plate_bbox: BBox
    rectified: np.ndarray  # rectified plate image
    quality: float


@dataclass(frozen=True)
class OCRItem:
    track_id: int
    frame_index: int
    quality: float
    text: str
    confidence: float


@dataclass(frozen=True)
class OCRTrackResult:
    track_id: int
    best_text: str
    confidence: float
    items: list[OCRItem]
