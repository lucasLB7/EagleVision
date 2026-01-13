# pipeline/plate_rectify.py
from __future__ import annotations

import cv2
import numpy as np

from .models import BBox


def rectify_plate(frame: np.ndarray, bbox: BBox, target_size: tuple[int, int], pad: int = 0) -> np.ndarray:
    """
    V1: crop + resize.
    V2: replace with perspective rectification using plate corner points / homography.
    """
    h, w = frame.shape[:2]
    b = bbox.clamp(w, h)

    x1 = max(0, b.x1 - pad)
    y1 = max(0, b.y1 - pad)
    x2 = min(w, b.x2 + pad)
    y2 = min(h, b.y2 + pad)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_CUBIC)
    return resized
