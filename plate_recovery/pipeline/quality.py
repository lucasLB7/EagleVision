# pipeline/quality.py
from __future__ import annotations

import cv2
import numpy as np
from collections import defaultdict

from .models import PlateObservation


def score_plate_patch(img: np.ndarray) -> float:
    """
    Simple quality heuristic (0..1):
    - Sharpness via Laplacian variance
    - Penalize over/under exposure
    """
    if img is None or img.size == 0:
        return 0.0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # sharpness
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sharp = float(lap.var())

    # exposure (prefer mid-range)
    mean = float(gray.mean())
    exposure_penalty = abs(mean - 128.0) / 128.0  # 0 best, 1 worst

    # normalize sharpness into 0..1-ish
    sharp_norm = min(1.0, sharp / 500.0)

    q = 0.75 * sharp_norm + 0.25 * (1.0 - exposure_penalty)
    return float(max(0.0, min(1.0, q)))


def select_best_observations(
    observations: list[PlateObservation],
    per_track_top_k: int,
    min_quality: float,
) -> list[PlateObservation]:
    by_track = defaultdict(list)

    for obs in observations:
        if obs.quality >= min_quality:
            by_track[obs.track_id].append(obs)

    selected: list[PlateObservation] = []
    for tid, obs_list in by_track.items():
        obs_list.sort(key=lambda o: o.quality, reverse=True)
        selected.extend(obs_list[:per_track_top_k])

    # deterministic ordering (optional)
    selected.sort(key=lambda o: (o.track_id, -o.quality, o.frame_index))
    return selected
