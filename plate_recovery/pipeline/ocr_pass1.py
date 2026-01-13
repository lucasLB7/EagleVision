# pipeline/ocr_pass1.py
from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List

from config import AppConfig
from .models import PlateObservation, OCRItem, OCRTrackResult


PLATE_CLEAN_RE = re.compile(r"[^A-Z0-9]+")


def _normalize_text(s: str) -> str:
    s = s.upper().strip()
    s = PLATE_CLEAN_RE.sub("", s)
    return s


def ocr_first_pass(selected: list[PlateObservation], cfg: AppConfig) -> Dict[int, OCRTrackResult]:
    """
    V1:
    - run OCR on each selected rectified image
    - return per-track best guess (simple confidence max)
    Later:
    - character-level probabilities + aggregation + priors
    """
    engine = cfg.ocr_engine.lower()
    if engine == "easyocr":
        return _easyocr_pass(selected, cfg)
    elif engine == "paddleocr":
        return _paddleocr_pass(selected, cfg)
    else:
        raise ValueError(f"Unknown OCR engine: {cfg.ocr_engine}")


def _easyocr_pass(selected: list[PlateObservation], cfg: AppConfig) -> Dict[int, OCRTrackResult]:
    # Lazy import so the project runs even if OCR libs not installed yet.
    import easyocr

    reader = easyocr.Reader(list(cfg.ocr_langs), gpu=False)

    by_track: dict[int, list[OCRItem]] = defaultdict(list)

    for obs in selected:
        # easyocr returns list of (bbox, text, conf)
        res = reader.readtext(obs.rectified)
        if not res:
            continue
        # take best line
        best = max(res, key=lambda x: float(x[2]))
        text = _normalize_text(str(best[1]))
        conf = float(best[2])

        if text:
            by_track[obs.track_id].append(
                OCRItem(
                    track_id=obs.track_id,
                    frame_index=obs.frame_index,
                    quality=obs.quality,
                    text=text,
                    confidence=conf,
                )
            )

    return _pack_results(by_track)


def _paddleocr_pass(selected: list[PlateObservation], cfg: AppConfig) -> Dict[int, OCRTrackResult]:
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(use_angle_cls=True, lang="en")  # keep simple for now
    by_track: dict[int, list[OCRItem]] = defaultdict(list)

    for obs in selected:
        out = ocr.ocr(obs.rectified, cls=True)
        if not out or not out[0]:
            continue
        # out[0] is list of [bbox, (text, conf)]
        best = max(out[0], key=lambda x: float(x[1][1]))
        text = _normalize_text(str(best[1][0]))
        conf = float(best[1][1])
        if text:
            by_track[obs.track_id].append(
                OCRItem(
                    track_id=obs.track_id,
                    frame_index=obs.frame_index,
                    quality=obs.quality,
                    text=text,
                    confidence=conf,
                )
            )

    return _pack_results(by_track)


def _pack_results(by_track: dict[int, list[OCRItem]]) -> Dict[int, OCRTrackResult]:
    results: dict[int, OCRTrackResult] = {}
    for tid, items in by_track.items():
        items.sort(key=lambda x: (x.confidence, x.quality), reverse=True)
        if items:
            best = items[0]
            results[tid] = OCRTrackResult(
                track_id=tid,
                best_text=best.text,
                confidence=best.confidence,
                items=items,
            )
    return results
