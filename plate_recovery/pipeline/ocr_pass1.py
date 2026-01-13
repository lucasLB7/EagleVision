from __future__ import annotations

from typing import Dict
from collections import defaultdict

from config import AppConfig
from pipeline.models import PlateObservation, OCRItem, OCRTrackResult

import re

PLATE_RE = re.compile(r"^[A-Z]{3}[0-9]{3}[A-Z]$")

# common OCR confusions
CONFUSIONS = {
    "0": "O", "O": "0",
    "1": "I", "I": "1", "L": "1",
    "2": "Z", "Z": "2",
    "4": "A", "A": "4",
    "5": "S", "S": "5",
    "8": "B", "B": "8",
}

def _normalize_for_plate(t: str) -> str:
    """Uppercase and keep only A–Z/0–9 (OCR cleanup)."""
    t = t.upper()
    return "".join(c for c in t if c.isalnum())


def _plate_score(t: str) -> float:
    """
    Score how well a string matches Kenyan plate shape: LLLDDDL (7 chars).
    Higher is better.
    """
    t = _normalize_for_plate(t)

    # perfect match gets a strong bonus
    if PLATE_RE.match(t):
        return 2.0

    # only score 7-char candidates; reject others
    if len(t) != 7:
        return -1.0

    score = 0.0
    for i, ch in enumerate(t):
        want_letter = i in (0, 1, 2, 6)
        want_digit = i in (3, 4, 5)

        if want_letter and ch.isalpha():
            score += 0.25
        elif want_digit and ch.isdigit():
            score += 0.25
        else:
            # allow common OCR confusions as partial credit
            score += 0.10 if ch in CONFUSIONS else -0.20

    return score


# ---------- public entry point (THIS is what main.py imports) ----------
def ocr_first_pass(
    selected: list[PlateObservation],
    cfg: AppConfig,
) -> Dict[int, OCRTrackResult]:
    return _easyocr_pass(selected, cfg)


# ---------- internal implementation ----------
def _easyocr_pass(
    selected: list[PlateObservation],
    cfg: AppConfig,
) -> Dict[int, OCRTrackResult]:
    import easyocr

    reader = easyocr.Reader(list(cfg.ocr_langs), gpu=False)
    by_track: dict[int, list[OCRItem]] = defaultdict(list)

    DEBUG_EVERY = 25
    MIN_LEN = 3  # drop junk partials like "1", "424", "42A"

    for i, obs in enumerate(selected):
        img = obs.rectified
        raw = reader.readtext(img, detail=1, paragraph=False)

        if i % DEBUG_EVERY == 0:
            print(
                f"[easyocr] track={obs.track_id} frame={obs.frame_index} "
                f"q={obs.quality:.3f} raw={raw[:3]}"
            )

        if not raw:
            continue

        # choose highest-confidence line from EasyOCR output
        best = max(raw, key=lambda x: float(x[2]))
        text = _normalize_text(str(best[1]))
        conf = float(best[2])

        if len(text) < MIN_LEN:
            continue

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



# ---------- helpers ----------
def _normalize_text(t: str) -> str:
    t = t.upper()
    t = "".join(c for c in t if c.isalnum())
    return t


def _pack_results(by_track: dict[int, list[OCRItem]]) -> Dict[int, OCRTrackResult]:
    results: Dict[int, OCRTrackResult] = {}

    for track_id, items in by_track.items():
        if not items:
            continue

        def rank(it: OCRItem) -> float:
            # normalize for plate scoring
            t = _normalize_for_plate(it.text)
            # combine OCR confidence + patch quality + plate-shape score
            return (0.65 * it.confidence) + (0.20 * it.quality) + (0.15 * _plate_score(t))

        best = max(items, key=rank)

        results[track_id] = OCRTrackResult(
            track_id=track_id,
            best_text=best.text,
            confidence=best.confidence,
            items=sorted(items, key=lambda x: rank(x), reverse=True),
        )

    if not results:
        print("OCR produced no results.")

    return results


    return results
