# pipeline/utils.py
import logging
import cv2

def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

# ──────────────────────────────────────────────
# Plate-crop helpers (NEW)
# ──────────────────────────────────────────────

def pad_bbox(b, img_w: int, img_h: int, pad: float = 0.18):
    bw = b.x2 - b.x1
    bh = b.y2 - b.y1
    px = int(bw * pad)
    py = int(bh * pad)
    x1 = max(0, int(b.x1 - px))
    y1 = max(0, int(b.y1 - py))
    x2 = min(img_w, int(b.x2 + px))
    y2 = min(img_h, int(b.y2 + py))
    return type(b)(x1, y1, x2, y2)

def sharpness_score(bgr_roi) -> float:
    if bgr_roi is None or bgr_roi.size == 0:
        return 0.0
    gray = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

