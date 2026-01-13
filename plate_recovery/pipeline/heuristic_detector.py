from __future__ import annotations
import cv2
import numpy as np
from .models import Detection, BBox

class HeuristicPlateDetector:
    def __init__(self, top_k: int = 12):
        self.top_k = top_k

    def _v1_candidates(self, gray: np.ndarray) -> list[BBox]:
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 80, 180)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes: list[BBox] = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            ar = w / float(h + 1e-6)
            if 1.8 < ar < 7.0 and w > 40 and h > 12:
                boxes.append(BBox(x, y, x + w, y + h))
        return boxes

    def _v2_candidates(self, gray: np.ndarray) -> list[BBox]:
        H, W = gray.shape[:2]
        frame_area = float(W * H)

        grayb = cv2.GaussianBlur(gray, (5, 5), 0)

        grad_x = cv2.Sobel(grayb, cv2.CV_16S, 1, 0, ksize=3)
        abs_x = cv2.convertScaleAbs(grad_x)

        # slightly more stable than pure Otsu in noisy CCTV
        _, bw = cv2.threshold(abs_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kw = max(9, (W // 90) | 1)
        kh = max(3, (H // 200) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

        bw = cv2.dilate(bw, None, iterations=1)

        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = max(120, int(0.00012 * frame_area))
        max_area = 0.35 * frame_area

        boxes: list[BBox] = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if area < min_area or area > max_area:
                continue
            ar = w / float(h + 1e-6)
            if 1.5 < ar < 8.0:
                boxes.append(BBox(x, y, x + w, y + h))
        return boxes

    def _score(self, gray: np.ndarray, box: BBox) -> float:
        H, W = gray.shape[:2]
        x1 = max(0, int(box.x1)); y1 = max(0, int(box.y1))
        x2 = min(W, int(box.x2)); y2 = min(H, int(box.y2))  # NOTE: use W/H not W-1/H-1

        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0

        gx = cv2.Sobel(roi, cv2.CV_16S, 1, 0, ksize=3)
        gy = cv2.Sobel(roi, cv2.CV_16S, 0, 1, ksize=3)
        mag = cv2.addWeighted(cv2.convertScaleAbs(gx), 1.0, cv2.convertScaleAbs(gy), 0.6, 0)

        edge_pixels = float((mag > 35).mean())
        mean = float(roi.mean())
        std = float(roi.std())

        bright_score = 1.0 - min(1.0, abs(mean - 140.0) / 140.0)
        contrast_score = min(1.0, std / 60.0)
        edge_score = min(1.0, edge_pixels / 0.22)

        return 0.45 * edge_score + 0.30 * contrast_score + 0.25 * bright_score

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # union candidates (dedupe roughly)
        boxes = self._v1_candidates(gray) + self._v2_candidates(gray)

        # cheap dedupe by rounding
        seen = set()
        uniq: list[BBox] = []
        for b in boxes:
            key = (b.x1//4, b.y1//4, b.x2//4, b.y2//4)
            if key not in seen:
                seen.add(key)
                uniq.append(b)

        scored = []
        for b in uniq:
            s = self._score(gray, b)
            scored.append((s, b))

        scored.sort(key=lambda t: t[0], reverse=True)
        keep = scored[: self.top_k]

        # IMPORTANT: use the bbox field your tracker consumes.
        return [Detection(plate_bbox=b, conf=float(s), cls="plate") for s, b in keep]
