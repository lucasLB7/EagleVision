# pipeline/heuristic_detector.py
import cv2
import numpy as np
from .models import Detection, BBox


class HeuristicPlateDetector:
    """
    Edge + contour based plate guesser.
    Works shockingly well on frontal / rear CCTV.
    """

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        h, w = image_bgr.shape[:2]

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 100, 200)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []

        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / float(bh)

            # plate-ish heuristics
            if 2.0 < aspect < 6.0 and bw > 60 and bh > 15:
                bbox = BBox(x, y, x + bw, y + bh)
                detections.append(Detection(bbox, conf=0.3))

        return detections
