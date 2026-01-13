# pipeline/tracking.py
from __future__ import annotations

from typing import List, Dict
import itertools
import logging

from config import AppConfig
from .models import Detection, TrackedPlate, BBox, FramePacket

# pick ONE detector for now (heuristic is easiest to get running)
from .heuristic_detector import HeuristicPlateDetector

log = logging.getLogger(__name__)


def iou(a: BBox, b: BBox) -> float:
    xi1 = max(a.x1, b.x1)
    yi1 = max(a.y1, b.y1)
    xi2 = min(a.x2, b.x2)
    yi2 = min(a.y2, b.y2)
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    inter = (xi2 - xi1) * (yi2 - yi1)
    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    return inter / (area_a + area_b - inter)


class SimplePlateTracker:
    """Minimal IoU-based tracker."""

    def __init__(self, iou_thresh: float = 0.3):
        self.iou_thresh = iou_thresh
        self._next_id = itertools.count(1)
        self._tracks: Dict[int, BBox] = {}

    def update(self, detections: List[Detection]) -> List[TrackedPlate]:
        results: List[TrackedPlate] = []
        used_tracks = set()

        for det in detections:
            best_id = None
            best_iou = 0.0

            for tid, prev_bbox in self._tracks.items():
                score = iou(det.plate_bbox, prev_bbox)
                if score > best_iou:
                    best_iou = score
                    best_id = tid

            if best_id is not None and best_iou >= self.iou_thresh:
                self._tracks[best_id] = det.plate_bbox
                used_tracks.add(best_id)
                results.append(TrackedPlate(best_id, det.plate_bbox))
            else:
                tid = next(self._next_id)
                self._tracks[tid] = det.plate_bbox
                used_tracks.add(tid)
                results.append(TrackedPlate(tid, det.plate_bbox))

        self._tracks = {tid: bb for tid, bb in self._tracks.items() if tid in used_tracks}
        return results


class PlateTracker:
    """
    Wrapper expected by main.py:
    detect -> track ids
    """

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.detector = HeuristicPlateDetector()  # swap to YOLO later
        self.tracker = SimplePlateTracker(iou_thresh=0.3)
        log.info("PlateTracker initialized (heuristic detector + IoU tracker).")

    def process_frame(self, frame: FramePacket) -> List[TrackedPlate]:
        detections = self.detector.detect(frame.image)  # must return List[Detection]
        return self.tracker.update(detections)