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
    denom = (area_a + area_b - inter)
    return float(inter / denom) if denom > 0 else 0.0

def center_suppress(dets: List[Detection], dist_frac: float = 0.35) -> List[Detection]:
    """
    Suppress detections whose centers are very close (relative to bbox size),
    even if IoU is low (common with tight vs wide boxes).
    """
    dets = sorted(dets, key=lambda d: d.conf, reverse=True)
    keep: List[Detection] = []

    for d in dets:
        bx = d.plate_bbox
        cx = 0.5 * (bx.x1 + bx.x2)
        cy = 0.5 * (bx.y1 + bx.y2)
        bw = max(1, bx.x2 - bx.x1)
        bh = max(1, bx.y2 - bx.y1)
        thresh = dist_frac * max(bw, bh)

        ok = True
        for k in keep:
            kb = k.plate_bbox
            kcx = 0.5 * (kb.x1 + kb.x2)
            kcy = 0.5 * (kb.y1 + kb.y2)
            if ((cx - kcx) ** 2 + (cy - kcy) ** 2) ** 0.5 < thresh:
                ok = False
                break

        if ok:
            keep.append(d)

    return keep

def nms(dets: List[Detection], iou_thresh: float = 0.5) -> List[Detection]:
    """Keep highest-conf detections, suppress overlaps."""
    dets = sorted(dets, key=lambda d: d.conf, reverse=True)
    keep: List[Detection] = []
    for d in dets:
        if all(iou(d.plate_bbox, k.plate_bbox) < iou_thresh for k in keep):
            keep.append(d)
    return keep


def plate_like_bbox(
    b: BBox,
    img_w: int,
    img_h: int,
    *,
    min_w: int = 60,
    min_h: int = 18,
    max_w_frac: float = 0.70,
    max_h_frac: float = 0.30,
    min_ar: float = 2.0,
    max_ar: float = 6.5,
    min_area: int = 1200,
) -> bool:
    """
    Simple geometry filter to kill random grass/bumper patches.
    Tune these for your camera.
    """
    w = b.x2 - b.x1
    h = b.y2 - b.y1
    if w <= 0 or h <= 0:
        return False

    # tiny -> reject
    if w < min_w or h < min_h:
        return False

    # huge -> reject
    if w > int(max_w_frac * img_w):
        return False
    if h > int(max_h_frac * img_h):
        return False

    # plates are wide rectangles
    ar = w / float(h)
    if ar < min_ar or ar > max_ar:
        return False

    # area sanity
    if (w * h) < min_area:
        return False

    return True


class SimplePlateTracker:
    """Minimal IoU-based tracker with one-to-one assignment + TTL (max_age)."""

    def __init__(self, iou_thresh: float = 0.3, max_age: int = 8):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self._next_id = itertools.count(1)
        self._tracks: Dict[int, BBox] = {}
        self._age: Dict[int, int] = {}

    def update(self, detections: List[Detection]) -> List[TrackedPlate]:
        # Age all tracks by 1 each frame
        for tid in list(self._age.keys()):
            self._age[tid] += 1

        # Greedy: match highest-confidence detections first
        detections = sorted(detections, key=lambda d: d.conf, reverse=True)

        results: List[TrackedPlate] = []
        assigned_tracks = set()

        # Start from existing tracks, then update in-place
        new_tracks: Dict[int, BBox] = dict(self._tracks)

        for det in detections:
            best_id = None
            best_iou = 0.0

            for tid, prev_bbox in self._tracks.items():
                if tid in assigned_tracks:
                    continue  # one detection per track per frame

                score = iou(det.plate_bbox, prev_bbox)
                if score > best_iou:
                    best_iou = score
                    best_id = tid

            if best_id is not None and best_iou >= self.iou_thresh:
                assigned_tracks.add(best_id)
                new_tracks[best_id] = det.plate_bbox
                self._age[best_id] = 0
                results.append(TrackedPlate(best_id, det.plate_bbox))
            else:
                tid = next(self._next_id)
                assigned_tracks.add(tid)
                new_tracks[tid] = det.plate_bbox
                self._age[tid] = 0
                results.append(TrackedPlate(tid, det.plate_bbox))

        # Keep tracks that haven't expired
        alive_tracks: Dict[int, BBox] = {}
        alive_age: Dict[int, int] = {}
        for tid, bb in new_tracks.items():
            age = self._age.get(tid, 9999)
            if age <= self.max_age:
                alive_tracks[tid] = bb
                alive_age[tid] = age

        self._tracks = alive_tracks
        self._age = alive_age
        return results


class PlateTracker:
    """
    Wrapper expected by main.py:
    detect -> (filter) -> NMS -> track ids
    """

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.detector = HeuristicPlateDetector()  # swap to YOLO later
        self.tracker = SimplePlateTracker(iou_thresh=0.3, max_age=8)

        # tuning knobs (start conservative)
        self.min_w = 60
        self.min_h = 18
        self.max_w_frac = 0.70
        self.max_h_frac = 0.30
        self.min_ar = 2.0
        self.max_ar = 6.5
        self.min_area = 1200

        # per-frame NMS overlap threshold - Change to adjust frame size 
        self.nms_iou = 0.25

        log.info("PlateTracker initialized (heuristic detector + IoU tracker).")

    def _filter_detections(self, dets: List[Detection], img_w: int, img_h: int) -> List[Detection]:
        kept: List[Detection] = []
        for d in dets:
            # clamp bbox to image bounds first
            bb = d.plate_bbox.clamp(img_w, img_h)

            if plate_like_bbox(
                bb,
                img_w,
                img_h,
                min_w=self.min_w,
                min_h=self.min_h,
                max_w_frac=self.max_w_frac,
                max_h_frac=self.max_h_frac,
                min_ar=self.min_ar,
                max_ar=self.max_ar,
                min_area=self.min_area,
            ):
                kept.append(Detection(plate_bbox=bb, conf=d.conf, cls=d.cls))

        return kept

    def process_frame(self, frame: FramePacket) -> List[TrackedPlate]:
        img = frame.image
        h, w = img.shape[:2]

        raw = self.detector.detect(img)  # List[Detection]
        filt = self._filter_detections(raw, w, h)

        # tighter NMS to merge "tight vs wide" plate boxes
        filt = nms(filt, iou_thresh=0.30)

        # suppress near-duplicate centers even if IoU is low
        filt = center_suppress(filt, dist_frac=0.35)

        # OPTIONAL but recommended: keep only top-N per frame to reduce false tracks
        filt = sorted(filt, key=lambda d: d.conf, reverse=True)[:4]

        if frame.index % 50 == 0:
            log.info(f"frame {frame.index}: raw dets={len(raw)} -> filtered={len(filt)}")

        return self.tracker.update(filt)

