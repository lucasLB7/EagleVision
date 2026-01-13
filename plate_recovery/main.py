# main.py
from __future__ import annotations

import argparse
from pathlib import Path
import cv2
import numpy as np

from config import AppConfig
from pipeline.video_io import iter_frames
from pipeline.tracking import PlateTracker
from pipeline.plate_rectify import rectify_plate
from pipeline.quality import score_plate_patch, select_best_observations
from pipeline.ocr_pass1 import ocr_first_pass
from pipeline.utils import setup_logging
from pipeline.models import PlateObservation


def sharpness_score(bgr: np.ndarray) -> float:
    """Simple blur metric: higher is sharper."""
    if bgr is None or bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def run(cfg: AppConfig) -> None:
    setup_logging(cfg.logging_level)

    # Ensure output dir exists
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    tracker = PlateTracker(cfg)
    observations: list[PlateObservation] = []

    # Debug output folders
    debug_crops_dir = cfg.output_dir / "debug_crops"
    debug_crops_dir.mkdir(parents=True, exist_ok=True)

    debug_frames_dir = cfg.output_dir / "debug_frames"
    debug_frames_dir.mkdir(parents=True, exist_ok=True)

    # Heuristic sharpness gate (tune after first run)
    SHARP_MIN = 40.0
    SHARP_NORM = 120.0

    # 1) Video -> frames
    for frame in iter_frames(cfg.video_path, stride=cfg.frame_stride, max_frames=cfg.max_frames):

        # 2) Detect + track
        tracked = tracker.process_frame(frame)

        if frame.index % 50 == 0:
            print(f"[frame {frame.index}] tracked={len(tracked)} obs={len(observations)}")

        # ✅ DEBUG: save annotated frames (every 20th frame)
        if frame.index % 20 == 0 and tracked:
            vis = frame.image.copy()
            for det in tracked:
                b = det.plate_bbox
                cv2.rectangle(vis, (b.x1, b.y1), (b.x2, b.y2), (0, 255, 0), 2)
                cv2.putText(
                    vis,
                    f"id{det.track_id}",
                    (b.x1, max(20, b.y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
            cv2.imwrite(str(debug_frames_dir / f"frame_{frame.index}.jpg"), vis)

        # 3) Crop + rectify + quality + collect observation
        for det in tracked:
            # NEW: use pad_frac if you added it to rectify_plate; otherwise this still works
            try:
                rectified = rectify_plate(
                    frame=frame.image,
                    bbox=det.plate_bbox,
                    target_size=cfg.rectify_target_size,
                    pad=cfg.plate_pad_px,
                    pad_frac=0.18,  # NEW: bbox-relative padding (recommended)
                )
            except TypeError:
                # Backwards compatible if rectify_plate doesn't yet accept pad_frac
                rectified = rectify_plate(
                    frame=frame.image,
                    bbox=det.plate_bbox,
                    target_size=cfg.rectify_target_size,
                    pad=cfg.plate_pad_px,
                )

            q0 = score_plate_patch(rectified)
            sh = sharpness_score(rectified)

            # NEW: combine quality with sharpness so sharp frames dominate selection
            sharp_factor = min(1.0, sh / SHARP_NORM)
            q = float(q0) * float(sharp_factor)

            # NEW: hard gate—skip super-blurry patches to avoid "05"/"1" outputs
            if sh < SHARP_MIN:
                # still optionally write debug crops so you can see what you skipped
                if frame.index % 20 == 0:
                    crop_path = debug_crops_dir / f"t{det.track_id}_f{frame.index}_SKIP_q{q:.3f}_sh{sh:.1f}.jpg"
                    cv2.imwrite(str(crop_path), rectified)
                continue

            # ✅ DEBUG: save some crops so you can judge detector quality
            if frame.index % 20 == 0:
                crop_path = debug_crops_dir / f"t{det.track_id}_f{frame.index}_q{q:.3f}_sh{sh:.1f}.jpg"
                cv2.imwrite(str(crop_path), rectified)

            observations.append(
                PlateObservation(
                    track_id=det.track_id,
                    frame_index=frame.index,
                    timestamp_s=frame.timestamp_s,
                    plate_bbox=det.plate_bbox,
                    rectified=rectified,
                    quality=q,  # store combined score
                )
            )

    # ✅ AFTER LOOP
    print(f"Total observations: {len(observations)}")
    if not observations:
        print("No observations collected — detector or filters too strict (or sharpness gate too high).")
        return

    selected = select_best_observations(
        observations,
        per_track_top_k=cfg.per_track_top_k,
        min_quality=cfg.min_quality_threshold,
    )

    print(f"Selected observations: {len(selected)}")
    if not selected:
        print("Observations exist, but none passed quality threshold.")
        print("Tip: lower cfg.min_quality_threshold OR lower SHARP_MIN/SHARP_NORM.")
        return

    # ✅ OCR PASS
    results = ocr_first_pass(selected, cfg)

    if not results:
        print("OCR produced no results.")
        return

    # Print / save
    for track_id, r in results.items():
        print(f"\n=== Track {track_id} ===")
        print(f"Best guess: {r.best_text}  (confidence={r.confidence:.3f})")
        for item in r.items[: min(10, len(r.items))]:
            print(
                f"  - frame {item.frame_index} q={item.quality:.3f} -> {item.text} (conf={item.confidence:.3f})"
            )


def parse_args() -> AppConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, help="Path to input video file")
    p.add_argument("--out", default="out", help="Output directory")
    p.add_argument("--stride", type=int, default=2, help="Process every Nth frame")
    p.add_argument("--max-frames", type=int, default=0, help="0 means no limit")
    args = p.parse_args()

    cfg = AppConfig(
        video_path=Path(args.video),
        output_dir=Path(args.out),
        frame_stride=max(1, args.stride),
        max_frames=(None if args.max_frames == 0 else args.max_frames),
    )
    return cfg


if __name__ == "__main__":
    run(parse_args())
