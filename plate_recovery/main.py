from __future__ import annotations

import argparse
from pathlib import Path
import cv2

from config import AppConfig
from pipeline.video_io import iter_frames
from pipeline.tracking import PlateTracker
from pipeline.plate_rectify import rectify_plate
from pipeline.quality import score_plate_patch, select_best_observations
from pipeline.ocr_pass1 import ocr_first_pass
from pipeline.utils import setup_logging
from pipeline.models import PlateObservation


def run(cfg: AppConfig) -> None:
    setup_logging(cfg.logging_level)

    tracker = PlateTracker(cfg)

    observations: list[PlateObservation] = []

    # Debug output folder for plate crops
    debug_dir = cfg.output_dir / "debug_crops"
    debug_dir.mkdir(parents=True, exist_ok=True)

    # 1) Video -> frames
    for frame in iter_frames(cfg.video_path, stride=cfg.frame_stride, max_frames=cfg.max_frames):
        # 2) Detect+track plates/vehicles (returns plate bboxes + track ids)
        tracked = tracker.process_frame(frame)

        # 3) Crop+rectify each plate candidate
        for det in tracked:
            rectified = rectify_plate(
                frame=frame.image,
                bbox=det.plate_bbox,
                target_size=cfg.rectify_target_size,
                pad=cfg.plate_pad_px,
            )

            # 4) Score quality (blur/glare/size/etc.)
            q = score_plate_patch(rectified)

            # Save some crops so you can inspect detections
            # (every 20th processed frame per track)
            if frame.index % 20 == 0:
                fn = debug_dir / f"t{det.track_id}_f{frame.index}_q{q:.3f}.jpg"
                cv2.imwrite(str(fn), rectified)

            observations.append(
                PlateObservation(
                    track_id=det.track_id,
                    frame_index=frame.index,
                    timestamp_s=frame.timestamp_s,
                    plate_bbox=det.plate_bbox,
                    rectified=rectified,
                    quality=q,
                )
            )

    # Group by track and select best frames per track
    selected = select_best_observations(
        observations,
        per_track_top_k=cfg.per_track_top_k,
        min_quality=cfg.min_quality_threshold,
    )

    # 5) First-pass OCR on selected frames
    results = ocr_first_pass(selected, cfg)

    # Print / save
    for track_id, r in results.items():
        print(f"\n=== Track {track_id} ===")
        print(f"Best guess: {r.best_text}  (confidence={r.confidence:.3f})")
        for item in r.items[: min(10, len(r.items))]:
            print(f"  - frame {item.frame_index} q={item.quality:.3f} -> {item.text} (conf={item.confidence:.3f})")


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
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    return cfg


if __name__ == "__main__":
    run(parse_args())