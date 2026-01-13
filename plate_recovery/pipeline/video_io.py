# pipeline/video_io.py
from __future__ import annotations

import cv2
from pathlib import Path
from typing import Iterator

from .models import FramePacket


def iter_frames(video_path: Path, stride: int = 1, max_frames: int | None = None) -> Iterator[FramePacket]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    idx = -1
    yielded = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1

        if idx % stride != 0:
            continue

        t = idx / fps
        yield FramePacket(index=idx, timestamp_s=t, image=frame)

        yielded += 1
        if max_frames is not None and yielded >= max_frames:
            break

    cap.release()
