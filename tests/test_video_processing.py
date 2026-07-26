from pathlib import Path

import cv2
import numpy as np

from archivistbot.media_processing import extract_video_frames


def test_extracts_distinct_video_frames(tmp_path: Path) -> None:
    path = tmp_path / "scenes.avi"
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5,
        (64, 64),
    )
    assert writer.isOpened()
    try:
        for index in range(20):
            color = (0, 0, 0) if index < 10 else (255, 255, 255)
            frame = np.full((64, 64, 3), color, dtype=np.uint8)
            cv2.putText(
                frame,
                str(index),
                (10, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                1,
            )
            writer.write(frame)
    finally:
        writer.release()

    frames, duration = extract_video_frames(path, max_frames=6)

    assert 2 <= len(frames) <= 6
    assert duration == 4
