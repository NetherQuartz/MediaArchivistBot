import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from faster_whisper import WhisperModel

from .config import Settings, get_settings
from .llm_api import ImageInput
from .models import MediaType

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedMedia:
    images: list[ImageInput]
    transcript: str | None = None
    duration: int | None = None


class MediaProcessor:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._whisper: WhisperModel | None = None

    async def prepare(
        self,
        path: Path,
        media_type: str,
    ) -> PreparedMedia:
        if media_type == MediaType.IMAGE:
            image = await asyncio.to_thread(_prepare_image, path)
            return PreparedMedia(images=[ImageInput(image)])

        frames, duration = await asyncio.to_thread(
            extract_video_frames,
            path,
            self.settings.max_video_frames,
        )
        transcript: str | None = None
        if self.settings.whisper_enabled and media_type == MediaType.VIDEO:
            transcript = await asyncio.to_thread(self._transcribe, path)
        return PreparedMedia(
            images=[ImageInput(frame) for frame in frames],
            transcript=transcript,
            duration=duration,
        )

    def _transcribe(self, path: Path) -> str | None:
        if self._whisper is None:
            self._whisper = WhisperModel(
                self.settings.whisper_model,
                device=self.settings.whisper_device,
                compute_type=self.settings.whisper_compute_type,
            )
        try:
            segments, _ = self._whisper.transcribe(
                str(path),
                beam_size=1,
                vad_filter=True,
            )
            text = " ".join(segment.text.strip() for segment in segments).strip()
            return text[:8_000] or None
        except Exception:
            logger.exception("Could not transcribe video audio")
            return None


def _prepare_image(path: Path, max_dimension: int = 1600) -> bytes:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not decode image: {path.name}")
    image = _resize(image, max_dimension)
    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 90],
    )
    if not success:
        raise ValueError(f"Could not encode image: {path.name}")
    return encoded.tobytes()


def extract_video_frames(
    path: Path,
    max_frames: int = 12,
) -> tuple[list[bytes], int | None]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"OpenCV could not open video: {path.name}")

    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        duration = round(total_frames / fps) if total_frames > 0 and fps > 0 else None
        if total_frames <= 0:
            frames = _extract_unknown_length(capture, max_frames)
        else:
            indices = _candidate_indices(capture, total_frames, max_frames)
            frames = _read_unique_frames(capture, indices, max_frames)
    finally:
        capture.release()

    if not frames:
        raise ValueError(f"No readable frames in video: {path.name}")
    return frames, duration


def _candidate_indices(
    capture: cv2.VideoCapture,
    total_frames: int,
    max_frames: int,
) -> list[int]:
    uniform_count = min(max_frames, 8, total_frames)
    uniform = {
        int(index)
        for index in np.linspace(0, total_frames - 1, uniform_count)
    }
    scan_count = min(total_frames, 160)
    scan_indices = [
        int(index)
        for index in np.linspace(0, total_frames - 1, scan_count)
    ]

    scene_scores: list[tuple[float, int]] = []
    previous: np.ndarray | None = None
    for index in scan_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        success, frame = capture.read()
        if not success:
            continue
        gray = cv2.resize(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            (96, 54),
        )
        if previous is not None:
            score = float(np.mean(cv2.absdiff(previous, gray)))
            scene_scores.append((score, index))
        previous = gray

    extra_count = max(0, max_frames - len(uniform))
    scene_indices = {
        index
        for _, index in sorted(scene_scores, reverse=True)[:extra_count]
    }
    return sorted(uniform | scene_indices)


def _read_unique_frames(
    capture: cv2.VideoCapture,
    indices: list[int],
    max_frames: int,
) -> list[bytes]:
    frames: list[bytes] = []
    hashes: list[np.ndarray] = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        success, frame = capture.read()
        if not success:
            continue
        frame_hash = _average_hash(frame)
        if any(np.count_nonzero(frame_hash != known) < 10 for known in hashes):
            continue
        hashes.append(frame_hash)
        frame = _resize(frame, 1280)
        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 85],
        )
        if success:
            frames.append(encoded.tobytes())
        if len(frames) >= max_frames:
            break
    return frames


def _extract_unknown_length(
    capture: cv2.VideoCapture,
    max_frames: int,
) -> list[bytes]:
    candidate_frames: list[np.ndarray] = []
    frame_number = 0
    while True:
        success, frame = capture.read()
        if not success:
            break
        if frame_number % 10 == 0:
            candidate_frames.append(frame)
        frame_number += 1
    if not candidate_frames:
        return []

    indices = {
        int(index)
        for index in np.linspace(
            0,
            len(candidate_frames) - 1,
            min(max_frames, len(candidate_frames)),
        )
    }
    frames: list[bytes] = []
    for index in sorted(indices):
        frame = _resize(candidate_frames[index], 1280)
        success, encoded = cv2.imencode(".jpg", frame)
        if success:
            frames.append(encoded.tobytes())
    return frames


def _average_hash(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (16, 16))
    return small > np.mean(small)


def _resize(frame: np.ndarray, max_dimension: int) -> np.ndarray:
    height, width = frame.shape[:2]
    largest = max(height, width)
    if largest <= max_dimension:
        return frame
    scale = max_dimension / largest
    return cv2.resize(
        frame,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
