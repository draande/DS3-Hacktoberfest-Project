"""Camera, file, and synthetic video sources with frame-drop protection."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - opencv is a required dependency
    cv2 = None


@dataclass
class VideoFrame:
    image: np.ndarray
    timestamp: float
    index: int


class FrameQueue:
    """Bounded queue that drops stale frames and frames arriving faster than target_fps."""

    def __init__(self, max_frames: int = 2, target_fps: float = 30.0) -> None:
        if max_frames < 1:
            raise ValueError("max_frames must be at least 1")
        self.max_frames = max_frames
        self.target_fps = target_fps
        self.dropped_overflow = 0
        self.dropped_pacing = 0
        self._frames: deque[VideoFrame] = deque()
        self._last_kept: float | None = None
        self._lock = threading.Lock()

    def push(self, frame: VideoFrame) -> bool:
        with self._lock:
            if self.target_fps > 0 and self._last_kept is not None:
                min_dt = 1.0 / self.target_fps
                if frame.timestamp - self._last_kept < min_dt - 1e-9:
                    self.dropped_pacing += 1
                    return False
            if len(self._frames) >= self.max_frames:
                self._frames.popleft()
                self.dropped_overflow += 1
            self._frames.append(frame)
            self._last_kept = frame.timestamp
            return True

    def pop(self) -> VideoFrame | None:
        with self._lock:
            if not self._frames:
                return None
            return self._frames.popleft()

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)

    @property
    def dropped(self) -> int:
        return self.dropped_overflow + self.dropped_pacing


def stream_drift_ms(video_timestamp: float, audio_timestamp: float) -> float:
    """Signed video-minus-audio drift in milliseconds."""
    return (video_timestamp - audio_timestamp) * 1000.0


class SyntheticVideoSource:
    """Generated frames for headless demo and tests. No camera required."""

    def __init__(
        self,
        frames: list[np.ndarray] | None = None,
        width: int = 320,
        height: int = 240,
        count: int = 30,
        fps: float = 30.0,
    ) -> None:
        self.fps = fps
        self._index = 0
        if frames is None:
            self._frames = [_synthetic_frame(i, width, height) for i in range(count)]
        else:
            self._frames = frames

    def read(self) -> VideoFrame | None:
        if self._index >= len(self._frames):
            return None
        image = self._frames[self._index]
        frame = VideoFrame(
            image=image, timestamp=self._index / self.fps, index=self._index
        )
        self._index += 1
        return frame

    def close(self) -> None:
        return None

    def __enter__(self) -> SyntheticVideoSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _synthetic_frame(index: int, width: int, height: int) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = (24, 18, 32)
    x = (index * 7) % max(width, 1)
    x2 = min(width, x + 3)
    image[:, x:x2] = (70, 60, 80)
    return image


class FileVideoSource:
    """Play a recorded video file. Used for headless playback tests."""

    def __init__(self, path: str | Path, fps: float | None = None) -> None:
        if cv2 is None:
            raise RuntimeError("opencv is required to read video files")
        self.path = str(path)
        self._fps_override = fps
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Could not open video file: {self.path}")
        reported = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.fps = fps if fps else (reported if reported > 1 else 30.0)
        self._index = 0

    def read(self) -> VideoFrame | None:
        ok, image = self._cap.read()
        if not ok or image is None:
            return None
        frame = VideoFrame(
            image=image, timestamp=self._index / self.fps, index=self._index
        )
        self._index += 1
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> FileVideoSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class CameraUnavailable(RuntimeError):
    """Raised when a webcam cannot be opened."""


class WebcamSource:
    """Live webcam capture. Not used by the demo or the test suite."""

    def __init__(
        self,
        index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: float = 30.0,
    ) -> None:
        if cv2 is None:
            raise CameraUnavailable("opencv is required for webcam capture")
        self.index = index
        self.fps = fps
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise CameraUnavailable(
                f"Could not open webcam index {index}. Run with --demo to skip the camera."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        self._index = 0
        self._t0 = time.monotonic()

    def read(self) -> VideoFrame | None:
        ok, image = self._cap.read()
        if not ok or image is None:
            return None
        frame = VideoFrame(
            image=image,
            timestamp=time.monotonic() - self._t0,
            index=self._index,
        )
        self._index += 1
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> WebcamSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
