"""Eye-contact and room-coverage tracking from Face Mesh iris landmarks."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

# Face Mesh indices. Iris centers are the refined landmarks 468 and 473.
LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
LEFT_IRIS = 468
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
RIGHT_IRIS = 473


@dataclass(frozen=True)
class GazeSample:
    horizontal: float
    vertical: float
    away: bool


@dataclass(frozen=True)
class GazeState:
    horizontal: float
    vertical: float
    away_seconds: float
    side: str | None
    cue: str | None
    display_text: str | None
    icon: str | None


def _ratio(value: float, start: float, end: float) -> float:
    span = end - start
    if abs(span) < 1e-6:
        return 0.5
    return (value - start) / span


def _horizontal_ratio(iris_x: float, corner_a: float, corner_b: float) -> float:
    left = min(corner_a, corner_b)
    right = max(corner_a, corner_b)
    return _ratio(iris_x, left, right)


def _vertical_ratio(iris_y: float, top_y: float, bottom_y: float) -> float:
    top = min(top_y, bottom_y)
    bottom = max(top_y, bottom_y)
    return _ratio(iris_y, top, bottom)


def measure_gaze(
    landmarks: np.ndarray,
    off_center: float = 0.18,
    look_down: float = 0.65,
) -> GazeSample:
    """Map iris position into 0..1 ratios. 0.5 is looking straight at the camera."""
    points = np.asarray(landmarks, dtype=np.float64)
    left_x = _horizontal_ratio(
        float(points[LEFT_IRIS, 0]),
        float(points[LEFT_EYE_OUTER, 0]),
        float(points[LEFT_EYE_INNER, 0]),
    )
    right_x = _horizontal_ratio(
        float(points[RIGHT_IRIS, 0]),
        float(points[RIGHT_EYE_OUTER, 0]),
        float(points[RIGHT_EYE_INNER, 0]),
    )
    left_y = _vertical_ratio(
        float(points[LEFT_IRIS, 1]),
        float(points[LEFT_EYE_TOP, 1]),
        float(points[LEFT_EYE_BOTTOM, 1]),
    )
    right_y = _vertical_ratio(
        float(points[RIGHT_IRIS, 1]),
        float(points[RIGHT_EYE_TOP, 1]),
        float(points[RIGHT_EYE_BOTTOM, 1]),
    )
    horizontal = (left_x + right_x) / 2.0
    vertical = (left_y + right_y) / 2.0
    away = abs(horizontal - 0.5) > off_center or vertical > look_down
    return GazeSample(horizontal=horizontal, vertical=vertical, away=away)


class GazeTracker:
    """Accumulate fixation time and one-sided room coverage."""

    def __init__(
        self,
        away_seconds: float = 4.0,
        off_center: float = 0.18,
        look_down: float = 0.65,
        side_bias_ratio: float = 0.75,
        history_seconds: float = 6.0,
    ) -> None:
        self.away_seconds_limit = away_seconds
        self.off_center = off_center
        self.look_down = look_down
        self.side_bias_ratio = side_bias_ratio
        self.history_seconds = history_seconds
        self.away_seconds = 0.0
        self._last_t: float | None = None
        self._history: deque[tuple[float, GazeSample]] = deque()

    def update(self, landmarks: np.ndarray | None, timestamp: float) -> GazeState:
        if landmarks is None:
            self.away_seconds = 0.0
            self._last_t = timestamp
            return GazeState(0.5, 0.5, 0.0, None, None, None, None)

        sample = measure_gaze(landmarks, self.off_center, self.look_down)
        dt = 0.0 if self._last_t is None else max(0.0, timestamp - self._last_t)
        self._last_t = timestamp
        if sample.away:
            self.away_seconds += dt
        else:
            self.away_seconds = 0.0

        self._history.append((timestamp, sample))
        cutoff = timestamp - self.history_seconds
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

        side = self._side_bias(timestamp)
        cue, text, icon = self._cue(sample, side)
        return GazeState(
            horizontal=sample.horizontal,
            vertical=sample.vertical,
            away_seconds=self.away_seconds,
            side=side,
            cue=cue,
            display_text=text,
            icon=icon,
        )

    def _side_bias(self, timestamp: float) -> str | None:
        if not self._history:
            return None
        span = timestamp - self._history[0][0]
        if span < self.away_seconds_limit:
            return None
        horizontals = [sample.horizontal for _, sample in self._history]
        left = sum(1 for value in horizontals if value < 0.5 - self.off_center)
        right = sum(1 for value in horizontals if value > 0.5 + self.off_center)
        total = len(horizontals)
        if left / total >= self.side_bias_ratio:
            return "left"
        if right / total >= self.side_bias_ratio:
            return "right"
        return None

    def _cue(
        self, sample: GazeSample, side: str | None
    ) -> tuple[str | None, str | None, str | None]:
        if self.away_seconds >= self.away_seconds_limit:
            if sample.vertical > self.look_down:
                return "eye_contact", "Make eye contact", "target"
            if sample.horizontal < 0.5:
                return "look_right", "Look right", "arrow_right"
            return "look_left", "Look left", "arrow_left"
        if side == "left":
            return "scan_room", "Scan the room", "arrow_right"
        if side == "right":
            return "scan_room", "Scan the room", "arrow_left"
        return None, None, None
