"""Hand-sign detection from the 21 MediaPipe hand landmarks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WRIST = 0
THUMB_TIP = 4
THUMB_IP = 3
INDEX_TIP = 8
MIDDLE_MCP = 9
# (tip, pip) for the four fingers.
FINGERS = ((8, 6), (12, 10), (16, 14), (20, 18))


@dataclass(frozen=True)
class GestureEvent:
    name: str = "none"
    open_palm: bool = False
    pinch: bool = False
    raised: bool = False
    swipe: str | None = None


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.hypot(float(a[0] - b[0]), float(a[1] - b[1])))


def finger_extended(
    landmarks: np.ndarray, tip: int, pip: int, ratio: float = 1.12
) -> bool:
    wrist = landmarks[WRIST, :2]
    return _distance(landmarks[tip], wrist) > _distance(landmarks[pip], wrist) * ratio


def is_open_palm(landmarks: np.ndarray) -> bool:
    points = np.asarray(landmarks, dtype=np.float64)
    if not all(finger_extended(points, tip, pip) for tip, pip in FINGERS):
        return False
    return finger_extended(points, THUMB_TIP, THUMB_IP, ratio=1.05)


def pinch_ratio(landmarks: np.ndarray) -> float:
    """Thumb-to-index gap divided by palm size. Smaller means a tighter pinch."""
    points = np.asarray(landmarks, dtype=np.float64)
    palm = max(_distance(points[MIDDLE_MCP], points[WRIST]), 1e-6)
    gap = _distance(points[THUMB_TIP], points[INDEX_TIP])
    return gap / palm


def is_pinch(landmarks: np.ndarray, ratio: float = 0.45) -> bool:
    return pinch_ratio(landmarks) < ratio


class GestureDetector:
    """Detect a raised open palm, a pinch, and horizontal swipes.

    Pinch uses hysteresis: it engages below ``pinch_on`` and stays engaged
    until the gap opens past ``pinch_off``. A one-frame flicker does not
    count as a pinch.
    """

    def __init__(
        self,
        swipe_distance: float = 0.12,
        swipe_window: float = 0.35,
        raised_y: float = 0.55,
        pinch_on: float = 0.35,
        pinch_off: float = 0.55,
    ) -> None:
        self.swipe_distance = swipe_distance
        self.swipe_window = swipe_window
        self.raised_y = raised_y
        self.pinch_on = pinch_on
        self.pinch_off = pinch_off
        self._history: list[tuple[float, float]] = []
        self._pinch_active = False

    def update(self, landmarks: np.ndarray | None, timestamp: float) -> GestureEvent:
        if landmarks is None:
            self._pinch_active = False
            return GestureEvent()
        points = np.asarray(landmarks, dtype=np.float64)
        palm = is_open_palm(points)
        ratio = pinch_ratio(points)
        if self._pinch_active:
            self._pinch_active = ratio < self.pinch_off
        else:
            self._pinch_active = ratio < self.pinch_on
        pinch = self._pinch_active
        raised = bool(palm and float(points[WRIST, 1]) < self.raised_y)
        wrist_x = float(points[WRIST, 0])
        self._history.append((timestamp, wrist_x))
        self._history = [
            (t, x) for t, x in self._history if timestamp - t <= self.swipe_window
        ]
        swipe = None
        if len(self._history) >= 2:
            dx = self._history[-1][1] - self._history[0][1]
            if dx >= self.swipe_distance:
                swipe = "right"
            elif dx <= -self.swipe_distance:
                swipe = "left"
        if swipe is not None:
            self._history = [(timestamp, wrist_x)]
        if swipe == "right":
            name = "swipe_right"
        elif swipe == "left":
            name = "swipe_left"
        elif raised or palm:
            name = "open_palm"
        elif pinch:
            name = "pinch"
        else:
            name = "none"
        return GestureEvent(
            name=name,
            open_palm=palm,
            pinch=pinch,
            raised=raised,
            swipe=swipe,
        )
