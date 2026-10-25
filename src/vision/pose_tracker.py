"""Posture metrics from MediaPipe Pose landmarks.

Landmark indices follow the Pose Landmarker topology: nose 0, ears 7 and 8,
shoulders 11 and 12, hips 23 and 24. Coordinates are normalized image x, y
and optional relative depth z (smaller z is closer to the camera).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

NOSE = 0
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24


@dataclass(frozen=True)
class PostureMetrics:
    shoulder_tilt_deg: float
    neck_slump_deg: float
    spine_tilt_deg: float
    ear_forward_offset: float
    straighten: bool

    @property
    def slouch_deg(self) -> float:
        return max(
            abs(self.shoulder_tilt_deg), self.neck_slump_deg, self.spine_tilt_deg
        )


def shoulder_tilt_degrees(landmarks: np.ndarray) -> float:
    """Shoulder line angle in degrees.

    theta = arctan((y_12 - y_11) / (x_12 - x_11)). arctan2 is used so a
    vertical shoulder line does not divide by zero. A level pair is 0 degrees.
    Positive means the image-right shoulder sits lower.
    """
    dy = float(landmarks[RIGHT_SHOULDER, 1] - landmarks[LEFT_SHOULDER, 1])
    dx = float(landmarks[RIGHT_SHOULDER, 0] - landmarks[LEFT_SHOULDER, 0])
    return float(np.degrees(np.arctan2(dy, dx)))


def neck_slump_degrees(landmarks: np.ndarray) -> float:
    """Angle of the ear midpoint from straight above the shoulder midpoint."""
    ear = (landmarks[LEFT_EAR, :2] + landmarks[RIGHT_EAR, :2]) / 2.0
    shoulder = (landmarks[LEFT_SHOULDER, :2] + landmarks[RIGHT_SHOULDER, :2]) / 2.0
    dx = float(ear[0] - shoulder[0])
    vertical = float(-(ear[1] - shoulder[1]))
    return float(np.degrees(np.arctan2(abs(dx), max(vertical, 1e-8))))


def spine_tilt_degrees(landmarks: np.ndarray) -> float:
    """Angle of the shoulder midpoint from straight above the hip midpoint."""
    hip = (landmarks[LEFT_HIP, :2] + landmarks[RIGHT_HIP, :2]) / 2.0
    shoulder = (landmarks[LEFT_SHOULDER, :2] + landmarks[RIGHT_SHOULDER, :2]) / 2.0
    dx = float(shoulder[0] - hip[0])
    vertical = float(-(shoulder[1] - hip[1]))
    return float(np.degrees(np.arctan2(abs(dx), max(vertical, 1e-8))))


def ear_forward_offset(landmarks: np.ndarray) -> float:
    """How far the head sits in front of the shoulders, as a fraction of shoulder width.

    With a z column, offset = (shoulder_z - ear_z) / shoulder_width. MediaPipe's
    z gets smaller as a point approaches the camera, so a forward head is positive.
    Without z, the horizontal ear-to-shoulder offset is used instead.
    """
    shoulder_width = float(
        np.hypot(
            landmarks[RIGHT_SHOULDER, 0] - landmarks[LEFT_SHOULDER, 0],
            landmarks[RIGHT_SHOULDER, 1] - landmarks[LEFT_SHOULDER, 1],
        )
    )
    shoulder_width = max(shoulder_width, 1e-6)
    if landmarks.shape[1] >= 3:
        ear_z = float(landmarks[LEFT_EAR, 2] + landmarks[RIGHT_EAR, 2]) / 2.0
        shoulder_z = (
            float(landmarks[LEFT_SHOULDER, 2] + landmarks[RIGHT_SHOULDER, 2]) / 2.0
        )
        return (shoulder_z - ear_z) / shoulder_width
    ear_x = float(landmarks[LEFT_EAR, 0] + landmarks[RIGHT_EAR, 0]) / 2.0
    shoulder_x = float(landmarks[LEFT_SHOULDER, 0] + landmarks[RIGHT_SHOULDER, 0]) / 2.0
    return abs(ear_x - shoulder_x) / shoulder_width


def analyze_posture(
    landmarks: np.ndarray,
    slouch_degrees: float = 12.0,
    ear_forward_limit: float = 0.15,
) -> PostureMetrics:
    """Return posture metrics and whether the straighten cue should fire.

    The cue fires when slouch exceeds ``slouch_degrees`` or the ear forward
    offset exceeds ``ear_forward_limit``. The boundary itself does not fire.
    """
    points = np.asarray(landmarks, dtype=np.float64)
    tilt = shoulder_tilt_degrees(points)
    neck = neck_slump_degrees(points)
    spine = spine_tilt_degrees(points)
    forward = ear_forward_offset(points)
    slouch = max(abs(tilt), neck, spine)
    straighten = slouch > slouch_degrees or forward > ear_forward_limit
    return PostureMetrics(
        shoulder_tilt_deg=tilt,
        neck_slump_deg=neck,
        spine_tilt_deg=spine,
        ear_forward_offset=forward,
        straighten=straighten,
    )


class PostureCalibrator:
    """Learn a laptop-webcam baseline, then cue on deviation from it.

    The first ``warmup_seconds`` are the speaker's normal sitting posture.
    A fixed 12° slouch line fires constantly when the camera looks up from
    the keyboard. After the warmup, the straighten cue is the extra slouch
    or forward-head past that baseline, and the warmup itself stays quiet.
    """

    def __init__(
        self,
        warmup_seconds: float = 3.0,
        slouch_margin: float = 8.0,
        forward_margin: float = 0.08,
    ) -> None:
        self.warmup_seconds = max(0.0, float(warmup_seconds))
        self.slouch_margin = float(slouch_margin)
        self.forward_margin = float(forward_margin)
        self._slouches: list[float] = []
        self._forwards: list[float] = []
        self._t0: float | None = None
        self.ready = False
        self.baseline_slouch = 0.0
        self.baseline_forward = 0.0

    def observe(self, metrics: PostureMetrics, now: float) -> PostureMetrics:
        if self._t0 is None:
            self._t0 = float(now)
        if not self.ready:
            self._slouches.append(float(metrics.slouch_deg))
            self._forwards.append(float(metrics.ear_forward_offset))
            elapsed = float(now) - self._t0
            if elapsed >= self.warmup_seconds and self._slouches:
                ordered = sorted(self._slouches)
                forwards = sorted(self._forwards)
                mid = len(ordered) // 2
                self.baseline_slouch = ordered[mid]
                self.baseline_forward = forwards[mid]
                self.ready = True
            else:
                return replace(metrics, straighten=False)
        straighten = (
            metrics.slouch_deg > self.baseline_slouch + self.slouch_margin
            or metrics.ear_forward_offset > self.baseline_forward + self.forward_margin
        )
        return replace(metrics, straighten=straighten)
