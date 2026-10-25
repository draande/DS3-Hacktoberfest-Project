"""Audience confusion and surprise from Face Mesh geometry or blendshapes."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_OUTER = 263
RIGHT_EYE_INNER = 362
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
LEFT_INNER_BROW = 107
RIGHT_INNER_BROW = 336
LEFT_BROW = 105
RIGHT_BROW = 334


@dataclass(frozen=True)
class AudienceTells:
    brow_furrow: float
    eye_widen: float
    brow_raise: float
    confused: bool
    surprised: bool
    blendshapes: dict[str, float] = field(default_factory=dict)


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.hypot(float(a[0] - b[0]), float(a[1] - b[1])))


def _eye_aspect(
    points: np.ndarray, outer: int, inner: int, top: int, bottom: int
) -> float:
    width = max(_distance(points[outer], points[inner]), 1e-6)
    height = _distance(points[top], points[bottom])
    return height / width


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def analyze_audience(
    landmarks: np.ndarray,
    brow_gap_ratio: float = 0.22,
    eye_aspect_wide: float = 0.42,
    brow_raise_ratio: float = 0.18,
) -> AudienceTells:
    """Score brow furrow, eye opening, and brow raise from landmark distances."""
    points = np.asarray(landmarks, dtype=np.float64)
    interocular = max(_distance(points[LEFT_EYE_OUTER], points[RIGHT_EYE_OUTER]), 1e-6)
    gap_ratio = (
        _distance(points[LEFT_INNER_BROW], points[RIGHT_INNER_BROW]) / interocular
    )
    if gap_ratio < brow_gap_ratio:
        furrow = _clamp01((brow_gap_ratio - gap_ratio) / brow_gap_ratio)
    else:
        furrow = 0.0

    aspect = (
        _eye_aspect(
            points, LEFT_EYE_OUTER, LEFT_EYE_INNER, LEFT_EYE_TOP, LEFT_EYE_BOTTOM
        )
        + _eye_aspect(
            points, RIGHT_EYE_OUTER, RIGHT_EYE_INNER, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM
        )
    ) / 2.0
    if aspect > eye_aspect_wide:
        widen = _clamp01((aspect - eye_aspect_wide) / eye_aspect_wide)
    else:
        widen = 0.0

    raise_ratio = (
        (
            _distance(points[LEFT_BROW], points[LEFT_EYE_TOP])
            + _distance(points[RIGHT_BROW], points[RIGHT_EYE_TOP])
        )
        / 2.0
        / interocular
    )
    if raise_ratio > brow_raise_ratio:
        raised = _clamp01(
            (raise_ratio - brow_raise_ratio) / max(brow_raise_ratio, 1e-6)
        )
    else:
        raised = 0.0

    blendshapes = {
        "browDown": furrow,
        "eyeWide": widen,
        "browRaise": raised,
        "browGapRatio": gap_ratio,
        "eyeAspect": aspect,
    }
    return AudienceTells(
        brow_furrow=furrow,
        eye_widen=widen,
        brow_raise=raised,
        confused=furrow > 0.0,
        surprised=widen > 0.0 and raised > 0.0,
        blendshapes=blendshapes,
    )


def _normalize_name(name: str) -> str:
    return name.replace("_", "").replace(" ", "").lower()


def tells_from_blendshapes(
    blendshapes: dict[str, float],
    furrow_threshold: float = 0.4,
    wide_threshold: float = 0.4,
    raise_threshold: float = 0.4,
) -> AudienceTells:
    """Map MediaPipe face blendshape scores onto the same tell flags."""
    scores = {_normalize_name(key): float(value) for key, value in blendshapes.items()}

    def score(*names: str) -> float:
        values = [scores.get(_normalize_name(name), 0.0) for name in names]
        return sum(values) / len(values)

    furrow = score("browDownLeft", "browDownRight")
    widen = score("eyeWideLeft", "eyeWideRight")
    raised = score("browInnerUp", "browOuterUpLeft", "browOuterUpRight")
    return AudienceTells(
        brow_furrow=furrow,
        eye_widen=widen,
        brow_raise=raised,
        confused=furrow >= furrow_threshold,
        surprised=widen >= wide_threshold and raised >= raise_threshold,
        blendshapes=dict(blendshapes),
    )
