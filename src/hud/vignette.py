"""Time-remaining border glow. Neutral above 25%, amber through 10%, red below."""

from __future__ import annotations

import numpy as np

from src.config_loader import Theme
from src.hud.widgets import hex_to_bgr

_MASKS: dict[tuple[int, int], np.ndarray] = {}


def edge_mask(height: int, width: int) -> np.ndarray:
    """0 at the center, rising toward the edges. Cached per resolution."""
    key = (height, width)
    cached = _MASKS.get(key)
    if cached is not None:
        return cached
    ys = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    xs = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    radius = np.sqrt(xs * xs + ys * ys)
    mask = np.clip((radius - 0.45) / 0.85, 0.0, 1.0) ** 1.6
    _MASKS[key] = mask
    return mask


def vignette_strength(
    remaining_ratio: float,
    warning_ratio: float = 0.25,
    critical_ratio: float = 0.10,
    pulse_t: float = 0.0,
) -> tuple[float, str]:
    """Return (edge alpha, phase). Alpha grows as the remaining fraction shrinks."""
    if remaining_ratio > warning_ratio:
        return 0.0, "none"
    if remaining_ratio > critical_ratio:
        span = max(warning_ratio - critical_ratio, 1e-6)
        base = (warning_ratio - remaining_ratio) / span
        pulse = 0.65 + 0.35 * float(np.sin(pulse_t * 4.0))
        return float(np.clip(base * 0.55 * pulse, 0.0, 1.0)), "warning"
    depth = 1.0 - (remaining_ratio / critical_ratio) if critical_ratio else 1.0
    pulse = 0.75 + 0.25 * float(np.sin(pulse_t * 6.0))
    return float(np.clip((0.55 + 0.40 * depth) * pulse, 0.0, 1.0)), "critical"


def vignette_color(theme: Theme, phase: str) -> tuple[int, int, int]:
    if phase == "critical":
        return hex_to_bgr(theme.colors["alert"])
    if phase == "warning":
        return hex_to_bgr(theme.colors["warning"])
    return hex_to_bgr(theme.colors.get("neutral", "#000000"))


def apply_vignette(
    frame: np.ndarray,
    remaining_ratio: float,
    theme: Theme,
    pulse_t: float = 0.0,
    warning_ratio: float = 0.25,
    critical_ratio: float = 0.10,
) -> np.ndarray:
    strength, phase = vignette_strength(
        remaining_ratio, warning_ratio, critical_ratio, pulse_t
    )
    if strength <= 0.0 or phase == "none":
        return frame.copy()
    mask = edge_mask(frame.shape[0], frame.shape[1]) * strength
    color = np.asarray(vignette_color(theme, phase), dtype=np.float32)
    base = frame.astype(np.float32)
    blended = base * (1.0 - mask[:, :, None]) + color * mask[:, :, None]
    return np.clip(blended, 0, 255).astype(np.uint8)
