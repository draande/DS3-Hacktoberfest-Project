"""RMS loudness on a short sliding window.

Digital samples have no sound-pressure level. ``rms_to_db`` uses

    dB = 20 log10(RMS / reference)

with ``reference`` defaulting to 1e-4. A full-scale sine then sits near 77 dB
and a quiet talker (amplitude around 0.004) falls under 40 dB, which is the
scale the HUD thresholds in PLAN.md assume.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.capture.audio_stream import RingBuffer


def rms(samples: np.ndarray) -> float:
    data = np.asarray(samples, dtype=np.float64).reshape(-1)
    if data.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(data * data)))


def rms_to_db(rms_value: float, reference: float = 1.0e-4) -> float:
    if rms_value <= 1e-12 or reference <= 0:
        return 0.0
    return float(20.0 * np.log10(rms_value / reference))


@dataclass(frozen=True)
class VolumeReading:
    rms: float
    db: float
    quiet: bool
    speak_up: bool
    quiet_seconds: float


class VolumeAnalyzer:
    """Track level on a short window and raise speak-up after sustained quiet."""

    def __init__(
        self,
        threshold_db: float = 42.0,
        quiet_duration: float = 3.0,
        reference: float = 1.0e-4,
        sample_rate: int = 16000,
        level_window_seconds: float = 0.5,
        buffer_seconds: float = 2.5,
    ) -> None:
        self.threshold_db = threshold_db
        self.quiet_duration = quiet_duration
        self.reference = reference
        self.sample_rate = sample_rate
        self.level_window_seconds = level_window_seconds
        self.buffer = RingBuffer(max(1, int(sample_rate * buffer_seconds)))
        self._quiet_since: float | None = None

    def update(self, samples: np.ndarray, timestamp: float) -> VolumeReading:
        self.buffer.write(samples)
        window = max(1, int(self.sample_rate * self.level_window_seconds))
        recent = self.buffer.read_latest(window)
        level = rms(recent)
        decibels = rms_to_db(level, self.reference)
        quiet = decibels < self.threshold_db
        if quiet:
            if self._quiet_since is None:
                self._quiet_since = timestamp
            quiet_seconds = timestamp - self._quiet_since
        else:
            self._quiet_since = None
            quiet_seconds = 0.0
        return VolumeReading(
            rms=level,
            db=decibels,
            quiet=quiet,
            speak_up=quiet and quiet_seconds >= self.quiet_duration,
            quiet_seconds=quiet_seconds,
        )
