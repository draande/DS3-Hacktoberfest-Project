"""Syllable-rate words-per-minute estimate and run-on detection.

Speech is not transcribed. Voiced energy peaks stand in for syllables, and
words are syllables divided by ``syllables_per_word`` (default 1.5). The
estimate uses the sliding 2.5 second buffer from the audio spec. Rushing is
withheld until that window has at least 1.5 seconds of audio so a single
syllable cannot look like 400 WPM.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

from src.audio.volume_analyzer import rms
from src.capture.audio_stream import RingBuffer


@dataclass(frozen=True)
class PaceReading:
    wpm: float
    syllables: int
    voiced_seconds: float
    pause_seconds: float
    continuous_speech_seconds: float
    rushing: bool
    run_on: bool
    in_target: bool


def frame_energies(
    samples: np.ndarray,
    sample_rate: int,
    frame_seconds: float = 0.02,
    hop_seconds: float = 0.01,
) -> np.ndarray:
    data = np.asarray(samples, dtype=np.float64).reshape(-1)
    frame = max(1, int(frame_seconds * sample_rate))
    hop = max(1, int(hop_seconds * sample_rate))
    if data.size < frame:
        return np.asarray([rms(data)], dtype=np.float64) if data.size else np.zeros(0)
    values = [
        rms(data[start : start + frame])
        for start in range(0, data.size - frame + 1, hop)
    ]
    return np.asarray(values, dtype=np.float64)


def count_syllables(
    samples: np.ndarray,
    sample_rate: int,
    vad_rms: float = 0.02,
    min_gap_seconds: float = 0.12,
) -> int:
    energies = frame_energies(samples, sample_rate)
    if energies.size < 3:
        return 0
    smooth = np.convolve(energies, np.ones(3) / 3.0, mode="same")
    kernel = min(smooth.size if smooth.size % 2 else smooth.size - 1, 21)
    kernel = max(3, kernel)
    # Remove the slow level so a steady tone has no peaks to count.
    contrast = smooth - median_filter(smooth, size=kernel)
    peak_level = float(np.percentile(contrast, 90))
    height = max(vad_rms * 0.5, peak_level * 0.5)
    hop_seconds = 0.01
    distance = max(1, int(min_gap_seconds / hop_seconds))
    peaks, _ = find_peaks(
        contrast, height=height, distance=distance, prominence=height * 0.5
    )
    return int(peaks.size)


def voiced_and_silence(
    samples: np.ndarray,
    sample_rate: int,
    vad_rms: float,
    hop_seconds: float = 0.01,
) -> tuple[float, float]:
    """Return (voiced seconds, longest internal silence seconds)."""
    energies = frame_energies(samples, sample_rate, hop_seconds=hop_seconds)
    if energies.size == 0:
        return 0.0, 0.0
    voiced_frames = int(np.count_nonzero(energies >= vad_rms))
    longest = current = 0
    for energy in energies:
        if energy < vad_rms:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return voiced_frames * hop_seconds, longest * hop_seconds


class PaceAnalyzer:
    def __init__(
        self,
        sample_rate: int = 16000,
        rush_wpm: float = 180.0,
        target_wpm_min: float = 130.0,
        target_wpm_max: float = 160.0,
        syllables_per_word: float = 1.5,
        pause_seconds: float = 0.35,
        continuous_speech_seconds: float = 8.0,
        vad_rms: float = 0.02,
        window_seconds: float = 2.5,
        min_window_seconds: float = 1.5,
    ) -> None:
        self.sample_rate = sample_rate
        self.rush_wpm = rush_wpm
        self.target_wpm_min = target_wpm_min
        self.target_wpm_max = target_wpm_max
        self.syllables_per_word = syllables_per_word
        self.pause_seconds = pause_seconds
        self.continuous_speech_seconds = continuous_speech_seconds
        self.vad_rms = vad_rms
        self.window_seconds = window_seconds
        self.min_window_seconds = min_window_seconds
        self.buffer = RingBuffer(max(1, int(sample_rate * window_seconds)))
        self.continuous_speech_seconds_value = 0.0
        self._trailing_silence = 0.0

    def update(self, samples: np.ndarray, timestamp: float) -> PaceReading:
        del timestamp
        data = np.asarray(samples, dtype=np.float32).reshape(-1)
        self.buffer.write(data)
        window = self.buffer.read_latest(self.buffer.capacity)
        duration = float(window.size) / self.sample_rate if self.sample_rate else 0.0
        syllables = count_syllables(window, self.sample_rate, self.vad_rms)
        minutes = duration / 60.0 if duration else 0.0
        words = syllables / self.syllables_per_word if self.syllables_per_word else 0.0
        wpm = words / minutes if minutes else 0.0
        voiced, _silence = voiced_and_silence(window, self.sample_rate, self.vad_rms)
        self._advance_speech_clock(data)

        rushing = (
            duration >= self.min_window_seconds and wpm > self.rush_wpm and voiced > 0.3
        )
        run_on = self.continuous_speech_seconds_value > self.continuous_speech_seconds
        in_target = self.target_wpm_min <= wpm <= self.target_wpm_max
        return PaceReading(
            wpm=wpm,
            syllables=syllables,
            voiced_seconds=voiced,
            pause_seconds=max(0.0, duration - voiced),
            continuous_speech_seconds=self.continuous_speech_seconds_value,
            rushing=rushing,
            run_on=run_on,
            in_target=in_target,
        )

    def _advance_speech_clock(self, data: np.ndarray) -> None:
        """Treat silence shorter than a real pause as still part of the same sentence."""
        if data.size == 0 or self.sample_rate <= 0:
            return
        energies = frame_energies(data, self.sample_rate)
        hop = 0.01
        for energy in energies:
            if energy < self.vad_rms:
                self._trailing_silence += hop
                if self._trailing_silence >= self.pause_seconds:
                    self.continuous_speech_seconds_value = 0.0
            else:
                self._trailing_silence = 0.0
                self.continuous_speech_seconds_value += hop
