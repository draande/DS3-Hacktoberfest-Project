"""Synthesized sine chimes. Playback is optional and never required for tests."""

from __future__ import annotations

import numpy as np

CHIME_FREQUENCIES = {
    "POSTURE": 523.25,
    "VOLUME": 659.25,
    "PACE": 440.0,
    "GAZE": 783.99,
    "AUDIENCE": 587.33,
    "ENCOURAGE": 880.0,
    "DISMISS": 349.23,
}


def synthesize_chime(
    frequency: float = 880.0,
    duration: float = 0.18,
    sample_rate: int = 16000,
    volume: float = 0.2,
) -> np.ndarray:
    """Return a decaying sine in float32, peak amplitude at most ``volume``."""
    count = max(1, int(duration * sample_rate))
    t = np.arange(count, dtype=np.float32) / sample_rate
    envelope = np.exp(-t * 8.0)
    wave = volume * envelope * np.sin(2.0 * np.pi * frequency * t)
    return wave.astype(np.float32)


def chime_for_cue(cue_type: str, sample_rate: int = 16000) -> np.ndarray:
    frequency = CHIME_FREQUENCIES.get(cue_type.upper(), 880.0)
    return synthesize_chime(frequency=frequency, sample_rate=sample_rate)


def play_chime(samples: np.ndarray, sample_rate: int = 16000) -> bool:
    """Play a chime if PortAudio is available. Returns False instead of raising."""
    try:
        import sounddevice as sd
    except (OSError, ImportError):
        return False
    sd.play(np.asarray(samples, dtype=np.float32), sample_rate)
    return True
