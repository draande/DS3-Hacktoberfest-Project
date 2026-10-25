import numpy as np
import pytest

from src.audio.audio_feedback import chime_for_cue, synthesize_chime
from src.audio.pace_analyzer import PaceAnalyzer
from src.audio.volume_analyzer import VolumeAnalyzer, rms, rms_to_db
from src.demo_scene import audio_chunk


def _sine(
    amplitude: float, seconds: float, sample_rate: int = 16000, frequency: float = 220.0
):
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def test_rms_and_db_follow_the_calibrated_formula():
    samples = np.array([3.0, 4.0], dtype=np.float64)
    assert rms(samples) == pytest.approx((12.5) ** 0.5)
    assert rms_to_db(0.1, reference=0.1) == pytest.approx(0.0)
    assert rms_to_db(1.0, reference=1e-4) == pytest.approx(80.0)
    assert rms_to_db(0.0) == 0.0


def test_quiet_buffer_is_detected_within_500ms():
    analyzer = VolumeAnalyzer(threshold_db=40.0, reference=1e-4)
    quiet = _sine(0.004, 0.5)
    reading = analyzer.update(quiet, timestamp=0.5)
    assert reading.db < 40.0
    assert reading.quiet is True
    # The level is visible immediately. The speak-up cue waits for the 3 s hold.
    assert reading.speak_up is False


def test_sustained_quiet_triggers_speak_up_and_loud_speech_does_not():
    analyzer = VolumeAnalyzer(threshold_db=42.0, quiet_duration=3.0, reference=1e-4)
    quiet = _sine(0.003, 0.5)
    first = analyzer.update(quiet, timestamp=1.0)
    assert first.quiet is True
    assert first.speak_up is False
    later = analyzer.update(quiet, timestamp=4.0)
    assert later.db < 42.0
    assert later.speak_up is True

    loud = VolumeAnalyzer(threshold_db=42.0, reference=1e-4)
    speaking = _sine(0.2, 0.5)
    loud_reading = loud.update(speaking, timestamp=0.5)
    assert loud_reading.db > 42.0
    assert loud_reading.quiet is False


def test_demo_signals_sit_on_opposite_sides_of_the_volume_threshold():
    sample_rate = 16000
    count = sample_rate // 2
    analyzer = VolumeAnalyzer(
        threshold_db=42.0, reference=1e-4, sample_rate=sample_rate
    )
    speech = analyzer.update(audio_chunk("speech", 0, count, sample_rate), 0.5)
    quiet_analyzer = VolumeAnalyzer(
        threshold_db=40.0, reference=1e-4, sample_rate=sample_rate
    )
    quiet = quiet_analyzer.update(audio_chunk("quiet", 0, count, sample_rate), 0.5)
    assert speech.db > 42.0
    assert quiet.db < 40.0


def test_comfortable_and_rushing_cadence():
    sample_rate = 16000
    count = int(2.5 * sample_rate)
    comfortable = PaceAnalyzer(sample_rate=sample_rate)
    comfort = comfortable.update(audio_chunk("speech", 0, count, sample_rate), 2.5)
    assert 100 <= comfort.wpm <= 170
    assert comfort.rushing is False

    rushing = PaceAnalyzer(sample_rate=sample_rate)
    rush = rushing.update(audio_chunk("rush", 0, count, sample_rate), 2.5)
    assert rush.wpm > 180
    assert rush.rushing is True


def test_eight_seconds_without_a_pause_is_a_run_on():
    sample_rate = 16000
    tone = _sine(0.15, 8.5, sample_rate)
    analyzer = PaceAnalyzer(sample_rate=sample_rate, continuous_speech_seconds=8.0)
    reading = analyzer.update(tone, 8.5)
    assert reading.run_on is True
    assert reading.rushing is False


def test_a_real_pause_resets_the_run_on_clock():
    sample_rate = 16000
    analyzer = PaceAnalyzer(sample_rate=sample_rate, continuous_speech_seconds=8.0)
    voiced = _sine(0.15, 3.0, sample_rate)
    silence = np.zeros(int(0.5 * sample_rate), dtype=np.float32)
    analyzer.update(voiced, 3.0)
    paused = analyzer.update(silence, 3.5)
    assert paused.continuous_speech_seconds == 0.0
    assert paused.run_on is False


def test_chime_is_a_bounded_tone_at_the_requested_frequency():
    sample_rate = 16000
    wave = synthesize_chime(880, duration=0.18, sample_rate=sample_rate, volume=0.2)
    spectrum = np.abs(np.fft.rfft(wave))
    frequencies = np.fft.rfftfreq(len(wave), 1 / sample_rate)
    peak = frequencies[int(np.argmax(spectrum))]
    assert peak == pytest.approx(880, abs=20)
    assert wave.dtype == np.float32
    assert float(np.max(np.abs(wave))) <= 0.2 + 1e-6
    posture = chime_for_cue("POSTURE", sample_rate)
    volume = chime_for_cue("VOLUME", sample_rate)
    assert posture.shape == volume.shape
    assert not np.allclose(posture, volume)
