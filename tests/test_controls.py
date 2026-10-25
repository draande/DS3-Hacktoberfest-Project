import numpy as np

from src.controls.session_control import SessionControl, TapDetector
from src.intelligence.gemma_client import make_cue
from src.vision.gesture_detector import GestureEvent


def _control() -> SessionControl:
    return SessionControl(
        themes=["ironman", "minimalist"],
        theme_name="ironman",
        slouch_degrees=12.0,
        ear_forward_offset=0.15,
        speak_up_db=42.0,
    )


def test_hotkeys_dismiss_snooze_theme_and_sensitivity():
    control = _control()
    cue = make_cue("HIGH", "POSTURE", "Straighten Posture", "arrow_up")
    assert control.filter.consider(cue, 0.0) is not None
    assert control.handle_key("space", 1.0) == "dismiss"
    assert control.filter.active is None
    assert control.handle_key("t", 1.2) == "dismiss"
    assert control.handle_key("enter", 1.3) == "dismiss"

    assert control.handle_key("s", 2.0) == "snooze"
    assert control.filter.is_snoozed(10.0) is True
    assert control.filter.consider(cue, 10.0) is None

    assert control.handle_key("m", 40.0) == "theme"
    assert control.theme_name == "minimalist"
    assert control.handle_key("[", 40.0) == "sensitivity"
    assert control.threshold_scale == pytest_approx(0.9)
    assert control.effective_slouch() == pytest_approx(10.8)
    assert control.handle_key("]", 40.0) == "sensitivity"
    assert control.threshold_scale == pytest_approx(1.0)


def test_raised_palm_snoozes_and_swipe_cycles_themes():
    control = _control()
    cue = make_cue("MEDIUM", "GAZE", "Look right", "arrow_right", duration_ms=5000)
    control.filter.consider(cue, 0.0)
    palm = GestureEvent(name="open_palm", open_palm=True, raised=True)
    assert control.handle_gesture(palm, 0.2) == "snooze"
    assert control.filter.active is None
    assert control.handle_gesture(palm, 0.4) is None
    assert control.filter.is_snoozed(1.0) is True

    later = _control()
    swipe = GestureEvent(name="swipe_right", swipe="right")
    assert later.handle_gesture(swipe, 0.0) == "theme"
    assert later.theme_name == "minimalist"
    pinch = GestureEvent(name="pinch", pinch=True)
    assert later.handle_gesture(pinch, 0.0) is None
    assert later.handle_gesture(pinch, 0.2) is None
    assert later.handle_gesture(GestureEvent(), 0.25) is None
    assert later.handle_gesture(pinch, 0.3) is None
    assert later.handle_gesture(pinch, 0.75) == "sensitivity"
    assert later.handle_gesture(pinch, 0.9) is None
    assert later.threshold_scale == pytest_approx(1.1)


def test_double_audio_spike_dismisses_the_active_cue():
    detector = TapDetector()
    loud = np.full(160, 0.8, dtype=np.float32)
    quiet = np.zeros(160, dtype=np.float32)
    assert detector.update(loud, 0.0) is False
    assert detector.update(quiet, 0.1) is False
    assert detector.update(loud, 0.2) is True

    control = _control()
    cue = make_cue("HIGH", "VOLUME", "Speak up", "volume_up")
    control.filter.consider(cue, 0.0)
    assert control.handle_audio(loud, 1.0) is None
    assert control.handle_audio(loud, 1.2) == "dismiss"
    assert control.filter.active is None


def pytest_approx(value: float):
    import pytest

    return pytest.approx(value)
