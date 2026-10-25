import math

import numpy as np
import pytest

from src.demo_scene import face_landmarks, hand_landmarks, neutral_pose, slouch_pose
from src.vision.audience_tells import analyze_audience, tells_from_blendshapes
from src.vision.gaze_tracker import GazeTracker, measure_gaze
from src.vision.gesture_detector import GestureDetector, is_open_palm, is_pinch
from src.vision.mediapipe_backend import MediaPipeBackend
from src.vision.pose_tracker import analyze_posture, shoulder_tilt_degrees


def _pose_with_shoulder_drop(degrees: float) -> np.ndarray:
    landmarks = neutral_pose()
    dx = 0.20
    landmarks[12, 1] = landmarks[11, 1] + dx * math.tan(math.radians(degrees))
    return landmarks


def test_shoulder_tilt_matches_the_arctan_formula():
    landmarks = _pose_with_shoulder_drop(13.0)
    expected = math.degrees(
        math.atan2(
            landmarks[12, 1] - landmarks[11, 1], landmarks[12, 0] - landmarks[11, 0]
        )
    )
    assert shoulder_tilt_degrees(landmarks) == pytest.approx(expected)
    assert expected == pytest.approx(13.0, abs=1e-4)


def test_level_pose_does_not_ask_the_speaker_to_straighten():
    metrics = analyze_posture(neutral_pose())
    assert metrics.shoulder_tilt_deg == pytest.approx(0.0, abs=1e-4)
    assert metrics.neck_slump_deg == pytest.approx(0.0, abs=1e-4)
    assert metrics.ear_forward_offset == pytest.approx(0.0, abs=1e-6)
    assert metrics.straighten is False


def test_twelve_degrees_is_the_slouch_boundary():
    assert analyze_posture(_pose_with_shoulder_drop(12.0)).straighten is False
    assert analyze_posture(_pose_with_shoulder_drop(13.0)).straighten is True


def test_forward_head_uses_depth_and_a_two_dimensional_fallback():
    forward = neutral_pose()
    forward[7, 2] = -0.032
    forward[8, 2] = -0.032
    metrics = analyze_posture(forward)
    assert metrics.ear_forward_offset == pytest.approx(0.16, abs=1e-6)
    assert metrics.straighten is True

    flat = neutral_pose()[:, :2].copy()
    flat[7, 0] -= 0.04
    flat[8, 0] -= 0.04
    flat_metrics = analyze_posture(flat)
    assert flat_metrics.ear_forward_offset == pytest.approx(0.2, abs=1e-6)
    assert flat_metrics.straighten is True


def test_demo_slouch_pose_triggers():
    assert analyze_posture(slouch_pose()).straighten is True


def test_centered_gaze_is_not_a_cue_until_it_lingers():
    tracker = GazeTracker()
    sample = measure_gaze(face_landmarks("center"))
    assert sample.horizontal == pytest.approx(0.5, abs=1e-6)
    assert sample.away is False
    state = tracker.update(face_landmarks("center"), 0.0)
    assert state.cue is None
    state = tracker.update(face_landmarks("left"), 1.0)
    assert state.away_seconds == pytest.approx(1.0)
    assert state.cue is None


def test_looking_away_for_four_seconds_requests_a_scan():
    tracker = GazeTracker()
    state = None
    for step in range(0, 9):
        state = tracker.update(face_landmarks("left"), step * 0.5)
    assert state is not None
    assert state.away_seconds >= 4.0
    assert state.cue == "look_right"
    assert state.display_text == "Look right"
    assert state.icon == "arrow_right"


def test_looking_down_requests_eye_contact():
    tracker = GazeTracker()
    state = None
    for step in range(0, 9):
        state = tracker.update(face_landmarks("down"), float(step))
    assert state is not None
    assert state.cue == "eye_contact"
    assert state.icon == "target"


def test_audience_furrow_and_surprise_from_landmarks():
    neutral = analyze_audience(face_landmarks())
    assert neutral.confused is False
    assert neutral.surprised is False
    confused = analyze_audience(face_landmarks(audience="confused"))
    assert confused.confused is True
    assert confused.brow_furrow > 0
    surprised = analyze_audience(face_landmarks(audience="surprised"))
    assert surprised.surprised is True
    assert surprised.eye_widen > 0
    assert surprised.brow_raise > 0


def test_blendshape_names_are_normalized():
    tells = tells_from_blendshapes(
        {
            "brow_down_left": 0.8,
            "browDownRight": 0.6,
            "eyeWideLeft": 0.7,
            "eyeWideRight": 0.7,
            "browInnerUp": 0.5,
            "browOuterUpLeft": 0.5,
            "browOuterUpRight": 0.5,
        }
    )
    assert tells.confused is True
    assert tells.surprised is True


def test_open_palm_pinch_and_swipe():
    assert is_open_palm(hand_landmarks("open", (0.5, 0.3))) is True
    assert is_open_palm(hand_landmarks("fist")) is False
    assert is_pinch(hand_landmarks("pinch")) is True
    raised = hand_landmarks("open", (0.5, 0.28))
    lowered = hand_landmarks("open", (0.5, 0.8))
    detector = GestureDetector()
    assert detector.update(raised, 0.0).raised is True
    assert detector.update(lowered, 0.1).raised is False

    swiper = GestureDetector(swipe_distance=0.12, swipe_window=0.4)
    events = [
        swiper.update(hand_landmarks("fist", (x, 0.7)), index * 0.08)
        for index, x in enumerate((0.20, 0.28, 0.36, 0.48))
    ]
    assert any(
        event.swipe == "right" and event.name == "swipe_right" for event in events
    )


def test_missing_mediapipe_models_do_not_raise(tmp_path):
    backend = MediaPipeBackend(tmp_path)
    assert backend.load() is False
    assert "Missing MediaPipe model" in (backend.error or "")
