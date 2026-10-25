"""Scripted landmarks and PCM for the hardware-free demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Sections are half-open [start, end) in seconds.
# posture, gaze, audience, audio, hand
_SECTIONS = (
    (0.0, 4.0, "slouch", "center", "neutral", "speech", None),
    (4.0, 8.0, "upright", "center", "neutral", "quiet", None),
    (8.0, 12.0, "upright", "center", "neutral", "rush", None),
    (12.0, 17.0, "upright", "left", "neutral", "speech", None),
    (17.0, 1.0e9, "upright", "center", "neutral", "speech", "palm"),
)


@dataclass(frozen=True)
class DemoSample:
    frame: np.ndarray
    pose: np.ndarray
    face: np.ndarray
    hand: np.ndarray | None
    audio: np.ndarray


def section_at(timestamp: float) -> tuple[str, str, str, str, str | None]:
    for start, end, posture, gaze, audience, audio, hand in _SECTIONS:
        if start <= timestamp < end:
            return posture, gaze, audience, audio, hand
    return _SECTIONS[-1][2:]


def neutral_pose() -> np.ndarray:
    landmarks = np.zeros((33, 3), dtype=np.float32)
    landmarks[0] = (0.50, 0.20, 0.0)
    landmarks[7] = (0.46, 0.24, 0.0)
    landmarks[8] = (0.54, 0.24, 0.0)
    landmarks[11] = (0.40, 0.45, 0.0)
    landmarks[12] = (0.60, 0.45, 0.0)
    landmarks[23] = (0.42, 0.75, 0.0)
    landmarks[24] = (0.58, 0.75, 0.0)
    return landmarks


def slouch_pose() -> np.ndarray:
    landmarks = neutral_pose()
    landmarks[7, 2] = -0.08
    landmarks[8, 2] = -0.08
    landmarks[12, 1] = 0.52
    return landmarks


def face_landmarks(gaze: str = "center", audience: str = "neutral") -> np.ndarray:
    landmarks = np.zeros((478, 3), dtype=np.float32)
    landmarks[33] = (0.35, 0.40, 0.0)
    landmarks[133] = (0.45, 0.40, 0.0)
    landmarks[159] = (0.40, 0.385, 0.0)
    landmarks[145] = (0.40, 0.415, 0.0)
    landmarks[468] = (0.40, 0.40, 0.0)
    landmarks[362] = (0.55, 0.40, 0.0)
    landmarks[263] = (0.65, 0.40, 0.0)
    landmarks[386] = (0.60, 0.385, 0.0)
    landmarks[374] = (0.60, 0.415, 0.0)
    landmarks[473] = (0.60, 0.40, 0.0)
    landmarks[107] = (0.44, 0.34, 0.0)
    landmarks[336] = (0.56, 0.34, 0.0)
    landmarks[105] = (0.40, 0.345, 0.0)
    landmarks[334] = (0.60, 0.345, 0.0)
    if gaze == "left":
        landmarks[468, 0] = 0.36
        landmarks[473, 0] = 0.56
    elif gaze == "right":
        landmarks[468, 0] = 0.44
        landmarks[473, 0] = 0.64
    elif gaze == "down":
        landmarks[468, 1] = 0.410
        landmarks[473, 1] = 0.410
    if audience == "confused":
        landmarks[107, 0] = 0.485
        landmarks[336, 0] = 0.515
    elif audience == "surprised":
        landmarks[159, 1] = 0.37
        landmarks[145, 1] = 0.43
        landmarks[386, 1] = 0.37
        landmarks[374, 1] = 0.43
        landmarks[105, 1] = 0.30
        landmarks[334, 1] = 0.30
    return landmarks


def hand_landmarks(
    kind: str = "open", wrist: tuple[float, float] = (0.5, 0.30)
) -> np.ndarray:
    landmarks = np.zeros((21, 3), dtype=np.float32)
    wrist_x, wrist_y = wrist
    landmarks[0] = (wrist_x, wrist_y, 0.0)
    knuckles = {
        5: (wrist_x - 0.08, wrist_y - 0.08),
        9: (wrist_x - 0.02, wrist_y - 0.10),
        13: (wrist_x + 0.04, wrist_y - 0.10),
        17: (wrist_x + 0.10, wrist_y - 0.08),
    }
    for index, position in knuckles.items():
        landmarks[index] = (position[0], position[1], 0.0)
    pip_for_mcp = {6: 5, 10: 9, 14: 13, 18: 17}
    tip_for_pip = {8: 6, 12: 10, 16: 14, 20: 18}
    for pip, mcp in pip_for_mcp.items():
        landmarks[pip] = landmarks[mcp] + np.array([0.0, -0.07, 0.0], dtype=np.float32)
    for tip, pip in tip_for_pip.items():
        landmarks[tip] = landmarks[pip] + np.array([0.0, -0.08, 0.0], dtype=np.float32)
    landmarks[2] = (wrist_x - 0.12, wrist_y - 0.04, 0.0)
    landmarks[3] = (wrist_x - 0.14, wrist_y - 0.08, 0.0)
    landmarks[4] = (wrist_x - 0.16, wrist_y - 0.13, 0.0)
    if kind == "fist":
        for tip, pip in tip_for_pip.items():
            landmarks[tip] = landmarks[pip] + np.array(
                [0.0, 0.04, 0.0], dtype=np.float32
            )
        landmarks[4] = landmarks[3] + np.array([0.03, 0.03, 0.0], dtype=np.float32)
    elif kind == "pinch":
        for tip, pip in tip_for_pip.items():
            landmarks[tip] = landmarks[pip] + np.array(
                [0.0, 0.04, 0.0], dtype=np.float32
            )
        landmarks[8] = (wrist_x - 0.02, wrist_y - 0.12, 0.0)
        landmarks[4] = (wrist_x - 0.015, wrist_y - 0.12, 0.0)
        landmarks[6] = (wrist_x - 0.03, wrist_y - 0.08, 0.0)
    return landmarks


def tone(
    start: int,
    count: int,
    sample_rate: int,
    frequency: float,
    amplitude: float,
) -> np.ndarray:
    t = (start + np.arange(count, dtype=np.float32)) / float(sample_rate)
    return (amplitude * np.sin(2.0 * np.pi * frequency * t)).astype(np.float32)


def pulsed_speech(
    start: int,
    count: int,
    sample_rate: int,
    syllables_per_sec: float,
    amplitude: float,
) -> np.ndarray:
    t = (start + np.arange(count, dtype=np.float32)) / float(sample_rate)
    period = 1.0 / syllables_per_sec
    phase = np.mod(t, period)
    dist = np.minimum(phase, period - phase)
    envelope = np.exp(-0.5 * (dist / 0.018) ** 2)
    carrier = np.sin(2.0 * np.pi * 170.0 * t)
    return (amplitude * envelope * carrier).astype(np.float32)


def audio_chunk(kind: str, start: int, count: int, sample_rate: int) -> np.ndarray:
    if kind == "quiet":
        return tone(start, count, sample_rate, 180.0, 0.003)
    if kind == "rush":
        return pulsed_speech(start, count, sample_rate, 6.5, 0.22)
    return pulsed_speech(start, count, sample_rate, 3.5, 0.2)


def background_frame(timestamp: float, width: int, height: int) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = (24, 18, 32)
    if width > 0:
        x = int((timestamp * 40) % width)
        frame[:, x : min(width, x + 2)] = (70, 55, 80)
    return frame


class DemoScene:
    def __init__(self, width: int, height: int, sample_rate: int = 16000) -> None:
        self.width = width
        self.height = height
        self.sample_rate = sample_rate

    def sample(
        self, timestamp: float, audio_start: int, audio_count: int
    ) -> DemoSample:
        posture, gaze, audience, audio_kind, hand = section_at(timestamp)
        pose = slouch_pose() if posture == "slouch" else neutral_pose()
        face = face_landmarks(gaze=gaze, audience=audience)
        hand_points = None
        if hand == "palm":
            hand_points = hand_landmarks("open", wrist=(0.50, 0.28))
        return DemoSample(
            frame=background_frame(timestamp, self.width, self.height),
            pose=pose,
            face=face,
            hand=hand_points,
            audio=audio_chunk(audio_kind, audio_start, audio_count, self.sample_rate),
        )
