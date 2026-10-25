"""Optional MediaPipe landmarker wrapper for live camera mode.

Demo mode and unit tests feed synthetic landmarks straight into the math
modules. This backend is only constructed when a webcam or video file is
used, and it stays inactive until the `.task` models are on disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = REPO_ROOT / "models"

MODEL_FILES = {
    "pose": "pose_landmarker_lite.task",
    "face": "face_landmarker.task",
    "hand": "hand_landmarker.task",
}

MODEL_URLS = {
    "pose": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    ),
    "face": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    ),
    "hand": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    ),
}


@dataclass
class VisionObservation:
    pose: np.ndarray | None = None
    face: np.ndarray | None = None
    hands: np.ndarray | None = None
    blendshapes: dict[str, float] | None = None


class MediaPipeBackend:
    def __init__(self, model_dir: Path | None = None) -> None:
        self.model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        self.ready = False
        self.error: str | None = None
        self._pose = None
        self._face = None
        self._hand = None

    def missing_models(self) -> list[str]:
        return [
            name
            for name, filename in MODEL_FILES.items()
            if not (self.model_dir / filename).is_file()
        ]

    def load(self) -> bool:
        missing = self.missing_models()
        if missing:
            self.ready = False
            self.error = (
                "Missing MediaPipe model files: "
                + ", ".join(missing)
                + f". Download them into {self.model_dir} (python main.py --download-models)."
            )
            return False
        try:
            from mediapipe.tasks.python.core import base_options as base_options_lib
            from mediapipe.tasks.python.vision import (
                FaceLandmarker,
                FaceLandmarkerOptions,
            )
            from mediapipe.tasks.python.vision import (
                HandLandmarker,
                HandLandmarkerOptions,
            )
            from mediapipe.tasks.python.vision import (
                PoseLandmarker,
                PoseLandmarkerOptions,
            )
            from mediapipe.tasks.python.vision.core import (
                vision_task_running_mode as running_mode,
            )

            image_mode = running_mode.VisionTaskRunningMode.IMAGE
            self._pose = PoseLandmarker.create_from_options(
                PoseLandmarkerOptions(
                    base_options=base_options_lib.BaseOptions(
                        model_asset_path=str(self.model_dir / MODEL_FILES["pose"])
                    ),
                    running_mode=image_mode,
                    num_poses=1,
                )
            )
            self._face = FaceLandmarker.create_from_options(
                FaceLandmarkerOptions(
                    base_options=base_options_lib.BaseOptions(
                        model_asset_path=str(self.model_dir / MODEL_FILES["face"])
                    ),
                    running_mode=image_mode,
                    num_faces=1,
                    output_face_blendshapes=True,
                )
            )
            self._hand = HandLandmarker.create_from_options(
                HandLandmarkerOptions(
                    base_options=base_options_lib.BaseOptions(
                        model_asset_path=str(self.model_dir / MODEL_FILES["hand"])
                    ),
                    running_mode=image_mode,
                    num_hands=1,
                )
            )
        except Exception as exc:  # pragma: no cover - depends on native model load
            self.ready = False
            self.error = f"MediaPipe failed to load: {exc}"
            return False
        self.ready = True
        self.error = None
        return True

    def detect(self, frame_bgr: np.ndarray) -> VisionObservation:
        if not self.ready:
            return VisionObservation()
        import cv2
        from mediapipe.tasks.python.vision.core import image as image_lib

        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        mp_image = image_lib.Image(image_lib.ImageFormat.SRGB, rgb)
        pose_result = self._pose.detect(mp_image)
        face_result = self._face.detect(mp_image)
        hand_result = self._hand.detect(mp_image)
        blendshapes = None
        if getattr(face_result, "face_blendshapes", None):
            blendshapes = {}
            for category in face_result.face_blendshapes[0]:
                if category.category_name:
                    blendshapes[category.category_name] = float(category.score or 0.0)
        hands = hand_result.hand_landmarks[0] if hand_result.hand_landmarks else None
        return VisionObservation(
            pose=(
                _to_array(pose_result.pose_landmarks[0])
                if pose_result.pose_landmarks
                else None
            ),
            face=(
                _to_array(face_result.face_landmarks[0])
                if face_result.face_landmarks
                else None
            ),
            hands=_to_array(hands) if hands else None,
            blendshapes=blendshapes,
        )

    def close(self) -> None:
        for detector in (self._pose, self._face, self._hand):
            close = getattr(detector, "close", None)
            if close is not None:
                close()
        self._pose = self._face = self._hand = None
        self.ready = False


def download_models(model_dir: Path | None = None) -> list[Path]:
    """Download the three lite/float16 task bundles. Requires network."""
    destination = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, url in MODEL_URLS.items():
        target = destination / MODEL_FILES[name]
        if not target.is_file():
            urlretrieve(url, target)
        written.append(target)
    return written


def _to_array(landmarks) -> np.ndarray:
    array = np.zeros((len(landmarks), 3), dtype=np.float32)
    for index, landmark in enumerate(landmarks):
        array[index, 0] = float(landmark.x or 0.0)
        array[index, 1] = float(landmark.y or 0.0)
        array[index, 2] = float(landmark.z or 0.0)
    return array
