"""Speech-to-text for spoken asks.

``gemma-4-26b-a4b-it`` and ``gemma-4-31b-it`` reject inline audio. E2B, E4B,
and 12B accept a wav part. Anything with ``transcribe(samples, sample_rate)``
can stand in for :class:`FasterWhisperTranscriber`.
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from typing import Protocol

import numpy as np

_AUDIO_TOKENS = ("e2b", "e4b", "12b", "gemma-3n")


class SpeechTranscriber(Protocol):
    def transcribe(self, samples: np.ndarray, sample_rate: int) -> str: ...


@dataclass
class SpokenRequest:
    """``text`` is the question. ``audio`` is a wav for models that accept it."""

    kind: str
    text: str = ""
    audio: bytes | None = None


def model_accepts_inline_audio(model: str) -> bool:
    """26B and 31B return 400 if a session sends them an audio part."""
    name = (model or "").lower()
    if "26b" in name or "31b" in name:
        return False
    return any(token in name for token in _AUDIO_TOKENS)


def pcm16_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    """16-bit little-endian mono wav, which is what ``audio/wav`` expects."""
    clipped = np.clip(np.asarray(samples, dtype=np.float32).reshape(-1), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


def prepare_spoken_ask(
    samples: np.ndarray,
    sample_rate: int,
    model: str,
    transcriber: SpeechTranscriber | None,
) -> SpokenRequest:
    """Turn a push-to-talk clip into a question, an audio part, or a sidebar notice."""
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    rate = int(sample_rate)
    if rate <= 0 or audio.size < max(1, int(rate * 0.3)):
        return SpokenRequest("notice", "Didn't catch that.")
    if model_accepts_inline_audio(model):
        return SpokenRequest(
            "audio",
            text="What should I change? Answer in 1-3 short lines.",
            audio=pcm16_wav(audio, rate),
        )
    if transcriber is None:
        return SpokenRequest(
            "notice",
            "Speech transcription is not installed. Type with G.",
        )
    try:
        text = " ".join(str(transcriber.transcribe(audio, rate)).split())
    except Exception:
        return SpokenRequest("notice", "Could not transcribe that. Type with G.")
    if not text:
        return SpokenRequest("notice", "Didn't catch that.")
    return SpokenRequest("text", text=text)


class FasterWhisperTranscriber:
    """Optional local model. ``tiny`` or ``base`` on CPU. Imported only when installed."""

    def __init__(
        self,
        model_size: str = "tiny",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        from faster_whisper import WhisperModel

        size = model_size if model_size in {"tiny", "base"} else "tiny"
        self._model = WhisperModel(size, device=device, compute_type=compute_type)

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        segments, _info = self._model.transcribe(
            audio,
            language="en",
            vad_filter=True,
            beam_size=1,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()


def load_transcriber(model_size: str = "tiny") -> SpeechTranscriber | None:
    """``None`` when the ``stt`` extra is not installed. Typed asks still work."""
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return None
    return FasterWhisperTranscriber(model_size=model_size)
