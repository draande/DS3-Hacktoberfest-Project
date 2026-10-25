"""Speech-to-text for spoken asks.

The default Gemma model (``gemma-4-26b-a4b-it``) does not accept audio.
E2B, E4B, and 12B do; those send a wav part instead of coming through here.

Anything with ``transcribe(samples, sample_rate) -> str`` can replace
:class:`FasterWhisperTranscriber`. That includes a teammate's transcriber:
pass the object into ``run_session`` and this module is not imported.
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from src.intelligence.gemma_client import model_accepts_inline_audio


class SpeechTranscriber(Protocol):
    def transcribe(self, samples: np.ndarray, sample_rate: int) -> str: ...


@dataclass
class SpokenRequest:
    """``text`` goes to Gemma as a question. ``audio`` is a wav for models that take it."""

    kind: str
    text: str = ""
    audio: bytes | None = None


def pcm16_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    """16-bit little-endian mono wav. That is the layout ``audio/wav`` expects."""
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
    if rate <= 0 or audio.size < int(rate * 0.3):
        return SpokenRequest("notice", "Didn't catch that.")
    if model_accepts_inline_audio(model):
        return SpokenRequest(
            "audio",
            text="What did I just ask? Answer in 1-3 short lines.",
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
    """Optional local model. ``tiny`` or ``base`` on CPU. Not imported unless installed."""

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
