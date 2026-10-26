"""Speak the session summary with xAI Grok text-to-speech.

The model is ``grok-tts``, served at ``POST https://api.x.ai/v1/tts``.
The official request body takes ``text``, ``voice_id``, and ``language``.
It does not take a separate model field. The key is ``XAI_API_KEY``.
"""

from __future__ import annotations

import io
import json
import os
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TTS_ENDPOINT = "https://api.x.ai/v1/tts"
TTS_MODEL = "grok-tts"
DEFAULT_VOICE_ID = "eve"
BUILTIN_VOICES = ("eve", "ara", "leo", "rex", "sal")
API_KEY_ENV = "XAI_API_KEY"


class VoiceError(RuntimeError):
    """The TTS call failed. The session summary is still available."""


@dataclass
class VoiceResult:
    skipped: str = ""
    path: Path | None = None
    played: bool = False
    error: str = ""


def resolve_voice_key(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    value = os.environ.get(API_KEY_ENV, "").strip()
    return value or None


def spoken_script(summary: str, strengths: list[str], improve: list[str]) -> str:
    """Short spoken recap. The written report stays the full version."""
    parts = [part.strip() for part in (summary or "",) if part and part.strip()]
    good = "; ".join(line.strip() for line in strengths[:3] if line and line.strip())
    work = "; ".join(line.strip() for line in improve[:3] if line and line.strip())
    if good:
        parts.append(f"What went well: {good}.")
    if work:
        parts.append(f"Work on next: {work}.")
    text = " ".join(" ".join(parts).split())
    if len(text) > 4000:
        text = text[:3997] + "..."
    return text


def tts_payload(
    text: str,
    voice_id: str = DEFAULT_VOICE_ID,
    language: str = "en",
    sample_rate: int = 16000,
) -> dict:
    return {
        "text": text,
        "voice_id": voice_id or DEFAULT_VOICE_ID,
        "language": language or "en",
        "text_normalization": True,
        "output_format": {"codec": "wav", "sample_rate": int(sample_rate)},
    }


def grok_tts(
    text: str,
    api_key: str,
    voice_id: str = DEFAULT_VOICE_ID,
    language: str = "en",
    sample_rate: int = 16000,
    timeout: float = 30.0,
    transport=None,
    endpoint: str = TTS_ENDPOINT,
) -> bytes:
    """Return wav bytes. ``transport`` receives the JSON body and skips the network."""
    payload = tts_payload(text, voice_id, language, sample_rate)
    if transport is not None:
        audio = transport(payload)
        return b"" if audio is None else bytes(audio)
    return _post_tts(endpoint, payload, api_key, timeout)


def _post_tts(endpoint: str, payload: dict, api_key: str, timeout: float) -> bytes:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise VoiceError(f"HTTP {exc.code} {detail}".strip()) from exc
    except urllib.error.URLError as exc:
        raise VoiceError(str(exc.reason or exc)) from exc


def wav_to_float32(audio: bytes) -> tuple[np.ndarray, int]:
    """Decode a 16-bit PCM wav. Mono is mixed down if the file has more channels."""
    with wave.open(io.BytesIO(audio), "rb") as handle:
        rate = int(handle.getframerate())
        channels = int(handle.getnchannels())
        width = int(handle.getsampwidth())
        frames = handle.readframes(handle.getnframes())
    if width != 2:
        raise VoiceError(f"Expected 16-bit wav, got {width * 8}-bit audio")
    if rate <= 0:
        raise VoiceError("Wav sample rate is missing")
    pcm = np.frombuffer(frames, dtype="<i2")
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    samples = (pcm.astype(np.float32) / 32767.0).reshape(-1)
    return samples, rate


def write_voice_file(audio: bytes, report_path: Path) -> Path:
    path = Path(report_path).with_suffix(".wav")
    path.write_bytes(audio)
    return path


def speak_after_session(
    summary: str,
    strengths: list[str],
    improve: list[str],
    enabled: bool = True,
    playback: bool = True,
    voice_id: str = DEFAULT_VOICE_ID,
    language: str = "en",
    api_key: str | None = None,
    report_path: Path | None = None,
    player=None,
    transport=None,
    timeout: float = 30.0,
) -> VoiceResult:
    """Synthesize, save beside the report, and play. Failures stay in the result."""
    if not enabled:
        return VoiceResult(skipped="disabled")
    script = spoken_script(summary, strengths, improve)
    if not script:
        return VoiceResult(skipped="nothing to say")
    key = resolve_voice_key(api_key)
    if not key and transport is None:
        return VoiceResult(skipped=f"{API_KEY_ENV} is not set")
    try:
        audio = grok_tts(
            script,
            api_key=key or "",
            voice_id=voice_id,
            language=language,
            timeout=timeout,
            transport=transport,
        )
    except Exception as exc:
        return VoiceResult(error=str(exc) or exc.__class__.__name__)
    if not audio:
        return VoiceResult(error="Grok voice returned no audio")
    path = None
    if report_path is not None:
        try:
            path = write_voice_file(audio, report_path)
        except OSError as exc:
            return VoiceResult(error=f"could not save voice audio ({exc})")
    if not playback:
        return VoiceResult(path=path, played=False)
    try:
        samples, rate = wav_to_float32(audio)
    except Exception as exc:
        return VoiceResult(path=path, error=str(exc) or exc.__class__.__name__)
    play_error = _play(samples, rate, player)
    if play_error:
        return VoiceResult(path=path, played=False, error=play_error)
    return VoiceResult(path=path, played=True)


def _play(samples: np.ndarray, rate: int, player) -> str:
    if player is not None:
        if not getattr(player, "active", True):
            return "No speaker is available."
        player.play(samples)
        return ""
    from src.audio.audio_feedback import AudioOutput

    output = AudioOutput(rate)
    if not output.start():
        return output.error or "No speaker is available."
    output.play(samples)
    # Hold the stream open until the clip has had time to drain.
    import time

    time.sleep(min(60.0, float(samples.size) / float(max(1, rate)) + 0.25))
    output.close()
    return ""
