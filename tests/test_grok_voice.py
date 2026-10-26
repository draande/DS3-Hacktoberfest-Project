"""Grok voice after the session. The HTTP call is a fake transport."""

import io
import wave

import numpy as np

from src.audio.grok_voice import (
    API_KEY_ENV,
    TTS_ENDPOINT,
    TTS_MODEL,
    VoiceError,
    grok_tts,
    speak_after_session,
    spoken_script,
    tts_payload,
    wav_to_float32,
)
from src.main import parse_args


def _tiny_wav(rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x10" * 320)
    return buffer.getvalue()


class _Player:
    def __init__(self, active: bool = True) -> None:
        self.active = active
        self.played: list[np.ndarray] = []

    def play(self, samples: np.ndarray) -> None:
        self.played.append(np.asarray(samples))


def test_payload_matches_the_tts_endpoint_and_skips_a_model_field():
    seen: list[dict] = []

    def transport(payload: dict) -> bytes:
        seen.append(payload)
        return _tiny_wav()

    audio = grok_tts(
        "Hello from Wingman.",
        api_key="test-key",
        voice_id="ara",
        language="en",
        transport=transport,
    )
    assert audio.startswith(b"RIFF")
    body = seen[0]
    assert "model" not in body
    assert body["text"] == "Hello from Wingman."
    assert body["voice_id"] == "ara"
    assert body["language"] == "en"
    assert body["output_format"] == {"codec": "wav", "sample_rate": 16000}
    assert TTS_ENDPOINT == "https://api.x.ai/v1/tts"
    assert TTS_MODEL == "grok-tts"
    assert tts_payload("Hi")["voice_id"] == "eve"


def test_missing_key_and_disabled_voice_do_not_call_out(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("network")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    missing = speak_after_session(
        "The pace settled.",
        ["Clear voice."],
        ["Pause more."],
        api_key=None,
    )
    assert API_KEY_ENV in missing.skipped
    assert missing.played is False
    disabled = speak_after_session(
        "The pace settled.",
        ["Clear voice."],
        ["Pause more."],
        enabled=False,
        api_key="test-key",
        transport=boom,
    )
    assert disabled.skipped == "disabled"


def test_a_failed_call_keeps_the_summary_path_and_skips_playback(tmp_path):
    def fail(_payload):
        raise VoiceError("HTTP 401 unauthorized")

    result = speak_after_session(
        "The pace settled.",
        ["Clear voice."],
        ["Pause more."],
        api_key="test-key",
        report_path=tmp_path / "session.md",
        transport=fail,
    )
    assert "401" in result.error
    assert result.played is False
    assert result.path is None
    assert not (tmp_path / "session.wav").exists()


def test_playback_flag_saves_wav_and_can_skip_the_speaker(tmp_path):
    player = _Player()
    script = spoken_script(
        "The pace settled in the middle.",
        ["Clear voice.", "Steady gaze."],
        ["Pause more."],
    )
    assert "The pace settled" in script
    assert "Clear voice" in script
    assert "Pause more" in script

    saved = speak_after_session(
        "The pace settled in the middle.",
        ["Clear voice."],
        ["Pause more."],
        playback=False,
        api_key="test-key",
        voice_id="rex",
        report_path=tmp_path / "session-1.md",
        player=player,
        transport=lambda _payload: _tiny_wav(),
    )
    assert saved.played is False
    assert player.played == []
    assert saved.path == tmp_path / "session-1.wav"
    samples, rate = wav_to_float32(saved.path.read_bytes())
    assert rate == 16000
    assert samples.size == 320

    quiet = _Player(active=False)
    silent = speak_after_session(
        "The pace settled in the middle.",
        ["Clear voice."],
        ["Pause more."],
        playback=True,
        api_key="test-key",
        report_path=tmp_path / "session-2.md",
        player=quiet,
        transport=lambda _payload: _tiny_wav(),
    )
    assert silent.path == tmp_path / "session-2.wav"
    assert silent.played is False
    assert "speaker" in silent.error.lower()
    assert quiet.played == []

    heard = speak_after_session(
        "The pace settled in the middle.",
        ["Clear voice."],
        ["Pause more."],
        playback=True,
        api_key="test-key",
        voice_id="eve",
        report_path=tmp_path / "session-3.md",
        player=_Player(),
        transport=lambda payload: _tiny_wav() if payload["voice_id"] == "eve" else b"",
    )
    assert heard.played is True
    assert parse_args(["--no-voice"]).no_voice is True
    assert parse_args(["--no-voice-playback"]).no_voice_playback is True
