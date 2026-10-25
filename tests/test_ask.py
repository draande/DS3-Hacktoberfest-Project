"""Direct Gemma asks: typed, spoken, and the one-shot CLI. No network."""

import io
import sys
import time

import numpy as np

from src.audio.speech_transcriber import (
    load_transcriber,
    pcm16_wav,
    prepare_spoken_ask,
)
from src.config_loader import load_config
from src.controls.session_control import SessionControl
from src.hud.ask_input import AskComposer, SpeechRecorder, dispatch_preview_key
from src.hud.renderer import HudRenderer, HudState
from src.hud.sidebar import FeedbackSidebar
from src.intelligence.gemma_client import (
    CoachWindow,
    GemmaClient,
    ask_error_line,
    build_ask_prompt,
    model_accepts_inline_audio,
    parse_ask_reply,
)
from src.main import deliver_spoken_ask, main, parse_args, perform_ask


def _wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("timed out waiting for the ask")


def _control() -> SessionControl:
    return SessionControl(["ironman", "minimalist"], "ironman")


class _FakeTranscriber:
    def __init__(self, text: str = "say hello") -> None:
        self.text = text
        self.calls = 0

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        self.calls += 1
        assert sample_rate > 0
        assert samples.size > 0
        return self.text


def test_default_gemma_models_do_not_accept_inline_audio():
    assert model_accepts_inline_audio("gemma-4-26b-a4b-it") is False
    assert model_accepts_inline_audio("gemma-4-31b-it") is False
    assert model_accepts_inline_audio("gemma-4-e2b-it") is True
    assert model_accepts_inline_audio("gemma-4-e4b-it") is True
    assert model_accepts_inline_audio("gemma-4-12b-it") is True


def test_ask_prompt_includes_the_question_and_recent_metrics():
    prompt = build_ask_prompt(
        "how's my pacing?",
        CoachWindow(avg_wpm=190, avg_db=36, seconds=45, posture_ratio=0.2),
    )
    assert "how's my pacing?" in prompt
    assert "wpm=190" in prompt
    assert "db=36" in prompt
    assert "1-3 lines" in prompt
    reply = parse_ask_reply('```json\n{"lines":["Hello.", "Pace is a bit fast"]}\n```')
    assert reply == ["Hello.", "Pace is a bit fast"]
    assert parse_ask_reply("Hello.") == ["Hello."]
    assert (
        ask_error_line(TimeoutError("timed out after 45.0s")) == "Timed out. Try again."
    )


def test_ask_returns_immediately_and_keeps_a_short_reply():
    def slow(prompt: str) -> str:
        time.sleep(0.6)
        assert "say hello" in prompt
        return '{"lines":["Hello there"]}'

    client = GemmaClient(api_key="test-key", transport=slow, min_interval=0, timeout=2)
    started = time.perf_counter()
    client.ask("say hello", CoachWindow(seconds=10), now=0)
    assert time.perf_counter() - started < 0.2
    assert client.ask_state.phase == "thinking"
    assert client.ask_state.question == "say hello"
    _wait_until(lambda: client.ask_state.phase == "reply")
    assert client.ask_state.lines == ("Hello there",)
    assert client.pending is False
    client.close()


def test_ask_cancels_an_insight_and_skips_the_next_one():
    def transport(prompt: str) -> str:
        if "Q:" in prompt:
            return '{"lines":["Hello there"]}'
        time.sleep(0.35)
        return '{"bullets":["Insight should wait"]}'

    client = GemmaClient(
        api_key="test-key", transport=transport, min_interval=0, timeout=2
    )
    window = CoachWindow(avg_wpm=180, seconds=30)
    assert client.poll(window, now=0) is None
    assert client.pending is True
    client.ask("say hello", window, now=0.1)
    _wait_until(lambda: client.ask_state.phase == "reply")
    time.sleep(0.45)
    assert client.ask_state.lines == ("Hello there",)
    assert client.bullets is None
    assert client.poll(window, now=1) is None
    assert client.pending is False
    client.poll(window, now=2)
    assert client.pending is True
    client.close()


def test_ask_timeout_and_429_are_short_sidebar_lines():
    def slow(_prompt: str) -> str:
        time.sleep(0.45)
        return '{"lines":["too late"]}'

    timed = GemmaClient(api_key="test-key", transport=slow, min_interval=0, timeout=0.1)
    started = time.perf_counter()
    timed.ask("say hello", CoachWindow(), now=0)
    _wait_until(lambda: timed.ask_state.phase == "error")
    assert time.perf_counter() - started < 0.4
    assert timed.ask_state.lines == ("Timed out. Try again.",)
    time.sleep(0.5)
    timed.close()

    holder = {"fn": lambda _prompt: '{"bullets":["Posture slipping often"]}'}

    def transport(prompt: str) -> str:
        return holder["fn"](prompt)

    client = GemmaClient(
        api_key="test-key", transport=transport, min_interval=10, timeout=2
    )
    window = CoachWindow(posture_ratio=0.5, seconds=40)
    client.poll(window, now=0)
    _wait_until(lambda: client.bullets == ["Posture slipping often"])

    def raise_429(_prompt: str) -> str:
        error = RuntimeError("RESOURCE_EXHAUSTED")
        error.code = 429
        raise error

    holder["fn"] = raise_429
    client.ask("how's my pacing?", window, now=1)
    _wait_until(lambda: client.ask_state.phase == "error" and not client.pending)
    assert client.ask_state.lines == ("Gemma is busy. Try again.",)
    assert client.bullets == ["Posture slipping often"]
    assert client.status == "gemma"
    shown = client.poll(window, now=5)
    assert shown == ["Posture slipping often"]
    assert client.pending is False
    client.poll(window, now=20)
    time.sleep(0.05)
    assert client.pending is False
    client.poll(window, now=40)
    assert client.pending is True

    def raise_504(_prompt: str) -> str:
        error = RuntimeError("DEADLINE_EXCEEDED")
        error.code = 504
        raise error

    holder["fn"] = raise_504
    _wait_until(lambda: not client.pending, timeout=2)
    client.ask("say hello", window, now=100)
    _wait_until(lambda: client.ask_state.phase == "error" and not client.pending)
    assert client.ask_state.lines == ("Gemma timed out. Try again.",)
    assert client.bullets == ["Posture slipping often"]
    client.close()


def test_typed_keys_do_not_trigger_shortcuts_and_f_still_works():
    composer = AskComposer()
    control = _control()
    opened = dispatch_preview_key("g", composer, control, 0.0)
    assert opened.action is None
    assert composer.open is True
    assert composer.text == ""
    for key in ("m", "s", "v", "f", "q", "space", "t"):
        dispatch_preview_key(key, composer, control, 1.0)
    assert composer.text == "msvfq t"
    assert control.theme_name == "ironman"
    assert control.filter.is_snoozed(1.0) is False
    cancelled = dispatch_preview_key("escape", composer, control, 2.0)
    assert cancelled.quit is False
    assert composer.open is False
    assert dispatch_preview_key("quit", composer, control, 3.0).quit is True
    assert dispatch_preview_key("fullscreen", composer, control, 4.0).fullscreen == (
        "fullscreen"
    )
    dispatch_preview_key("g", composer, control, 5.0)
    dispatch_preview_key("h", composer, control, 5.1)
    dispatch_preview_key("i", composer, control, 5.2)
    dispatch_preview_key("backspace", composer, control, 5.3)
    dispatch_preview_key("i", composer, control, 5.4)
    sent = dispatch_preview_key("enter", composer, control, 5.5)
    assert sent.submitted == "hi"
    assert composer.open is False
    assert dispatch_preview_key("v", composer, control, 6.0).speech_toggle is True
    assert dispatch_preview_key("m", composer, control, 7.0).action == "theme"


def test_sidebar_ask_outlasts_cues_and_shows_elapsed_time():
    bar = FeedbackSidebar(ttl_seconds=10, fade_seconds=1)
    bar.push(["Straighten your posture"], 0.0)
    bar.push_insights(["Pace steadier now"], 1.0)
    bar.show_ask("how's my pacing?", "reply", ["Pace is a bit fast"], started=0.0)
    assert bar.visible(30.0) == []
    assert bar.visible_insights(30.0)[0][0] == "Pace steadier now"
    rows = bar.visible_ask()
    assert rows[0] == "YOU: how's my pacing?"
    assert rows[1] == "ASK"
    assert rows[2] == "Pace is a bit fast"
    bar.push_insights(["Volume stays too quiet"], 31.0)
    assert bar.visible_ask()[0] == "YOU: how's my pacing?"
    bar.show_ask("say hello", "thinking", [], started=time.monotonic() - 12.2)
    assert bar.visible_ask() == ["YOU: say hello", "THINKING... 12s"]


def test_ask_bar_is_drawn_only_while_typing():
    config = load_config()
    renderer = HudRenderer(
        config.theme("ironman"),
        sidebar_config=config.hud.sidebar,
        avatar_config=config.hud.avatar,
    )
    frame = np.zeros((180, 320, 3), np.uint8)
    state = HudState(
        remaining_seconds=100,
        remaining_ratio=0.8,
        cue=None,
        volume_db=40,
        wpm=140,
        pulse_t=0,
    )
    plain = renderer.render(frame.copy(), state)
    state.ask_draft = "hello"
    asked = renderer.render(frame.copy(), state)
    assert asked.shape == plain.shape
    assert not np.array_equal(plain, asked)


def test_spoken_ask_uses_audio_only_on_models_that_accept_it():
    tone = np.full(16000, 0.2, dtype=np.float32)
    audio_model = prepare_spoken_ask(tone, 16000, "gemma-4-12b-it", None)
    assert audio_model.kind == "audio"
    assert audio_model.audio is not None and audio_model.audio[:4] == b"RIFF"
    missing = prepare_spoken_ask(tone, 16000, "gemma-4-26b-a4b-it", None)
    assert missing.kind == "notice"
    assert "Type with G" in missing.text
    assert load_transcriber() is None
    short = prepare_spoken_ask(
        tone[:100], 16000, "gemma-4-26b-a4b-it", _FakeTranscriber()
    )
    assert short.text == "Didn't catch that."
    heard = prepare_spoken_ask(
        tone, 16000, "gemma-4-31b-it", _FakeTranscriber("  say hello ")
    )
    assert heard.kind == "text"
    assert heard.text == "say hello"

    seen: dict = {}

    def transport(prompt: str, audio: bytes | None = None) -> str:
        seen["prompt"] = prompt
        seen["audio"] = audio
        return '{"lines":["You said hello"]}'

    client = GemmaClient(
        model="gemma-4-12b-it",
        api_key="test-key",
        transport=transport,
        min_interval=0,
        timeout=2,
    )
    request = deliver_spoken_ask(
        tone, 16000, client, CoachWindow(seconds=12), 1.0, None
    )
    assert request.kind == "audio"
    _wait_until(lambda: client.ask_state.phase == "reply")
    assert seen["audio"][:4] == b"RIFF"
    assert "sec=12" in seen["prompt"]
    assert pcm16_wav(tone, 16000)[:4] == b"RIFF"

    text_client = GemmaClient(
        model="gemma-4-26b-a4b-it",
        api_key="test-key",
        transport=lambda prompt: '{"lines":["Hello there"]}',
        min_interval=0,
        timeout=2,
    )
    fake = _FakeTranscriber()
    typed = deliver_spoken_ask(
        tone, 16000, text_client, CoachWindow(avg_wpm=150, seconds=20), 2.0, fake
    )
    assert typed.kind == "text"
    assert fake.calls == 1
    _wait_until(lambda: text_client.ask_state.phase == "reply")
    assert text_client.ask_state.question == "say hello"
    assert text_client.ask_state.lines == ("Hello there",)

    quiet = GemmaClient(api_key="test-key", transport=lambda _prompt: "unused")
    notice = deliver_spoken_ask(tone, 16000, quiet, CoachWindow(), 0.0, None)
    assert notice.kind == "notice"
    assert quiet.pending is False
    assert "not installed" in quiet.ask_state.lines[0]
    client.close()
    text_client.close()


def test_recorder_stops_itself_after_the_cap():
    recorder = SpeechRecorder(max_seconds=1.0)
    assert recorder.toggle() == "start"
    assert recorder.add(np.zeros(8000, np.float32), 16000) is False
    assert recorder.add(np.zeros(8000, np.float32), 16000) is True
    assert recorder.active is False
    audio, rate = recorder.take()
    assert rate == 16000
    assert audio.size == 16000


def test_ask_flag_prints_a_reply_without_starting_the_session(capsys, monkeypatch):
    assert parse_args(["--ask", "say hello"]).ask == "say hello"
    assert parse_args(["--ask"]).ask == "-"
    assert parse_args([]).ask is None

    code = main(["--ask", "say hello", "--offline"])
    captured = capsys.readouterr()
    assert code == 1
    assert "YOU: say hello" in captured.out
    assert "ASK: Gemma is offline." in captured.out
    assert "frames=" not in captured.out

    monkeypatch.setattr(sys, "stdin", io.StringIO("how's my pacing?\n"))
    code = main(["--ask"])
    captured = capsys.readouterr()
    assert code == 1
    assert "YOU: how's my pacing?" in captured.out
    assert "frames=" not in captured.out

    seen: dict = {}

    def transport(prompt: str) -> str:
        seen["prompt"] = prompt
        return '{"lines":["Hello there", "Pace is fine"]}'

    client = GemmaClient(api_key="test-key", transport=transport, timeout=2)
    code, text = perform_ask("how's my pacing?", client=client)
    assert code == 0
    assert text.startswith("YOU: how's my pacing?")
    assert "ASK: Hello there" in text
    assert "ASK: Pace is fine" in text
    assert "how's my pacing?" in seen["prompt"]
    assert "wpm=" in seen["prompt"]
    client.close()
