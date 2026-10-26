"""Ask Gemma stays on the Gemma path and leaves the insight loop alone."""

import threading
import time
import wave
from io import BytesIO

import numpy as np

from src.audio.speech_transcriber import (
    model_accepts_inline_audio,
    prepare_spoken_ask,
)
from src.controls.session_control import SessionControl
from src.hud.ask_input import AskComposer, SpeechRecorder, dispatch_preview_key
from src.hud.renderer import PreviewWindow
from src.hud.sidebar import FeedbackSidebar
from src.intelligence.ask_gemma import (
    AskGemma,
    ask_error_line,
    build_ask_prompt,
    parse_ask_reply,
)
from src.intelligence.gemma_client import CoachWindow, GemmaClient
from src.intelligence.insight_client import InsightClient
from src.main import parse_args, run_cli_ask


def _window() -> CoachWindow:
    return CoachWindow(
        cue_counts={"posture": 2},
        avg_db=48,
        avg_wpm=160,
        gaze_away_ratio=0.2,
        posture_ratio=0.4,
        seconds=30,
    )


class _Keys:
    def __init__(self) -> None:
        self.keys: list[str] = []

    def handle_key(self, key: str, now: float) -> str | None:
        self.keys.append(key)
        if key == "m":
            return "theme"
        return None


def test_typed_ask_consumes_shortcut_letters_until_enter():
    composer = AskComposer()
    control = _Keys()
    assert dispatch_preview_key("g", composer, control, 0).submitted is None
    assert composer.open
    for key in ("m", "s", "[", "]", "f", "q", "space", "v"):
        dispatch_preview_key(key, composer, control, 1)
    assert control.keys == []
    assert composer.text == "ms[]fq v"
    sent = dispatch_preview_key("enter", composer, control, 2)
    assert sent.submitted == "ms[]fq v"
    assert not composer.open


def test_escape_cancels_the_bar_and_closed_keys_still_quit():
    composer = AskComposer()
    control = _Keys()
    window = PreviewWindow(fullscreen=False)
    assert window.interpret_key("escape") == "quit"
    window.text_entry = True
    assert window.interpret_key("escape") == "escape"
    assert window.interpret_key("f") == "f"
    dispatch_preview_key("g", composer, control, 0)
    assert dispatch_preview_key("escape", composer, control, 1).submitted is None
    assert not composer.open
    window.text_entry = False
    assert window.interpret_key("q") == "quit"


def test_push_to_talk_stops_at_thirty_seconds():
    recorder = SpeechRecorder(max_seconds=30)
    assert recorder.toggle() == "start"
    assert not recorder.add(np.zeros(16000, dtype=np.float32), 16000)
    assert recorder.add(np.ones(16000 * 29, dtype=np.float32), 16000)
    audio, rate = recorder.take()
    assert rate == 16000
    assert audio.size == 16000 * 30
    assert not recorder.active


def test_text_models_transcribe_and_audio_models_keep_the_wav():
    assert not model_accepts_inline_audio("gemma-4-26b-a4b-it")
    assert not model_accepts_inline_audio("gemma-4-31b-it")
    assert model_accepts_inline_audio("gemma-4-e4b-it")
    samples = np.linspace(-0.2, 0.2, 8000, dtype=np.float32)

    class Transcriber:
        def transcribe(self, audio, sample_rate: int) -> str:
            assert sample_rate == 16000
            assert audio.size == 8000
            return "  how   was my pace  "

    spoken = prepare_spoken_ask(samples, 16000, "gemma-4-26b-a4b-it", Transcriber())
    assert spoken.kind == "text"
    assert spoken.text == "how was my pace"
    assert spoken.audio is None

    missing = prepare_spoken_ask(samples, 16000, "gemma-4-26b-a4b-it", None)
    assert missing.kind == "notice"
    assert "Type with G" in missing.text

    audio_model = prepare_spoken_ask(samples, 16000, "gemma-4-e4b-it", None)
    assert audio_model.kind == "audio"
    assert audio_model.audio is not None
    with wave.open(BytesIO(audio_model.audio), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getframerate() == 16000
        assert handle.getsampwidth() == 2

    short = prepare_spoken_ask(np.zeros(100, np.float32), 16000, "gemma-4-e4b-it", None)
    assert short.kind == "notice"


def test_parse_reply_keeps_three_plain_lines():
    assert parse_ask_reply('{"lines":["Slow the pace", "Look up"]}') == [
        "Slow the pace",
        "Look up",
    ]
    text = parse_ask_reply("- First line\n2. Second line\n* Third\nFourth")
    assert text == ["First line", "Second line", "Third"]
    prompt = build_ask_prompt("how was my pace?", _window())
    assert "Question: how was my pace?" in prompt
    assert "wpm=160" in prompt
    assert "posture:2" in prompt


def test_ask_does_not_delay_the_flash_loop():
    ask_started = threading.Event()
    release_ask = threading.Event()

    def gemma_transport(prompt: str) -> str:
        assert "Question:" in prompt
        ask_started.set()
        assert release_ask.wait(2)
        return "Keep the pace steady."

    gemma = GemmaClient(api_key="test-key", transport=gemma_transport, timeout=2)
    insight = InsightClient(
        api_key="test-key",
        transport=lambda _prompt: '{"bullets":["Posture slipping often"]}',
        min_interval=0,
        timeout=2,
    )
    gemma._sdk = object()
    insight._sdk = object()
    window = _window()
    asker = AskGemma(gemma)
    asker.submit("how is my pace", window, now=0)
    assert ask_started.wait(1)
    began = time.perf_counter()
    insight.poll(window, now=0)
    deadline = time.monotonic() + 2
    while insight.bullets is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert time.perf_counter() - began < 0.4
    assert insight.bullets == ["Posture slipping often"]
    assert insight.status == "flash"
    assert insight.failed == 0
    assert insight._lock is not gemma._lock
    assert insight._call_lock is not gemma._call_lock
    assert insight._sdk is not gemma._sdk
    assert gemma.pending is False
    assert gemma.bullets is None
    release_ask.set()
    answer = asker.wait(2)
    assert answer.lines == ["Keep the pace steady."]
    assert insight.bullets == ["Posture slipping often"]
    insight.close()
    gemma.close()


def test_a_second_ask_waits_and_a_failed_ask_leaves_insights_up():
    current = 0
    peak = 0
    lock = threading.Lock()

    def transport(prompt: str) -> str:
        nonlocal current, peak
        if "Question:" not in prompt:
            return '{"bullets":["Steady pace today"]}'
        with lock:
            current += 1
            peak = max(peak, current)
        time.sleep(0.08)
        with lock:
            current -= 1
        if "first" in prompt:
            return "First reply stays hidden."
        return "Second reply is shown."

    gemma = GemmaClient(api_key="test-key", transport=transport, timeout=2)
    insight = InsightClient(
        api_key="test-key",
        transport=lambda _prompt: '{"bullets":["Steady pace today"]}',
        min_interval=100,
        timeout=2,
    )
    window = _window()
    insight.poll(window, now=0)
    deadline = time.monotonic() + 2
    while insight.bullets is None and time.monotonic() < deadline:
        time.sleep(0.01)
    asker = AskGemma(gemma)
    asker.submit("first question", window, now=1)
    time.sleep(0.02)
    asker.submit("second question", window, now=2)
    state = asker.wait(2)
    assert peak == 1
    assert state.lines == ["Second reply is shown."]
    assert insight.bullets == ["Steady pace today"]
    assert insight.failed == 0
    assert insight.status == "flash"

    class Limited(Exception):
        code = 429

    def fail(prompt: str) -> str:
        raise Limited("resource exhausted")

    gemma.transport = fail
    asker.submit("again", window, now=3)
    failed = asker.wait(2)
    assert failed.phase == "error"
    assert failed.lines == ["Gemma is rate limited. Try again shortly."]
    assert gemma.failed == 0
    assert insight.failed == 0
    assert insight.bullets == ["Steady pace today"]
    assert insight.status == "flash"
    insight.close()
    gemma.close()


def test_timeout_and_offline_are_short_lines():
    def slow(prompt: str) -> str:
        time.sleep(0.4)
        return "too late"

    client = GemmaClient(
        api_key="test-key", transport=slow, min_interval=18, timeout=0.05
    )
    asker = AskGemma(client)
    asker.submit("hello", _window(), now=0)
    state = asker.wait(1)
    assert state.phase == "error"
    assert state.lines == ["Gemma timed out. Try again."]
    assert client.failed == 0
    client.close()

    offline = GemmaClient(offline=True)
    quiet = AskGemma(offline)
    quiet.submit("hello", now=0)
    assert quiet.wait(1).lines == ["Gemma is offline."]
    assert ask_error_line(TimeoutError("deadline")) == "Gemma timed out. Try again."


def test_sidebar_holds_an_answer_longer_than_a_cue_and_shows_elapsed():
    bar = FeedbackSidebar(ttl_seconds=12, fade_seconds=3)
    bar.show_ask("how was my pace", "thinking", [], 10.0, 10.0)
    thinking = bar.visible_ask(12.4)
    assert thinking is not None
    assert thinking.elapsed == 2
    assert thinking.phase == "thinking"
    bar.show_ask("how was my pace", "thinking", [], 10.0, 40.0)
    assert bar.visible_ask(40.0) is not None
    bar.show_ask("how was my pace", "answer", ["Keep the pace steady."], 10.0, 13.0)
    assert bar.visible_ask(13.0).lines == ("Keep the pace steady.",)
    assert bar.visible_ask(57.0) is not None
    assert bar.visible_ask(58.1) is None


def test_cli_ask_prints_the_question_and_does_not_open_a_session():
    options = parse_args(["--ask", "how was my pace?"])
    assert options.ask == "how was my pace?"
    assert options.display is False
    assert parse_args(["--ask"]).ask == "-"

    def transport(prompt: str) -> str:
        assert "how was my pace" in prompt
        return "Pace is a little fast.\nLook up more often."

    client = GemmaClient(api_key="test-key", transport=transport, timeout=2)
    text = run_cli_ask("how was my pace?", client)
    assert text.splitlines() == [
        "YOU: how was my pace?",
        "Pace is a little fast.",
        "Look up more often.",
    ]
    client.close()


def test_session_control_is_what_closed_preview_keys_still_reach():
    control = SessionControl(["ironman", "minimalist"], "ironman")
    composer = AskComposer()
    dispatched = dispatch_preview_key("m", composer, control, 1.0)
    assert dispatched.action == "theme"
    assert control.theme_name == "minimalist"
