"""Running Flash Lite summary. No network, and not the Gemma path."""

import threading
import time

import numpy as np

from src.intelligence.gemma_client import CoachWindow, GemmaClient
from src.intelligence.insight_client import InsightClient
from src.intelligence.talk_summary import (
    FlashSummaryClient,
    TalkSummarizer,
    build_final_prompt,
    parse_summary_text,
)
from src.main import parse_args, run_session


def _tone(seconds: float, rate: int = 16000, amplitude: float = 0.2) -> np.ndarray:
    count = int(seconds * rate)
    ramp = np.linspace(0, 40, count, dtype=np.float32)
    return (amplitude * np.sin(ramp)).astype(np.float32)


class _Lines:
    def __init__(self, lines: list[str]) -> None:
        self.lines = list(lines)

    def transcribe(self, samples, sample_rate: int) -> str:
        del samples, sample_rate
        return self.lines.pop(0)


def _wait_for(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("timed out")


def test_folds_keep_earlier_speech_out_of_later_prompts():
    prompts: list[str] = []

    def transport(prompt: str) -> str:
        prompts.append(prompt)
        if "Write the final summary" in prompt:
            return '{"summary":"Final wrap of the talk."}'
        if "beta speech about the room" in prompt:
            return '{"summary":"Heard the second stretch."}'
        return '{"summary":"Heard the first stretch."}'

    client = FlashSummaryClient(api_key="test-key", transport=transport, timeout=2)
    summarizer = TalkSummarizer(
        client,
        transcriber=_Lines(["alpha speech about pacing", "beta speech about the room"]),
        interval_seconds=0.5,
    )
    metrics = "level 42 dB, pace 150 WPM, gaze forward, posture steady"
    summarizer.add_audio(_tone(0.6), 16000, metrics)
    _wait_for(lambda: summarizer.text() == "Heard the first stretch.")
    summarizer.add_audio(_tone(0.6), 16000, metrics)
    _wait_for(lambda: summarizer.text() == "Heard the second stretch.")
    final = summarizer.finish(metrics, timeout=2)
    assert final == "Final wrap of the talk."
    assert "alpha speech about pacing" in prompts[0]
    assert "beta speech about the room" in prompts[1]
    assert "Heard the first stretch." in prompts[1]
    assert "alpha speech about pacing" not in prompts[1]
    assert "Write the final summary" in prompts[-1]
    assert "Heard the second stretch." in prompts[-1]
    assert "alpha speech about pacing" not in prompts[-1]
    assert "beta speech about the room" not in prompts[-1]
    assert "alpha speech" not in build_final_prompt("Heard the second stretch.")


def test_add_audio_does_not_wait_and_insight_keeps_its_own_client():
    release = threading.Event()
    started = threading.Event()

    def slow_summary(prompt: str) -> str:
        del prompt
        started.set()
        release.wait(2)
        return '{"summary":"Talk summary."}'

    summary_client = FlashSummaryClient(
        api_key="test-key", transport=slow_summary, timeout=2
    )
    insight = InsightClient(
        api_key="test-key",
        transport=lambda _prompt: '{"bullets":["Steady pace today"]}',
        min_interval=0,
        timeout=2,
    )
    assert summary_client._call_lock is not insight._call_lock
    summary_client._sdk_client()
    insight._sdk_client()
    assert summary_client._sdk is not insight._sdk
    summarizer = TalkSummarizer(
        summary_client,
        transcriber=_Lines(["hello there from the room"]),
        interval_seconds=0.5,
    )
    began = time.perf_counter()
    summarizer.add_audio(
        _tone(0.6),
        16000,
        "level 40 dB, pace 140 WPM, gaze forward, posture steady",
    )
    assert time.perf_counter() - began < 0.3
    assert started.wait(1)
    insight.poll(CoachWindow(avg_wpm=140, seconds=10), now=0)
    _wait_for(lambda: insight.bullets == ["Steady pace today"])
    assert time.perf_counter() - began < 0.8
    assert insight.pending is False
    release.set()
    _wait_for(lambda: summarizer.text() == "Talk summary.")
    insight.close()
    assert parse_summary_text('{"summary":"Kept it short."}') == "Kept it short."


def test_session_report_includes_the_summary_and_not_the_gemma_prompt():
    class Scripted:
        def add_audio(self, samples, sample_rate, metrics) -> None:
            del samples, sample_rate, metrics

        def finish(self, metrics: str = "") -> str:
            del metrics
            return "unique-speech-token"

    gemma_prompts: list[str] = []

    def gemma_transport(prompt: str) -> str:
        gemma_prompts.append(prompt)
        return (
            '{"strengths":["Clear voice.","Steady gaze.","Even pace."],'
            '"improve":["Sit taller.","Pause more.","Lift the middle."]}'
        )

    gemma = GemmaClient(api_key="test-key", transport=gemma_transport, report_timeout=2)
    summary = run_session(
        parse_args(
            [
                "--offline",
                "--frames",
                "2",
                "--fps",
                "10",
                "--width",
                "80",
                "--height",
                "48",
                "--no-voice",
            ]
        ),
        gemma_client=gemma,
        talk_summarizer=Scripted(),
    )
    assert summary.talk_summary == "unique-speech-token"
    assert "## Conversation" in summary.report
    assert "unique-speech-token" in summary.report
    assert gemma_prompts
    assert all("unique-speech-token" not in prompt for prompt in gemma_prompts)
    assert "Source: gemma" in summary.report
