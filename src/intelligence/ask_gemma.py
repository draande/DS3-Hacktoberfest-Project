"""Direct questions for Gemma.

Each ask calls ``GemmaClient.generate`` on its own thread. That path does
not use the Flash Lite client, its lock, its SDK, or its pending flag, so
an ask cannot delay the insight loop. A second ask waits until the current
Gemma call returns.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field

from src.intelligence.gemma_client import (
    CoachWindow,
    GemmaClient,
    _metric_line,
    extract_json_object,
    format_gemma_error,
)

logger = logging.getLogger("wingman.gemma")

_ANSWER_TTL = 45.0
_NOTICE_TTL = 20.0


@dataclass
class AskState:
    """What the sidebar should show for the latest direct question."""

    phase: str = "idle"
    question: str = ""
    lines: list[str] = field(default_factory=list)
    started_at: float | None = None
    updated_at: float = 0.0

    def elapsed(self, now: float) -> int:
        if self.started_at is None:
            return 0
        return max(0, int(float(now) - self.started_at))


def build_ask_prompt(question: str, window: CoachWindow) -> str:
    """Question plus the same compact window the insight call already uses."""
    return (
        "Answer the presenter in 1-3 short lines. Plain text.\n"
        f"Question: {' '.join(question.split())}\n" + _metric_line(window)
    )


def parse_ask_reply(text: str | None) -> list[str]:
    """Keep at most three short lines from JSON or plain text."""
    raw: list[str] = []
    data = extract_json_object(text)
    if isinstance(data, dict):
        for key in ("lines", "answer", "reply", "text"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                raw = [value]
                break
            if isinstance(value, list):
                raw = [str(item) for item in value]
                break
    if not raw and text:
        raw = text.splitlines()
    lines: list[str] = []
    for line in raw:
        cleaned = re.sub(r"^[\s\-\*\d\.\)\]]+", "", str(line)).strip()
        cleaned = " ".join(cleaned.split())
        if not cleaned or cleaned.lower() in {"json", "```"}:
            continue
        lines.append(cleaned[:160])
        if len(lines) == 3:
            break
    return lines


def ask_error_line(exc: BaseException) -> str:
    """One sidebar line for a failed ask. Insights keep their own errors."""
    if isinstance(exc, TimeoutError):
        return "Gemma timed out. Try again."
    text = format_gemma_error(exc)
    code = getattr(exc, "code", None)
    if code is None:
        code = getattr(exc, "status_code", None)
    blob = f"{code or ''} {text}".lower()
    if "429" in blob or "resource_exhausted" in blob:
        return "Gemma is rate limited. Try again shortly."
    if "504" in blob or "deadline" in blob:
        return "Gemma timed out. Try again."
    if "offline" in blob:
        return "Gemma is offline."
    return "Gemma could not answer."


class AskGemma:
    """One Gemma question at a time, on a thread the frame loop does not join."""

    def __init__(self, client: GemmaClient) -> None:
        self.client = client
        self._cond = threading.Condition()
        self._job = 0
        self._queued: tuple | None = None
        self._busy = False
        self._running = False
        self._closed = False
        self._state = AskState()

    def snapshot(self) -> AskState:
        with self._cond:
            return self._copy()

    def submit(
        self,
        question: str,
        window: CoachWindow | None = None,
        audio: bytes | None = None,
        now: float | None = None,
    ) -> AskState:
        """Show THINKING immediately. A newer ask waits until the current call returns."""
        text = " ".join(str(question).split())
        stamp = time.monotonic() if now is None else float(now)
        with self._cond:
            if self._closed or not text:
                return self._copy()
            self._job += 1
            self._queued = (
                self._job,
                text,
                window or CoachWindow(),
                audio,
                stamp,
            )
            self._state = AskState(
                phase="thinking", question=text, started_at=stamp, updated_at=stamp
            )
            start = not self._running
            self._running = True
            self._cond.notify_all()
            state = self._copy()
        if start:
            threading.Thread(
                target=self._worker, name="wingman-gemma-ask", daemon=True
            ).start()
        return state

    def notice(self, text: str, now: float | None = None) -> AskState:
        """Sidebar line that does not call the model."""
        stamp = time.monotonic() if now is None else float(now)
        line = " ".join(str(text).split()) or "Gemma could not answer."
        with self._cond:
            self._job += 1
            self._queued = None
            self._state = AskState(
                phase="notice", lines=[line], started_at=stamp, updated_at=stamp
            )
            self._cond.notify_all()
            return self._copy()

    def ask_blocking(
        self, question: str, window: CoachWindow | None = None
    ) -> AskState:
        """One-shot path. The preview frame loop does not use this."""
        self.submit(question, window, now=time.monotonic())
        return self.wait(self.client.timeout + 1.0)

    def wait(self, timeout: float = 5.0) -> AskState:
        deadline = time.monotonic() + max(0.0, float(timeout))
        while True:
            with self._cond:
                state = self._copy()
                pending = self._queued is not None or self._busy
                if state.phase != "thinking" and self._queued is None:
                    return state
                if not pending and state.phase != "thinking":
                    return state
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return state
            time.sleep(min(0.02, remaining))

    def close(self) -> None:
        """Drop a queued ask. An in-flight call is a daemon thread and is not joined."""
        with self._cond:
            self._closed = True
            self._queued = None
            self._cond.notify_all()

    def _worker(self) -> None:
        while True:
            with self._cond:
                while self._queued is None and not self._closed:
                    self._busy = False
                    self._cond.notify_all()
                    self._cond.wait()
                if self._closed and self._queued is None:
                    self._busy = False
                    self._running = False
                    self._cond.notify_all()
                    return
                job = self._queued
                self._queued = None
                self._busy = True
            assert job is not None
            self._run_job(job)

    def _run_job(self, job: tuple) -> None:
        job_id, question, window, audio, started = job
        prompt = build_ask_prompt(question, window)
        done = threading.Event()
        box: dict = {}

        def invoke() -> None:
            try:
                box["text"] = self.client.generate(prompt, audio)
            except Exception as exc:
                box["error"] = exc
            finally:
                done.set()

        started_call = time.perf_counter()
        threading.Thread(
            target=invoke, name="wingman-gemma-ask-call", daemon=True
        ).start()
        if not done.wait(self.client.timeout):
            self._finish_job(
                job_id,
                question,
                started,
                TimeoutError(f"timed out after {self.client.timeout:.1f}s"),
                time.perf_counter() - started_call,
            )
            done.wait()
            return
        latency = time.perf_counter() - started_call
        if "error" in box:
            self._finish_job(job_id, question, started, box["error"], latency)
            return
        lines = parse_ask_reply(box.get("text"))
        if not lines:
            self._finish_job(
                job_id,
                question,
                started,
                ValueError("empty Gemma reply"),
                latency,
            )
            return
        self._finish_job(job_id, question, started, None, latency, lines)

    def _finish_job(
        self,
        job_id: int,
        question: str,
        started: float,
        error: BaseException | None,
        latency: float,
        lines: list[str] | None = None,
    ) -> None:
        logger.info("Gemma ask took %.2fs", latency)
        stamp = time.monotonic()
        with self._cond:
            if job_id != self._job:
                return
            if error is None:
                self._state = AskState(
                    phase="answer",
                    question=question,
                    lines=list(lines or []),
                    started_at=started,
                    updated_at=stamp,
                )
            else:
                self._state = AskState(
                    phase="error",
                    question=question,
                    lines=[ask_error_line(error)],
                    started_at=started,
                    updated_at=stamp,
                )
            self._cond.notify_all()

    def _copy(self) -> AskState:
        return AskState(
            phase=self._state.phase,
            question=self._state.question,
            lines=list(self._state.lines),
            started_at=self._state.started_at,
            updated_at=self._state.updated_at,
        )


def answer_ttl(phase: str) -> float:
    """How long a finished ask stays up. Thinking stays until the reply lands."""
    if phase == "answer":
        return _ANSWER_TTL
    return _NOTICE_TTL
