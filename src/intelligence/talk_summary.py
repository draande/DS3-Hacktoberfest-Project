"""Running conversation summary on Flash Lite.

Each update sends only the previous summary plus the new stretch of speech
and a one-line coaching snapshot. The full transcript is never sent again.
The worker is a daemon thread with its own SDK client and lock, so it does
not take the insight client's in-flight slot and it never calls Gemma.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from src.intelligence.gemma_client import (
    extract_json_object,
    http_timeout_ms,
    resolve_api_key,
    resolve_model,
)
from src.intelligence.insight_client import (
    DEFAULT_INSIGHT_MODEL,
    DEFAULT_THINKING_LEVEL,
    flash_request_config,
)

logger = logging.getLogger("wingman.summary")

MIN_CHUNK_SECONDS = 0.5
QUIET_RMS = 0.01


def stretch_metrics(db: float, wpm: float, gaze_away: bool, posture_alert: bool) -> str:
    """One line for this stretch. Not a history of frames."""
    gaze = "away" if gaze_away else "forward"
    posture = "alert" if posture_alert else "steady"
    return (
        f"level {float(db):.0f} dB, pace {float(wpm):.0f} WPM, "
        f"gaze {gaze}, posture {posture}"
    )


def build_fold_prompt(previous: str, speech: str, metrics: str) -> str:
    """Previous summary plus the new stretch only."""
    prior = previous.strip() or "(none yet)"
    words = speech.strip() or "(no words caught)"
    coaching = metrics.strip() or "(none)"
    return (
        'JSON {"summary":"2-4 sentences"}.\n'
        "Update the running summary of this talk. Use only the previous "
        "summary and the new stretch. Do not invent earlier events.\n\n"
        f"Previous summary:\n{prior}\n\n"
        f"New speech:\n{words}\n\n"
        f"Coaching for this stretch:\n{coaching}\n"
    )


def build_final_prompt(summary: str) -> str:
    """Polish the running summary. The transcript is not included."""
    return (
        'JSON {"summary":"2-4 sentences"}.\n'
        "Write the final summary of this talk from the running summary "
        "below. Do not add events that are not already in it.\n\n"
        f"Running summary:\n{summary.strip()}\n"
    )


def parse_summary_text(raw: str) -> str:
    parsed = extract_json_object(raw)
    if parsed and str(parsed.get("summary") or "").strip():
        return " ".join(str(parsed["summary"]).split())
    text = " ".join(str(raw or "").split())
    if text.startswith("{") or text.startswith("["):
        return ""
    return text


def local_update(previous: str, speech: str, metrics: str) -> str:
    """Short offline line. Capped so a missing key cannot grow a transcript."""
    bits = [part.strip() for part in (speech, metrics) if part and part.strip()]
    fresh = " ".join(bits)
    if not fresh:
        return previous.strip()
    if len(fresh) > 280:
        fresh = fresh[:277] + "..."
    combined = f"{previous.strip()} {fresh}".strip() if previous.strip() else fresh
    if len(combined) > 700:
        combined = combined[:697] + "..."
    return combined


class FlashSummaryClient:
    """Blocking Flash Lite completion. Not InsightClient and not GemmaClient.

    ``transport`` replaces the network in tests. The SDK client is created
    lazily and is never the insight client's client.
    """

    def __init__(
        self,
        model: str = DEFAULT_INSIGHT_MODEL,
        api_key: str | None = None,
        transport=None,
        timeout: float = 10.0,
        max_output_tokens: int = 512,
        thinking_level: str = DEFAULT_THINKING_LEVEL,
        offline: bool = False,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.model = model
        self.api_key = api_key
        self.transport = transport
        self.timeout = float(timeout)
        self.max_output_tokens = int(max_output_tokens)
        self.thinking_level = thinking_level or DEFAULT_THINKING_LEVEL
        self.offline = offline or (api_key is None and transport is None)
        self._lock = threading.Lock()
        self._call_lock = threading.Lock()
        self._sdk = None

    @classmethod
    def from_env(
        cls,
        model: str,
        offline: bool = False,
        timeout: float = 10.0,
        max_output_tokens: int = 512,
        thinking_level: str = DEFAULT_THINKING_LEVEL,
    ) -> FlashSummaryClient:
        api_key = None if offline else resolve_api_key()
        return cls(
            model=resolve_model(model, "INSIGHT_MODEL"),
            api_key=api_key,
            timeout=timeout,
            max_output_tokens=max_output_tokens,
            thinking_level=thinking_level,
            offline=offline,
        )

    def complete(self, prompt: str) -> str:
        """One Flash call. Call this off the frame-loop thread."""
        if self.offline or (self.transport is None and not self.api_key):
            raise RuntimeError("Flash summary is offline")
        if self.transport is not None:
            return str(self.transport(prompt) or "")
        with self._call_lock:
            return self._call_google(prompt)

    def _sdk_client(self):
        if self._sdk is not None:
            return self._sdk
        from google import genai
        from google.genai import types

        self._sdk = genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(timeout=http_timeout_ms(self.timeout)),
        )
        return self._sdk

    def _call_google(self, prompt: str) -> str:
        response = self._sdk_client().models.generate_content(
            model=self.model,
            contents=prompt,
            config=flash_request_config(self.max_output_tokens, self.thinking_level),
        )
        return response.text or ""


class TalkSummarizer:
    """Fold each new stretch into the running summary.

    ``add_audio`` only copies samples. Transcription and Flash run on a
    daemon thread. ``finish`` waits for that thread, folds whatever audio
    is left, then asks for a final wording from the running summary alone.
    """

    def __init__(
        self,
        client: FlashSummaryClient | None = None,
        transcriber=None,
        load_transcriber=None,
        interval_seconds: float = 20.0,
        min_rms: float = QUIET_RMS,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.client = client
        self._transcriber = transcriber
        self._load_transcriber = load_transcriber
        self._transcriber_loaded = transcriber is not None or load_transcriber is None
        self.interval_seconds = float(interval_seconds)
        self.min_rms = float(min_rms)
        self._chunks: list[np.ndarray] = []
        self._count = 0
        self._rate = 16000
        self._metrics = ""
        self._summary = ""
        self._lock = threading.Lock()
        self._pending = False
        self._closed = False
        self._warned = False

    def text(self) -> str:
        with self._lock:
            return self._summary

    def add_audio(
        self,
        samples: np.ndarray,
        sample_rate: int,
        metrics: str,
    ) -> None:
        """Queue audio and return. Does not transcribe and does not call Flash."""
        data = np.asarray(samples, dtype=np.float32).reshape(-1)
        if data.size == 0 or int(sample_rate) <= 0:
            return
        launch = False
        taken = None
        with self._lock:
            if self._closed:
                return
            self._chunks.append(data.copy())
            self._count += int(data.size)
            self._rate = int(sample_rate)
            if metrics:
                self._metrics = str(metrics)
            ready = self._count >= int(self.interval_seconds * self._rate)
            if ready and not self._pending:
                taken = self._take()
                if taken is not None:
                    self._pending = True
                    launch = True
        if launch and taken is not None:
            self._start(taken)

    def finish(self, metrics: str = "", timeout: float = 12.0) -> str:
        """Final summary. Safe to call after the frame loop has stopped."""
        with self._lock:
            self._closed = True
            if metrics:
                self._metrics = str(metrics)
            leftover = self._take() if self._count else None
        deadline = time.monotonic() + max(0.0, float(timeout))
        while time.monotonic() < deadline:
            with self._lock:
                pending = self._pending
            if not pending:
                break
            time.sleep(0.02)
        if leftover is not None and leftover[0].size:
            self._fold(*leftover)
        summary = self.text()
        if summary and self._can_call():
            try:
                polished = parse_summary_text(
                    self.client.complete(build_final_prompt(summary))
                )
            except Exception as exc:
                self._warn(exc)
                polished = ""
            if polished:
                summary = polished
                with self._lock:
                    self._summary = polished
        if not summary:
            summary = local_update("", "", self._metrics or metrics)
        if not summary:
            summary = "No speech was captured during this session."
        return summary

    def _can_call(self) -> bool:
        client = self.client
        return client is not None and not client.offline

    def _start(self, taken: tuple[np.ndarray, int, str]) -> None:
        threading.Thread(
            target=self._worker,
            args=(taken,),
            name="wingman-summary",
            daemon=True,
        ).start()

    def _worker(self, taken: tuple[np.ndarray, int, str]) -> None:
        try:
            self._fold(*taken)
        finally:
            follow = None
            with self._lock:
                self._pending = False
                if not self._closed and self._count >= int(
                    self.interval_seconds * max(1, self._rate)
                ):
                    follow = self._take()
                    if follow is not None:
                        self._pending = True
            if follow is not None:
                self._worker(follow)

    def _take(self) -> tuple[np.ndarray, int, str] | None:
        if not self._chunks:
            return None
        audio = (
            self._chunks[0] if len(self._chunks) == 1 else np.concatenate(self._chunks)
        )
        self._chunks = []
        self._count = 0
        return audio, self._rate, self._metrics

    def _fold(self, samples: np.ndarray, rate: int, metrics: str) -> None:
        if samples.size < int(MIN_CHUNK_SECONDS * max(1, rate)):
            return
        speech = self._transcribe(samples, rate)
        if not speech and self._rms(samples) < self.min_rms:
            return
        previous = self.text()
        if not self._can_call():
            updated = local_update(previous, speech, metrics)
        else:
            try:
                raw = self.client.complete(build_fold_prompt(previous, speech, metrics))
                updated = parse_summary_text(raw) or local_update(
                    previous, speech, metrics
                )
            except Exception as exc:
                self._warn(exc)
                updated = local_update(previous, speech, metrics)
        if updated:
            with self._lock:
                self._summary = updated

    def _transcribe(self, samples: np.ndarray, rate: int) -> str:
        transcriber = self._ensure_transcriber()
        if transcriber is None:
            if self._rms(samples) >= self.min_rms:
                return "(speech was not transcribed)"
            return ""
        try:
            return " ".join(str(transcriber.transcribe(samples, rate)).split())
        except Exception as exc:
            self._warn(exc)
            return ""

    def _ensure_transcriber(self):
        if self._transcriber_loaded:
            return self._transcriber
        self._transcriber_loaded = True
        loader = self._load_transcriber
        if loader is None:
            return None
        try:
            self._transcriber = loader()
        except Exception as exc:
            self._warn(exc)
            self._transcriber = None
        return self._transcriber

    def _warn(self, exc: BaseException) -> None:
        with self._lock:
            if self._warned:
                return
            self._warned = True
        logger.warning("Talk summary update failed: %s", exc)

    @staticmethod
    def _rms(samples: np.ndarray) -> float:
        if samples.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
