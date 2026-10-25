"""Periodic coaching trends on Gemini Flash Lite.

This client is independent of :class:`src.intelligence.gemma_client.GemmaClient`.
A slow Gemma report or a future Ask Gemma call cannot take its lock, its
in-flight slot, or its HTTP client. Real-time HUD cues still come from the
local analyzers; this loop only refreshes the sidebar trend lines.
"""

from __future__ import annotations

import logging
import threading
import time

from src.intelligence.gemma_client import (
    CoachWindow,
    _is_backoff,
    build_insight_prompt,
    format_gemma_error,
    http_timeout_ms,
    parse_insight_json,
    resolve_api_key,
    resolve_model,
)

logger = logging.getLogger("wingman.insight")

DEFAULT_INSIGHT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_THINKING_LEVEL = "minimal"


def flash_request_config(
    max_output_tokens: int, thinking_level: str = DEFAULT_THINKING_LEVEL
):
    """JSON mode, an explicit minimal thinking level, and a small output cap.

    Gemini 3.5 no longer accepts ``thinking_budget``; ``thinking_level`` is
    the supported control. Gemma must not use this config.
    """
    from google.genai import types

    level = types.ThinkingLevel(str(thinking_level or DEFAULT_THINKING_LEVEL).upper())
    return types.GenerateContentConfig(
        max_output_tokens=int(max_output_tokens),
        response_mime_type="application/json",
        thinking_config=types.ThinkingConfig(thinking_level=level),
    )


class InsightClient:
    """Background trend client. ``poll`` returns immediately.

    At most one Flash call is in flight. A 429 or 504 stretches the wait
    exponentially. The SDK client is created once and reused.
    """

    def __init__(
        self,
        model: str = DEFAULT_INSIGHT_MODEL,
        api_key: str | None = None,
        transport=None,
        min_interval: float = 8.0,
        timeout: float = 10.0,
        max_output_tokens: int = 256,
        thinking_level: str = DEFAULT_THINKING_LEVEL,
        offline: bool = False,
    ) -> None:
        if min_interval < 0:
            raise ValueError("min_interval must be >= 0")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.model = model
        self.api_key = api_key
        self.transport = transport
        self.min_interval = float(min_interval)
        self.timeout = float(timeout)
        self.max_output_tokens = int(max_output_tokens)
        self.thinking_level = thinking_level or DEFAULT_THINKING_LEVEL
        self.offline = offline or (api_key is None and transport is None)
        self.used_fallback = True
        self._lock = threading.Lock()
        self._call_lock = threading.Lock()
        self._pending = False
        self._live = False
        self._bullets: list[str] | None = None
        self._succeeded = 0
        self._failed = 0
        self._seen_errors: set[str] = set()
        self._latencies: list[float] = []
        self._penalty = 0.0
        self._last_submit: float | None = None
        self._job_id = 0
        self._settled: set[int] = set()
        self._closed = False
        self._sdk = None

    @classmethod
    def from_env(
        cls,
        model: str,
        offline: bool = False,
        min_interval: float = 8.0,
        timeout: float = 10.0,
        max_output_tokens: int = 256,
        thinking_level: str = DEFAULT_THINKING_LEVEL,
    ) -> InsightClient:
        api_key = None if offline else resolve_api_key()
        return cls(
            model=resolve_model(model, "INSIGHT_MODEL"),
            api_key=api_key,
            min_interval=min_interval,
            timeout=timeout,
            max_output_tokens=max_output_tokens,
            thinking_level=thinking_level,
            offline=offline,
        )

    @property
    def succeeded(self) -> int:
        with self._lock:
            return self._succeeded

    @property
    def failed(self) -> int:
        with self._lock:
            return self._failed

    @property
    def pending(self) -> bool:
        with self._lock:
            return self._pending

    @property
    def status(self) -> str:
        """``thinking`` while a call is in flight, ``flash`` after a success, else ``offline``."""
        with self._lock:
            if self._pending:
                return "thinking"
            if self._live:
                return "flash"
            return "offline"

    @property
    def bullets(self) -> list[str] | None:
        with self._lock:
            if not self._bullets:
                return None
            return list(self._bullets)

    @property
    def average_latency(self) -> float | None:
        with self._lock:
            if not self._latencies:
                return None
            return sum(self._latencies) / len(self._latencies)

    def insights(self, window: CoachWindow) -> list[str] | None:
        """Blocking trend call. Does not produce a HUD cue."""
        if self.offline or (self.transport is None and not self.api_key):
            self.used_fallback = True
            return None
        started = time.perf_counter()
        try:
            bullets = self._evaluate(window)
        except Exception as exc:
            self._note_failure(exc, latency=time.perf_counter() - started)
            return None
        self._note_success(bullets, latency=time.perf_counter() - started)
        return bullets

    def poll(self, window: CoachWindow, now: float | None = None) -> list[str] | None:
        """Start a trend if one is due. Never waits on this call or on Gemma."""
        if not self.offline and (self.transport is not None or self.api_key):
            self._maybe_start(window, time.monotonic() if now is None else float(now))
        with self._lock:
            pending = self._pending
            live = self._live
            bullets = None if not self._bullets else list(self._bullets)
        self.used_fallback = pending or not live
        if not live:
            return None
        return bullets

    def close(self) -> None:
        """Stop submitting. In-flight calls are daemon threads and are not joined."""
        with self._lock:
            self._closed = True

    def _wait_seconds(self) -> float:
        return self.min_interval + self._penalty

    def _maybe_start(self, window: CoachWindow, now: float) -> None:
        with self._lock:
            if self._closed or self._pending:
                return
            if (
                self._last_submit is not None
                and now - self._last_submit < self._wait_seconds()
            ):
                return
            self._pending = True
            self._last_submit = now
            self._job_id += 1
            job_id = self._job_id
        threading.Thread(
            target=self._worker,
            args=(window, job_id),
            name="wingman-insight",
            daemon=True,
        ).start()

    def _worker(self, window: CoachWindow, job_id: int) -> None:
        done = threading.Event()
        box: dict = {}

        def invoke() -> None:
            try:
                box["bullets"] = self._evaluate(window)
            except Exception as exc:
                box["error"] = exc
            finally:
                done.set()

        started = time.perf_counter()
        threading.Thread(
            target=invoke, name="wingman-insight-call", daemon=True
        ).start()
        if not done.wait(self.timeout):
            self._note_failure(
                TimeoutError(f"timed out after {self.timeout:.1f}s"),
                job_id,
                latency=time.perf_counter() - started,
            )
            done.wait()
        elif "error" in box:
            self._note_failure(
                box["error"], job_id, latency=time.perf_counter() - started
            )
        else:
            self._note_success(
                box.get("bullets"), job_id, latency=time.perf_counter() - started
            )
        with self._lock:
            if self._job_id == job_id:
                self._pending = False

    def _evaluate(self, window: CoachWindow) -> list[str]:
        parsed = parse_insight_json(self._fetch(build_insight_prompt(window)))
        if not parsed:
            raise ValueError("unparseable insight response")
        return parsed

    def _fetch(self, prompt: str) -> str:
        if self.transport is not None:
            return self.transport(prompt)
        with self._call_lock:
            return self._call_google(prompt)

    def _note_success(
        self,
        bullets: list[str] | None,
        job_id: int | None = None,
        latency: float | None = None,
    ) -> None:
        with self._lock:
            if not self._claim(job_id):
                return
            self._succeeded += 1
            self._live = True
            self._penalty = 0.0
            self.used_fallback = False
            if bullets:
                self._bullets = list(bullets)
            if latency is not None:
                self._latencies.append(float(latency))
                logger.info("%s call took %.2fs", self.model, latency)

    def _note_failure(
        self,
        exc: BaseException,
        job_id: int | None = None,
        latency: float | None = None,
    ) -> None:
        line = format_gemma_error(exc)
        retryable = _is_backoff(exc)
        with self._lock:
            if not self._claim(job_id):
                return
            self._failed += 1
            self._live = False
            self.used_fallback = True
            if latency is not None:
                self._latencies.append(float(latency))
                logger.info("%s call took %.2fs", self.model, latency)
            if retryable:
                if self._penalty <= 0:
                    self._penalty = self.min_interval
                else:
                    self._penalty = min(180.0, self._penalty * 2)
            fresh = line not in self._seen_errors
            if fresh:
                self._seen_errors.add(line)
        if fresh:
            logger.warning("%s call failed: %s", self.model, line)

    def _claim(self, job_id: int | None) -> bool:
        if job_id is None:
            return True
        if job_id != self._job_id or job_id in self._settled:
            return False
        self._settled.add(job_id)
        return True

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
