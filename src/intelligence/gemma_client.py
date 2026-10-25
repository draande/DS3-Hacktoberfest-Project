"""Google Gemma cue arbitration with a deterministic offline fallback.

When ``GEMMA_API_KEY`` (or ``GOOGLE_API_KEY`` / ``GEMINI_API_KEY``) is unset,
``GemmaClient`` never touches the network. The fallback ranks the same signals
the prompt describes and returns the HUD JSON schema from PLAN.md.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger("wingman.gemma")

DEFAULT_MODEL = "gemma-4-26b-a4b-it"

CUE_KEYS = (
    "priority",
    "cue_type",
    "display_text",
    "icon",
    "duration_ms",
    "audio_chime",
)
API_KEY_ENV_VARS = ("GEMMA_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY")


@dataclass
class CoachMetrics:
    shoulder_tilt_deg: float = 0.0
    neck_slump_deg: float = 0.0
    ear_forward_offset: float = 0.0
    posture_alert: bool = False
    gaze_away_seconds: float = 0.0
    gaze_cue: str | None = None
    gaze_text: str | None = None
    gaze_icon: str | None = None
    volume_db: float = 60.0
    speak_up: bool = False
    wpm: float = 145.0
    rushing: bool = False
    run_on: bool = False
    audience_confused: bool = False
    audience_surprised: bool = False
    blendshapes: dict[str, float] = field(default_factory=dict)


# Short HUD labels expanded so the sidebar still gets a 3–6 word line.
_BULLET_EXPANSIONS = {
    "speak up": "Speak up a bit",
    "straighten posture": "Straighten your posture",
    "slow down": "Slow down your pace",
    "look right": "Look toward the right",
    "look left": "Look toward the left",
    "take a breath": "Take a breath now",
    "make eye contact": "Make eye contact now",
}


def make_cue(
    priority: str,
    cue_type: str,
    display_text: str,
    icon: str,
    duration_ms: int = 3000,
    audio_chime: bool = False,
    bullets=None,
) -> dict:
    return {
        "priority": priority,
        "cue_type": cue_type,
        "display_text": display_text,
        "icon": icon,
        "duration_ms": int(duration_ms),
        "audio_chime": bool(audio_chime),
        "bullets": bullets_for_display(display_text, bullets),
    }


def normalize_bullet(text: str | None) -> str | None:
    """One sidebar line: 3 to 6 words, or None when the text is empty."""
    cleaned = str(text or "").replace("•", " ").strip()
    while cleaned[:1] in {"-", "*"}:
        cleaned = cleaned[1:].strip()
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        return None
    expanded = _BULLET_EXPANSIONS.get(cleaned.lower().rstrip("."))
    if expanded:
        cleaned = expanded
    words = cleaned.split()
    if len(words) > 6:
        words = words[:6]
    if len(words) == 2:
        words = words + ["now"]
    elif len(words) == 1:
        words = ["Focus", "on", words[0].lower()]
    return " ".join(words)


def normalize_bullets(value) -> list[str]:
    """Accept a list or a newline string. Drop empties and keep at most four."""
    if value is None:
        return []
    if isinstance(value, str):
        parts = [line.strip(" \t") for line in value.splitlines()]
    elif isinstance(value, (list, tuple)):
        parts = [str(item) for item in value]
    else:
        return []
    bullets: list[str] = []
    for part in parts:
        bullet = normalize_bullet(part)
        if bullet and bullet not in bullets:
            bullets.append(bullet)
        if len(bullets) == 4:
            break
    return bullets


def bullets_for_display(display_text: str, raw=None) -> list[str]:
    bullets = normalize_bullets(raw)
    if bullets:
        return bullets
    fallback = normalize_bullet(display_text)
    return [fallback] if fallback else []


def resolve_model(config_model: str) -> str:
    """``GEMMA_MODEL`` wins over the YAML model. A blank value does not."""
    override = os.environ.get("GEMMA_MODEL", "").strip()
    return override or config_model


def format_gemma_error(exc: BaseException) -> str:
    """One line with the status code and message, when the exception has them."""
    code = getattr(exc, "code", None)
    if code is None:
        code = getattr(exc, "status_code", None)
    detail = " ".join(str(exc).split()) or exc.__class__.__name__
    if code is not None and str(code) not in detail.split():
        return f"{code} {detail}"
    return detail


def resolve_api_key(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    for name in API_KEY_ENV_VARS:
        value = os.environ.get(name)
        if value:
            return value
    return None


def build_prompt(metrics: CoachMetrics) -> str:
    blendshapes = json.dumps(metrics.blendshapes or {})
    return (
        "You are Wingman, a presentation coach inside smart glasses. "
        "Filter transient noise and only flag persistent issues. "
        "Return ONLY a JSON object with keys priority, cue_type, display_text, "
        "icon, duration_ms, audio_chime, bullets. "
        "priority is one of HIGH, MEDIUM, LOW, NONE. "
        "cue_type is one of POSTURE, GAZE, VOLUME, PACE, AUDIENCE, ENCOURAGE, NONE. "
        "bullets is an array of 1 to 4 strings, most urgent first. "
        "Each bullet is 3 to 6 words, a terse sidebar line, with no paragraphs. "
        "If nothing needs a cue, use priority NONE and an empty bullets array.\n"
        "Metrics:\n"
        f"- shoulder_tilt_deg: {metrics.shoulder_tilt_deg:.1f}\n"
        f"- neck_slump_deg: {metrics.neck_slump_deg:.1f}\n"
        f"- ear_forward_offset: {metrics.ear_forward_offset:.3f}\n"
        f"- posture_alert: {metrics.posture_alert}\n"
        f"- gaze_away_seconds: {metrics.gaze_away_seconds:.1f}\n"
        f"- gaze_cue: {metrics.gaze_cue}\n"
        f"- volume_db: {metrics.volume_db:.1f}\n"
        f"- speak_up: {metrics.speak_up}\n"
        f"- wpm: {metrics.wpm:.0f}\n"
        f"- rushing: {metrics.rushing}\n"
        f"- run_on: {metrics.run_on}\n"
        f"- audience_confused: {metrics.audience_confused}\n"
        f"- audience_surprised: {metrics.audience_surprised}\n"
        f"- blendshapes: {blendshapes}\n"
    )


def parse_cue_json(text: str | None) -> dict | None:
    if not text:
        return None
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or any(key not in data for key in CUE_KEYS):
        return None
    return make_cue(
        priority=str(data["priority"]).upper(),
        cue_type=str(data["cue_type"]).upper(),
        display_text=str(data["display_text"]),
        icon=str(data["icon"]),
        duration_ms=int(data["duration_ms"]),
        audio_chime=bool(data["audio_chime"]),
        bullets=data.get("bullets"),
    )


def offline_cue(metrics: CoachMetrics, duration_ms: int = 3000) -> dict | None:
    """Deterministic ranking used when Gemma is offline or returns junk."""
    if metrics.posture_alert:
        return make_cue(
            "HIGH", "POSTURE", "Straighten Posture", "arrow_up", duration_ms, True
        )
    if metrics.speak_up:
        return make_cue("HIGH", "VOLUME", "Speak up", "volume_up", duration_ms, True)
    if metrics.rushing or metrics.run_on:
        if metrics.rushing:
            text = "Slow down"
        else:
            text = "Take a breath"
        return make_cue("MEDIUM", "PACE", text, "slow_down", duration_ms, False)
    if metrics.gaze_cue and metrics.gaze_text:
        return make_cue(
            "MEDIUM",
            "GAZE",
            metrics.gaze_text,
            metrics.gaze_icon or "target",
            duration_ms,
            False,
        )
    if metrics.audience_confused:
        return make_cue(
            "MEDIUM",
            "AUDIENCE",
            "Reiterate your last point",
            "clarify",
            duration_ms,
            False,
        )
    if metrics.audience_surprised:
        return make_cue(
            "LOW",
            "AUDIENCE",
            "They're with you",
            "interest",
            duration_ms,
            False,
        )
    return None


class GemmaClient:
    """Cue client. ``arbitrate`` is synchronous. ``poll`` never blocks the caller.

    ``poll`` keeps at most one request in flight, waits ``min_interval`` seconds
    between submits, and times out each call. The frame loop reads the latest
    finished result. Until that result exists, and again after an error or
    while a call is still running, ``poll`` returns the offline cue.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        duration_ms: int = 3000,
        transport=None,
        min_interval: float = 2.5,
        timeout: float = 8.0,
        offline: bool = False,
    ) -> None:
        if min_interval < 0:
            raise ValueError("min_interval must be >= 0")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.model = model
        self.api_key = api_key
        self.duration_ms = duration_ms
        self.transport = transport
        self.min_interval = float(min_interval)
        self.timeout = float(timeout)
        self.offline = offline or (api_key is None and transport is None)
        self.used_fallback = True
        self._lock = threading.Lock()
        self._pending = False
        self._live = False
        self._cue: dict | None = None
        self._succeeded = 0
        self._failed = 0
        self._seen_errors: set[str] = set()
        self._last_submit: float | None = None
        self._job_id = 0
        self._settled: set[int] = set()
        self._closed = False

    @classmethod
    def from_env(
        cls,
        model: str,
        duration_ms: int,
        offline: bool = False,
        min_interval: float = 2.5,
        timeout: float = 8.0,
    ) -> GemmaClient:
        api_key = None if offline else resolve_api_key()
        return cls(
            model=resolve_model(model),
            api_key=api_key,
            duration_ms=duration_ms,
            min_interval=min_interval,
            timeout=timeout,
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

    def arbitrate(self, metrics: CoachMetrics) -> dict | None:
        """Blocking call used by tests and by the worker thread."""
        if self.offline or (self.transport is None and not self.api_key):
            self.used_fallback = True
            return offline_cue(metrics, self.duration_ms)
        try:
            cue = self._evaluate(metrics)
        except Exception as exc:
            self._note_failure(exc)
            return offline_cue(metrics, self.duration_ms)
        self._note_success(cue)
        self.used_fallback = False
        return cue

    def poll(self, metrics: CoachMetrics, now: float | None = None) -> dict | None:
        """Return a cue immediately. Never waits on the network."""
        if not self.offline and (self.transport is not None or self.api_key):
            self._maybe_start(metrics, time.monotonic() if now is None else float(now))
        with self._lock:
            pending = self._pending
            live = self._live
            cue = self._cue
        if pending or not live:
            self.used_fallback = True
            return offline_cue(metrics, self.duration_ms)
        self.used_fallback = False
        return cue

    def close(self) -> None:
        """Stop submitting. In-flight calls are daemon threads and are not joined."""
        with self._lock:
            self._closed = True

    def _maybe_start(self, metrics: CoachMetrics, now: float) -> None:
        with self._lock:
            if self._closed or self._pending:
                return
            if (
                self._last_submit is not None
                and now - self._last_submit < self.min_interval
            ):
                return
            self._pending = True
            self._last_submit = now
            self._job_id += 1
            job_id = self._job_id
        threading.Thread(
            target=self._worker,
            args=(metrics, job_id),
            name="wingman-gemma",
            daemon=True,
        ).start()

    def _worker(self, metrics: CoachMetrics, job_id: int) -> None:
        done = threading.Event()
        box: dict = {}

        def invoke() -> None:
            try:
                box["cue"] = self._evaluate(metrics)
            except Exception as exc:
                box["error"] = exc
            finally:
                done.set()

        threading.Thread(target=invoke, name="wingman-gemma-call", daemon=True).start()
        if not done.wait(self.timeout):
            self._note_failure(
                TimeoutError(f"timed out after {self.timeout:.1f}s"), job_id
            )
            done.wait()
        elif "error" in box:
            self._note_failure(box["error"], job_id)
        else:
            self._note_success(box.get("cue"), job_id)
        with self._lock:
            if self._job_id == job_id:
                self._pending = False

    def _evaluate(self, metrics: CoachMetrics) -> dict | None:
        parsed = parse_cue_json(self._fetch(build_prompt(metrics)))
        if parsed is None:
            raise ValueError("unparseable Gemma response")
        if parsed["priority"] == "NONE" or parsed["cue_type"] == "NONE":
            return None
        return parsed

    def _fetch(self, prompt: str) -> str:
        if self.transport is not None:
            return self.transport(prompt)
        return self._call_google(prompt)

    def _note_success(self, cue: dict | None, job_id: int | None = None) -> None:
        with self._lock:
            if not self._claim(job_id):
                return
            self._succeeded += 1
            self._live = True
            self._cue = cue

    def _note_failure(self, exc: BaseException, job_id: int | None = None) -> None:
        line = format_gemma_error(exc)
        with self._lock:
            if not self._claim(job_id):
                return
            self._failed += 1
            self._live = False
            self._cue = None
            self.used_fallback = True
            fresh = line not in self._seen_errors
            if fresh:
                self._seen_errors.add(line)
        if fresh:
            logger.warning("Gemma call failed: %s", line)

    def _claim(self, job_id: int | None) -> bool:
        if job_id is None:
            return True
        if job_id != self._job_id or job_id in self._settled:
            return False
        self._settled.add(job_id)
        return True

    def _call_google(self, prompt: str) -> str:
        from google import genai
        from google.genai import types

        timeout_ms = max(1, int(self.timeout * 1000))
        client = genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(timeout=timeout_ms),
        )
        response = client.models.generate_content(model=self.model, contents=prompt)
        return response.text or ""
