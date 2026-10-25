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


@dataclass
class CoachWindow:
    """Compact rolling summary. Gemma sees this, not raw frame history."""

    cue_counts: dict[str, int] = field(default_factory=dict)
    avg_db: float = 0.0
    avg_wpm: float = 0.0
    gaze_away_ratio: float = 0.0
    posture_ratio: float = 0.0
    seconds: float = 0.0


def http_timeout_ms(timeout_seconds: float) -> int:
    """Google rejects deadlines under 10 seconds."""
    return max(10_000, int(float(timeout_seconds) * 1000))


def _metric_line(window: CoachWindow) -> str:
    counts = (
        ",".join(f"{key}:{value}" for key, value in sorted(window.cue_counts.items()))
        or "none"
    )
    return (
        f"sec={window.seconds:.0f} cues={counts} db={window.avg_db:.0f} "
        f"wpm={window.avg_wpm:.0f} gaze={window.gaze_away_ratio:.2f} "
        f"posture={window.posture_ratio:.2f}"
    )


def build_insight_prompt(window: CoachWindow) -> str:
    """Short trend request. No per-frame history and no long instructions."""
    return 'JSON {"bullets":["3-6 words"]}. 1-3 trends.\n' + _metric_line(window)


def build_report_prompt(window: CoachWindow) -> str:
    return (
        'JSON {"strengths":["..."],"improve":["..."]}. 3 and 3. Short lines.\n'
        + _metric_line(window)
    )


def build_prompt(window: CoachWindow) -> str:
    """Alias kept so older callers still build the insight prompt."""
    return build_insight_prompt(window)


def model_accepts_inline_audio(model: str) -> bool:
    """True only for Gemma 4 sizes whose model card lists audio input.

    ``google-genai`` can attach an inline ``audio/wav`` part, but
    ``gemma-4-26b-a4b-it`` and ``gemma-4-31b-it`` reject it with
    "Audio input modality is not enabled for this model". Audio is
    documented for E2B, E4B, and 12B only.
    """
    name = (model or "").lower().replace("_", "-")
    if any(token in name for token in ("26b", "31b", "a4b")):
        return False
    return any(token in name for token in ("e2b", "e4b", "12b"))


def build_ask_prompt(question: str, window: CoachWindow) -> str:
    """The user's question plus the same compact metric line as an insight."""
    question = " ".join(str(question or "").split())
    return (
        'JSON {"lines":["short answer"]}. 1-3 lines. Answer the question.\n'
        f"Q: {question}\n" + _metric_line(window)
    )


def extract_json_object(text: str | None) -> dict | None:
    """Parse the first JSON object, including one wrapped in a ```json fence."""
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match is None:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return data


def parse_cue_json(text: str | None) -> dict | None:
    data = extract_json_object(text)
    if data is None or any(key not in data for key in CUE_KEYS):
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


def parse_insight_json(text: str | None) -> list[str] | None:
    data = extract_json_object(text)
    if data is None or "bullets" not in data:
        return None
    bullets = normalize_bullets(data.get("bullets"))[:3]
    return bullets or None


def _short_answer_lines(value, limit: int = 3, words: int = 14) -> list[str]:
    """Keep a direct reply readable. Do not pad one-word answers."""
    if isinstance(value, str):
        parts = [line.strip() for line in value.splitlines()]
    elif isinstance(value, (list, tuple)):
        parts = [str(item).strip() for item in value]
    else:
        return []
    lines: list[str] = []
    for part in parts:
        cleaned = " ".join(part.replace("•", " ").split())
        while cleaned[:1] in {"-", "*"}:
            cleaned = cleaned[1:].strip()
        if not cleaned:
            continue
        clipped = " ".join(cleaned.split()[:words])
        if clipped not in lines:
            lines.append(clipped)
        if len(lines) == limit:
            break
    return lines


def parse_ask_reply(text: str | None) -> list[str] | None:
    """1–3 short lines from JSON ``lines`` or ``bullets``, or from plain text."""
    data = extract_json_object(text)
    if data is not None:
        raw = data.get("lines")
        if raw is None:
            raw = data.get("bullets")
        return _short_answer_lines(raw) or None
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    if not cleaned or cleaned.startswith("{"):
        return None
    return _short_answer_lines(cleaned) or None


def _report_lines(value, limit: int = 3) -> list[str]:
    if isinstance(value, str):
        parts = [line.strip() for line in value.splitlines()]
    elif isinstance(value, (list, tuple)):
        parts = [str(item).strip() for item in value]
    else:
        return []
    lines: list[str] = []
    for part in parts:
        words = " ".join(part.replace("•", " ").split())
        if not words:
            continue
        clipped = " ".join(words.split()[:16])
        if clipped not in lines:
            lines.append(clipped)
        if len(lines) == limit:
            break
    return lines


def parse_report_json(text: str | None) -> tuple[list[str], list[str]] | None:
    data = extract_json_object(text)
    if data is None:
        return None
    strengths = _report_lines(data.get("strengths"))
    improve = _report_lines(data.get("improve") or data.get("improvements"))
    if len(strengths) < 3 or len(improve) < 3:
        return None
    return strengths[:3], improve[:3]


def active_cues(metrics: CoachMetrics, duration_ms: int = 3000) -> list[dict]:
    """Every condition that is true right now, in display order.

    Real-time cues come from this list. Gemma does not add or replace them.
    """
    cues: list[dict] = []
    if metrics.posture_alert:
        cues.append(
            make_cue(
                "HIGH", "POSTURE", "Straighten Posture", "arrow_up", duration_ms, True
            )
        )
    if metrics.speak_up:
        cues.append(
            make_cue("HIGH", "VOLUME", "Speak up", "volume_up", duration_ms, True)
        )
    if metrics.rushing or metrics.run_on:
        text = "Slow down" if metrics.rushing else "Take a breath"
        cues.append(make_cue("MEDIUM", "PACE", text, "slow_down", duration_ms, False))
    if metrics.gaze_cue and metrics.gaze_text:
        cues.append(
            make_cue(
                "MEDIUM",
                "GAZE",
                metrics.gaze_text,
                metrics.gaze_icon or "target",
                duration_ms,
                False,
            )
        )
    if metrics.audience_confused:
        cues.append(
            make_cue(
                "MEDIUM",
                "AUDIENCE",
                "Reiterate your last point",
                "clarify",
                duration_ms,
                False,
            )
        )
    elif metrics.audience_surprised:
        cues.append(
            make_cue(
                "LOW",
                "AUDIENCE",
                "They're with you",
                "interest",
                duration_ms,
                False,
            )
        )
    return cues


def offline_cue(metrics: CoachMetrics, duration_ms: int = 3000) -> dict | None:
    """Highest-priority local cue. Prefer :func:`active_cues` when several apply."""
    cues = active_cues(metrics, duration_ms)
    return cues[0] if cues else None


def offline_insights(window: CoachWindow) -> list[str]:
    """Trend lines used when Gemma has not returned any."""
    lines: list[str] = []
    if window.posture_ratio >= 0.35:
        lines.append("Posture slipping often")
    elif window.seconds >= 10 and window.posture_ratio <= 0.05:
        lines.append("Posture holding steady")
    if window.seconds >= 5 and window.avg_db < 42:
        lines.append("Volume stays too quiet")
    elif window.avg_db >= 50:
        lines.append("Volume is carrying well")
    if window.avg_wpm > 180:
        lines.append("Pace running ahead now")
    elif window.seconds >= 8 and 0 < window.avg_wpm < 130:
        lines.append("Pace a bit slow")
    elif 130 <= window.avg_wpm <= 170:
        lines.append("Pace steadier now")
    if window.gaze_away_ratio >= 0.35:
        lines.append("Gaze drifting off often")
    if not lines:
        lines.append("Delivery looks steady")
    return normalize_bullets(lines)[:3]


def deterministic_report(window: CoachWindow) -> tuple[list[str], list[str]]:
    """Three strengths and three improvements with no network call."""
    strengths: list[str] = []
    improve: list[str] = []
    if window.avg_db >= 42:
        strengths.append("Your volume stayed easy to hear.")
    else:
        improve.append("Bring the volume up above a quiet room.")
    if window.posture_ratio < 0.25:
        strengths.append("Posture stayed close to your baseline.")
    else:
        improve.append("Reset your posture when the cue repeats.")
    if window.gaze_away_ratio < 0.25:
        strengths.append("You kept looking toward the room.")
    else:
        improve.append("Sweep the room when your gaze sticks.")
    if 120 <= window.avg_wpm <= 180:
        strengths.append("Pace stayed in a comfortable range.")
    elif window.avg_wpm > 180:
        improve.append("Leave short pauses so the pace can settle.")
    else:
        improve.append("Pick the pace up a little between points.")
    strength_fillers = (
        "You finished the session.",
        "The live cues stayed on the device.",
        "The level meter stayed in view.",
    )
    improve_fillers = (
        "Glance at the level meter while you talk.",
        "Use the opening seconds to sit how you mean to.",
        "Let one cue land before the next correction.",
    )
    for line in strength_fillers:
        if len(strengths) >= 3:
            break
        strengths.append(line)
    for line in improve_fillers:
        if len(improve) >= 3:
            break
        improve.append(line)
    return strengths[:3], improve[:3]


def render_report_markdown(
    window: CoachWindow,
    strengths: list[str],
    improve: list[str],
    source: str,
) -> str:
    counts = (
        ", ".join(f"{key} {value}" for key, value in sorted(window.cue_counts.items()))
        or "none"
    )
    strength_lines = "\n".join(f"- {line}" for line in strengths)
    improve_lines = "\n".join(f"- {line}" for line in improve)
    return (
        "# Wingman session\n\n"
        f"Source: {source}\n\n"
        "## Strengths\n\n"
        f"{strength_lines}\n\n"
        "## Work on\n\n"
        f"{improve_lines}\n\n"
        "## Numbers\n\n"
        f"- Window: {window.seconds:.0f}s\n"
        f"- Average level: {window.avg_db:.0f} dB\n"
        f"- Average pace: {window.avg_wpm:.0f} WPM\n"
        f"- Gaze away: {window.gaze_away_ratio:.0%}\n"
        f"- Posture alert: {window.posture_ratio:.0%}\n"
        f"- Cues: {counts}\n"
    )


def _is_backoff(exc: BaseException) -> bool:
    code = getattr(exc, "code", None)
    if code is None:
        code = getattr(exc, "status_code", None)
    if code in (429, 504):
        return True
    text = format_gemma_error(exc).lower()
    return "429" in text or "504" in text or "deadline_exceeded" in text


def ask_error_line(exc: BaseException) -> str:
    """One sidebar line. The log still keeps the raw status text."""
    text = format_gemma_error(exc).lower()
    code = getattr(exc, "code", None)
    if code is None:
        code = getattr(exc, "status_code", None)
    if code == 429 or "429" in text:
        return "Gemma is busy. Try again."
    if code == 504 or "504" in text or "deadline_exceeded" in text:
        return "Gemma timed out. Try again."
    if isinstance(exc, TimeoutError) or "timed out" in text:
        return "Timed out. Try again."
    return "Gemma could not answer."


@dataclass
class AskState:
    """What the sidebar shows for a direct question. Idle until the user asks."""

    question: str = ""
    phase: str = "idle"
    lines: tuple[str, ...] = ()
    started: float = 0.0


class GemmaClient:
    """Background insight client. Real-time cues never come from here.

    ``poll`` keeps at most one request in flight and returns immediately.
    A 504 or 429 stretches the wait exponentially. ``report`` is the one
    blocking call, used after the session ends.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        duration_ms: int = 3000,
        transport=None,
        min_interval: float = 18.0,
        timeout: float = 45.0,
        report_timeout: float = 90.0,
        max_output_tokens: int = 2048,
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
        self.report_timeout = float(report_timeout)
        self.max_output_tokens = int(max_output_tokens)
        self.offline = offline or (api_key is None and transport is None)
        self.used_fallback = True
        self._lock = threading.Lock()
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
        self._skip_next_insight = False
        self._ask = AskState()

    @classmethod
    def from_env(
        cls,
        model: str,
        duration_ms: int,
        offline: bool = False,
        min_interval: float = 18.0,
        timeout: float = 45.0,
        report_timeout: float = 90.0,
        max_output_tokens: int = 2048,
    ) -> GemmaClient:
        api_key = None if offline else resolve_api_key()
        return cls(
            model=resolve_model(model),
            api_key=api_key,
            duration_ms=duration_ms,
            min_interval=min_interval,
            timeout=timeout,
            report_timeout=report_timeout,
            max_output_tokens=max_output_tokens,
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
        """``thinking`` while a call is in flight, ``gemma`` after a success, else ``offline``."""
        with self._lock:
            if self._pending:
                return "thinking"
            if self._live:
                return "gemma"
            return "offline"

    @property
    def bullets(self) -> list[str] | None:
        with self._lock:
            if not self._bullets:
                return None
            return list(self._bullets)

    @property
    def ask_state(self) -> AskState:
        with self._lock:
            state = self._ask
            return AskState(
                state.question, state.phase, tuple(state.lines), state.started
            )

    @property
    def average_latency(self) -> float | None:
        with self._lock:
            if not self._latencies:
                return None
            return sum(self._latencies) / len(self._latencies)

    def insights(self, window: CoachWindow) -> list[str] | None:
        """Blocking insight call. Does not produce a HUD cue."""
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
        """Start an insight if one is due. Never waits, never returns a live cue."""
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

    def ask(
        self,
        question: str,
        window: CoachWindow,
        now: float | None = None,
        audio: bytes | None = None,
    ) -> None:
        """Start a direct question and return immediately.

        Drops an in-flight insight or ask so only the newest result is applied,
        and skips the next scheduled insight. The frame loop must not join this.
        """
        question = " ".join(str(question or "").split())
        if audio is not None and not model_accepts_inline_audio(self.model):
            audio = None
        if not question and not audio:
            return
        stamp = time.monotonic()
        when = stamp if now is None else float(now)
        shown = question or "Voice question"
        if self.offline or (self.transport is None and not self.api_key):
            with self._lock:
                self._skip_next_insight = True
                self._ask = AskState(
                    question=shown,
                    phase="error",
                    lines=("Gemma is offline.",),
                    started=stamp,
                )
            return
        prompt = build_ask_prompt(
            question or "What did I just ask? Answer in 1-3 short lines.",
            window,
        )
        with self._lock:
            if self._closed:
                return
            self._job_id += 1
            job_id = self._job_id
            self._pending = True
            self._last_submit = when
            self._skip_next_insight = True
            self._ask = AskState(question=shown, phase="thinking", started=stamp)
        threading.Thread(
            target=self._ask_worker,
            args=(prompt, audio, job_id),
            name="wingman-gemma-ask",
            daemon=True,
        ).start()

    def note_local(self, lines: list[str], question: str = "") -> None:
        """Sidebar message that does not call Gemma. Typed asks stay available."""
        shown = tuple(line for line in lines if str(line).strip())
        if not shown:
            return
        with self._lock:
            self._ask = AskState(
                question=" ".join(question.split()),
                phase="error",
                lines=shown[:3],
                started=time.monotonic(),
            )

    def ask_blocking(
        self, question: str, window: CoachWindow | None = None
    ) -> tuple[list[str], bool]:
        """One-shot ask for ``--ask``. Waits, but the frame loop never calls this."""
        window = window or CoachWindow()
        question = " ".join(str(question or "").split())
        if not question:
            return ["Type a question for Gemma."], False
        if self.offline or (self.transport is None and not self.api_key):
            return ["Gemma is offline."], False
        prompt = build_ask_prompt(question, window)
        done = threading.Event()
        box: dict = {}

        def invoke() -> None:
            try:
                box["text"] = self._fetch(prompt)
            except Exception as exc:
                box["error"] = exc
            finally:
                done.set()

        started = time.perf_counter()
        threading.Thread(
            target=invoke, name="wingman-gemma-ask-once", daemon=True
        ).start()
        if not done.wait(self.timeout):
            exc = TimeoutError(f"timed out after {self.timeout:.1f}s")
            self._note_failure(exc, latency=time.perf_counter() - started)
            return [ask_error_line(exc)], False
        latency = time.perf_counter() - started
        if "error" in box:
            self._note_failure(box["error"], latency=latency)
            return [ask_error_line(box["error"])], False
        lines = parse_ask_reply(box.get("text"))
        if not lines:
            exc = ValueError("unparseable Gemma response")
            self._note_failure(exc, latency=latency)
            return [ask_error_line(exc)], False
        self._note_success(None, latency=latency)
        return lines, True

    def report(self, window: CoachWindow) -> tuple[str, list[str], list[str]]:
        """One end-of-session call. Falls back locally if Gemma does not answer.

        ``report_timeout`` of 0 skips the network and writes the local summary.
        """
        local = deterministic_report(window)
        if (
            self.report_timeout <= 0
            or self.offline
            or (self.transport is None and not self.api_key)
        ):
            return ("offline", local[0], local[1])
        done = threading.Event()
        box: dict = {}

        def invoke() -> None:
            try:
                box["text"] = self._fetch(build_report_prompt(window))
            except Exception as exc:
                box["error"] = exc
            finally:
                done.set()

        started = time.perf_counter()
        threading.Thread(
            target=invoke, name="wingman-gemma-report", daemon=True
        ).start()
        if not done.wait(self.report_timeout):
            self._note_failure(
                TimeoutError(f"timed out after {self.report_timeout:.1f}s"),
                latency=time.perf_counter() - started,
            )
            return ("offline", local[0], local[1])
        latency = time.perf_counter() - started
        if "error" in box:
            self._note_failure(box["error"], latency=latency)
            return ("offline", local[0], local[1])
        parsed = parse_report_json(box.get("text"))
        if parsed is None:
            self._note_failure(
                ValueError("unparseable Gemma response"), latency=latency
            )
            return ("offline", local[0], local[1])
        self._note_success(self.bullets, latency=latency)
        return ("gemma", parsed[0], parsed[1])

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
            if self._skip_next_insight:
                self._skip_next_insight = False
                self._last_submit = now
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
            name="wingman-gemma",
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
        threading.Thread(target=invoke, name="wingman-gemma-call", daemon=True).start()
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

    def _ask_worker(self, prompt: str, audio: bytes | None, job_id: int) -> None:
        done = threading.Event()
        box: dict = {}

        def invoke() -> None:
            try:
                box["text"] = self._fetch(prompt, audio)
            except Exception as exc:
                box["error"] = exc
            finally:
                done.set()

        started = time.perf_counter()
        threading.Thread(
            target=invoke, name="wingman-gemma-ask-call", daemon=True
        ).start()
        if not done.wait(self.timeout):
            self._finish_ask(
                job_id,
                error=TimeoutError(f"timed out after {self.timeout:.1f}s"),
                latency=time.perf_counter() - started,
            )
            return
        latency = time.perf_counter() - started
        if "error" in box:
            self._finish_ask(job_id, error=box["error"], latency=latency)
            return
        lines = parse_ask_reply(box.get("text"))
        if not lines:
            self._finish_ask(
                job_id,
                error=ValueError("unparseable Gemma response"),
                latency=latency,
            )
            return
        self._finish_ask(job_id, lines=lines, latency=latency)

    def _finish_ask(
        self,
        job_id: int,
        lines: list[str] | None = None,
        error: BaseException | None = None,
        latency: float | None = None,
    ) -> None:
        line = None if error is None else format_gemma_error(error)
        retryable = error is not None and _is_backoff(error)
        fresh = False
        with self._lock:
            if not self._claim(job_id):
                return
            if latency is not None:
                self._latencies.append(float(latency))
                logger.info("Gemma call took %.2fs", latency)
            if error is None:
                self._succeeded += 1
                self._live = True
                self._penalty = 0.0
                self.used_fallback = False
                self._ask = AskState(
                    question=self._ask.question,
                    phase="reply",
                    lines=tuple(lines or ()),
                    started=self._ask.started,
                )
            else:
                self._failed += 1
                if retryable:
                    if self._penalty <= 0:
                        self._penalty = self.min_interval
                    else:
                        self._penalty = min(180.0, self._penalty * 2)
                fresh = line not in self._seen_errors
                if fresh and line is not None:
                    self._seen_errors.add(line)
                self._ask = AskState(
                    question=self._ask.question,
                    phase="error",
                    lines=(ask_error_line(error),),
                    started=self._ask.started,
                )
            if self._job_id == job_id:
                self._pending = False
        if fresh and line:
            logger.warning("Gemma call failed: %s", line)

    def _evaluate(self, window: CoachWindow) -> list[str]:
        parsed = parse_insight_json(self._fetch(build_insight_prompt(window)))
        if not parsed:
            raise ValueError("unparseable Gemma response")
        return parsed

    def _fetch(self, prompt: str, audio: bytes | None = None) -> str:
        if self.transport is not None:
            if audio is None:
                return self.transport(prompt)
            try:
                return self.transport(prompt, audio)
            except TypeError as exc:
                message = str(exc)
                if "argument" not in message and "positional" not in message:
                    raise
                return self.transport(prompt)
        return self._call_google(prompt, audio)

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
                logger.info("Gemma call took %.2fs", latency)

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
                logger.info("Gemma call took %.2fs", latency)
            if retryable:
                if self._penalty <= 0:
                    self._penalty = self.min_interval
                else:
                    self._penalty = min(180.0, self._penalty * 2)
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

    def _call_google(self, prompt: str, audio: bytes | None = None) -> str:
        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(timeout=http_timeout_ms(self.timeout)),
        )
        contents: str | list = prompt
        if audio and model_accepts_inline_audio(self.model):
            contents = [
                prompt,
                types.Part.from_bytes(data=audio, mime_type="audio/wav"),
            ]
        response = client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                max_output_tokens=self.max_output_tokens
            ),
        )
        return response.text or ""
