"""Google Gemma cue arbitration with a deterministic offline fallback.

When ``GEMMA_API_KEY`` (or ``GOOGLE_API_KEY`` / ``GEMINI_API_KEY``) is unset,
``GemmaClient`` never touches the network. The fallback ranks the same signals
the prompt describes and returns the HUD JSON schema from PLAN.md.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

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


def make_cue(
    priority: str,
    cue_type: str,
    display_text: str,
    icon: str,
    duration_ms: int = 3000,
    audio_chime: bool = False,
) -> dict:
    return {
        "priority": priority,
        "cue_type": cue_type,
        "display_text": display_text,
        "icon": icon,
        "duration_ms": int(duration_ms),
        "audio_chime": bool(audio_chime),
    }


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
        "icon, duration_ms, audio_chime. "
        "priority is one of HIGH, MEDIUM, LOW, NONE. "
        "cue_type is one of POSTURE, GAZE, VOLUME, PACE, AUDIENCE, ENCOURAGE, NONE. "
        "If nothing needs a cue, use priority NONE.\n"
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
    def __init__(
        self,
        model: str = "gemma-3-12b-it",
        api_key: str | None = None,
        duration_ms: int = 3000,
        transport=None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.duration_ms = duration_ms
        self.transport = transport
        self.used_fallback = api_key is None and transport is None

    @classmethod
    def from_env(
        cls, model: str, duration_ms: int, offline: bool = False
    ) -> GemmaClient:
        api_key = None if offline else resolve_api_key()
        return cls(model=model, api_key=api_key, duration_ms=duration_ms)

    def arbitrate(self, metrics: CoachMetrics) -> dict | None:
        if self.transport is not None:
            parsed = parse_cue_json(self.transport(build_prompt(metrics)))
            if parsed is None:
                self.used_fallback = True
                return offline_cue(metrics, self.duration_ms)
            if parsed["priority"] == "NONE" or parsed["cue_type"] == "NONE":
                return None
            return parsed
        if not self.api_key:
            self.used_fallback = True
            return offline_cue(metrics, self.duration_ms)
        try:
            parsed = parse_cue_json(self._call_google(build_prompt(metrics)))
        except Exception:
            self.used_fallback = True
            return offline_cue(metrics, self.duration_ms)
        if parsed is None:
            self.used_fallback = True
            return offline_cue(metrics, self.duration_ms)
        if parsed["priority"] == "NONE" or parsed["cue_type"] == "NONE":
            return None
        self.used_fallback = False
        return parsed

    def _call_google(self, prompt: str) -> str:
        from google import genai

        client = genai.Client(api_key=self.api_key)
        response = client.models.generate_content(model=self.model, contents=prompt)
        return response.text or ""
