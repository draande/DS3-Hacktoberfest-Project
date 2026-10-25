"""Cooldown, snooze, and active-cue lifetime for HUD alerts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ActiveCue:
    payload: dict
    started_at: float

    def expired(self, now: float) -> bool:
        return (now - self.started_at) * 1000.0 >= float(self.payload["duration_ms"])


class CueFilter:
    """Pass at most one instance of a cue type every ``cooldown_seconds``.

    A second, shorter ``global_gap_seconds`` keeps two different cues from
    appearing in the same instant. Snooze suppresses every cue until it ends.
    """

    def __init__(
        self,
        cooldown_seconds: float = 5.0,
        snooze_seconds: float = 30.0,
        global_gap_seconds: float = 1.0,
    ) -> None:
        self.cooldown_seconds = cooldown_seconds
        self.snooze_seconds = snooze_seconds
        self.global_gap_seconds = global_gap_seconds
        self._last_by_type: dict[str, float] = {}
        self._last_any = -1.0e9
        self.snooze_until = -1.0e9
        self.active: ActiveCue | None = None
        self.fired_at: float | None = None

    def consider(self, cue: dict | None, now: float) -> dict | None:
        self.fired_at = None
        if self.active is not None and self.active.expired(now):
            self.active = None
        if now < self.snooze_until:
            self.active = None
            return None
        if self.active is not None:
            return self.active.payload
        if not cue or str(cue.get("priority", "")).upper() == "NONE":
            return None
        if str(cue.get("cue_type", "")).upper() == "NONE":
            return None
        cue_type = str(cue["cue_type"])
        last_same = self._last_by_type.get(cue_type, -1.0e9)
        if now - last_same < self.cooldown_seconds:
            return None
        if now - self._last_any < self.global_gap_seconds:
            return None
        self._last_by_type[cue_type] = now
        self._last_any = now
        self.active = ActiveCue(payload=cue, started_at=now)
        self.fired_at = now
        return cue

    def dismiss(self, now: float) -> None:
        self.active = None
        self._last_any = now

    def snooze(self, now: float, seconds: float | None = None) -> None:
        duration = self.snooze_seconds if seconds is None else seconds
        self.snooze_until = now + duration
        self.active = None

    def is_snoozed(self, now: float) -> bool:
        return now < self.snooze_until
