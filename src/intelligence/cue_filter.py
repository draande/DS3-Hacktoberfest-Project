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
        self._last_fired_type: str | None = None
        self.snooze_until = -1.0e9
        self.active: ActiveCue | None = None
        self.fired_at: float | None = None

    def consider(self, cue: dict | None, now: float) -> dict | None:
        return self.consider_many([] if not cue else [cue], now)

    def consider_many(self, cues: list[dict] | None, now: float) -> dict | None:
        """Show one eligible cue, and let a type that just fired yield.

        Several conditions can be true at once. The type that was just on
        screen waits its cooldown and, when another type is also ready, steps
        aside so posture cannot occupy every slot.
        """
        self.fired_at = None
        if self.active is not None and self.active.expired(now):
            self.active = None
        if now < self.snooze_until:
            self.active = None
            return None
        if self.active is not None:
            return self.active.payload
        eligible: list[dict] = []
        for cue in cues or []:
            if not cue or str(cue.get("priority", "")).upper() == "NONE":
                continue
            cue_type = str(cue.get("cue_type", "")).upper()
            if not cue_type or cue_type == "NONE":
                continue
            last_same = self._last_by_type.get(cue_type, -1.0e9)
            if now - last_same < self.cooldown_seconds:
                continue
            eligible.append(cue)
        if not eligible:
            return None
        if now - self._last_any < self.global_gap_seconds:
            return None
        chosen = eligible[0]
        if self._last_fired_type is not None and len(eligible) > 1:
            others = [
                cue
                for cue in eligible
                if str(cue.get("cue_type", "")).upper() != self._last_fired_type
            ]
            if others:
                chosen = others[0]
        cue_type = str(chosen["cue_type"]).upper()
        self._last_by_type[cue_type] = now
        self._last_any = now
        self._last_fired_type = cue_type
        self.active = ActiveCue(payload=chosen, started_at=now)
        self.fired_at = now
        return chosen

    def dismiss(self, now: float) -> None:
        self.active = None
        self._last_any = now

    def snooze(self, now: float, seconds: float | None = None) -> None:
        duration = self.snooze_seconds if seconds is None else seconds
        self.snooze_until = now + duration
        self.active = None

    def is_snoozed(self, now: float) -> bool:
        return now < self.snooze_until
