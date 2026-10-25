"""Connect gestures and hotkeys to the cue filter and HUD theme."""

from __future__ import annotations

from src.audio.volume_analyzer import rms
from src.intelligence.cue_filter import CueFilter
from src.vision.gesture_detector import GestureEvent


class TapDetector:
    """Two loud spikes within a short window simulate an earbud double-tap."""

    def __init__(
        self,
        spike_rms: float = 0.45,
        min_gap: float = 0.05,
        max_gap: float = 0.45,
    ) -> None:
        self.spike_rms = spike_rms
        self.min_gap = min_gap
        self.max_gap = max_gap
        self._spikes: list[float] = []

    def update(self, samples, now: float) -> bool:
        level = rms(samples)
        if level >= self.spike_rms:
            if not self._spikes or now - self._spikes[-1] >= self.min_gap:
                self._spikes.append(now)
        self._spikes = [
            stamp for stamp in self._spikes if now - stamp <= self.max_gap * 2
        ]
        if (
            len(self._spikes) >= 2
            and self.min_gap <= self._spikes[-1] - self._spikes[-2] <= self.max_gap
        ):
            self._spikes.clear()
            return True
        return False


class SessionControl:
    def __init__(
        self,
        themes: list[str],
        theme_name: str,
        slouch_degrees: float = 12.0,
        ear_forward_offset: float = 0.15,
        speak_up_db: float = 42.0,
        cooldown_seconds: float = 5.0,
        snooze_seconds: float = 30.0,
        global_gap_seconds: float = 1.0,
    ) -> None:
        if theme_name not in themes:
            raise ValueError(f"Unknown theme {theme_name}")
        self.themes = list(themes)
        self.theme_index = self.themes.index(theme_name)
        self.slouch_degrees = slouch_degrees
        self.ear_forward_offset = ear_forward_offset
        self.speak_up_db = speak_up_db
        self.threshold_scale = 1.0
        self.filter = CueFilter(cooldown_seconds, snooze_seconds, global_gap_seconds)
        self.taps = TapDetector()
        self._palm_held = False
        self._pinch_held = False

    @property
    def theme_name(self) -> str:
        return self.themes[self.theme_index]

    def effective_slouch(self) -> float:
        return self.slouch_degrees * self.threshold_scale

    def effective_forward(self) -> float:
        return self.ear_forward_offset * self.threshold_scale

    def effective_speak_up_db(self) -> float:
        # A higher scale raises the quiet cutoff, so soft speech is flagged sooner.
        return self.speak_up_db / self.threshold_scale

    def cycle_theme(self, direction: int = 1) -> str:
        self.theme_index = (self.theme_index + direction) % len(self.themes)
        return self.theme_name

    def adjust_threshold_scale(self, delta: float) -> float:
        self.threshold_scale = float(min(2.0, max(0.5, self.threshold_scale + delta)))
        return self.threshold_scale

    def handle_key(self, key: str, now: float) -> str | None:
        name = key.lower().strip()
        if name in {"space", "enter", "t", "return"}:
            self.filter.dismiss(now)
            return "dismiss"
        if name == "m":
            self.cycle_theme(1)
            return "theme"
        if name in {"[", "-"}:
            self.adjust_threshold_scale(-0.1)
            return "sensitivity"
        if name in {"]", "+", "="}:
            self.adjust_threshold_scale(0.1)
            return "sensitivity"
        if name == "s":
            self.filter.snooze(now)
            return "snooze"
        if name in {"quit", "escape", "q"}:
            return "quit"
        return None

    def handle_gesture(self, event: GestureEvent, now: float) -> str | None:
        action = None
        if event.swipe == "right":
            self.cycle_theme(1)
            action = "theme"
        elif event.swipe == "left":
            self.cycle_theme(-1)
            action = "theme"
        elif event.raised and event.open_palm:
            self.filter.snooze(now)
            if not self._palm_held:
                action = "snooze"
            self._palm_held = True
        else:
            self._palm_held = False
        if event.pinch and not self._pinch_held and action is None:
            self.adjust_threshold_scale(0.1)
            action = "sensitivity"
        self._pinch_held = event.pinch
        return action

    def handle_audio(self, samples, now: float) -> str | None:
        if self.taps.update(samples, now):
            self.filter.dismiss(now)
            return "dismiss"
        return None
