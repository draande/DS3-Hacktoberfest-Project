"""Typed and push-to-talk asks. Shortcuts stay quiet while a line is being edited."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class AskComposer:
    """One line of text. Open with G, Enter sends, Esc cancels."""

    def __init__(self, max_chars: int = 180) -> None:
        self.max_chars = max_chars
        self.open = False
        self.text = ""

    def handle(self, key: str | None) -> str | None:
        """``opened``, ``submit``, ``cancel``, ``typing``, or None when the key is not for us.

        While the bar is open every key is consumed, including letters that are
        also shortcuts.
        """
        if not key:
            return None
        name = key.lower()
        if not self.open:
            if name == "g":
                self.open = True
                self.text = ""
                return "opened"
            return None
        if name in {"escape", "esc"}:
            self.open = False
            self.text = ""
            return "cancel"
        if name in {"enter", "return"}:
            self.open = False
            if self.text.strip():
                return "submit"
            self.text = ""
            return "cancel"
        if name in {"backspace", "\b"}:
            self.text = self.text[:-1]
            return "typing"
        if name == "space":
            self._append(" ")
            return "typing"
        if len(name) == 1 and name.isprintable():
            self._append(name)
            return "typing"
        symbol = {
            "question": "?",
            "apostrophe": "'",
            "period": ".",
            "comma": ",",
            "exclaim": "!",
            "slash": "/",
            "minus": "-",
            "quotedbl": '"',
        }.get(name)
        if symbol:
            self._append(symbol)
        return "typing"

    def _append(self, char: str) -> None:
        if len(self.text) < self.max_chars:
            self.text += char

    def take(self) -> str:
        text = self.text.strip()
        self.text = ""
        self.open = False
        return text


@dataclass
class KeyDispatch:
    quit: bool = False
    fullscreen: str | None = None
    submitted: str | None = None
    speech_toggle: bool = False
    action: str | None = None


def dispatch_preview_key(
    key: str | None, composer: AskComposer, control, now: float
) -> KeyDispatch:
    """Route one preview key. ``key`` is already interpreted unless the bar is open.

    F, Esc, and Q are decided before this runs, so they still fullscreen and
    quit when the user is not typing. Esc while typing arrives as ``escape``.
    """
    if composer.open:
        outcome = composer.handle(key)
        if outcome == "submit":
            return KeyDispatch(submitted=composer.take())
        return KeyDispatch()
    if not key:
        return KeyDispatch()
    if key == "quit":
        return KeyDispatch(quit=True)
    if key in {"fullscreen", "exit_fullscreen"}:
        return KeyDispatch(fullscreen=key)
    if key == "v":
        return KeyDispatch(speech_toggle=True)
    if composer.handle(key) == "opened":
        return KeyDispatch()
    action = control.handle_key(key, now)
    if action == "quit":
        return KeyDispatch(quit=True)
    return KeyDispatch(action=action)


class SpeechRecorder:
    """Toggle recording on the samples the frame loop already pulled from the mic."""

    def __init__(self, max_seconds: float = 30.0) -> None:
        self.max_seconds = float(max_seconds)
        self.active = False
        self._parts: list[np.ndarray] = []
        self._rate = 16000

    def toggle(self) -> str:
        if self.active:
            self.active = False
            return "stop"
        self.active = True
        self._parts = []
        return "start"

    def add(self, samples: np.ndarray, sample_rate: int) -> bool:
        """Append one frame. True when the clip hit ``max_seconds`` and should be sent."""
        if not self.active:
            return False
        self._rate = int(sample_rate)
        self._parts.append(np.asarray(samples, dtype=np.float32).reshape(-1).copy())
        total = sum(part.size for part in self._parts)
        if self._rate > 0 and total >= int(self.max_seconds * self._rate):
            self.active = False
            return True
        return False

    def take(self) -> tuple[np.ndarray, int]:
        if not self._parts:
            audio = np.zeros(0, dtype=np.float32)
        else:
            audio = np.concatenate(self._parts)
        self._parts = []
        self.active = False
        return audio, self._rate
