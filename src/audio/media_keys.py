"""Global earbud button for Ask Gemma push-to-talk.

On Windows this is a low-level keyboard hook (``WH_KEYBOARD_LL``) on a
background thread, so the Wingman window does not have to be focused.
Play/Pause is swallowed by default, which keeps music from pausing too.
Other platforms, and a hook that fails to install, log once and leave the
``V`` key working. Tests pass a :class:`FakeKeySource` and never install a hook.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from typing import Callable

logger = logging.getLogger("wingman.media_key")

DEBOUNCE_SECONDS = 0.3
LISTEN_SECONDS = 15.0

# https://learn.microsoft.com/windows/win32/inputdev/virtual-key-codes
VK_VOLUME_MUTE = 0xAD
VK_VOLUME_DOWN = 0xAE
VK_VOLUME_UP = 0xAF
VK_MEDIA_NEXT_TRACK = 0xB0
VK_MEDIA_PREV_TRACK = 0xB1
VK_MEDIA_STOP = 0xB2
VK_MEDIA_PLAY_PAUSE = 0xB3

# Windows reports most Bluetooth earbud buttons as Play/Pause. ``headset`` is
# that same virtual key so the config can name it; ``--listen-keys`` prints
# the code if a headset sends something else.
KEY_NAMES = {
    "play_pause": VK_MEDIA_PLAY_PAUSE,
    "play": VK_MEDIA_PLAY_PAUSE,
    "pause": VK_MEDIA_PLAY_PAUSE,
    "vk_media_play_pause": VK_MEDIA_PLAY_PAUSE,
    "next": VK_MEDIA_NEXT_TRACK,
    "next_track": VK_MEDIA_NEXT_TRACK,
    "previous": VK_MEDIA_PREV_TRACK,
    "prev": VK_MEDIA_PREV_TRACK,
    "prev_track": VK_MEDIA_PREV_TRACK,
    "previous_track": VK_MEDIA_PREV_TRACK,
    "stop": VK_MEDIA_STOP,
    "headset": VK_MEDIA_PLAY_PAUSE,
    "hook": VK_MEDIA_PLAY_PAUSE,
    "headset_hook": VK_MEDIA_PLAY_PAUSE,
    "volume_mute": VK_VOLUME_MUTE,
    "volume_down": VK_VOLUME_DOWN,
    "volume_up": VK_VOLUME_UP,
}

VK_LABELS = {
    VK_MEDIA_PLAY_PAUSE: "play_pause",
    VK_MEDIA_NEXT_TRACK: "next_track",
    VK_MEDIA_PREV_TRACK: "previous_track",
    VK_MEDIA_STOP: "stop",
    VK_VOLUME_MUTE: "volume_mute",
    VK_VOLUME_DOWN: "volume_down",
    VK_VOLUME_UP: "volume_up",
}

_logged_unavailable = False


def parse_media_key(value) -> int:
    """A config name (``play_pause``, ``next``, ``previous``, ``headset``) or a VK code."""
    if isinstance(value, bool):
        raise ValueError(f"Unknown media key {value!r}")
    if isinstance(value, int):
        return int(value)
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if text in KEY_NAMES:
        return KEY_NAMES[text]
    try:
        return int(text, 0)
    except ValueError as exc:
        known = "play_pause, next, previous, stop, headset"
        raise ValueError(
            f"Unknown media key {value!r}. Use {known}, or a virtual-key code."
        ) from exc


def format_key_event(vk: int, is_down: bool) -> str:
    label = VK_LABELS.get(int(vk), "")
    name = f" {label}" if label else ""
    edge = "down" if is_down else "up"
    return f"vk=0x{int(vk):02X} ({int(vk)}){name} {edge}"


def log_hook_unavailable(reason: str) -> None:
    """One warning per process. ``V`` keeps working after this."""
    global _logged_unavailable
    if _logged_unavailable:
        return
    _logged_unavailable = True
    logger.warning(
        "Media-key hook is unavailable (%s). Push-to-talk with V still works.",
        reason,
    )


class FakeKeySource:
    """Test double. ``emit`` delivers one event and returns whether it was swallowed."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.callback: Callable | None = None
        self.started = False
        self.stopped = False

    def start(self, callback) -> bool:
        if self.fail:
            return False
        self.callback = callback
        self.started = True
        self.stopped = False
        return True

    def stop(self) -> None:
        self.stopped = True
        self.started = False

    def emit(self, vk: int, is_down: bool = True, now: float | None = None) -> bool:
        if self.callback is None:
            raise RuntimeError("FakeKeySource has not been started")
        return bool(self.callback(int(vk), bool(is_down), now))


class UnavailableKeySource:
    """Used off Windows, and when the real hook cannot be installed."""

    def __init__(self, reason: str = "not Windows") -> None:
        self.reason = reason
        self.stop_calls = 0

    def start(self, callback) -> bool:
        del callback
        return False

    def stop(self) -> None:
        self.stop_calls += 1


class WindowsLowLevelHook:
    """``WH_KEYBOARD_LL`` via ctypes. The installing thread pumps messages.

    Returning 1 from the hook, without calling ``CallNextHookEx``, swallows
    that key. Only keys the listener asks to suppress are swallowed.
    """

    def __init__(self) -> None:
        self._callback: Callable | None = None
        self._thread: threading.Thread | None = None
        self._thread_id = 0
        self._ready = threading.Event()
        self._ok = False
        self._hook = None
        self._proc = None
        self.reason = "hook was not installed"

    def start(self, callback) -> bool:
        if sys.platform != "win32":
            self.reason = "not Windows"
            return False
        self._callback = callback
        self._ready.clear()
        self._thread = threading.Thread(
            target=self._run, name="wingman-media-key", daemon=True
        )
        self._thread.start()
        self._ready.wait(2.0)
        return self._ok

    def stop(self) -> None:
        thread = self._thread
        thread_id = self._thread_id
        if thread is None:
            return
        if thread_id and sys.platform == "win32":
            import ctypes

            ctypes.windll.user32.PostThreadMessageW(thread_id, 0x0012, 0, 0)
        thread.join(timeout=1.5)
        self._thread = None
        self._ok = False

    def _run(self) -> None:
        try:
            self._install_and_pump()
        except Exception as exc:
            self.reason = str(exc) or exc.__class__.__name__
            self._ok = False
            self._ready.set()

    def _install_and_pump(self) -> None:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        wh_keyboard_ll = 13
        hc_action = 0
        wm_keydown = 0x0100
        wm_syskeydown = 0x0104
        llkhf_up = 0x80
        lresult = (
            ctypes.c_longlong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_long
        )

        class KBDLLHOOKSTRUCT(ctypes.Structure):
            _fields_ = [
                ("vkCode", wintypes.DWORD),
                ("scanCode", wintypes.DWORD),
                ("flags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.c_size_t),
            ]

        hookproc = ctypes.WINFUNCTYPE(
            lresult, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM
        )

        def _low_level(n_code, w_param, l_param):
            if n_code == hc_action and self._callback is not None:
                info = ctypes.cast(l_param, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
                is_down = w_param in (wm_keydown, wm_syskeydown) and not (
                    info.flags & llkhf_up
                )
                if self._callback(int(info.vkCode), bool(is_down), None):
                    return 1
            return user32.CallNextHookEx(self._hook, n_code, w_param, l_param)

        self._proc = hookproc(_low_level)
        user32.SetWindowsHookExW.restype = ctypes.c_void_p
        user32.CallNextHookEx.restype = lresult
        self._thread_id = int(kernel32.GetCurrentThreadId())
        self._hook = user32.SetWindowsHookExW(wh_keyboard_ll, self._proc, None, 0)
        if not self._hook:
            self.reason = f"SetWindowsHookExW failed ({ctypes.get_last_error()})"
            self._ok = False
            self._ready.set()
            return
        message = wintypes.MSG()
        user32.PeekMessageW(ctypes.byref(message), None, 0, 0, 0)
        self._ok = True
        self._ready.set()
        while user32.GetMessageW(ctypes.byref(message), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(message))
            user32.DispatchMessageW(ctypes.byref(message))
        user32.UnhookWindowsHookEx(self._hook)
        self._hook = None


def default_key_source():
    if sys.platform != "win32":
        return UnavailableKeySource("not Windows")
    return WindowsLowLevelHook()


class MediaKeyListener:
    """Turn configured media keys into taps. Down/up pairs count as one tap."""

    def __init__(
        self,
        keys=("play_pause",),
        suppress: bool = True,
        debounce: float = DEBOUNCE_SECONDS,
        source=None,
        on_tap: Callable[[], None] | None = None,
    ) -> None:
        self.codes = tuple(parse_media_key(key) for key in keys)
        if not self.codes:
            raise ValueError("ask.media_key.keys must list at least one key")
        self.suppress = bool(suppress)
        self.debounce = float(debounce)
        self.source = default_key_source() if source is None else source
        self.on_tap = on_tap
        self._watched = set(self.codes)
        self._last: float | None = None
        self._taps = 0
        self._lock = threading.Lock()
        self.started = False

    def start(self) -> bool:
        if self.started:
            return True
        ok = bool(self.source.start(self.handle_event))
        self.started = ok
        if not ok:
            reason = getattr(self.source, "reason", "hook was not installed")
            log_hook_unavailable(str(reason))
        return ok

    def stop(self) -> None:
        self.source.stop()
        self.started = False

    def handle_event(
        self, vk: int, is_down: bool = True, now: float | None = None
    ) -> bool:
        """Return True when this event should be swallowed.

        The first watched event in a debounce window is one tap, whether it
        is a key-down or a key-up. Repeats and the matching up/down are
        ignored until the window passes. Unwatched keys are never swallowed.
        """
        code = int(vk)
        if code not in self._watched:
            return False
        stamp = time.monotonic() if now is None else float(now)
        fire = False
        with self._lock:
            if self._last is None or stamp - self._last >= self.debounce:
                self._last = stamp
                self._taps += 1
                fire = True
        if fire and self.on_tap is not None:
            self.on_tap()
        return self.suppress

    def pop_taps(self) -> int:
        """How many taps arrived since the last pop. The frame loop drains this."""
        with self._lock:
            count = self._taps
            self._taps = 0
            return count


def listen_for_keys(
    seconds: float = LISTEN_SECONDS,
    source=None,
    output: Callable[[str], None] = print,
) -> int:
    """Print every virtual key for ``seconds``. Does not swallow, and does not record.

    This is how to see what an earbud button actually sends. It does not start
    the camera or the session.
    """
    hook = default_key_source() if source is None else source
    seen: list[str] = []

    def handle(vk: int, is_down: bool = True, now: float | None = None) -> bool:
        del now
        line = format_key_event(vk, is_down)
        seen.append(line)
        output(line)
        return False

    if not hook.start(handle):
        reason = getattr(hook, "reason", "hook was not installed")
        log_hook_unavailable(str(reason))
        output(f"Media-key hook is unavailable ({reason}).")
        return 1
    output(
        f"Listening for {float(seconds):.0f}s. Press the earbud button. "
        "Do not type passwords."
    )
    try:
        time.sleep(max(0.0, float(seconds)))
    finally:
        hook.stop()
    output(f"Done. {len(seen)} events.")
    return 0
