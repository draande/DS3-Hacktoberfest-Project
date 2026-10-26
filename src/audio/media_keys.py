"""Global earbud button for Ask Gemma push-to-talk.

Bluetooth buttons on Windows arrive as AVRCP media-session events, not as
virtual keys. The primary backend owns a System Media Transport Controls
session and listens for ``ButtonPressed``. A low-level keyboard hook remains
as a fallback for keys Windows still delivers that way.

``ask.media_key.backend`` is ``auto`` (try the media session, then the hook),
``smtc``, or ``hook``. Tests pass fakes and never import WinRT or install a
hook. ``--listen-keys`` reports events from both backends.
"""

from __future__ import annotations

import ctypes
import logging
import sys
import tempfile
import threading
import time
import wave
from ctypes import wintypes
from pathlib import Path
from typing import Callable

logger = logging.getLogger("wingman.media_key")

DEBOUNCE_SECONDS = 0.3
LISTEN_SECONDS = 15.0
JOIN_TIMEOUT_SECONDS = 1.0
SMTC_TITLE = "Wingman – Ask Gemma"
SMTC_ARTIST = "Wingman"
BACKENDS = ("auto", "smtc", "hook")

# https://learn.microsoft.com/windows/win32/inputdev/virtual-key-codes
VK_VOLUME_MUTE = 0xAD
VK_VOLUME_DOWN = 0xAE
VK_VOLUME_UP = 0xAF
VK_MEDIA_NEXT_TRACK = 0xB0
VK_MEDIA_PREV_TRACK = 0xB1
VK_MEDIA_STOP = 0xB2
VK_MEDIA_PLAY_PAUSE = 0xB3
WM_QUIT = 0x0012

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

# Volume and transport keys. The hook ignores every other vkCode.
MEDIA_VK_CODES = frozenset(VK_LABELS)

# Official SystemMediaTransportControlsButton values, plus PlayPause by name.
_SMTC_BUTTON_VALUES = {
    0: "play",
    1: "pause",
    2: "stop",
    6: "next",
    7: "previous",
}
_SMTC_PLAY_NAMES = frozenset({"play", "pause", "playpause"})

_logged_unavailable = False
_hookproc_type = None


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


def smtc_actions_for(codes) -> frozenset[str]:
    """Which media-session buttons count as a tap for these virtual keys."""
    actions: set[str] = set()
    for code in codes:
        if int(code) == VK_MEDIA_PLAY_PAUSE:
            actions.update(_SMTC_PLAY_NAMES)
        elif int(code) == VK_MEDIA_NEXT_TRACK:
            actions.add("next")
        elif int(code) == VK_MEDIA_PREV_TRACK:
            actions.add("previous")
        elif int(code) == VK_MEDIA_STOP:
            actions.add("stop")
    return frozenset(actions)


def normalize_smtc_button(button) -> str:
    """Map a WinRT button, enum, or test name to ``play`` / ``next`` / ..."""
    raw = getattr(button, "name", None)
    if raw:
        text = str(raw)
    else:
        try:
            return _SMTC_BUTTON_VALUES.get(int(button), str(int(button)))
        except (TypeError, ValueError):
            text = str(button)
    return text.strip().lower().replace("_", "").replace("-", "").replace(" ", "")


def consider_hook_event(vk, is_down, watched, callback) -> bool:
    """Return True to swallow. Unwatched keys return before any callback.

    The hook calls this only after the frozen-set check. It still rechecks so
    tests can prove a letter never reaches ``callback``, and so a raising
    callback cannot escape into the hook.
    """
    try:
        code = int(vk)
    except (TypeError, ValueError):
        return False
    try:
        if code not in watched:
            return False
    except TypeError:
        return False
    if callback is None:
        return False
    try:
        return bool(callback(code, bool(is_down), None))
    except Exception:
        return False


def log_hook_unavailable(reason: str) -> None:
    """One warning per process. ``V`` keeps working after this."""
    global _logged_unavailable
    if _logged_unavailable:
        return
    _logged_unavailable = True
    logger.warning(
        "Earbud button is unavailable (%s). Push-to-talk with V still works.",
        reason,
    )


def bind_user32_signatures(user32):
    """Pointer-sized WPARAM and LPARAM for the low-level hook.

    Without these, ``CallNextHookEx`` treats a 64-bit ``lParam`` as a C int
    and raises inside the callback. The returned ``WINFUNCTYPE`` is the
    callback type ``SetWindowsHookExW`` expects.
    """
    global _hookproc_type
    lresult = ctypes.c_ssize_t
    wparam = ctypes.c_size_t
    lparam = ctypes.c_ssize_t
    if _hookproc_type is None:
        # WINFUNCTYPE is the stdcall hook on Windows. Linux tests only check
        # the argument sizes, so the cdecl factory is enough there.
        factory = getattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE)
        _hookproc_type = factory(lresult, ctypes.c_int, wparam, lparam)
    user32.SetWindowsHookExW.argtypes = [
        ctypes.c_int,
        _hookproc_type,
        wintypes.HINSTANCE,
        wintypes.DWORD,
    ]
    user32.SetWindowsHookExW.restype = ctypes.c_void_p
    user32.CallNextHookEx.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        wparam,
        lparam,
    ]
    user32.CallNextHookEx.restype = lresult
    user32.UnhookWindowsHookEx.argtypes = [ctypes.c_void_p]
    user32.UnhookWindowsHookEx.restype = wintypes.BOOL
    user32.GetMessageW.argtypes = [
        ctypes.POINTER(wintypes.MSG),
        wintypes.HWND,
        wintypes.UINT,
        wintypes.UINT,
    ]
    user32.GetMessageW.restype = ctypes.c_int
    user32.PostThreadMessageW.argtypes = [
        wintypes.DWORD,
        wintypes.UINT,
        wparam,
        lparam,
    ]
    user32.PostThreadMessageW.restype = wintypes.BOOL
    user32.PeekMessageW.argtypes = [
        ctypes.POINTER(wintypes.MSG),
        wintypes.HWND,
        wintypes.UINT,
        wintypes.UINT,
        wintypes.UINT,
    ]
    user32.PeekMessageW.restype = wintypes.BOOL
    user32.TranslateMessage.argtypes = [ctypes.POINTER(wintypes.MSG)]
    user32.TranslateMessage.restype = wintypes.BOOL
    user32.DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]
    user32.DispatchMessageW.restype = lresult
    return _hookproc_type


def write_silent_wav(path: Path, seconds: float = 0.25, rate: int = 8000) -> None:
    """A short silent wav. Looping it keeps the media session the active one."""
    frames = max(1, int(float(seconds) * int(rate)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(rate))
        handle.writeframes(b"\x00\x00" * frames)


def _load_winrt():
    """Import WinRT projections. Called only after a Windows check."""
    from winrt.windows.foundation import Uri
    from winrt.windows.media import MediaPlaybackStatus, MediaPlaybackType
    from winrt.windows.media.core import MediaSource
    from winrt.windows.media.playback import MediaPlayer

    return {
        "Uri": Uri,
        "MediaPlaybackStatus": MediaPlaybackStatus,
        "MediaPlaybackType": MediaPlaybackType,
        "MediaSource": MediaSource,
        "MediaPlayer": MediaPlayer,
    }


def _reason(source, default: str = "not started") -> str:
    text = str(getattr(source, "reason", "") or "").strip()
    return text or default


class FakeKeySource:
    """Test double for the keyboard hook. ``emit`` returns whether it was swallowed."""

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


class FakeMediaSession:
    """Test double for the Windows media session. Never touches WinRT."""

    name = "smtc"

    def __init__(
        self, fail: bool = False, reason: str = "fake media session failed"
    ) -> None:
        self.fail = fail
        self.reason = reason
        self.callback: Callable | None = None
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self, callback) -> bool:
        if self.fail:
            return False
        self.callback = callback
        self.started = True
        self.stopped = False
        self.closed = False
        self.reason = ""
        return True

    def stop(self) -> None:
        self.stopped = True
        self.closed = True
        self.started = False
        self.callback = None

    def emit(self, button: str, now: float | None = None) -> None:
        if self.callback is None:
            raise RuntimeError("FakeMediaSession has not been started")
        self.callback(button, now)


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


class SmtcMediaSession:
    """Own the system media session so earbud buttons are delivered here.

    Importing this class does not import WinRT. ``start`` imports it only on
    Windows, and a missing install says to run ``uv sync --group earbud``.
    """

    name = "smtc"

    def __init__(self) -> None:
        self.reason = "not started"
        self.started = False
        self._player = None
        self._controls = None
        self._token = None
        self._wav_path: Path | None = None
        self._callback: Callable | None = None

    def start(self, callback) -> bool:
        if sys.platform != "win32":
            self.reason = "not Windows"
            return False
        self._callback = callback
        try:
            api = _load_winrt()
        except ImportError as exc:
            self.reason = (
                "WinRT media controls are not installed "
                f"({exc}). Run: uv sync --group earbud"
            )
            return False
        try:
            self._open_player(api)
        except Exception as exc:
            self.reason = str(exc) or exc.__class__.__name__
            self._close_player()
            return False
        self.started = True
        self.reason = ""
        return True

    def stop(self) -> None:
        self._callback = None
        self._close_player()
        self.started = False

    def _open_player(self, api) -> None:
        fd, name = tempfile.mkstemp(prefix="wingman-smtc-", suffix=".wav")
        import os

        os.close(fd)
        self._wav_path = Path(name)
        write_silent_wav(self._wav_path)
        player = api["MediaPlayer"]()
        self._player = player
        player.command_manager.is_enabled = False
        controls = player.system_media_transport_controls
        self._controls = controls
        controls.is_enabled = True
        controls.is_play_enabled = True
        controls.is_pause_enabled = True
        controls.is_next_enabled = True
        controls.is_previous_enabled = True
        controls.is_stop_enabled = True
        playing = api["MediaPlaybackStatus"].PLAYING
        controls.playback_status = playing
        updater = controls.display_updater
        updater.type = api["MediaPlaybackType"].MUSIC
        updater.music_properties.title = SMTC_TITLE
        updater.music_properties.artist = SMTC_ARTIST
        updater.update()
        uri = api["Uri"](self._wav_path.resolve().as_uri())
        player.source = api["MediaSource"].create_from_uri(uri)
        player.is_looping_enabled = True
        player.volume = 0.0

        def _on_button(*args) -> None:
            try:
                event = args[-1] if args else None
                button = getattr(event, "button", event)
                name = normalize_smtc_button(button)
                callback = self._callback
                if callback is not None:
                    callback(name, None)
            except Exception:
                return
            try:
                controls.playback_status = playing
                player.play()
            except Exception:
                return

        self._token = controls.add_button_pressed(_on_button)
        player.play()

    def _close_player(self) -> None:
        controls = self._controls
        token = self._token
        player = self._player
        self._controls = None
        self._token = None
        self._player = None
        if controls is not None and token is not None:
            try:
                controls.remove_button_pressed(token)
            except Exception:
                pass
        if player is not None:
            try:
                player.pause()
            except Exception:
                pass
            try:
                player.close()
            except Exception:
                pass
        path = self._wav_path
        self._wav_path = None
        if path is not None:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


class WindowsLowLevelHook:
    """``WH_KEYBOARD_LL`` via ctypes. The installing thread pumps messages.

    Only virtual keys in the frozen watched set reach Python. Everything else
    is passed on immediately, with no logging and no lock. The callback cannot
    raise. ``stop`` unhooks from the caller so a stuck ``GetMessageW`` cannot
    keep the process alive: the thread is a daemon and join times out.
    """

    def __init__(self) -> None:
        self._callback: Callable | None = None
        self._thread: threading.Thread | None = None
        self._thread_id = 0
        self._ready = threading.Event()
        self._ok = False
        self._hook = None
        self._proc = None
        self._watched: frozenset[int] = MEDIA_VK_CODES
        self.reason = "hook was not installed"

    def set_watched(self, codes) -> None:
        self._watched = frozenset(int(code) for code in codes)

    def start(self, callback) -> bool:
        if sys.platform != "win32":
            self.reason = "not Windows"
            return False
        self._callback = callback
        self._ready.clear()
        self._ok = False
        self._thread = threading.Thread(
            target=self._run, name="wingman-media-key", daemon=True
        )
        self._thread.start()
        try:
            self._ready.wait(2.0)
        except KeyboardInterrupt:
            self.stop()
            raise
        return self._ok

    def stop(self) -> None:
        self._callback = None
        thread = self._thread
        thread_id = self._thread_id
        if sys.platform == "win32" and (self._hook or thread_id):
            user32 = ctypes.windll.user32
            bind_user32_signatures(user32)
            self._unhook(user32)
            if thread_id:
                try:
                    user32.PostThreadMessageW(int(thread_id), WM_QUIT, 0, 0)
                except Exception:
                    pass
        if thread is not None:
            thread.join(timeout=JOIN_TIMEOUT_SECONDS)
        self._thread = None
        self._ok = False

    def _unhook(self, user32) -> None:
        hook = self._hook
        self._hook = None
        if not hook:
            return
        try:
            user32.UnhookWindowsHookEx(hook)
        except Exception:
            pass

    def _run(self) -> None:
        try:
            kernel32 = ctypes.windll.kernel32
            self._thread_id = int(kernel32.GetCurrentThreadId())
            self._install_and_pump()
        except Exception as exc:
            self.reason = str(exc) or exc.__class__.__name__
            self._ok = False
            self._ready.set()

    def _install_and_pump(self) -> None:
        user32 = ctypes.windll.user32
        hookproc = bind_user32_signatures(user32)
        wh_keyboard_ll = 13
        hc_action = 0
        wm_keydown = 0x0100
        wm_syskeydown = 0x0104
        llkhf_up = 0x80

        class KBDLLHOOKSTRUCT(ctypes.Structure):
            _fields_ = [
                ("vkCode", wintypes.DWORD),
                ("scanCode", wintypes.DWORD),
                ("flags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.c_size_t),
            ]

        def _low_level(n_code, w_param, l_param):
            try:
                if n_code == hc_action:
                    info = ctypes.cast(
                        l_param, ctypes.POINTER(KBDLLHOOKSTRUCT)
                    ).contents
                    vk = int(info.vkCode)
                    if vk not in self._watched:
                        return user32.CallNextHookEx(
                            self._hook, n_code, w_param, l_param
                        )
                    is_down = int(w_param) in (wm_keydown, wm_syskeydown) and not (
                        int(info.flags) & llkhf_up
                    )
                    if consider_hook_event(vk, is_down, self._watched, self._callback):
                        return 1
                return user32.CallNextHookEx(self._hook, n_code, w_param, l_param)
            except Exception:
                return 0

        self._proc = hookproc(_low_level)
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
        while True:
            result = int(user32.GetMessageW(ctypes.byref(message), None, 0, 0))
            if result <= 0:
                break
            user32.TranslateMessage(ctypes.byref(message))
            user32.DispatchMessageW(ctypes.byref(message))
        self._unhook(user32)


def default_key_source():
    if sys.platform != "win32":
        return UnavailableKeySource("not Windows")
    return WindowsLowLevelHook()


class MediaKeyListener:
    """Turn configured earbud buttons into taps. The frame loop drains them.

    Pass ``source`` for a hook-only listener (the existing tests). Otherwise
    ``backend`` selects the media session, the hook, or auto. Auto keeps the
    hook idle when the media session starts. One debounce window covers both.
    """

    def __init__(
        self,
        keys=("play_pause",),
        suppress: bool = True,
        debounce: float = DEBOUNCE_SECONDS,
        source=None,
        on_tap: Callable[[], None] | None = None,
        smtc=None,
        hook=None,
        backend: str = "auto",
    ) -> None:
        choice = str(backend).strip().lower()
        if choice not in BACKENDS:
            raise ValueError("ask.media_key.backend must be auto, smtc, or hook")
        self.codes = tuple(parse_media_key(key) for key in keys)
        if not self.codes:
            raise ValueError("ask.media_key.keys must list at least one key")
        self.suppress = bool(suppress)
        self.debounce = float(debounce)
        self.on_tap = on_tap
        self._watched = frozenset(self.codes)
        self._smtc_actions = smtc_actions_for(self.codes)
        self._last: float | None = None
        self._taps = 0
        self._lock = threading.Lock()
        self.started = False
        self._smtc_on = False
        self._hook_on = False
        if source is not None:
            self.backend = "hook"
            self._hook = source
            self._smtc = None
        else:
            self.backend = choice
            if choice == "hook":
                self._smtc = None
            elif smtc is not None:
                self._smtc = smtc
            else:
                self._smtc = SmtcMediaSession()
            if choice == "smtc":
                self._hook = None
            elif hook is not None:
                self._hook = hook
            else:
                self._hook = default_key_source()
        self.source = self._hook

    @property
    def active(self) -> bool:
        return bool(self._smtc_on or self._hook_on)

    @property
    def active_backend(self) -> str:
        if self._smtc_on:
            return "smtc"
        if self._hook_on:
            return "hook"
        return "none"

    def status_line(self) -> str:
        """Which backend is active, or why none is."""
        if self._smtc_on:
            return (
                "Earbud: smtc active "
                f'(Windows media session "{SMTC_TITLE}"). '
                "Play, Pause, and Play/Pause toggle Ask Gemma."
            )
        if self._hook_on:
            extra = ""
            if self.backend == "auto" and self._smtc is not None:
                extra = (
                    " System media controls unavailable " f"({_reason(self._smtc)})."
                )
            return f"Earbud: keyboard hook active.{extra}"
        parts: list[str] = []
        if self._smtc is not None and self.backend != "hook":
            parts.append(f"system media controls unavailable ({_reason(self._smtc)})")
        if self._hook is not None and self.backend != "smtc":
            parts.append(f"keyboard hook unavailable ({_reason(self._hook)})")
        detail = "; ".join(parts) if parts else "no backend was configured"
        return f"Earbud: none. {detail}. Press V for push-to-talk."

    def start(self) -> bool:
        if self.started:
            return self.active
        self._smtc_on = False
        self._hook_on = False
        try:
            if self._smtc is not None and self.backend in {"auto", "smtc"}:
                self._smtc_on = bool(self._smtc.start(self.handle_smtc))
            if self._hook is not None and (
                self.backend == "hook" or (self.backend == "auto" and not self._smtc_on)
            ):
                if hasattr(self._hook, "set_watched"):
                    self._hook.set_watched(self._watched)
                self._hook_on = bool(self._hook.start(self.handle_event))
        except KeyboardInterrupt:
            self.stop()
            raise
        self.started = self._smtc_on or self._hook_on
        if not self.started:
            log_hook_unavailable(self._failure_reason())
        return self.started

    def stop(self) -> None:
        for backend in (self._smtc, self._hook):
            if backend is None:
                continue
            try:
                backend.stop()
            except Exception:
                logger.debug("earbud backend stop failed", exc_info=True)
        self.started = False
        self._smtc_on = False
        self._hook_on = False

    def _failure_reason(self) -> str:
        parts: list[str] = []
        if self._smtc is not None:
            parts.append(f"smtc: {_reason(self._smtc)}")
        if self._hook is not None:
            parts.append(f"hook: {_reason(self._hook)}")
        return "; ".join(parts) or "no backend"

    def handle_smtc(self, button, now: float | None = None) -> None:
        name = normalize_smtc_button(button)
        if name not in self._smtc_actions:
            return
        self._note_tap(now)

    def handle_event(
        self, vk: int, is_down: bool = True, now: float | None = None
    ) -> bool:
        """Return True when this hook event should be swallowed.

        The first watched event in a debounce window is one tap, whether it
        is a key-down or a key-up. Repeats and the matching edge are ignored
        until the window passes. Unwatched keys are never swallowed.
        """
        code = int(vk)
        if code not in self._watched:
            return False
        self._note_tap(now)
        return self.suppress

    def _note_tap(self, now: float | None) -> None:
        stamp = time.monotonic() if now is None else float(now)
        fire = False
        with self._lock:
            if self._last is None or stamp - self._last >= self.debounce:
                self._last = stamp
                self._taps += 1
                fire = True
        if fire and self.on_tap is not None:
            self.on_tap()

    def pop_taps(self) -> int:
        """How many taps arrived since the last pop. The frame loop drains this."""
        with self._lock:
            count = self._taps
            self._taps = 0
            return count


def _wait_until(seconds: float, stop: Callable[[], bool] | None) -> None:
    """Sleep in short slices so Ctrl+C and ``stop`` are noticed quickly."""
    deadline = time.monotonic() + max(0.0, float(seconds))
    while True:
        if stop is not None and stop():
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(0.05, remaining))


def listen_for_keys(
    seconds: float = LISTEN_SECONDS,
    source=None,
    output: Callable[[str], None] = print,
    smtc=None,
    hook=None,
    stop: Callable[[], bool] | None = None,
) -> int:
    """Print media-session buttons and media virtual keys, naming the backend.

    Does not swallow and does not record. Letters are ignored, so this window
    does not run Python on every keystroke. Ctrl+C, or ``stop`` returning
    true, ends it. Returns 1 only when neither backend starts.
    """
    if hook is None:
        hook = source
    if smtc is None and hook is None:
        smtc = SmtcMediaSession()
        hook = default_key_source()
    seen: list[str] = []

    def emit(backend: str, line: str) -> None:
        text = f"{backend} {line}"
        seen.append(text)
        output(text)

    def on_hook(vk: int, is_down: bool = True, now: float | None = None) -> bool:
        del now
        code = int(vk)
        if code not in MEDIA_VK_CODES:
            return False
        emit("hook", format_key_event(code, is_down))
        return False

    def on_smtc(button, now: float | None = None) -> None:
        del now
        emit("smtc", normalize_smtc_button(button))

    started: list[str] = []
    try:
        if smtc is not None and smtc.start(on_smtc):
            started.append("smtc")
        elif smtc is not None:
            output(f"smtc unavailable ({_reason(smtc)}).")
        if hook is not None:
            if hasattr(hook, "set_watched"):
                hook.set_watched(MEDIA_VK_CODES)
            if hook.start(on_hook):
                started.append("hook")
            else:
                output(f"hook unavailable ({_reason(hook)}).")
        if not started:
            detail = []
            if smtc is not None:
                detail.append(f"smtc: {_reason(smtc)}")
            if hook is not None:
                detail.append(f"hook: {_reason(hook)}")
            output("No earbud backend is available (" + "; ".join(detail) + ").")
            return 1
        names = ", ".join(started)
        output(
            f"Listening for {float(seconds):.0f}s on {names}. "
            "Press the earbud button. Ctrl+C stops. Do not type passwords."
        )
        try:
            _wait_until(seconds, stop)
        except KeyboardInterrupt:
            pass
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        if smtc is not None:
            try:
                smtc.stop()
            except Exception:
                logger.debug("smtc stop failed", exc_info=True)
        if hook is not None:
            try:
                hook.stop()
            except Exception:
                logger.debug("hook stop failed", exc_info=True)
        if started:
            output(f"Done. {len(seen)} events.")
