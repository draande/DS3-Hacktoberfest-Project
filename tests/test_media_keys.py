"""Earbud media keys, with fakes so CI never installs a hook or imports WinRT."""

import ctypes
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from src.audio.media_keys import (
    DEBOUNCE_SECONDS,
    VK_MEDIA_NEXT_TRACK,
    VK_MEDIA_PLAY_PAUSE,
    FakeKeySource,
    SMTC_TITLE,
    FakeMediaSession,
    MediaKeyListener,
    SmtcMediaSession,
    UnavailableKeySource,
    bind_user32_signatures,
    consider_hook_event,
    default_key_source,
    format_key_event,
    listen_for_keys,
    log_hook_unavailable,
    parse_media_key,
    write_silent_wav,
)
from src.audio import media_keys
from src.hud.ask_input import SpeechRecorder
from src.main import parse_args, run_session


def test_play_pause_down_up_is_one_tap_and_can_be_swallowed():
    source = FakeKeySource()
    listener = MediaKeyListener(
        keys=["play_pause", "next", "headset"], source=source, suppress=True
    )
    assert listener.start()
    assert source.emit(VK_MEDIA_PLAY_PAUSE, True, 1.0) is True
    assert source.emit(VK_MEDIA_PLAY_PAUSE, False, 1.05) is True
    assert source.emit(ord("A"), True, 1.1) is False
    assert listener.pop_taps() == 1
    assert source.emit(VK_MEDIA_NEXT_TRACK, True, 1.0 + DEBOUNCE_SECONDS) is True
    assert listener.pop_taps() == 1
    assert parse_media_key("headset") == VK_MEDIA_PLAY_PAUSE
    assert parse_media_key("0xB0") == VK_MEDIA_NEXT_TRACK
    listener.stop()
    assert source.stopped
    assert not source.started


def test_repeat_inside_the_debounce_window_does_not_toggle_twice():
    source = FakeKeySource()
    taps: list[int] = []
    listener = MediaKeyListener(
        keys=["previous"],
        source=source,
        suppress=False,
        on_tap=lambda: taps.append(1),
    )
    listener.start()
    assert source.emit(0xB1, True, 0.0) is False
    assert source.emit(0xB1, True, 0.2) is False
    assert source.emit(0xB1, False, 0.29) is False
    assert taps == [1]
    assert source.emit(0xB1, True, 0.3) is False
    assert taps == [1, 1]
    recorder = SpeechRecorder()
    assert listener.pop_taps() == 2
    assert recorder.toggle() == "start"
    assert recorder.toggle() == "stop"
    listener.stop()


def test_unavailable_hook_logs_once_and_stop_is_safe(caplog):
    media_keys._logged_unavailable = False
    source = UnavailableKeySource("not Windows")
    listener = MediaKeyListener(keys=["play_pause"], source=source)
    with caplog.at_level(logging.WARNING, logger="wingman.media_key"):
        assert listener.start() is False
        assert listener.start() is False
    assert source.stop_calls == 0
    listener.stop()
    assert source.stop_calls == 1
    warnings = [record.message for record in caplog.records]
    assert len(warnings) == 1
    assert "V still works" in warnings[0]
    log_hook_unavailable("again")
    assert len(caplog.records) == 1


def test_listen_keys_names_the_backend_and_ignores_letters():
    class EmitOnStart(FakeKeySource):
        def start(self, callback):
            super().start(callback)
            assert self.emit(VK_MEDIA_PLAY_PAUSE, True) is False
            assert self.emit(0x41, False) is False
            return True

    source = EmitOnStart()
    lines: list[str] = []
    code = listen_for_keys(seconds=0, source=source, output=lines.append)
    assert code == 0
    assert lines[0] == f"hook {format_key_event(VK_MEDIA_PLAY_PAUSE, True)}"
    assert "play_pause down" in lines[0]
    assert not any("0x41" in line for line in lines)
    assert lines[-2].startswith("Listening for 0s")
    assert lines[-1] == "Done. 1 events."
    assert source.stopped

    media_keys._logged_unavailable = False
    missed: list[str] = []
    failed = listen_for_keys(
        seconds=0, source=UnavailableKeySource("not Windows"), output=missed.append
    )
    assert failed == 1
    assert "unavailable" in missed[0]
    assert not any(line.startswith("Done.") for line in missed)


def test_default_source_does_not_install_a_hook_off_windows():
    if sys.platform == "win32":
        return
    source = default_key_source()
    assert source.start(lambda *_args: False) is False


def test_flags_and_session_use_a_fake_listener_without_a_real_hook():
    assert parse_args(["--earbud"]).earbud is True
    assert parse_args(["--listen-keys"]).listen_keys is True
    source = FakeKeySource()
    listener = MediaKeyListener(keys=["play_pause"], source=source)
    listener.start()
    source.emit(VK_MEDIA_PLAY_PAUSE, True, 0.0)
    summary = run_session(
        parse_args(
            [
                "--offline",
                "--frames",
                "2",
                "--fps",
                "10",
                "--width",
                "80",
                "--height",
                "48",
            ]
        ),
        media_listener=listener,
    )
    assert summary.error is None
    assert summary.frames == 2
    assert source.stopped
    assert "Earbud button unavailable" not in " ".join(summary.notes)


def test_smtc_play_pause_and_playpause_share_one_debounce_with_the_hook():
    smtc = FakeMediaSession()
    hook = FakeKeySource()
    listener = MediaKeyListener(
        keys=["play_pause", "next"],
        backend="auto",
        smtc=smtc,
        hook=hook,
        suppress=True,
    )
    assert listener.start()
    assert listener.active_backend == "smtc"
    assert smtc.started
    assert not hook.started
    assert "smtc active" in listener.status_line()
    assert SMTC_TITLE in listener.status_line()
    smtc.emit("play", 1.0)
    smtc.emit("pause", 1.1)
    smtc.emit("playpause", 1.2)
    listener.handle_event(VK_MEDIA_PLAY_PAUSE, True, 1.2)
    assert listener.pop_taps() == 1
    smtc.emit("next", 1.0 + DEBOUNCE_SECONDS)
    assert listener.pop_taps() == 1
    smtc.emit("previous", 2.0)
    assert listener.pop_taps() == 0
    listener.stop()
    assert smtc.stopped
    assert smtc.closed
    assert hook.stopped


def test_auto_falls_back_to_the_hook_and_smtc_only_does_not():
    smtc = FakeMediaSession(fail=True, reason="no media session")
    hook = FakeKeySource()
    listener = MediaKeyListener(keys=["next"], backend="auto", smtc=smtc, hook=hook)
    assert listener.start()
    assert listener.active_backend == "hook"
    line = listener.status_line()
    assert line.startswith("Earbud: keyboard hook active.")
    assert "no media session" in line
    assert hook.emit(VK_MEDIA_NEXT_TRACK, True, 0.0) is True
    assert hook.emit(VK_MEDIA_PLAY_PAUSE, True, 1.0) is False
    assert listener.pop_taps() == 1
    listener.stop()

    skipped = FakeKeySource()
    only = MediaKeyListener(
        keys=["play_pause"],
        backend="smtc",
        smtc=FakeMediaSession(fail=True, reason="missing"),
        hook=skipped,
    )
    assert only.start() is False
    assert not skipped.started
    assert only.status_line().startswith("Earbud: none.")
    assert "missing" in only.status_line()
    assert "keyboard hook" not in only.status_line()
    only.stop()


def test_listen_keys_reports_both_backends_and_stops_promptly():
    class EmitHook(FakeKeySource):
        def start(self, callback):
            super().start(callback)
            self.emit(VK_MEDIA_NEXT_TRACK, True)
            self.emit(0x41, True)
            return True

    class EmitSmtc(FakeMediaSession):
        def start(self, callback):
            super().start(callback)
            self.emit("Play")
            self.emit("channel_up")
            return True

    lines: list[str] = []
    code = listen_for_keys(
        seconds=0, smtc=EmitSmtc(), hook=EmitHook(), output=lines.append
    )
    assert code == 0
    assert "smtc play" in lines
    assert "smtc channelup" in lines
    assert any(line.startswith("hook vk=0xB0") for line in lines)
    assert not any("0x41" in line for line in lines)
    assert "on smtc, hook" in lines[-2]

    smtc = FakeMediaSession()
    hook = FakeKeySource()
    started = time.monotonic()
    stopped = listen_for_keys(
        seconds=30,
        smtc=smtc,
        hook=hook,
        stop=lambda: True,
        output=lambda _line: None,
    )
    assert time.monotonic() - started < 1.0
    assert stopped == 0
    assert smtc.stopped and hook.stopped

    smtc = FakeMediaSession()
    hook = FakeKeySource()

    def interrupt():
        raise KeyboardInterrupt

    code = listen_for_keys(
        seconds=30, smtc=smtc, hook=hook, stop=interrupt, output=lambda _line: None
    )
    assert code == 0
    assert smtc.stopped and hook.stopped


def test_session_shutdown_stops_a_running_listener():
    smtc = FakeMediaSession()
    hook = FakeKeySource()
    listener = MediaKeyListener(
        keys=["play_pause"], backend="auto", smtc=smtc, hook=hook
    )
    assert listener.start()
    assert smtc.started

    def halt():
        raise KeyboardInterrupt

    started = time.monotonic()
    summary = run_session(
        parse_args(
            [
                "--offline",
                "--frames",
                "30",
                "--fps",
                "10",
                "--width",
                "80",
                "--height",
                "48",
            ]
        ),
        media_listener=listener,
        should_stop=halt,
    )
    assert time.monotonic() - started < 10
    assert summary.frames == 0
    assert summary.error is None
    assert "Stopped." in summary.notes
    assert smtc.stopped
    assert smtc.closed


def test_session_records_the_backend_it_starts(capsys):
    source = FakeKeySource()
    listener = MediaKeyListener(keys=["play_pause"], source=source)
    summary = run_session(
        parse_args(
            [
                "--offline",
                "--frames",
                "1",
                "--fps",
                "10",
                "--width",
                "80",
                "--height",
                "48",
            ]
        ),
        media_listener=listener,
    )
    assert source.stopped
    assert any(
        note.startswith("Earbud: keyboard hook active") for note in summary.notes
    )
    assert "Earbud: keyboard hook active" in capsys.readouterr().out


def test_smtc_import_is_skipped_until_windows_needs_it(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")

    def boom():
        raise AssertionError("winrt imported")

    monkeypatch.setattr(media_keys, "_load_winrt", boom)
    session = SmtcMediaSession()
    assert session.start(lambda *_args: None) is False
    assert session.reason == "not Windows"
    session.stop()

    monkeypatch.setattr(sys, "platform", "win32")

    def missing():
        raise ImportError("No module named winrt")

    monkeypatch.setattr(media_keys, "_load_winrt", missing)
    session = SmtcMediaSession()
    assert session.start(lambda *_args: None) is False
    assert "uv sync --group earbud" in session.reason
    session.stop()


def test_hook_fast_path_and_pointer_sized_signatures(tmp_path):
    watched = frozenset({VK_MEDIA_PLAY_PAUSE})
    seen: list[int] = []
    assert (
        consider_hook_event(0x41, True, watched, lambda *_args: seen.append(1)) is False
    )
    assert seen == []
    assert consider_hook_event(VK_MEDIA_PLAY_PAUSE, True, watched, lambda *_a: True)

    def boom(*_args):
        raise RuntimeError("hook callback")

    assert consider_hook_event(VK_MEDIA_PLAY_PAUSE, False, watched, boom) is False

    class Api:
        def __init__(self) -> None:
            self.argtypes = None
            self.restype = None

    user32 = SimpleNamespace(
        SetWindowsHookExW=Api(),
        CallNextHookEx=Api(),
        UnhookWindowsHookEx=Api(),
        GetMessageW=Api(),
        PostThreadMessageW=Api(),
        PeekMessageW=Api(),
        TranslateMessage=Api(),
        DispatchMessageW=Api(),
    )
    bind_user32_signatures(user32)
    assert user32.CallNextHookEx.argtypes[2] is ctypes.c_size_t
    assert user32.CallNextHookEx.argtypes[3] is ctypes.c_ssize_t
    assert user32.CallNextHookEx.restype is ctypes.c_ssize_t
    assert user32.PostThreadMessageW.argtypes[2] is ctypes.c_size_t
    assert user32.PostThreadMessageW.argtypes[3] is ctypes.c_ssize_t
    assert user32.GetMessageW.restype is ctypes.c_int
    assert user32.SetWindowsHookExW.restype is ctypes.c_void_p
    assert user32.UnhookWindowsHookEx.argtypes == [ctypes.c_void_p]

    path = Path(tmp_path) / "silence.wav"
    write_silent_wav(path)
    assert path.stat().st_size > 44


def test_auto_without_fakes_does_not_import_winrt_off_windows(monkeypatch):
    if sys.platform == "win32":
        return

    def boom():
        raise AssertionError("winrt imported")

    monkeypatch.setattr(media_keys, "_load_winrt", boom)
    media_keys._logged_unavailable = False
    listener = MediaKeyListener(keys=["play_pause"], backend="auto")
    assert listener.start() is False
    assert "not Windows" in listener.status_line()
    assert "Press V" in listener.status_line()
    listener.stop()
