"""Earbud media keys, with a fake source so CI never installs a hook."""

import logging
import sys

from src.audio.media_keys import (
    DEBOUNCE_SECONDS,
    VK_MEDIA_NEXT_TRACK,
    VK_MEDIA_PLAY_PAUSE,
    FakeKeySource,
    MediaKeyListener,
    UnavailableKeySource,
    default_key_source,
    format_key_event,
    listen_for_keys,
    log_hook_unavailable,
    parse_media_key,
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


def test_listen_keys_prints_codes_and_does_not_swallow():
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
    assert lines[0] == format_key_event(VK_MEDIA_PLAY_PAUSE, True)
    assert "play_pause down" in lines[0]
    assert "vk=0x41 (65) up" in lines[1]
    assert lines[-2].startswith("Listening for 0s")
    assert lines[-1] == "Done. 2 events."
    assert source.stopped

    media_keys._logged_unavailable = False
    missed: list[str] = []
    failed = listen_for_keys(
        seconds=0, source=UnavailableKeySource("not Windows"), output=missed.append
    )
    assert failed == 1
    assert "unavailable" in missed[0]


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
