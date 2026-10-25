import numpy as np

from src.hud.display import (
    configure_preview_window,
    detect_screen_size,
    prepare_display_frame,
)
from src.hud.renderer import PreviewWindow


class FakeCv2:
    WINDOW_NORMAL = 0
    WINDOW_FULLSCREEN = 1
    WND_PROP_FULLSCREEN = 2

    def __init__(self, key: int = 255) -> None:
        self.key = key
        self.calls: list[tuple] = []

    def namedWindow(self, title, flag) -> None:
        self.calls.append(("namedWindow", title, flag))

    def resizeWindow(self, title, width, height) -> None:
        self.calls.append(("resizeWindow", title, width, height))

    def moveWindow(self, title, x, y) -> None:
        self.calls.append(("moveWindow", title, x, y))

    def setWindowProperty(self, title, prop, value) -> None:
        self.calls.append(("setWindowProperty", title, prop, value))

    def imshow(self, title, frame) -> None:
        self.calls.append(("imshow", title, frame.shape))

    def waitKey(self, delay: int) -> int:
        self.calls.append(("waitKey", delay))
        return self.key


def test_fullscreen_window_is_sized_to_the_native_resolution():
    fake = FakeCv2()
    configure_preview_window("Wingman", (1920, 1080), True, cv2_module=fake)
    assert ("namedWindow", "Wingman", fake.WINDOW_NORMAL) in fake.calls
    assert ("resizeWindow", "Wingman", 1920, 1080) in fake.calls
    assert ("moveWindow", "Wingman", 0, 0) in fake.calls
    assert (
        "setWindowProperty",
        "Wingman",
        fake.WND_PROP_FULLSCREEN,
        fake.WINDOW_FULLSCREEN,
    ) in fake.calls


def test_leaving_fullscreen_restores_a_normal_window():
    fake = FakeCv2()
    configure_preview_window("Wingman", (1280, 720), False, cv2_module=fake)
    assert (
        "setWindowProperty",
        "Wingman",
        fake.WND_PROP_FULLSCREEN,
        fake.WINDOW_NORMAL,
    ) in fake.calls
    assert ("resizeWindow", "Wingman", 1280, 720) in fake.calls
    assert not any(call[0] == "moveWindow" for call in fake.calls)


def test_prepare_display_frame_matches_the_screen_exactly():
    frame = np.zeros((90, 160, 3), np.uint8)
    filled = prepare_display_frame(frame, 1920, 1080)
    assert filled.shape == (1080, 1920, 3)
    same = prepare_display_frame(filled, 1920, 1080)
    assert same is filled


def test_escape_exits_fullscreen_and_f_toggles_it():
    window = PreviewWindow(
        screen_size=(1920, 1080), windowed_size=(1280, 720), fullscreen=True
    )
    assert window.interpret_key("escape") == "exit_fullscreen"
    assert window.fullscreen is False
    assert window.target_size() == (1280, 720)
    assert window.interpret_key("escape") == "quit"
    assert window.interpret_key("f") == "fullscreen"
    assert window.fullscreen is True
    assert window.target_size() == (1920, 1080)
    assert window.interpret_key("f") == "exit_fullscreen"
    assert window.target_size() == (1280, 720)
    assert window.interpret_key("q") == "quit"


def test_fullscreen_property_is_set_again_after_the_frame_is_shown():
    fake = FakeCv2()
    window = PreviewWindow(
        screen_size=(200, 100),
        windowed_size=(80, 40),
        fullscreen=True,
        cv2_module=fake,
    )
    small = np.zeros((40, 80, 3), np.uint8)
    assert window.show(small) is None
    assert not any(call[0] == "namedWindow" for call in fake.calls)

    fake.calls.clear()
    assert window.show(np.zeros((100, 200, 3), np.uint8)) is None
    names = [call[0] for call in fake.calls]
    imshow_at = names.index("imshow")
    prop_at = [index for index, name in enumerate(names) if name == "setWindowProperty"]
    assert prop_at[-1] > imshow_at
    assert ("resizeWindow", "Wingman", 200, 100) in fake.calls


def test_show_escape_leaves_fullscreen_before_a_second_escape_quits():
    fake = FakeCv2(key=27)
    window = PreviewWindow(
        screen_size=(200, 100),
        windowed_size=(80, 40),
        fullscreen=True,
        cv2_module=fake,
    )
    frame = np.zeros((100, 200, 3), np.uint8)
    assert window.show(frame) == "exit_fullscreen"
    assert window.fullscreen is False
    assert window.target_size() == (80, 40)
    assert window.show(frame) == "quit"


def test_detect_screen_size_does_not_crash_without_a_display():
    size = detect_screen_size()
    assert size is None or (size[0] > 0 and size[1] > 0)
