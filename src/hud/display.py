"""Borderless fullscreen preview at the screen's native resolution.

``--display`` asks OpenCV for a real fullscreen window and renders the HUD on a
frame that is already the screen size. Stretching a 1280x720 canvas is not
enough: Windows then letterboxes or stops short of the edges. Headless mode
never calls :func:`detect_screen_size` and never opens a window.
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np


def detect_screen_size() -> tuple[int, int] | None:
    """Return ``(width, height)`` in physical pixels, or None when unknown.

    On Windows the process is marked per-monitor DPI aware first, so
    ``GetSystemMetrics`` reports the same pixels OpenCV will cover. Other
    platforms ask Tk. No window is mapped.
    """
    if sys.platform == "win32":
        return _windows_screen_size()
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        return None
    return _tk_screen_size()


def _windows_screen_size() -> tuple[int, int] | None:
    try:
        import ctypes

        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass
        user32 = ctypes.windll.user32
        width = int(user32.GetSystemMetrics(0))
        height = int(user32.GetSystemMetrics(1))
    except Exception:
        return None
    if width > 0 and height > 0:
        return width, height
    return None


def _tk_screen_size() -> tuple[int, int] | None:
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        width = int(root.winfo_screenwidth())
        height = int(root.winfo_screenheight())
        root.destroy()
    except Exception:
        return None
    if width > 0 and height > 0:
        return width, height
    return None


def prepare_display_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Return a frame that is exactly ``height`` by ``width``.

    Matching frames are returned unchanged. Anything else is stretched to fill
    the screen. There is no letterbox.
    """
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


def configure_preview_window(
    title: str, size: tuple[int, int], fullscreen: bool, cv2_module=cv2
) -> None:
    """Size a named OpenCV window and set the fullscreen property.

    Windows often ignores ``WND_PROP_FULLSCREEN`` until after the first
    ``imshow``. Callers that stay fullscreen should set the property again
    after showing the frame.
    """
    width, height = size
    cv2_module.namedWindow(title, cv2_module.WINDOW_NORMAL)
    if fullscreen:
        cv2_module.resizeWindow(title, int(width), int(height))
        cv2_module.moveWindow(title, 0, 0)
        cv2_module.setWindowProperty(
            title, cv2_module.WND_PROP_FULLSCREEN, cv2_module.WINDOW_FULLSCREEN
        )
        return
    cv2_module.setWindowProperty(
        title, cv2_module.WND_PROP_FULLSCREEN, cv2_module.WINDOW_NORMAL
    )
    cv2_module.resizeWindow(title, int(width), int(height))
