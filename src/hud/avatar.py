"""Placeholder slot for the avatar another part of the team is building.

This module does not draw a character. It reserves a rectangle and paints a
labeled stand-in so the layout is obvious. Drop the real avatar in by replacing
:func:`draw_avatar` or by assigning a custom :class:`AvatarWidget` to
``HudRenderer.avatar_widget``.

Hook::

    def draw_avatar(frame, rect, state) -> None:
        x, y, width, height = rect  # pixels on the BGR frame
        # state.theme_name, state.cue_type, state.display_text
        # state.speaking, state.volume_db, state.wpm, state.remaining_seconds

``rect`` is ``(x, y, width, height)`` and stays inside the frame. The default
layout puts that rectangle at the top of the feedback sidebar. ``hud.avatar``
in ``config/default_config.yaml`` sets the anchor and how much of the sidebar
the slot uses.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class AvatarState:
    """Snapshot the avatar can read. The placeholder only uses a few fields."""

    theme_name: str
    cue_type: str | None
    display_text: str | None
    speaking: bool
    volume_db: float
    wpm: float
    remaining_seconds: float


def draw_avatar(
    frame: np.ndarray, rect: tuple[int, int, int, int], state: AvatarState
) -> None:
    """Paint the placeholder. Replace this function with the real avatar."""
    x, y, width, height = (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))
    if width <= 1 or height <= 1:
        return
    frame_h, frame_w = frame.shape[:2]
    x0 = max(0, min(x, frame_w - 1))
    y0 = max(0, min(y, frame_h - 1))
    x1 = max(x0 + 1, min(x + width, frame_w))
    y1 = max(y0 + 1, min(y + height, frame_h))
    if state.theme_name == "minimalist":
        fill = (48, 48, 48)
        border = (255, 255, 255)
        caption = (220, 220, 220)
    else:
        fill = (48, 32, 8)
        border = (255, 240, 0)
        caption = (255, 240, 0)
    # Draw on a crop so antialiased strokes cannot spill past the slot.
    slot = frame[y0:y1, x0:x1].copy()
    local_w = x1 - x0
    local_h = y1 - y0
    cv2.rectangle(slot, (0, 0), (local_w - 1, local_h - 1), fill, -1)
    cv2.rectangle(slot, (0, 0), (local_w - 1, local_h - 1), border, 1, cv2.LINE_AA)
    scale = max(0.35, min(0.7, local_w / 280.0, local_h / 120.0))
    cv2.putText(
        slot,
        "AVATAR",
        (8, max(16, local_h // 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        border,
        1,
        cv2.LINE_AA,
    )
    if state.cue_type and local_h >= 36:
        cv2.putText(
            slot,
            str(state.cue_type),
            (8, min(local_h - 6, max(16, local_h // 2) + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.3, scale * 0.6),
            caption,
            1,
            cv2.LINE_AA,
        )
    frame[y0:y1, x0:x1] = slot


class AvatarWidget:
    """Widget wrapper around :func:`draw_avatar`.

    Assign an instance with a different ``draw`` method to
    ``HudRenderer.avatar_widget`` and the renderer will call it with the same
    ``(frame, rect, state)`` contract.
    """

    def draw(
        self, frame: np.ndarray, rect: tuple[int, int, int, int], state: AvatarState
    ) -> None:
        draw_avatar(frame, rect, state)
