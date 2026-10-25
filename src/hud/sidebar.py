"""Short Gemma feedback bullets, newest on top, fading as they expire."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.config_loader import Theme
from src.hud.avatar import AvatarState, AvatarWidget
from src.hud.widgets import WidgetBox, hex_to_bgr
from src.intelligence.gemma_client import normalize_bullets


@dataclass
class _Bullet:
    text: str
    shown_at: float


class FeedbackSidebar:
    """Keep at most four bullets. ``visible`` returns ``(text, alpha)`` newest first."""

    def __init__(
        self,
        max_bullets: int = 4,
        ttl_seconds: float = 12.0,
        fade_seconds: float = 3.0,
    ) -> None:
        self.max_bullets = max(1, min(4, int(max_bullets)))
        self.ttl_seconds = float(ttl_seconds)
        self.fade_seconds = max(0.0, float(fade_seconds))
        self._items: list[_Bullet] = []

    def push(self, texts, now: float) -> None:
        incoming = normalize_bullets(texts)
        for text in reversed(incoming):
            self._items = [item for item in self._items if item.text != text]
            self._items.insert(0, _Bullet(text, float(now)))
        self._items = self._items[: self.max_bullets]

    def visible(self, now: float) -> list[tuple[str, float]]:
        shown: list[tuple[str, float]] = []
        fade = self.fade_seconds
        for item in self._items:
            age = float(now) - item.shown_at
            if age >= self.ttl_seconds:
                continue
            if fade <= 0 or age <= self.ttl_seconds - fade:
                alpha = 1.0
            else:
                alpha = max(0.0, (self.ttl_seconds - age) / fade)
            if alpha <= 0:
                continue
            shown.append((item.text, alpha))
        return shown


def draw_sidebar(
    image: np.ndarray,
    panel: WidgetBox,
    avatar_box: WidgetBox,
    bullets: list[tuple[str, float]],
    theme: Theme,
    avatar_state: AvatarState,
    avatar_widget: AvatarWidget,
) -> None:
    """Fill the side panel, paint the avatar slot, then the bullet lines."""
    if panel.w <= 1 or panel.h <= 1:
        return
    _fill_panel(image, panel, theme)
    _draw_border(image, panel, theme)
    avatar_widget.draw(
        image,
        (avatar_box.x, avatar_box.y, avatar_box.w, avatar_box.h),
        avatar_state,
    )
    _draw_bullets(image, panel, avatar_box, bullets, theme)


def _fill_panel(image: np.ndarray, panel: WidgetBox, theme: Theme) -> None:
    y1 = min(image.shape[0], panel.y + panel.h)
    x1 = min(image.shape[1], panel.x + panel.w)
    y0 = max(0, panel.y)
    x0 = max(0, panel.x)
    if y1 <= y0 or x1 <= x0:
        return
    roi = image[y0:y1, x0:x1]
    overlay = np.empty_like(roi)
    if theme.brackets:
        overlay[:] = hex_to_bgr(theme.colors.get("background", "#001018"))
        strength = 0.9
    else:
        overlay[:] = (230, 230, 230)
        strength = 0.38
    cv2.addWeighted(overlay, strength, roi, 1.0 - strength, 0.0, roi)


def _draw_border(image: np.ndarray, panel: WidgetBox, theme: Theme) -> None:
    color = hex_to_bgr(theme.colors.get("primary", "#FFFFFF"))
    x1 = min(image.shape[1] - 1, panel.x + panel.w - 1)
    y1 = min(image.shape[0] - 1, panel.y + panel.h - 1)
    cv2.rectangle(image, (panel.x, panel.y), (x1, y1), color, 1, cv2.LINE_AA)
    if not theme.brackets:
        return
    length = max(6, min(panel.w, panel.h) // 14)
    corners = (
        ((panel.x, panel.y), (panel.x + length, panel.y), (panel.x, panel.y + length)),
        ((x1, panel.y), (x1 - length, panel.y), (x1, panel.y + length)),
        ((panel.x, y1), (panel.x + length, y1), (panel.x, y1 - length)),
        ((x1, y1), (x1 - length, y1), (x1, y1 - length)),
    )
    for corner, horizontal, vertical in corners:
        cv2.line(image, corner, horizontal, color, 1, cv2.LINE_AA)
        cv2.line(image, corner, vertical, color, 1, cv2.LINE_AA)


def _draw_bullets(
    image: np.ndarray,
    panel: WidgetBox,
    avatar_box: WidgetBox,
    bullets: list[tuple[str, float]],
    theme: Theme,
) -> None:
    text_color = hex_to_bgr(theme.colors.get("text", "#FFFFFF"))
    font = (
        cv2.FONT_HERSHEY_COMPLEX
        if theme.font == "monospace"
        else cv2.FONT_HERSHEY_SIMPLEX
    )
    scale = max(0.32, min(0.5, panel.w / 460.0))
    line_h = max(14, int(round(28 * scale)))
    x = panel.x + max(6, panel.w // 18)
    y = avatar_box.y + avatar_box.h + line_h
    bottom = panel.y + panel.h - 6
    for text, alpha in bullets:
        if y > bottom:
            break
        color = tuple(
            max(0, min(255, int(round(channel * alpha)))) for channel in text_color
        )
        cv2.putText(
            image,
            f"- {text}",
            (x, y),
            font,
            scale,
            color,
            1,
            cv2.LINE_AA,
        )
        y += line_h
