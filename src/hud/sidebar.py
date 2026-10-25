"""Short Gemma feedback bullets, newest on top, fading as they expire."""

from __future__ import annotations

import time
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
        self._insights: list[_Bullet] = []
        self._ask_question = ""
        self._ask_phase = "idle"
        self._ask_lines: list[str] = []
        self._ask_started = 0.0

    def push(self, texts, now: float) -> None:
        incoming = normalize_bullets(texts)
        for text in reversed(incoming):
            self._items = [item for item in self._items if item.text != text]
            self._items.insert(0, _Bullet(text, float(now)))
        self._items = self._items[: self.max_bullets]

    def push_insights(self, texts, now: float) -> None:
        """Replace the Gemma trend lines. Same text does not reset the clock."""
        incoming = normalize_bullets(texts)[:3]
        if incoming == [item.text for item in self._insights]:
            return
        self._insights = [_Bullet(text, float(now)) for text in incoming]

    def visible_insights(self, now: float) -> list[tuple[str, float]]:
        del now
        return [(item.text, 1.0) for item in self._insights]

    def show_ask(
        self,
        question: str,
        phase: str,
        lines,
        started: float | None = None,
    ) -> None:
        """Remember a direct question. It stays until the next ask replaces it."""
        self._ask_question = " ".join(str(question or "").split())
        self._ask_phase = phase or "idle"
        self._ask_lines = [str(line) for line in lines if str(line).strip()]
        if started is None:
            self._ask_started = time.monotonic()
        else:
            self._ask_started = float(started)

    def visible_ask(self) -> list[str]:
        """``YOU``, then ``THINKING... Ns`` or an ``ASK`` block. Not trimmed to six words."""
        if self._ask_phase == "idle":
            return []
        rows: list[str] = []
        if self._ask_question:
            rows.append(f"YOU: {self._ask_question}")
        if self._ask_phase == "thinking":
            elapsed = max(0, int(time.monotonic() - self._ask_started))
            rows.append(f"THINKING... {elapsed}s")
        elif self._ask_phase == "listening":
            rows.append("LISTENING...")
        else:
            rows.append("ASK")
            rows.extend(self._ask_lines)
        return rows

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


BADGE_LABELS = {"gemma": "GEMMA", "thinking": "THINKING", "offline": "OFFLINE"}


def coaching_badge_rect(panel: WidgetBox) -> tuple[int, int, int, int]:
    """Small strip at the bottom of the sidebar for the Gemma status."""
    height = 16 if panel.h >= 40 else max(4, panel.h // 5)
    height = min(height, max(1, panel.h))
    inset = 2 if panel.w > 8 else 0
    width = max(1, panel.w - 2 * inset)
    x = panel.x + inset if panel.w > width else panel.x
    y = panel.y + panel.h - height
    return int(x), int(y), int(width), int(height)


def draw_sidebar(
    image: np.ndarray,
    panel: WidgetBox,
    avatar_box: WidgetBox,
    bullets: list[tuple[str, float]],
    theme: Theme,
    avatar_state: AvatarState,
    avatar_widget: AvatarWidget,
    coaching_status: str = "offline",
    insights: list[tuple[str, float]] | None = None,
    ask_rows: list[str] | None = None,
) -> None:
    """Fill the side panel, paint the avatar slot, then cue, insight, and ask lines."""
    if panel.w <= 1 or panel.h <= 1:
        return
    _fill_panel(image, panel, theme)
    _draw_border(image, panel, theme)
    avatar_widget.draw(
        image,
        (avatar_box.x, avatar_box.y, avatar_box.w, avatar_box.h),
        avatar_state,
    )
    _draw_bullets(
        image, panel, avatar_box, bullets, theme, insights or [], ask_rows or []
    )
    _draw_coaching_badge(image, panel, theme, coaching_status)


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


def draw_ask_bar(
    image: np.ndarray,
    draft: str | None,
    listening: bool,
    theme: Theme,
    content_x: int,
    content_w: int,
    height: int,
) -> None:
    """One line under the camera view. Hidden unless the user is typing or recording."""
    if draft is None and not listening:
        return
    bar_h = 22 if height >= 80 else 12
    margin = 8 if height >= 80 else 2
    y1 = max(bar_h + 1, height - (28 if height >= 80 else 16))
    y0 = y1 - bar_h
    x0 = max(0, content_x + margin)
    x1 = min(image.shape[1] - 1, content_x + content_w - margin)
    if x1 - x0 < 16 or y0 < 0 or y1 > image.shape[0]:
        return
    fill = hex_to_bgr(theme.colors.get("background", "#001018"))
    border = hex_to_bgr(theme.colors.get("primary", "#FFFFFF"))
    cv2.rectangle(image, (x0, y0), (x1, y1), fill, -1)
    cv2.rectangle(image, (x0, y0), (x1, y1), border, 1)
    if listening and draft is None:
        label = "Listening... press V to send"
    else:
        label = f"Ask Gemma: {draft}_"
    max_chars = max(8, (x1 - x0 - 8) // 7)
    if len(label) > max_chars:
        label = label[: max_chars - 1] + "_"
    cv2.putText(
        image,
        label,
        (x0 + 4, min(image.shape[0] - 2, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4 if height >= 80 else 0.3,
        border,
        1,
        cv2.LINE_8,
    )


def _draw_bullets(
    image: np.ndarray,
    panel: WidgetBox,
    avatar_box: WidgetBox,
    bullets: list[tuple[str, float]],
    theme: Theme,
    insights: list[tuple[str, float]],
    ask_rows: list[str],
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
    _badge_x, badge_y, _badge_w, _badge_h = coaching_badge_rect(panel)
    bottom = min(panel.y + panel.h - 6, badge_y - 4)
    max_chars = max(12, (panel.w - 16) // 7)
    label_color = hex_to_bgr(theme.colors.get("primary", "#FFFFFF"))
    for row in ask_rows:
        if y > bottom:
            break
        shown = row if len(row) <= max_chars else row[: max_chars - 3] + "..."
        color = (
            label_color
            if row in {"ASK", "LISTENING..."} or row.startswith("THINKING")
            else text_color
        )
        cv2.putText(
            image,
            shown,
            (x, y),
            font,
            scale,
            color,
            1,
            cv2.LINE_AA,
        )
        y += line_h
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
    if insights and y + line_h <= bottom:
        label = hex_to_bgr(theme.colors.get("primary", "#FFFFFF"))
        cv2.putText(
            image,
            "INSIGHT",
            (x, y),
            font,
            scale,
            label,
            1,
            cv2.LINE_AA,
        )
        y += line_h
        for text, alpha in insights:
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


def _draw_coaching_badge(
    image: np.ndarray, panel: WidgetBox, theme: Theme, status: str
) -> None:
    x, y, width, height = coaching_badge_rect(panel)
    frame_h, frame_w = image.shape[:2]
    x1 = min(frame_w, x + width)
    y1 = min(frame_h, y + height)
    x0 = max(0, min(x, frame_w))
    y0 = max(0, min(y, frame_h))
    if x1 <= x0 or y1 <= y0:
        return
    kind = status if status in BADGE_LABELS else "offline"
    role = {"gemma": "primary", "thinking": "warning", "offline": "neutral"}[kind]
    color = hex_to_bgr(theme.colors.get(role, "#7A8B99"))
    cv2.rectangle(image, (x0, y0), (x1 - 1, y1 - 1), color, -1)
    if height < 12 or width < 36:
        return
    label = BADGE_LABELS[kind]
    text = hex_to_bgr(theme.colors.get("background", "#000000"))
    cv2.putText(
        image,
        label,
        (x0 + 4, y1 - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        text,
        1,
        cv2.LINE_AA,
    )
