"""Short Gemma feedback bullets, newest on top, fading as they expire."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.config_loader import Theme
from src.hud.avatar import AvatarState, AvatarWidget
from src.hud.widgets import WidgetBox, hex_to_bgr
from src.intelligence.ask_gemma import answer_ttl
from src.intelligence.gemma_client import normalize_bullets


@dataclass
class AskOverlay:
    """Direct question drawn under the cue lines. It outlives a transient bullet."""

    question: str
    phase: str
    lines: tuple[str, ...]
    elapsed: int


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
        self._ask_lines: tuple[str, ...] = ()
        self._ask_started: float | None = None
        self._ask_signature: tuple | None = None
        self._ask_at = 0.0

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
        started_at: float | None,
        now: float,
    ) -> None:
        """Remember the latest ask. Repeating the same reply does not reset its clock."""
        signature = (phase, question, tuple(lines or ()))
        if signature != self._ask_signature:
            self._ask_signature = signature
            self._ask_at = float(now)
        self._ask_question = question
        self._ask_phase = phase
        self._ask_lines = tuple(lines or ())
        self._ask_started = started_at

    def visible_ask(self, now: float) -> AskOverlay | None:
        phase = self._ask_phase
        if phase == "idle":
            return None
        if phase != "thinking" and float(now) - self._ask_at >= answer_ttl(phase):
            return None
        elapsed = 0
        if self._ask_started is not None:
            elapsed = max(0, int(float(now) - self._ask_started))
        return AskOverlay(
            question=self._ask_question,
            phase=phase,
            lines=self._ask_lines,
            elapsed=elapsed,
        )

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


BADGE_LABELS = {
    "flash": "GEMMA",
    "gemma": "GEMMA",
    "thinking": "THINKING",
    "offline": "OFFLINE",
}


def coaching_badge_rect(panel: WidgetBox) -> tuple[int, int, int, int]:
    """Small strip at the bottom of the sidebar for the insight-model status."""
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
    ask: AskOverlay | None = None,
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
    _draw_bullets(image, panel, avatar_box, bullets, theme, insights or [], ask)
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


def _draw_bullets(
    image: np.ndarray,
    panel: WidgetBox,
    avatar_box: WidgetBox,
    bullets: list[tuple[str, float]],
    theme: Theme,
    insights: list[tuple[str, float]],
    ask: AskOverlay | None = None,
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
    ask_rows = _ask_rows(ask)
    reserve = line_h * len(ask_rows)
    bottom = min(panel.y + panel.h - 6, badge_y - 4) - reserve
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
    y_ask = bottom + line_h
    for row in ask_rows:
        if y_ask > badge_y - 2:
            break
        cv2.putText(
            image,
            row,
            (x, y_ask),
            font,
            scale,
            text_color,
            1,
            cv2.LINE_AA,
        )
        y_ask += line_h


def _ask_rows(ask: AskOverlay | None) -> list[str]:
    if ask is None:
        return []
    rows: list[str] = []
    if ask.question:
        question = ask.question
        if len(question) > 48:
            question = question[:47] + "..."
        rows.append(f"YOU: {question}")
    if ask.phase == "thinking":
        rows.append(f"THINKING... {ask.elapsed}s")
        return rows
    rows.append("ASK")
    rows.extend(ask.lines[:3])
    return rows


def draw_ask_bar(image: np.ndarray, text: str, listening: bool, theme: Theme) -> None:
    """One line at the bottom of the preview. The frame loop does not wait on it."""
    height, width = image.shape[:2]
    if height < 16 or width < 20:
        return
    bar_h = min(28, max(14, height // 12))
    y0 = height - bar_h
    color = hex_to_bgr(theme.colors.get("background", "#001018"))
    accent = hex_to_bgr(theme.colors.get("primary", "#FFFFFF"))
    ink = hex_to_bgr(theme.colors.get("text", "#FFFFFF"))
    cv2.rectangle(image, (0, y0), (width - 1, height - 1), color, -1)
    cv2.line(image, (0, y0), (width - 1, y0), accent, 1, cv2.LINE_AA)
    if listening:
        label = "Listening... press V to send"
    else:
        label = f"Ask Gemma: {text}_"
    cv2.putText(
        image,
        label[: max(8, width // 8)],
        (8, height - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        ink,
        1,
        cv2.LINE_AA,
    )


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
    role = {
        "flash": "primary",
        "gemma": "primary",
        "thinking": "warning",
        "offline": "neutral",
    }[kind]
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
