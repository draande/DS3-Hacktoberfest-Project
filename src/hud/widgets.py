"""HUD widget layout and drawing. All coordinates are pixel bounds on the frame."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.config_loader import AvatarConfig, SidebarConfig, Theme

REFERENCE_WIDTH = 1280
REFERENCE_HEIGHT = 720

CUE_COLOR_ROLE = {
    "POSTURE": "primary",
    "GAZE": "primary",
    "VOLUME": "warning",
    "PACE": "warning",
    "AUDIENCE": "warning",
    "ENCOURAGE": "primary",
}


@dataclass(frozen=True)
class WidgetBox:
    name: str
    x: int
    y: int
    w: int
    h: int

    def inside(self, frame_width: int, frame_height: int) -> bool:
        return (
            self.w > 0
            and self.h > 0
            and self.x >= 0
            and self.y >= 0
            and self.x + self.w <= frame_width
            and self.y + self.h <= frame_height
        )


def hex_to_bgr(value: str) -> tuple[int, int, int]:
    text = value.strip().lstrip("#")
    if len(text) == 8:
        text = text[:6]
    if len(text) != 6:
        raise ValueError(f"Expected RRGGBB color, got {value}")
    red = int(text[0:2], 16)
    green = int(text[2:4], 16)
    blue = int(text[4:6], 16)
    return blue, green, red


def format_timer(remaining_seconds: float) -> str:
    total = max(0, int(remaining_seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


def widget_names_for_cue(cue: dict | None) -> list[str]:
    names = ["timer"]
    if not cue:
        return names
    cue_type = str(cue.get("cue_type", "")).upper()
    icon = str(cue.get("icon", ""))
    if cue_type == "POSTURE":
        names.append("posture")
    elif cue_type == "GAZE":
        names.append("gaze" if icon == "target" else "scan")
    elif cue_type == "VOLUME":
        names.append("volume")
    elif cue_type == "PACE":
        names.append("pace")
    elif cue_type == "AUDIENCE":
        names.append("audience")
    elif cue_type == "ENCOURAGE":
        names.append("pace")
    return names


def sidebar_bounds(width: int, height: int, sidebar: SidebarConfig) -> WidgetBox:
    """Pixel box for the feedback column. It always touches the chosen edge."""
    if width <= 1 or height <= 0:
        return WidgetBox("sidebar", 0, 0, max(width, 0), max(height, 0))
    ratio = min(0.9, max(0.05, float(sidebar.width_ratio)))
    panel_w = max(1, int(round(width * ratio)))
    panel_w = min(panel_w, width - 1)
    x = 0 if sidebar.side == "left" else width - panel_w
    return WidgetBox("sidebar", x, 0, panel_w, height)


def avatar_bounds(panel: WidgetBox, avatar: AvatarConfig) -> WidgetBox:
    """Top of the sidebar. The team draws the avatar inside this rectangle."""
    ratio = min(1.0, max(0.05, float(avatar.height_ratio)))
    height = max(1, int(round(panel.h * ratio)))
    height = min(height, max(1, panel.h))
    return WidgetBox("avatar", panel.x, panel.y, panel.w, height)


def layout_hud(
    width: int,
    height: int,
    names: list[str],
    sidebar: SidebarConfig,
    avatar: AvatarConfig,
) -> tuple[list[WidgetBox], WidgetBox, WidgetBox]:
    """Place coaching widgets in the area the sidebar does not cover.

    Returns ``(widgets, sidebar_panel, avatar_slot)``. Widget coordinates are
    already shifted when the sidebar sits on the left.
    """
    panel = sidebar_bounds(width, height, sidebar)
    if sidebar.side == "left":
        origin_x = panel.w
        content_w = max(1, width - panel.w)
    else:
        origin_x = 0
        content_w = max(1, width - panel.w)
    shifted = [
        WidgetBox(box.name, box.x + origin_x, box.y, box.w, box.h)
        for box in layout_widgets(content_w, height, names)
    ]
    return shifted, panel, avatar_bounds(panel, avatar)


def layout_widgets(width: int, height: int, names: list[str]) -> list[WidgetBox]:
    """Place widgets using the 1280x720 reference, then fit them to the frame."""
    scale_x = width / REFERENCE_WIDTH
    scale_y = height / REFERENCE_HEIGHT
    specs = {
        "timer": (24, 24, 180, 56),
        "volume": (1040, 24, 216, 56),
        "posture": (500, 200, 280, 90),
        "gaze": (570, 300, 140, 140),
        "scan": (980, 320, 250, 64),
        "pace": (520, 620, 240, 56),
        "audience": (24, 620, 420, 56),
    }
    boxes: list[WidgetBox] = []
    for name in names:
        if name not in specs:
            continue
        x, y, w, h = specs[name]
        box = WidgetBox(
            name,
            int(x * scale_x),
            int(y * scale_y),
            max(8, int(w * scale_x)),
            max(8, int(h * scale_y)),
        )
        boxes.append(_fit(box, width, height))
    return boxes


def _fit(box: WidgetBox, width: int, height: int, margin: int = 4) -> WidgetBox:
    margin = min(margin, max(0, width // 10), max(0, height // 10))
    max_w = max(1, width - 2 * margin)
    max_h = max(1, height - 2 * margin)
    w = min(box.w, max_w)
    h = min(box.h, max_h)
    x = min(max(box.x, margin), max(margin, width - w - margin))
    y = min(max(box.y, margin), max(margin, height - h - margin))
    return WidgetBox(box.name, int(x), int(y), int(w), int(h))


def cue_bgr(theme: Theme, cue: dict | None) -> tuple[int, int, int]:
    role = "primary"
    if cue:
        role = CUE_COLOR_ROLE.get(str(cue.get("cue_type", "")).upper(), "primary")
    return hex_to_bgr(theme.colors.get(role, theme.colors.get("primary", "#FFFFFF")))


def draw_widgets(
    image: np.ndarray,
    boxes: list[WidgetBox],
    theme: Theme,
    cue: dict | None,
    remaining_seconds: float,
    volume_db: float,
) -> None:
    color = cue_bgr(theme, cue)
    text_color = hex_to_bgr(theme.colors.get("text", "#FFFFFF"))
    font = (
        cv2.FONT_HERSHEY_COMPLEX
        if theme.font == "monospace"
        else cv2.FONT_HERSHEY_SIMPLEX
    )
    for box in boxes:
        if theme.brackets:
            _draw_brackets(image, box, color)
        if box.name == "timer":
            _draw_text(
                image, box, format_timer(remaining_seconds), text_color, font, scale=0.9
            )
        elif box.name == "posture":
            _draw_arrow(image, box, "up", color)
            _draw_text(
                image,
                box,
                _label(cue, "Straighten Posture"),
                text_color,
                font,
                scale=0.5,
            )
        elif box.name == "gaze":
            _draw_reticle(image, box, color)
            _draw_text(
                image, box, _label(cue, "Make eye contact"), text_color, font, scale=0.4
            )
        elif box.name == "scan":
            direction = "left" if cue and cue.get("icon") == "arrow_left" else "right"
            _draw_arrow(image, box, direction, color)
            _draw_text(
                image, box, _label(cue, "Scan the room"), text_color, font, scale=0.5
            )
        elif box.name == "volume":
            _draw_text(
                image, box, _label(cue, "Speak up"), text_color, font, scale=0.55
            )
            _draw_level(image, box, volume_db, color)
        elif box.name == "pace":
            _draw_text(
                image, box, _label(cue, "Slow down"), text_color, font, scale=0.6
            )
        elif box.name == "audience":
            _draw_text(
                image,
                box,
                _label(cue, "Reiterate your last point"),
                text_color,
                font,
                scale=0.45,
            )


def _label(cue: dict | None, default: str) -> str:
    if cue and cue.get("display_text"):
        return str(cue["display_text"])
    return default


def _draw_brackets(
    image: np.ndarray, box: WidgetBox, color: tuple[int, int, int]
) -> None:
    length = max(4, min(box.w, box.h) // 4)
    x, y, w, h = box.x, box.y, box.w, box.h
    points = (
        ((x, y), (x + length, y), (x, y + length)),
        ((x + w - 1, y), (x + w - 1 - length, y), (x + w - 1, y + length)),
        ((x, y + h - 1), (x + length, y + h - 1), (x, y + h - 1 - length)),
        (
            (x + w - 1, y + h - 1),
            (x + w - 1 - length, y + h - 1),
            (x + w - 1, y + h - 1 - length),
        ),
    )
    for corner, horizontal, vertical in points:
        cv2.line(image, corner, horizontal, color, 1, cv2.LINE_AA)
        cv2.line(image, corner, vertical, color, 1, cv2.LINE_AA)


def _draw_arrow(
    image: np.ndarray,
    box: WidgetBox,
    direction: str,
    color: tuple[int, int, int],
) -> None:
    if box.w < 6 or box.h < 6:
        return
    pad = max(2, min(box.w, box.h) // 6)
    cx = box.x + box.w // 2
    cy = box.y + box.h // 2
    if direction == "up":
        start = (cx, box.y + box.h - pad)
        end = (cx, box.y + pad)
    elif direction == "down":
        start = (cx, box.y + pad)
        end = (cx, box.y + box.h - pad)
    elif direction == "left":
        start = (box.x + box.w - pad, cy)
        end = (box.x + pad, cy)
    else:
        start = (box.x + pad, cy)
        end = (box.x + box.w - pad, cy)
    thickness = max(1, min(box.w, box.h) // 18)
    cv2.arrowedLine(image, start, end, color, thickness, cv2.LINE_AA, tipLength=0.3)


def _draw_reticle(
    image: np.ndarray, box: WidgetBox, color: tuple[int, int, int]
) -> None:
    cx = box.x + box.w // 2
    cy = box.y + box.h // 2
    radius = max(4, min(box.w, box.h) // 3)
    cv2.circle(image, (cx, cy), radius, color, 1, cv2.LINE_AA)
    cv2.circle(image, (cx, cy), max(1, radius // 8), color, -1, cv2.LINE_AA)
    tick = max(3, radius // 3)
    cv2.line(
        image, (cx - radius - tick, cy), (cx - radius + tick, cy), color, 1, cv2.LINE_AA
    )
    cv2.line(
        image, (cx + radius - tick, cy), (cx + radius + tick, cy), color, 1, cv2.LINE_AA
    )


LEVEL_DB_MIN = 20.0
LEVEL_DB_MAX = 80.0


def mic_meter_box(width: int, height: int, sidebar: SidebarConfig) -> WidgetBox:
    """Always-on level bar in the area the sidebar does not cover."""
    panel = sidebar_bounds(width, height, sidebar)
    margin = 8 if min(width, height) >= 80 else 4
    bar_h = 10 if height >= 80 else 6
    available = max(24, width - panel.w - 2 * margin)
    bar_w = min(160, available)
    if sidebar.side == "left":
        x = min(max(0, width - bar_w - margin), panel.w + margin)
    else:
        x = margin
    y = max(0, height - bar_h - margin)
    return WidgetBox("mic", int(x), int(y), int(bar_w), int(bar_h))


def _db_fraction(decibels: float) -> float:
    span = LEVEL_DB_MAX - LEVEL_DB_MIN
    return max(0.0, min(1.0, (float(decibels) - LEVEL_DB_MIN) / span))


def draw_mic_meter(
    image: np.ndarray,
    box: WidgetBox,
    volume_db: float,
    threshold_db: float,
    theme: Theme,
) -> None:
    """Horizontal dB bar with a marker on the speak-up line."""
    if box.w < 8 or box.h < 4:
        return
    frame_h, frame_w = image.shape[:2]
    x0 = max(0, min(frame_w - 1, box.x))
    y0 = max(0, min(frame_h - 1, box.y))
    x1 = max(x0 + 1, min(frame_w, box.x + box.w))
    y1 = max(y0 + 1, min(frame_h, box.y + box.h))
    neutral = hex_to_bgr(theme.colors.get("neutral", "#334455"))
    fill = hex_to_bgr(
        theme.colors.get(
            "warning" if volume_db < threshold_db else "primary", "#FFFFFF"
        )
    )
    marker = hex_to_bgr(theme.colors.get("alert", "#FF003C"))
    cv2.rectangle(image, (x0, y0), (x1 - 1, y1 - 1), neutral, 1, cv2.LINE_AA)
    inner_w = max(1, x1 - x0 - 2)
    filled = int(round(inner_w * _db_fraction(volume_db)))
    if filled > 0 and y1 - y0 > 2:
        cv2.rectangle(image, (x0 + 1, y0 + 1), (x0 + filled, y1 - 2), fill, -1)
    mark = x0 + 1 + int(round(inner_w * _db_fraction(threshold_db)))
    mark = max(x0 + 1, min(x1 - 2, mark))
    cv2.line(image, (mark, y0), (mark, y1 - 1), marker, 1)


def _draw_level(
    image: np.ndarray,
    box: WidgetBox,
    volume_db: float,
    color: tuple[int, int, int],
) -> None:
    span = max(4, box.w // 3)
    height = max(3, box.h // 5)
    x = box.x + box.w - span - 6
    y = box.y + box.h - height - 6
    if x < box.x or y < box.y:
        return
    cv2.rectangle(image, (x, y), (x + span, y + height), color, 1, cv2.LINE_AA)
    filled = int(span * max(0.0, min(volume_db / 80.0, 1.0)))
    if filled > 2 and y + 2 < y + height - 1:
        cv2.rectangle(image, (x + 1, y + 1), (x + filled, y + height - 1), color, -1)


def _draw_text(
    image: np.ndarray,
    box: WidgetBox,
    text: str,
    color: tuple[int, int, int],
    font: int,
    scale: float,
) -> None:
    fitted = max(0.35, scale * min(box.w / 180.0, box.h / 40.0))
    thickness = 1
    baseline_y = box.y + box.h - max(4, box.h // 8)
    origin_x = box.x + max(4, box.w // 20)
    cv2.putText(
        image,
        text,
        (origin_x, baseline_y),
        font,
        fitted,
        color,
        thickness,
        cv2.LINE_AA,
    )


def pygame_surface_size(frame_bgr: np.ndarray) -> tuple[int, int]:
    """Build an offscreen pygame surface. Does not open a window."""
    import pygame

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    surface = pygame.image.frombuffer(rgb.tobytes(), (width, height), "RGB")
    return surface.get_size()
