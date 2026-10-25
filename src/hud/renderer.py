"""Composite the vignette and widgets onto a camera frame.

OpenCV draws the overlay so the demo can run without a window. Pygame is used
only when a preview is requested with ``--display pygame``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.config_loader import Theme
from src.hud.vignette import apply_vignette
from src.hud.widgets import draw_widgets, layout_widgets, widget_names_for_cue


@dataclass
class HudState:
    remaining_seconds: float
    remaining_ratio: float
    cue: dict | None
    volume_db: float
    wpm: float
    pulse_t: float = 0.0


class HudRenderer:
    def __init__(
        self,
        theme: Theme,
        opacity: float | None = None,
        warning_ratio: float = 0.25,
        critical_ratio: float = 0.10,
    ) -> None:
        self.theme = theme
        self.opacity = theme.opacity if opacity is None else opacity
        self.warning_ratio = warning_ratio
        self.critical_ratio = critical_ratio

    def set_theme(self, theme: Theme) -> None:
        self.theme = theme
        self.opacity = theme.opacity

    def render(self, frame: np.ndarray, state: HudState) -> np.ndarray:
        canvas = apply_vignette(
            frame,
            state.remaining_ratio,
            self.theme,
            pulse_t=state.pulse_t,
            warning_ratio=self.warning_ratio,
            critical_ratio=self.critical_ratio,
        )
        names = widget_names_for_cue(state.cue)
        boxes = layout_widgets(canvas.shape[1], canvas.shape[0], names)
        draw_widgets(
            canvas,
            boxes,
            self.theme,
            state.cue,
            state.remaining_seconds,
            state.volume_db,
        )
        if self.opacity >= 0.999:
            return canvas
        return cv2.addWeighted(canvas, self.opacity, frame, 1.0 - self.opacity, 0.0)


class PreviewWindow:
    """Optional on-screen preview. Headless demo mode never constructs one."""

    def __init__(self, backend: str = "opencv", title: str = "Wingman") -> None:
        self.backend = backend
        self.title = title
        self._pygame = None
        self._screen = None

    def show(self, frame: np.ndarray) -> str | None:
        """Show a frame. Return a hotkey name, 'quit', or None."""
        try:
            if self.backend == "pygame":
                return self._show_pygame(frame)
            return self._show_opencv(frame)
        except Exception:
            return None

    def _show_opencv(self, frame: np.ndarray) -> str | None:
        cv2.imshow(self.title, frame)
        key = cv2.waitKey(1) & 0xFF
        return _key_name(key)

    def _show_pygame(self, frame: np.ndarray) -> str | None:
        import pygame

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        if self._screen is None:
            pygame.init()
            self._screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption(self.title)
        surface = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
        self._screen.blit(surface, (0, 0))
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.KEYDOWN:
                name = pygame.key.name(event.key)
                if name in {"escape", "q"}:
                    return "quit"
                return name
        return None

    def close(self) -> None:
        if self.backend == "pygame" and self._screen is not None:
            import pygame

            pygame.display.quit()
            self._screen = None
        elif self.backend == "opencv":
            cv2.destroyAllWindows()


def _key_name(key: int) -> str | None:
    if key in (255, -1):
        return None
    mapping = {
        32: "space",
        13: "enter",
        27: "quit",
        ord("q"): "quit",
        ord("t"): "t",
        ord("m"): "m",
        ord("s"): "s",
        ord("["): "[",
        ord("]"): "]",
        ord("+"): "+",
        ord("-"): "-",
        ord("="): "=",
    }
    return mapping.get(key)
