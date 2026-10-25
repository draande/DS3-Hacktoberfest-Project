"""Composite the vignette and widgets onto a camera frame.

OpenCV draws the overlay so the demo can run without a window. Pygame is used
only when a preview is requested with ``--display pygame``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.config_loader import AvatarConfig, SidebarConfig, Theme
from src.hud.avatar import AvatarState, AvatarWidget
from src.hud.display import configure_preview_window
from src.hud.sidebar import FeedbackSidebar, draw_sidebar
from src.hud.vignette import apply_vignette
from src.hud.widgets import draw_widgets, layout_hud, widget_names_for_cue


@dataclass
class HudState:
    remaining_seconds: float
    remaining_ratio: float
    cue: dict | None
    volume_db: float
    wpm: float
    pulse_t: float = 0.0
    coaching_live: bool = False


class HudRenderer:
    def __init__(
        self,
        theme: Theme,
        opacity: float | None = None,
        warning_ratio: float = 0.25,
        critical_ratio: float = 0.10,
        sidebar_config: SidebarConfig | None = None,
        avatar_config: AvatarConfig | None = None,
        sidebar: FeedbackSidebar | None = None,
        avatar_widget: AvatarWidget | None = None,
    ) -> None:
        self.theme = theme
        self.opacity = theme.opacity if opacity is None else opacity
        self.warning_ratio = warning_ratio
        self.critical_ratio = critical_ratio
        self.sidebar_config = sidebar_config or SidebarConfig()
        self.avatar_config = avatar_config or AvatarConfig()
        self.sidebar = sidebar or FeedbackSidebar(
            max_bullets=self.sidebar_config.max_bullets,
            ttl_seconds=self.sidebar_config.ttl_seconds,
            fade_seconds=self.sidebar_config.fade_seconds,
        )
        self.avatar_widget = avatar_widget or AvatarWidget()

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
        boxes, panel, avatar_box = layout_hud(
            canvas.shape[1],
            canvas.shape[0],
            names,
            self.sidebar_config,
            self.avatar_config,
        )
        draw_widgets(
            canvas,
            boxes,
            self.theme,
            state.cue,
            state.remaining_seconds,
            state.volume_db,
        )
        cue = state.cue or {}
        draw_sidebar(
            canvas,
            panel,
            avatar_box,
            self.sidebar.visible(state.pulse_t),
            self.theme,
            AvatarState(
                theme_name=self.theme.key,
                cue_type=None if not cue else str(cue.get("cue_type")),
                display_text=None if not cue else str(cue.get("display_text") or ""),
                speaking=state.volume_db >= 42.0,
                volume_db=state.volume_db,
                wpm=state.wpm,
                remaining_seconds=state.remaining_seconds,
            ),
            self.avatar_widget,
            coaching_live=state.coaching_live,
        )
        if self.opacity >= 0.999:
            return canvas
        return cv2.addWeighted(canvas, self.opacity, frame, 1.0 - self.opacity, 0.0)


class PreviewWindow:
    """Optional on-screen preview. Headless demo mode never constructs one.

    ``--display`` starts fullscreen at ``screen_size`` when that is known.
    ``windowed_size`` is the size used after Esc leaves fullscreen, and the
    size used when the screen cannot be detected. F toggles. Esc leaves
    fullscreen first and quits only once the window is already windowed.
    """

    def __init__(
        self,
        backend: str = "opencv",
        title: str = "Wingman",
        screen_size: tuple[int, int] | None = None,
        windowed_size: tuple[int, int] = (1280, 720),
        fullscreen: bool = False,
        cv2_module=cv2,
    ) -> None:
        self.backend = backend
        self.title = title
        self.screen_size = screen_size
        self.windowed_size = windowed_size
        self.fullscreen = fullscreen
        self._cv2 = cv2_module
        self._pygame = None
        self._screen = None
        self._applied: tuple[bool, tuple[int, int]] | None = None

    def target_size(self) -> tuple[int, int]:
        if self.fullscreen and self.screen_size is not None:
            return self.screen_size
        return self.windowed_size

    def interpret_key(self, name: str | None) -> str | None:
        """Map a key name to quit, a fullscreen action, or the key itself.

        This does not touch OpenCV, so tests can drive it directly.
        """
        if not name:
            return None
        key = name.lower()
        if key == "f":
            self.fullscreen = not self.fullscreen
            return "fullscreen" if self.fullscreen else "exit_fullscreen"
        if key == "escape":
            if self.fullscreen:
                self.fullscreen = False
                return "exit_fullscreen"
            return "quit"
        if key in {"q", "quit"}:
            return "quit"
        return key

    def show(self, frame: np.ndarray) -> str | None:
        """Show a frame. Return a hotkey name, 'quit', or None."""
        try:
            if self.backend == "pygame":
                return self._show_pygame(frame)
            return self._show_opencv(frame)
        except Exception:
            return None

    def _show_opencv(self, frame: np.ndarray) -> str | None:
        height, width = frame.shape[:2]
        target = self.target_size()
        matches = (width, height) == target
        mode = (self.fullscreen, target)
        if matches and self._applied != mode:
            configure_preview_window(self.title, target, self.fullscreen, self._cv2)
            self._applied = mode
        self._cv2.imshow(self.title, frame)
        if matches and self.fullscreen:
            self._cv2.setWindowProperty(
                self.title,
                self._cv2.WND_PROP_FULLSCREEN,
                self._cv2.WINDOW_FULLSCREEN,
            )
        key = self._cv2.waitKey(1) & 0xFF
        return self.interpret_key(_key_name(key))

    def _show_pygame(self, frame: np.ndarray) -> str | None:
        import pygame

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        target = self.target_size()
        matches = (width, height) == target
        mode = (self.fullscreen, target)
        if self._screen is None or (matches and self._applied != mode):
            pygame.init()
            flags = pygame.FULLSCREEN if self.fullscreen and matches else 0
            size = target if matches else (width, height)
            self._screen = pygame.display.set_mode(size, flags)
            pygame.display.set_caption(self.title)
            if matches:
                self._applied = mode
        surface = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
        self._screen.blit(surface, (0, 0))
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.KEYDOWN:
                return self.interpret_key(pygame.key.name(event.key))
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
        27: "escape",
        ord("q"): "q",
        ord("f"): "f",
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
