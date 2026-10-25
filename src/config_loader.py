"""Load Wingman YAML config and HUD themes."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, get_type_hints

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default_config.yaml"
DEFAULT_THEMES_PATH = REPO_ROOT / "config" / "themes.yaml"


@dataclasses.dataclass
class TimerConfig:
    duration_seconds: float = 300.0
    warning_ratio: float = 0.25
    critical_ratio: float = 0.10


@dataclasses.dataclass
class PostureConfig:
    slouch_degrees: float = 12.0
    ear_forward_offset: float = 0.15
    calibrate_seconds: float = 3.0
    slouch_margin_degrees: float = 8.0
    forward_margin: float = 0.08


@dataclasses.dataclass
class GazeConfig:
    away_seconds: float = 4.0
    off_center: float = 0.18
    look_down: float = 0.65
    side_bias_ratio: float = 0.75
    history_seconds: float = 6.0


@dataclasses.dataclass
class AudienceConfig:
    brow_gap_ratio: float = 0.22
    eye_aspect_wide: float = 0.42
    brow_raise_ratio: float = 0.18


@dataclasses.dataclass
class VolumeConfig:
    speak_up_db: float = 42.0
    quiet_duration_seconds: float = 3.0
    reference_amplitude: float = 1.0e-4
    level_window_seconds: float = 0.5


@dataclasses.dataclass
class PaceConfig:
    target_wpm_min: float = 130.0
    target_wpm_max: float = 160.0
    rush_wpm: float = 180.0
    continuous_speech_seconds: float = 8.0
    syllables_per_word: float = 1.5
    pause_seconds: float = 0.35
    vad_rms: float = 0.02
    window_seconds: float = 2.5


@dataclasses.dataclass
class CueConfig:
    cooldown_seconds: float = 5.0
    snooze_seconds: float = 30.0
    duration_ms: int = 3000
    global_gap_seconds: float = 1.0


@dataclasses.dataclass
class SidebarConfig:
    side: str = "right"
    width_ratio: float = 0.22
    max_bullets: int = 4
    ttl_seconds: float = 12.0
    fade_seconds: float = 3.0


@dataclasses.dataclass
class AvatarConfig:
    anchor: str = "sidebar_top"
    height_ratio: float = 0.24


@dataclasses.dataclass
class HudConfig:
    theme: str = "ironman"
    opacity: float = 0.85
    width: int = 1280
    height: int = 720
    fps: int = 30
    sidebar: SidebarConfig = dataclasses.field(default_factory=SidebarConfig)
    avatar: AvatarConfig = dataclasses.field(default_factory=AvatarConfig)


@dataclasses.dataclass
class AudioStreamConfig:
    sample_rate: int = 16000
    channels: int = 1
    block_size: int = 256
    buffer_seconds: float = 2.5


@dataclasses.dataclass
class InsightConfig:
    """Fast periodic trends. Separate from the slow Gemma client."""

    model: str = "gemini-3.5-flash-lite"
    min_interval_seconds: float = 8.0
    timeout_seconds: float = 10.0
    max_output_tokens: int = 256
    thinking_level: str = "minimal"


@dataclasses.dataclass
class GemmaConfig:
    """Slow report and Ask Gemma settings. The insight loop does not read these."""

    model: str = "gemma-4-26b-a4b-it"
    min_interval_seconds: float = 18.0
    timeout_seconds: float = 45.0
    report_timeout_seconds: float = 90.0
    max_output_tokens: int = 2048


@dataclasses.dataclass
class Theme:
    key: str
    name: str
    font: str
    brackets: bool
    opacity: float
    colors: dict[str, str]


@dataclasses.dataclass
class AppConfig:
    timer: TimerConfig = dataclasses.field(default_factory=TimerConfig)
    posture: PostureConfig = dataclasses.field(default_factory=PostureConfig)
    gaze: GazeConfig = dataclasses.field(default_factory=GazeConfig)
    audience: AudienceConfig = dataclasses.field(default_factory=AudienceConfig)
    volume: VolumeConfig = dataclasses.field(default_factory=VolumeConfig)
    pace: PaceConfig = dataclasses.field(default_factory=PaceConfig)
    cues: CueConfig = dataclasses.field(default_factory=CueConfig)
    hud: HudConfig = dataclasses.field(default_factory=HudConfig)
    audio: AudioStreamConfig = dataclasses.field(default_factory=AudioStreamConfig)
    insight: InsightConfig = dataclasses.field(default_factory=InsightConfig)
    gemma: GemmaConfig = dataclasses.field(default_factory=GemmaConfig)
    themes: dict[str, Theme] = dataclasses.field(default_factory=dict)

    def theme(self, name: str | None = None) -> Theme:
        key = name or self.hud.theme
        try:
            return self.themes[key]
        except KeyError as exc:
            known = ", ".join(sorted(self.themes)) or "(none)"
            raise KeyError(f"Unknown HUD theme '{key}'. Known themes: {known}") from exc


def _apply(instance: Any, raw: dict[str, Any]) -> None:
    hints = get_type_hints(type(instance))
    for key, value in raw.items():
        if key not in hints:
            continue
        current = getattr(instance, key)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            _apply(current, value)
        else:
            setattr(instance, key, value)


def load_themes(path: Path | None = None) -> dict[str, Theme]:
    theme_path = Path(path) if path else DEFAULT_THEMES_PATH
    raw = yaml.safe_load(theme_path.read_text(encoding="utf-8")) or {}
    themes: dict[str, Theme] = {}
    for key, value in raw.items():
        colors = dict(value.get("colors", {}))
        themes[key] = Theme(
            key=key,
            name=str(value.get("name", key)),
            font=str(value.get("font", "sans")),
            brackets=bool(value.get("brackets", False)),
            opacity=float(value.get("opacity", 0.85)),
            colors=colors,
        )
    return themes


def load_config(
    path: Path | None = None,
    themes_path: Path | None = None,
) -> AppConfig:
    """Load thresholds and themes. Paths default to the repo config directory."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must be a mapping: {config_path}")
    config = AppConfig()
    _apply(config, raw)
    config.themes = load_themes(themes_path)
    _validate(config)
    return config


def _validate(config: AppConfig) -> None:
    if config.hud.theme not in config.themes:
        raise ValueError(
            f"hud.theme '{config.hud.theme}' is not defined in the themes file"
        )
    if config.timer.warning_ratio <= config.timer.critical_ratio:
        raise ValueError("timer.warning_ratio must be greater than critical_ratio")
    if config.audio.sample_rate <= 0 or config.audio.block_size <= 0:
        raise ValueError("audio sample_rate and block_size must be positive")
    if config.hud.fps <= 0:
        raise ValueError("hud.fps must be positive")
    if not 0 < config.volume.reference_amplitude < 1:
        raise ValueError("volume.reference_amplitude must be between 0 and 1")
    if config.hud.sidebar.side not in {"left", "right"}:
        raise ValueError("hud.sidebar.side must be 'left' or 'right'")
    if not 0 < float(config.hud.sidebar.width_ratio) < 1:
        raise ValueError("hud.sidebar.width_ratio must be between 0 and 1")
    if int(config.hud.sidebar.max_bullets) < 1:
        raise ValueError("hud.sidebar.max_bullets must be at least 1")
    if float(config.hud.sidebar.ttl_seconds) <= 0:
        raise ValueError("hud.sidebar.ttl_seconds must be positive")
    if float(config.hud.sidebar.fade_seconds) < 0:
        raise ValueError("hud.sidebar.fade_seconds cannot be negative")
    if float(config.hud.sidebar.fade_seconds) >= float(config.hud.sidebar.ttl_seconds):
        raise ValueError("hud.sidebar.fade_seconds must be shorter than ttl_seconds")
    if float(config.insight.min_interval_seconds) <= 0:
        raise ValueError("insight.min_interval_seconds must be positive")
    if float(config.insight.timeout_seconds) < 10:
        raise ValueError(
            "insight.timeout_seconds must be at least 10 seconds; "
            "shorter deadlines are rejected by the API"
        )
    if int(config.insight.max_output_tokens) < 32:
        raise ValueError("insight.max_output_tokens must be at least 32")
    if str(config.insight.thinking_level).lower() not in {
        "minimal",
        "low",
        "medium",
        "high",
    }:
        raise ValueError("insight.thinking_level must be minimal, low, medium, or high")
    if float(config.gemma.min_interval_seconds) <= 0:
        raise ValueError("gemma.min_interval_seconds must be positive")
    if float(config.gemma.timeout_seconds) < 10:
        raise ValueError(
            "gemma.timeout_seconds must be at least 10 seconds; "
            "shorter deadlines are rejected by the API"
        )
    if float(config.gemma.report_timeout_seconds) < 10:
        raise ValueError("gemma.report_timeout_seconds must be at least 10 seconds")
    if int(config.gemma.max_output_tokens) < 512:
        raise ValueError(
            "gemma.max_output_tokens must be at least 512 so thinking tokens "
            "do not consume the whole reply"
        )
    if float(config.posture.calibrate_seconds) < 0:
        raise ValueError("posture.calibrate_seconds cannot be negative")
    if float(config.posture.slouch_margin_degrees) <= 0:
        raise ValueError("posture.slouch_margin_degrees must be positive")
    if float(config.posture.forward_margin) <= 0:
        raise ValueError("posture.forward_margin must be positive")
    if config.hud.avatar.anchor != "sidebar_top":
        raise ValueError("hud.avatar.anchor must be 'sidebar_top'")
    if not 0 < float(config.hud.avatar.height_ratio) <= 1:
        raise ValueError("hud.avatar.height_ratio must be between 0 and 1")
