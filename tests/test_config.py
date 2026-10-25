from pathlib import Path

import pytest

from src.config_loader import load_config


def test_default_config_loads_plan_thresholds():
    config = load_config()
    assert config.posture.slouch_degrees == 12.0
    assert config.posture.ear_forward_offset == 0.15
    assert config.gaze.away_seconds == 4.0
    assert config.volume.speak_up_db == 42.0
    assert config.volume.quiet_duration_seconds == 3.0
    assert config.pace.rush_wpm == 180
    assert config.pace.target_wpm_min == 130
    assert config.pace.target_wpm_max == 160
    assert config.pace.continuous_speech_seconds == 8.0
    assert config.cues.cooldown_seconds == 5.0
    assert config.cues.snooze_seconds == 30.0
    assert config.cues.duration_ms == 3000
    assert config.timer.warning_ratio == 0.25
    assert config.timer.critical_ratio == 0.10
    assert config.timer.duration_seconds == 300
    assert config.audio.sample_rate == 16000
    assert config.hud.fps == 30
    assert config.gemma.model == "gemma-4-26b-a4b-it"
    assert config.gemma.min_interval_seconds == 2.5
    assert config.gemma.timeout_seconds == 8
    assert config.hud.theme == "ironman"
    assert config.hud.sidebar.side == "right"
    assert config.hud.sidebar.max_bullets == 4
    assert config.hud.sidebar.ttl_seconds == 12
    assert config.hud.avatar.anchor == "sidebar_top"
    assert config.hud.avatar.height_ratio == 0.24


def test_themes_match_the_spec_palettes():
    config = load_config()
    ironman = config.theme("ironman")
    minimalist = config.theme("minimalist")
    assert ironman.colors["primary"] == "#00F0FF"
    assert ironman.colors["warning"] == "#FFB800"
    assert ironman.colors["alert"] == "#FF003C"
    assert ironman.brackets is True
    assert ironman.font == "monospace"
    assert minimalist.colors["primary"] == "#FFFFFF"
    assert minimalist.colors["warning"] == "#FFA500"
    assert minimalist.colors["alert"] == "#FF3B30"
    assert minimalist.brackets is False


def test_load_config_is_independent_of_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = load_config()
    assert config.gemma.model == "gemma-4-26b-a4b-it"


def test_unknown_theme_is_rejected(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("hud:\n  theme: missing\n", encoding="utf-8")
    themes = tmp_path / "themes.yaml"
    themes.write_text(
        "ironman:\n  name: Ironman\n  colors:\n    primary: '#00F0FF'\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing"):
        load_config(config_path, themes)


def test_sidebar_side_must_be_left_or_right(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "hud:\n  theme: ironman\n  sidebar:\n    side: middle\n",
        encoding="utf-8",
    )
    themes = tmp_path / "themes.yaml"
    themes.write_text(
        "ironman:\n  name: Ironman\n  colors:\n    primary: '#00F0FF'\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sidebar"):
        load_config(config_path, themes)


def test_missing_config_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config(Path("/tmp/does-not-exist-wingman.yaml"))
