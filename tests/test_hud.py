import numpy as np
import pytest

from src.config_loader import load_config
from src.hud.renderer import HudRenderer, HudState
from src.hud.vignette import (
    apply_vignette,
    edge_mask,
    vignette_color,
    vignette_strength,
)
from src.hud.widgets import (
    format_timer,
    hex_to_bgr,
    layout_widgets,
    pygame_surface_size,
    widget_names_for_cue,
)
from src.intelligence.gemma_client import make_cue


def test_timer_format_and_colors():
    assert format_timer(125) == "02:05"
    assert format_timer(0) == "00:00"
    assert format_timer(59.9) == "00:59"
    assert hex_to_bgr("#00F0FF") == (255, 240, 0)
    assert hex_to_bgr("#FF003C") == (60, 0, 255)


def test_widget_boxes_stay_inside_the_frame():
    names = ["timer", "posture", "gaze", "scan", "volume", "pace", "audience"]
    for width, height in ((1280, 720), (640, 360), (320, 180)):
        boxes = layout_widgets(width, height, names)
        assert [box.name for box in boxes] == names
        assert all(box.inside(width, height) for box in boxes)


def test_each_coaching_cue_selects_its_widget():
    assert widget_names_for_cue(None) == ["timer"]
    assert "posture" in widget_names_for_cue(
        make_cue("HIGH", "POSTURE", "Straighten Posture", "arrow_up")
    )
    assert "gaze" in widget_names_for_cue(
        make_cue("MEDIUM", "GAZE", "Make eye contact", "target")
    )
    assert "scan" in widget_names_for_cue(
        make_cue("MEDIUM", "GAZE", "Look right", "arrow_right")
    )
    assert "volume" in widget_names_for_cue(
        make_cue("HIGH", "VOLUME", "Speak up", "volume_up")
    )
    assert "pace" in widget_names_for_cue(
        make_cue("MEDIUM", "PACE", "Slow down", "slow_down")
    )


def test_vignette_alpha_grows_as_time_runs_out_and_falls_off_toward_center():
    normal, phase = vignette_strength(0.80)
    warning, warning_phase = vignette_strength(0.15, pulse_t=0.0)
    critical, critical_phase = vignette_strength(0.02, pulse_t=0.0)
    assert normal == 0.0 and phase == "none"
    assert warning_phase == "warning"
    assert critical_phase == "critical"
    assert critical > warning > normal

    mask = edge_mask(120, 160)
    center = float(mask[60, 80])
    edge = float(mask[60, 0])
    corner = float(mask[0, 0])
    assert center < edge <= corner


def test_warning_and_critical_hues_follow_the_theme():
    config = load_config()
    ironman = config.theme("ironman")
    assert vignette_color(ironman, "warning") == hex_to_bgr("#FFB800")
    assert vignette_color(ironman, "critical") == hex_to_bgr("#FF003C")
    frame = np.full((90, 120, 3), 128, np.uint8)
    warning = apply_vignette(frame, 0.12, ironman, pulse_t=0.0)
    critical = apply_vignette(frame, 0.0, ironman, pulse_t=0.0)
    # Amber adds green and red at the corner. Critical red drops green.
    assert int(warning[0, 0, 1]) > int(frame[0, 0, 1])
    assert int(critical[0, 0, 2]) > int(warning[0, 0, 2])
    assert int(critical[0, 0, 1]) < int(warning[0, 0, 1])
    # Spatial decay: the center stays closer to the original gray.
    assert abs(int(critical[45, 60, 2]) - 128) < abs(int(critical[0, 0, 2]) - 128)


def test_renderer_draws_a_posture_cue_inside_the_frame():
    config = load_config()
    renderer = HudRenderer(config.theme("ironman"), opacity=1.0)
    frame = np.zeros((360, 640, 3), np.uint8)
    cue = make_cue("HIGH", "POSTURE", "Straighten Posture", "arrow_up")
    rendered = renderer.render(
        frame,
        HudState(
            remaining_seconds=125,
            remaining_ratio=0.9,
            cue=cue,
            volume_db=60,
            wpm=140,
        ),
    )
    assert rendered.shape == frame.shape
    assert int(rendered.sum()) > 0
    posture = next(box for box in layout_widgets(640, 360, ["posture"]))
    region = rendered[
        posture.y : posture.y + posture.h, posture.x : posture.x + posture.w
    ]
    arrow = region[: max(1, region.shape[0] // 2)]
    # Ironman primary #00F0FF is bright in blue and green and dark in red.
    assert int(arrow[:, :, 0].max()) > 100
    assert int(arrow[:, :, 1].max()) > 100
    assert int(arrow[:, :, 2].max()) < 80


def test_minimalist_theme_switches_the_palette():
    config = load_config()
    renderer = HudRenderer(config.theme("minimalist"), opacity=1.0)
    frame = np.zeros((180, 320, 3), np.uint8)
    cue = make_cue("HIGH", "VOLUME", "Speak up", "volume_up")
    rendered = renderer.render(
        frame,
        HudState(
            remaining_seconds=10, remaining_ratio=0.5, cue=cue, volume_db=30, wpm=100
        ),
    )
    assert rendered.shape == frame.shape
    assert int(rendered.max()) > 200


def test_pygame_surface_matches_the_frame_size():
    pytest.importorskip("pygame")
    frame = np.zeros((40, 64, 3), np.uint8)
    frame[:, :, 2] = 255
    assert pygame_surface_size(frame) == (64, 40)
