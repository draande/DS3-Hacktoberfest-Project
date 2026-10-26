import numpy as np
import pytest

from src.config_loader import SidebarConfig, load_config
from src.hud.avatar import AvatarState, draw_avatar
from src.hud.renderer import HudRenderer, HudState
from src.hud.sidebar import BADGE_LABELS, FeedbackSidebar, coaching_badge_rect
from src.hud.vignette import (
    apply_vignette,
    edge_mask,
    vignette_color,
    vignette_strength,
)
from src.hud.widgets import (
    format_timer,
    hex_to_bgr,
    layout_hud,
    layout_widgets,
    mic_meter_box,
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
    boxes, _panel, _avatar = layout_hud(
        640,
        360,
        ["posture"],
        renderer.sidebar_config,
        renderer.avatar_config,
    )
    posture = next(box for box in boxes)
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


def test_sidebar_keeps_newest_on_top_and_fades_them_out():
    bar = FeedbackSidebar(max_bullets=4, ttl_seconds=10, fade_seconds=4)
    bar.push(["First note here", "Second note here"], 0.0)
    assert [text for text, _alpha in bar.visible(0.0)] == [
        "First note here",
        "Second note here",
    ]
    bar.push(["Third note here"], 1.0)
    assert bar.visible(1.0)[0][0] == "Third note here"
    bar.push(["Second note here"], 2.0)
    visible = bar.visible(2.0)
    assert visible[0] == ("Second note here", 1.0)
    assert bar.visible(9.0)[0][1] == pytest.approx(0.75)
    assert bar.visible(12.0) == []


def test_sidebar_caps_visible_bullets_at_four_even_if_config_is_larger():
    bar = FeedbackSidebar(max_bullets=8, ttl_seconds=30, fade_seconds=1)
    assert bar.max_bullets == 4
    bar.push(
        ["one two three", "four five six", "seven eight nine", "ten eleven twelve"],
        0.0,
    )
    bar.push(["thirteen fourteen fifteen"], 1.0)
    visible = [text for text, _alpha in bar.visible(1.0)]
    assert visible[0] == "thirteen fourteen fifteen"
    assert len(visible) == 4
    assert "ten eleven twelve" not in visible


def test_avatar_slot_is_the_top_of_the_sidebar_and_widgets_stay_clear():
    config = load_config()
    names = ["timer", "posture", "volume", "scan"]
    boxes, panel, avatar = layout_hud(
        640, 360, names, config.hud.sidebar, config.hud.avatar
    )
    assert panel.x + panel.w == 640
    assert avatar.x == panel.x and avatar.y == 0
    assert avatar.w == panel.w
    assert 0 < avatar.h < panel.h
    assert avatar.inside(640, 360)
    for box in boxes:
        assert box.inside(640, 360)
        assert box.x + box.w <= panel.x

    left = SidebarConfig(side="left", width_ratio=0.22)
    left_boxes, left_panel, _left_avatar = layout_hud(
        640, 360, names, left, config.hud.avatar
    )
    assert left_panel.x == 0
    for box in left_boxes:
        assert box.x >= left_panel.w

    frame = np.zeros((360, 640, 3), np.uint8)
    draw_avatar(
        frame,
        (avatar.x, avatar.y, avatar.w, avatar.h),
        AvatarState("ironman", "POSTURE", "Straighten Posture", False, 20, 140, 100),
    )
    region = frame[avatar.y : avatar.y + avatar.h, avatar.x : avatar.x + avatar.w]
    assert int(region.sum()) > 0
    assert int(frame[:, : panel.x].sum()) == 0


def test_renderer_paints_the_sidebar_for_both_themes():
    config = load_config()
    cue = make_cue("HIGH", "POSTURE", "Straighten Posture", "arrow_up")
    for theme_name in ("ironman", "minimalist"):
        renderer = HudRenderer(
            config.theme(theme_name),
            opacity=1.0,
            sidebar_config=config.hud.sidebar,
            avatar_config=config.hud.avatar,
        )
        renderer.sidebar.push(cue["bullets"], 0.0)
        rendered = renderer.render(
            np.zeros((180, 320, 3), np.uint8),
            HudState(
                remaining_seconds=10,
                remaining_ratio=0.5,
                cue=cue,
                volume_db=30,
                wpm=140,
                pulse_t=0.0,
            ),
        )
        _boxes, panel, _avatar = layout_hud(
            320, 180, ["posture"], renderer.sidebar_config, renderer.avatar_config
        )
        strip = rendered[:, panel.x : panel.x + panel.w]
        assert int(strip.max()) > 0


def test_coaching_badge_reads_gemma_thinking_or_offline():
    assert BADGE_LABELS["flash"] == "GEMMA"
    assert BADGE_LABELS["gemma"] == "GEMMA"
    assert BADGE_LABELS["thinking"] == "THINKING"
    assert BADGE_LABELS["offline"] == "OFFLINE"
    config = load_config()
    theme = config.theme("ironman")
    renderer = HudRenderer(
        theme,
        opacity=1.0,
        sidebar_config=config.hud.sidebar,
        avatar_config=config.hud.avatar,
    )
    frame = np.zeros((180, 320, 3), np.uint8)
    kwargs = dict(
        remaining_seconds=10, remaining_ratio=0.8, cue=None, volume_db=50, wpm=140
    )
    live = renderer.render(frame.copy(), HudState(**kwargs, coaching_status="flash"))
    thinking = renderer.render(
        frame.copy(), HudState(**kwargs, coaching_status="thinking")
    )
    offline = renderer.render(
        frame.copy(), HudState(**kwargs, coaching_status="offline")
    )
    _boxes, panel, _avatar = layout_hud(
        320, 180, ["timer"], renderer.sidebar_config, renderer.avatar_config
    )
    x, y, _w, _h = coaching_badge_rect(panel)
    assert tuple(int(channel) for channel in live[y, x + 2]) == hex_to_bgr(
        theme.colors["primary"]
    )
    assert tuple(int(channel) for channel in thinking[y, x + 2]) == hex_to_bgr(
        theme.colors["warning"]
    )
    assert tuple(int(channel) for channel in offline[y, x + 2]) == hex_to_bgr(
        theme.colors["neutral"]
    )


def test_mic_meter_shows_level_and_the_speak_up_line():
    config = load_config()
    theme = config.theme("ironman")
    renderer = HudRenderer(
        theme,
        opacity=1.0,
        sidebar_config=config.hud.sidebar,
        avatar_config=config.hud.avatar,
    )
    kwargs = dict(
        remaining_seconds=10, remaining_ratio=0.8, cue=None, wpm=140, pulse_t=0.0
    )
    quiet = renderer.render(
        np.zeros((180, 320, 3), np.uint8), HudState(**kwargs, volume_db=30)
    )
    loud = renderer.render(
        np.zeros((180, 320, 3), np.uint8), HudState(**kwargs, volume_db=70)
    )
    box = mic_meter_box(320, 180, config.hud.sidebar)
    mid = box.x + box.w // 2
    assert int(quiet[box.y + 2, mid].sum()) == 0
    assert int(loud[box.y + 2, mid].sum()) > 0
    span = 60.0
    mark = box.x + 1 + int(round((box.w - 2) * (42.0 - 20.0) / span))
    assert tuple(int(channel) for channel in quiet[box.y + 1, mark]) == hex_to_bgr(
        theme.colors["alert"]
    )


def test_pygame_surface_matches_the_frame_size():
    pytest.importorskip("pygame")
    frame = np.zeros((40, 64, 3), np.uint8)
    frame[:, :, 2] = 255
    assert pygame_surface_size(frame) == (64, 40)
