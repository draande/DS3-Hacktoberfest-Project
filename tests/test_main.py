import time

import numpy as np

from main import main
from src.intelligence.gemma_client import GemmaClient
from src.main import RunSummary, format_summary, parse_args, run_session


def test_main_banner_and_short_headless_demo(capsys):
    code = main(
        [
            "--offline",
            "--frames",
            "4",
            "--width",
            "160",
            "--height",
            "90",
            "--fps",
            "10",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "Hello from Wingman!" in captured.out
    assert "Wingman demo" in captured.out
    assert "coaching=offline" in captured.out


def test_summary_reports_gemma_success_and_failure_counts():
    text = format_summary(RunSummary(fallback=False, gemma_ok=2, gemma_failed=1))
    assert "coaching=gemma" in text
    assert "gemma_ok=2" in text
    assert "gemma_failed=1" in text


def test_demo_loop_does_not_wait_for_a_slow_gemma_call():
    def slow(_prompt: str) -> str:
        time.sleep(1.5)
        return (
            '{"priority":"LOW","cue_type":"ENCOURAGE","display_text":"Great pace",'
            '"icon":"ok","duration_ms":2000,"audio_chime":false}'
        )

    client = GemmaClient(
        api_key="test-key", transport=slow, min_interval=2.5, timeout=5
    )
    started = time.perf_counter()
    summary = run_session(
        parse_args(["--frames", "6", "--fps", "10", "--width", "80", "--height", "48"]),
        client=client,
    )
    elapsed = time.perf_counter() - started
    assert elapsed < 1.0
    assert summary.frames == 6
    assert summary.fallback is True
    assert "POSTURE" in summary.cue_types


def test_parse_args_defaults_to_demo_and_live_camera_opts_out():
    demo = parse_args([])
    assert demo.demo is True
    assert demo.display is False
    live = parse_args(["--camera", "0"])
    assert live.demo is False
    assert live.camera == 0
    both = parse_args(["--camera", "1", "--demo"])
    assert both.demo is True


def test_demo_emits_the_coaching_sequence_and_a_palm_snooze():
    summary = run_session(
        parse_args(
            [
                "--demo",
                "--offline",
                "--duration",
                "18",
                "--fps",
                "10",
                "--width",
                "160",
                "--height",
                "90",
                "--timer",
                "20",
            ]
        )
    )
    assert summary.error is None
    assert summary.fallback is True
    assert summary.frames == 180
    assert "POSTURE" in summary.cue_types
    assert "VOLUME" in summary.cue_types
    assert "PACE" in summary.cue_types
    assert "GAZE" in summary.cue_types
    assert any(action == "snooze" for _, action in summary.actions)
    assert summary.chimes >= 1
    assert summary.first_frame is not None and summary.last_frame is not None
    assert summary.first_frame.shape == (90, 160, 3)
    # With a 20 s timer the closing frames are inside the warning vignette.
    assert int(summary.last_frame[0, 0].sum()) > int(summary.first_frame[0, 0].sum())
    text = format_summary(summary)
    assert "Straighten Posture" in text
    assert "Speak up" in text
    assert "Slow down" in text
    assert "Look right" in text


def test_demo_palm_is_a_short_snooze_then_the_talk_continues():
    from src.demo_scene import section_at

    assert section_at(17.2)[-1] == "palm"
    assert section_at(20.0)[-1] is None


def test_demo_frame_is_finite():
    summary = run_session(
        parse_args(
            [
                "--offline",
                "--frames",
                "2",
                "--width",
                "80",
                "--height",
                "48",
                "--fps",
                "5",
            ]
        )
    )
    assert summary.last_frame is not None
    assert np.isfinite(summary.last_frame).all()
