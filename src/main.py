"""Orchestrate capture, perception, coaching, and the HUD."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np

from src.audio.audio_feedback import chime_for_cue, play_chime
from src.audio.pace_analyzer import PaceAnalyzer
from src.audio.volume_analyzer import VolumeAnalyzer
from src.capture.audio_stream import AudioCaptureError, MicrophoneStream
from src.capture.video_stream import CameraUnavailable, FileVideoSource, WebcamSource
from src.config_loader import AppConfig, load_config
from src.controls.session_control import SessionControl
from src.demo_scene import DemoScene
from src.hud.renderer import HudRenderer, HudState, PreviewWindow
from src.intelligence.gemma_client import CoachMetrics, GemmaClient
from src.vision.audience_tells import analyze_audience, tells_from_blendshapes
from src.vision.gaze_tracker import GazeTracker
from src.vision.gesture_detector import GestureDetector
from src.vision.mediapipe_backend import MediaPipeBackend, download_models
from src.vision.pose_tracker import analyze_posture


@dataclass
class RunOptions:
    demo: bool = True
    camera: int | None = None
    video: str | None = None
    display: bool = False
    display_backend: str = "opencv"
    frames: int | None = None
    duration: float | None = None
    theme: str | None = None
    fps: int | None = None
    width: int | None = None
    height: int | None = None
    timer: float | None = None
    chimes: bool = False
    offline: bool = False
    download_models: bool = False


@dataclass
class RunSummary:
    demo: bool = True
    theme: str = "ironman"
    fallback: bool = True
    frames: int = 0
    duration: float = 0.0
    cues: list[tuple[float, str, str]] = field(default_factory=list)
    actions: list[tuple[float, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    chimes: int = 0
    first_frame: np.ndarray | None = None
    last_frame: np.ndarray | None = None
    error: str | None = None

    @property
    def cue_types(self) -> list[str]:
        return [cue_type for _, cue_type, _ in self.cues]


def parse_args(argv: list[str] | None = None) -> RunOptions:
    parser = argparse.ArgumentParser(
        description="Wingman smart-glasses presentation and social-cue coach"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run on synthetic landmarks and audio (the default when no camera or video is set)",
    )
    parser.add_argument(
        "--camera", type=int, default=None, help="Webcam index for live mode"
    )
    parser.add_argument(
        "--video", type=str, default=None, help="Play a video file instead of a webcam"
    )
    parser.add_argument("--display", action="store_true", help="Open a preview window")
    parser.add_argument(
        "--display-backend",
        choices=("opencv", "pygame"),
        default="opencv",
        help="Preview toolkit. The HUD itself is always drawn with OpenCV.",
    )
    parser.add_argument(
        "--frames", type=int, default=None, help="Stop after this many frames"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Seconds to run. Demo default is 18 so posture, volume, pace, and gaze each fire.",
    )
    parser.add_argument("--theme", choices=("ironman", "minimalist"), default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument(
        "--timer",
        type=float,
        default=None,
        help="Presentation length in seconds. Default comes from config (300).",
    )
    parser.add_argument(
        "--chimes",
        action="store_true",
        help="Play synthesized cue chimes when PortAudio is available",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force the deterministic coach even if a Gemma API key is set",
    )
    parser.add_argument(
        "--download-models",
        action="store_true",
        help="Download MediaPipe .task models into models/ and exit",
    )
    args = parser.parse_args(argv)
    demo = bool(args.demo or (args.camera is None and args.video is None))
    if args.camera is not None or args.video is not None:
        demo = bool(args.demo)
    return RunOptions(
        demo=demo and not args.download_models,
        camera=args.camera,
        video=args.video,
        display=args.display,
        display_backend=args.display_backend,
        frames=args.frames,
        duration=args.duration,
        theme=args.theme,
        fps=args.fps,
        width=args.width,
        height=args.height,
        timer=args.timer,
        chimes=args.chimes,
        offline=args.offline,
        download_models=args.download_models,
    )


def format_summary(summary: RunSummary) -> str:
    mode = "demo" if summary.demo else "live"
    coaching = "offline" if summary.fallback else "gemma"
    lines = [
        f"Wingman {mode} | theme={summary.theme} | coaching={coaching} | frames={summary.frames}"
    ]
    events = [
        (timestamp, f"t={timestamp:05.2f} {cue_type} {text}")
        for timestamp, cue_type, text in summary.cues
    ]
    events.extend(
        (timestamp, f"t={timestamp:05.2f} action {action}")
        for timestamp, action in summary.actions
    )
    lines.extend(line for _, line in sorted(events, key=lambda item: item[0]))
    lines.extend(summary.notes)
    if summary.error:
        lines.append(summary.error)
    lines.append(f"cues={len(summary.cues)} duration={summary.duration:.1f}s")
    return "\n".join(lines)


def run_session(options: RunOptions, config: AppConfig | None = None) -> RunSummary:
    app_config = config or load_config()
    if options.download_models:
        paths = download_models()
        summary = RunSummary(demo=False, theme=app_config.hud.theme, fallback=True)
        summary.notes.extend(str(path) for path in paths)
        return summary

    fps = options.fps or app_config.hud.fps
    width = options.width or app_config.hud.width
    height = options.height or app_config.hud.height
    theme_name = options.theme or app_config.hud.theme
    timer_seconds = (
        options.timer
        if options.timer is not None
        else app_config.timer.duration_seconds
    )
    if options.frames is not None:
        frame_limit: int | None = options.frames
    elif options.duration is not None:
        frame_limit = max(1, int(round(options.duration * fps)))
    elif options.demo:
        frame_limit = max(1, int(round(18 * fps)))
    else:
        frame_limit = None

    control = SessionControl(
        themes=list(app_config.themes),
        theme_name=theme_name,
        slouch_degrees=app_config.posture.slouch_degrees,
        ear_forward_offset=app_config.posture.ear_forward_offset,
        speak_up_db=app_config.volume.speak_up_db,
        cooldown_seconds=app_config.cues.cooldown_seconds,
        snooze_seconds=app_config.cues.snooze_seconds,
        global_gap_seconds=app_config.cues.global_gap_seconds,
    )
    renderer = HudRenderer(
        app_config.theme(control.theme_name),
        warning_ratio=app_config.timer.warning_ratio,
        critical_ratio=app_config.timer.critical_ratio,
    )
    volume = VolumeAnalyzer(
        threshold_db=app_config.volume.speak_up_db,
        quiet_duration=app_config.volume.quiet_duration_seconds,
        reference=app_config.volume.reference_amplitude,
        sample_rate=app_config.audio.sample_rate,
        level_window_seconds=app_config.volume.level_window_seconds,
        buffer_seconds=app_config.audio.buffer_seconds,
    )
    pace = PaceAnalyzer(
        sample_rate=app_config.audio.sample_rate,
        rush_wpm=app_config.pace.rush_wpm,
        target_wpm_min=app_config.pace.target_wpm_min,
        target_wpm_max=app_config.pace.target_wpm_max,
        syllables_per_word=app_config.pace.syllables_per_word,
        pause_seconds=app_config.pace.pause_seconds,
        continuous_speech_seconds=app_config.pace.continuous_speech_seconds,
        vad_rms=app_config.pace.vad_rms,
        window_seconds=app_config.pace.window_seconds,
    )
    gaze = GazeTracker(
        away_seconds=app_config.gaze.away_seconds,
        off_center=app_config.gaze.off_center,
        look_down=app_config.gaze.look_down,
        side_bias_ratio=app_config.gaze.side_bias_ratio,
        history_seconds=app_config.gaze.history_seconds,
    )
    gestures = GestureDetector()
    client = GemmaClient.from_env(
        model=app_config.gemma.model,
        duration_ms=app_config.cues.duration_ms,
        offline=options.offline,
    )
    summary = RunSummary(
        demo=options.demo, theme=control.theme_name, fallback=client.used_fallback
    )

    scene = (
        DemoScene(width, height, app_config.audio.sample_rate) if options.demo else None
    )
    video = None
    microphone = None
    backend = None
    preview = PreviewWindow(options.display_backend) if options.display else None
    if scene is None:
        try:
            if options.video:
                video = FileVideoSource(options.video, fps=fps)
            else:
                video = WebcamSource(
                    options.camera or 0, width=width, height=height, fps=fps
                )
        except (CameraUnavailable, FileNotFoundError, RuntimeError) as exc:
            summary.error = str(exc)
            return summary
        microphone = MicrophoneStream(
            sample_rate=app_config.audio.sample_rate,
            block_size=app_config.audio.block_size,
            buffer_seconds=app_config.audio.buffer_seconds,
        )
        try:
            microphone.start()
        except AudioCaptureError as exc:
            summary.notes.append(str(exc))
            microphone = None
        backend = MediaPipeBackend()
        if not backend.load():
            summary.notes.append(backend.error or "MediaPipe models are not loaded")

    sample_rate = app_config.audio.sample_rate
    audio_cursor = 0
    frame_index = 0
    try:
        while frame_limit is None or frame_index < frame_limit:
            timestamp = frame_index / fps
            hand = None
            blendshapes = None
            if scene is not None:
                chunk_count = max(1, int(round(sample_rate / fps)))
                sample = scene.sample(timestamp, audio_cursor, chunk_count)
                audio_cursor += chunk_count
                frame = sample.frame
                pose_landmarks = sample.pose
                face_landmarks = sample.face
                hand = sample.hand
                audio = sample.audio
            else:
                assert video is not None
                video_frame = video.read()
                if video_frame is None:
                    break
                frame = video_frame.image
                timestamp = video_frame.timestamp
                pose_landmarks = None
                face_landmarks = None
                if backend is not None and backend.ready:
                    observation = backend.detect(frame)
                    pose_landmarks = observation.pose
                    face_landmarks = observation.face
                    hand = observation.hands
                    blendshapes = observation.blendshapes
                if microphone is not None:
                    audio = microphone.pop_ready().samples
                else:
                    audio = np.zeros(max(1, int(sample_rate / fps)), dtype=np.float32)

            posture = (
                analyze_posture(
                    pose_landmarks,
                    slouch_degrees=control.effective_slouch(),
                    ear_forward_limit=control.effective_forward(),
                )
                if pose_landmarks is not None
                else None
            )
            gaze_state = gaze.update(face_landmarks, timestamp)
            if blendshapes:
                audience = tells_from_blendshapes(blendshapes)
            elif face_landmarks is not None:
                audience = analyze_audience(
                    face_landmarks,
                    brow_gap_ratio=app_config.audience.brow_gap_ratio,
                    eye_aspect_wide=app_config.audience.eye_aspect_wide,
                    brow_raise_ratio=app_config.audience.brow_raise_ratio,
                )
            else:
                audience = None
            gesture = gestures.update(hand, timestamp)
            gesture_action = control.handle_gesture(gesture, timestamp)
            if gesture_action:
                summary.actions.append((timestamp, gesture_action))
            if control.theme_name != renderer.theme.key:
                renderer.set_theme(app_config.theme(control.theme_name))
                summary.theme = control.theme_name

            volume.threshold_db = control.effective_speak_up_db()
            volume_reading = volume.update(audio, timestamp)
            pace_reading = pace.update(audio, timestamp)
            audio_action = control.handle_audio(audio, timestamp)
            if audio_action:
                summary.actions.append((timestamp, audio_action))

            metrics = CoachMetrics(
                shoulder_tilt_deg=0.0 if posture is None else posture.shoulder_tilt_deg,
                neck_slump_deg=0.0 if posture is None else posture.neck_slump_deg,
                ear_forward_offset=(
                    0.0 if posture is None else posture.ear_forward_offset
                ),
                posture_alert=False if posture is None else posture.straighten,
                gaze_away_seconds=gaze_state.away_seconds,
                gaze_cue=gaze_state.cue,
                gaze_text=gaze_state.display_text,
                gaze_icon=gaze_state.icon,
                volume_db=volume_reading.db,
                speak_up=volume_reading.speak_up,
                wpm=pace_reading.wpm,
                rushing=pace_reading.rushing,
                run_on=pace_reading.run_on,
                audience_confused=False if audience is None else audience.confused,
                audience_surprised=False if audience is None else audience.surprised,
                blendshapes={} if audience is None else dict(audience.blendshapes),
            )
            proposal = client.arbitrate(metrics)
            active = control.filter.consider(proposal, timestamp)
            if control.filter.fired_at == timestamp and active is not None:
                summary.cues.append(
                    (timestamp, active["cue_type"], active["display_text"])
                )
                if active.get("audio_chime"):
                    chime = chime_for_cue(active["cue_type"], sample_rate)
                    summary.chimes += 1
                    if options.chimes:
                        play_chime(chime, sample_rate)

            remaining = max(0.0, timer_seconds - timestamp)
            ratio = remaining / timer_seconds if timer_seconds else 0.0
            rendered = renderer.render(
                frame,
                HudState(
                    remaining_seconds=remaining,
                    remaining_ratio=ratio,
                    cue=active,
                    volume_db=volume_reading.db,
                    wpm=pace_reading.wpm,
                    pulse_t=timestamp,
                ),
            )
            if summary.first_frame is None:
                summary.first_frame = rendered
            summary.last_frame = rendered
            summary.frames = frame_index + 1
            summary.duration = timestamp
            summary.fallback = client.used_fallback
            summary.theme = control.theme_name

            if preview is not None:
                key = preview.show(rendered)
                if key == "quit":
                    break
                if key:
                    action = control.handle_key(key, timestamp)
                    if action == "quit":
                        break
                    if action:
                        summary.actions.append((timestamp, action))
                    if control.theme_name != renderer.theme.key:
                        renderer.set_theme(app_config.theme(control.theme_name))
            frame_index += 1
    finally:
        if video is not None:
            video.close()
        if microphone is not None:
            microphone.close()
        if backend is not None:
            backend.close()
        if preview is not None:
            preview.close()
    return summary


def main(argv: list[str] | None = None) -> int:
    options = parse_args(argv)
    if options.download_models:
        summary = run_session(options)
        print(format_summary(summary))
        return 0 if summary.error is None else 1
    summary = run_session(options)
    print(format_summary(summary))
    return 0 if summary.error is None else 1
