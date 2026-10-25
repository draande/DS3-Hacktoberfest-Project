# Wingman

Wingman is a smart-glasses presentation coach. It watches posture, gaze, audience expressions, and speaking pace, then shows a short cue on a heads-up display. The demo runs on synthetic landmarks and generated audio, so you do not need glasses, a webcam, a microphone, or an API key.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Run the demo

Headless demo (no camera, microphone, or window). It plays an 18 second scripted talk: a slouch, a quiet stretch, a rush, a long look to one side, then a raised palm that snoozes the cue.

```bash
uv run python main.py --demo
```

Useful flags:

```bash
uv run python main.py --demo --duration 18 --fps 10 --width 640 --height 360
uv run python main.py --demo --display                 # borderless fullscreen preview
uv run python main.py --demo --timer 20 --display          # preview the edge vignette
uv run python main.py --demo --theme minimalist
uv run python main.py --offline                            # ignore a Gemma API key
```

`--timer 20` shortens the presentation clock so the border glow moves from clear, to amber under 25% remaining, to red under 10%. The config default is a 5 minute talk.

## Live camera

Live mode needs a webcam, PortAudio (`libportaudio2` on Debian/Ubuntu), and three MediaPipe task files. The math and the demo do not need those files.

```bash
uv run python main.py --download-models
uv run python main.py --camera 0 --display
```

`--video clip.mp4` plays a file instead of the webcam. Vision cues appear only after the models load. Audio cues need a working microphone; if PortAudio is missing, the HUD still runs and prints the reason.

## Gemma

If `GEMMA_API_KEY`, `GOOGLE_API_KEY`, or `GEMINI_API_KEY` is set, Wingman asks Google Gemma for the cue JSON. The default model is `gemma-4-26b-a4b-it` (`gemma.model` in `config/default_config.yaml`). `GEMMA_MODEL` overrides that. Calls run on a background thread: at most one at a time, at least `gemma.min_interval_seconds` apart (2.5s), each limited by `gemma.timeout_seconds` (8s). The frame loop never waits. Until a reply arrives, and again while a call is in flight or after an error, the HUD keeps the deterministic offline cues. A small badge at the bottom of the sidebar reads `GEMMA` when the cue on screen came from the model and `OFFLINE` otherwise. The end summary includes `gemma_ok` and `gemma_failed`. A failed call logs one warning per distinct error. With no key, the offline coach is used and nothing is sent over the network. The test suite clears the key and model variables.

## Hotkeys

Shown while a preview window is open (`--display`):

| Key | Action |
| --- | --- |
| F | Toggle borderless fullscreen |
| Esc | Leave fullscreen. When the window is already windowed, quit |
| Q | Quit the preview |
| Space, Enter, T | Dismiss the active cue (earbud double-tap) |
| S | Snooze every cue for 30 seconds |
| M | Switch between Ironman and Minimalist |
| `[` / `]` | More sensitive / less sensitive coaching thresholds |

Gestures, from hand landmarks:

| Gesture | Action |
| --- | --- |
| Raised open palm | Snooze cues for 30 seconds |
| Horizontal swipe | Switch theme |
| Pinch | Raise the threshold scale one step |

Two loud spikes about 50–450 ms apart also dismiss the active cue.

`--display` opens a borderless fullscreen window at the screen resolution and lays the HUD out at that size, so the picture is not a small canvas stretched to fit. On Windows the process is marked DPI-aware before the screen size is read. If the screen size cannot be detected, the window still requests fullscreen at the configured `hud.width` and `hud.height`. Headless mode never opens a window. Esc leaves fullscreen; press Esc again, or Q, to quit.

## Feedback sidebar

Gemma's JSON includes a `bullets` array: 1 to 4 lines, each 3 to 6 words, most urgent first. Those lines appear in a sidebar, on the right by default. Newest lines sit on top. Each line fades out and then expires (`hud.sidebar.ttl_seconds` and `fade_seconds`). Set `hud.sidebar.side` to `left` to flip the column. Offline mode, and any reply that omits `bullets`, fills the sidebar from the cue text so the demo still shows feedback. Ironman draws a dark bracketed panel. Minimalist draws a thin frosted panel without brackets.

## Avatar slot

The avatar itself is not part of this build. The top of the sidebar is reserved for it (`hud.avatar.anchor: sidebar_top`, `hud.avatar.height_ratio`). `src/hud/avatar.py` paints a labeled placeholder and documents the hook:

```python
draw_avatar(frame, rect, state)  # rect is (x, y, width, height)
```

`state` is an `AvatarState` with the theme name, the active cue, whether the speaker is talking, volume, pace, and time remaining. Replace `draw_avatar`, or assign your own `AvatarWidget` to `HudRenderer.avatar_widget`. The renderer calls `widget.draw(frame, rect, state)` with the same contract.

## Configuration

`config/default_config.yaml` holds the plan thresholds (12° slouch, 15% forward head, 4 s gaze, 42 dB speak-up, 180 WPM rush, 5 s cue cooldown, vignette at 25% and 10%) plus the sidebar and avatar slot. `config/themes.yaml` holds the Ironman and Minimalist palettes.

Digital audio has no sound-pressure level. Volume uses `dB = 20 log10(RMS / 1e-4)`, so a full-scale tone is about 80 dB and the 42 dB cue lines up with a quiet talker.

## Tests

```bash
uv run pytest
uv run black --check .
```

Tests use synthetic landmark arrays and generated PCM. They do not open a camera, microphone, GPU, or network connection.
