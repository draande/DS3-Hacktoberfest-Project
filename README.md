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

If `GEMMA_API_KEY`, `GOOGLE_API_KEY`, or `GEMINI_API_KEY` is set, Wingman asks Google Gemma (`gemma-3-12b-it` by default, override with `GEMMA_MODEL`) for the cue JSON. With no key, or when the response is not valid JSON, a deterministic offline coach emits the same schema. The test suite clears those variables so it never calls the network.

## Hotkeys

Shown while a preview window is open (`--display`):

| Key | Action |
| --- | --- |
| Space, Enter, T | Dismiss the active cue (earbud double-tap) |
| S | Snooze every cue for 30 seconds |
| M | Switch between Ironman and Minimalist |
| `[` / `]` | More sensitive / less sensitive coaching thresholds |
| Q or Esc | Quit the preview |

Gestures, from hand landmarks:

| Gesture | Action |
| --- | --- |
| Raised open palm | Snooze cues for 30 seconds |
| Horizontal swipe | Switch theme |
| Pinch | Raise the threshold scale one step |

Two loud spikes about 50–450 ms apart also dismiss the active cue.

## Configuration

`config/default_config.yaml` holds the plan thresholds (12° slouch, 15% forward head, 4 s gaze, 42 dB speak-up, 180 WPM rush, 5 s cue cooldown, vignette at 25% and 10%). `config/themes.yaml` holds the Ironman and Minimalist palettes.

Digital audio has no sound-pressure level. Volume uses `dB = 20 log10(RMS / 1e-4)`, so a full-scale tone is about 80 dB and the 42 dB cue lines up with a quiet talker.

## Tests

```bash
uv run pytest
uv run black --check .
```

Tests use synthetic landmark arrays and generated PCM. They do not open a camera, microphone, GPU, or network connection.
