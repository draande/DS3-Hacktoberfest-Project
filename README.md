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
uv run python main.py --camera 0 --display             # live cues, meter, Gemma insights
uv run python main.py --demo --display                 # borderless fullscreen preview
uv run python main.py --demo --timer 20 --display          # preview the edge vignette
uv run python main.py --demo --theme minimalist
uv run python main.py --offline                            # ignore a Gemma API key
uv run python main.py --ask "how was my pace?"             # one Gemma reply, then exit
```

`--timer 20` shortens the presentation clock so the border glow moves from clear, to amber under 25% remaining, to red under 10%. The config default is a 5 minute talk.

## Live camera

Live mode needs a webcam, PortAudio (`libportaudio2` on Debian/Ubuntu), and three MediaPipe task files. The math and the demo do not need those files.

```bash
uv run python main.py --download-models
uv run python main.py --camera 0 --display
```

`--video clip.mp4` plays a file instead of the webcam. Vision cues appear only after the models load. Audio cues need a working microphone; if PortAudio is missing, the HUD still runs and prints the reason.

The microphone is opened at the device's own sample rate (often 44.1 or 48 kHz) and resampled to 16 kHz for the volume and pace math. A fixed 16 kHz mono capture is accepted by many devices and then never delivers samples. The input stream is also restarted once after open, because PortAudio can report it active while the callback never runs. Cue chimes play on a speaker stream at that device's rate. `--display` and live mode play those chimes. `--display` and `--chimes` also play the demo's scripted audio in real time. On Windows, if the mic stays closed, allow desktop apps under Settings → Privacy → Microphone. The pygame preview uses a dummy audio driver so it does not take the sound card away from PortAudio.

A level meter sits at the bottom of the HUD with a mark on the 42 dB speak-up line, so you can see the microphone working even when no volume cue is up. Live posture spends the first `posture.calibrate_seconds` (3s) learning how you sit in front of the laptop camera, then cues only when you slouch further than that baseline. A pinch has to be held for about 0.4s, with hysteresis on the finger gap, before it changes sensitivity.

Posture, volume, pace, and gaze are decided on the device every frame. If several are true, the one that just appeared yields so another type can take the next slot. Gemma does not replace those cues.

## Gemma

Gemma writes two slow things: a trend every `gemma.min_interval_seconds` (18s) and a short report when the session ends. The default model is `gemma-4-26b-a4b-it`. `GEMMA_MODEL` overrides `gemma.model`. Set `GEMMA_API_KEY`, `GOOGLE_API_KEY`, or `GEMINI_API_KEY` to enable it. With no key, the same slots are filled locally and nothing is sent over the network.

Each insight call sends a compact summary of the last 45 seconds (cue counts, average dB, WPM, gaze-away, posture), not the raw frame history. Output room is `gemma.max_output_tokens` (2048) so the model's thinking tokens do not come back as an empty reply. The client timeout defaults to 45 seconds and cannot be configured below 10, which the API rejects. A 429 or 504 backs off exponentially. Every call logs its latency, and the end summary includes `gemma_avg`.

The sidebar shows those trend lines under an `INSIGHT` label, next to the instant cue bullets. The badge reads `GEMMA` after a reply, `THINKING` while a call is in flight, and `OFFLINE` otherwise.

When the session ends, Wingman asks for three strengths and three things to improve, prints them, and saves `reports/session-<timestamp>.md`. If that call fails, the file is still written from the local numbers. The frame loop never waits on either call. The test suite clears the key and model variables and uses fake transports.

## Ask Gemma

A direct question uses the Gemma model and `gemma.timeout_seconds`. It does not cancel the periodic insight call, and it does not read the insight interval. Only one ask is sent at a time. A second question waits until that call returns.

In a preview, `G` opens a one-line bar. Enter sends it, Esc cancels it, and letters typed there do not fire other shortcuts. `F`, Esc, and `Q` behave as usual while the bar is closed. `V` toggles push-to-talk on the microphone samples the frame loop is already reading, and stops on its own after 30 seconds. `--ask "how was my pace?"` prints the question and a 1–3 line reply, then exits without a window or a camera. `--ask` on its own, or `--ask -`, reads one line from stdin.

The sidebar shows `YOU`, then `THINKING...` with the elapsed seconds, then `ASK` and the reply. The reply stays up longer than a cue bullet. A timeout, 429, or 504 is one short error line. The frame loop never waits on the call.

`gemma-4-26b-a4b-it` does not accept audio, so a spoken ask is transcribed first. `uv sync --group stt` installs `faster-whisper` for that. E2B, E4B, and 12B models can take the wav directly. With no transcriber and a text-only model, the sidebar says to type with `G` instead.

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
| G | Ask Gemma. Type a line, Enter sends, Esc cancels |
| V | Push-to-talk. Press again to send, or wait 30 seconds |

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
