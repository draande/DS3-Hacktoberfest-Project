# Wingman

Wingman is a smart-glasses presentation coach. It watches posture, gaze, audience expressions, and speaking pace, then shows a short cue on a heads-up display. The demo runs on synthetic landmarks and generated audio, so you do not need glasses, a webcam, a microphone, or an API key.

## Walkthrough
https://youtu.be/U2cPfduU50E

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
uv sync --group earbud                               # Windows media-session buttons
uv run python main.py --listen-keys                    # smtc and hook events; Ctrl+C stops
uv run python main.py --camera 0 --display --earbud    # earbud button toggles Ask Gemma
uv sync --group stt                                  # transcribe the live talk
uv run python main.py --demo --no-voice              # summary without Grok speech
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

Two models, used separately. The periodic trend uses `insight.model` (`gemini-3.5-flash-lite`). `INSIGHT_MODEL` overrides it. That call sets thinking to `minimal`, asks for JSON, and caps the reply at `insight.max_output_tokens` (256). The timeout is `insight.timeout_seconds` (10, the shortest deadline Google accepts) and the client is created once and reused. With paid credits the gap between trends is `insight.min_interval_seconds` (8). A 429 or 504 still backs off exponentially. One trend call is in flight at a time, and that loop does not share a lock or an HTTP client with the slow Gemma client.

Gemma (`gemma.model`, default `gemma-4-26b-a4b-it`, override `GEMMA_MODEL`) is only the slow path: the end-of-session report, and `GemmaClient.generate` for a direct question. It keeps the large output cap (2048) and does not send a thinking config or JSON response mode, because gemma-4 spends hidden thinking tokens and rejects those controls. A slow Gemma question or report cannot delay the periodic trend loop. Set `GEMMA_API_KEY`, `GOOGLE_API_KEY`, or `GEMINI_API_KEY` to enable either model. With no key, the same slots are filled locally and nothing is sent over the network.

Each insight call sends a compact summary of the last 45 seconds (cue counts, average dB, WPM, gaze-away, posture), not the raw frame history. Every call logs its latency with the model name. The end summary includes `insight_avg` and `gemma_avg` separately.

The sidebar shows those trend lines under an `INSIGHT` label, next to the instant cue bullets. The badge reads `GEMMA` after a reply, `THINKING` while a call is in flight, and `OFFLINE` otherwise.

When the session ends, Wingman asks for three strengths and three things to improve, prints them, and saves `reports/session-<timestamp>.md`. If that call fails, the file is still written from the local numbers. The frame loop never waits on either call. The test suite clears the key and model variables and uses fake transports.

## Ask Gemma

A direct question calls `GemmaClient.generate` on Gemma's own thread. It uses `gemma.model` and `gemma.timeout_seconds`. It does not use the insight client, that client's lock, its SDK client, or its pending flag, and the request does not include a thinking config or a JSON response type. An ask cannot delay the periodic trend loop. Only one ask is sent at a time. A second question waits until that call returns.

In a preview, `G` opens a one-line bar. Enter sends it, Esc cancels it, and letters typed there do not fire other shortcuts. `F`, Esc, and `Q` behave as usual while the bar is closed. `V` toggles push-to-talk on the microphone samples the frame loop is already reading, and stops on its own after 30 seconds. `--ask "how was my pace?"` prints the question and a 1–3 line reply, then exits without a window or a camera. `--ask` on its own, or `--ask -`, reads one line from stdin.

The sidebar shows `YOU`, then `THINKING...` with the elapsed seconds, then `ASK` and the reply. The reply stays up longer than a cue bullet. A timeout, 429, or 504 is one short error line. The frame loop never waits on the call.

`gemma-4-26b-a4b-it` does not accept audio, so a spoken ask is transcribed first. `uv sync --group stt` installs `faster-whisper` for that. E2B, E4B, and 12B models can take the wav directly. With no transcriber and a text-only model, the sidebar says to type with `G` instead.

The earbud button is the same push-to-talk action as `V`. `--earbud`, or `ask.media_key.enabled: true`, installs a global listener, so the Wingman window does not have to be focused. The first tap starts recording and the second sends it. It also stops on its own after 30 seconds. The default button is Play/Pause. `ask.media_key.keys` can also list `next`, `previous`, `stop`, `headset`, or a virtual-key code such as `0xB3`. Down/up pairs and repeats closer than 300 ms count as one tap. The HUD shows `EARBUD` while a backend is active, and startup prints which backend that is.

Bluetooth buttons on Windows arrive through the system media session, not as keyboard keys. That backend is the primary one. On Windows:

```bash
uv sync --group earbud
uv run python main.py --listen-keys
uv run python main.py --camera 0 --display --earbud
```

`uv sync --group earbud` installs the WinRT media packages (`winrt-Windows.Media`, `winrt-Windows.Media.Playback`, `winrt-Windows.Media.Core`, and `winrt-Windows.Foundation`). They are Windows-only. Wingman then owns a silent looping session titled "Wingman – Ask Gemma", enables play, pause, next, and previous, and listens for `ButtonPressed`. Play, Pause, and Play/Pause are a tap. Next and previous count only when those keys are listed. While the session is active, Windows routes those buttons to Wingman, so another music app may pause until Wingman exits.

`ask.media_key.backend` is `auto` (default), `smtc`, or `hook`. Auto uses the media session when it starts, and otherwise a low-level keyboard hook (`WH_KEYBOARD_LL`) on a daemon thread. The hook ignores every key that is not in the configured set, so it does not run Python on ordinary typing. `ask.media_key.suppress` defaults to true and applies only to that hook: those keys are swallowed. The media session does not use the flag.

`--listen-keys` starts both backends for about 15 seconds and prefixes each line with `smtc` or `hook`. Ordinary keys are not printed. Ctrl+C stops it. If a button prints nothing, Windows is not exposing it as a media-session button or a media virtual key. Do not type passwords during that window.

Esc, Q, closing the preview, and Ctrl+C in the terminal all end the session and stop the listener (unhook, stop the media player, and join threads). A second Ctrl+C kills the process if the first one is stuck. Off Windows, or if neither backend can start, Wingman logs that once and `V` still works.

## Running conversation summary

During the session, Flash Lite (`insight.model`, default `gemini-3.5-flash-lite`, override `INSIGHT_MODEL`) keeps a running summary of the talk. Each update sends only the previous summary, the new stretch of speech, and one line of coaching numbers. It does not send the whole transcript again. The work runs on its own thread, with its own client and lock, so it does not wait on the frame loop, the insight bullets, or Gemma. It never calls `GemmaClient`.

Speech text uses the same local transcriber as a spoken ask. Install it with:

```bash
uv sync --group stt
```

Without that group, the summary still updates from the coaching line and says the speech was not transcribed. `summary.enabled` turns the feature off. `summary.interval_seconds` (default 20) is how often a new stretch is folded in.

When the session ends, Wingman asks Flash for a final wording of that running summary and prints it as `Conversation: ...`. The same text is the `## Conversation` section in `reports/session-<timestamp>.md`. If Flash is offline or the call fails, a short local line is used and the strengths report is still written.

## Grok voice

After that summary and the strengths list exist, Wingman can speak them. The call is xAI's text-to-speech API:

- Endpoint: `POST https://api.x.ai/v1/tts`
- Model: `grok-tts` (the request body does not take a separate model field)
- Voices: `eve` (default), `ara`, `leo`, `rex`, `sal`, or a custom voice id
- Key: `XAI_API_KEY` (never put it in the config file)

```bash
export XAI_API_KEY="your key"
uv run python main.py --demo
uv run python main.py --camera 0 --display
uv run python main.py --demo --no-voice-playback   # save the wav, do not play it
uv run python main.py --demo --no-voice            # skip the call
```

`voice.enabled`, `voice.voice_id`, `voice.language`, and `voice.playback` are in `config/default_config.yaml`. Playback goes through the same speaker stream as the cue chimes. The wav is saved next to the report as `reports/session-<timestamp>.wav`. If the key is missing, the call fails, or there is no speaker, Wingman prints the summary and continues.

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
| Earbud Play/Pause | Same as V when `--earbud` or `ask.media_key.enabled` is on. Works while the window is unfocused. The HUD shows EARBUD |

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
