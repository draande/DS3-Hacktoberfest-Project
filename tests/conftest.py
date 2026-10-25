"""Keep the suite hardware-free and deterministic."""

import os

# A developer key in the environment must not turn unit tests into network calls.
for _name in (
    "GEMMA_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "GEMMA_MODEL",
    "INSIGHT_MODEL",
):
    os.environ.pop(_name, None)

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")
