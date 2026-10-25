"""Rolling coach numbers for Gemma insights and the end-of-talk report."""

from __future__ import annotations

from pathlib import Path

from src.intelligence.gemma_client import CoachWindow


class SessionStats:
    """Keep every sample, and also a recent window for the insight prompt."""

    def __init__(self, window_seconds: float = 45.0) -> None:
        self.window_seconds = float(window_seconds)
        self._rows: list[tuple[float, float, float, bool, bool]] = []
        self._cues: list[tuple[float, str]] = []

    def observe(
        self,
        now: float,
        db: float,
        wpm: float,
        gaze_away: bool,
        posture_alert: bool,
    ) -> None:
        self._rows.append(
            (float(now), float(db), float(wpm), bool(gaze_away), bool(posture_alert))
        )

    def note_cue(self, now: float, cue_type: str) -> None:
        self._cues.append((float(now), str(cue_type)))

    def recent(self, now: float) -> CoachWindow:
        return self._slice(now, self.window_seconds)

    def overall(self, now: float) -> CoachWindow:
        return self._slice(now, None)

    def _slice(self, now: float, seconds: float | None) -> CoachWindow:
        if seconds is None:
            rows = self._rows
            cues = self._cues
            if len(rows) >= 2:
                span = rows[-1][0] - rows[0][0]
            elif rows:
                span = rows[-1][0]
            else:
                span = 0.0
        else:
            start = float(now) - float(seconds)
            rows = [row for row in self._rows if row[0] >= start]
            cues = [cue for cue in self._cues if cue[0] >= start]
            span = float(seconds)
        if not rows:
            return CoachWindow(seconds=span)
        count = float(len(rows))
        counts: dict[str, int] = {}
        for _stamp, kind in cues:
            counts[kind] = counts.get(kind, 0) + 1
        return CoachWindow(
            cue_counts=counts,
            avg_db=sum(row[1] for row in rows) / count,
            avg_wpm=sum(row[2] for row in rows) / count,
            gaze_away_ratio=sum(1.0 for row in rows if row[3]) / count,
            posture_ratio=sum(1.0 for row in rows if row[4]) / count,
            seconds=span,
        )


def save_report(text: str, directory: Path) -> Path:
    from datetime import datetime, timezone

    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = directory / f"session-{stamp}.md"
    # Two sessions in the same second should not clobber each other.
    suffix = 1
    while path.exists():
        path = directory / f"session-{stamp}-{suffix}.md"
        suffix += 1
    path.write_text(text, encoding="utf-8")
    return path
