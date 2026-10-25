"""Synthesized sine chimes. Playback is optional and never required for tests.

``sd.play`` opens a new stream at whatever rate it is given and, on Windows,
that call often fails at 16 kHz or fights the microphone stream. :class:`AudioOutput`
keeps one output stream at the speaker's native rate and mixes chimes onto it.
"""

from __future__ import annotations

import threading

import numpy as np

from src.capture.audio_stream import (
    StreamResampler,
    _load_sounddevice,
    _native_blocksize,
)

CHIME_FREQUENCIES = {
    "POSTURE": 523.25,
    "VOLUME": 659.25,
    "PACE": 440.0,
    "GAZE": 783.99,
    "AUDIENCE": 587.33,
    "ENCOURAGE": 880.0,
    "DISMISS": 349.23,
}


def synthesize_chime(
    frequency: float = 880.0,
    duration: float = 0.18,
    sample_rate: int = 16000,
    volume: float = 0.2,
) -> np.ndarray:
    """Return a decaying sine in float32, peak amplitude at most ``volume``."""
    count = max(1, int(duration * sample_rate))
    t = np.arange(count, dtype=np.float32) / sample_rate
    envelope = np.exp(-t * 8.0)
    wave = volume * envelope * np.sin(2.0 * np.pi * frequency * t)
    return wave.astype(np.float32)


def chime_for_cue(cue_type: str, sample_rate: int = 16000) -> np.ndarray:
    frequency = CHIME_FREQUENCIES.get(cue_type.upper(), 880.0)
    return synthesize_chime(frequency=frequency, sample_rate=sample_rate)


class _SampleQueue:
    """Mono float32 queue. ``mix`` adds onto the samples about to be played."""

    def __init__(self, capacity: int) -> None:
        self.capacity = max(1, int(capacity))
        self._buf = np.zeros(self.capacity, dtype=np.float32)
        self._read = 0
        self._count = 0
        self._lock = threading.Lock()

    def write(self, samples: np.ndarray) -> None:
        data = np.asarray(samples, dtype=np.float32).reshape(-1)
        if data.size == 0:
            return
        with self._lock:
            if data.size >= self.capacity:
                data = data[-self.capacity :]
                self._buf[:] = data
                self._read = 0
                self._count = data.size
                return
            overflow = self._count + data.size - self.capacity
            if overflow > 0:
                self._read = (self._read + overflow) % self.capacity
                self._count -= overflow
            write_at = (self._read + self._count) % self.capacity
            first = min(data.size, self.capacity - write_at)
            self._buf[write_at : write_at + first] = data[:first]
            if first < data.size:
                self._buf[: data.size - first] = data[first:]
            self._count += data.size

    def mix(self, samples: np.ndarray) -> None:
        data = np.clip(np.asarray(samples, dtype=np.float32).reshape(-1), -1.0, 1.0)
        if data.size == 0:
            return
        if data.size > self.capacity:
            data = data[: self.capacity]
        with self._lock:
            missing = data.size - self._count
            if missing > 0:
                write_at = (self._read + self._count) % self.capacity
                first = min(missing, self.capacity - write_at)
                self._buf[write_at : write_at + first] = 0
                if first < missing:
                    self._buf[: missing - first] = 0
                self._count += missing
            index = self._read
            for value in data:
                mixed = float(self._buf[index]) + float(value)
                if mixed > 1.0:
                    mixed = 1.0
                elif mixed < -1.0:
                    mixed = -1.0
                self._buf[index] = mixed
                index = (index + 1) % self.capacity

    def read(self, count: int) -> np.ndarray:
        with self._lock:
            take = min(int(count), self._count)
            if take <= 0:
                return np.zeros(0, dtype=np.float32)
            start = self._read
            if start + take <= self.capacity:
                out = self._buf[start : start + take].copy()
            else:
                head = self.capacity - start
                out = np.concatenate((self._buf[start:], self._buf[: take - head]))
            self._read = (self._read + take) % self.capacity
            self._count -= take
            return out


class AudioOutput:
    """Non-blocking speaker. Analysis audio is 16 kHz; the device is not."""

    def __init__(self, analysis_rate: int = 16000) -> None:
        self.analysis_rate = int(analysis_rate)
        self.output_rate = self.analysis_rate
        self.error: str | None = None
        self.active = False
        self.underruns = 0
        self._queue = _SampleQueue(max(1, self.analysis_rate))
        self._resampler = StreamResampler(self.analysis_rate, self.analysis_rate)
        self._stream = None

    def start(self, sd_module=None) -> bool:
        try:
            sd = sd_module if sd_module is not None else _load_sounddevice()
            info = sd.query_devices(kind="output")
            rate = int(round(float(info.get("default_samplerate") or 0)))
            max_channels = int(info.get("max_output_channels") or 0)
        except Exception as exc:
            self.error = f"Could not open audio output ({exc})."
            self.active = False
            return False
        if rate <= 0:
            rate = self.analysis_rate
        if max_channels < 1:
            self.error = "No speaker output channels are available."
            self.active = False
            return False
        self.output_rate = rate
        self._resampler = StreamResampler(self.analysis_rate, rate)
        self._queue = _SampleQueue(max(rate, int(rate * 2)))
        try:
            self._stream = _open_output_stream(sd, self._fill, rate, max_channels)
        except Exception as exc:
            self.error = f"Could not open audio output ({exc})."
            self.active = False
            return False
        self.active = True
        return True

    def write(self, samples: np.ndarray) -> None:
        """Queue continuous audio, such as the demo soundtrack."""
        if not self.active:
            return
        self._queue.write(self._resampler.process(samples))

    def play(self, samples: np.ndarray) -> None:
        """Mix a one-shot (a cue chime) onto whatever is already queued."""
        if not self.active:
            return
        converted = StreamResampler(self.analysis_rate, self.output_rate).process(
            samples
        )
        self._queue.mix(converted)

    def _fill(self, outdata, frames, time_info, status) -> None:  # noqa: ARG001
        if status:
            self.underruns += 1
        samples = self._queue.read(frames)
        rendered = np.zeros(frames, dtype=np.float32)
        rendered[: samples.size] = samples
        if getattr(outdata, "ndim", 1) == 2:
            for channel in range(outdata.shape[1]):
                outdata[:, channel] = rendered
        else:
            outdata[:] = rendered

    def close(self) -> None:
        self.active = False
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                return None


def _open_output_stream(sd, callback, rate: int, max_channels: int):
    from src.capture.audio_stream import _channel_choices

    choices = _channel_choices(max_channels)
    last: Exception | None = None
    for channels in choices:
        for blocksize in (_native_blocksize(rate), 0):
            stream = None
            try:
                stream = sd.OutputStream(
                    samplerate=rate,
                    channels=channels,
                    dtype="float32",
                    blocksize=blocksize,
                    latency="high",
                    callback=callback,
                )
                stream.start()
                return stream
            except Exception as exc:
                last = exc
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass
    if last is None:
        raise RuntimeError("No speaker output channels are available.")
    raise last


def play_chime(samples: np.ndarray, sample_rate: int = 16000) -> bool:
    """Play a chime if PortAudio is available. Returns False instead of raising.

    The output stream is kept on this function so the callback can finish
    after the call returns. Replacing it closes the previous chime.
    """
    output = AudioOutput(sample_rate)
    if not output.start():
        return False
    output.play(samples)
    previous = getattr(play_chime, "_output", None)
    play_chime._output = output
    if previous is not None:
        previous.close()
    return True
