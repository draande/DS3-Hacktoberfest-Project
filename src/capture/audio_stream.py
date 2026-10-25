"""Microphone and synthetic audio capture with a thread-safe ring buffer.

The analyzers stay at 16 kHz. Capture opens at the device's native rate
(commonly 48 kHz on Windows) and :class:`StreamResampler` converts on the way
into the ring the rest of the app reads. A 16 kHz mono stream is a common
reason the device accepts the open and then stays silent. The input stream
is also stopped and started once after open: PortAudio otherwise reports the
stream active without ever calling the callback.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np


def block_latency_ms(sample_rate: int, block_size: int) -> float:
    """Capture latency of one audio block, in milliseconds."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    return 1000.0 * block_size / sample_rate


class RingBuffer:
    """Fixed-capacity float32 ring buffer. Writes past capacity drop the oldest samples."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._buf = np.zeros(self.capacity, dtype=np.float32)
        self._write = 0
        self._count = 0
        self.overflows = 0
        self.samples_dropped = 0
        self._lock = threading.Lock()

    def write(self, samples: np.ndarray) -> int:
        data = np.asarray(samples, dtype=np.float32).reshape(-1)
        n = int(data.size)
        if n == 0:
            return 0
        with self._lock:
            if n >= self.capacity:
                self.samples_dropped += (n - self.capacity) + self._count
                self.overflows += 1
                self._buf[:] = data[-self.capacity :]
                self._write = 0
                self._count = self.capacity
                return self.capacity
            free = self.capacity - self._count
            if n > free:
                drop = n - free
                self.overflows += 1
                self.samples_dropped += drop
                self._count -= drop
            first = min(n, self.capacity - self._write)
            self._buf[self._write : self._write + first] = data[:first]
            if first < n:
                self._buf[: n - first] = data[first:]
            self._write = (self._write + n) % self.capacity
            self._count += n
            return n

    def read_latest(self, n: int) -> np.ndarray:
        with self._lock:
            count = min(int(n), self._count)
            if count <= 0:
                return np.zeros(0, dtype=np.float32)
            start = (self._write - count) % self.capacity
            if start + count <= self.capacity:
                return self._buf[start : start + count].copy()
            head = self.capacity - start
            return np.concatenate((self._buf[start:], self._buf[: count - head]))

    def available(self) -> int:
        with self._lock:
            return self._count

    def pop_all(self) -> np.ndarray:
        """Remove and return every sample currently stored, oldest first."""
        with self._lock:
            count = self._count
            if count <= 0:
                return np.zeros(0, dtype=np.float32)
            start = (self._write - count) % self.capacity
            if start + count <= self.capacity:
                data = self._buf[start : start + count].copy()
            else:
                head = self.capacity - start
                data = np.concatenate((self._buf[start:], self._buf[: count - head]))
            self._count = 0
            return data

    def clear(self) -> None:
        with self._lock:
            self._write = 0
            self._count = 0


@dataclass
class AudioChunk:
    samples: np.ndarray
    timestamp: float
    sample_rate: int


class SyntheticAudioSource:
    """Pull-based generator. `generator(start_sample, count)` returns PCM float32."""

    def __init__(
        self,
        generator,
        sample_rate: int = 16000,
        block_size: int = 256,
    ) -> None:
        self.generator = generator
        self.sample_rate = sample_rate
        self.block_size = block_size
        self._cursor = 0

    def read(self) -> AudioChunk:
        samples = np.asarray(
            self.generator(self._cursor, self.block_size), dtype=np.float32
        ).reshape(-1)
        timestamp = self._cursor / self.sample_rate
        self._cursor += int(samples.size)
        return AudioChunk(
            samples=samples, timestamp=timestamp, sample_rate=self.sample_rate
        )

    def close(self) -> None:
        return None


class StreamResampler:
    """Streaming linear resampler. Chunks concatenate without a gap or overlap."""

    def __init__(self, source_rate: int, target_rate: int) -> None:
        if source_rate <= 0 or target_rate <= 0:
            raise ValueError("sample rates must be positive")
        self.source_rate = int(source_rate)
        self.target_rate = int(target_rate)
        self._pending = np.zeros(0, dtype=np.float64)
        self._pos = 0.0

    def process(self, samples: np.ndarray) -> np.ndarray:
        incoming = np.asarray(samples, dtype=np.float64).reshape(-1)
        if incoming.size == 0:
            return np.zeros(0, dtype=np.float32)
        if self.source_rate == self.target_rate:
            return incoming.astype(np.float32)
        data = (
            np.concatenate((self._pending, incoming))
            if self._pending.size
            else incoming
        )
        if data.size < 2:
            self._pending = data
            return np.zeros(0, dtype=np.float32)
        step = self.source_rate / self.target_rate
        limit = data.size - 1
        if self._pos >= limit:
            self._pending = data
            return np.zeros(0, dtype=np.float32)
        count = int((limit - self._pos) / step)
        if count <= 0:
            self._pending = data
            return np.zeros(0, dtype=np.float32)
        positions = self._pos + step * np.arange(count, dtype=np.float64)
        index = positions.astype(np.int64)
        fraction = positions - index
        values = data[index] * (1.0 - fraction) + data[index + 1] * fraction
        end = float(positions[-1] + step)
        keep = min(data.size, int(end))
        self._pending = data[keep:]
        self._pos = end - keep
        return values.astype(np.float32)


def sine_generator(frequency: float, amplitude: float, sample_rate: int):
    def generate(start: int, count: int) -> np.ndarray:
        t = (start + np.arange(count, dtype=np.float32)) / sample_rate
        return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)

    return generate


class AudioCaptureError(RuntimeError):
    """Raised when the microphone cannot be opened."""


class MicrophoneStream:
    """Non-blocking mic capture into a ring buffer.

    sounddevice is imported lazily so tests and demo mode do not need PortAudio.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        block_size: int = 256,
        buffer_seconds: float = 2.5,
    ) -> None:
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.buffer_seconds = buffer_seconds
        self.capture_rate = sample_rate
        self.buffer = RingBuffer(max(block_size, int(sample_rate * buffer_seconds)))
        self._resampler = StreamResampler(sample_rate, sample_rate)
        self._stream = None
        self._status_overflows = 0

    def start(self, sd_module=None) -> None:
        sd = sd_module if sd_module is not None else _load_sounddevice()
        try:
            info = sd.query_devices(kind="input")
            rate = int(round(float(info.get("default_samplerate") or 0)))
            max_channels = int(info.get("max_input_channels") or 0)
        except Exception as exc:
            raise AudioCaptureError(_microphone_error(exc)) from exc
        if rate <= 0:
            rate = self.sample_rate
        if max_channels < 1:
            raise AudioCaptureError(
                "No microphone input channels are available. "
                "Check that a microphone is connected and that the OS allows apps to use it."
            )
        self.capture_rate = rate
        self.buffer = RingBuffer(max(self.block_size, int(rate * self.buffer_seconds)))
        self._resampler = StreamResampler(rate, self.sample_rate)
        try:
            self._stream = _open_input_stream(sd, self._on_audio, rate, max_channels)
        except Exception as exc:
            raise AudioCaptureError(_microphone_error(exc)) from exc

    def _on_audio(self, indata, frames, time_info, status) -> None:  # noqa: ARG001
        try:
            if status:
                self._status_overflows += 1
            data = np.asarray(indata, dtype=np.float32)
            if data.ndim == 2 and data.shape[1] > 1:
                mono = data.mean(axis=1)
            elif data.ndim == 2:
                mono = data[:, 0]
            else:
                mono = data.reshape(-1)
            self.buffer.write(mono)
        except Exception:
            self._status_overflows += 1

    def read_latest(self, seconds: float) -> AudioChunk:
        count = int(seconds * self.capture_rate)
        native = self.buffer.read_latest(count)
        samples = StreamResampler(self.capture_rate, self.sample_rate).process(native)
        return AudioChunk(samples=samples, timestamp=0.0, sample_rate=self.sample_rate)

    def pop_ready(self) -> AudioChunk:
        samples = self._resampler.process(self.buffer.pop_all())
        return AudioChunk(samples=samples, timestamp=0.0, sample_rate=self.sample_rate)

    def close(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is not None:
            stream.stop()
            stream.close()

    def __enter__(self) -> MicrophoneStream:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _load_sounddevice():
    try:
        import sounddevice as sd
    except OSError as exc:
        raise AudioCaptureError(
            "PortAudio library not found. Install libportaudio2 "
            "(or the PortAudio runtime on Windows), "
            "or run `python main.py --demo` without a microphone."
        ) from exc
    except ImportError as exc:  # pragma: no cover
        raise AudioCaptureError("sounddevice is not installed") from exc
    return sd


def _microphone_error(exc: Exception) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    hint = ""
    lowered = text.lower()
    if any(word in lowered for word in ("denied", "permission", "access", "privacy")):
        hint = (
            " On Windows, allow microphone access in Settings > Privacy > Microphone."
        )
    return (
        f"Could not open the microphone ({text}).{hint} "
        "Run with --demo to skip audio hardware."
    )


def _native_blocksize(rate: int) -> int:
    """About 16 ms, which stays under the 20 ms capture budget.

    ``blocksize=0`` lets PortAudio pick the period. On the Pulse plugin that
    period is short enough that a Python callback underruns on every block, so
    playback collapses into clicks and the recording is mostly silence.
    """
    return max(64, int(round(rate * 0.016)))


def _channel_choices(max_channels: int) -> list[int]:
    """Prefer stereo. A mono stream can open and then never deliver samples."""
    if max_channels >= 2:
        return [2, 1]
    if max_channels == 1:
        return [1]
    return []


def _prime_input_stream(stream) -> None:
    """Make PortAudio actually call the input callback.

    On the ALSA Pulse plugin the first ``start()`` reports the stream active
    and then never delivers a buffer. The same stall shows up on some other
    PortAudio hosts. Stopping and starting again is what makes samples arrive.
    A later output stream does not repair an input that was left unprimed, so
    capture stays silent for the whole session.
    """
    stream.start()
    stream.stop()
    stream.start()


def _open_input_stream(sd, callback, rate: int, max_channels: int):
    """Open at the native rate. Stereo first, then mono.

    A ~16 ms block is requested first. If the device rejects it, PortAudio's
    own period (``blocksize=0``) is tried before the next channel count.
    """
    choices = _channel_choices(max_channels)
    last: Exception | None = None
    for channels in choices:
        for blocksize in (_native_blocksize(rate), 0):
            stream = None
            try:
                stream = sd.InputStream(
                    samplerate=rate,
                    channels=channels,
                    dtype="float32",
                    blocksize=blocksize,
                    latency="high",
                    callback=callback,
                )
                _prime_input_stream(stream)
                return stream
            except Exception as exc:
                last = exc
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass
    if last is None:
        raise AudioCaptureError("No microphone input channels are available.")
    raise last
