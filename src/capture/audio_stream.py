"""Microphone and synthetic audio capture with a thread-safe ring buffer."""

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
        self.buffer = RingBuffer(max(block_size, int(sample_rate * buffer_seconds)))
        self._stream = None
        self._status_overflows = 0

    def start(self) -> None:
        try:
            import sounddevice as sd
        except OSError as exc:
            raise AudioCaptureError(
                "PortAudio library not found. Install libportaudio2, "
                "or run `python main.py --demo` without a microphone."
            ) from exc
        except ImportError as exc:  # pragma: no cover
            raise AudioCaptureError("sounddevice is not installed") from exc

        def callback(indata, frames, time_info, status) -> None:  # noqa: ARG001
            if status:
                self._status_overflows += 1
            self.buffer.write(indata[:, 0] if indata.ndim == 2 else indata)

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                blocksize=self.block_size,
                dtype="float32",
                callback=callback,
            )
            self._stream.start()
        except Exception as exc:
            raise AudioCaptureError(
                f"Could not open the microphone ({exc}). Run with --demo to skip audio hardware."
            ) from exc

    def read_latest(self, seconds: float) -> AudioChunk:
        count = int(seconds * self.sample_rate)
        samples = self.buffer.read_latest(count)
        return AudioChunk(samples=samples, timestamp=0.0, sample_rate=self.sample_rate)

    def pop_ready(self) -> AudioChunk:
        samples = self.buffer.pop_all()
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
