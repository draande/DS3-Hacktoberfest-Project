import threading

import numpy as np
import pytest

from src.capture.audio_stream import (
    MicrophoneStream,
    RingBuffer,
    StreamResampler,
    SyntheticAudioSource,
    block_latency_ms,
    sine_generator,
)
from src.capture.video_stream import (
    FileVideoSource,
    FrameQueue,
    SyntheticVideoSource,
    VideoFrame,
    stream_drift_ms,
)
from src.config_loader import load_config


def test_default_audio_block_is_under_20ms():
    config = load_config()
    latency = block_latency_ms(config.audio.sample_rate, config.audio.block_size)
    assert latency == pytest.approx(16.0)
    assert latency < 20.0


def test_ring_buffer_keeps_the_newest_samples_on_overflow():
    buffer = RingBuffer(8)
    buffer.write(np.arange(6, dtype=np.float32))
    buffer.write(np.arange(10, 14, dtype=np.float32))
    assert buffer.overflows == 1
    assert buffer.samples_dropped == 2
    assert buffer.read_latest(8).tolist() == [2, 3, 4, 5, 10, 11, 12, 13]


def test_ring_buffer_overwrite_when_write_exceeds_capacity():
    buffer = RingBuffer(4)
    buffer.write(np.arange(10, dtype=np.float32))
    assert buffer.read_latest(4).tolist() == [6, 7, 8, 9]
    assert buffer.overflows == 1


def test_ring_buffer_pop_all_empties_the_buffer():
    buffer = RingBuffer(8)
    buffer.write(np.arange(5, dtype=np.float32))
    assert buffer.pop_all().tolist() == [0, 1, 2, 3, 4]
    assert buffer.available() == 0
    assert buffer.pop_all().size == 0


def test_ring_buffer_is_safe_for_concurrent_writes():
    buffer = RingBuffer(1000)

    def writer():
        for _ in range(50):
            buffer.write(np.ones(10, dtype=np.float32))

    threads = [threading.Thread(target=writer) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert buffer.available() == 1000
    assert buffer.overflows >= 1
    assert np.all(buffer.read_latest(1000) == 1)


def test_frame_queue_drops_frames_that_arrive_too_fast():
    queue = FrameQueue(max_frames=4, target_fps=10)
    kept = []
    for index, timestamp in enumerate((0.0, 0.01, 0.02, 0.10, 0.20)):
        frame = VideoFrame(np.zeros((2, 2, 3), np.uint8), timestamp, index)
        if queue.push(frame):
            kept.append(timestamp)
    assert kept == [0.0, 0.10, 0.20]
    assert queue.dropped_pacing == 2


def test_frame_queue_drops_the_oldest_frame_when_full():
    queue = FrameQueue(max_frames=2, target_fps=0)
    for index in range(4):
        queue.push(VideoFrame(np.full((2, 2, 3), index, np.uint8), index / 10, index))
    assert queue.dropped_overflow == 2
    first = queue.pop()
    second = queue.pop()
    assert first.image[0, 0, 0] == 2
    assert second.image[0, 0, 0] == 3
    assert queue.pop() is None


def test_synthetic_sources_are_monotonic_and_drift_is_in_milliseconds():
    assert stream_drift_ms(1.0, 0.984) == pytest.approx(16.0)
    assert abs(stream_drift_ms(1.016, 1.0)) < 20.0
    video = SyntheticVideoSource(width=16, height=12, count=5, fps=30)
    audio = SyntheticAudioSource(
        sine_generator(440, 0.1, 16000), sample_rate=16000, block_size=256
    )
    previous_video = -1.0
    previous_audio = -1.0
    for _ in range(5):
        frame = video.read()
        chunk = audio.read()
        assert frame is not None
        assert frame.timestamp > previous_video
        assert chunk.timestamp > previous_audio
        previous_video = frame.timestamp
        previous_audio = chunk.timestamp
    assert video.read() is None
    video.close()
    audio.close()


def test_file_video_round_trip(tmp_path):
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10,
        (32, 24),
    )
    if not writer.isOpened():
        pytest.skip("MJPG video writer is not available")
    for index in range(5):
        writer.write(np.full((24, 32, 3), index * 20, np.uint8))
    writer.release()
    source = FileVideoSource(path, fps=10)
    frames = []
    while (frame := source.read()) is not None:
        frames.append(frame)
    source.close()
    assert len(frames) == 5
    assert frames[1].timestamp == pytest.approx(0.1)


def test_resampler_keeps_level_and_joins_chunks():
    source_rate = 48000
    target_rate = 16000
    seconds = 0.25
    times = np.arange(int(seconds * source_rate)) / source_rate
    tone = (0.25 * np.sin(2 * np.pi * 440.0 * times)).astype(np.float32)
    whole = StreamResampler(source_rate, target_rate).process(tone)
    joined = []
    chunked = StreamResampler(source_rate, target_rate)
    for piece in np.array_split(tone, 8):
        joined.append(chunked.process(piece))
    stitched = np.concatenate(joined) if joined else np.zeros(0, np.float32)
    assert whole.size == pytest.approx(seconds * target_rate, abs=5)
    count = min(whole.size, stitched.size)
    assert count > 1000
    assert np.allclose(whole[:count], stitched[:count], atol=1e-5)
    assert float(np.max(np.abs(whole))) == pytest.approx(0.25, abs=0.02)


def test_microphone_opens_at_the_device_rate_and_returns_16k():
    class FakeInput:
        def __init__(self) -> None:
            self.calls: list[int] = []
            self.kwargs = None
            self.callback = None

        def query_devices(self, kind=None):
            assert kind == "input"
            return {
                "default_samplerate": 48000.0,
                "max_input_channels": 2,
                "name": "mic",
            }

        def InputStream(self, **kwargs):
            self.calls.append(kwargs["channels"])
            if kwargs["channels"] == 1:
                raise OSError("Invalid number of channels")
            self.kwargs = kwargs
            self.callback = kwargs["callback"]
            return self

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

        def close(self) -> None:
            return None

    device = FakeInput()
    device.events = []
    device.start = lambda: device.events.append("start")
    device.stop = lambda: device.events.append("stop")
    mic = MicrophoneStream(sample_rate=16000, buffer_seconds=1.0)
    mic.start(sd_module=device)
    assert device.calls == [2]
    assert device.kwargs["samplerate"] == 48000
    assert device.kwargs["blocksize"] == 768
    # The first start() alone leaves the callback silent on PortAudio.
    assert device.events[:3] == ["start", "stop", "start"]
    stereo = np.zeros((4800, 2), np.float32)
    stereo[:, 0] = 0.4
    device.callback(stereo, 4800, None, None)
    chunk = mic.pop_ready()
    mic.close()
    assert chunk.sample_rate == 16000
    assert chunk.samples.size == pytest.approx(1600, abs=5)
    assert float(np.mean(chunk.samples)) == pytest.approx(0.2, abs=0.02)


def test_microphone_falls_back_to_mono_when_stereo_is_refused():
    class MonoOnly:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def query_devices(self, kind=None):
            return {
                "default_samplerate": 48000.0,
                "max_input_channels": 2,
                "name": "mic",
            }

        def InputStream(self, **kwargs):
            self.calls.append(kwargs["channels"])
            if kwargs["channels"] != 1:
                raise OSError("stereo refused")
            return self

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

        def close(self) -> None:
            return None

    device = MonoOnly()
    mic = MicrophoneStream(sample_rate=16000)
    mic.start(sd_module=device)
    # Stereo is tried at the 16 ms block and again with PortAudio's period
    # before the open falls back to mono.
    assert device.calls == [2, 2, 1]
    mic.close()


def test_microphone_reports_missing_portaudio(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sounddevice":
            raise OSError("PortAudio library not found")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    stream = MicrophoneStream()
    with pytest.raises(Exception, match="PortAudio"):
        stream.start()
