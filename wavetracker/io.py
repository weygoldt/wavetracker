"""Loading recordings and streaming them in frame-aligned blocks."""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from thunderlab.dataloader import DataLoader

AUDIO_SUFFIXES = (".wav", ".flac", ".ogg", ".mp3", ".raw")


def resolve_input(path: str | Path) -> str | list[str]:
    """Turn a file or recording directory into something `DataLoader` opens.

    A directory either contains a fishgrid recording (``traces-grid*.raw``),
    or a sequence of audio files that are loaded as one continuous recording.
    """
    path = Path(path)
    if path.is_file():
        return str(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    raw = sorted(path.glob("traces-grid*.raw"))
    if raw:
        return str(raw[0])
    files = sorted(p for p in path.iterdir() if p.suffix.lower() in AUDIO_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"No recordings found in {path}")
    return str(files[0]) if len(files) == 1 else [str(f) for f in files]


@dataclass(frozen=True)
class RecordingInfo:
    path: str
    rate: float
    channels: int
    frames: int

    @property
    def duration(self) -> float:
        return self.frames / self.rate


def open_recording(path: str | Path, buffersize: float = 60.0) -> DataLoader:
    return DataLoader(resolve_input(path), buffersize=buffersize)


def recording_info(path: str | Path) -> RecordingInfo:
    with open_recording(path, buffersize=1.0) as data:
        return RecordingInfo(str(path), float(data.rate), data.channels, len(data))


@dataclass(frozen=True)
class Block:
    data: np.ndarray
    """float32 array of shape (channels, samples)."""
    first_frame: int
    """Index of the first STFT frame computed from this block."""
    n_frames: int


@dataclass(frozen=True)
class FrameLayout:
    """How a sample range maps onto contiguous STFT frames.

    Frame ``k`` covers samples ``[start + k*step, start + k*step + nfft)``,
    so blocks can be processed independently without edge artifacts.
    """

    start: int
    stop: int
    nfft: int
    step: int
    rate: float

    @property
    def n_frames(self) -> int:
        n = self.stop - self.start
        return 0 if n < self.nfft else (n - self.nfft) // self.step + 1

    def times(self) -> np.ndarray:
        """Time of each frame's window center relative to recording start [s]."""
        k = np.arange(self.n_frames)
        return (self.start + k * self.step + self.nfft / 2) / self.rate


def iter_blocks(
    data: DataLoader,
    layout: FrameLayout,
    frames_per_block: int,
    channels: np.ndarray | None = None,
    prefetch: int = 2,
) -> Iterator[Block]:
    """Yield frame-aligned float32 blocks, reading ahead in a background thread.

    `channels` selects a subset of channels (default: all).
    """
    n_frames = layout.n_frames
    q: queue.Queue = queue.Queue(maxsize=prefetch)
    sentinel = object()
    stop = threading.Event()

    def reader() -> None:
        try:
            for f0 in range(0, n_frames, frames_per_block):
                if stop.is_set():
                    return
                nf = min(frames_per_block, n_frames - f0)
                s0 = layout.start + f0 * layout.step
                s1 = s0 + (nf - 1) * layout.step + layout.nfft
                block = data[s0:s1]
                if channels is not None:
                    block = block[:, channels]
                block = np.ascontiguousarray(block.T, dtype=np.float32)
                q.put(Block(block, f0, nf))
        except BaseException as e:  # forwarded to the consumer
            q.put(e)
        finally:
            q.put(sentinel)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    try:
        while (item := q.get()) is not sentinel:
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        while thread.is_alive():  # unblock a reader waiting on a full queue
            try:
                q.get_nowait()
            except queue.Empty:
                thread.join(0.05)
