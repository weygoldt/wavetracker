"""Loading recordings and streaming them in frame-aligned blocks."""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from thunderlab.dataloader import DataLoader

AUDIO_SUFFIXES = (".wav", ".flac", ".ogg", ".mp3", ".raw")

#: Feature flag that callers (e.g. the audian runner) check: `resolve_input`,
#: `open_recording`, `recording_info` and `pipeline.detect` accept a sequence
#: of paths as one recording split over several files.
MULTI_INPUT = True


def resolve_input(path: str | Path | Sequence[str | Path]) -> str | list[str]:
    """Turn a file or recording directory into something `DataLoader` opens.

    A directory either contains a fishgrid recording (``traces-grid*.raw``),
    or a sequence of audio files that are loaded as one continuous recording.

    A list or tuple of paths is a recording split over several files, in the
    order given; each must be an existing file.  A one-element sequence is the
    same as that path.
    """
    if isinstance(path, (list, tuple)):
        files = [Path(p) for p in path]
        if not files:
            raise FileNotFoundError("empty list of recordings")
        missing = [str(p) for p in files if not p.is_file()]
        if missing:
            raise FileNotFoundError(", ".join(missing))
        return str(files[0]) if len(files) == 1 else [str(p) for p in files]
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


def open_recording(
    path: str | Path | Sequence[str | Path], buffersize: float = 60.0
) -> DataLoader | JoinedRecording:
    """Open a recording file, directory, or sequence of files (see
    `resolve_input`).  Several files are joined by `JoinedRecording`."""
    if isinstance(path, (list, tuple)) and len(path) > 1:
        return _open_joined(resolve_input(path), buffersize)
    return DataLoader(resolve_input(path), buffersize=buffersize)


def _open_joined(files: list[str], buffersize: float) -> JoinedRecording:
    """Concatenate every file, whatever their timestamps say.

    thunderlab compares each file's start time with the end of the previous
    one and, depending on its version, refuses the sequence or drops that file
    and every later one on a mismatch.  That loses the tail of TASCAM sessions
    (bext time is when a file was closed).  The caller named these files; all
    of them are read, in the order given.
    """
    return JoinedRecording(files, buffersize)


class JoinedRecording:
    """Several recording files read as one, in the order given.

    Read-only and with the part of the `DataLoader` interface the pipeline
    uses: ``rate``, ``channels``, ``frames``, ``len()``, ``[rows]`` and
    ``[rows, channels]`` returning ``(samples, channels)`` arrays, ``close()``
    and the context manager.  All files must share rate and channel count.
    """

    #: Loaders kept open at once.  A read touches one file, or two at a
    #: boundary; every other file is closed so that neither its buffer nor
    #: its handle grows with the number of files.
    max_open = 2

    def __init__(self, files: Sequence[str | Path], buffersize: float = 60.0):
        self.file_paths = [str(f) for f in files]
        self.filepath = self.file_paths[0]
        self.buffersize = buffersize
        self._open: dict[int, DataLoader] = {}
        sizes = []
        rate = channels = None
        for f in self.file_paths:
            # Probe with a tiny buffer and close again: only the shape is kept.
            ld = DataLoader(f, buffersize=1.0)
            try:
                if rate is None:
                    rate, channels = float(ld.rate), int(ld.channels)
                elif float(ld.rate) != rate:
                    raise ValueError(
                        f"sampling rates differ: {ld.rate} Hz in {f} versus "
                        f"{rate} Hz in {self.filepath}"
                    )
                elif int(ld.channels) != channels:
                    raise ValueError(
                        f"number of channels differs: {ld.channels} in {f} "
                        f"versus {channels} in {self.filepath}"
                    )
                sizes.append(len(ld))
            finally:
                ld.close()
        self.rate = rate
        self.channels = channels
        sizes = np.array(sizes, dtype=np.int64)
        self.end_indices = np.cumsum(sizes)
        self.start_indices = self.end_indices - sizes
        self.frames = int(self.end_indices[-1])

    def _loader(self, i: int) -> DataLoader:
        ld = self._open.pop(i, None)
        if ld is None:
            ld = DataLoader(self.file_paths[i], buffersize=self.buffersize)
        self._open[i] = ld  # most recently used last
        while len(self._open) > self.max_open:
            oldest = next(iter(self._open))
            self._open.pop(oldest).close()
        return ld

    def __len__(self) -> int:
        return self.frames

    def __getitem__(self, key):
        cols = slice(None)
        if isinstance(key, tuple):
            key, cols = key
        if isinstance(key, (int, np.integer)):
            k = int(key) + (self.frames if key < 0 else 0)
            if not 0 <= k < self.frames:
                raise IndexError(key)
            return self[k : k + 1, cols][0]
        if not isinstance(key, slice):
            raise TypeError(f"unsupported index {key!r}")
        s0, s1, step = key.indices(self.frames)
        if step != 1:
            raise IndexError("JoinedRecording supports contiguous slices only")
        parts = []
        for i, (a, b) in enumerate(
            zip(self.start_indices, self.end_indices, strict=True)
        ):
            lo, hi = max(s0, a), min(s1, b)
            if lo < hi:
                parts.append(np.asarray(self._loader(i)[lo - a : hi - a])[:, cols])
        if not parts:
            return np.empty((0, self.channels))[:, cols]
        return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)

    def close(self) -> None:
        while self._open:
            self._open.popitem()[1].close()

    def __enter__(self) -> JoinedRecording:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def recording_info(path: str | Path | Sequence[str | Path]) -> RecordingInfo:
    with open_recording(path, buffersize=1.0) as data:
        name = str(path) if not isinstance(path, (list, tuple)) else str(path[0])
        return RecordingInfo(name, float(data.rate), data.channels, len(data))


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
