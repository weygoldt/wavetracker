"""Synthetic multi-electrode recordings of wave-type fish with ground truth.

Used for tests and benchmarks. Each fish has a smoothly drifting EOD
frequency, a harmonic waveform and a slowly moving position along a linear
electrode array that sets its amplitude on every channel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Fish:
    base_freq: float
    """Mean EOD frequency [Hz]."""
    drift: float = 2.0
    """Standard deviation of the slow frequency drift [Hz]."""
    harmonics: tuple[float, ...] = (1.0, 0.6, 0.35, 0.2, 0.1)
    """Relative amplitudes of the harmonics."""
    amplitude: float = 1.0
    position: float = 0.5
    """Mean position along the electrode array (0-1)."""
    movement: float = 0.2
    """Standard deviation of the position random walk."""
    rises: list[tuple[float, float, float]] = field(default_factory=list)
    """(time [s], size [Hz], decay [s]) of rises."""
    spread: float = 0.25
    """Width of the amplitude falloff along the array (smaller: fewer
    electrodes see the fish)."""


@dataclass
class Hum:
    """Stationary interference comb: lines at k * spacing for k in teeth."""

    spacing: float = 95.4
    teeth: tuple[int, int] = (4, 30)
    """First and last harmonic number."""
    amplitude: float = 0.05
    channel_gains: tuple[float, ...] | None = None
    """Gain per channel (default: equal on all channels)."""
    flutter: float = 0.05
    """Relative amplitude fluctuation (random modulation on a ~1 min scale;
    the 95.4 Hz hum in the 2022 tube recordings varies by ~0.4 dB)."""


def _smooth_walk(
    n: int, sd: float, smooth: int, rng: np.random.Generator
) -> np.ndarray:
    """Zero-mean smooth random process with standard deviation `sd`."""
    if sd == 0 or n == 0:
        return np.zeros(n)
    x = np.cumsum(rng.standard_normal(n + smooth))
    x = np.convolve(x, np.hanning(smooth) / np.hanning(smooth).sum(), mode="valid")[:n]
    x -= x.mean()
    s = x.std()
    return x / s * sd if s > 0 else x


def frequency_trace(
    fish: Fish, times: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    f = fish.base_freq + _smooth_walk(len(times), fish.drift, 200, rng)
    for t0, size, tau in fish.rises:
        after = times >= t0
        f[after] += size * np.exp(-(times[after] - t0) / tau)
    return f


@dataclass
class SyntheticRecording:
    data: np.ndarray
    """float32 samples, shape (samples, channels)."""
    rate: float
    truth_times: np.ndarray
    """Times of the ground truth traces [s]."""
    truth_freqs: np.ndarray
    """Ground truth frequencies, shape (fish, times)."""
    truth_amps: np.ndarray
    """Ground truth amplitude on each channel, shape (fish, times, channels)."""


def random_fish(
    n: int,
    duration: float,
    fmin: float = 450.0,
    fmax: float = 1100.0,
    min_df: float = 8.0,
    rng: np.random.Generator | None = None,
) -> list[Fish]:
    """`n` fish with base frequencies at least `min_df` apart."""
    rng = rng or np.random.default_rng()
    freqs: list[float] = []
    while len(freqs) < n:
        f = rng.uniform(fmin, fmax)
        if all(abs(f - g) >= min_df for g in freqs):
            freqs.append(f)
    return [
        Fish(
            base_freq=f,
            drift=rng.uniform(0.5, 3.0),
            amplitude=rng.uniform(0.3, 1.0),
            position=rng.uniform(0.0, 1.0),
            movement=rng.uniform(0.05, 0.3),
            rises=[(rng.uniform(0, duration), rng.uniform(5, 20), rng.uniform(2, 10))]
            if rng.random() < 0.3
            else [],
        )
        for f in freqs
    ]


def synthesize(
    fish: list[Fish],
    duration: float,
    hum: list[Hum] | None = None,
    rate: float = 20000.0,
    channels: int = 8,
    noise: float = 0.05,
    mains: float = 0.02,
    truth_rate: float = 20.0,
    rng: np.random.Generator | None = None,
) -> SyntheticRecording:
    rng = rng or np.random.default_rng()
    n = int(duration * rate)
    t = np.arange(n) / rate
    truth_times = np.arange(0, duration, 1.0 / truth_rate)
    electrodes = np.linspace(0, 1, channels)

    data = noise * rng.standard_normal((n, channels)).astype(np.float32)
    for h in range(1, 6):
        data += (mains / h * np.sin(2 * np.pi * 50 * h * t)).astype(np.float32)[:, None]

    truth_f = np.empty((len(fish), len(truth_times)))
    truth_a = np.empty((len(fish), len(truth_times), channels))
    for k, fi in enumerate(fish):
        f = frequency_trace(fi, truth_times, rng)
        pos = np.clip(
            fi.position + _smooth_walk(len(truth_times), fi.movement, 400, rng),
            -0.2,
            1.2,
        )
        # amplitude decays with distance to each electrode
        amp = fi.amplitude / (
            1.0 + ((electrodes[None, :] - pos[:, None]) / fi.spread) ** 2
        )
        truth_f[k], truth_a[k] = f, amp

        phase = 2 * np.pi * np.cumsum(np.interp(t, truth_times, f)) / rate
        wave = np.zeros(n)
        for h, a in enumerate(fi.harmonics, start=1):
            wave += a * np.sin(h * phase + rng.uniform(0, 2 * np.pi))
        for c in range(channels):
            data[:, c] += (wave * np.interp(t, truth_times, amp[:, c])).astype(
                np.float32
            )

    for h in hum or []:
        gains = (
            np.ones(channels)
            if h.channel_gains is None
            else np.asarray(h.channel_gains)
        )
        wave = np.zeros(n)
        for k in range(h.teeth[0], h.teeth[1] + 1):
            if k * h.spacing >= rate / 2:
                break
            wave += np.sin(2 * np.pi * k * h.spacing * t + rng.uniform(0, 2 * np.pi))
        env = 1.0 + _smooth_walk(len(truth_times), h.flutter, int(60 * truth_rate), rng)
        wave *= h.amplitude * np.interp(t, truth_times, env)
        data += (wave[:, None] * gains[None, :]).astype(np.float32)

    return SyntheticRecording(data, rate, truth_times, truth_f, truth_a)


def save_recording(rec: SyntheticRecording, path: str | Path) -> Path:
    """Write the data as wav and the ground truth as ``<name>_truth.npz``."""
    from audioio import write_audio

    path = Path(path)
    scale = 0.9 / np.abs(rec.data).max()
    write_audio(str(path), rec.data * scale, rec.rate)
    truth = path.with_name(path.stem + "_truth.npz")
    np.savez(truth, times=rec.truth_times, freqs=rec.truth_freqs, amps=rec.truth_amps)
    return truth
