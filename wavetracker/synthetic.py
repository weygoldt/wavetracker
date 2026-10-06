"""Synthetic multi-electrode recordings of wave-type fish with ground truth.

Used for tests and benchmarks. Each fish has a smoothly drifting EOD
frequency, a harmonic waveform and a slowly moving position along a linear
electrode array that sets its amplitude on every channel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .results import Results


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


# --- moving electrodes ---------------------------------------------------------


@dataclass
class StationaryFish:
    """A resting fish for the moving-electrode scenario."""

    freq: float
    """EOD frequency at the start of the survey [Hz]."""
    x: float
    y: float
    depth: float = 0.4
    heading: float = 0.0
    """Head direction [rad]."""
    strength: float = 1.0
    """Factor on the forward model's potential."""
    drift: float = 0.0
    """Linear frequency change over the survey [Hz]."""


def boat_survey(
    length: float = 10.0,
    width: float = 6.0,
    lane_spacing: float = 1.0,
    speed: float = 0.3,
    separation: float = 0.6,
    electrode_z: float = -0.05,
    step: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """A boat towing two electrodes and a hull reference in lanes.

    The boat travels back and forth along x (lawnmower pattern, lanes
    `lane_spacing` apart in y). Returns times (T,) and positions (T, 3, 3) of
    the port tip, starboard tip and hull reference (centre between the tips),
    i.e. channels electrode 0 - 2 and 1 - 2 (``reference=2``).
    """
    ys = np.arange(0.0, width + 1e-9, lane_spacing)
    vertices = []
    for k, y in enumerate(ys):
        xs = (0.0, length) if k % 2 == 0 else (length, 0.0)
        vertices += [(xs[0], y), (xs[1], y)]
    v = np.array(vertices)
    seg_len = np.hypot(*np.diff(v, axis=0).T)
    s_v = np.r_[0.0, np.cumsum(seg_len)]
    s = np.arange(0.0, s_v[-1], speed * step)
    centre = np.c_[np.interp(s, s_v, v[:, 0]), np.interp(s, s_v, v[:, 1])]
    seg = np.clip(np.searchsorted(s_v, s, side="right") - 1, 0, len(seg_len) - 1)
    direction = np.diff(v, axis=0)[seg] / seg_len[seg][:, None]
    left = np.c_[-direction[:, 1], direction[:, 0]]
    pos = np.empty((len(s), 3, 3))
    pos[:, 0, :2] = centre + left * separation / 2
    pos[:, 1, :2] = centre - left * separation / 2
    pos[:, 2, :2] = centre
    pos[:, :, 2] = electrode_z
    return s / speed, pos


@dataclass
class SurveyDetections:
    results: Results
    """Detections as wavetracker writes them (ident_v: one identity per pass)."""
    fish: np.ndarray
    """True fish of each detection (-1: clutter)."""
    time: np.ndarray
    """Electrode track times [s]."""
    positions: np.ndarray
    """Electrode positions (T, E, 3)."""
    reference: int | None


def simulate_survey(
    fish: list[StationaryFish],
    time: np.ndarray,
    positions: np.ndarray,
    reference: int | None = 2,
    model=None,
    frame_step: float = 0.15,
    noise: float = 0.03,
    snr_threshold: float = 8.0,
    miss_prob: float = 0.05,
    amp_jitter: float = 0.1,
    freq_noise: float = 0.05,
    link_gap: float = 2.0,
    n_clutter: int = 0,
    fmin: float = 400.0,
    fmax: float = 900.0,
    rng: np.random.Generator | None = None,
) -> SurveyDetections:
    """Detections of stationary fish by moving electrodes.

    Channel signals come from `model` (a
    :class:`wavetracker.position.efield.SourceModel`, default: monopole line
    with a bottom at 1.2 m) with a common random phase per frame, a log-normal
    amplitude jitter and complex Gaussian noise of rms `noise` per channel. A
    fish is detected in a frame if its total power exceeds `snr_threshold`
    times the total noise power. Its detections are linked into one identity
    as long as the gaps are at most `link_gap` seconds. Clutter identities
    (5-12 detections near the noise level at random frequencies) are added.
    """
    from .position.efield import SourceModel
    from .position.electrodes import channel_map

    rng = rng or np.random.default_rng()
    model = model or SourceModel(water_depth=1.2)
    plus, minus = channel_map(positions.shape[1], reference)
    times = np.arange(time[0], time[-1], frame_step)
    pos = np.stack(
        [
            np.stack([np.interp(times, time, positions[:, e, a]) for a in range(3)], -1)
            for e in range(positions.shape[1])
        ],
        axis=1,
    )
    duration = times[-1] - times[0]
    n_ch = len(plus)
    rows = []  # (frame, freq, complex values, fish, identity)
    ident = 0
    for k, fi in enumerate(fish):
        v = model.potential(pos, fi.x, fi.y, fi.depth, fi.heading) * fi.strength
        sig = v[:, plus] - np.where(minus < 0, 0.0, v[:, np.maximum(minus, 0)])
        sig = sig * np.exp(amp_jitter * rng.standard_normal((len(times), 1)))
        z = sig + noise / np.sqrt(2) * (
            rng.standard_normal(sig.shape) + 1j * rng.standard_normal(sig.shape)
        )
        z *= np.exp(2j * np.pi * rng.random((len(times), 1)))
        det = (np.abs(z) ** 2).sum(1) > snr_threshold * n_ch * noise**2
        det &= rng.random(len(times)) > miss_prob
        frames = np.flatnonzero(det)
        f = fi.freq + fi.drift * (times[frames] - times[0]) / duration
        f = f + freq_noise * rng.standard_normal(len(frames))
        last = -np.inf
        for fr, ff in zip(frames, f, strict=True):
            if times[fr] - last > link_gap:
                ident += 1
            last = times[fr]
            rows.append((fr, ff, z[fr], k, ident))
    for _ in range(n_clutter):
        ident += 1
        f0 = rng.uniform(fmin, fmax)
        start = rng.integers(0, len(times) - 30)
        n = rng.integers(5, 13)
        for fr in np.sort(rng.choice(np.arange(start, start + 30), n, replace=False)):
            z = noise * 2 * (rng.standard_normal(n_ch) + 1j * rng.standard_normal(n_ch))
            rows.append((fr, f0 + 0.3 * rng.standard_normal(), z, -1, ident))
    rows.sort(key=lambda r: r[0])
    cplx = np.array([r[2] for r in rows]).reshape(-1, n_ch)
    results = Results(
        fund_v=np.array([r[1] for r in rows]),
        idx_v=np.array([r[0] for r in rows], dtype=int),
        sign_v=np.abs(cplx) ** 2,
        ident_v=np.array([r[4] for r in rows], dtype=float),
        times=times,
        meta={"synthetic": "moving electrodes"},
        cplx_v=cplx,
    )
    truth = np.array([r[3] for r in rows], dtype=int)
    return SurveyDetections(results, truth, np.asarray(time), positions, reference)
