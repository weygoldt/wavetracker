"""Dense synthetic mixtures of real two-fish tube recordings, with ground truth.

Each source (a fishgrid recording of two fish in tubes, 2022 tube competition)
is time-warped and the warped recordings are summed electrode by electrode.
The warp of source s is

    tau_s(t) = start + a_s * (t + D(t)),   D(t) = integral of eps(t) dt,

so all its frequencies are scaled by ``a_s * (1 + eps(t))``: ``a_s`` spreads
the fish of different sources over a wide range (harmonics of low fish land
among the fundamentals of high fish) and ``eps`` is a slow drift common to
all sources (temperature acting on every fish with the same Q10). The fish
keep their natural frequency modulations, rises, waveforms and electrode
patterns; modulation timescales are stretched by 1 / a_s.

Hum combs of the sources (stationary lines at multiples of 25, 50, 55.5,
95.5 Hz, ...) are removed before warping: warped, they would drift with
``eps`` and be tracked as fish. Teeth within 3 Hz of a fish band are kept.

Ground truth: every source segment is analysed alone (two fish, reliable),
its detections are assigned to the lower and upper fish with the pseudo
ground truth of ``tube_competition.py``, and the per-frame frequencies are
mapped into mixture time, f_mix(t) = f_src(tau_s(t)) * tau_s'(t).

Usage::

    python benchmarks/tube_mixture.py output/dev/mix0 --n-sources 8 --seed 0
    wavetracker run output/dev/mix0 -c benchmarks/tube_mixture.yaml -o output/dev/mix0/wt
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.signal import filtfilt, firwin, resample_poly

sys.path.insert(0, str(Path(__file__).parent))
from tube_competition import pseudo_truth

from wavetracker.config import Config
from wavetracker.pipeline import detect

RAW = Path("/mnt/data2/2022_tube_competition/raw")
RATE = 20000.0
CHANNELS = 11
UP = 4  # upsampling before interpolation
TRUTH_RATE = 10.0  # Hz
SCALES = (0.4, 0.5, 0.65, 0.8, 1.0, 1.2, 1.4, 1.6)


def common_drift(duration: float, amplitude: float, timescale: float, seed: int):
    """eps(t) on a 1 s grid: smooth random process, std `amplitude`."""
    rng = np.random.default_rng(seed)
    n = int(duration) + 2
    x = np.cumsum(rng.standard_normal(n + int(timescale)))
    k = np.hanning(max(int(timescale), 3))
    x = np.convolve(x, k / k.sum(), mode="valid")[:n]
    x -= x.mean()
    eps = x / x.std() * amplitude if amplitude > 0 else np.zeros(n)
    t = np.arange(n, dtype=float)
    d = np.concatenate([[0.0], np.cumsum(0.5 * (eps[1:] + eps[:-1]))])
    return t, eps, d


def warp(t, a, start, grid_t, grid_d):
    return start + a * (t + np.interp(t, grid_t, grid_d))


def raw_memmap(source: Path) -> np.memmap:
    return np.memmap(source / "traces-grid1.raw", dtype=np.float32, mode="r").reshape(
        -1, CHANNELS
    )


def hum_lines(x: np.ndarray, spacings, fish_bands, fmax=9500.0) -> np.ndarray:
    """Exact frequencies of the hum comb teeth in `x` (samples, channels):
    peaks of the long-term spectrum within 0.8 Hz of k * spacing that are
    10 dB above their surroundings and not inside a fish band."""
    from scipy.signal import welch

    f, p = welch(x, RATE, nperseg=2**18, axis=0)
    db = 10 * np.log10(p.sum(1) + 1e-30)
    df = f[1]
    lines = []
    for sp in spacings:
        for k in range(1, int(fmax / sp) + 1):
            m = np.abs(f - k * sp) <= 0.8
            if not m.any():
                continue
            i = np.flatnonzero(m)[db[m].argmax()]
            ctx = db[max(0, i - int(5 / df)) : i + int(5 / df)]
            if db[i] - np.median(ctx) < 10.0:
                continue
            a, b, c = db[i - 1 : i + 2]
            fi = f[i] + 0.5 * (a - c) / (a - 2 * b + c) * df
            if any(lo - 3 <= fi <= hi + 3 for lo, hi in fish_bands):
                continue
            lines.append(fi)
    return np.unique(np.round(lines, 3))


def remove_lines(x: np.ndarray, lines: np.ndarray, width: float = 0.45) -> np.ndarray:
    """Zero +-`width` Hz around `lines` in a long-window STFT, per channel."""
    from scipy.signal import istft, stft

    out = np.empty_like(x, dtype=np.float32)
    for c in range(x.shape[1]):
        f, _, z = stft(x[:, c], RATE, nperseg=2**16, noverlap=3 * 2**14)
        z = z.astype(np.complex64)
        mask = np.zeros(len(f), bool)
        for fl in lines:
            mask |= np.abs(f - fl) <= width
        z[mask] = 0
        _, y = istft(z, RATE, nperseg=2**16, noverlap=3 * 2**14)
        out[:, c] = y[: x.shape[0]]
    return out


def mix(args, segments, scales, grid):
    out = Path(args.out)
    n_out = int(args.duration * RATE)
    dst = np.memmap(
        out / "traces-grid1.raw", dtype=np.float32, mode="w+", shape=(n_out, CHANNELS)
    )
    chunk = int(30 * RATE)
    pad = 64
    filters = {}
    for s, a in enumerate(scales):
        if a > 1.0:  # source content above rate / 2 / a would alias
            filters[s] = firwin(255, 0.9 / a, fs=2.0)  # in units of Nyquist
    for c0 in range(0, n_out, chunk):
        c1 = min(n_out, c0 + chunk)
        t = np.arange(c0, c1) / RATE
        acc = np.zeros((c1 - c0, CHANNELS), np.float64)
        for s, ((data, seg0), a) in enumerate(zip(segments, scales, strict=True)):
            tau = warp(t, a, args.start, grid[0], grid[2]) - seg0
            i0 = int(np.floor(tau[0] * RATE)) - pad
            i1 = int(np.ceil(tau[-1] * RATE)) + pad
            x = np.asarray(data[i0:i1], dtype=np.float64)
            if s in filters:
                x = filtfilt(filters[s], [1.0], x, axis=0)
            xu = resample_poly(x, UP, 1, axis=0)
            pos = (tau * RATE - i0) * UP
            j = np.floor(pos).astype(int)
            w = (pos - j)[:, None]
            acc += (1 - w) * xu[j] + w * xu[j + 1]
        dst[c0:c1] = acc.astype(np.float32)
        print(f"  mixed {c1 / RATE:.0f} / {args.duration:.0f} s", flush=True)
    dst.flush()


def source_segment(args, source: str, a: float, grid):
    """Analyse the source segment used for scale `a` (cached): ground truth
    (2, T) in mixture time and frequency, NaN where the fish was not
    detected, and the segment without hum (memmap, first sample time)."""
    from wavetracker.results import Results

    t_mix = np.arange(0, args.duration, 1 / TRUTH_RATE)
    t_end = warp(np.array([args.duration]), a, args.start, grid[0], grid[2])[0]
    seg0, seg1 = args.start - 5.0, t_end + 5.0
    cache = Path(args.cache) / f"{source}_{seg0:.0f}_{seg1:.0f}"
    if not (cache / "fund_v.npy").exists():
        cfg = Config.from_dict({"harmonic_groups": {"min_freq": 400.0, "max_freq": 1200.0}})
        detect(RAW / source, cache, cfg, start=seg0, duration=seg1 - seg0)
    res = Results.load(cache)
    truth = pseudo_truth(res)  # (2, frames), source frequency
    t_det = res.times[res.idx_v]
    ref = truth[:, res.idx_v]
    k = np.abs(ref - res.fund_v).argmin(0)
    ok = np.abs(ref[k, np.arange(len(k))] - res.fund_v) < 30.0

    clean = cache / "clean.npy"
    if not clean.exists():
        x = np.asarray(
            raw_memmap(RAW / source)[int(seg0 * RATE) : int(seg1 * RATE)], np.float64
        )
        combs = json.loads((cache / "wavetracker.json").read_text())["interference_combs"]
        bands = [
            (h * res.fund_v[ok & (k == fish)].min(), h * res.fund_v[ok & (k == fish)].max())
            for fish in range(2)
            for h in range(1, 6)
        ]
        lines = hum_lines(x, [float(sp) for sp in combs], bands)
        print(f"  removing {len(lines)} hum lines", flush=True)
        np.save(cache / "hum_lines.npy", lines)
        np.save(clean, remove_lines(x, lines))
    data = np.load(clean, mmap_mode="r")

    tau = warp(t_mix, a, args.start, grid[0], grid[2])
    rate = a * (1.0 + np.interp(t_mix, grid[0], np.gradient(grid[2], grid[0])))
    out = np.full((2, len(t_mix)), np.nan)
    for fish in range(2):
        m = ok & (k == fish)
        tf, ff = t_det[m], res.fund_v[m]
        order = np.argsort(tf)
        tf, ff = tf[order], ff[order]
        f = np.interp(tau, tf, ff)
        # gaps longer than 5 s (source time) are unknown
        nxt = np.clip(np.searchsorted(tf, tau), 1, len(tf) - 1)
        gap = tf[nxt] - tf[nxt - 1]
        f[(gap > 5.0) | (tau < tf[0]) | (tau > tf[-1])] = np.nan
        out[fish] = f * rate
    return t_mix, out, data, seg0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out")
    ap.add_argument("--n-sources", type=int, default=8)
    ap.add_argument("--start", type=float, default=7200.0)
    ap.add_argument("--duration", type=float, default=1200.0)
    ap.add_argument("--drift", type=float, default=0.005, help="std of eps")
    ap.add_argument("--drift-timescale", type=float, default=300.0, help="[s]")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default="output/dev/tube_sources")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    dirs = sorted(p.name for p in RAW.iterdir() if (p / "traces-grid1.raw").exists())
    sources = list(rng.choice(dirs, args.n_sources, replace=False))
    scales = list(rng.permutation(SCALES)[: args.n_sources])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(RAW / sources[0] / "fishgrid.cfg", out / "fishgrid.cfg")
    grid = common_drift(args.duration, args.drift, args.drift_timescale, args.seed + 1)

    truth, segments = [], []
    for s, a in zip(sources, scales, strict=True):
        print(f"source {s}: scale {a}", flush=True)
        t_mix, f, data, seg0 = source_segment(args, s, a, grid)
        truth.append(f)
        segments.append((data, seg0))
    np.savez(
        out / "truth.npz",
        times=t_mix,
        freqs=np.concatenate(truth),  # (2 * n_sources, T)
        source=np.repeat(np.arange(len(sources)), 2),
    )
    (out / "mixture.json").write_text(
        json.dumps(
            {
                "sources": sources,
                "scales": scales,
                "start": args.start,
                "duration": args.duration,
                "drift": args.drift,
                "drift_timescale": args.drift_timescale,
                "seed": args.seed,
            },
            indent=2,
        )
    )
    print("mixing", flush=True)
    mix(args, segments, scales, grid)
    print("median fish frequencies:", np.round(np.nanmedian(np.concatenate(truth), 1)))


if __name__ == "__main__":
    main()
