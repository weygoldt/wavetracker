"""Dense mixtures of real two-fish recordings, with per-fish power augmentation.

Each source segment of the library (see ``library.py``) is split into its two
fish and a residual in a short-time Fourier transform: the bins around every
harmonic of a fish's ground-truth trace belong to that fish. Under the fish,
the residual is filled with noise of the local noise spectrum, so the parts
add up to the original exactly, and a fish attenuated to nothing leaves the
noise of the recording, not a hole. Stationary hum lines are replaced by the
same noise.

The parts of every source are then

* **scaled**: each fish gets its own gain over time (static offset, ramps,
  slow wander, fades of a fish that swims away) and a slowly changing gain per
  electrode (its pattern changes as it moves); with ``per: dyad`` both fish of
  a source share the gains;
* **time-warped**: ``tau_s(t) = start + a_s (t + D(t))`` scales all frequencies
  of source s by ``a_s (1 + eps(t))``; ``a_s`` places the source's fish in the
  band of the mixture and ``eps`` is a slow drift common to all sources;
* **summed** electrode by electrode onto ``MIX_CHANNELS`` electrodes (each
  source's best electrodes, in random order).

The residual of source s is scaled by ``sqrt(a_s / n_sources)``: warping
scales a noise spectrum by ``1 / a_s`` and the residuals of all sources add up,
so this keeps the noise floor of the mixture at that of one recording, for any
number of sources and any frequency scale. ``noise_db`` raises it.

The truth (``truth.npz``) holds per fish and frame of a 10 Hz grid the
frequency, the gain and its SNR in the mixture (power at the fundamental,
summed over electrodes, over the noise floor at that frequency, both with the
analysis resolution of wavetracker's default ``nfft``).

Usage::

    python benchmarks/dense/mixture.py                       # whole suite
    python benchmarks/dense/mixture.py --only dev/density-10-s0
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.ndimage import median_filter

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA, MIX_CHANNELS, RATE, Trial, Warp, common_drift

SUITE = Path(__file__).with_name("suite.yaml")
MIXTURES = DATA / "mixtures"
TRUTH_RATE = 10.0  # Hz
GAIN_RATE = 1.0  # Hz, grid of the gain processes
SNR_NFFT = 2**15  # wavetracker's default at 20 kHz (0.61 Hz)
MAX_GAP = 1.0  # s of source time: longer gaps in the source truth are unknown
CLIP = 6.0  # residual near a fish keeps at most 6x the mean noise power (7.8 dB)
BETA = 4.0  # over-subtraction in the fish masks: noise bins rarely go to a fish


# --- splitting a source into its fish and a residual -------------------------


def _device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def fish_labels(
    freqs: np.ndarray,
    frame_t: np.ndarray,
    traces: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    t_win: float,
    dev: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Label of every STFT bin (bins, frames): 0 residual, k + 1 fish k, and
    the fish whose neighbourhood a bin is in (0: none).

    `traces` holds per fish (times, frequency without gaps, in-gap flag). A
    bin belongs to a fish if it is within ``w_h = 2 df + h (0.75 Hz +
    |f'| t_win / 2)`` of its h-th harmonic; in gaps of the truth (a missed rise?)
    the band extends 15 Hz * h upwards. Bins claimed by two fish go to the
    closer one (in units of the fundamental). The neighbourhood extends
    ``min(40 + 10 h, 0.45 f)`` Hz around every harmonic."""
    df = freqs[1] - freqs[0]
    fb = torch.as_tensor(freqs, dtype=torch.float32, device=dev)[:, None]
    best = torch.full((len(freqs), len(frame_t)), np.inf, device=dev)
    best_near = torch.full((len(freqs), len(frame_t)), np.inf, device=dev)
    label = torch.zeros((len(freqs), len(frame_t)), dtype=torch.int8, device=dev)
    near = torch.zeros((len(freqs), len(frame_t)), dtype=torch.int8, device=dev)
    for k, (tt, ff, gap) in enumerate(traces):
        f = np.interp(frame_t, tt, ff)
        slope = np.abs(np.gradient(f, frame_t))
        g = np.interp(frame_t, tt, gap.astype(float)) > 0
        f_t = torch.as_tensor(f, dtype=torch.float32, device=dev)[None]
        h = torch.clamp(torch.round(fb / f_t), min=1)
        off = fb - h * f_t
        w = (
            2 * df
            + h
            * torch.as_tensor(
                0.75 + slope * t_win / 2, dtype=torch.float32, device=dev
            )[None]
        )
        up = h * torch.as_tensor(15.0 * g, dtype=torch.float32, device=dev)[None]
        inside = (off >= -w) & (off <= w + up)
        dist = torch.abs(off) / h
        take = inside & (dist < best)
        label[take] = k + 1
        best = torch.where(take, dist, best)
        nb = torch.minimum(40.0 + 10.0 * h, 0.45 * f_t)
        take = (torch.abs(off) <= nb) & (dist < best_near)
        near[take] = k + 1
        best_near = torch.where(take, dist, best_near)
        del h, off, w, up, inside, dist, take, nb
    return label, near


def split_source(
    x: np.ndarray,
    traces: list,
    hum: np.ndarray,
    a: float,
    rng: np.random.Generator,
    nperseg: int = 2**14,
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split `x` (samples, channels) into fish 0, fish 1 and residual (each
    float32, same shape). Traces are in seconds from the first sample.

    For a > 1 everything above 0.9 / a of the Nyquist frequency is removed,
    which would alias after warping."""
    dev = _device(device)
    hop = nperseg // 4
    n = x.shape[0]
    win = torch.hann_window(nperseg, periodic=True, device=dev)
    freqs = np.fft.rfftfreq(nperseg, 1 / RATE)
    n_frames = n // hop + 1
    frame_t = np.arange(n_frames) * hop / RATE
    label, near = fish_labels(freqs, frame_t, traces, nperseg / RATE, dev)
    df = freqs[1]
    hum_bins = np.zeros(len(freqs), bool)
    for fl in hum:
        hum_bins |= np.abs(freqs - fl) <= 1.5 * df
    hum_t = torch.as_tensor(hum_bins, device=dev)[:, None]
    resid = label == 0
    replace = resid & hum_t  # residual bins filled with noise
    cut = 0.9 * RATE / 2 / a if a > 1.0 else np.inf
    alias = torch.as_tensor(freqs > cut, device=dev)[:, None]
    gen = torch.Generator(device=dev).manual_seed(int(rng.integers(2**31)))

    outs = [np.empty_like(x, dtype=np.float32) for _ in range(3)]
    n_lines, moved = [], []
    for c in range(x.shape[1]):
        xc = torch.as_tensor(
            np.ascontiguousarray(x[:, c]), dtype=torch.float32, device=dev
        )
        z = torch.stft(xc, nperseg, hop, window=win, center=True, return_complex=True)
        p = z.real**2 + z.imag**2
        # noise spectrum and lines from residual bins away from the fish
        # (near a fish, its skirts would count as lines; strong residual there
        # goes to the fish below)
        p_res = torch.where(resid & ~hum_t & (near == 0), p, torch.nan)
        med = torch.nanmedian(p_res, dim=1).values.cpu().numpy()
        known = torch.isfinite(p_res).float().mean(1).cpu().numpy() >= 0.1
        known &= np.isfinite(med) & (med > 0)
        # stationary lines the comb search missed (weak hum, crosstalk of
        # other setups): residual bins whose median is 4 dB above +-10 Hz
        logm = np.interp(freqs, freqs[known], np.log(med[known]))
        ref = median_filter(logm, size=2 * int(10 / df) + 1, mode="nearest")
        line = known & (logm - ref > 0.4 * np.log(10))
        line = np.convolve(line, np.ones(3), "same") > 0  # +-1 bin
        known &= ~line
        mu = np.exp(np.interp(freqs, freqs[known], np.log(med[known]))) / np.log(2)
        rep = replace | torch.as_tensor(line, device=dev)[:, None]
        n_lines.append(int(line.sum()))
        # the noise level changes over time (transients, fish moving): scale
        # the fill per frame by the residual away from the fish, in blocks of
        # ~200 Hz (1 for stationary noise: median of exponential / ln 2)
        mu_t = torch.as_tensor(mu, dtype=torch.float32, device=dev)[:, None]
        away = (near == 0) & ~rep
        q = torch.where(away, p / mu_t, torch.nan)
        nb = max(1, int(200 / df))
        n_blk = -(-len(freqs) // nb)
        q = torch.nn.functional.pad(q, (0, 0, 0, n_blk * nb - len(freqs)), value=np.nan)
        act = torch.nanmedian(q.reshape(n_blk, nb, -1), dim=1).values / np.log(2)
        act = torch.nan_to_num(act, nan=1.0).clamp(0.25, 100.0)
        level = mu_t * act.repeat_interleave(nb, dim=0)[: len(freqs)]  # (bins, frames)
        del q, away, act
        # the fill is the transform of white noise, shaped: independent random
        # coefficients would lose up to 1 - hop / nperseg of their power in
        # the overlap-add of the inverse transform (real noise is consistent
        # across overlapping frames)
        white = torch.randn(n, generator=gen, device=dev)
        fill = torch.stft(
            white, nperseg, hop, window=win, center=True, return_complex=True
        )
        fill = fill * torch.sqrt(level / (win**2).sum())
        del white
        zero = torch.zeros((), dtype=z.dtype, device=dev)
        # in its masks a fish takes what stands out of the noise (spectral
        # subtraction, gain w); what is left, and the noise under the fish,
        # stay in the residual, which is scaled like all noise. Exact:
        # fish + residual = w z - sqrt(w) fill + (1 - w) z + sqrt(w) fill = z
        in_mask = (label > 0) & ~alias
        w = torch.clamp(1 - BETA * level / torch.clamp(p, min=1e-30), 0, 1)
        w = torch.where(in_mask, w, 0)
        sw = torch.sqrt(w)
        fish_z = w * z - sw * fill
        res_z = torch.where(rep, fill, z)
        res_z = torch.where(in_mask, (1 - w) * z + sw * fill, res_z)
        del w, sw
        # near a fish, residual power far above the noise (chirp smears, a
        # fish line off its trace) belongs to that fish: keep at most
        # CLIP x the mean noise power in the residual, move the excess
        thr = torch.sqrt(CLIP * level)
        mag = torch.abs(res_z)
        excess = res_z * torch.clamp(1 - thr / torch.clamp(mag, min=1e-30), min=0)
        excess = torch.where((near > 0) & (label == 0), excess, zero)
        res_z = res_z - excess
        parts = []
        for k in (1, 2):
            fk = torch.where(label == k, fish_z, zero)
            fk = fk + torch.where((near == k) & ~alias, excess, zero)
            parts.append(fk)
        del fish_z
        parts.append(torch.where(alias, zero, res_z))
        moved.append(float((excess.abs() ** 2).sum() / (z.abs() ** 2).sum()))
        del mag, excess, res_z, level, thr
        for j, zz in enumerate(parts):
            y = torch.istft(zz, nperseg, hop, window=win, center=True, length=n)
            outs[j][:, c] = y.cpu().numpy()
        del z, p, p_res, fill, parts, rep
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    # QC: line bins replaced and fraction of the power moved to fish, per channel
    split_source.last_qc = {"line_bins": n_lines, "moved_power": moved}
    return outs[0], outs[1], outs[2]


# --- gains -------------------------------------------------------------------


def _ou(n: int, sd: float, tau: float, dt: float, rng) -> np.ndarray:
    if sd == 0 or n == 0:
        return np.zeros(n)
    phi = np.exp(-dt / tau)
    x = np.empty(n)
    x[0] = rng.standard_normal() * sd
    e = rng.standard_normal(n) * sd * np.sqrt(1 - phi**2)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def _uniform(v, rng, size=None):
    if v is None:
        return np.zeros(size) if size else 0.0
    if np.isscalar(v):
        return np.full(size, float(v)) if size else float(v)
    return rng.uniform(v[0], v[1], size)


def gains(spec: dict | None, n_fish: int, duration: float, rng) -> dict:
    """Gain processes in dB on a ``GAIN_RATE`` grid.

    Returns times, common gain per fish (n_fish, T), electrode gain per fish
    (n_fish, MIX_CHANNELS, T) and which fish were modulated.

    spec keys (all optional): per (fish|dyad), fraction (of fish/dyads that
    are modulated), static_db (value or [lo, hi]), ramp_db (span of a linear
    ramp, random direction), wander_db / wander_tau (Ornstein-Uhlenbeck in dB),
    fades: {rate (per 10 min), depth_db [lo, hi], duration_s [lo, hi]},
    channel_db / channel_tau (independent per electrode, for every fish)."""
    spec = spec or {}
    t = np.arange(0.0, duration + 2.0, 1 / GAIN_RATE)
    T = len(t)
    per_dyad = spec.get("per", "fish") == "dyad"
    n_units = n_fish // 2 if per_dyad else n_fish
    unit = np.arange(n_fish) // 2 if per_dyad else np.arange(n_fish)
    frac = spec.get("fraction", 1.0)
    chosen = rng.permutation(n_units)[: round(frac * n_units)]
    active_u = np.zeros(n_units, bool)
    active_u[chosen] = bool(spec)  # draws stay the same with and without a spec
    common_u = np.zeros((n_units, T))
    for u in range(n_units):
        if not active_u[u]:
            continue
        g = np.full(T, _uniform(spec.get("static_db"), rng))
        if spec.get("ramp_db"):
            ramp = -abs(float(spec["ramp_db"])) * t / duration
            g += ramp if rng.random() < 0.5 else ramp[::-1]
        g += _ou(
            T,
            spec.get("wander_db", 0.0),
            spec.get("wander_tau", 60.0),
            1 / GAIN_RATE,
            rng,
        )
        fades = spec.get("fades")
        if fades:
            n_ev = rng.poisson(fades.get("rate", 1.0) * duration / 600.0)
            for _ in range(n_ev):
                depth = _uniform(fades.get("depth_db", [10, 40]), rng)
                length = _uniform(fades.get("duration_s", [20, 120]), rng)
                t0 = rng.uniform(-length / 2, duration - length / 2)
                u_ = np.clip((t - t0) / length, 0, 1)
                g -= depth * np.sin(np.pi * u_) ** 2
        common_u[u] = g
    common = common_u[unit]
    chan = np.stack(
        [
            [
                _ou(
                    T,
                    spec.get("channel_db", 0.0),
                    spec.get("channel_tau", 90.0),
                    1 / GAIN_RATE,
                    rng,
                )
                for _ in range(MIX_CHANNELS)
            ]
            for _ in range(n_fish)
        ]
    )
    return {
        "times": t,
        "common_db": common,
        "channel_db": chan,
        "modulated": active_u[unit],
    }


# --- mixture design ----------------------------------------------------------


def place_sources(design: dict, pool: list[dict], rng) -> list[dict]:
    """Choose sources and frequency scales: dyad centres stratified over the
    band (linear or log), so the density of fish is controlled."""
    n_fish = int(design["n_fish"])
    if n_fish % 2:
        raise ValueError("n_fish must be even (two fish per source)")
    n_src = n_fish // 2
    if n_src > len(pool):
        raise ValueError(f"{n_src} sources requested, {len(pool)} in the pool")
    lo, hi = design["band"]
    log = design.get("placement", "linear") == "log"
    srcs = [pool[i] for i in rng.choice(len(pool), n_src, replace=False)]
    edges = np.geomspace(lo, hi, n_src + 1) if log else np.linspace(lo, hi, n_src + 1)
    strata = rng.permutation(n_src)
    duration = design["duration"]
    spare = [p for p in pool if p not in srcs]

    def centre_range(e):
        """Dyad centres that keep all of the dyad (1-99 % range) in the band."""
        f1, f2 = e["median_hz"]
        r1 = min(e["range_hz"][0][0], e["range_hz"][1][0]) / np.sqrt(f1 * f2)
        r2 = max(e["range_hz"][0][1], e["range_hz"][1][1]) / np.sqrt(f1 * f2)
        return lo / r1, hi / r2

    out = []
    for s, e in zip(strata, srcs, strict=True):
        m_lo, m_hi = centre_range(e)
        while m_hi < m_lo and spare:  # the dyad is wider than the band
            e = spare.pop(int(rng.integers(len(spare))))
            m_lo, m_hi = centre_range(e)
        f1, f2 = e["median_hz"]
        u = rng.uniform(edges[s], edges[s + 1])
        if m_hi < m_lo:
            m = m_lo
        elif not m_lo <= u <= m_hi:
            # stratum outside what this dyad allows: anywhere it fits (clipping
            # to the edge stacked several dyads on the same frequencies)
            m = float(rng.uniform(m_lo, m_hi))
        else:
            m = u
        # the segment limits the scale: high strata get the fish a bit lower
        a = min(m / np.sqrt(f1 * f2), (e["length"] - 20.0) / (1.05 * duration))
        span = a * duration * 1.05 + 20.0
        offset = float(rng.uniform(10.0, e["length"] - span + 10.0))
        out.append(
            {
                "trial": e["name"],
                "scale": a,
                "start": e["start"] + offset,  # source time of mixture t = 0
                "channels": e["channels_used"],
                "to_mix": [int(c) for c in rng.permutation(MIX_CHANNELS)],
                "fish_hz": [f1 * a, f2 * a],
            }
        )
    return out


def source_truth(entry: dict):
    """Source truth (times, (2, frames) with NaN) of a library entry."""
    tr = np.load(DATA / "sources" / entry["name"] / "truth.npz")
    return tr["times"], tr["freqs"]


def _gap_flags(t, f, max_gap):
    """Filled trace and per-frame flag 'inside a gap longer than max_gap'."""
    ok = np.isfinite(f)
    tt = t[ok]
    filled = np.interp(t, tt, f[ok])
    nxt = np.clip(np.searchsorted(tt, t), 1, len(tt) - 1)
    gap = (tt[nxt] - tt[nxt - 1]) > max_gap
    gap &= ~ok
    gap |= (t < tt[0]) | (t > tt[-1])
    return filled, gap


def _kernel(d: torch.Tensor, half: int, cutoff: float = 0.45, beta: float = 8.0):
    """Kaiser-windowed sinc at distances `d` [samples], low-pass at `cutoff`
    of the sampling rate, rows normalised to unit DC gain."""
    h = 2 * cutoff * torch.sinc(2 * cutoff * d)
    u = torch.clamp(1 - (d / half) ** 2, min=0)
    h = (
        h
        * torch.special.i0(beta * torch.sqrt(u))
        / torch.special.i0(torch.tensor(beta))
    )
    return h / h.sum(-1, keepdim=True)


def warp_add(accs, y, src0, w: Warp, dev, chunk_s: float = 5.0, taps: int = 32):
    """acc[t] += y(tau(t)) for the accumulators `accs` (mixture time; their
    channels side by side are the columns of `y`) and `y` (source time from
    src0, (samples, channels) at RATE): band-limited interpolation with a
    32-tap Kaiser-windowed sinc (errors below -80 dB up to 0.3 of the rate).
    Content above 0.45 / a of the rate must be removed before for a > 1
    (``split_source`` does)."""
    yt = torch.as_tensor(y, dtype=torch.float32, device=dev)
    widths = np.cumsum([0] + [acc.shape[1] for acc in accs])
    half = taps // 2
    k = torch.arange(-half + 1, half + 1, device=dev)
    n_out = accs[0].shape[0]
    chunk = int(chunk_s * RATE)
    for c0 in range(0, n_out, chunk):
        c1 = min(n_out, c0 + chunk)
        pos = (w.tau(np.arange(c0, c1) / RATE) - src0) * RATE
        i = np.floor(pos)
        if i[0] - half < 0 or i[-1] + half + 1 > len(yt):
            raise ValueError("source span too short for the warp")
        i_t = torch.as_tensor(i.astype(np.int64), device=dev)
        frac = torch.as_tensor(pos - i, dtype=torch.float32, device=dev)
        h = _kernel(frac[:, None] - k[None, :].float(), half)
        out = torch.einsum("nk,nkc->nc", h, yt[i_t[:, None] + k[None, :]])
        out = out.cpu().numpy()
        for acc, a, b in zip(accs, widths[:-1], widths[1:], strict=True):
            acc[c0:c1] += out[:, a:b]
    del yt


def snr_db(mix, noise, truth_t, truth_f, dev, hop_s: float = 0.5):
    """SNR of every fish at its fundamental [dB], on frames every `hop_s`.

    Power summed over electrodes; fish power = mixture power at the bin
    closest to the truth (max over +-1 bin) minus the noise power, noise =
    median of the noise-only spectrogram over +-5 bins."""
    hop = int(hop_s * RATE)
    win = torch.hann_window(SNR_NFFT, periodic=True, device=dev)
    scale = 1.0 / win.sum() ** 2

    def power(x):
        p = 0
        for c in range(x.shape[1]):
            xc = torch.as_tensor(
                np.ascontiguousarray(x[:, c]), dtype=torch.float32, device=dev
            )
            z = torch.stft(
                xc, SNR_NFFT, hop, window=win, center=False, return_complex=True
            )
            p = p + (z.real**2 + z.imag**2) * scale
        return p  # (bins, frames)

    pm = power(mix)
    pn = power(noise)
    pn = torch.nn.functional.avg_pool1d(pn.T[:, None], 11, 1, 5)[:, 0].T  # ~median
    freqs = np.fft.rfftfreq(SNR_NFFT, 1 / RATE)
    times = (np.arange(pm.shape[1]) * hop + SNR_NFFT / 2) / RATE
    out = np.full((len(truth_f), len(times)), np.nan)
    fish_p = np.full((len(truth_f), len(times)), np.nan, np.float32)
    noise_p = np.full((len(truth_f), len(times)), np.nan, np.float32)
    pm_np, pn_np = pm.cpu().numpy(), pn.cpu().numpy()
    cols = np.arange(len(times))
    for k, f in enumerate(truth_f):
        fk = np.interp(times, truth_t, f)
        ok = np.isfinite(fk)
        b = np.clip(np.round(fk[ok] / freqs[1]).astype(int), 1, len(freqs) - 2)
        c = cols[ok]
        pf = np.max([pm_np[b + d, c] for d in (-1, 0, 1)], axis=0)
        nb = pn_np[b, c]
        fp = np.maximum(pf - nb, 1e-3 * nb)
        out[k, ok] = 10 * np.log10(fp / nb)
        fish_p[k, ok], noise_p[k, ok] = fp, nb
    # fish and noise power (summed over electrodes) let other noise floors
    # be put under the same fish later (field-noise variants)
    return times, out, (pm_np, freqs, times), (fish_p, noise_p)


def generate(name: str, design: dict, library: list[dict], device: str = "auto"):
    out = MIXTURES / name
    if (out / "band.yaml").exists():  # written last
        print(f"{name}: exists", flush=True)
        return
    t_start = time.perf_counter()
    split = name.split("/")[0]
    seed = int(design["seed"])
    rng = np.random.default_rng([seed, *map(ord, name)])
    pool = [e for e in library if e["ok"] and e["split"] == split]
    sources = place_sources(design, pool, rng)
    duration = float(design["duration"])
    grid = common_drift(
        duration,
        design.get("drift", 0.005),
        design.get("drift_timescale", 300.0),
        seed + 1,
    )
    n_fish = 2 * len(sources)
    gp = gains(design.get("gain"), n_fish, duration, rng)
    noise_gain = 10 ** (design.get("noise_db", 0.0) / 20)
    n_out = int(duration * RATE)
    mix = np.zeros((n_out, MIX_CHANNELS), np.float32)
    noise = np.zeros((n_out, MIX_CHANNELS), np.float32)
    t_truth = np.arange(0.0, duration, 1 / TRUTH_RATE)
    truth = np.full((n_fish, len(t_truth)), np.nan)
    filled = np.full((n_fish, len(t_truth)), np.nan)
    by_name = {e["name"]: e for e in library}
    dev = _device(device)

    for s, src in enumerate(sources):
        e = by_name[src["trial"]]
        trial = Trial(**{k: e[k] for k in Trial.__dataclass_fields__})
        w = Warp(src["scale"], src["start"], grid[0], grid[1], grid[2])
        tau = w.tau(np.array([0.0, duration]))
        src0, src1 = tau[0] - 5.0, tau[1] + 5.0
        i0 = int(np.floor(src0 * RATE))
        x = np.asarray(
            trial.raw()[i0 : int(np.ceil(src1 * RATE))][:, src["channels"]], np.float32
        )
        src0 = i0 / RATE
        st, sf = source_truth(e)
        traces = []
        for k in range(2):
            ff, gap = _gap_flags(st, sf[k], MAX_GAP)
            traces.append((st - src0, ff, gap))
            # truth in mixture time: frequency times the warp rate
            tq = w.tau(t_truth)
            fk = np.interp(tq, st, ff) * w.rate(t_truth)
            gq = np.interp(tq, st, gap.astype(float)) > 0
            filled[2 * s + k] = fk
            truth[2 * s + k] = np.where(gq, np.nan, fk)
        hum = np.load(DATA / "sources" / e["name"] / "hum_lines.npy")
        f0, f1, r = split_source(x, traces, hum, src["scale"], rng, device=device)
        n_src = len(x)
        del x
        # gains in source time: per sample and electrode, interpolated
        # linearly from the gain grid (slow processes). Composed electrode by
        # electrode into one array, in mixture order: [fish + noise | noise]
        t_of = w.t_of_tau(np.arange(n_src) / RATE + src0, duration)
        gpos = np.clip(t_of * GAIN_RATE, 0, len(gp["times"]) - 1.001)
        del t_of
        gi = gpos.astype(np.int64)
        gf = (gpos - gi).astype(np.float32)
        del gpos
        c_res = np.float32(np.sqrt(src["scale"] / len(sources)) * noise_gain)
        y = np.empty((n_src, 2 * MIX_CHANNELS), np.float32)
        for c, m in enumerate(src["to_mix"]):  # source column c -> electrode m
            acc = r[:, c] * c_res
            y[:, MIX_CHANNELS + m] = acc
            for k, part in enumerate((f0, f1)):
                j = 2 * s + k
                tab = (gp["common_db"][j] + gp["channel_db"][j, m]).astype(np.float32)
                db = tab[gi] * (1 - gf) + tab[gi + 1] * gf
                acc += np.power(np.float32(10.0), db / 20) * part[:, c]
            y[:, m] = acc
        del f0, f1, r, gi, gf, acc, db
        warp_add([mix, noise], y, src0, w, dev)
        del y
        print(
            f"  {name}: source {s + 1}/{len(sources)} {src['trial']} x{src['scale']:.2f} "
            f"-> {src['fish_hz'][0]:.0f}/{src['fish_hz'][1]:.0f} Hz",
            flush=True,
        )

    snr_t, snr, (pm, freqs, ptimes), (fish_p, noise_p) = snr_db(
        mix, noise, t_truth, filled, dev
    )
    out.mkdir(parents=True, exist_ok=True)
    mix.tofile(out / "traces-grid1.raw")
    shutil.copy(Path(trial_cfg_source(library)) / "fishgrid.cfg", out / "fishgrid.cfg")
    gain_t = gp["times"]
    common = np.stack([np.interp(t_truth, gain_t, g) for g in gp["common_db"]])
    chan = np.stack(
        [[np.interp(t_truth, gain_t, g) for g in ch] for ch in gp["channel_db"]]
    )
    np.savez(
        out / "truth.npz",
        times=t_truth,
        freqs=truth,  # (fish, T) NaN where unknown
        freqs_filled=filled,  # through the gaps, for ignoring detections there
        source=np.repeat(np.arange(len(sources)), 2),
        gain_db=common,
        channel_gain_db=chan.astype(np.float32),
        modulated=gp["modulated"],
        snr_times=snr_t,
        snr_db=snr,
        fish_power=fish_p,  # at the fundamental, summed over electrodes
        noise_power=noise_p,
    )
    # overview spectrogram for figures (sum over electrodes, <= 2.6 kHz)
    fmax = min(RATE / 2, 1.3 * np.nanmax(filled) * 2)
    nb = int(np.searchsorted(freqs, fmax))
    pool_f = 4
    nb -= nb % pool_f
    ov = pm[:nb].reshape(nb // pool_f, pool_f, -1).mean(1)
    np.savez_compressed(
        out / "overview.npz",
        power=ov.astype(np.float32),
        freqs=freqs[:nb].reshape(-1, pool_f).mean(1),
        times=ptimes,
    )
    lo = float(np.nanpercentile(filled, 0.5))
    hi = float(np.nanpercentile(filled, 99.5))
    meta = {
        "name": name,
        "design": design,
        "sources": sources,
        "n_fish": n_fish,
        "duration": duration,
        "noise_scaling": "residual * sqrt(scale / n_sources) * 10**(noise_db / 20)",
        "fish_band_hz": [lo, hi],
        "seconds": time.perf_counter() - t_start,
    }
    (out / "mixture.json").write_text(json.dumps(meta, indent=1))
    cfg = {
        "harmonic_groups": {
            "min_freq": float(np.floor(0.85 * lo)),
            "max_freq": float(np.ceil(1.15 * hi)),
            "mains_freq": 0.0,  # hum was removed from the sources
            "min_good_peak_power": None,
        }
    }
    (out / "band.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"{name}: done in {meta['seconds']:.0f} s", flush=True)


def trial_cfg_source(library: list[dict]) -> str:
    """A fishgrid.cfg of a 3x3 (9 electrode) grid."""
    for e in library:
        if e["channels"] == MIX_CHANNELS:
            return e["path"]
    raise RuntimeError(f"no {MIX_CHANNELS}-electrode trial in the library")


def expand_suite(path: Path = SUITE) -> dict[str, dict]:
    """{name: design}; names are <split>/<scenario>-s<seed>."""
    suite = yaml.safe_load(path.read_text())
    defaults = suite.get("defaults", {})
    out = {}
    for sc in suite["scenarios"]:
        for split in sc.get("splits", ["dev", "test"]):
            for seed in sc.get("seeds", [0]):
                d = defaults | {
                    k: v for k, v in sc.items() if k not in ("seeds", "splits")
                }
                d["seed"] = seed
                out[f"{split}/{sc['name']}-s{seed}"] = d
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--only", nargs="*", help="mixture names (split/scenario-sSEED)")
    p.add_argument("--suite", type=Path, default=SUITE)
    p.add_argument("--split", help="dev or test (default both)")
    p.add_argument("--device", default="auto")
    p.add_argument("--worker", default="0/1", help="i/n: every n-th mixture from i")
    args = p.parse_args(argv)
    library = json.loads((DATA / "library.json").read_text())
    designs = expand_suite(args.suite)
    names = [
        n
        for n in designs
        if (not args.only or n in args.only)
        and (not args.split or n.startswith(args.split + "/"))
    ]
    # biggest first, so parallel workers finish together
    names.sort(key=lambda n: -designs[n]["n_fish"])
    i, n_workers = map(int, args.worker.split("/"))
    for name in names[i::n_workers]:
        for attempt in range(60):  # other sessions share the GPU
            try:
                generate(name, designs[name], library, args.device)
                break
            except torch.cuda.OutOfMemoryError:
                if attempt == 59:
                    raise
                torch.cuda.empty_cache()
                print(f"{name}: GPU out of memory, retrying in 60 s", flush=True)
                time.sleep(60)


if __name__ == "__main__":
    main()
