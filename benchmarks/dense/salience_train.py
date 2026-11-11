"""Train the learned (salience) detector on the `ltrain` mixtures.

Every training example is a fresh crop of a training mixture (lab or field
noise), turned into a spectrogram on the GPU exactly as wavetracker does it
(so beats and leakage between close fish are as in a recording), summed over
a random subset of electrodes. Targets and loss weights come from the
mixture's truth:

* positives: a Gaussian bump (sigma 0.6 Hz) at every truth fundamental,
  merged by maximum;
* ignored (weight 0): fish frames below ``--snr-ignore`` dB (the fish was
  attenuated below the noise: neither fish nor no-fish), stretches where the
  truth does not know the fish, and persistent lines the truth does not know
  (found once per mixture on its lab variant, away from every truth
  fundamental and harmonic, see ``unknown_lines``);
* half weight: frames the truth completed by ridge following;
* everything else, harmonics of truth fish included, is a negative.

The loss is binary cross-entropy on the salience plus an L1 loss on the
sub-bin offset at the truth bins. Each example uses a random window of
candidate frequencies (the network is the same at every frequency), with the
harmonic stack gathered from the whole spectrum.

Usage::

    python benchmarks/dense/salience_train.py --tag v1 --steps 6000
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA, MIX_CHANNELS, RATE

from wavetracker.salience import (
    SalienceNet,
    SalienceSpec,
    decode,
    electrode_features,
    harmonic_stack,
    load_model,
    normalise,
    save_model,
)
from wavetracker.spectrogram import PowerSpectrogram, decibel

NFFT = 2**15
STEP = int(NFFT * 0.1)  # overlap_frac 0.9, as wavetracker's default
DF = RATE / NFFT
FT = STEP / RATE  # frame step [s]
SIGMA_HZ = 0.6
F_MIN, F_MAX = 40.0, 1500.0  # candidate fundamentals seen in training [Hz]
LINE_SMOOTH_S = 10.0  # unknown lines: mean feature over this many seconds
LINE_Z = 2.0  # ... above this
LINE_CLEAR_HZ = 2.5  # unknown lines farther than this from truth (harmonics)
LINE_HARMONICS = 12
TILT_MAX_DB = 15.0  # spectral tilt: at most this gain or attenuation
HARD_HARMONICS = (1 / 3, 1 / 2, 2, 3, 4, 5)  # hard negatives at h * f of strong fish
HARD_TOL_HZ = 1.2  # ... within this many Hz times max(1, h / 2)
MIXTURES = DATA / "mixtures"
CACHE = DATA / "salience" / "cache"
OUT = DATA / "salience"


def mixture_dirs(split: str) -> list[Path]:
    return sorted(p for p in (MIXTURES / split).glob("*") if (p / "band.yaml").exists())


class Mixture:
    """Raw traces and truth of one lab mixture and its field-noise variant
    (lab traces plus scaled field noise), resampled to spectrogram frames.
    ``traces(j0, n, g)`` mixes in the field noise with any gain g (0: lab,
    1: the field variant) and ``snr_at(g)`` gives the matching SNR."""

    def __init__(self, path: Path):
        self.path = path
        n = (path / "traces-grid1.raw").stat().st_size // 4 // MIX_CHANNELS
        self.raw = np.memmap(
            path / "traces-grid1.raw",
            dtype=np.float32,
            mode="r",
            shape=(n, MIX_CHANNELS),
        )
        fpath = path.parent.parent / (path.parent.name + "-field") / path.name
        self.field = None
        if (fpath / "band.yaml").exists():
            self.field = np.memmap(
                fpath / "traces-grid1.raw",
                dtype=np.float32,
                mode="r",
                shape=(n, MIX_CHANNELS),
            )
            zf = np.load(fpath / "truth.npz")
        self.n_frames = (n - NFFT) // STEP + 1
        t = (np.arange(self.n_frames) * STEP + NFFT / 2) / RATE
        z = np.load(path / "truth.npz")
        self.freqs = np.stack(
            [np.interp(t, z["times"], f, left=np.nan, right=np.nan) for f in z["freqs"]]
        )
        # interpolation across NaN gaps gives NaN; keep where the truth knows the fish
        known = np.stack(
            [
                np.interp(t, z["times"], np.isfinite(f).astype(float)) > 0.999
                for f in z["freqs"]
            ]
        )
        self.freqs[~known] = np.nan
        # where the fish is unknown, still keep a guess of its frequency to ignore it
        self.guess = np.stack([_fill(t, z["times"], f) for f in z["freqs"]])
        det = z["freqs_detected"] if "freqs_detected" in z.files else z["freqs"]
        detk = np.stack(
            [
                np.interp(t, z["times"], np.isfinite(f).astype(float)) > 0.999
                for f in det
            ]
        )
        self.completed = known & ~detk
        st = z["snr_times"]

        def frames(a):
            return np.stack([np.interp(t, st, np.nan_to_num(r, nan=0.0)) for r in a])

        self.fish_power = frames(z["fish_power"])
        self.noise_lab = frames(z["noise_power"])
        self.noise_field = None if self.field is None else frames(zf["noise_power"])
        self.snr = self.snr_at(0.0)

    def snr_at(self, g: float) -> np.ndarray:
        """SNR [dB] per fish and frame with the field noise at gain g."""
        noise = self.noise_lab
        if g and self.noise_field is not None:
            noise = noise + g**2 * (self.noise_field - self.noise_lab)
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = 10 * np.log10(self.fish_power / noise)
        return np.nan_to_num(snr, nan=-99.0, neginf=-99.0)

    def traces(self, j0: int, n_frames: int, g: float) -> np.ndarray:
        a, b = j0 * STEP, (j0 + n_frames - 1) * STEP + NFFT
        x = np.asarray(self.raw[a:b])
        if g and self.field is not None:
            x = x + np.float32(g) * (np.asarray(self.field[a:b]) - x)
        return x

    def lab_name(self) -> str:
        split, name = self.path.parent.name, self.path.name
        return f"{split.removesuffix('-field')}/{name}"


def _smooth(n: int) -> int:
    """The smallest number >= n whose prime factors are 2, 3 and 5."""
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


def _fill(t, tt, f):
    ok = np.isfinite(f)
    if ok.sum() < 2:
        return np.full(len(t), np.nan)
    return np.interp(t, tt[ok], f[ok])


def spectrogram(spec: PowerSpectrogram, x: np.ndarray, electrodes, dev) -> torch.Tensor:
    """Raw (samples, channels) -> summed log power (bins, frames) [dB]."""
    tr = torch.from_numpy(np.ascontiguousarray(x[:, electrodes].T)).to(dev)
    return decibel(spec(tr).sum(0))


def summed_power(
    spec: PowerSpectrogram,
    x: torch.Tensor,
    windows: list[int] | None = None,
    bins: int = 0,
    frames: slice = slice(None),
):
    """Power (bins, frames) of traces (channels, samples) summed over
    channels, one channel at a time (a ninth of the memory). With `windows`
    (first bins), also the complex spectra of every channel in each window
    of `bins` bins and the given frames: (windows, channels, bins, frames)."""
    p, per = None, []
    for c in range(len(x)):
        X = spec.stft(x[c : c + 1])[0]
        pc = spec.power(X)
        p = pc if p is None else p + pc
        if windows is not None:
            per.append(torch.stack([X[k : k + bins, frames] for k in windows]))
    if windows is None:
        return p
    return p, torch.stack(per, 1)


def unknown_lines(mix: Mixture, spec: PowerSpectrogram, dev) -> np.ndarray:
    """(frames, bins up to F_MAX) bool: persistent lines in the lab variant of
    the mixture that are no truth fundamental or harmonic. Cached per mixture."""
    lab = mix.lab_name()
    cache = CACHE / (lab.replace("/", "__") + ".npz")
    kmax = int(F_MAX / DF) + 1
    if cache.exists():
        z = np.load(cache)
        return np.unpackbits(z["bits"], axis=1, count=kmax).astype(bool)
    lab_mix = (
        mix if mix.path.parent.name == lab.split("/")[0] else Mixture(MIXTURES / lab)
    )
    rows = []
    block = 400
    for f0 in range(0, lab_mix.n_frames, block):
        f1 = min(lab_mix.n_frames, f0 + block)
        x = lab_mix.raw[f0 * STEP : (f1 - 1) * STEP + NFFT]
        log = spectrogram(spec, x, list(range(MIX_CHANNELS)), dev)
        rows.append(normalise(log[:kmax].T, DF).cpu())
    zf = torch.cat(rows).numpy()  # (frames, kmax)
    n = max(1, round(LINE_SMOOTH_S / FT))
    cs = np.cumsum(np.vstack([np.zeros((1, kmax)), zf]), axis=0)
    i = np.arange(len(zf))
    a, b = np.clip(i - n // 2, 0, len(zf)), np.clip(i - n // 2 + n, 0, len(zf))
    smooth = (cs[b] - cs[a]) / (b - a)[:, None]
    line = smooth > LINE_Z
    fk = np.arange(kmax) * DF
    clear = int(np.ceil(LINE_CLEAR_HZ / DF))
    for g in lab_mix.guess:
        for h in range(1, LINE_HARMONICS + 1):
            k = np.round(h * g / DF)
            ok = np.isfinite(k) & (k < kmax + clear)
            for d in range(-clear, clear + 1):
                kk = (k[ok] + d).astype(int)
                m = (kk >= 0) & (kk < kmax)
                line[np.flatnonzero(ok)[m], kk[m]] = False
    line[:, fk < F_MIN / 2] = False
    line = line | np.roll(line, 1, 1) | np.roll(line, -1, 1)  # +-1 bin
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, bits=np.packbits(line, axis=1))
    return line


@dataclass
class Crop:
    """What one training crop is made of (drawn in the main thread, read in
    a worker thread, built on the GPU)."""

    mix: Mixture
    j0: int  # first output frame, in frames of the mixture
    electrodes: list[int]
    gain: float  # field noise gain
    k_los: list[int]  # first candidate bin of each frequency window
    r: float = 1.0  # speed: r > 1 plays the source faster (frequencies x r)
    tilt: float = 0.0  # dB per octave on the lab traces (fish), around f_ref
    f_ref: float = 500.0
    shelf: tuple[float, float, float] | None = None
    """Noise shelf: (edge [Hz], raise [dB], width [Hz]), see Batches."""


class Batches:
    """Training examples drawn from the mixtures of a split, each with field
    noise at a random gain (0: lab, 1: the field variant, up to 3).

    A crop's spectrogram serves ``windows`` random windows of candidate
    frequencies. Augmentations (all off by default): ``speed`` resamples a
    crop by a random r (frequencies and modulation speeds times r, the truth
    with them; the field noise stays at its own rate); ``tilt_db`` tilts the
    spectrum of the lab traces (fish and lab noise) by a random slope in dB
    per octave, which changes the power ratios of the harmonics against the
    field noise; ``harm_weight`` weights the bins at harmonics and
    subharmonics of fish at least ``harm_snr`` dB above the noise as hard
    negatives. ``shelf_p`` adds, to that share of the crops, white noise
    above a random edge frequency (a smooth step ``shelf_width`` Hz wide)
    that raises the crop's noise floor there by a random ``shelf_db`` dB: a
    fish below the edge keeps its fundamental but loses its harmonics under
    the raised floor, as fish below a dense chorus do in the field; the SNR
    of every fish (and so which frames are ignored) follows the added noise
    at its fundamental."""

    def __init__(
        self,
        split: str,
        frames: int,
        ctx: int,
        bins: int,
        snr_ignore: float,
        dev,
        seed=0,
        limit: int | None = None,
        fraction: float = 1.0,
        harmonics: tuple[float, ...] = SalienceSpec(DF, FT).harmonics,
        windows: int = 1,
        speed: tuple[float, float] | None = None,
        speed_p: float = 0.0,
        tilt_db: float = 0.0,
        tilt_p: float = 0.0,
        harm_weight: float = 1.0,
        harm_snr: float = 10.0,
        hard_harmonics: tuple[float, ...] = HARD_HARMONICS,
        sigma_hz: float = SIGMA_HZ,
        workers: int = 4,
        pair_hz: float = 0.0,
        pair_focus: float = 0.0,
        el_frames: slice | None = None,
        shelf_p: float = 0.0,
        shelf_db: tuple[float, float] = (10.0, 30.0),
        shelf_edge: tuple[float, float] = (150.0, 1200.0),
        shelf_width: tuple[float, float] = (5.0, 40.0),
    ):
        splits = split.split(",")
        paths = [p for s in splits for p in mixture_dirs(s)[:limit]]
        if fraction < 1.0:  # evenly spaced, so every scenario family keeps examples
            n = max(1, round(fraction * len(paths)))
            paths = [
                paths[i]
                for i in np.unique(
                    np.round(np.linspace(0, len(paths) - 1, n)).astype(int)
                )
            ]
        self.mixes = [Mixture(p) for p in paths]
        if not self.mixes:
            raise SystemExit(f"no mixtures in {MIXTURES / split}")
        self.frames, self.ctx, self.bins = frames, ctx, bins
        self.snr_ignore = snr_ignore
        self.dev = dev
        self.harmonics = tuple(harmonics)
        self.windows = windows
        self.speed, self.speed_p = speed, speed_p
        self.tilt_db, self.tilt_p = tilt_db, tilt_p
        self.harm_weight, self.harm_snr = harm_weight, harm_snr
        self.hard_harmonics = tuple(hard_harmonics)
        self.sigma_hz = sigma_hz
        self.pair_hz = pair_hz
        self.pair_focus = pair_focus if pair_hz else 0.0
        self.el_frames = el_frames
        """Electrode features of these crop frames (None: no electrodes)."""
        self.shelf_p, self.shelf_db = shelf_p, shelf_db
        self.shelf_edge, self.shelf_width = shelf_edge, shelf_width
        self._pairs: dict[str, np.ndarray] = {}
        self.spec = PowerSpectrogram(NFFT, STEP, RATE, dev)
        self.rng = np.random.default_rng(seed)
        self.lines = {}
        for m in self.mixes:
            key = m.lab_name()
            if key not in self.lines:
                self.lines[key] = unknown_lines(m, self.spec, dev)
        self._gpu: dict[str, dict] = {}
        self.pool = ThreadPoolExecutor(workers) if workers else None
        self._pending: deque = deque()

    # --- drawing and reading ----------------------------------------------

    def _span(self, crop: Crop):
        """Source samples of the lab traces [s0, s0 + n_in), the output
        samples n_out of the resampled segment (with margins M on both
        sides), and the unwarped range [a, a + N) of the field noise."""
        T = self.frames + 2 * self.ctx
        N = (T - 1) * STEP + NFFT
        a = crop.j0 * STEP
        if crop.r == 1.0 and crop.tilt == 0.0:
            return a, N, N, 0, a, N
        # FFT lengths with small prime factors only (cuFFT fails on some
        # large primes); the exact ratio n_in / n_out is the speed used
        n_out = _smooth(N + 2 * (NFFT // 2))
        M = (n_out - N) // 2
        n_in = _smooth(round(n_out * crop.r))
        s0 = a - round(M * n_in / n_out)
        return s0, n_in, n_out, M, a, N

    def draw(self) -> Crop:
        rng = self.rng
        mix = self.mixes[rng.integers(len(self.mixes))]
        n_el = (
            MIX_CHANNELS if rng.random() < 0.5 else int(rng.integers(2, MIX_CHANNELS))
        )
        electrodes = sorted(rng.choice(MIX_CHANNELS, n_el, replace=False).tolist())
        u = rng.random()
        gain = 0.0 if u < 0.25 else 1.0 if u < 0.5 else float(rng.uniform(0, 3))
        r = 1.0
        if self.speed and rng.random() < self.speed_p:
            lo, hi = np.log(self.speed[0]), np.log(self.speed[1])
            r = float(np.exp(rng.uniform(lo, hi)))
        tilt, f_ref = 0.0, 500.0
        if self.tilt_db and rng.random() < self.tilt_p:
            tilt = float(rng.uniform(-self.tilt_db, self.tilt_db))
            f_ref = float(np.exp(rng.uniform(np.log(100.0), np.log(1500.0))))
        # candidate fundamentals of the windows: in the crop's frequencies
        k0, k1 = int(F_MIN / DF), int(F_MAX / DF) - self.bins
        k_los = [int(rng.integers(k0, k1)) for _ in range(self.windows)]
        T = self.frames + 2 * self.ctx
        crop = Crop(mix, 0, electrodes, gain, k_los, r, tilt, f_ref)
        if self.shelf_p and rng.random() < self.shelf_p:
            lo, hi = np.log(self.shelf_edge[0]), np.log(self.shelf_edge[1])
            crop.shelf = (
                float(np.exp(rng.uniform(lo, hi))),
                float(rng.uniform(*self.shelf_db)),
                float(rng.uniform(*self.shelf_width)),
            )
        # output frames whose source span stays inside the recording
        n_samples = mix.raw.shape[0]
        s0, n_in, *_ = self._span(crop)
        lead = crop.j0 * STEP - s0  # samples the lab span starts before a
        need = max(n_in - lead, (T - 1) * STEP + NFFT)
        j_lo = int(np.ceil(max(lead, 0) / STEP))
        j_hi = (n_samples - need) // STEP
        crop.j0 = int(rng.integers(j_lo, max(j_lo + 1, j_hi)))
        pairs = self.close_pairs(mix) if self.pair_focus else None
        if pairs is not None and len(pairs) and rng.random() < self.pair_focus:
            # centre the crop on a close pair: a random output frame and
            # position in the first frequency window
            p_frame, p_freq = pairs[rng.integers(len(pairs))]
            j_out = int(rng.integers(0, self.frames))
            j0 = round(p_frame - (self.ctx + j_out) * r)
            crop.j0 = int(np.clip(j0, j_lo, max(j_lo, j_hi - 1)))
            k = round(p_freq * r / DF) - int(rng.integers(8, self.bins - 8))
            crop.k_los[0] = int(np.clip(k, k0, k1 - 1))
        return crop

    def close_pairs(self, mix: Mixture) -> np.ndarray:
        """(frame, mean frequency) of every pair of known fish closer than
        ``pair_hz`` (cached per mixture)."""
        key = str(mix.path)
        if key not in self._pairs:
            F = mix.freqs
            rows = []
            for i in range(len(F)):
                d = np.abs(F[i + 1 :] - F[i])
                a, j = np.nonzero(d < self.pair_hz)
                rows.append(np.stack([j, 0.5 * (F[i, j] + F[i + 1 + a, j])], 1))
            self._pairs[key] = np.concatenate(rows) if rows else np.zeros((0, 2))
        return self._pairs[key]

    def read(self, crop: Crop):
        """Raw (samples, channels) arrays: the lab traces over the union of
        the warped and the unwarped span, and the field variant over the
        unwarped one (None without field noise)."""
        s0, n_in, _, _, a, N = self._span(crop)
        u0, u1 = min(s0, a), max(s0 + n_in, a + N)
        lab = np.array(crop.mix.raw[u0:u1])
        field = None
        if crop.gain and crop.mix.field is not None:
            field = np.array(crop.mix.field[a : a + N])
        return u0, lab, field

    def prefetch(self, n: int) -> list[tuple[Crop, Any]]:
        """The next n crops with their (pending) reads."""
        while len(self._pending) < 3 * n:
            c = self.draw()
            fut = self.pool.submit(self.read, c) if self.pool else None
            self._pending.append((c, fut))
        out = []
        for _ in range(n):
            c, fut = self._pending.popleft()
            out.append((c, fut.result() if fut else self.read(c)))
        return out

    # --- building on the GPU ----------------------------------------------

    def _mix_gpu(self, mix: Mixture) -> dict:
        key = str(mix.path)
        if key not in self._gpu:
            t = lambda a: torch.as_tensor(np.ascontiguousarray(a), device=self.dev)  # noqa: E731
            self._gpu[key] = {
                "freqs": t(mix.freqs).float(),
                "guess": t(mix.guess).float(),
                "completed": t(mix.completed),
                "fish_power": t(mix.fish_power).float(),
                "noise_lab": t(mix.noise_lab).float(),
                "noise_field": None
                if mix.noise_field is None
                else t(mix.noise_field).float(),
                "lines": t(self.lines[mix.lab_name()]),
            }
        return self._gpu[key]

    def traces(self, crop: Crop, raw) -> torch.Tensor:
        """(channels, N) output traces on the GPU: lab traces resampled and
        tilted, plus the field noise at its gain."""
        u0, lab_np, field_np = raw
        s0, n_in, n_out, M, a, N = self._span(crop)
        lab = torch.from_numpy(lab_np).to(self.dev, non_blocking=True)
        lab = lab[:, crop.electrodes].T.contiguous()  # (ch, n)
        if crop.r == 1.0 and crop.tilt == 0.0:
            y = lab[:, a - u0 : a - u0 + N]
        else:
            seg = lab[:, s0 - u0 : s0 - u0 + n_in]
            X = torch.fft.rfft(seg)
            nb = n_out // 2 + 1
            X = X[..., :nb] if nb <= X.shape[-1] else F.pad(X, (0, nb - X.shape[-1]))
            if crop.tilt:
                fo = torch.arange(nb, device=self.dev, dtype=torch.float32) * (
                    RATE / n_out
                )
                g = crop.tilt * torch.log2(fo.clamp_min(20.0) / crop.f_ref)
                X = X * (10.0 ** (g.clamp(-TILT_MAX_DB, TILT_MAX_DB) / 20.0))
            y = torch.fft.irfft(X, n=n_out) * (n_out / n_in)
            y = y[:, M : M + N]
        if field_np is not None:
            fld = torch.from_numpy(field_np).to(self.dev, non_blocking=True)
            fld = fld[:, crop.electrodes].T
            y = y + crop.gain * (fld - lab[:, a - u0 : a - u0 + N])
        return y

    def add_shelf(self, crop: Crop, x: torch.Tensor):
        """Traces (channels, N) + the crop's noise shelf, and a function
        giving the SNR loss [dB] of a fish at frequencies f (any shape):
        10 log10(1 + added / present noise), both from the crop's own
        spectrogram (time median of the power in 100-Hz bands)."""
        edge, raise_db, width = crop.shelf
        n_el, N = x.shape
        p0 = summed_power(self.spec, x)  # (bins, frames)
        nb = round(100.0 / DF)
        K = (p0.shape[0] // nb) * nb
        floor = p0[:K].reshape(-1, nb, p0.shape[1]).median(1).values.median(1).values
        fc = (torch.arange(len(floor), device=self.dev) + 0.5) * nb * DF
        k_e = min(len(floor) - 1, int((edge + 100.0) / (nb * DF)))
        add = floor[k_e] * (10.0 ** (raise_db / 10.0) - 1.0)  # summed PSD
        sigma = torch.sqrt(add * RATE / (2.0 * n_el))
        w = torch.randn((n_el, N), device=self.dev, generator=None)
        W = torch.fft.rfft(w)
        fo = torch.arange(W.shape[-1], device=self.dev, dtype=torch.float32) * (
            RATE / N
        )
        step = torch.sigmoid((fo - edge) / (width / 4.0))
        noise = torch.fft.irfft(W * step.sqrt(), n=N) * sigma

        def loss(f: torch.Tensor) -> torch.Tensor:
            s = torch.sigmoid((f - edge) / (width / 4.0))
            i = torch.clamp(torch.bucketize(f, fc), 1, len(fc) - 1)
            a = ((f - fc[i - 1]) / (fc[i] - fc[i - 1])).clamp(0, 1)
            present = floor[i - 1] * (1 - a) + floor[i] * a
            return 10 * torch.log10(1 + add * s / present.clamp_min(1e-30))

        return x + noise.to(x.dtype), loss

    def source_frames(self, crop: Crop, j: np.ndarray) -> tuple[np.ndarray, float]:
        """Fractional source frame of output frames j, and the exact speed."""
        if crop.r == 1.0 and crop.tilt == 0.0:
            return (crop.j0 + j).astype(np.float64), 1.0
        s0, n_in, n_out, M, _, _ = self._span(crop)
        r = n_in / n_out
        s = s0 + (j * STEP + NFFT / 2 + M) * r
        return (s - NFFT / 2) / STEP, r

    def build(self, crop: Crop, raw):
        """-> features (windows, harmonics, T, bins) and targets, weights,
        offset targets and weights (windows, frames, bins)."""
        dev = self.dev
        x = self.traces(crop, raw)
        el = None
        shelf = None
        if crop.shelf is not None:
            x, shelf = self.add_shelf(crop, x)
        if self.el_frames is None:
            power = summed_power(self.spec, x)
        else:
            power, X = summed_power(self.spec, x, crop.k_los, self.bins, self.el_frames)
            el = torch.stack(
                [electrode_features(Xw, 0, self.bins - 1) for Xw in X]
            ).permute(0, 2, 3, 1, 4)  # (windows, electrodes, 3, frames, bins)
            n = el.shape[1]
            el = F.pad(el, (0, 0, 0, 0, 0, 0, 0, MIX_CHANNELS - n))
            mask = torch.zeros((len(el), MIX_CHANNELS), dtype=torch.bool, device=dev)
            mask[:, :n] = True
        log = decibel(power)
        k_hi_max = max(crop.k_los) + self.bins - 1
        k_need = min(log.shape[0], math.ceil(max(self.harmonics) * k_hi_max) + 4)
        z = normalise(log[:k_need].T, DF)
        feats = torch.stack(
            [
                harmonic_stack(z, k, k + self.bins - 1, self.harmonics)
                for k in crop.k_los
            ]
        )
        # truth on the frames the loss sees
        g = self._mix_gpu(crop.mix)
        j = np.arange(self.ctx, self.ctx + self.frames)
        q, r = self.source_frames(crop, j)
        nf = crop.mix.n_frames
        q = np.clip(q, 0, nf - 1)
        i0 = np.floor(q).astype(np.int64)
        i1 = np.minimum(i0 + 1, nf - 1)
        w1 = torch.as_tensor(q - i0, device=dev, dtype=torch.float32)
        i0t = torch.as_tensor(i0, device=dev)
        i1t = torch.as_tensor(i1, device=dev)
        near = torch.as_tensor(np.round(q).astype(np.int64), device=dev)

        def lerp(a):  # NaN where a neighbour with weight is unknown
            return torch.where(w1 > 0, a[:, i0t] * (1 - w1) + a[:, i1t] * w1, a[:, i0t])

        f = lerp(g["freqs"]) * r  # NaN where either neighbour is unknown
        guess = lerp(g["guess"]) * r
        comp = g["completed"][:, near]
        fish_p = g["fish_power"][:, near]
        noise_lab = g["noise_lab"][:, near]
        tilt = torch.ones_like(fish_p)
        if crop.tilt:
            gdb = crop.tilt * torch.log2(
                torch.nan_to_num(f, nan=crop.f_ref).clamp_min(20.0) / crop.f_ref
            )
            tilt = 10.0 ** (gdb.clamp(-TILT_MAX_DB, TILT_MAX_DB) / 10.0)
        noise = noise_lab * tilt
        if crop.gain and g["noise_field"] is not None:
            noise = noise + crop.gain**2 * (g["noise_field"][:, near] - noise_lab)
        snr = 10 * torch.log10(fish_p * tilt / noise)
        snr = torch.nan_to_num(snr, nan=-99.0, neginf=-99.0, posinf=99.0)
        if shelf is not None:
            snr = snr - shelf(torch.nan_to_num(f, nan=0.0))
        ok = torch.isfinite(f) & (snr >= self.snr_ignore)  # (fish, frames)
        f0 = torch.where(torch.isfinite(f), f, torch.full_like(f, -1e9))
        gs = torch.where(torch.isfinite(guess), guess, torch.full_like(guess, -1e9))
        half = math.ceil(1.5 / DF) * DF
        strong = ok & (snr >= self.harm_snr)
        lines_all = g["lines"]
        out = []
        for k_lo in crop.k_los:
            fk = (
                torch.arange(k_lo, k_lo + self.bins, device=dev, dtype=torch.float32)
                * DF
            )
            d = fk[None, None, :] - f0[:, :, None]  # (fish, frames, bins)
            gauss = torch.exp(-0.5 * (d / self.sigma_hz) ** 2) * ok[:, :, None]
            target = gauss.amax(0)
            around = (fk[None, None, :] - gs[:, :, None]).abs() <= half
            ignore = ((~ok)[:, :, None] & around).any(0)
            halfw = ((ok & comp)[:, :, None] & around).any(0)
            weight = torch.where(
                ignore,
                torch.zeros_like(target),
                torch.where(
                    halfw, torch.full_like(target, 0.5), torch.ones_like(target)
                ),
            )
            if self.harm_weight != 1.0:
                hard = torch.zeros_like(target, dtype=torch.bool)
                for h in self.hard_harmonics:
                    tol = HARD_TOL_HZ * max(1.0, h / 2)
                    hd = (fk[None, None, :] - h * f0[:, :, None]).abs() <= tol
                    hard |= (hd & strong[:, :, None]).any(0)
                hard &= (target < 0.1) & (weight > 0)
                weight = torch.where(hard, weight * self.harm_weight, weight)
            # unknown lines of the lab variant, at the source's frames and bins
            kb = torch.round(
                torch.arange(k_lo, k_lo + self.bins, device=dev, dtype=torch.float64)
                / r
            ).long()
            inb = kb < lines_all.shape[1]
            lines = lines_all[near][:, kb.clamp(max=lines_all.shape[1] - 1)] & inb
            weight = weight.masked_fill(lines, 0.0)
            # sub-bin offsets at the truth bins
            off_t = torch.zeros_like(target)
            off_w = torch.zeros_like(target)
            kf = f0 / DF
            kr = torch.round(kf)
            kk = (kr - k_lo).long()
            m = ok & (kk >= 0) & (kk < self.bins)
            fi, fr = torch.nonzero(m, as_tuple=True)
            off_t[fr, kk[fi, fr]] = (kf - kr)[fi, fr]
            off_w[fr, kk[fi, fr]] = 1.0
            parts = {"target": target, "weight": weight, "off_t": off_t, "off_w": off_w}
            if self.pair_hz:
                q = pair_targets(f0, ok, k_lo, self.bins, weight, self.pair_hz)
                parts |= dict(zip(("q_t", "q_w", "po_t", "po_w"), q, strict=True))
            out.append(parts)
        ex = {k: torch.stack([o[k] for o in out]) for k in out[0]}
        ex["feats"] = feats
        if el is not None:
            ex["el"], ex["el_mask"] = el, mask
        return ex

    def example(self, mix, j0, k_lo, electrodes, gain):
        """One fixed window without augmentation (validation)."""
        crop = Crop(mix, j0, list(electrodes), gain, [k_lo])
        return {k: v[0] for k, v in self.build(crop, self.read(crop)).items()}

    def batch(self, n_crops: int) -> dict[str, torch.Tensor]:
        ex = [self.build(c, raw) for c, raw in self.prefetch(n_crops)]
        return {k: torch.cat([e[k] for e in ex]) for k in ex[0]}


def electrode_window(model: SalienceNet, frames: int) -> tuple[slice | None, int]:
    """Crop frames whose electrode features a model needs for the loss
    frames, and the first of them: into the pair branch, the loss frames with
    the pair branch's and the encoder's context; into the trunk, all."""
    spec = model.spec
    if not spec.electrode_channels:
        return None, 0
    if spec.electrode_into != "pair":
        return slice(None), 0
    ctx = model.context_frames
    reach = model.electrode_context + sum(
        d * (spec.kernel_frames // 2) for d in spec.pair_dilations
    )
    return slice(ctx - reach, ctx + frames + reach), ctx - reach


PAIR_NEAR_BINS = 2.5  # pair-head negatives at full weight this close to a fish
PAIR_FAR_WEIGHT = 0.1  # ... and elsewhere at this weight (no peaks there)


def pair_targets(f0, ok, k_lo, n_bins, weight, pair_hz):
    """Targets of the pair head in one window: (pair, weight, offsets (2, ...),
    offset weight), each (frames, bins). A bin is a pair bin if its centre is
    within half a bin of the span of two fish (at least -3 dB, see ``ok``)
    closer than `pair_hz`; the offsets are those of the lower and the upper
    fish from the bin [bins]. Bins one more bin outside a span are ignored
    (a peak may sit there or not), bins near single fish are negatives, bins
    away from fish weak negatives."""
    dev = f0.device
    n, T = f0.shape
    fb = f0 / DF
    q = torch.zeros((T, n_bins), device=dev)
    amb = torch.zeros((T, n_bins), dtype=torch.bool, device=dev)
    po = torch.zeros((2, T, n_bins), device=dev)
    kk = torch.arange(k_lo, k_lo + n_bins, device=dev, dtype=torch.float32)
    near = (
        ((kk[None, None, :] - fb[:, :, None]).abs() <= PAIR_NEAR_BINS) & ok[:, :, None]
    ).any(0)
    fi = torch.where(ok, fb, torch.full_like(fb, float("nan")))
    close = (fi[:, None, :] - fi[None, :, :]).abs() < pair_hz / DF
    close &= torch.triu(torch.ones(n, n, dtype=torch.bool, device=dev), 1)[:, :, None]
    a, b, t = torch.nonzero(close, as_tuple=True)
    if len(t):
        lo = torch.minimum(fb[a, t], fb[b, t])
        hi = torch.maximum(fb[a, t], fb[b, t])
        k0 = torch.ceil(lo - 1.5)
        for s in range(math.ceil(pair_hz / DF) + 4):
            k = k0 + s
            idx = (k - k_lo).long()
            inb = (idx >= 0) & (idx < n_bins)
            inside = inb & (k >= lo - 0.5) & (k <= hi + 0.5)
            edge = inb & ~inside & (k >= lo - 1.5) & (k <= hi + 1.5)
            q[t[inside], idx[inside]] = 1.0
            po[0, t[inside], idx[inside]] = (lo - k)[inside]
            po[1, t[inside], idx[inside]] = (hi - k)[inside]
            amb[t[edge], idx[edge]] = True
    w = weight * torch.where(near, 1.0, PAIR_FAR_WEIGHT)
    w = torch.where(amb & (q == 0), torch.zeros_like(w), w)
    po_w = q * (weight > 0)
    return q, w, po, po_w


def losses(
    model,
    batch: dict,
    ctx: int,
    focal: float = 0.0,
    pair_pos_weight: float = 1.0,
    el_start: int = 0,
):
    target, weight, off_t, off_w = (
        batch[k] for k in ("target", "weight", "off_t", "off_w")
    )
    feats = batch["feats"].contiguous(memory_format=torch.channels_last)
    out = model.outputs(feats, batch.get("el"), batch.get("el_mask"), el_start)
    logits, offset = out["logits"], out["offset"]
    sl = slice(ctx, ctx + target.shape[1])
    logits, offset = logits[:, sl].float(), offset[:, sl].float()
    if focal:  # down-weight what is already right: |target - p|^gamma
        p = torch.sigmoid(logits).detach()
        weight = weight * (target - p).abs().pow(focal)
    bce = F.binary_cross_entropy_with_logits(
        logits, target, weight=weight, reduction="sum"
    )
    bce = bce / weight.sum().clamp_min(1.0)
    l1 = ((offset - off_t).abs() * off_w).sum() / off_w.sum().clamp_min(1.0)
    if "q_t" not in batch or "pair_logits" not in out:
        return bce, l1
    q_t, q_w, po_t, po_w = (batch[k] for k in ("q_t", "q_w", "po_t", "po_w"))
    q_w = q_w * (1.0 + (pair_pos_weight - 1.0) * q_t)
    ql = out["pair_logits"][:, sl].float()
    bce_q = F.binary_cross_entropy_with_logits(ql, q_t, weight=q_w, reduction="sum")
    bce_q = bce_q / q_w.sum().clamp_min(1.0)
    po = torch.sort(out["pair_offset"][:, :, sl].float(), dim=1).values
    l1_q = ((po - po_t).abs().sum(1) * po_w).sum() / po_w.sum().clamp_min(1.0)
    return bce, l1, bce_q, l1_q


THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
PAIR_THRESHOLD = 0.5  # validation: pair probability that splits a peak


def frame_scores(model, data: Batches, examples, ctx, tol=1.2, el_start: int = 0):
    """Recall per SNR band and precision of salience peaks on fixed crops, at
    every threshold in THRESHOLDS; ``best_f1`` is the best F1 (recall over all
    fish frames at >= 0 dB) over thresholds, so that checkpoints are compared
    independently of how their salience is calibrated."""
    th = np.array(THRESHOLDS)
    hits = {b: np.zeros(len(th)) for b in ("faint", "visible", "pair")}
    tot = {"faint": 0, "visible": 0, "pair": 0}
    n_det, n_match = np.zeros(len(th)), np.zeros(len(th))
    for mix, j0, k_lo, el, gain in examples:
        ex = data.example(mix, j0, k_lo, el, gain)
        feats = ex["feats"]
        with (
            torch.no_grad(),
            torch.autocast("cuda", torch.bfloat16, enabled=feats.is_cuda),
        ):
            out = model.outputs(
                feats[None],
                ex["el"][None] if "el" in ex else None,
                ex["el_mask"][None] if "el" in ex else None,
                el_start,
            )
        sl = slice(ctx, ctx + data.frames)
        pair = {}
        if "pair_logits" in out:
            pair = {
                "pair_logits": out["pair_logits"][0, sl].float(),
                "pair_offset": out["pair_offset"][0, :, sl].float(),
                "pair_threshold": PAIR_THRESHOLD,
            }
        fr, _, f, sal = decode(
            out["logits"][0, sl].float(),
            out["offset"][0, sl].float(),
            th.min(),
            k_lo,
            DF,
            **pair,
        )
        rows = slice(j0 + ctx, j0 + ctx + data.frames)
        tf, snr = mix.freqs[:, rows], mix.snr_at(gain)[:, rows]
        lo, hi = k_lo * DF + 1.5, (k_lo + data.bins - 1) * DF - 1.5
        for j in range(data.frames):
            sel = fr == j
            d_all, s_all = f[sel], sal[sel]
            t = tf[:, j]
            inb = np.isfinite(t) & (t > lo) & (t < hi)
            bands = []
            for i in np.flatnonzero(inb):
                b = "visible" if snr[i, j] >= 6 else "faint" if snr[i, j] >= 0 else None
                if b:
                    tot[b] += 1
                close = np.abs(np.delete(t, i) - t[i]) < 1.2
                in_pair = b is not None and bool(close.any())
                tot["pair"] += in_pair
                bands.append((i, b, in_pair))
            tt = t[np.isfinite(t)]
            for q, thr in enumerate(th):
                keep = s_all >= thr
                d = d_all[keep]
                used = np.zeros(len(d), bool)
                for i, b, in_pair in bands:
                    if not len(d) or b is None:
                        continue
                    dist = np.where(used, np.inf, np.abs(d - t[i]))
                    m = int(np.argmin(dist))
                    if dist[m] <= tol:
                        used[m] = True
                        hits[b][q] += 1
                        hits["pair"][q] += in_pair
                dd = d[(d > lo) & (d < hi)]
                n_det[q] += len(dd)
                if len(dd) and len(tt):
                    n_match[q] += np.sum(
                        np.abs(dd[:, None] - tt[None, :]).min(1) <= tol
                    )
    prec = n_match / np.maximum(n_det, 1)
    rec = (hits["faint"] + hits["visible"]) / max(tot["faint"] + tot["visible"], 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    q5 = int(np.flatnonzero(th == 0.5)[0])
    best = int(np.argmax(f1))
    return {
        "recall_visible": float(hits["visible"][q5] / max(tot["visible"], 1)),
        "recall_faint": float(hits["faint"][q5] / max(tot["faint"], 1)),
        "recall_pair": float(hits["pair"][q5] / max(tot["pair"], 1)),
        "precision": float(prec[q5]),
        "best_f1": float(f1[best]),
        "best_thr": float(th[best]),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tag", required=True)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=8, help="crops per step")
    p.add_argument("--windows", type=int, default=1, help="frequency windows per crop")
    p.add_argument(
        "--frames", type=int, default=300, help="frames per crop seen by the loss"
    )
    p.add_argument("--bins", type=int, default=512, help="candidate bins per window")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument(
        "--dilations", default="1,2,4,8,16,1", help="time dilation per layer"
    )
    p.add_argument(
        "--temporal-dilations",
        default="",
        help="time dilations of the temporal stack (empty: none)",
    )
    p.add_argument("--temporal-kernel", type=int, default=3)
    p.add_argument("--temporal-kernel-bins", type=int, default=1)
    p.add_argument(
        "--temporal-bin-dilations",
        default="",
        help="frequency dilation per temporal layer (empty: 1)",
    )
    p.add_argument(
        "--harmonics",
        default="1/3,1/2,1,2,3,4,5",
        help="channels of the harmonic stack",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--train-fraction",
        type=float,
        default=1.0,
        help="evenly spaced share of mixtures",
    )
    p.add_argument("--snr-ignore", type=float, default=-3.0)
    p.add_argument(
        "--speed", default="", help="speed perturbation range lo,hi (e.g. 0.85,1.18)"
    )
    p.add_argument("--speed-p", type=float, default=0.75)
    p.add_argument("--tilt-db", type=float, default=0.0, help="max dB per octave")
    p.add_argument("--tilt-p", type=float, default=0.5)
    p.add_argument(
        "--shelf-p",
        type=float,
        default=0.0,
        help="share of crops with a noise shelf (harmonics under a raised floor)",
    )
    p.add_argument("--shelf-db", default="10,30", help="raise of the floor [dB]")
    p.add_argument("--shelf-edge", default="150,1200", help="edge frequency [Hz]")
    p.add_argument(
        "--harm-weight",
        type=float,
        default=1.0,
        help="loss weight of hard negatives at (sub)harmonics of strong fish",
    )
    p.add_argument("--harm-snr", type=float, default=10.0)
    p.add_argument(
        "--hard-harmonics",
        default="1/3,1/2,2,3,4,5",
        help="multiples of strong fish that are hard negatives",
    )
    p.add_argument(
        "--focal", type=float, default=0.0, help="focal-loss exponent (0: BCE)"
    )
    p.add_argument(
        "--sigma-hz", type=float, default=SIGMA_HZ, help="width of the target bumps"
    )
    p.add_argument(
        "--pair-head",
        action="store_true",
        help="train a pair head (two fish in one peak, see pair_targets)",
    )
    p.add_argument("--pair-hz", type=float, default=1.2, help="pairs closer than this")
    p.add_argument(
        "--pair-dilations", default="", help="time dilations of the pair branch"
    )
    p.add_argument("--pair-weight", type=float, default=1.0, help="pair loss weight")
    p.add_argument(
        "--pair-focus",
        type=float,
        default=0.0,
        help="share of crops centred on a close pair (time and first window)",
    )
    p.add_argument(
        "--pair-pos-weight",
        type=float,
        default=1.0,
        help="loss weight of pair bins relative to other bins",
    )
    p.add_argument(
        "--electrode-channels",
        type=int,
        default=0,
        help="width of the electrode branch (0: none)",
    )
    p.add_argument("--electrode-dilations", default="1,2,4,8")
    p.add_argument("--electrode-into", default="pair", choices=("pair", "trunk"))
    p.add_argument(
        "--init", default="", help="start from this model's weights (new parts random)"
    )
    p.add_argument(
        "--freeze-trunk",
        action="store_true",
        help="with --init: train only what the init model does not have",
    )
    p.add_argument("--checkpoint", action="store_true", help="recompute layers")
    p.add_argument(
        "--resume",
        action="store_true",
        help="continue from the last validation step's state.pt, if there is one",
    )
    p.add_argument("--workers", type=int, default=4, help="reader threads")
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=0, help="keep sSTEP.pt")
    p.add_argument("--train-split", default="ltrain")
    p.add_argument("--val-split", default="ldev")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--limit", type=int, help="use only the first N mixtures per split (tests)"
    )
    args = p.parse_args(argv)
    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    spec = SalienceSpec(
        freq_resolution=DF,
        frame_step=FT,
        harmonics=tuple(float(Fraction(h)) for h in args.harmonics.split(",")),
        channels=args.channels,
        dilations=tuple(int(d) for d in args.dilations.split(",")),
        temporal_dilations=tuple(
            int(d) for d in args.temporal_dilations.split(",") if d
        ),
        temporal_kernel=args.temporal_kernel,
        temporal_kernel_bins=args.temporal_kernel_bins,
        temporal_bin_dilations=tuple(
            int(d) for d in args.temporal_bin_dilations.split(",") if d
        ),
    )
    if args.pair_head:
        spec = replace(
            spec,
            pair_head=True,
            pair_dilations=tuple(int(d) for d in args.pair_dilations.split(",") if d),
        )
    if args.electrode_channels:
        spec = replace(
            spec,
            electrode_channels=args.electrode_channels,
            electrode_dilations=tuple(
                int(d) for d in args.electrode_dilations.split(",") if d
            ),
            electrode_into=args.electrode_into,
        )
    frozen: list[str] = []
    if args.init:
        init_model, init_spec, _ = load_model(args.init, "cpu")
        new = (
            "pair_head",
            "pair_dilations",
            "electrode_channels",
            "electrode_dilations",
            "electrode_into",
        )
        spec = replace(init_spec, **{k: getattr(spec, k) for k in new})
        state = init_model.state_dict()
    model = SalienceNet(spec).to(dev, memory_format=torch.channels_last)
    if args.init:
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            raise SystemExit(f"--init: unexpected keys {unexpected}")
        print(
            f"init from {args.init}; new: {sorted({k.split('.')[0] for k in missing})}"
        )
        if args.freeze_trunk:
            frozen = [n for n, _ in model.named_parameters() if n in state]
            for n, prm in model.named_parameters():
                prm.requires_grad_(n not in state)
    model.checkpoint = args.checkpoint
    ctx = model.context_frames
    el_frames, el_start = electrode_window(model, args.frames)
    out = OUT / args.tag
    out.mkdir(parents=True, exist_ok=True)
    train = Batches(
        args.train_split,
        args.frames,
        ctx,
        args.bins,
        args.snr_ignore,
        dev,
        1 + 10 * args.seed,
        args.limit,
        args.train_fraction,
        harmonics=spec.harmonics,
        windows=args.windows,
        speed=tuple(float(v) for v in args.speed.split(",")) if args.speed else None,
        speed_p=args.speed_p,
        tilt_db=args.tilt_db,
        tilt_p=args.tilt_p,
        harm_weight=args.harm_weight,
        harm_snr=args.harm_snr,
        hard_harmonics=tuple(
            float(Fraction(h)) for h in args.hard_harmonics.split(",")
        ),
        sigma_hz=args.sigma_hz,
        workers=args.workers,
        pair_hz=args.pair_hz if spec.pair_head or args.pair_focus else 0.0,
        pair_focus=args.pair_focus,
        el_frames=el_frames,
        shelf_p=args.shelf_p,
        shelf_db=tuple(float(v) for v in args.shelf_db.split(",")),
        shelf_edge=tuple(float(v) for v in args.shelf_edge.split(",")),
    )
    val = Batches(
        args.val_split,
        args.frames,
        ctx,
        args.bins,
        args.snr_ignore,
        dev,
        2,
        args.limit,
        harmonics=spec.harmonics,
        workers=0,
        el_frames=el_frames,
    )
    vr = np.random.default_rng(3)
    T = args.frames + 2 * ctx
    val_examples = {0.0: [], 1.0: []}  # lab and field variant
    for m in val.mixes:
        for _ in range(2):
            j0 = int(vr.integers(0, m.n_frames - T))
            k_lo = int(vr.integers(int(150 / DF), int(1300 / DF) - args.bins))
            for g, ex in val_examples.items():
                ex.append((m, j0, k_lo, list(range(MIX_CHANNELS)), g))
    n_par = sum(p.numel() for p in model.parameters())
    print(
        f"{args.tag}: {n_par} parameters, context +-{ctx} frames ({ctx * FT:.1f} s), "
        f"{len(train.mixes)} train / {len(val.mixes)} val mixtures",
        flush=True,
    )
    params = [prm for prm in model.parameters() if prm.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05
    )
    try:
        from torch.utils.tensorboard import SummaryWriter

        tb = SummaryWriter(str(out / "tb"))
    except ImportError:
        tb = None
    log, best, t0 = [], -1.0, time.perf_counter()
    first = 1
    state_path = out / "state.pt"
    if args.resume and state_path.exists():  # continue after a crash
        st = torch.load(state_path, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        train.rng = st["rng"]
        log, best, first = st["log"], st["best"], st["step"] + 1
        print(f"resumed after step {st['step']}", flush=True)
    t_mark = t0
    for step in range(first, args.steps + 1):
        model.train()
        if frozen:  # batch norms of the frozen parts keep their statistics
            for name, mod in model.named_modules():
                if isinstance(mod, torch.nn.BatchNorm2d) and f"{name}.weight" in frozen:
                    mod.eval()
        batch = train.batch(args.batch)
        with torch.autocast("cuda", torch.bfloat16, enabled=dev.type == "cuda"):
            bce, l1, *pl = losses(
                model, batch, ctx, args.focal, args.pair_pos_weight, el_start
            )
            loss = bce + 0.5 * l1
            if pl:
                loss = loss + args.pair_weight * (pl[0] + 0.5 * pl[1])
            if frozen:  # only the new parts learn
                loss = args.pair_weight * (pl[0] + 0.5 * pl[1])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0:
            now = time.perf_counter()
            pmsg = f" pair bce {pl[0].item():.4f} l1 {pl[1].item():.3f}" if pl else ""
            print(  # speed over the last 50 steps, without validation
                f"step {step}: bce {bce.item():.4f} l1 {l1.item():.3f}{pmsg} "
                f"({(now - t_mark) / 50:.2f} s/step)",
                flush=True,
            )
            t_mark = now
            if tb is not None:
                tb.add_scalar("train/bce", bce.item(), step)
                tb.add_scalar("train/offset_l1", l1.item(), step)
                if pl:
                    tb.add_scalar("train/pair_bce", pl[0].item(), step)
                    tb.add_scalar("train/pair_offset_l1", pl[1].item(), step)
                tb.add_scalar("train/lr", sched.get_last_lr()[0], step)
        if step % args.val_every == 0 or step == args.steps:
            model.eval()
            sc = {}
            for g, name in ((0.0, "lab"), (1.0, "field")):
                r = frame_scores(model, val, val_examples[g], ctx, el_start=el_start)
                sc |= {f"{k}_{name}": v for k, v in r.items()}
            row = {
                "step": step,
                "bce": bce.item(),
                "l1": l1.item(),
                **sc,
                "seconds": time.perf_counter() - t0,
            }
            log.append(row)
            t_mark = time.perf_counter()
            print(json.dumps(row), flush=True)
            meta = {"tag": args.tag, "step": step, "args": vars(args), "val": sc}
            save_model(out / "last.pt", model, spec, meta)
            if args.save_every and step % args.save_every == 0:
                save_model(out / f"s{step}.pt", model, spec, meta)
            if tb is not None:
                for k, v in sc.items():
                    metric, variant = k.rsplit("_", 1)
                    tb.add_scalar(f"val_{variant}/{metric}", v, step)
                tb.flush()
            # threshold-free: the best F1 over thresholds, lab and field
            score = sc["best_f1_lab"] + sc["best_f1_field"]
            if score > best:
                best = score
                save_model(out / "model.pt", model, spec, meta)
            (out / "log.json").write_text(json.dumps(log, indent=1))
            torch.save(
                {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "sched": sched.state_dict(),
                    "rng": train.rng,
                    "log": log,
                    "best": best,
                    "step": step,
                },
                state_path,
            )
    print(
        f"done in {(time.perf_counter() - t0) / 60:.1f} min; best model {out / 'model.pt'}"
    )


if __name__ == "__main__":
    main()
