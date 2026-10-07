"""Removal of stationary interference combs from per-electrode spectra.

Electrical interference often appears as a *comb*: persistent lines at
consecutive multiples ``k * spacing`` of a fundamental below the fish range
(for example 95.4 Hz -> 477, 573, 669, 764 Hz, ...). Within one spectrum such
a comb is indistinguishable from fish, because every tooth has harmonics that
are other teeth.

Per block and electrode the canceller

1. estimates the persistent spectrum: a low quantile over time within each
   block (a line must be present in most frames), followed by the minimum
   over the last ``history_blocks`` blocks (it must stay on the same bin for
   minutes), and a smooth noise baseline,
2. finds persistent lines that stand out from the baseline,
3. searches them for combs: a spacing below ``max_spacing`` with at least
   ``min_run`` consecutive teeth,
4. remembers the teeth found on each electrode for ``tooth_memory_blocks``
   blocks (single teeth intermittently miss the line test), and
5. gates the remembered teeth: in every frame, a tooth bin whose power is
   below a high level of the tooth (``subtract_quantile`` over time plus
   ``subtract_margin``) holds only interference and is set to the noise
   floor; a bin above it holds a fish and is left untouched.

What is *not* removed: a resting fish, even if it is stable for hours and
seen on a single electrode, because its harmonics are spaced by its own
fundamental (above ``max_spacing``); its sub-multiples only match every
2nd/3rd tooth, never consecutive runs. Subtracting (instead of zeroing) the
stationary level keeps a fish visible while it passes a tooth if it is
stronger than the tooth. A fish that is weaker than a tooth and stays within
~1 Hz (the spectral resolution) of it for the whole history (default 5
blocks of 60 s) is lost on the electrodes carrying the tooth: at this
resolution the two cannot be separated.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from .config import InterferenceConfig


@dataclass
class Comb:
    channel: int
    spacing: float
    """Fundamental of the comb [Hz]."""
    teeth: np.ndarray
    """Harmonic numbers of the detected teeth."""
    longest_run: int


def find_combs(
    freqs: np.ndarray,
    min_spacing: float,
    max_spacing: float,
    tol: float,
    min_run: int,
    max_combs: int = 4,
) -> list[tuple[float, np.ndarray, int]]:
    """Find combs among line frequencies.

    Returns a list of (spacing, indices of matched lines, longest run of
    consecutive teeth), best comb first.
    """
    freqs = np.asarray(freqs, dtype=float)
    remaining = np.arange(len(freqs))
    combs = []
    for _ in range(max_combs):
        f = freqs[remaining]
        if len(f) < min_run:
            break
        d = np.abs(f[:, None] - f[None, :])
        cand = np.unique(np.round(d[(d >= min_spacing) & (d <= max_spacing)], 2))
        if len(cand) == 0:
            break
        delta = cand[:, None]
        for _ in range(3):  # refine spacings by least squares on matched teeth
            k = np.round(f[None, :] / delta)
            ok = (k >= 1) & (np.abs(f[None, :] - k * delta) < tol)
            num = (f[None, :] * k * ok).sum(1, keepdims=True)
            den = (k * k * ok).sum(1, keepdims=True)
            delta = np.where(den > 0, num / np.maximum(den, 1), delta)
        k = np.round(f[None, :] / delta)
        ok = (k >= 1) & (np.abs(f[None, :] - k * delta) < tol)
        delta = delta[:, 0]

        best = None
        for i in np.nonzero(ok.sum(1) >= min_run)[0]:
            teeth = np.unique(k[i, ok[i]]).astype(int)
            run = _longest_run(teeth)
            if run < min_run:
                continue
            key = (ok[i].sum(), run, -delta[i])
            if best is None or key > best[0]:
                best = (key, i)
        if best is None:
            break
        i = best[1]
        combs.append((float(delta[i]), remaining[ok[i]], best[0][1]))
        remaining = remaining[~ok[i]]
    return combs


def merge_tooth_neighbours(
    frame, freq, power, teeth, max_sep, tol, tooth_sep=None
) -> np.ndarray:
    """Mask of detections to keep.

    Where a fish overlaps an interference tooth (within the ~2 Hz main lobe),
    removing the tooth can split the fish's peak into two detections on
    either side of it: two detections of a frame closer than `max_sep` with
    a tooth between them count as one, the stronger is kept. Near a strong
    fish its spectral leakage can also lift a tooth above the gate: a
    detection on a tooth (within `tol`) with a stronger non-tooth detection
    within `tooth_sep` is dropped. A lone detection on a tooth is kept (a
    fish may rest there).
    """
    keep = np.ones(len(freq), dtype=bool)
    if len(teeth) == 0 or len(freq) < 2:
        return keep
    tooth_sep = max_sep if tooth_sep is None else tooth_sep
    teeth = np.sort(teeth)
    order = np.lexsort((freq, frame))
    fr, fq, pw = frame[order], freq[order], power[order]
    pos = np.clip(np.searchsorted(teeth, fq), 1, len(teeth) - 1)
    on_tooth = np.minimum(np.abs(fq - teeth[pos - 1]), np.abs(fq - teeth[pos])) <= tol
    n = len(fr)
    for j in range(n):
        # split peak: adjacent detection with a tooth between them
        k = j + 1
        if k < n and fr[k] == fr[j] and fq[k] - fq[j] <= max_sep:
            lo = np.searchsorted(teeth, fq[j] - tol)
            if lo < len(teeth) and teeth[lo] <= fq[k] + tol:
                keep[order[j if pw[j] < pw[k] else k]] = False
        # tooth lifted by a nearby stronger fish
        if on_tooth[j]:
            for k in range(j - 1, -1, -1):
                if fr[k] != fr[j] or fq[j] - fq[k] > tooth_sep:
                    break
                if not on_tooth[k] and pw[k] > pw[j]:
                    keep[order[j]] = False
            for k in range(j + 1, n):
                if fr[k] != fr[j] or fq[k] - fq[j] > tooth_sep:
                    break
                if not on_tooth[k] and pw[k] > pw[j]:
                    keep[order[j]] = False
    return keep


def _extend_comb(
    freqs: np.ndarray, spacing: float, tol: float, start: float, free: np.ndarray
) -> tuple[float, np.ndarray]:
    """Match teeth over all `freqs`, refining the spacing while going up."""
    limit = start
    while True:
        k = np.round(freqs / spacing)
        ok = free & (k >= 1) & (np.abs(freqs - k * spacing) < tol) & (freqs < limit)
        if ok.any():
            spacing = float((freqs[ok] * k[ok]).sum() / (k[ok] ** 2).sum())
        if limit > freqs.max():
            return spacing, np.nonzero(ok)[0]
        limit *= 2


def _longest_run(sorted_ints: np.ndarray) -> int:
    if len(sorted_ints) == 0:
        return 0
    breaks = np.nonzero(np.diff(sorted_ints) != 1)[0]
    edges = np.concatenate([[-1], breaks, [len(sorted_ints) - 1]])
    return int(np.diff(edges).max())


def _running_median(x: torch.Tensor, width: int, step: int = 8) -> torch.Tensor:
    """Median over `width` bins along the last axis (subsampled by `step`)."""
    half = width // 2
    xp = torch.nn.functional.pad(x[None], (half, half), mode="replicate")[0]
    med = xp.unfold(-1, width, step).median(-1).values
    # interpolate back to full resolution
    pos = torch.arange(x.shape[-1], device=x.device, dtype=torch.float32) / step
    i0 = pos.floor().long().clamp_max(med.shape[-1] - 1)
    i1 = (i0 + 1).clamp_max(med.shape[-1] - 1)
    w = (pos - i0).clamp(0, 1)
    return med[..., i0] * (1 - w) + med[..., i1] * w


class CombCanceller:
    """Detects and subtracts interference combs block by block."""

    def __init__(self, cfg: InterferenceConfig, freqs: np.ndarray):
        self.cfg = cfg
        self.freqs = freqs
        self.df = float(freqs[1] - freqs[0])
        self.subtract: torch.Tensor | None = None
        self.floor: torch.Tensor | None = None
        self.combs: list[Comb] = []
        self.history: deque[torch.Tensor] = deque(maxlen=cfg.history_blocks)
        self.n_blocks = 0
        self.last_seen: np.ndarray | None = None
        self.tooth_freq: dict[int, float] = {}
        """Frequency (k * spacing) of every bin that was ever a tooth."""
        """Block count at which each (channel, bin) was last a comb tooth."""

    def __call__(self, power: torch.Tensor) -> torch.Tensor:
        """Return `power` (channels, freqs, frames) with comb teeth removed."""
        if power.shape[-1] >= self.cfg.min_frames:
            self._update(power)
        if self.subtract is not None:
            # Gate, don't subtract: a bin at or below the tooth's level holds
            # only interference and drops to the noise floor; a bin above it
            # holds something else (a fish) and is left untouched. Subtracting
            # would carve notches into fish peaks that overlap a tooth and
            # split them into two detections.
            level = self.subtract[..., None]
            hum_only = (level > 0) & (power <= level)
            power = torch.where(hum_only, self.floor[..., None], power)
        return power

    def _update(self, power: torch.Tensor) -> None:
        cfg = self.cfg
        # consecutive frames overlap strongly, a subset suffices for quantiles
        sub = power[..., :: cfg.frame_stride].contiguous()
        n_frames = sub.shape[-1]
        k_low = max(1, math.ceil(cfg.persistence_quantile * n_frames))
        k_high = max(1, math.ceil(cfg.subtract_quantile * n_frames))
        self.history.append(
            (
                sub.kthvalue(k_low, dim=-1).values,  # (channels, freqs)
                sub.kthvalue(k_high, dim=-1).values,
            )
        )
        del sub
        # minimum over blocks: a line must stay on its bin for the whole
        # history, and a fish resting on a tooth for part of it does not raise
        # the subtracted level (interference drifts of < subtract_margin are ok)
        persistent = torch.stack([h[0] for h in self.history]).amin(0)
        recent = list(self.history)[-cfg.level_history_blocks :]
        level = torch.stack([h[1] for h in recent]).amin(0)
        level = level * 10 ** (cfg.subtract_margin / 10)
        pers_db = 10 * torch.log10(persistent.clamp_min(1e-30))
        base_db = _running_median(pers_db, cfg.baseline_width)
        excess = (pers_db - base_db).cpu().numpy()
        pers_np = pers_db.cpu().numpy()

        self.n_blocks += 1
        if self.last_seen is None:
            self.last_seen = np.full(persistent.shape, -(10**9), dtype=np.int64)
        combs = []
        fmax_bin = len(self.freqs) - 2
        if cfg.max_line_freq is not None:
            fmax_bin = min(fmax_bin, int(cfg.max_line_freq / self.df))
        for c in range(power.shape[0]):
            e = excess[c, : fmax_bin + 1]
            peak = (
                (e[1:-1] > cfg.line_threshold) & (e[1:-1] >= e[:-2]) & (e[1:-1] > e[2:])
            )
            bins = np.nonzero(peak)[0] + 1
            if len(bins) < cfg.min_run:
                continue
            line_f = self._refine(pers_np[c], bins)
            # search on the low lines (fast), then extend to all lines
            low = line_f < cfg.search_max_freq
            used = np.zeros(len(line_f), dtype=bool)
            for spacing, _, run in find_combs(
                line_f[low],
                cfg.min_spacing,
                cfg.max_spacing,
                cfg.tooth_tolerance,
                cfg.min_run,
            ):
                spacing, idx = _extend_comb(
                    line_f, spacing, cfg.tooth_tolerance, cfg.search_max_freq, ~used
                )
                used[idx] = True
                teeth = np.round(line_f[idx] / spacing).astype(int)
                combs.append(Comb(c, spacing, teeth, run))
                self.last_seen[c, bins[idx]] = self.n_blocks
                for b_, k_ in zip(bins[idx], teeth, strict=True):
                    self.tooth_freq[int(b_)] = float(k_ * spacing)
        self.combs = combs

        # Subtract every tooth seen on a channel within the tooth memory, so a
        # tooth that misses the line test in one block is still removed.
        remembered = self.n_blocks - self.last_seen <= cfg.tooth_memory_blocks
        self.floor = 10 ** (base_db / 10)
        if not remembered.any():
            self.subtract = None
            return
        centers = torch.from_numpy(remembered).to(power.device, torch.float32)
        lobe = torch.nn.functional.max_pool1d(centers[:, None], 5, 1, 2)[:, 0] > 0
        is_line = persistent > self.floor
        self.subtract = torch.where(lobe & is_line, level, 0.0)

    def tooth_frequencies(self) -> np.ndarray:
        """Frequencies of all teeth currently remembered on any electrode."""
        if self.last_seen is None:
            return np.empty(0)
        recent = self.n_blocks - self.last_seen <= self.cfg.tooth_memory_blocks
        bins = np.unique(np.nonzero(recent)[1])
        return np.array(
            [self.tooth_freq[int(b)] for b in bins if int(b) in self.tooth_freq]
        )

    def _refine(self, row: np.ndarray, bins: np.ndarray) -> np.ndarray:
        a, m, c = row[bins - 1], row[bins], row[bins + 1]
        denom = a - 2 * m + c
        with np.errstate(invalid="ignore", divide="ignore"):
            delta = np.where(denom < 0, 0.5 * (a - c) / denom, 0.0)
        return (bins + np.clip(np.nan_to_num(delta), -0.5, 0.5)) * self.df
