"""Detection of harmonic groups (individual fish) in power spectra.

A faithful, parallel CPU port of the GPU kernels of Raab et al. (2022):

1. Hysteresis peak detection on each dB spectrum (`low_threshold`); peaks
   with a prominence above `high_threshold` inside the fundamental frequency
   range that are not mains harmonics are "good" peaks.
2. Every good peak divided by 1..`max_divisor` is a candidate fundamental.
   For each candidate the nearest peaks to its harmonics are collected; the
   candidate is scored by the mean power of its `min_group_size` lowest
   harmonics.
3. Greedy assignment: starting with the strongest good peak, the best
   scoring candidate containing it whose peaks are still unused becomes a
   fish. With ``exclusive_harmonics="all"`` (original behaviour) all
   harmonics of accepted fish are claimed; with "core" only the
   ``min_group_size`` lowest ones, so that a chance overlap of high
   harmonics does not suppress a fish.

Frames are processed independently and in parallel with numba.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from .config import HarmonicGroupsConfig


@njit(cache=True, nogil=True)
def _detect_peaks(
    data, freqs, low_th, high_th, min_freq, max_freq, mains, mains_tol, min_good, peaks
):
    """Mark peaks (1) and good peaks (2) of one dB spectrum in `peaks`."""
    direction = 0
    min_inx = 0
    max_inx = 0
    last_min_idx = 0
    last_max_idx = 0
    trough_count = 0
    peak_count = 0
    min_value = data[0]
    max_value = min_value
    p = 0
    t = 0
    for i in range(data.shape[0]):
        v = data[i]
        if direction > 0:
            if v > max_value:
                max_inx = i
                max_value = v
            if v <= max_value - low_th:
                peaks[max_inx] = 1
                p = 1
                last_max_idx = max_inx
                peak_count += 1
                direction = -1
                min_inx = i
                min_value = v
        if direction < 0:
            if v < min_value:
                min_inx = i
                min_value = v
            if v >= min_value + low_th:
                t = 1
                last_min_idx = min_inx
                trough_count += 1
                direction = 1
                max_inx = i
                max_value = v
        if direction == 0:
            if v <= max_value - low_th:
                direction = -1
            if v >= min_value + low_th:
                direction = 1
            if v > max_value:
                max_inx = i
                max_value = v
            if v < min_value:
                min_inx = i
                min_value = v
        if p != 0 and t != 0:
            p = 0
            t = 0
            if not data[last_max_idx] - data[last_min_idx] > high_th:
                continue
            f = freqs[last_max_idx]
            if f < min_freq or f > max_freq:
                continue
            if f % mains < mains_tol or abs(f % mains - mains) < mains_tol:
                continue
            if data[last_max_idx] < min_good:
                continue
            peaks[last_max_idx] = 2
    if peak_count > trough_count:
        peaks[last_max_idx] = 0


@njit(cache=True, nogil=True)
def _get_group(
    f0, pk_bins, freqs, log_row, out, min_group_size, max_freq_tol, mains, mains_tol
):
    """Collect harmonics of candidate `f0` into `out`; return its score."""
    fzero = f0
    fzero_h = 1
    for h in range(1, out.shape[0] + 1):
        ioi = 0
        fe = 1e6
        for k in range(pk_bins.shape[0]):
            new_fe = abs(freqs[pk_bins[k]] / h - fzero / fzero_h)
            if new_fe < fe and new_fe < max_freq_tol:
                ioi = pk_bins[k]
                fe = new_fe
            if new_fe > fe and ioi != 0:
                fzero = freqs[ioi]
                fzero_h = h
                out[h - 1] = ioi

    peak_sum = 0.0
    n = 0
    nn = 0
    for i in range(min_group_size):
        if out[i] != 0:
            nn += 1
            f = freqs[out[i]]
            if f % mains < mains_tol or abs(f % mains - mains) < mains_tol:
                continue
            n += 1
            peak_sum += log_row[out[i]]
    if nn < min_group_size - 1 or n == 0:
        return -1e6
    return peak_sum / n


@njit(cache=True, nogil=True)
def _frame_groups(
    log_row,
    freqs,
    low_th,
    high_th,
    min_freq,
    max_freq,
    mains,
    mains_tol,
    min_good,
    max_divisor,
    min_group_size,
    n_harmonics,
    max_freq_tol,
    n_exclusive,
    out_bins,
):
    """Detect fish in one spectrum; write fundamental bins to `out_bins`."""
    nf = log_row.shape[0]
    peaks = np.zeros(nf, dtype=np.int8)
    _detect_peaks(
        log_row,
        freqs,
        low_th,
        high_th,
        min_freq,
        max_freq,
        mains,
        mains_tol,
        min_good,
        peaks,
    )
    pk_bins = np.nonzero(peaks)[0]
    good = np.nonzero(peaks == 2)[0]
    good_f = good[(freqs[good] < max_freq) & (freqs[good] > min_freq)]
    n_cand = good_f.shape[0] * max_divisor
    if n_cand == 0:
        return 0

    groups = np.zeros((n_cand, n_harmonics), dtype=np.int64)
    values = np.empty(n_cand)
    for d in range(max_divisor):
        for j in range(good_f.shape[0]):
            c = d * good_f.shape[0] + j
            values[c] = _get_group(
                freqs[good_f[j]] / (d + 1),
                pk_bins,
                freqs,
                log_row,
                groups[c],
                min_group_size,
                max_freq_tol,
                mains,
                mains_tol,
            )

    # Candidates in descending score whose lowest harmonics are all present.
    order = np.argsort(-values, kind="mergesort")
    valid = np.empty(n_cand, dtype=np.int64)
    n_valid = 0
    for i in order:
        complete = True
        for h in range(min_group_size):
            if groups[i, h] == 0:
                complete = False
                break
        if complete:
            valid[n_valid] = i
            n_valid += 1

    assigned = np.zeros(nf, dtype=np.int8)
    good_order = good[np.argsort(-log_row[good], kind="mergesort")]
    n_found = 0
    for search_peak in good_order:
        for vi in range(n_valid):
            i = valid[vi]
            contains = False
            for h in range(min_group_size):
                if groups[i, h] == search_peak:
                    contains = True
                    break
            if not contains:
                continue
            if groups[i, 0] == 0 or log_row[groups[i, 0]] < min_good:
                continue
            used = False
            lowest = nf
            for h in range(n_harmonics):
                b = groups[i, h]
                if b != 0:
                    if h < n_exclusive and assigned[b] != 0:
                        used = True
                    lowest = min(lowest, b)
            if used:
                continue
            for h in range(n_exclusive):
                if groups[i, h] != 0:
                    assigned[groups[i, h]] = 1
            if n_found < out_bins.shape[0]:
                out_bins[n_found] = lowest
                n_found += 1
            break
    return n_found


@njit(cache=True, parallel=True)
def _detect_groups(
    log_spec,
    freqs,
    low_th,
    high_th,
    min_freq,
    max_freq,
    mains,
    mains_tol,
    min_good,
    max_divisor,
    min_group_size,
    n_harmonics,
    max_freq_tol,
    n_exclusive,
    max_groups,
):
    n_frames = log_spec.shape[0]
    bins = np.full((n_frames, max_groups), -1, dtype=np.int64)
    counts = np.zeros(n_frames, dtype=np.int64)
    for t in prange(n_frames):
        counts[t] = _frame_groups(
            log_spec[t],
            freqs,
            low_th,
            high_th,
            min_freq,
            max_freq,
            mains,
            mains_tol,
            min_good,
            max_divisor,
            min_group_size,
            n_harmonics,
            max_freq_tol,
            n_exclusive,
            bins[t],
        )
    return bins, counts


@dataclass
class Detections:
    """Fundamentals detected in a block of spectra."""

    frame: np.ndarray
    """Frame index (relative to the block) of each detection."""
    bin: np.ndarray
    """Frequency bin of each fundamental."""
    freq: np.ndarray
    """Fundamental frequency [Hz], refined to sub-bin precision if enabled."""


def n_harmonics(cfg: HarmonicGroupsConfig) -> int:
    """Number of harmonics collected per group (as in the original code)."""
    return max(
        cfg.min_group_size, int(cfg.max_freq * cfg.min_group_size // cfg.min_freq) - 1
    )


def detect_harmonic_groups(
    log_spec: np.ndarray,
    freqs: np.ndarray,
    cfg: HarmonicGroupsConfig,
    low_threshold: float,
    high_threshold: float,
) -> Detections:
    """Detect fish in a dB spectrogram of shape (frames, freqs)."""
    if cfg.exclusive_harmonics == "all":
        n_exclusive = n_harmonics(cfg)
    elif cfg.exclusive_harmonics == "core":
        n_exclusive = cfg.min_group_size
    else:
        raise ValueError(
            f"exclusive_harmonics must be 'core' or 'all', not {cfg.exclusive_harmonics!r}"
        )
    log_spec = np.ascontiguousarray(log_spec, dtype=np.float32)
    freqs = np.ascontiguousarray(freqs, dtype=np.float64)
    bins, counts = _detect_groups(
        log_spec,
        freqs,
        float(low_threshold),
        float(high_threshold),
        float(cfg.min_freq),
        float(cfg.max_freq),
        float(cfg.mains_freq),
        float(cfg.mains_freq_tol),
        float(cfg.min_good_peak_power),
        int(cfg.max_divisor),
        int(cfg.min_group_size),
        n_harmonics(cfg),
        float(cfg.max_freq_tol),
        n_exclusive,
        int(cfg.max_groups_per_frame),
    )
    frame = np.repeat(np.arange(len(counts)), counts)
    mask = np.arange(bins.shape[1])[None, :] < counts[:, None]
    fbin = bins[mask]
    if cfg.refine_frequency:
        freq = refine_peak_frequency(log_spec, freqs, frame, fbin)
    else:
        freq = freqs[fbin]
    # Fundamentals of groups found via a divisor may lie below min_freq.
    keep = (freq >= cfg.min_freq) & (freq <= cfg.max_freq)
    return Detections(frame[keep], fbin[keep], freq[keep])


def refine_peak_frequency(
    log_spec: np.ndarray, freqs: np.ndarray, frame: np.ndarray, fbin: np.ndarray
) -> np.ndarray:
    """Parabolic interpolation of peak positions on the dB spectrum."""
    if len(fbin) == 0:
        return freqs[fbin]
    b = np.clip(fbin, 1, log_spec.shape[1] - 2)
    a, m, c = (log_spec[frame, b + k].astype(np.float64) for k in (-1, 0, 1))
    denom = a - 2 * m + c
    with np.errstate(invalid="ignore", divide="ignore"):
        delta = np.where(np.isfinite(denom) & (denom < 0), 0.5 * (a - c) / denom, 0.0)
    delta = np.clip(np.nan_to_num(delta), -0.5, 0.5)
    df = freqs[1] - freqs[0]
    return freqs[fbin] + delta * df
