"""Identity tracking of EOD frequency detections.

A numba port of ``freq_tracking_v6`` from Raab et al. (2022) that reproduces
the original assignments while scaling linearly with recording length.

Detections are linked by a combined error of (2/3) the relative difference of
their spatial amplitude patterns across electrodes and (1/3) a sigmoid of
their frequency difference. Tracking proceeds in windows of ``3 * max_dt``:
within a window, temporary identities are formed by greedily linking the
lowest-error pairs; the central third of each window is then attached to the
identities established so far.

One bug of the original is fixed: when attaching temporary identities, it
skipped link targets whose *detection offset* within the window was smaller
than the window's frame count, so the skipped time span shrank with the
number of fish. Targets are now skipped if their *frame* lies before the
central third of the window. ``track(..., v6_compat=True)`` restores the old
behaviour; it exists so the tests can verify the rest of the port against
the original implementation.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .config import TrackingConfig

A_WEIGHT = 2.0 / 3.0
F_WEIGHT = 1.0 / 3.0


@njit(cache=True)
def _boltzmann(x, x0=0.35, dx=0.08):
    return 1.0 / (1.0 + np.exp(-(x - x0) / dx))


@njit(cache=True)
def _amp_dist(nsign, a, b):
    s = 0.0
    for k in range(nsign.shape[1]):
        d = nsign[a, k] - nsign[b, k]
        s += d * d
    return np.sqrt(s)


@njit(cache=True)
def _amplitude_error_distribution(idx, nsign, frame_start, f0, f1, comp_range):
    """All amplitude errors between detections at frames [f0, f1) and their
    potential partners up to `comp_range` frames later."""
    n = 0
    for i in range(f0, f1):
        n += (frame_start[i + 1] - frame_start[i]) * (
            frame_start[i + comp_range + 1] - frame_start[i + 1]
        )
    out = np.empty(n)
    k = 0
    for i in range(f0, f1):
        for a in range(frame_start[i], frame_start[i + 1]):
            for b in range(frame_start[i + 1], frame_start[i + comp_range + 1]):
                out[k] = _amp_dist(nsign, a, b)
                k += 1
    return np.sort(out)


@njit(cache=True)
def _connections(fund, nsign, frame_start, f0, f1, comp_range, freq_tol, a_dist):
    """Candidate links (a, b, error) for origins in frames [f0, f1), sorted by
    error; ties keep the (origin, target) order of the original error cube."""
    # First pass: count, second pass: fill.
    n = 0
    for i in range(f0, f1):
        for a in range(frame_start[i], frame_start[i + 1]):
            for b in range(frame_start[i + 1], frame_start[i + comp_range + 1]):
                if abs(fund[a] - fund[b]) < freq_tol:
                    n += 1
    ca = np.empty(n, dtype=np.int64)
    cb = np.empty(n, dtype=np.int64)
    ce = np.empty(n)
    k = 0
    n_dist = a_dist.shape[0]
    for i in range(f0, f1):
        for a in range(frame_start[i], frame_start[i + 1]):
            for b in range(frame_start[i + 1], frame_start[i + comp_range + 1]):
                df = abs(fund[a] - fund[b])
                if df < freq_tol:
                    if n_dist > 0:
                        rel_a = (
                            A_WEIGHT
                            * np.searchsorted(a_dist, _amp_dist(nsign, a, b))
                            / n_dist
                        )
                    else:
                        rel_a = 1.0
                    ca[k] = a
                    cb[k] = b
                    ce[k] = rel_a + F_WEIGHT * _boltzmann(df)
                    k += 1
    order = np.argsort(ce, kind="mergesort")
    return ca[order], cb[order], ce[order]


@njit(cache=True)
def _tmp_identities(idx, fund, ca, cb, lo, hi, n_window_frames, f_lo):
    """Greedily link detections in [lo, hi] into temporary identities."""
    w = hi - lo + 1
    tmp = np.full(w, -1, dtype=np.int64)
    next_tmp = 0
    marks = np.zeros(n_window_frames, dtype=np.int64)
    stamp = 0
    for k in range(ca.shape[0]):
        a = ca[k] - lo
        b = cb[k] - lo
        ta = tmp[a]
        tb = tmp[b]
        if ta < 0:
            if tb < 0:
                tmp[a] = next_tmp
                tmp[b] = next_tmp
                next_tmp += 1
                continue
            # attach a to b's trace if a's frame is free and freqs are close
            clash = False
            f_after = np.nan
            f_before = np.nan
            for j in range(w):
                if tmp[j] == tb:
                    if idx[lo + j] == idx[lo + a]:
                        clash = True
                        break
                    if j > a and np.isnan(f_after):
                        f_after = fund[lo + j]
                    if j < a:
                        f_before = fund[lo + j]
            if clash:
                continue
            fa = fund[lo + a]
            close = (not np.isnan(f_after) and abs(f_after - fa) <= 0.5) or (
                not np.isnan(f_before) and abs(f_before - fa) <= 0.5
            )
            if not close:
                continue
            tmp[a] = tb
        elif tb < 0:
            clash = False
            for j in range(w):
                if tmp[j] == ta and idx[lo + j] == idx[lo + b]:
                    clash = True
                    break
            if clash:
                continue
            tmp[b] = ta
        else:
            if ta == tb:
                continue
            # merge traces only if they never occupy the same frame
            stamp += 1
            clash = False
            for j in range(w):
                if tmp[j] == ta:
                    marks[idx[lo + j] - f_lo] = stamp
            for j in range(w):
                if tmp[j] == tb and marks[idx[lo + j] - f_lo] == stamp:
                    clash = True
                    break
            if clash:
                continue
            for j in range(w):
                if tmp[j] == ta:
                    tmp[j] = tb
    return tmp


@njit(cache=True)
def _assign(
    ident,
    idx,
    tmp,
    ca,
    cb,
    lo,
    hi,
    s,
    comp_range,
    next_identity,
    n_window_frames,
    f_lo,
    v6_compat,
):
    """Attach the central part of the window's temporary identities to the
    established identities (`ident`, modified in place)."""
    w = hi - lo + 1
    c0 = s + comp_range
    c1 = s + 2 * comp_range

    def central(j):
        return idx[lo + j] > c0 and idx[lo + j] <= c1

    open_count = 0
    for j in range(w):
        if tmp[j] >= 0 and central(j) and ident[lo + j] < 0:
            open_count += 1

    marks = np.zeros(n_window_frames, dtype=np.int64)
    stamp = 0
    taken = np.zeros(w, dtype=np.bool_)
    for k in range(ca.shape[0]):
        if open_count == 0:
            break
        a = ca[k] - lo
        b = cb[k] - lo
        if ident[lo + b] >= 0 or tmp[b] < 0:
            continue
        if v6_compat:
            if b < comp_range:  # original bug: detection offset vs. frame count
                continue
        elif idx[lo + b] <= c0:  # target before the central third
            continue
        ia = ident[lo + a]
        if ia < 0:
            continue
        tb = tmp[b]
        stamp += 1
        clash = False
        for j in range(w):
            if ident[lo + j] == ia and central(j):
                f = idx[lo + j] - f_lo
                if marks[f] == stamp:
                    clash = True
                marks[f] = stamp
        if not clash:
            for j in range(w):
                if tmp[j] == tb and ident[lo + j] < 0 and central(j):
                    f = idx[lo + j] - f_lo
                    if marks[f] == stamp:
                        clash = True
                        break
                    marks[f] = stamp
        if clash or taken[b]:
            continue
        taken[b] = True
        for j in range(w):
            if tmp[j] == tb and ident[lo + j] < 0 and central(j):
                ident[lo + j] = ia
                open_count -= 1

    # Temporary identities without any established identity start a new one.
    max_tmp = -1
    for j in range(w):
        max_tmp = max(max_tmp, tmp[j])
    has_ident = np.zeros(max_tmp + 1, dtype=np.bool_)
    present = np.zeros(max_tmp + 1, dtype=np.bool_)
    for j in range(w):
        if tmp[j] >= 0:
            present[tmp[j]] = True
            if ident[lo + j] >= 0:
                has_ident[tmp[j]] = True
    for t in range(max_tmp + 1):
        if present[t] and not has_ident[t]:
            for j in range(w):
                if tmp[j] == t and central(j):
                    ident[lo + j] = next_identity
            next_identity += 1
    return next_identity


@njit(cache=True)
def _track(fund, idx, nsign, comp_range, freq_tol, v6_compat):
    n_frames = idx[-1] + 4 * comp_range + 2
    frame_start = np.searchsorted(idx, np.arange(n_frames + 1))
    start = idx[0]
    last = idx[-1]
    a_dist = _amplitude_error_distribution(
        idx, nsign, frame_start, start, start + 3 * comp_range, comp_range
    )
    ident = np.full(fund.shape[0], -1, dtype=np.int64)
    next_identity = 0
    n_window_frames = 3 * comp_range + 1
    for s in range(start, last + 1, comp_range):
        lo = frame_start[s]
        hi = frame_start[s + 3 * comp_range] - 1
        # skip windows without origins or without targets
        if frame_start[s + 2 * comp_range] == lo or hi < frame_start[s + 1]:
            continue
        ca, cb, _ = _connections(
            fund,
            nsign,
            frame_start,
            s + 1,
            s + 2 * comp_range,
            comp_range,
            freq_tol,
            a_dist,
        )
        tmp = _tmp_identities(idx, fund, ca, cb, lo, hi, n_window_frames, s)
        if s == start:
            max_tmp = -1
            for j in range(tmp.shape[0]):
                max_tmp = max(max_tmp, tmp[j])
            present = np.zeros(max_tmp + 1, dtype=np.bool_)
            for j in range(tmp.shape[0]):
                if tmp[j] >= 0:
                    present[tmp[j]] = True
            for t in range(max_tmp + 1):
                if not present[t]:
                    continue
                for j in range(tmp.shape[0]):
                    if tmp[j] == t and idx[lo + j] <= s + comp_range:
                        ident[lo + j] = next_identity
                next_identity += 1
        ca, cb, _ = _connections(
            fund, nsign, frame_start, s, s + comp_range, comp_range, freq_tol, a_dist
        )
        next_identity = _assign(
            ident,
            idx,
            tmp,
            ca,
            cb,
            lo,
            hi,
            s,
            comp_range,
            next_identity,
            n_window_frames,
            s,
            v6_compat,
        )
    return ident


def normalize_signatures(sign_v: np.ndarray) -> np.ndarray:
    """Scale each detection's electrode powers to the range 0..1."""
    lo = sign_v.min(axis=1, keepdims=True)
    hi = sign_v.max(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (sign_v - lo) / (hi - lo)


def track(
    fund_v: np.ndarray,
    idx_v: np.ndarray,
    sign_v: np.ndarray,
    times: np.ndarray,
    cfg: TrackingConfig,
    min_freq: float = -np.inf,
    max_freq: float = np.inf,
    *,
    v6_compat: bool = False,
) -> np.ndarray:
    """Assign identities to detections.

    Parameters
    ----------
    fund_v, idx_v, sign_v
        Fundamental frequency, frame index (non-decreasing) and per-electrode
        power of every detection.
    times
        Time of each frame [s].
    min_freq, max_freq
        Detections outside this range are not tracked.
    v6_compat
        Reproduce the original ``freq_tracking_v6`` exactly, including its
        window-check bug (for verification only).

    Returns
    -------
    ident_v
        Identity of every detection (float, NaN if unassigned).
    """
    ident_v = np.full(len(fund_v), np.nan)
    valid = (fund_v >= min_freq) & (fund_v <= max_freq)
    if valid.sum() < 2 or len(times) < 2:
        return ident_v
    if np.any(np.diff(idx_v) < 0):
        raise ValueError("idx_v must be sorted")
    comp_range = int(np.floor(cfg.max_dt / (times[1] - times[0])))
    if comp_range < 1:
        raise ValueError("max_dt must be larger than the frame interval")
    ident = _track(
        np.ascontiguousarray(fund_v[valid], dtype=np.float64),
        np.ascontiguousarray(idx_v[valid], dtype=np.int64),
        np.ascontiguousarray(normalize_signatures(sign_v[valid]), dtype=np.float64),
        comp_range,
        float(cfg.freq_tolerance),
        v6_compat,
    )
    ident_v[valid] = np.where(ident >= 0, ident, np.nan)
    return ident_v
