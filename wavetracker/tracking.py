"""Identity tracking of EOD frequency detections.

A numba port of ``freq_tracking_v6`` from Raab et al. (2022) that reproduces
the original assignments while scaling linearly with recording length.

Detections are linked by a combined error of (2/3) the relative difference of
their spatial amplitude patterns across electrodes and (1/3) a sigmoid of
their frequency difference. Tracking proceeds in windows of ``3 * max_dt``:
within a window, temporary identities are formed by greedily linking the
lowest-error pairs; the central third of each window is then attached to the
identities established so far. Temporary identities that attach to none
start a new identity; with ``start_salience`` only those holding enough
confident detections do (ByteTrack-style: weak detections may extend
identities, but not start them).

One bug of the original is fixed: when attaching temporary identities, it
skipped link targets whose *detection offset* within the window was smaller
than the window's frame count, so the skipped time span shrank with the
number of fish. Targets are now skipped if their *frame* lies before the
central third of the window. ``track(..., v6_compat=True)`` restores the old
behaviour; it exists so the tests can verify the rest of the port against
the original implementation.
"""

from __future__ import annotations

import logging

import numpy as np
from numba import njit, prange

from .config import TrackingConfig

log = logging.getLogger(__name__)

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
def _error_distribution_start(frame_start, start, last, comp_range):
    """First frame of the window to estimate the amplitude errors from.

    Normally the first detection. But a recording may hold a few stray detections
    long before the first fish, and errors among those say nothing. So: the first
    window (from the first detection, in steps of `comp_range`) with detections in
    at least half of its frames, or else the one with the most detections.
    """
    frames = 3 * comp_range
    best = start
    best_count = -1
    for s in range(start, last + 1, comp_range):
        occupied = 0
        for f in range(s, s + frames):
            if frame_start[f + 1] > frame_start[f]:
                occupied += 1
        if 2 * occupied >= frames:
            return s
        count = frame_start[s + frames] - frame_start[s]
        if count > best_count:
            best = s
            best_count = count
    return best


@njit(cache=True)
def _freq_error(fund, idx, a, b, slope, frame_dt, motion, horizon):
    """|f_a - f_b|, or with `motion` the mean error of predicting the later
    detection b from a along a's slope over its past (slope[0]) and a from b
    along b's slope over its future (slope[1]) [Hz/s], at most `horizon` [s]
    ahead (as far as the slopes were measured)."""
    if not motion:
        return abs(fund[a] - fund[b])
    d = fund[b] - fund[a]
    t = min(max((idx[b] - idx[a]) * frame_dt, -horizon), horizon)
    if t < 0:  # b before a: a's future, b's past
        return 0.5 * (abs(d - slope[1, a] * t) + abs(d - slope[0, b] * t))
    return 0.5 * (abs(d - slope[0, a] * t) + abs(d - slope[1, b] * t))


@njit(cache=True)
def _learned_prob(l_start, l_dst, l_prob, a, b):
    """Learned probability of the link a -> b (-1: not a learned edge)."""
    lo = l_start[a]
    hi = l_start[a + 1]
    j = lo + np.searchsorted(l_dst[lo:hi], b)
    if j < hi and l_dst[j] == b:
        return l_prob[j]
    return -1.0


@njit(cache=True)
def _connections(
    fund,
    idx,
    nsign,
    frame_start,
    f0,
    f1,
    comp_range,
    freq_tol,
    tol0,
    tol_rate,
    a_dist,
    slope,
    frame_dt,
    motion,
    horizon,
    l_start,
    l_dst,
    l_prob,
    l_frames,
    l_threshold,
):
    """Candidate links (a, b, error) for origins in frames [f0, f1), sorted by
    error; ties keep the (origin, target) order of the original error cube.

    With learned link probabilities (`l_frames` > 0: edges of a in
    l_dst[l_start[a]:l_start[a + 1]], sorted), a candidate link up to
    `l_frames` frames is kept if its probability is at least `l_threshold`,
    with error 1 - probability; longer links keep the rule-based error plus
    1, so that they come after every learned one."""
    # First pass: count, second pass: fill.
    n = 0
    for i in range(f0, f1):
        for a in range(frame_start[i], frame_start[i + 1]):
            for b in range(frame_start[i + 1], frame_start[i + comp_range + 1]):
                df = _freq_error(fund, idx, a, b, slope, frame_dt, motion, horizon)
                if df < freq_tol and df < tol0 + tol_rate * (idx[b] - idx[a]):
                    if l_frames > 0 and idx[b] - idx[a] <= l_frames:
                        if _learned_prob(l_start, l_dst, l_prob, a, b) >= l_threshold:
                            n += 1
                    else:
                        n += 1
    ca = np.empty(n, dtype=np.int64)
    cb = np.empty(n, dtype=np.int64)
    ce = np.empty(n)
    k = 0
    n_dist = a_dist.shape[0]
    for i in range(f0, f1):
        for a in range(frame_start[i], frame_start[i + 1]):
            for b in range(frame_start[i + 1], frame_start[i + comp_range + 1]):
                df = _freq_error(fund, idx, a, b, slope, frame_dt, motion, horizon)
                if df < freq_tol and df < tol0 + tol_rate * (idx[b] - idx[a]):
                    if l_frames > 0 and idx[b] - idx[a] <= l_frames:
                        pr = _learned_prob(l_start, l_dst, l_prob, a, b)
                        if pr >= l_threshold:
                            ca[k] = a
                            cb[k] = b
                            ce[k] = 1.0 - pr
                            k += 1
                        continue
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
                    if l_frames > 0:
                        ce[k] += 1.0
                    k += 1
    order = np.argsort(ce, kind="mergesort")
    return ca[order], cb[order], ce[order]


@njit(cache=True)
def _tmp_identities(
    idx, fund, ca, cb, lo, hi, n_window_frames, f_lo, slope, frame_dt, motion, horizon
):
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
            j_after = -1
            j_before = -1
            for j in range(w):
                if tmp[j] == tb:
                    if idx[lo + j] == idx[lo + a]:
                        clash = True
                        break
                    if j > a and j_after < 0:
                        j_after = j
                    if j < a:
                        j_before = j
            if clash:
                continue
            close = False
            for j in (j_after, j_before):
                if j >= 0:
                    e = _freq_error(
                        fund, idx, lo + a, lo + j, slope, frame_dt, motion, horizon
                    )
                    close = close or e <= 0.5
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
def _startable(tmp, confident, lo, start_count):
    """Which temporary identities are present in the window and hold at least
    `start_count` confident detections, i.e. may start a new identity."""
    max_tmp = -1
    for j in range(tmp.shape[0]):
        max_tmp = max(max_tmp, tmp[j])
    count = np.zeros(max_tmp + 1, dtype=np.int64)
    present = np.zeros(max_tmp + 1, dtype=np.bool_)
    for j in range(tmp.shape[0]):
        if tmp[j] >= 0:
            present[tmp[j]] = True
            if confident[lo + j]:
                count[tmp[j]] += 1
    return present & (count >= start_count)


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
    confident,
    start_count,
):
    """Attach the central part of the window's temporary identities to the
    established identities (`ident`, modified in place); the others start new
    identities if they hold `start_count` confident detections."""
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
    startable = _startable(tmp, confident, lo, start_count)
    has_ident = np.zeros(startable.shape[0], dtype=np.bool_)
    for j in range(w):
        if tmp[j] >= 0 and ident[lo + j] >= 0:
            has_ident[tmp[j]] = True
    for t in range(startable.shape[0]):
        if startable[t] and not has_ident[t]:
            for j in range(w):
                if tmp[j] == t and central(j):
                    ident[lo + j] = next_identity
            next_identity += 1
    return next_identity


@njit(cache=True)
def _track(
    fund,
    idx,
    nsign,
    comp_range,
    freq_tol,
    tol0,
    tol_rate,
    v6_compat,
    confident,
    start_count,
    slope,
    frame_dt,
    motion,
    horizon,
    l_start,
    l_dst,
    l_prob,
    l_frames,
    l_threshold,
):
    n_frames = idx[-1] + 4 * comp_range + 2
    frame_start = np.searchsorted(idx, np.arange(n_frames + 1))
    start = idx[0]
    last = idx[-1]
    a_start = (
        start
        if v6_compat
        else _error_distribution_start(frame_start, start, last, comp_range)
    )
    a_dist = _amplitude_error_distribution(
        idx, nsign, frame_start, a_start, a_start + 3 * comp_range, comp_range
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
            idx,
            nsign,
            frame_start,
            s + 1,
            s + 2 * comp_range,
            comp_range,
            freq_tol,
            tol0,
            tol_rate,
            a_dist,
            slope,
            frame_dt,
            motion,
            horizon,
            l_start,
            l_dst,
            l_prob,
            l_frames,
            l_threshold,
        )
        tmp = _tmp_identities(
            idx,
            fund,
            ca,
            cb,
            lo,
            hi,
            n_window_frames,
            s,
            slope,
            frame_dt,
            motion,
            horizon,
        )
        if s == start:
            startable = _startable(tmp, confident, lo, start_count)
            for t in range(startable.shape[0]):
                if not startable[t]:
                    continue
                for j in range(tmp.shape[0]):
                    if tmp[j] == t and idx[lo + j] <= s + comp_range:
                        ident[lo + j] = next_identity
                next_identity += 1
        ca, cb, _ = _connections(
            fund,
            idx,
            nsign,
            frame_start,
            s,
            s + comp_range,
            comp_range,
            freq_tol,
            tol0,
            tol_rate,
            a_dist,
            slope,
            frame_dt,
            motion,
            horizon,
            l_start,
            l_dst,
            l_prob,
            l_frames,
            l_threshold,
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
            confident,
            start_count,
        )
    return ident


@njit(cache=True)
def _local_slopes(fund, idx, ident, frame_dt, window, max_slope, past, future):
    """Slope [Hz/s] of a straight line through the detections of the same
    identity within `window` frames before (`past`) and/or after (`future`)
    each detection, itself included (0 without an identity or with fewer than
    3 such detections), clipped to +-max_slope. The slope is shrunk towards 0
    by its own uncertainty, s * s^2 / (s^2 + var(s)), so that the scatter of
    a flat track is not taken for a slope."""
    n = fund.shape[0]
    out = np.zeros(n)
    order = np.argsort(ident, kind="mergesort")  # by identity, then by frame
    k = 0
    while k < n:
        if ident[order[k]] < 0:
            k += 1
            continue
        e = k
        while e < n and ident[order[e]] == ident[order[k]]:
            e += 1
        lo = k
        hi = k
        for c in range(k, e):
            i = order[c]
            while idx[order[lo]] < idx[i] - (window if past else 0):
                lo += 1
            while hi < e and idx[order[hi]] <= idx[i] + (window if future else 0):
                hi += 1
            m = hi - lo
            if m < 3:
                continue
            st = 0.0
            sf = 0.0
            for q in range(lo, hi):
                st += idx[order[q]]
                sf += fund[order[q]]
            st /= m
            sf /= m
            num = 0.0
            den = 0.0
            for q in range(lo, hi):
                dt = idx[order[q]] - st
                num += dt * (fund[order[q]] - sf)
                den += dt * dt
            if den > 0:
                b = num / den  # Hz per frame
                res = 0.0
                for q in range(lo, hi):
                    r = fund[order[q]] - sf - b * (idx[order[q]] - st)
                    res += r * r
                var_b = res / max(m - 2, 1) / den
                if b * b + var_b > 0:
                    b *= b * b / (b * b + var_b)
                s = b / frame_dt
                out[i] = min(max(s, -max_slope), max_slope)
        k = e
    return out


def normalize_signatures(sign_v: np.ndarray, feature: str = "minmax") -> np.ndarray:
    """Electrode amplitude pattern of each detection.

    "minmax": powers scaled to 0..1 per detection (original). With two
    electrodes this degenerates to [0, 1] / [1, 0].
    "db": powers in dB relative to the detection's mean over electrodes; with
    two electrodes this is their level ratio, which changes smoothly when
    fish or electrodes move.
    """
    if feature == "db":
        db = 10.0 * np.log10(np.maximum(sign_v, 1e-30))
        return db - db.mean(axis=1, keepdims=True)
    if feature != "minmax":
        raise ValueError(f"amplitude_feature must be 'minmax' or 'db', not {feature!r}")
    lo = sign_v.min(axis=1, keepdims=True)
    hi = sign_v.max(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (sign_v - lo) / (hi - lo)


@njit(cache=True, parallel=True)
def _support(fund, idx, window, df):
    """Number of other detections within +-`window` frames and +-`df` Hz."""
    n = fund.shape[0]
    frame_start = np.searchsorted(idx, np.arange(idx[-1] + 2))
    out = np.zeros(n, dtype=np.int64)
    for i in prange(n):
        f0 = max(idx[i] - window, 0)
        f1 = min(idx[i] + window + 1, idx[-1] + 1)
        c = 0
        for j in range(frame_start[f0], frame_start[f1]):
            if j != i and abs(fund[j] - fund[i]) <= df:
                c += 1
        out[i] = c
    return out


def support(fund_v, idx_v, times, window: float, df: float) -> np.ndarray:
    """For each detection, the number of other detections within `window`
    seconds and `df` Hz (idx_v must be sorted)."""
    if len(fund_v) == 0:
        return np.zeros(0, dtype=np.int64)
    w = max(1, int(np.round(window / (times[1] - times[0]))))
    return _support(
        np.ascontiguousarray(fund_v, dtype=np.float64),
        np.ascontiguousarray(idx_v, dtype=np.int64),
        w,
        float(df),
    )


def _no_salience(key: str) -> str:
    return (
        f"tracking.{key} needs a salience per detection (sal_v), but these "
        "detections have none (results of an older version?): run detect again, "
        f"or set tracking.{key} to null"
    )


def drop_weak_identities(
    ident_v: np.ndarray, sal_v: np.ndarray, min_salience: float
) -> np.ndarray:
    """The identities, without those whose mean salience is below
    `min_salience` (their detections become NaN)."""
    out = np.array(ident_v, dtype=float)
    has = np.isfinite(out)
    if not has.any():
        return out
    _, inv = np.unique(out[has], return_inverse=True)
    mean = np.bincount(inv, np.asarray(sal_v, dtype=float)[has]) / np.bincount(inv)
    out[np.flatnonzero(has)[mean[inv] < min_salience]] = np.nan
    return out


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
    sal_v: np.ndarray | None = None,
    cplx_v: np.ndarray | None = None,
    trace: dict | None = None,
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
    sal_v
        Salience (0-1) of every detection; needed with
        ``cfg.start_salience``, which lets only temporary identities with
        enough confident detections start a new identity (ByteTrack-style).
        Weak detections still join identities that exist.
    cplx_v
        Complex spectra of every detection (n, channels); the learned
        association (``cfg.method == "learned"``) takes the relative signs of
        the electrodes from them (without: all in phase).
    trace
        Optional, for `wavetracker.review`: the learned association
        (``assoc_mode: link``) fills it with ``link_p`` and ``alt_p`` per
        detection (see `assocnet.link_evidence`; NaN and 0 where not
        tracked). The result does not depend on it.

    Returns
    -------
    ident_v
        Identity of every detection (float, NaN if unassigned).
    """
    if cfg.start_salience is not None and sal_v is None:
        raise ValueError(_no_salience("start_salience"))
    ident_v = np.full(len(fund_v), np.nan)
    valid = (fund_v >= min_freq) & (fund_v <= max_freq)
    if valid.sum() < 2 or len(times) < 2:
        return ident_v
    if np.any(np.diff(idx_v) < 0):
        raise ValueError("idx_v must be sorted")
    if cfg.method == "learned" and cfg.assoc_mode == "link":
        return _track_learned(
            fund_v, idx_v, sign_v, cplx_v, sal_v, times, cfg, valid, trace
        )
    if cfg.min_support > 0:  # isolated detections are not tracked
        valid &= (
            support(fund_v, idx_v, times, cfg.support_window, cfg.support_freq)
            >= cfg.min_support
        )
        if valid.sum() < 2:
            return ident_v
    dt = times[1] - times[0]
    comp_range = int(np.floor(cfg.max_dt / dt))
    if comp_range < 1:
        raise ValueError("max_dt must be larger than the frame interval")
    if cfg.start_salience is None:  # every temporary identity may start one
        confident = np.ones(valid.sum(), dtype=np.bool_)
        start_count = 0
    else:
        confident = np.asarray(sal_v)[valid] >= cfg.start_salience
        start_count = max(1, int(np.round(cfg.start_evidence / dt)))
    fund = np.ascontiguousarray(fund_v[valid], dtype=np.float64)
    idx = np.ascontiguousarray(idx_v[valid], dtype=np.int64)
    args = (
        fund,
        idx,
        np.ascontiguousarray(
            normalize_signatures(sign_v[valid], cfg.amplitude_feature), dtype=np.float64
        ),
        comp_range,
        float(cfg.freq_tolerance),
        np.inf if cfg.gap_tolerance is None else float(cfg.gap_tolerance),
        float(cfg.gap_tolerance_rate) * dt,
        v6_compat,
        confident,
        start_count,
    )
    if cfg.method == "learned":  # assoc_mode "cost": learned errors of short links
        learned = _learned_links(
            fund, idx, sign_v[valid], cplx_v, sal_v, valid, times, cfg
        )
    else:
        learned = (np.zeros(1, np.int64), np.zeros(0, np.int64), np.zeros(0), 0, 0.0)
    args = (*args, *learned)
    ident = _track(
        *args[:10], np.zeros((2, len(fund))), float(dt), False, 0.0, *args[10:]
    )
    if cfg.motion_window is not None:  # second pass along the slopes of the first
        w = max(1, int(np.round(cfg.motion_window / dt)))
        m = float(cfg.motion_max_slope)
        slope = np.stack(
            [
                _local_slopes(fund, idx, ident, float(dt), w, m, True, False),
                _local_slopes(fund, idx, ident, float(dt), w, m, False, True),
            ]
        )
        ident = _track(
            *args[:10], slope, float(dt), True, float(cfg.motion_window), *args[10:]
        )
    ident_v[valid] = np.where(ident >= 0, ident, np.nan)
    return ident_v


def _on_device(fn, device: str):
    """fn(device) on `device` ("auto": GPU first); on the CPU if the GPU runs
    out of memory."""
    import torch

    from .spectrogram import free_gpu_memory, get_device, out_of_memory

    dev = get_device(device)
    try:
        return fn(str(dev))
    except (RuntimeError, torch.OutOfMemoryError) as err:
        if dev.type == "cpu" or not out_of_memory(err):
            raise
        log.warning("%s ran out of memory; learned association on the CPU instead", dev)
        log.debug("%s", err)
    free_gpu_memory()
    return fn("cpu")


def _track_learned(fund_v, idx_v, sign_v, cplx_v, sal_v, times, cfg, valid, trace=None):
    """`track` with the learned association (wavetracker.assocnet)."""
    from . import assocnet

    sel = np.flatnonzero(valid)
    sub = {} if trace is not None else None

    def run(device):
        if sub is not None:
            sub.clear()
        return assocnet.track_learned(
            fund_v[sel],
            idx_v[sel],
            sign_v[sel],
            None if cplx_v is None else cplx_v[sel],
            None if sal_v is None else sal_v[sel],
            times,
            assocnet.load_model(cfg.assoc_model, device),
            threshold=cfg.assoc_threshold,
            min_detections=cfg.assoc_min_detections,
            device=device,
            trace=sub,
        )

    ident = _on_device(run, cfg.assoc_device)
    ident_v = np.full(len(fund_v), np.nan)
    ident_v[sel] = ident
    if trace is not None:
        trace["link_p"] = np.full(len(fund_v), np.nan)
        trace["alt_p"] = np.zeros(len(fund_v))
        trace["link_p"][sel] = sub["link_p"]
        trace["alt_p"][sel] = sub["alt_p"]
    return ident_v


def _learned_links(fund, idx, sign, cplx_v, sal_v, valid, times, cfg):
    """Learned link probabilities of the (valid) detections, as the arrays
    `_connections` takes: (edge start per detection, targets, probabilities,
    frames, threshold)."""
    from . import assocnet

    frame_dt = float(times[1] - times[0])

    def run(device):
        model = assocnet.load_model(cfg.assoc_model, device)
        k = max(1, round(model.max_dt / frame_dt))
        src, dst = assocnet.candidate_edges(idx, fund, k, model.max_df)
        d = assocnet.detection_inputs(
            fund,
            idx,
            times,
            sign,
            None if cplx_v is None else cplx_v[valid],
            None if sal_v is None else sal_v[valid],
            device,
        )
        prob = assocnet.edge_probabilities(model, d, idx, src, dst, frame_dt)
        return k, src, dst, prob

    k, src, dst, prob = _on_device(run, cfg.assoc_device)
    start = np.searchsorted(src, np.arange(len(fund) + 1)).astype(np.int64)
    return (
        start,
        np.ascontiguousarray(dst, dtype=np.int64),
        prob.astype(np.float64),
        int(k),
        float(cfg.assoc_threshold),
    )
