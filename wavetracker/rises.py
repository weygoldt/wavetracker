"""Joining identities across rise onsets.

At a rise a fish's frequency jumps up by a few to about 60 Hz within about a
second and then decays back over 10-60 s. The jump is far beyond the
tracker's tolerances and the linker's candidates (``linking.max_df``), and
the analysis window often misses the onset frames, so the fish goes on as a
new identity, or the linker leaves it on two identities that take turns at
its rises.

After linking (or stitching), ``join_rises`` pairs every *stop* of an
identity (no detection of it for ``stop_gap`` s after) with the *starts* of
other identities (no detection for ``stop_gap`` s before) that follow
within ``max_gap`` s (or begin up to ``max_overlap`` s earlier) between
``min_jump`` and ``max_jump`` Hz higher. Such a pair is one fish if the
electrode patterns before the stop and after the start agree
(``max_pattern_rms``, ``min_pattern_corr``): a fish does not move far within
a rise, while another fish that happens to start there sits elsewhere on the
grid. Near the noise the patterns are noise and two faint fish look alike,
so both sides need a mean detection salience of ``min_salience``. Pairs are
joined best first (smallest pattern difference), unless the two identities
would share frames (``max_shared``).

The learned detector often reports the old frequency for a few seconds
after the onset (and the new one a little before it). When such a pair is
joined, the earlier identity's detections after the later one's start are
dropped, since a fish is at one frequency.

The decay that follows a rise is not required: on the dense benchmark many
rise pieces end before they decay, or the next rise starts first, and
requiring a falling slope lost more true joins than false ones.
"""

from __future__ import annotations

import itertools

import numpy as np

from .config import RisesConfig
from .linking import _pattern, _pattern_distance
from .tracking import _no_salience


def _edges(t: np.ndarray, gap: float):
    """Positions in t (sorted) where segments separated by more than `gap`
    start and stop."""
    brk = np.flatnonzero(np.diff(t) > gap)
    return np.r_[0, brk + 1], np.r_[brk, len(t) - 1]


def candidates(fund_v, sign_v, t_det, members, cfg: RisesConfig, sal_v=None):
    """Rise pairs (pattern rms, earlier identity, later identity, the later
    one's start, the earlier one's stop [s]) by position in `members`, the
    detection indices of every identity sorted by time."""
    stops, starts = [], []

    def weak(m):
        return cfg.min_salience is not None and sal_v[m].mean() < cfg.min_salience

    for p, m in enumerate(members):
        t, f = t_det[m], fund_v[m]
        first, last = _edges(t, cfg.stop_gap)
        for s in first:
            if t[-1] - t[s] < cfg.min_duration:
                continue
            edge = (t >= t[s]) & (t <= t[s] + cfg.edge_window)
            pat = (t >= t[s]) & (t <= t[s] + cfg.pattern_window)
            if weak(m[pat]):
                continue
            starts.append((t[s], float(np.median(f[edge])), p, m[pat]))
        for e in last:
            edge = (t >= t[e] - cfg.edge_window) & (t <= t[e])
            pat = (t >= t[e] - cfg.pattern_window) & (t <= t[e])
            if weak(m[pat]):
                continue
            stops.append((t[e], float(np.median(f[edge])), p, m[pat]))
    if not stops or not starts:
        return []
    starts.sort(key=lambda s: s[0])
    t_start = np.array([s[0] for s in starts])
    pairs = []
    for t_e, f_e, p_e, m_e in stops:
        lo = np.searchsorted(t_start, t_e - cfg.max_overlap, "left")
        hi = np.searchsorted(t_start, t_e + cfg.max_gap, "right")
        pat_e = None
        for t_s, f_s, p_s, m_s in starts[lo:hi]:
            jump = f_s - f_e
            if p_s == p_e or not cfg.min_jump <= jump <= cfg.max_jump:
                continue
            if pat_e is None:
                pat_e = _pattern(sign_v[m_e])[0]
            rms, corr = _pattern_distance(pat_e, _pattern(sign_v[m_s])[0])
            if rms <= cfg.max_pattern_rms and corr >= cfg.min_pattern_corr:
                pairs.append((rms, p_e, p_s, t_s, t_e))
    pairs.sort(key=lambda x: x[:3])
    return pairs


def join_rises(
    fund_v: np.ndarray,
    idx_v: np.ndarray,
    sign_v: np.ndarray,
    ident_v: np.ndarray,
    times: np.ndarray,
    cfg: RisesConfig,
    sal_v: np.ndarray | None = None,
) -> np.ndarray:
    """Identities with the two sides of rise onsets joined (see module doc).
    `sal_v` (salience per detection) is needed for ``cfg.min_salience``."""
    from .stitching import resolve_duplicates

    if not cfg.enabled or not np.isfinite(ident_v).any():
        return ident_v.copy()
    if cfg.min_salience is not None and sal_v is None:
        raise ValueError(_no_salience("min_salience").replace("tracking.", "rises."))
    t_det = times[idx_v]
    valid = np.flatnonzero(np.isfinite(ident_v))
    order = valid[np.lexsort((t_det[valid], ident_v[valid]))]
    ids, first = np.unique(ident_v[order], return_index=True)
    bounds = np.append(first, len(order))
    members = [order[a:b] for a, b in itertools.pairwise(bounds)]
    pairs = candidates(fund_v, sign_v, t_det, members, cfg, sal_v)
    if not pairs:
        return ident_v.copy()

    parent = np.arange(len(ids))
    frames = [np.unique(idx_v[m]) for m in members]

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    drop = []
    for _, a, b, t_s, t_e in pairs:
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        # the earlier identity's tail after the later one starts (a ghost)
        m = members[a]
        tail = m[(t_det[m] >= t_s) & (t_det[m] <= t_e)]
        fa = np.setdiff1d(frames[ra], idx_v[tail], assume_unique=True)
        fb = frames[rb]
        shared = len(np.intersect1d(fa, fb, assume_unique=True))
        if shared > cfg.max_shared * min(len(fa), len(fb)):
            continue
        drop.append(tail)
        if len(fa) < len(fb) or (len(fa) == len(fb) and rb < ra):
            ra, rb, fa, fb = rb, ra, fb, fa
        parent[rb] = ra
        frames[ra] = np.union1d(fa, fb)
        frames[rb] = None
    out = ident_v.copy()
    for p, m in enumerate(members):
        r = find(p)
        if r != p:
            out[m] = ids[r]
    for tail in drop:
        out[tail] = np.nan
    return resolve_duplicates(fund_v, idx_v, out)
