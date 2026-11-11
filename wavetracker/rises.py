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
grid. Pairs are joined best first (smallest pattern difference), unless the
two identities would share frames (``max_shared``).

The decay that follows a rise is not required: on the dense benchmark many
rise pieces end before they decay, or the next rise starts first, and
requiring a falling slope lost more true joins than false ones.
"""

from __future__ import annotations

import numpy as np

from .config import RisesConfig
from .linking import _pattern, _pattern_distance


def _edges(t: np.ndarray, gap: float):
    """Positions in t (sorted) where segments separated by more than `gap`
    start and stop."""
    brk = np.flatnonzero(np.diff(t) > gap)
    return np.r_[0, brk + 1], np.r_[brk, len(t) - 1]


def candidates(fund_v, sign_v, t_det, members, cfg: RisesConfig):
    """Rise pairs (pattern rms, earlier identity, later identity) by position
    in `members`, the detection indices of every identity sorted by time."""
    stops, starts = [], []
    for p, m in enumerate(members):
        t, f = t_det[m], fund_v[m]
        first, last = _edges(t, cfg.stop_gap)
        for s in first:
            if t[-1] - t[s] < cfg.min_duration:
                continue
            edge = (t >= t[s]) & (t <= t[s] + cfg.edge_window)
            pat = (t >= t[s]) & (t <= t[s] + cfg.pattern_window)
            starts.append((t[s], float(np.median(f[edge])), p, m[pat]))
        for e in last:
            edge = (t >= t[e] - cfg.edge_window) & (t <= t[e])
            pat = (t >= t[e] - cfg.pattern_window) & (t <= t[e])
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
        for _, f_s, p_s, m_s in starts[lo:hi]:
            jump = f_s - f_e
            if p_s == p_e or not cfg.min_jump <= jump <= cfg.max_jump:
                continue
            if pat_e is None:
                pat_e = _pattern(sign_v[m_e])[0]
            rms, corr = _pattern_distance(pat_e, _pattern(sign_v[m_s])[0])
            if rms <= cfg.max_pattern_rms and corr >= cfg.min_pattern_corr:
                pairs.append((rms, p_e, p_s))
    pairs.sort(key=lambda x: (x[0], x[1], x[2]))
    return pairs


def join_rises(
    fund_v: np.ndarray,
    idx_v: np.ndarray,
    sign_v: np.ndarray,
    ident_v: np.ndarray,
    times: np.ndarray,
    cfg: RisesConfig,
) -> np.ndarray:
    """Identities with the two sides of rise onsets joined (see module doc)."""
    from .stitching import resolve_duplicates

    if not cfg.enabled or not np.isfinite(ident_v).any():
        return ident_v.copy()
    t_det = times[idx_v]
    valid = np.flatnonzero(np.isfinite(ident_v))
    order = valid[np.lexsort((t_det[valid], ident_v[valid]))]
    ids, first = np.unique(ident_v[order], return_index=True)
    bounds = np.append(first, len(order))
    members = [order[a:b] for a, b in zip(bounds[:-1], bounds[1:], strict=True)]
    pairs = candidates(fund_v, sign_v, t_det, members, cfg)
    if not pairs:
        return ident_v.copy()

    parent = np.arange(len(ids))
    frames = [np.unique(idx_v[m]) for m in members]

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for _, a, b in pairs:
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        fa, fb = frames[ra], frames[rb]
        shared = len(np.intersect1d(fa, fb, assume_unique=True))
        if shared > cfg.max_shared * min(len(fa), len(fb)):
            continue
        if len(fa) < len(fb) or (len(fa) == len(fb) and rb < ra):
            ra, rb = rb, ra
        parent[rb] = ra
        frames[ra] = np.union1d(fa, fb)
        frames[rb] = None
    out = ident_v.copy()
    for p, m in enumerate(members):
        r = find(p)
        if r != p:
            out[m] = ids[r]
    return resolve_duplicates(fund_v, idx_v, out)
