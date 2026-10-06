"""Joining track fragments across rises and short dropouts.

The tracker links detections only if their frequencies differ by less than
``freq_tolerance`` within ``max_dt``. A rise (a fast frequency excursion of
up to tens of Hz that decays back over seconds to minutes) breaks a track:
the onset jump exceeds the tolerance, and by the time the frequency has
decayed back the gap exceeds ``max_dt``.

Fragment A is joined to a later fragment B if

* B starts at most ``max_gap`` after A ends, or overlaps it by at most
  ``max_overlap`` while running at the same frequency (the detector
  occasionally reports a fish twice),
* the jump from A's last to B's first frequency is rise-shaped: between
  ``-max_drop`` and ``+max_rise``,
* their *baselines* agree within ``baseline_tolerance``: the low quantile of
  A's frequencies over its last ``baseline_window`` and of B's over its first
  ``2 * baseline_window`` (rises only go up, so a low quantile ignores them).
  At a clear rise onset (gap <= ``rise_max_gap``, upward jump >=
  ``rise_min_jump``) the looser ``rise_baseline_tolerance`` applies,
* optionally, their spatial amplitude patterns across electrodes are
  similar (``max_pattern_distance``).

Longer dropouts (up to ``max_dropout``) are bridged only with a tight
baseline match (``dropout_tolerance``) and if no other track occupies that
frequency during the gap. Candidate joins are accepted greedily, best first,
each fragment end and start used at most once. Finally, frames with two
detections of one identity keep the one closer to the local track frequency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import StitchingConfig


@dataclass
class Fragment:
    ident: float
    start: float
    end: float
    f_start: float
    f_end: float
    base_start: float
    base_end: float
    pattern_start: np.ndarray
    pattern_end: np.ndarray
    n: int


def _pattern(sign: np.ndarray) -> np.ndarray:
    """Mean electrode power pattern, scaled to unit maximum."""
    p = sign.mean(0)
    m = p.max()
    return p / m if m > 0 else p


def fragments(
    fund_v, idx_v, sign_v, ident_v, times, cfg: StitchingConfig
) -> list[Fragment]:
    valid = ~np.isnan(ident_v)
    order = np.argsort(ident_v[valid], kind="stable")
    sel = np.nonzero(valid)[0][order]
    ids, starts = np.unique(ident_v[sel], return_index=True)
    out = []
    for i, j0, j1 in zip(ids, starts, np.append(starts[1:], len(sel)), strict=True):
        k = sel[j0:j1]
        if len(k) < cfg.min_detections:
            continue
        t, f, s = times[idx_v[k]], fund_v[k], sign_v[k]
        a = t <= t[0] + 2 * cfg.baseline_window
        b = t >= t[-1] - cfg.baseline_window
        pa = t <= t[0] + cfg.pattern_window
        pb = t >= t[-1] - cfg.pattern_window
        n_edge = min(len(k), 5)
        out.append(
            Fragment(
                ident=float(i),
                start=float(t[0]),
                end=float(t[-1]),
                f_start=float(np.median(f[:n_edge])),
                f_end=float(np.median(f[-n_edge:])),
                base_start=float(np.quantile(f[a], cfg.baseline_quantile)),
                base_end=float(np.quantile(f[b], cfg.baseline_quantile)),
                pattern_start=_pattern(s[pa]),
                pattern_end=_pattern(s[pb]),
                n=len(k),
            )
        )
    return out


class _Detections:
    """Assigned detections sorted by time, for window queries."""

    def __init__(self, t, f, ident):
        order = np.argsort(t, kind="stable")
        self.t, self.f, self.ident = t[order], f[order], ident[order]

    def window(self, t0, t1):
        lo = np.searchsorted(self.t, t0, "left")
        hi = np.searchsorted(self.t, t1, "right")
        return self.f[lo:hi], self.ident[lo:hi]


def candidate_joins(
    frags: list[Fragment], dets: _Detections, cfg: StitchingConfig, n_channels: int
):
    """All admissible (cost, A index, B index) pairs, cheapest first."""
    starts = np.array([f.start for f in frags])
    order = np.argsort(starts)
    sorted_starts = starts[order]
    pairs = []
    for ia, a in enumerate(frags):
        lo = np.searchsorted(sorted_starts, a.end - cfg.max_overlap, "left")
        hi = np.searchsorted(sorted_starts, a.end + cfg.max_dropout, "right")
        for ib in order[lo:hi]:
            b = frags[ib]
            if ib == ia or b.start <= a.start or b.end <= a.end:
                continue
            gap = b.start - a.end
            jump = b.f_start - a.f_end
            # (for overlapping fragments the overlap check below replaces this)
            if gap >= 0 and not -cfg.max_drop <= jump <= cfg.max_rise:
                continue
            dbase = abs(b.base_start - a.base_end)
            if gap > cfg.max_gap:  # dropout: tight match, nobody else there
                tol = cfg.dropout_tolerance
                if dbase > tol:
                    continue
                f, ident = dets.window(a.end, b.start)
                other = (ident != a.ident) & (ident != b.ident)
                near = np.abs(f[other] - 0.5 * (a.base_end + b.base_start))
                if np.sum(near <= cfg.baseline_tolerance) >= cfg.min_detections:
                    continue
            else:
                rise = gap <= cfg.rise_max_gap and jump >= cfg.rise_min_jump
                tol = cfg.rise_baseline_tolerance if rise else cfg.baseline_tolerance
                if dbase > tol:
                    continue
                if gap < 0:  # overlap: both must run at the same frequency
                    f, ident = dets.window(b.start, a.end)
                    fa, fb = f[ident == a.ident], f[ident == b.ident]
                    differ = (
                        len(fa)
                        and len(fb)
                        and abs(np.median(fa) - np.median(fb)) > cfg.overlap_tolerance
                    )
                    if differ:
                        continue
            cost = dbase / tol + max(gap, 0.0) / cfg.max_gap
            if cfg.max_pattern_distance is not None:
                dpat = np.linalg.norm(b.pattern_start - a.pattern_end)
                dpat /= np.sqrt(n_channels)
                if dpat > cfg.max_pattern_distance:
                    continue
                cost += dpat / cfg.max_pattern_distance
            pairs.append((cost, ia, int(ib)))
    pairs.sort()
    return pairs


def resolve_duplicates(fund_v, idx_v, ident_v, window: int = 30) -> np.ndarray:
    """Keep one detection per identity and frame: the one closest to the
    identity's median frequency in the surrounding `window` frames."""
    out = ident_v.copy()
    valid = np.nonzero(~np.isnan(out))[0]
    order = valid[np.lexsort((idx_v[valid], out[valid]))]
    key_id, key_idx = out[order], idx_v[order]
    dup = (np.diff(key_id) == 0) & (np.diff(key_idx) == 0)
    if not dup.any():
        return out
    starts = np.nonzero(dup & ~np.r_[False, dup[:-1]])[0]
    for s0 in starts:
        s1 = s0 + 1
        while s1 < len(dup) and dup[s1]:
            s1 += 1
        group = order[s0 : s1 + 1]
        i, frame = key_id[s0], key_idx[s0]
        m = (key_id == i) & (np.abs(key_idx - frame) <= window) & (key_idx != frame)
        ref = np.median(fund_v[order[m]]) if m.any() else np.median(fund_v[group])
        keep = group[np.argmin(np.abs(fund_v[group] - ref))]
        out[group[group != keep]] = np.nan
    return out


def stitch(
    fund_v: np.ndarray,
    idx_v: np.ndarray,
    sign_v: np.ndarray,
    ident_v: np.ndarray,
    times: np.ndarray,
    cfg: StitchingConfig,
) -> np.ndarray:
    """Return identities with fragments joined across rises and dropouts."""
    if not cfg.enabled or np.all(np.isnan(ident_v)):
        return ident_v.copy()
    frags = fragments(fund_v, idx_v, sign_v, ident_v, times, cfg)
    valid = ~np.isnan(ident_v)
    dets = _Detections(times[idx_v[valid]], fund_v[valid], ident_v[valid])
    pairs = candidate_joins(frags, dets, cfg, sign_v.shape[1])

    nxt = {}
    prv = {}
    for _, ia, ib in pairs:
        if ia in nxt or ib in prv:
            continue
        nxt[ia] = ib
        prv[ib] = ia

    out = ident_v.copy()
    for head in range(len(frags)):
        if head in prv:
            continue
        j = head
        while j in nxt:
            j = nxt[j]
            out[ident_v == frags[j].ident] = frags[head].ident
    return resolve_duplicates(fund_v, idx_v, out)
