"""Two fish in one spectral peak: pair episodes decided as a unit.

Two fundamentals closer than about 1.2 Hz make one peak in the spectrogram.
A detector that finds both (a pair head) reports two detections of that
peak with the same electrode spectra, so the tracker's link error, two thirds
of which is the difference of the electrode patterns, cannot tell the two
apart, and below about 0.2 Hz neither can its frequency term. Within a pair
episode the tracker hands the two fish back and forth, and which identity
each fish leaves on is a coin toss: on the dense benchmark a perfect pair
head lowered IDF1 because fish left long episodes on their partner's
identity.

The electrode patterns *outside* the episode still tell the two fish apart.
``resolve_pairs`` finds every episode of two identities with detections in
one frame closer than ``max_df`` (frames less than ``merge_gap`` apart are
one episode) and compares the patterns (mean power per electrode in dB,
relative to their mean) of both identities just before and just after it:
over ``pattern_window`` of detections without another detection within
``clear_df`` in their frame, the nearest such up to ``pattern_reach`` away.
If the fish fit better exchanged (the sum of the two rms differences
smaller by ``margin``), the two identities swap everything after the
episode's closest approach. Episodes are decided in
time order, on the identities as earlier decisions left them.

With ``order``, the frames of the episode in which both identities have one
detection are ordered as well: the higher detection goes to the identity
that was higher before the episode, up to the closest approach, and to the
one that is higher after it from there, so two fish that crossed swap order
once and two that only came close keep theirs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import PairsConfig
from .linking import _pattern


@dataclass
class Episode:
    """Frames `first`..`last` in which identities ident[a] and ident[b] (one
    detection each in `a`, `b`, per close frame) are close."""

    first: int
    last: int
    a: np.ndarray
    b: np.ndarray


def _by_frame(fund_v, idx_v, sel):
    """Detection indices `sel` sorted by frame, then frequency."""
    return sel[np.lexsort((fund_v[sel], idx_v[sel]))]


def close_pairs(fund_v, idx_v, ident_v, max_df, max_neighbours=6):
    """Pairs of detections (a, b) in one frame, closer than max_df, on two
    different identities (a on the smaller one)."""
    order = _by_frame(fund_v, idx_v, np.flatnonzero(np.isfinite(ident_v)))
    f, fr, ids = fund_v[order], idx_v[order], ident_v[order]
    a, b = [], []
    for k in range(1, max_neighbours + 1):
        m = np.flatnonzero(
            (fr[k:] == fr[:-k]) & (f[k:] - f[:-k] < max_df) & (ids[k:] != ids[:-k])
        )
        a.append(order[m])
        b.append(order[m + k])
    a, b = np.concatenate(a), np.concatenate(b)
    swap = ident_v[a] > ident_v[b]
    a[swap], b[swap] = b[swap], a[swap]
    return a, b


def episodes(fund_v, idx_v, ident_v, max_df, merge_frames) -> list[Episode]:
    """Pair episodes of two identities, sorted by their first frame."""
    a, b = close_pairs(fund_v, idx_v, ident_v, max_df)
    if not len(a):
        return []
    ia, ib, fr = ident_v[a], ident_v[b], idx_v[a]
    o = np.lexsort((fr, ib, ia))
    a, b, ia, ib, fr = a[o], b[o], ia[o], ib[o], fr[o]
    new = np.r_[True, (ia[1:] != ia[:-1]) | (ib[1:] != ib[:-1])]
    new[1:] |= np.diff(fr) > merge_frames
    starts = np.flatnonzero(new)
    ends = np.r_[starts[1:], len(a)]
    out = [
        Episode(int(fr[s]), int(fr[e - 1]), a[s:e], b[s:e])
        for s, e in zip(starts, ends, strict=True)
    ]
    out.sort(key=lambda ep: (ep.first, ep.last))
    return out


def nearest_other(fund_v, idx_v) -> np.ndarray:
    """Distance [Hz] of every detection to the nearest other detection in its
    frame (inf if alone)."""
    order = _by_frame(fund_v, idx_v, np.arange(len(fund_v)))
    f, fr = fund_v[order], idx_v[order]
    gap = np.where(fr[1:] == fr[:-1], np.diff(f), np.inf)
    d = np.full(len(f), np.inf)
    d[:-1] = gap
    d[1:] = np.fmin(d[1:], gap)
    out = np.empty(len(f))
    out[order] = d
    return out


def _mode(x):
    x = x[np.isfinite(x)]
    if not len(x):
        return np.nan
    u, c = np.unique(x, return_counts=True)
    return u[c.argmax()]


def _rms(pa, pb):
    d = pa - pb
    return float(np.sqrt(np.mean(d * d)))


def _window(idx_v, f0, f1):
    """Detection positions with frames in [f0, f1)."""
    return np.searchsorted(idx_v, f0, "left"), np.searchsorted(idx_v, f1, "left")


def _swap_after(ident, idx_v, cut, i, j):
    """Identities i and j exchange their detections after frame `cut`."""
    s = np.searchsorted(idx_v, cut, "right")
    tail = ident[s:]
    mi, mj = tail == i, tail == j
    tail[mi], tail[mj] = j, i


def _order(ident, fund_v, idx_v, ep, i, j, upper_pre, upper_post, cut):
    """In the frames of the episode where i and j have one detection each,
    the higher goes to upper_pre up to `cut`, to upper_post after it."""
    lo, hi = _window(idx_v, ep.first, ep.last + 1)
    sel = lo + np.flatnonzero((ident[lo:hi] == i) | (ident[lo:hi] == j))
    if len(sel) < 2:
        return
    d = _by_frame(fund_v, idx_v, sel)
    fr = idx_v[d]
    same = fr[1:] == fr[:-1]
    alone_before = np.r_[True, fr[1:-1] != fr[:-2]]
    alone_after = np.r_[fr[2:] != fr[1:-1], True]
    two = np.flatnonzero(same & alone_before & alone_after)
    two = two[ident[d[two]] != ident[d[two + 1]]]
    lo_det, hi_det = d[two], d[two + 1]
    up = np.where(fr[two] <= cut, upper_pre, upper_post)
    ident[hi_det] = up
    ident[lo_det] = np.where(up == i, j, i)


def resolve_pairs(
    fund_v: np.ndarray,
    idx_v: np.ndarray,
    sign_v: np.ndarray,
    ident_v: np.ndarray,
    times: np.ndarray,
    cfg: PairsConfig,
    log: list | None = None,
) -> np.ndarray:
    """Identities with every pair episode decided as a unit (module doc).
    idx_v must be sorted. `log`, if given, receives a dict per episode with
    both identities: frames, pattern costs and the decision."""
    if not cfg.enabled or not np.isfinite(ident_v).any() or len(times) < 2:
        return ident_v.copy()
    if np.any(np.diff(idx_v) < 0):
        raise ValueError("idx_v must be sorted")
    ident = np.array(ident_v, dtype=float)
    dt = float(times[1] - times[0])
    eps = episodes(fund_v, idx_v, ident, cfg.max_df, max(1, round(cfg.merge_gap / dt)))
    if not eps:
        return ident
    clear = nearest_other(fund_v, idx_v) >= cfg.clear_df
    w = max(1, round(cfg.pattern_window / dt))
    reach = max(w, round(cfg.pattern_reach / dt))
    for ep in eps:
        i, j = _mode(ident[ep.a]), _mode(ident[ep.b])
        if not (np.isfinite(i) and np.isfinite(j)) or i == j:
            continue
        sides = {"pre": _window(idx_v, ep.first - reach, ep.first)}
        sides["post"] = _window(idx_v, ep.last + 1, ep.last + 1 + reach)
        df = np.abs(fund_v[ep.a] - fund_v[ep.b])
        cut = int(idx_v[ep.a[np.argmin(df)]])
        pats, freq = {}, {}
        for side, (lo, hi) in sides.items():
            for lab in (i, j):
                own = lo + np.flatnonzero(ident[lo:hi] == lab)
                m = own[clear[own]]
                if side == "pre":  # the last clear stretch before the episode
                    m = m[idx_v[m] > idx_v[m[-1]] - w] if len(m) else m
                    own = own[idx_v[own] >= ep.first - w]
                else:
                    m = m[idx_v[m] < idx_v[m[0]] + w] if len(m) else m
                    own = own[idx_v[own] <= ep.last + w]
                if len(m) >= cfg.min_pattern:
                    pats[side, lab] = _pattern(sign_v[m])[0]
                if len(own):
                    freq[side, lab] = float(np.median(fund_v[own]))
        keep = swap = np.nan
        if len(pats) == 4:
            keep = _rms(pats["pre", i], pats["post", i]) + _rms(
                pats["pre", j], pats["post", j]
            )
            swap = _rms(pats["pre", i], pats["post", j]) + _rms(
                pats["pre", j], pats["post", i]
            )
        if log is not None:
            log.append(
                {
                    "first": ep.first,
                    "last": ep.last,
                    "cut": cut,
                    "i": i,
                    "j": j,
                    "keep": keep,
                    "swap": swap,
                    "swapped": bool(swap < keep - cfg.margin),
                }
            )
        if swap < keep - cfg.margin:  # False without the four patterns (NaN)
            _swap_after(ident, idx_v, cut, i, j)
            fi, fj = freq.pop(("post", i), None), freq.pop(("post", j), None)
            if fj is not None:
                freq["post", i] = fj
            if fi is not None:
                freq["post", j] = fi
        if cfg.order:
            up = {}
            for side in ("pre", "post"):
                if (side, i) in freq and (side, j) in freq:
                    up[side] = i if freq[side, i] > freq[side, j] else j
            if up:
                pre = up.get("pre", up.get("post"))
                post = up.get("post", pre)
                _order(ident, fund_v, idx_v, ep, i, j, pre, post, cut)
    return ident


def _grow(fund_v, idx_v, ident, partner, gap, df, reverse):
    """One sweep through the frames (forward, or backward with `reverse`):
    every open partner goes to an identity without a detection in its frame
    whose latest detection (a partner it took included) lies at most `gap`
    frames and `df` Hz away, the closest in frequency first."""
    ids = np.unique(ident[np.isfinite(ident)])
    lab = {v: i for i, v in enumerate(ids)}
    far = np.iinfo(np.int64).max // 4
    last_fr = np.full(len(ids), far if reverse else -far, dtype=np.int64)
    last_f = np.zeros(len(ids))
    order = np.argsort(idx_v, kind="stable")
    if reverse:
        order = order[::-1]
    for grp in np.split(order, np.flatnonzero(np.diff(idx_v[order]) != 0) + 1):
        t = idx_v[grp[0]]
        open_p = grp[partner[grp] & ~np.isfinite(ident[grp])]
        if len(open_p):
            present = np.zeros(len(ids), bool)
            present[[lab[v] for v in ident[grp] if np.isfinite(v)]] = True
            near = (np.abs(t - last_fr) <= gap) & ~present
            cand = []
            for q in open_p:
                err = np.abs(fund_v[q] - last_f)
                cand += [(err[i], q, i) for i in np.flatnonzero(near & (err <= df))]
            for _, q, i in sorted(cand):
                if not (np.isfinite(ident[q]) or present[i]):
                    ident[q] = ids[i]
                    present[i] = True
        known = grp[np.isfinite(ident[grp])]
        k = np.array([lab[v] for v in ident[known]], dtype=np.int64)
        last_fr[k] = t
        last_f[k] = fund_v[known]
    return ident


def fill_partners(fund_v, idx_v, ident_v, partner, times, cfg: PairsConfig):
    """Identities with the split partners (`partner`, unassigned in ident_v)
    given to the identities whose gaps they fill (see PairsConfig.partners):
    identities grow into their gaps frame by frame, forward in time and then
    backward, from their own detections and the partners they took. The
    other detections keep their identities."""
    ident = np.array(ident_v, dtype=float)
    ident[partner] = np.nan
    if not partner.any() or not np.isfinite(ident).any() or len(times) < 2:
        return ident
    gap = max(1, round(cfg.fill_gap / float(times[1] - times[0])))
    for reverse in (False, True):
        ident = _grow(fund_v, idx_v, ident, partner, gap, cfg.fill_df, reverse)
    return ident
