"""Confidence of a tracked result, and the places that most likely need a
correction.

Each stage of tracking decides with evidence it then throws away: the
fragment linker's probability of a join, the rise joiner's pattern match, the
learned association's link probabilities, the saliences of an identity's
detections. With ``review.enabled``, `pipeline.track_results` keeps that
evidence (a `Trace`), and this module turns it into *items*, each a place
where the result may be wrong, with a suggested fix:

=============== ============================================== ===========
kind            what may be wrong                              fix
=============== ============================================== ===========
``join``        two fragments joined by the linker (source     ``split``
                ``linker``) or across a rise (``rise``) are
                two fish
``track``       an identity is not a fish (harmonic, clutter)  ``delete``
``missed_join`` two identities that were candidates for a      ``merge``
                join (linker or rise joiner) but stayed apart
                are one fish
=============== ============================================== ===========

Every item gets features (`Item.x`) from that evidence and from the result
around it (sizes, saliences, electrode patterns of both sides, harmonic
relations to other identities). A `ReviewModel`, fitted per kind on
benchmark truth (``benchmarks/dense/review.py fit``), maps them to a
calibrated *confidence*: the probability that the result is right at that
place (the join is right; the identity is a fish; not joining is right).
Without a model the stages' own scores are used, uncalibrated. Lowest
confidence first is the review order. `report` gives the content of
``review.json`` (see docs/review-flags.md).

A join is undone exactly by cutting the identity into the two parts the
join connected: the joins of an identity form a tree over its fragments,
and removing one join leaves two subtrees (`Item.rows`). In ``review.json``
the split is given as a frame ``k`` (the later side's first frame at the
junction) for a reader that cuts in time.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import Config
from .linking import _pattern, _pattern_distance, parallel_evidence

VERSION = 1
EPS = 1e-12

GROUPS = ("join_linker", "join_rise", "track", "missed_linker", "missed_rise")
"""Items by kind and source; each has its own features and model."""
KIND = {
    "join_linker": ("join", "linker"),
    "join_rise": ("join", "rise"),
    "track": ("track", "identity"),
    "missed_linker": ("missed_join", "linker"),
    "missed_rise": ("missed_join", "rise"),
}
ACTION = {"join": "split", "track": "delete", "missed_join": "merge"}

MAX_H = 5  # harmonics looked for
HARM_TOL = 1.0  # Hz, |f - h f_other| of a harmonic (on the harmonic's scale)
DUP_TOL = 1.0  # Hz, another identity this close in the same frame: parallel
NN_CAP = 20.0  # Hz, nearest other identity, capped
CHUNK = 60.0  # s, pattern stability: patterns of consecutive chunks
CONTEXT = 10.0  # s shown around a place (t0, t1)


@dataclass
class Trace:
    """Evidence the tracking stages leave behind (`pipeline.track_results`
    fills it when ``review.enabled``)."""

    frag_v: np.ndarray | None = None
    """Tracker identity of every detection (the fragments; NaN: none)."""
    linked_v: np.ndarray | None = None
    """Identity after linking (or stitching)."""
    joined_v: np.ndarray | None = None
    """Identity after the rise joiner (as linked_v without it)."""
    track: dict = field(default_factory=dict)
    """Learned association: ``link_p``, ``alt_p`` per detection."""
    link: dict = field(default_factory=dict)
    """Fragment linker: candidate pairs (see `linking.link`)."""
    rises: dict = field(default_factory=dict)
    """Rise joiner: candidate pairs (see `rises.join_rises`)."""

    def expand(self, keep: np.ndarray) -> Trace:
        """The trace of a subset of detections (`keep`, bool) on all of them
        (NaN or 0 elsewhere)."""

        def full(v, fill=np.nan):
            if v is None:
                return None
            out = np.full(len(keep), fill, dtype=float)
            out[keep] = v
            return out

        track = {
            k: full(v, 0.0 if k == "alt_p" else np.nan) for k, v in self.track.items()
        }
        return Trace(
            full(self.frag_v),
            full(self.linked_v),
            full(self.joined_v),
            track,
            self.link,
            self.rises,
        )


@dataclass
class Item:
    group: str
    t: float
    f: float
    t0: float
    t1: float
    ids: tuple
    raw: float
    """The stage's own score, as a probability that the result is right here
    (uncalibrated)."""
    x: dict
    """Features by name (`FEATURES[group]`)."""
    why: str
    impact: int
    """Detections a fix would move (the smaller side of a join)."""
    rows: tuple
    """Detections involved: the two sides of a join or missed join (the
    later side second), the identity of a track."""
    k: int = -1
    """First frame of a join's later side at the junction."""
    fragments: tuple = ()
    """Fragments (tracker identities, `Trace.frag_v`) of a join's later side:
    a split gives the identity's detections on them a new id."""

    @property
    def kind(self) -> str:
        return KIND[self.group][0]

    @property
    def source(self) -> str:
        return KIND[self.group][1]

    @property
    def action(self) -> str:
        return ACTION[self.kind]


# --- features ------------------------------------------------------------------

SIDE = [
    "log_n_a",  # detections of the earlier side
    "log_n_b",
    "log_dur_a",  # s
    "log_dur_b",
    "sal_a",  # mean salience
    "sal_b",
    "side_rms",  # dB, electrode patterns of the two sides (all detections)
    "side_corr",
    "side_dmed",  # Hz, |median frequency b - a|
    "side_shared",  # frames with both / the smaller side's frames
    "side_parallel",  # simultaneous detections > 3 Hz apart / simultaneous
]
LINKER = [
    "logit_p",  # the linker's logit
    "alt_a",  # best logit of a's other candidates (as the earlier side)
    "alt_b",  # best logit of b's other candidates (as the later side)
    "gap",
    "abs_df_edge",
    "df_extrap",
    "pattern_rms",
    "pattern_rms_long",
    "pattern_corr_long",
    "sign_agree",
    "occupancy",
    "nn_a",
    "nn_b",
    "margin_a",
    "margin_b",
]
RISE = [
    "rise_rms",  # dB, the joiner's pattern difference
    "rise_corr",
    "rise_jump",  # Hz
    "rise_gap",  # s, start of b - stop of a (< 0: overlap)
    "rise_sal_a",
    "rise_sal_b",
    "rise_n_alt",  # other candidates of the same stop or start
]
TRACK = [
    "log_n",
    "log_dur",
    "fill",  # frames with a detection / frames spanned
    "sal_mean",
    "sal_q10",
    "sal_q90",
    "log_freq",
    "f_iqr",  # Hz
    "level",  # dB, median total power relative to all identities' median
    "log_units",  # fragments joined into it
    "n_rise",  # rise joins in it
    "min_join",  # lowest linker logit of its joins (10 without joins)
    "harm_low",  # detections with another identity at f / h (it is a harmonic)
    "harm_high",  # detections with another identity at h f
    "parallel",  # detections with another identity within DUP_TOL
    "nn_df",  # Hz, median distance to the nearest other identity (same frame)
    "pattern_drift",  # dB, median rms change of the pattern between minutes
]
ASSOC = [
    "assoc_mean",  # mean learned link probability along the identity
    "assoc_q05",
    "assoc_alt",  # mean best link into it from another track
]
RAW = ["raw_logit"]  # every item: the logit of its raw score
FEATURES = {
    "join_linker": SIDE + LINKER,
    "join_rise": SIDE + RISE,
    "track": TRACK,
    "missed_linker": SIDE + LINKER,
    "missed_rise": SIDE + RISE,
}
"""Features of each group (the learned association adds `ASSOC` to tracks)."""


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


class _Dets:
    """The detections and the per-identity summaries the features use."""

    def __init__(self, fund_v, idx_v, sign_v, sal_v, ident_v, times):
        self.fund = np.asarray(fund_v, dtype=float)
        self.idx = np.asarray(idx_v, dtype=np.int64)
        self.sign = np.asarray(sign_v, dtype=float)
        self.sal = None if sal_v is None else np.asarray(sal_v, dtype=float)
        self.ident = np.asarray(ident_v, dtype=float)
        self.times = np.asarray(times, dtype=float)
        self.t = self.times[self.idx]
        self.dt = float(times[1] - times[0]) if len(times) > 1 else 1.0
        has = np.flatnonzero(np.isfinite(self.ident))
        order = has[np.lexsort((self.idx[has], self.ident[has]))]
        ids, first = np.unique(self.ident[order], return_index=True)
        bounds = np.append(first, len(order))
        self.ids = ids
        self.rows = {
            float(i): order[a:b]
            for i, a, b in zip(ids, bounds[:-1], bounds[1:], strict=True)
        }
        power = (
            np.maximum(self.sign.sum(1), EPS)
            if self.sign.ndim == 2
            else np.ones(len(self.fund))
        )
        self.level_v = 10 * np.log10(power)
        self.level_ref = float(np.median(self.level_v[has])) if len(has) else 0.0
        self._frame_relations(has)

    def _frame_relations(self, has):
        """Per assigned detection: another identity at f / h, at h f, within
        DUP_TOL, and the nearest other identity in the same frame."""
        n = len(self.fund)
        self.harm_low_v = np.zeros(n, bool)
        self.harm_high_v = np.zeros(n, bool)
        self.dup_v = np.zeros(n, bool)
        self.nn_v = np.full(n, NN_CAP)
        if len(has) < 2:
            return
        f, k, i = self.fund[has], self.idx[has], self.ident[has]
        scale = 4.0 * max(float(f.max()), 1.0) * (MAX_H + 1)
        key = k * scale + f
        o = np.argsort(key, kind="stable")
        ks, iss = key[o], i[o]

        def nearest_other(target, tol):
            """Is there a detection of another identity in the same frame
            within tol of target (per detection of `has`)?"""
            hit = np.zeros(len(has), bool)
            q = k * scale + target
            lo = np.searchsorted(ks, q - tol, "left")
            hi = np.searchsorted(ks, q + tol, "right")
            for j in np.flatnonzero(hi > lo):
                sel = slice(lo[j], hi[j])
                if np.any(iss[sel] != i[j]):
                    hit[j] = True
            return hit

        low = np.zeros(len(has), bool)
        high = np.zeros(len(has), bool)
        for h in range(2, MAX_H + 1):
            low |= nearest_other(f / h, HARM_TOL / h)
            high |= nearest_other(f * h, HARM_TOL * h)
        # nearest other identity in the same frame: neighbours in (frame, f)
        pos = np.empty(len(has), np.int64)
        pos[o] = np.arange(len(has))
        nn = np.full(len(has), NN_CAP)
        for step in (-1, 1):
            nb = pos + step
            ok = (nb >= 0) & (nb < len(has))
            j = np.flatnonzero(ok)
            other = o[nb[ok]]
            same = (k[other] == k[j]) & (i[other] != i[j])
            # skip neighbours of the same identity (duplicates are rare)
            d = np.abs(f[other] - f[j])
            nn[j[same]] = np.minimum(nn[j[same]], d[same])
        self.harm_low_v[has] = low
        self.harm_high_v[has] = high
        self.nn_v[has] = np.minimum(nn, NN_CAP)
        self.dup_v[has] = nn <= DUP_TOL

    def mean_sal(self, rows) -> float:
        if self.sal is None or len(rows) == 0:
            return np.nan
        return float(self.sal[rows].mean())

    def span(self, rows) -> float:
        return float(self.t[rows].max() - self.t[rows].min()) if len(rows) else 0.0


def _parallel(da: _Dets, ra, rb, max_dt=0.3, df=3.0) -> tuple[float, float]:
    """Shared frames (/ the smaller side's frames) and the fraction of
    simultaneous detections more than df apart (two fish at once)."""
    fa, fb = np.unique(da.idx[ra]), np.unique(da.idx[rb])
    if not len(fa) or not len(fb):
        return 0.0, 0.0
    shared = len(np.intersect1d(fa, fb, assume_unique=True)) / min(len(fa), len(fb))

    oa, ob = np.argsort(da.t[ra], kind="stable"), np.argsort(da.t[rb], kind="stable")
    n_par, n_near = parallel_evidence(
        da.t[ra][oa], da.fund[ra][oa], da.t[rb][ob], da.fund[rb][ob], max_dt, df
    )
    return float(shared), float(n_par / n_near) if n_near else 0.0


def _side_features(da: _Dets, ra, rb) -> dict:
    pa, pb = _pattern(da.sign[ra])[0], _pattern(da.sign[rb])[0]
    rms, corr = _pattern_distance(pa, pb)
    shared, par = _parallel(da, ra, rb)
    return {
        "log_n_a": np.log1p(len(ra)),
        "log_n_b": np.log1p(len(rb)),
        "log_dur_a": np.log1p(da.span(ra)),
        "log_dur_b": np.log1p(da.span(rb)),
        "sal_a": da.mean_sal(ra),
        "sal_b": da.mean_sal(rb),
        "side_rms": rms,
        "side_corr": corr,
        "side_dmed": abs(float(np.median(da.fund[rb]) - np.median(da.fund[ra]))),
        "side_shared": shared,
        "side_parallel": par,
    }


def _pattern_drift(da: _Dets, rows) -> float:
    """Median rms change [dB] of the electrode pattern from one CHUNK of the
    identity to the next (NaN if shorter than two chunks)."""
    t = da.t[rows]
    if not len(t) or t[-1] - t[0] < 2 * CHUNK:
        return np.nan
    c = ((t - t[0]) // CHUNK).astype(int)
    pats = [
        _pattern(da.sign[rows[c == j]])[0] for j in np.unique(c) if (c == j).sum() >= 5
    ]
    if len(pats) < 2:
        return np.nan
    d = [_pattern_distance(a, b)[0] for a, b in itertools.pairwise(pats)]
    return float(np.median(d))


# --- items -----------------------------------------------------------------------


class _Forest:
    """The joins of all identities as edges between fragments (tracker
    identities). The joins of one identity form a tree: every join connected
    two identities that were apart."""

    def __init__(self, da: _Dets, frag_v: np.ndarray):
        self.frag = np.asarray(frag_v, dtype=float)
        ok = np.isfinite(self.frag) & np.isfinite(da.ident)
        # final identity of a fragment: where most of its detections went
        self.final: dict[float, float] = {}
        if ok.any():
            pairs = np.stack([self.frag[ok], da.ident[ok]], 1)
            uniq, cnt = np.unique(pairs, axis=0, return_counts=True)
            o = np.lexsort((-cnt, uniq[:, 0]))
            first = np.r_[True, uniq[o[1:], 0] != uniq[o[:-1], 0]]
            for fr, idn in uniq[o[first]]:
                self.final[float(fr)] = float(idn)
        self.edges: list[tuple] = []  # (node a, node b, group, j)
        self._by_ident: dict[float, list[int]] = {}

    def add(self, a: float, b: float, group: str, j: int) -> None:
        self.edges.append((a, b, group, j))
        ia = self.final.get(a)
        if ia is not None and ia == self.final.get(b):
            self._by_ident.setdefault(ia, []).append(len(self.edges) - 1)

    def node_of(self, rows) -> float | None:
        """The fragment most of `rows` belong to."""
        fr = self.frag[rows]
        fr = fr[np.isfinite(fr)]
        if not len(fr):
            return None
        u, c = np.unique(fr, return_counts=True)
        return float(u[np.argmax(c)])

    def sides(self, da: _Dets, e: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Rows of the two parts of an identity that removing join e leaves
        (a's side first), or None if the join is not inside one identity.
        Fragments joined by neither side (untraced joins) are left out."""
        a, b, _, _ = self.edges[e]
        ident = self.final.get(a)
        if ident is None or ident != self.final.get(b) or ident not in da.rows:
            return None
        parent: dict[float, float] = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for q in self._by_ident.get(ident, []):
            if q != e:
                x, y = self.edges[q][:2]
                parent[find(x)] = find(y)
        rows = da.rows[ident]
        fr = self.frag[rows]
        ok = np.isfinite(fr)
        u, inv = np.unique(fr[ok], return_inverse=True)
        ra_root, rb_root = find(a), find(b)
        root = np.array([find(float(x)) for x in u])
        side = np.full(len(rows), -1)
        side[np.flatnonzero(ok)] = np.select(
            [root[inv] == ra_root, root[inv] == rb_root], [0, 1], -1
        )
        if not (side == 0).any() or not (side == 1).any():
            return None
        return rows[side == 0], rows[side == 1]


def build_items(
    fund_v, idx_v, sign_v, sal_v, ident_v, times, trace: Trace, cfg: Config
) -> list[Item]:
    """Every join, identity and rejected candidate join of a tracked result,
    with its features (see module doc)."""
    da = _Dets(fund_v, idx_v, sign_v, sal_v, ident_v, times)
    if not len(da.ids):
        return []
    frag = trace.frag_v if trace.frag_v is not None else da.ident
    linked = trace.linked_v if trace.linked_v is not None else frag
    forest = _Forest(da, frag)
    join_logits: dict[float, list] = {}
    rise_count: dict[float, int] = {}

    # linker joins and candidates ------------------------------------------------
    lk = trace.link
    lk_ok = len(lk.get("a", [])) > 0
    if lk_ok:
        logit = _logit(lk["prob"])
        col = {k: i for i, k in enumerate(lk["features"])}
        best_a: dict[float, list] = {}
        best_b: dict[float, list] = {}
        for j in range(len(logit)):
            best_a.setdefault(float(lk["a"][j]), []).append(j)
            best_b.setdefault(float(lk["b"][j]), []).append(j)

        def alt(group, key, j):
            others = [logit[q] for q in group.get(key, []) if q != j]
            return max(others) if others else -10.0

        def linker_x(j):
            x = {
                "logit_p": float(logit[j]),
                "alt_a": alt(best_a, float(lk["a"][j]), j),
                "alt_b": alt(best_b, float(lk["b"][j]), j),
            }
            for name in LINKER[3:]:
                x[name] = float(lk["X"][j, col[name]])
            return x

    if lk_ok:
        for j in np.flatnonzero(lk["accepted"]):
            forest.add(float(lk["a"][j]), float(lk["b"][j]), "join_linker", int(j))
    rs = trace.rises
    rs_ok = len(rs.get("a", [])) > 0
    if rs_ok:
        for j in np.flatnonzero(rs["accepted"]):
            na = forest.node_of(np.flatnonzero(linked == rs["a"][j]))
            nb = forest.node_of(np.flatnonzero(linked == rs["b"][j]))
            if na is not None and nb is not None:
                forest.add(na, nb, "join_rise", int(j))

    out: list[Item] = []
    for e, (na, nb, group, j) in enumerate(forest.edges):
        sides = forest.sides(da, e)
        if sides is None:
            continue
        ra, rb = sides
        x = _side_features(da, ra, rb)
        ident = forest.final[na]
        if group == "join_linker":
            x |= linker_x(j)
            p = float(lk["prob"][j])
            t_j, f_j = _edge(da, np.flatnonzero(frag == nb), first=True)
            why = (
                f"linker p = {p:.2f} across {x['gap']:.1f} s, "
                f"df {x['abs_df_edge']:.1f} Hz, patterns {x['pattern_rms']:.1f} dB apart"
            )
            join_logits.setdefault(ident, []).append(x["logit_p"])
            raw = p
        else:
            x |= _rise_x(da, linked, rs, j, cfg)
            t_j = float(rs["t_start"][j])
            f_j = _edge(da, rb[da.t[rb] >= t_j - 0.5 * da.dt], first=True)[1]
            raw = float(
                np.clip(1 - rs["rms"][j] / (2 * cfg.rises.max_pattern_rms), 0, 1)
            )
            why = (
                f"rise join: jump {x['rise_jump']:+.1f} Hz, patterns "
                f"{x['rise_rms']:.1f} dB apart (r = {x['rise_corr']:.2f})"
            )
            rise_count[ident] = rise_count.get(ident, 0) + 1
        later = rb[da.t[rb] >= t_j - 0.5 * da.dt]
        k = int(da.idx[later].min()) if len(later) else int(da.idx[rb].min())
        out.append(
            Item(
                group,
                t_j,
                f_j,
                t_j - CONTEXT,
                t_j + CONTEXT,
                (float(ident),),
                raw,
                x,
                why,
                int(min(len(ra), len(rb))),
                (ra, rb),
                k,
                tuple(float(x) for x in np.unique(frag[rb][np.isfinite(frag[rb])])),
            )
        )

    # identities -----------------------------------------------------------------
    assoc = trace.track.get("link_p") is not None
    units = {}
    for idn in forest.final.values():
        units[idn] = units.get(idn, 0) + 1
    for idn in da.ids:
        idn = float(idn)
        rows = da.rows[idn]
        t = da.t[rows]
        span = float(t[-1] - t[0])
        n_frames = len(np.unique(da.idx[rows]))
        sal = da.sal[rows] if da.sal is not None else np.full(len(rows), np.nan)
        f = da.fund[rows]
        x = {
            "log_n": np.log1p(len(rows)),
            "log_dur": np.log1p(span),
            "fill": n_frames / (span / da.dt + 1),
            "sal_mean": float(np.mean(sal)),
            "sal_q10": float(np.quantile(sal, 0.1)),
            "sal_q90": float(np.quantile(sal, 0.9)),
            "log_freq": float(np.log(max(np.median(f), 1.0))),
            "f_iqr": float(np.subtract(*np.quantile(f, [0.75, 0.25]))),
            "level": float(np.median(da.level_v[rows]) - da.level_ref),
            "log_units": np.log1p(units.get(idn, 1)),
            "n_rise": float(rise_count.get(idn, 0)),
            "min_join": float(min(join_logits.get(idn, [10.0]))),
            "harm_low": float(da.harm_low_v[rows].mean()),
            "harm_high": float(da.harm_high_v[rows].mean()),
            "parallel": float(da.dup_v[rows].mean()),
            "nn_df": float(np.median(da.nn_v[rows])),
            "pattern_drift": _pattern_drift(da, rows),
        }
        if assoc:
            lp = trace.track["link_p"][rows]
            lp = lp[np.isfinite(lp)]
            x["assoc_mean"] = float(lp.mean()) if len(lp) else np.nan
            x["assoc_q05"] = float(np.quantile(lp, 0.05)) if len(lp) else np.nan
            x["assoc_alt"] = float(trace.track["alt_p"][rows].mean())
        raw = x["sal_mean"] if np.isfinite(x["sal_mean"]) else 0.5
        why = f"{len(rows)} detections over {span:.0f} s, mean salience {x['sal_mean']:.2f}"
        if x["harm_low"] >= 0.3:
            why += f", {100 * x['harm_low']:.0f} % at a harmonic of another identity"
        out.append(
            Item(
                "track",
                float(t[0]),
                float(np.median(f)),
                float(t[0]),
                float(t[-1]),
                (idn,),
                float(raw),
                x,
                why,
                len(rows),
                (rows,),
            )
        )

    # rejected candidates (missed joins) ----------------------------------------------
    if cfg.review.missed_joins:
        out += _missed(
            da,
            frag,
            linked,
            forest,
            lk if lk_ok else None,
            rs if rs_ok else None,
            cfg,
            linker_x if lk_ok else None,
        )
    for it in out:  # the raw score as a feature (a model may only map it)
        it.x["raw_logit"] = float(_logit(it.raw))
    return out


def _edge(da: _Dets, rows, first: bool) -> tuple[float, float]:
    """Time and frequency (median of 5 detections) at the start (first) or
    end of rows."""
    if not len(rows):
        return np.nan, np.nan
    o = np.argsort(da.t[rows], kind="stable")
    r = rows[o][:5] if first else rows[o][-5:]
    return float(da.t[rows[o][0 if first else -1]]), float(np.median(da.fund[r]))


def _rise_x(da: _Dets, linked, rs, j, cfg: Config) -> dict:
    """Features of a rise candidate j (from the identities the joiner saw)."""
    rc = cfg.rises
    ma = np.flatnonzero(linked == rs["a"][j])
    mb = np.flatnonzero(linked == rs["b"][j])
    t_s, t_e = float(rs["t_start"][j]), float(rs["t_stop"][j])
    ta, tb = da.t[ma], da.t[mb]
    pa = ma[(ta >= t_e - rc.pattern_window) & (ta <= t_e)]
    pb = mb[(tb >= t_s) & (tb <= t_s + rc.pattern_window)]
    ea = ma[(ta >= t_e - rc.edge_window) & (ta <= t_e)]
    eb = mb[(tb >= t_s) & (tb <= t_s + rc.edge_window)]
    if len(pa) and len(pb):
        rms, corr = _pattern_distance(
            _pattern(da.sign[pa])[0], _pattern(da.sign[pb])[0]
        )
    else:
        rms, corr = float(rs["rms"][j]), np.nan
    jump = (
        float(np.median(da.fund[eb]) - np.median(da.fund[ea]))
        if len(ea) and len(eb)
        else np.nan
    )
    same = (rs["a"] == rs["a"][j]) | (rs["b"] == rs["b"][j])
    return {
        "rise_rms": rms,
        "rise_corr": corr,
        "rise_jump": jump,
        "rise_gap": t_s - t_e,
        "rise_sal_a": da.mean_sal(pa),
        "rise_sal_b": da.mean_sal(pb),
        "rise_n_alt": float(same.sum() - 1),
    }


def _missed(da, frag, linked, forest, lk, rs, cfg, linker_x) -> list[Item]:
    """Rejected candidates between two identities: the best per pair."""
    best: dict[tuple, tuple] = {}  # (id x, id y) -> (group, j, score)
    if lk is not None:
        for j in np.flatnonzero(
            ~lk["accepted"] & (lk["prob"] >= cfg.review.min_missed_prob)
        ):
            ix = forest.final.get(float(lk["a"][j]))
            iy = forest.final.get(float(lk["b"][j]))
            if ix is None or iy is None or ix == iy:
                continue
            key = (min(ix, iy), max(ix, iy))
            p = float(lk["prob"][j])
            if key not in best or (
                best[key][0] == "missed_linker" and p > best[key][2]
            ):
                best[key] = ("missed_linker", j, p)
    if rs is not None:
        for j in np.flatnonzero(~rs["accepted"]):
            na = forest.node_of(np.flatnonzero(linked == rs["a"][j]))
            nb = forest.node_of(np.flatnonzero(linked == rs["b"][j]))
            ix, iy = forest.final.get(na, None), forest.final.get(nb, None)
            if ix is None or iy is None or ix == iy:
                continue
            key = (min(ix, iy), max(ix, iy))
            if key not in best:
                best[key] = ("missed_rise", j, 0.0)
    out = []
    for group, j, _ in best.values():
        if group == "missed_linker":
            na, nb = float(lk["a"][j]), float(lk["b"][j])
        else:
            na = forest.node_of(np.flatnonzero(linked == rs["a"][j]))
            nb = forest.node_of(np.flatnonzero(linked == rs["b"][j]))
        ix, iy = forest.final[na], forest.final[nb]
        rx, ry = da.rows.get(ix), da.rows.get(iy)
        if rx is None or ry is None:
            continue
        x = _side_features(da, rx, ry)
        t_j, f_j = _edge(da, np.flatnonzero(frag == nb), first=True)
        if not np.isfinite(t_j):
            continue
        if group == "missed_linker":
            x |= linker_x(j)
            p = float(lk["prob"][j])
            raw = 1 - p
            why = (
                f"not joined: linker p = {p:.2f} across {x['gap']:.1f} s, "
                f"df {x['abs_df_edge']:.1f} Hz"
            )
        else:
            x |= _rise_x(da, linked, rs, j, cfg)
            t_j = float(rs["t_start"][j])
            raw = float(np.clip(rs["rms"][j] / (2 * cfg.rises.max_pattern_rms), 0, 1))
            why = f"not joined across a rise: jump {x['rise_jump']:+.1f} Hz"
        out.append(
            Item(
                group,
                t_j,
                f_j,
                t_j - CONTEXT,
                t_j + CONTEXT,
                (float(ix), float(iy)),
                raw,
                x,
                why,
                int(min(len(rx), len(ry))),
                (rx, ry),
            )
        )
    return out


# --- the model ------------------------------------------------------------------


@dataclass
class GroupModel:
    """Standardised features -> MLP (no hidden layer: logistic regression)
    -> probability, then an optional monotone calibration map."""

    features: list
    mean: np.ndarray
    std: np.ndarray
    weights: list
    biases: list
    cal_x: np.ndarray | None = None
    cal_y: np.ndarray | None = None

    def matrix(self, items: list[Item]) -> np.ndarray:
        X = np.array(
            [[it.x.get(k, np.nan) for k in self.features] for it in items], float
        )
        return X.reshape(len(items), len(self.features))

    def prob(self, X: np.ndarray) -> np.ndarray:
        h = (X - self.mean) / self.std
        h = np.where(np.isfinite(h), h, 0.0)  # missing: the training mean
        h = np.clip(h, -8, 8)
        for i, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            h = h @ w + b
            if i < len(self.weights) - 1:
                h = np.maximum(h, 0.0)
        p = 1.0 / (1.0 + np.exp(-h[:, 0]))
        if self.cal_x is not None:
            p = np.interp(p, self.cal_x, self.cal_y)
        return p


@dataclass
class ReviewModel:
    groups: dict
    meta: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        arrays = {"meta": np.array(json.dumps(self.meta))}
        for g, m in self.groups.items():
            arrays[f"{g}/features"] = np.array(m.features)
            arrays[f"{g}/mean"] = m.mean
            arrays[f"{g}/std"] = m.std
            for i, (w, b) in enumerate(zip(m.weights, m.biases, strict=True)):
                arrays[f"{g}/w{i}"] = w
                arrays[f"{g}/b{i}"] = b
            if m.cal_x is not None:
                arrays[f"{g}/cal_x"] = m.cal_x
                arrays[f"{g}/cal_y"] = m.cal_y
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> ReviewModel:
        with np.load(path) as z:
            files = set(z.files)
            meta = json.loads(str(z["meta"])) if "meta" in files else {}
            groups = {}
            for g in GROUPS:
                if f"{g}/features" not in files:
                    continue
                n = sum(1 for k in files if k.startswith(f"{g}/w"))
                groups[g] = GroupModel(
                    features=[str(s) for s in z[f"{g}/features"]],
                    mean=z[f"{g}/mean"],
                    std=z[f"{g}/std"],
                    weights=[z[f"{g}/w{i}"] for i in range(n)],
                    biases=[z[f"{g}/b{i}"] for i in range(n)],
                    cal_x=z[f"{g}/cal_x"] if f"{g}/cal_x" in files else None,
                    cal_y=z[f"{g}/cal_y"] if f"{g}/cal_y" in files else None,
                )
        return cls(groups, meta)


_MODELS: dict[str, ReviewModel] = {}


def load_model(path: str | Path) -> ReviewModel:
    key = str(Path(path).resolve())
    if key not in _MODELS:
        _MODELS[key] = ReviewModel.load(path)
    return _MODELS[key]


def confidences(items: list[Item], model: ReviewModel | None) -> np.ndarray:
    """Confidence of every item: the model's for groups it has, else the
    raw score."""
    conf = np.array([it.raw for it in items], dtype=float)
    if model is None:
        return conf
    groups = np.array([it.group for it in items])
    for g, m in model.groups.items():
        sel = np.flatnonzero(groups == g)
        if len(sel):
            conf[sel] = m.prob(m.matrix([items[i] for i in sel]))
    return conf


# --- review.json -----------------------------------------------------------------


def report(
    items: list[Item], conf: np.ndarray, calibration: str | None, cfg: Config
) -> dict:
    """Content of ``review.json``: items, lowest confidence first."""
    order = np.argsort(conf, kind="stable")
    rows = []
    for i in order:
        it = items[i]
        c = float(conf[i])
        if cfg.review.max_confidence is not None and c >= cfg.review.max_confidence:
            continue
        d = {
            "kind": it.kind,
            "source": it.source,
            "confidence": round(c, 4),
            "t": round(it.t, 3),
            "f": round(it.f, 2),
            "t0": round(it.t0, 3),
            "t1": round(it.t1, 3),
            "ids": [float(x) for x in it.ids],
            "action": it.action,
            "impact": it.impact,
            "why": it.why,
        }
        if it.action == "split":
            d["k"] = int(it.k)
            d["fragments"] = [float(x) for x in it.fragments]
        rows.append(d)
    return {"version": VERSION, "calibration": calibration, "items": rows}


def review(results, trace: Trace, cfg: Config) -> tuple[dict, list[Item], np.ndarray]:
    """``review.json`` content, the items and their confidences of tracked
    `results` (see module doc)."""
    items = build_items(
        results.fund_v,
        results.idx_v,
        results.sign_v,
        results.sal_v,
        results.ident_v,
        results.times,
        trace,
        cfg,
    )
    model = load_model(cfg.review.model) if cfg.review.model else None
    conf = confidences(items, model)
    return report(items, conf, cfg.review.model, cfg), items, conf
