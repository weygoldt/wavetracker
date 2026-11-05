"""Joining track fragments into identities with a learned link score.

The tracker (``tracking.track``) produces fragments that are almost always
pure (one fish), but a fish is spread over many of them: tracks break at
rises, where a fish is not detected for longer than ``max_dt``, and next to
close neighbours. This module joins fragments into identities:

1. **Fragments**: every tracker identity with at least ``min_detections``
   detections, summarised at its edges (frequency, slope, baseline, electrode
   power pattern, relative signs between electrodes, level).
2. **Candidates**: pairs (A, B), B starting after A, within ``max_gap`` of A's
   end (or overlapping it), whose frequencies at the junction, baselines or
   medians agree within ``max_df``.
3. **Features** per pair (``FEATURES``): time gap and overlap, frequency jump
   and extrapolation error, baseline and median differences, electrode
   pattern distance and correlation, relative-sign agreement, level change,
   shared frames, fragment sizes, how crowded the frequency is between them,
   and how the pair ranks among the competing candidates of A and of B.
4. **Score**: a small neural network (weights in ``wavetracker/models``)
   gives the probability that A and B are the same fish. It is trained on the
   dense-mixture benchmark (``benchmarks/dense/linker.py``), whose truth says
   which fragments belong to one fish.
5. **Association**: pairs are accepted greedily, most probable first, into
   identities (union-find), unless the two identities would share more than
   a few frames (one fish is detected once per frame) or run at the same time
   at different frequencies (a fish is never at two frequencies at once).

Frames with two detections of one identity keep the one closer to the local
track frequency (``stitching.resolve_duplicates``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import LinkingConfig

EPS = 1e-30
DEFAULT_MODEL = Path(__file__).with_name("models") / "linker.npz"


@dataclass(frozen=True)
class Windows:
    """Where the fragment summaries are measured. Part of the feature
    definitions, so fixed by the model: change only when retraining."""

    edge_detections: int = 5  # first/last detections for the edge frequency
    slope_window: float = 10.0  # s, local slope and scatter at an edge
    baseline_window: float = 60.0  # s (twice at a start: rises decay)
    baseline_quantile: float = 0.1  # low quantile: ignores rises
    median_window: float = 30.0  # s, median frequency and long trend
    pattern_window: float = 10.0  # s, electrode pattern and relative signs
    long_window: float = 60.0  # s, long electrode pattern
    nn_window: float = 5.0  # s, nearest other detection around an edge
    occupancy_df: float = 1.0  # Hz, band around the line joining A and B


WINDOWS = Windows()

FEATURES = [
    "gap",  # s, B start - A end, >= 0 (0 when overlapping)
    "log_gap",
    "overlap",  # s, overlap of the time spans
    "overlap_frac",  # overlap / shorter duration
    "shared_frac",  # frames with detections of both / smaller fragment
    "df_edge",  # Hz, B's first - A's last frequency (signed)
    "abs_df_edge",
    "df_extrap",  # Hz, |B's first - A's last extrapolated by its slope|
    "df_extrap_back",  # Hz, |A's last - B's first extrapolated backwards|
    "dbase",  # Hz, baseline (low quantile) B start - A end (signed)
    "abs_dbase",
    "abs_dmed",  # Hz, |median of B's start - median of A's end|
    "df_overlap",  # Hz, median |f_B - f_A| where both run (0 without overlap)
    "rel_df",  # |df_edge| / frequency * 1000 (per mille)
    "pattern_rms",  # dB, rms difference of the electrode patterns
    "pattern_corr",  # correlation of the electrode patterns (dB)
    "pattern_z",  # pattern_rms over the patterns' own scatter (split halves)
    "pattern_rms_long",  # the same over long_window
    "pattern_corr_long",
    "sign_agree",  # power-weighted agreement of the relative signs
    "dlevel",  # dB, level B start - A end
    "fill_a",  # fraction of A's frames with a detection
    "fill_b",
    "log_dur_a",
    "log_dur_b",
    "log_freq",
    "log_channels",  # number of electrodes
    "jitter_a",  # Hz, scatter of A's frequency around its trend at the end
    "jitter_b",
    "occupancy",  # other detections per frame near the joining line
    "df_mid",  # Hz, A and B extrapolated (long_window trends) to the gap middle
    "nn_a",  # Hz, nearest other detection around A's end (+-nn_window)
    "nn_b",  # Hz, ... around B's start
    "rank_a",  # rank of B among A's candidates by frequency mismatch
    "rank_b",  # rank of A among B's candidates
    "n_cand_a",  # number of candidates of A
    "n_cand_b",
    "margin_a",  # Hz, next-best mismatch of A's candidates - this one's
    "margin_b",
]


@dataclass
class Fragments:
    """Edge summaries of all fragments (one row each)."""

    ident: np.ndarray
    start: np.ndarray
    end: np.ndarray
    n: np.ndarray
    f_start: np.ndarray
    f_end: np.ndarray
    slope_start: np.ndarray
    slope_end: np.ndarray
    base_start: np.ndarray
    base_end: np.ndarray
    med_start: np.ndarray
    med_end: np.ndarray
    jitter_start: np.ndarray
    jitter_end: np.ndarray
    pat_start: np.ndarray  # (n, channels) dB relative to the mean
    pat_end: np.ndarray
    level_start: np.ndarray  # dB, total power
    level_end: np.ndarray
    cplx_start: np.ndarray  # (n, channels) mean complex spectrum, phase-aligned
    cplx_end: np.ndarray
    pat_noise_start: np.ndarray  # dB, rms difference of the two halves' patterns
    pat_noise_end: np.ndarray
    patl_start: np.ndarray  # (n, channels) pattern over long_window
    patl_end: np.ndarray
    fit_start: np.ndarray  # (n, 2) frequency at the edge and slope, long_window
    fit_end: np.ndarray
    members: list  # detection indices, sorted by time
    t: list  # times of the members
    f: list  # frequencies of the members

    def __len__(self):
        return len(self.ident)


def _trend(t, f):
    """Slope and scatter around a straight line."""
    if len(t) < 3 or t[-1] - t[0] <= 0:
        return 0.0, 0.0
    tc = t - t.mean()
    slope = float((tc * (f - f.mean())).sum() / max((tc * tc).sum(), EPS))
    resid = f - f.mean() - slope * tc
    return slope, float(np.median(np.abs(resid)))


def _aligned_mean(c):
    """Mean complex spectrum over detections, each rotated so that its
    strongest electrode (on average) has zero phase: the relative signs
    between electrodes survive, the arbitrary common phase does not."""
    if len(c) == 0:
        return np.zeros(c.shape[1], complex)
    ref = int(np.argmax((np.abs(c) ** 2).mean(0)))
    ph = c[:, ref] / np.maximum(np.abs(c[:, ref]), EPS)
    return (c * np.conj(ph)[:, None]).mean(0)


def _pattern(p):
    db = 10 * np.log10(np.maximum(p.mean(0), EPS))
    return db - db.mean(), float(10 * np.log10(max(p.mean(0).sum(), EPS)))


def _pattern_noise(p, default=6.0):
    """Rms difference [dB] of the patterns of the two halves of p."""
    if len(p) < 4:
        return default
    h = len(p) // 2
    d = _pattern(p[:h])[0] - _pattern(p[h:])[0]
    return float(np.sqrt(np.mean(d * d)))


def _edge_fit(t, f, t_edge):
    """Frequency at t_edge and slope of a straight line through (t, f)."""
    slope, _ = _trend(t, f)
    return float(np.median(f - slope * (t - t_edge))), slope


def fragments(fund_v, idx_v, sign_v, cplx_v, ident_v, times, cfg: LinkingConfig):
    valid = np.isfinite(ident_v)
    sel_all = np.flatnonzero(valid)
    order = sel_all[np.lexsort((idx_v[sel_all], ident_v[sel_all]))]
    ids, first = np.unique(ident_v[order], return_index=True)
    bounds = np.append(first, len(order))
    n_ch = sign_v.shape[1]
    rows = {k: [] for k in Fragments.__dataclass_fields__}
    for i, j0, j1 in zip(ids, bounds[:-1], bounds[1:], strict=True):
        k = order[j0:j1]
        if len(k) < cfg.min_detections:
            continue
        t, f = times[idx_v[k]], fund_v[k]
        ne = min(len(k), WINDOWS.edge_detections)
        sw_a = t <= t[0] + WINDOWS.slope_window
        sw_b = t >= t[-1] - WINDOWS.slope_window
        slope_a, jit_a = _trend(t[sw_a], f[sw_a])
        slope_b, jit_b = _trend(t[sw_b], f[sw_b])
        pa = t <= t[0] + WINDOWS.pattern_window
        pb = t >= t[-1] - WINDOWS.pattern_window
        pat_a, lev_a = _pattern(sign_v[k[pa]])
        pat_b, lev_b = _pattern(sign_v[k[pb]])
        if cplx_v is not None:
            ca, cb = _aligned_mean(cplx_v[k[pa]]), _aligned_mean(cplx_v[k[pb]])
        else:
            ca = cb = np.zeros(n_ch, complex)
        la = t <= t[0] + WINDOWS.long_window
        lb = t >= t[-1] - WINDOWS.long_window
        rows["pat_noise_start"].append(_pattern_noise(sign_v[k[pa]]))
        rows["pat_noise_end"].append(_pattern_noise(sign_v[k[pb]]))
        rows["patl_start"].append(_pattern(sign_v[k[la]])[0])
        rows["patl_end"].append(_pattern(sign_v[k[lb]])[0])
        ma = t <= t[0] + WINDOWS.median_window
        mb = t >= t[-1] - WINDOWS.median_window
        rows["fit_start"].append(_edge_fit(t[ma], f[ma], t[0]))
        rows["fit_end"].append(_edge_fit(t[mb], f[mb], t[-1]))
        rows["ident"].append(float(i))
        rows["start"].append(float(t[0]))
        rows["end"].append(float(t[-1]))
        rows["n"].append(len(k))
        rows["f_start"].append(float(np.median(f[:ne])))
        rows["f_end"].append(float(np.median(f[-ne:])))
        rows["slope_start"].append(slope_a)
        rows["slope_end"].append(slope_b)
        a = t <= t[0] + 2 * WINDOWS.baseline_window
        b = t >= t[-1] - WINDOWS.baseline_window
        rows["base_start"].append(float(np.quantile(f[a], WINDOWS.baseline_quantile)))
        rows["base_end"].append(float(np.quantile(f[b], WINDOWS.baseline_quantile)))
        ma = t <= t[0] + WINDOWS.median_window
        mb = t >= t[-1] - WINDOWS.median_window
        rows["med_start"].append(float(np.median(f[ma])))
        rows["med_end"].append(float(np.median(f[mb])))
        rows["jitter_start"].append(jit_a)
        rows["jitter_end"].append(jit_b)
        rows["pat_start"].append(pat_a)
        rows["pat_end"].append(pat_b)
        rows["level_start"].append(lev_a)
        rows["level_end"].append(lev_b)
        rows["cplx_start"].append(ca)
        rows["cplx_end"].append(cb)
        rows["members"].append(k)
        rows["t"].append(t)
        rows["f"].append(f)
    out = {}
    for key, v in rows.items():
        if key in ("members", "t", "f"):
            out[key] = v
        elif key in ("pat_start", "pat_end", "patl_start", "patl_end"):
            out[key] = np.array(v, dtype=float).reshape(len(v), n_ch)
        elif key.startswith("cplx_"):
            out[key] = np.array(v, dtype=complex).reshape(len(v), n_ch)
        elif key.startswith("fit_"):
            out[key] = np.array(v, dtype=float).reshape(len(v), 2)
        else:
            out[key] = np.asarray(v, dtype=float)
    return Fragments(**out)


class _Detections:
    """Detections with identity, sorted by time, for window queries."""

    def __init__(self, t, f, ident):
        order = np.argsort(t, kind="stable")
        self.t, self.f, self.ident = t[order], f[order], ident[order]

    def window(self, t0, t1):
        lo = np.searchsorted(self.t, t0, "left")
        hi = np.searchsorted(self.t, t1, "right")
        return self.t[lo:hi], self.f[lo:hi], self.ident[lo:hi]


def _pattern_distance(pa, pb):
    """Rms difference [dB] and correlation of two electrode patterns."""
    d = pa - pb
    sa, sb = pa - pa.mean(), pb - pb.mean()
    den = np.sqrt((sa * sa).sum() * (sb * sb).sum())
    return float(np.sqrt(np.mean(d * d))), float((sa * sb).sum() / max(den, EPS))


def _sign_agreement(ca, cb):
    """Power-weighted agreement of relative signs between electrodes, with
    A's strongest electrode as the reference for both (-1..1)."""
    ref = int(np.argmax(np.abs(ca)))
    ra = ca * np.conj(ca[ref] / max(abs(ca[ref]), EPS))
    rb = cb * np.conj(cb[ref] / max(abs(cb[ref]), EPS))
    w = np.sqrt(np.abs(ra) * np.abs(rb))
    w[ref] = 0.0
    if w.sum() <= 0:
        return 0.0
    agree = np.sign(ra.real) * np.sign(rb.real)
    return float((w * agree).sum() / w.sum())


def candidates(fr: Fragments, cfg: LinkingConfig) -> np.ndarray:
    """(A, B) index pairs: B starts after A (or with it, larger ident), at most
    max_gap after A's end, and some frequency measure agrees within max_df, or
    within rise_df for gaps up to rise_gap."""
    order = np.lexsort((fr.ident, fr.start))
    starts = fr.start[order]
    pairs = []
    for rank, ia in enumerate(order):
        hi = np.searchsorted(starts, fr.end[ia] + cfg.max_gap, "right")
        ib = order[rank + 1 : hi]
        if len(ib) == 0:
            continue
        mis = np.minimum.reduce(
            [
                np.abs(fr.f_start[ib] - fr.f_end[ia]),
                np.abs(fr.base_start[ib] - fr.base_end[ia]),
                np.abs(fr.med_start[ib] - fr.med_end[ia]),
            ]
        )
        # rises: a jump of up to rise_df at a short gap (onset, or the
        # return to the baseline from a fragment of the rise)
        gap = fr.start[ib] - fr.end[ia]
        jump = np.abs(fr.f_start[ib] - fr.f_end[ia])
        rise = (gap <= cfg.rise_gap) & (jump <= cfg.rise_df) & (cfg.rise_df > 0)
        ib = ib[(mis <= cfg.max_df) | rise]
        pairs.extend((ia, b) for b in ib)
    return np.array(pairs, dtype=np.int64).reshape(-1, 2)


def features(
    fr: Fragments, pairs: np.ndarray, dets: _Detections, cfg: LinkingConfig
) -> np.ndarray:
    """Feature matrix (pairs, len(FEATURES))."""
    n = len(pairs)
    X = np.zeros((n, len(FEATURES)))
    col = {k: i for i, k in enumerate(FEATURES)}
    frame_dt = _frame_dt(fr)
    for p, (a, b) in enumerate(pairs):
        gap = fr.start[b] - fr.end[a]
        ov = max(0.0, min(fr.end[a], fr.end[b]) - fr.start[b])
        dur_a = fr.end[a] - fr.start[a]
        dur_b = fr.end[b] - fr.start[b]
        g = max(gap, 0.0)
        df_edge = fr.f_start[b] - fr.f_end[a]
        ext = fr.f_end[a] + np.clip(fr.slope_end[a] * g, -5, 5)
        back = fr.f_start[b] - np.clip(fr.slope_start[b] * g, -5, 5)
        shared = 0.0
        df_ov = 0.0
        if ov > 0:
            ta, fa = fr.t[a], fr.f[a]
            tb, fb = fr.t[b], fr.f[b]
            inb = tb <= fr.end[a]
            if inb.any():
                df_ov = float(np.median(np.abs(fb[inb] - np.interp(tb[inb], ta, fa))))
            ina = ta >= fr.start[b]
            shared = len(np.intersect1d(ta[ina], tb[inb])) / min(fr.n[a], fr.n[b])
        # crowding: detections of others near the line joining A and B
        if g > 0:
            tt, ff, ii = dets.window(fr.end[a], fr.start[b])
            other = (ii != fr.ident[a]) & (ii != fr.ident[b])
            line = fr.f_end[a] + (tt - fr.end[a]) / g * (fr.f_start[b] - fr.f_end[a])
            near = np.abs(ff - line) <= WINDOWS.occupancy_df
            occ = (other & near).sum() / max(g / frame_dt, 1.0)
        else:
            occ = 0.0
        rms, corr = _pattern_distance(fr.pat_end[a], fr.pat_start[b])
        rms_l, corr_l = _pattern_distance(fr.patl_end[a], fr.patl_start[b])
        noise = fr.pat_noise_end[a] + fr.pat_noise_start[b] + 0.5
        # both trends extrapolated to the middle of the gap (or overlap)
        tm = 0.5 * (fr.end[a] + fr.start[b])
        fa_m = fr.fit_end[a, 0] + np.clip(fr.fit_end[a, 1] * (tm - fr.end[a]), -3, 3)
        fb_m = fr.fit_start[b, 0] + np.clip(
            fr.fit_start[b, 1] * (tm - fr.start[b]), -3, 3
        )
        nn = []
        for t_edge, f_edge in ((fr.end[a], fr.f_end[a]), (fr.start[b], fr.f_start[b])):
            _, ff, ii = dets.window(
                t_edge - WINDOWS.nn_window, t_edge + WINDOWS.nn_window
            )
            other = (ii != fr.ident[a]) & (ii != fr.ident[b])
            nn.append(
                min(float(np.min(np.abs(ff[other] - f_edge))), 20.0)
                if other.any()
                else 20.0
            )
        X[p] = [
            g,
            np.log1p(g),
            ov,
            ov / max(min(dur_a, dur_b), 1e-3),
            shared,
            df_edge,
            abs(df_edge),
            abs(fr.f_start[b] - ext),
            abs(fr.f_end[a] - back),
            fr.base_start[b] - fr.base_end[a],
            abs(fr.base_start[b] - fr.base_end[a]),
            abs(fr.med_start[b] - fr.med_end[a]),
            df_ov,
            1000 * abs(df_edge) / max(fr.f_end[a], 1.0),
            rms,
            corr,
            rms / noise,
            rms_l,
            corr_l,
            _sign_agreement(fr.cplx_end[a], fr.cplx_start[b]),
            fr.level_start[b] - fr.level_end[a],
            min(fr.n[a] * frame_dt / max(dur_a + frame_dt, EPS), 1.0),
            min(fr.n[b] * frame_dt / max(dur_b + frame_dt, EPS), 1.0),
            np.log1p(dur_a),
            np.log1p(dur_b),
            np.log(max(fr.f_end[a], 1.0)),
            np.log(fr.pat_end.shape[1]),
            fr.jitter_end[a],
            fr.jitter_start[b],
            occ,
            abs(fa_m - fb_m),
            nn[0],
            nn[1],
            0,
            0,
            0,
            0,
            0,
            0,
        ]
    if n:
        # competition: how this pair ranks among the candidates of A and of B
        mis = np.minimum(X[:, col["abs_df_edge"]], X[:, col["abs_dbase"]])
        mis = np.minimum(mis, X[:, col["abs_dmed"]])
        for side, key in ((0, "a"), (1, "b")):
            owner = pairs[:, side]
            order = np.lexsort((mis, owner))
            o = owner[order]
            first = np.r_[True, o[1:] != o[:-1]]
            grp = np.cumsum(first) - 1
            starts = np.flatnonzero(first)
            rank = np.arange(n) - starts[grp]
            counts = np.diff(np.append(starts, n))
            best = mis[order][starts]
            second = np.full(len(starts), 20.0)
            has2 = counts > 1
            second[has2] = mis[order][starts[has2] + 1]
            m_sorted = mis[order]
            # margin: for the best, to the second; for others, best - this (< 0)
            margin = np.where(rank == 0, second[grp] - m_sorted, best[grp] - m_sorted)
            X[order, col[f"rank_{key}"]] = rank
            X[order, col[f"n_cand_{key}"]] = counts[grp]
            X[order, col[f"margin_{key}"]] = np.clip(margin, -20, 20)
    return X


def _frame_dt(fr: Fragments) -> float:
    for t in fr.t:
        if len(t) > 1:
            d = np.diff(t)
            d = d[d > 0]
            if len(d):
                return float(d.min())
    return 1.0


# --- the model -----------------------------------------------------------------


@dataclass
class LinkModel:
    """A small MLP: standardised features -> hidden layers (ReLU) -> logit."""

    mean: np.ndarray
    std: np.ndarray
    weights: list
    biases: list
    features: list

    def logit(self, X: np.ndarray) -> np.ndarray:
        if list(self.features) != FEATURES:
            raise ValueError("model was trained with other features")
        h = (X - self.mean) / self.std
        for i, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            h = h @ w + b
            if i < len(self.weights) - 1:
                h = np.maximum(h, 0.0)
        return h[:, 0]

    def prob(self, X: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.logit(X)))

    def save(self, path: str | Path) -> None:
        arrays = {"mean": self.mean, "std": self.std}
        for i, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            arrays[f"w{i}"] = w
            arrays[f"b{i}"] = b
        np.savez(path, features=np.array(self.features), **arrays)

    @classmethod
    def load(cls, path: str | Path | None = None) -> LinkModel:
        """Load weights; the model shipped with wavetracker if path is None."""
        path = Path(path) if path is not None else DEFAULT_MODEL
        with np.load(path) as z:
            n = sum(1 for k in z.files if k.startswith("w"))
            return cls(
                mean=z["mean"],
                std=z["std"],
                weights=[z[f"w{i}"] for i in range(n)],
                biases=[z[f"b{i}"] for i in range(n)],
                features=[str(s) for s in z["features"]],
            )


# --- association -----------------------------------------------------------------


def parallel_evidence(ta, fa, tb, fb, max_dt: float, df: float) -> tuple[int, int]:
    """Where two identities run at the same time: the number of detections of
    the smaller with a detection of the other within max_dt, and how many of
    those differ by more than df in frequency (two fish, not one)."""
    if len(ta) < len(tb):
        ta, fa, tb, fb = tb, fb, ta, fa
    lo, hi = max(ta[0], tb[0]) - max_dt, min(ta[-1], tb[-1]) + max_dt
    if hi < lo:
        return 0, 0
    sel = (tb >= lo) & (tb <= hi)
    tq, fq = tb[sel], fb[sel]
    j = np.clip(np.searchsorted(ta, tq), 1, len(ta) - 1)
    left = np.abs(ta[j - 1] - tq) <= np.abs(ta[j] - tq)
    j = np.where(left, j - 1, j)
    near = np.abs(ta[j] - tq) <= max_dt
    differ = np.abs(fa[j] - fq)[near] > df
    return int(differ.sum()), int(near.sum())


def associate(
    fr: Fragments,
    pairs: np.ndarray,
    prob: np.ndarray,
    n_frames_of,
    cfg: LinkingConfig,
    accepted: list | None = None,
) -> np.ndarray:
    """Greedy union of fragments, most probable pair first, unless the two
    identities would share frames or run in parallel at other frequencies.
    Returns the root fragment of every fragment; appends the accepted pair
    indices to `accepted` if given."""
    parent = np.arange(len(fr))
    frames = [n_frames_of(i) for i in range(len(fr))]
    tt = list(fr.t)
    ff = list(fr.f)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    order = np.argsort(-prob, kind="stable")
    for k in order:
        if prob[k] < cfg.threshold:
            break
        a, b = pairs[k]
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        fa, fb = frames[ra], frames[rb]
        shared = len(np.intersect1d(fa, fb, assume_unique=True))
        if shared > cfg.max_shared * min(len(fa), len(fb)):
            continue
        n_par, n_near = parallel_evidence(
            tt[ra], ff[ra], tt[rb], ff[rb], cfg.parallel_dt, cfg.parallel_df
        )
        if n_par >= 3 and n_par > cfg.max_parallel * n_near:
            continue
        if len(fa) < len(fb):
            ra, rb = rb, ra
        parent[rb] = ra
        if accepted is not None:
            accepted.append(int(k))
        frames[ra] = np.union1d(frames[ra], frames[rb])
        frames[rb] = None
        t = np.concatenate([tt[ra], tt[rb]])
        o = np.argsort(t, kind="stable")
        tt[ra], ff[ra] = t[o], np.concatenate([ff[ra], ff[rb]])[o]
        tt[rb] = ff[rb] = None
    return np.array([find(i) for i in range(len(fr))])


def link(
    fund_v: np.ndarray,
    idx_v: np.ndarray,
    sign_v: np.ndarray,
    cplx_v: np.ndarray | None,
    ident_v: np.ndarray,
    times: np.ndarray,
    cfg: LinkingConfig | None = None,
    model: LinkModel | None = None,
) -> np.ndarray:
    """Identities with the tracker's fragments joined (see module doc)."""
    from .stitching import resolve_duplicates

    cfg = cfg or LinkingConfig()
    if np.all(np.isnan(ident_v)):
        return ident_v.copy()
    model = model or LinkModel.load(cfg.model)
    fr = fragments(fund_v, idx_v, sign_v, cplx_v, ident_v, times, cfg)
    if len(fr) < 2:
        return ident_v.copy()
    valid = np.isfinite(ident_v)
    dets = _Detections(times[idx_v[valid]], fund_v[valid], ident_v[valid])
    pairs = candidates(fr, cfg)
    if len(pairs) == 0:
        return ident_v.copy()
    X = features(fr, pairs, dets, cfg)
    if hasattr(model, "bind"):  # an oracle (benchmarks/dense/linker.py)
        model.bind(fr)
        prob = model.prob(X, pairs)
    else:
        prob = model.prob(X)
    roots = associate(fr, pairs, prob, lambda i: np.unique(idx_v[fr.members[i]]), cfg)
    out = ident_v.copy()
    for i, r in enumerate(roots):
        if r != i:
            out[fr.members[i]] = fr.ident[r]
    return resolve_duplicates(fund_v, idx_v, out)
