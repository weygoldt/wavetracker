"""Harmonics tracked as fish, found by co-modulation of identities.

A harmonic follows its fundamental exactly: f_h(t) = h * f_1(t) at every
moment, including drifts, rises and chirps, and it has the fundamental's
electrode pattern and amplitude changes. Two different fish near an integer
frequency ratio may also drift together on slow timescales, e.g. when
temperature changes all fish frequencies by the same factor (Q10), but they
have an offset (f_2 = h * f_1 + c) and independent fast modulations.

For every pair of identities that overlap in time, with h = round(f_hi / f_lo),
the following are computed on their common frames:

* ``offset``: median of f_hi - h * f_lo [Hz] (0 for a harmonic);
* ``freq_corr`` / ``freq_explained``: correlation of the fast frequency
  modulations (each trace minus its running median over `timescale`
  seconds) of h * f_lo and f_hi, and the fraction of the variance of f_hi's
  fast modulation explained by h * f_lo's (1 - var(y - x) / var(y));
* ``amp_corr``: correlation of the fast modulations of the log power;
* ``pattern``: cosine similarity of the electrode amplitude patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import pandas as pd

from .results import Results


@dataclass
class ComodulationConfig:
    timescale: float = 30.0
    """Window of the running median removed from frequency and amplitude
    traces [s]: only faster modulations are compared. Shorter windows leave
    mostly estimation noise (resting fish), longer ones let slow drifts
    shared by all fish (temperature) correlate different fish. 30 s was best
    on tube mixtures (benchmarks/harmonic_comodulation.py)."""
    min_overlap: float = 10.0
    """Minimum common time of two identities [s]."""
    max_harmonic: int = 5
    """Highest harmonic number considered."""
    max_offset: float | None = 3.0
    """Pairs whose |median(f_hi - h f_lo)| exceeds this are not scored [Hz];
    None: all overlapping pairs (for benchmarks)."""
    harmonic_offset: float = 0.25
    """A harmonic is at most this far from h * f_lo [Hz]."""
    min_freq_corr: float = 0.5
    """Fast frequency co-modulation that identifies a harmonic ..."""
    min_amp_corr: float | None = None
    """... or fast amplitude co-modulation (None: not used; with moving
    electrodes nearby fish share amplitude changes) ..."""
    min_pattern: float | None = None
    """... or electrode pattern similarity (None: not used; useful with many
    electrodes, e.g. 0.98 for a grid)."""


def identity_traces(res: Results) -> dict[float, pd.DataFrame]:
    """Per identity, one row per frame (the strongest detection): frame, time,
    freq, log power and the electrode amplitudes (sqrt of power)."""
    ok = np.isfinite(res.ident_v)
    total = res.sign_v.sum(1)
    order = np.lexsort((-total, res.idx_v, res.ident_v))
    order = order[ok[order]]
    ident, frame = res.ident_v[order], res.idx_v[order]
    first = np.r_[True, (np.diff(ident) != 0) | (np.diff(frame) != 0)]
    order = order[first]
    out = {}
    ids = res.ident_v[order]
    bounds = np.flatnonzero(np.r_[True, np.diff(ids) != 0, True])
    for a, b in pairwise(bounds):
        o = order[a:b]
        out[float(ids[a])] = pd.DataFrame(
            {
                "frame": res.idx_v[o],
                "time": res.times[res.idx_v[o]],
                "freq": res.fund_v[o],
                "logpow": np.log10(total[o] + 1e-30),
            }
        ).assign(
            **{f"a{c}": np.sqrt(res.sign_v[o, c]) for c in range(res.sign_v.shape[1])}
        )
    return out


def _highpass(time: np.ndarray, x: np.ndarray, timescale: float) -> np.ndarray:
    s = pd.Series(x, index=pd.to_timedelta(time, unit="s"))
    trend = s.rolling(
        pd.Timedelta(seconds=timescale), center=True, min_periods=1
    ).median()
    return x - trend.to_numpy()


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    sx, sy = x.std(), y.std()
    if sx == 0 or sy == 0:
        return np.nan
    return float(np.mean((x - x.mean()) * (y - y.mean())) / (sx * sy))


def score_pairs(res: Results, cfg: ComodulationConfig | None = None) -> pd.DataFrame:
    """Co-modulation features of all overlapping identity pairs near an
    integer frequency ratio (see module docstring), one row per pair."""
    cfg = cfg or ComodulationConfig()
    traces = identity_traces(res)
    dt = float(np.median(np.diff(res.times)))
    min_frames = int(cfg.min_overlap / dt)
    ids = [i for i, tr in traces.items() if len(tr) >= min_frames]
    span = {i: (traces[i].frame.iloc[0], traces[i].frame.iloc[-1]) for i in ids}
    med = {i: float(traces[i].freq.median()) for i in ids}
    chans = (
        [c for c in next(iter(traces.values())).columns if c.startswith("a")]
        if traces
        else []
    )
    rows = []
    for lo in ids:
        for hi in ids:
            if med[hi] <= med[lo] * 1.5:
                continue
            h = round(med[hi] / med[lo])
            if h < 2 or h > cfg.max_harmonic:
                continue
            if span[hi][0] > span[lo][1] or span[lo][0] > span[hi][1]:
                continue
            m = traces[lo].merge(
                traces[hi], on=["frame", "time"], suffixes=("_lo", "_hi")
            )
            if len(m) < min_frames:
                continue
            x = h * m.freq_lo.to_numpy()
            y = m.freq_hi.to_numpy()
            offset = float(np.median(y - x))
            if cfg.max_offset is not None and abs(offset) > cfg.max_offset:
                continue
            t = m.time.to_numpy()
            xf, yf = _highpass(t, x, cfg.timescale), _highpass(t, y, cfg.timescale)
            xa = _highpass(t, m.logpow_lo.to_numpy(), cfg.timescale)
            ya = _highpass(t, m.logpow_hi.to_numpy(), cfg.timescale)
            p_lo = m[[f"{c}_lo" for c in chans]].to_numpy()
            p_hi = m[[f"{c}_hi" for c in chans]].to_numpy()
            cos = (p_lo * p_hi).sum(1) / (
                np.linalg.norm(p_lo, axis=1) * np.linalg.norm(p_hi, axis=1) + 1e-30
            )
            vy = yf.var()
            rows.append(
                {
                    "low": lo,
                    "high": hi,
                    "h": h,
                    "overlap": len(m) * dt,
                    "f_low": med[lo],
                    "f_high": med[hi],
                    "offset": offset,
                    "residual_mad": float(np.median(np.abs(y - x - offset))),
                    "fast_std_low": float((xf / h).std()),
                    "fast_std_high": float(yf.std()),
                    "freq_corr": _corr(xf, yf),
                    "freq_explained": float(1 - (yf - xf).var() / vy)
                    if vy > 0
                    else np.nan,
                    "amp_corr": _corr(xa, ya),
                    "pattern": float(np.median(cos)),
                }
            )
    return pd.DataFrame(rows)


def classify(
    pairs: pd.DataFrame, cfg: ComodulationConfig | None = None
) -> pd.DataFrame:
    """Add a boolean column ``harmonic`` and the ``evidence`` for it."""
    cfg = cfg or ComodulationConfig()
    pairs = pairs.copy()
    if pairs.empty:
        pairs["harmonic"] = pd.Series(dtype=bool)
        pairs["evidence"] = pd.Series(dtype=str)
        return pairs
    close = pairs.offset.abs() <= cfg.harmonic_offset
    tests = {"frequency": pairs.freq_corr >= cfg.min_freq_corr}
    if cfg.min_amp_corr is not None:
        tests["amplitude"] = pairs.amp_corr >= cfg.min_amp_corr
    if cfg.min_pattern is not None:
        tests["pattern"] = pairs.pattern >= cfg.min_pattern
    evidence = pd.DataFrame(tests)
    pairs["harmonic"] = close & evidence.any(axis=1)
    pairs["evidence"] = [
        "+".join(c for c in evidence.columns if row[c]) if ok else ""
        for (_, row), ok in zip(evidence.iterrows(), pairs.harmonic, strict=True)
    ]
    return pairs


def find_harmonics(res: Results, cfg: ComodulationConfig | None = None) -> pd.DataFrame:
    """Identities that are harmonics of another identity: one row per such
    identity (``high``) with its fundamental (``low``) and the evidence."""
    cfg = cfg or ComodulationConfig()
    pairs = classify(score_pairs(res, cfg), cfg)
    hits = pairs[pairs.harmonic]
    if hits.empty:
        return hits
    score = hits[["freq_corr", "amp_corr"]].max(axis=1)
    best = hits.assign(score=score).sort_values("score", ascending=False)
    return best.drop_duplicates("high").drop(columns="score").reset_index(drop=True)
