"""Separating harmonics from fish by co-modulation, on tube mixtures.

Labels the identities of a ``tube_mixture.py`` mixture from its ground truth
(fundamental or h-th harmonic of fish k, or other), scores all overlapping
identity pairs with `wavetracker.comodulation.score_pairs` for several
timescales, and reports how well each feature separates harmonic pairs (same
fish) from pairs of different fish of the same source (shared temperature)
and of different sources.

Usage::

    python benchmarks/harmonic_comodulation.py output/dev/mix0 --timescales 2 5 10 30 120
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from wavetracker.comodulation import ComodulationConfig, score_pairs
from wavetracker.results import Results

FEATURES = ["offset_abs", "freq_corr", "freq_explained", "amp_corr", "pattern"]


def label_identities(res: Results, truth, min_share: float = 0.6, max_h: int = 5):
    """Identity -> (fish, harmonic), (-1, 0) if not explained by the truth."""
    t = res.times[res.idx_v]
    F = np.stack([np.interp(t, truth["times"], f) for f in truth["freqs"]])
    best = np.full(len(t), -1)
    best_h = np.zeros(len(t), int)
    err = np.full(len(t), np.inf)
    for h in range(1, max_h + 1):
        e = np.abs(res.fund_v[None] - h * F) / h  # in fundamental units
        k = np.nanargmin(np.where(np.isnan(e), np.inf, e), axis=0)
        ek = e[k, np.arange(len(k))]
        better = (ek < 0.5) & (ek < err)
        best[better], best_h[better], err[better] = k[better], h, ek[better]
    labels = {}
    for i in np.unique(res.ident_v[np.isfinite(res.ident_v)]):
        m = res.ident_v == i
        code = best[m] * 10 + best_h[m]
        vals, counts = np.unique(code, return_counts=True)
        j = counts.argmax()
        if vals[j] >= 0 and counts[j] >= min_share * m.sum():
            labels[float(i)] = (int(vals[j] // 10), int(vals[j] % 10))
        else:
            labels[float(i)] = (-1, 0)
    return labels


def pair_classes(pairs: pd.DataFrame, labels, source) -> pd.Series:
    out = []
    for lo, hi in zip(pairs.low, pairs.high, strict=True):
        (kl, hl), (kh, hh) = labels[lo], labels[hi]
        if kl < 0 or kh < 0:
            out.append("other")
        elif kl == kh:
            out.append("harmonic" if hh % hl == 0 and hh > hl else "same fish")
        elif source[kl] == source[kh]:
            out.append("same tank")
        else:
            out.append("different tank")
    return pd.Series(out, index=pairs.index)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    allv = np.r_[pos, neg]
    ranks = pd.Series(allv).rank().to_numpy()
    return float(
        (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
        / (len(pos) * len(neg))
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mixtures", nargs="+")
    ap.add_argument("--timescales", type=float, nargs="+", default=[2, 5, 10, 30, 120])
    ap.add_argument("--min-overlap", type=float, default=10.0)
    ap.add_argument("--results", default="wt", help="results subdirectory")
    args = ap.parse_args()

    tables = []
    for d in map(Path, args.mixtures):
        res = Results.load(d / args.results)
        truth = np.load(d / "truth.npz")
        labels = label_identities(res, truth)
        n_lab = pd.Series([lab for lab in labels.values()]).value_counts()
        harm_ids = sum(1 for k, h in labels.values() if k >= 0 and h > 1)
        print(
            f"{d.name}: {len(labels)} identities, {harm_ids} are harmonics of a fish, "
            f"{sum(1 for k, _ in labels.values() if k < 0)} other"
        )
        del n_lab
        for ts in args.timescales:
            cfg = ComodulationConfig(
                timescale=ts, min_overlap=args.min_overlap, max_offset=None
            )
            p = score_pairs(res, cfg)
            p["class"] = pair_classes(p, labels, truth["source"])
            p["timescale"] = ts
            p["mixture"] = d.name
            p["results"] = args.results
            tables.append(p)
    p = pd.concat(tables, ignore_index=True)
    p["offset_abs"] = -p.offset.abs()  # higher = more harmonic-like
    out = Path(args.mixtures[0]).parent / f"comodulation_pairs_{args.results}.csv"
    p.to_csv(out, index=False)
    print("pairs per class (first timescale):")
    print(p[p.timescale == args.timescales[0]]["class"].value_counts().to_string())
    for ts in args.timescales:
        q = p[p.timescale == ts]
        pos = q[q["class"] == "harmonic"]
        line = [f"timescale {ts:6.1f} s:"]
        for neg_name in ("same tank", "different tank"):
            neg = q[q["class"] == neg_name]
            line.append(
                f"  vs {neg_name}: "
                + " ".join(
                    f"{f} {auc(pos[f].to_numpy(), neg[f].to_numpy()):.2f}"
                    for f in FEATURES
                )
            )
        print("\n".join(line))
    print("wrote", out)


if __name__ == "__main__":
    main()
