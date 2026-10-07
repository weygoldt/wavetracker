"""Identity-level evaluation of wavetracker.comodulation.find_harmonics on
tube mixtures: flagged identities that are / are not harmonics of the fish of
their partner, for several evidence rules.

Usage::

    python benchmarks/harmonic_rule.py output/dev/mix0 output/dev/mix1 --results wt2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from harmonic_comodulation import label_identities

from wavetracker.comodulation import ComodulationConfig, classify, score_pairs
from wavetracker.results import Results

RULES = {
    "frequency": {},
    "frequency or amplitude": dict(min_amp_corr=0.5),
    "frequency or amplitude or pattern": dict(min_amp_corr=0.5, min_pattern=0.98),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mixtures", nargs="+")
    ap.add_argument("--results", default="wt2")
    ap.add_argument("--timescale", type=float, default=30.0)
    ap.add_argument("--min-overlap", type=float, default=10.0)
    args = ap.parse_args()
    totals = {r: np.zeros(3, int) for r in RULES}
    n_dup = 0
    for d in map(Path, args.mixtures):
        res = Results.load(d / args.results)
        labels = label_identities(res, np.load(d / "truth.npz"))
        base = ComodulationConfig(
            timescale=args.timescale, min_overlap=args.min_overlap
        )
        pairs = score_pairs(res, base)
        # harmonic identities that overlap one of their fish's fundamental
        # identities long enough to be found at all
        dup = set()
        for lo, hi in zip(pairs.low, pairs.high, strict=True):
            (kl, hl), (kh, hh) = labels[lo], labels[hi]
            if kl >= 0 and kl == kh and hl == 1 and hh > 1:
                dup.add(hi)
        n_dup += len(dup)
        for name, kw in RULES.items():
            cfg = ComodulationConfig(
                timescale=args.timescale, min_overlap=args.min_overlap, **kw
            )
            hits = classify(pairs, cfg)
            hits = hits[hits.harmonic].drop_duplicates("high")
            tp = sum(
                1
                for lo, hi in zip(hits.low, hits.high, strict=True)
                if labels[hi][0] >= 0
                and labels[hi][1] > 1
                and labels[lo][0] == labels[hi][0]
            )
            fish = sum(
                1 for hi in hits.high if labels[hi][0] >= 0 and labels[hi][1] == 1
            )
            totals[name] += (tp, len(hits) - tp, fish)
    print(
        f"timescale {args.timescale} s, min overlap {args.min_overlap} s: "
        f"{n_dup} harmonic identities overlap their fundamental (candidates)"
    )
    for name, (tp, fp, fish) in totals.items():
        print(
            f"  {name:36s}: flagged {tp + fp}, correct {tp} ({tp / max(n_dup, 1):.0%} of candidates), "
            f"wrong {fp} (of which real fish {fish})"
        )


if __name__ == "__main__":
    main()
