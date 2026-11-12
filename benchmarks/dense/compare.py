"""Paired comparison of two runs on the dense mixtures: the mean difference of
a score (b - a), a bootstrap 95 % CI over mixtures, and how many mixtures are
better or worse (lab and field variants counted apart); overall, per split,
lab / field, and per scenario group (``--by-scenario``: per scenario).

A run is given by its tag, or by tags per split when the same configuration
ran under different tags (``tag:split`` entries separated by commas; an entry
for ``ldev`` also covers ``ldev-field`` unless that has its own, a bare tag
covers the rest)::

    python benchmarks/dense/compare.py --a classic-ldev --b ref-learned-b --splits ldev2 ldev2-field
    python benchmarks/dense/compare.py --a classic-ldev \\
        --b B-aug-s0-t0.6:ldev,ref-learned-b:ldev2 \\
        --splits ldev ldev-field ldev2 ldev2-field --plot

Mixtures are paired by name, and the lab and field variants of a mixture (the
same fish) are resampled together: the bootstrap unit is the mixture, not the
variant, since the two are far from independent. ``--stratify`` resamples
mixtures within their scenario (the designs are fixed, only the seeds vary);
a scenario with a single mixture then adds no spread, so use it only where
every scenario has several (ldev + ldev2: 3 or 6).

``--truth rises`` compares the scores on the truth that follows the steep
part of rises (``score_rises.json``, see ``run.py rescore --truth rises``);
its rise-onset stratum is ``--metric recall_rise`` or ``idf1_coverage_rise``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA
from score import TRUTHS, score_file
from strata import GROUPS

RUNS = DATA / "runs"
FIG = DATA / "figures"
# scores for which smaller is better (counts "better" the other way round)
LOWER_IS_BETTER = {
    "false_per_frame",
    "harmonic_per_frame",
    "false_ids_30s",
    "switches_per_fish_min",
    "ids",
    "ids_30s",
}


def parse_spec(spec: str) -> dict[str | None, str]:
    """``tag`` or ``tag:split,tag:split,...`` -> {split or None: tag}."""
    out: dict[str | None, str] = {}
    for part in spec.split(","):
        tag, _, split = part.strip().partition(":")
        key = split or None
        if key in out:
            raise ValueError(f"{spec}: two tags for {split or 'the rest'}")
        out[key] = tag
    return out


def tag_for(spec: dict[str | None, str], split: str) -> str:
    """The tag of a split: its own entry, that of its lab split, the rest."""
    for key in (split, split.removesuffix("-field"), None):
        if key in spec:
            return spec[key]
    raise ValueError(f"no tag for split {split} in {spec}")


def load(
    spec: dict, splits: list[str], metric: str, truth: str = "default"
) -> dict[str, float]:
    """{split/mixture: score} of the runs a spec names."""
    out = {}
    for sp in splits:
        tag = tag_for(spec, sp)
        files = sorted((RUNS / tag / sp).glob(f"*/{score_file(truth)}"))
        if not files:
            print(f"warning: no runs {tag}/{sp}", file=sys.stderr)
        for f in files:
            out[f"{sp}/{f.parent.name}"] = float(json.loads(f.read_text())[metric])
    return out


def scenario(name: str) -> str:
    return name.split("/")[1].rsplit("-s", 1)[0]


def group(name: str) -> str:
    sc = name.split("/")[1]
    return next((g for g, pre in GROUPS.items() if sc.startswith(pre)), "other")


def unit(name: str) -> str:
    """The mixture a variant belongs to: lab and field share it."""
    sp, rest = name.split("/", 1)
    return f"{sp.removesuffix('-field')}/{rest}"


def bootstrap_ci(
    d: np.ndarray, units: np.ndarray, n_boot: int, rng, strata=None
) -> tuple:
    """Percentile 95 % CI of the mean of `d`, resampling units (mixtures)
    with replacement, within their stratum if `strata` (per element) is
    given; every resampled unit brings all of its variants."""
    u, first, inv = np.unique(units, return_index=True, return_inverse=True)
    s = np.bincount(inv, weights=d, minlength=len(u))
    c = np.bincount(inv, minlength=len(u)).astype(float)
    if len(u) < 2:
        return np.nan, np.nan
    st = np.zeros(len(u), int) if strata is None else np.asarray(strata)[first]
    tot_s, tot_c = np.zeros(n_boot), np.zeros(n_boot)
    for k in np.unique(st):
        members = np.flatnonzero(st == k)
        idx = members[rng.integers(len(members), size=(n_boot, len(members)))]
        tot_s += s[idx].sum(1)
        tot_c += c[idx].sum(1)
    lo, hi = np.percentile(tot_s / tot_c, [2.5, 97.5])
    return float(lo), float(hi)


def summarise(names, a, b, metric, n_boot, tie, stratify=False, seed=0) -> dict:
    da = np.array([a[n] for n in names])
    db = np.array([b[n] for n in names])
    d = db - da
    units = np.array([unit(n) for n in names])
    strata = [scenario(n) for n in names] if stratify else None
    rng = np.random.default_rng(seed)
    lo, hi = bootstrap_ci(d, units, n_boot, rng, strata)
    sign = -1.0 if metric in LOWER_IS_BETTER else 1.0
    return {
        "mixtures": len(set(units)),
        "variants": len(names),
        "a": float(da.mean()),
        "b": float(db.mean()),
        "diff": float(d.mean()),
        "ci_lo": lo,
        "ci_hi": hi,
        "ci_width": hi - lo,
        "better": int((sign * d > tie).sum()),
        "worse": int((sign * d < -tie).sum()),
        "tied": int((np.abs(d) <= tie).sum()),
    }


def compare(args) -> list[tuple[str, dict]]:
    spec_a, spec_b = parse_spec(args.a), parse_spec(args.b)
    a = load(spec_a, args.splits, args.metric, args.truth)
    b = load(spec_b, args.splits, args.metric, args.truth)
    names = sorted(set(a) & set(b))
    for n in sorted(set(a) ^ set(b)):
        side = "a" if n in a else "b"
        print(f"warning: {n} only in {side}", file=sys.stderr)
    undefined = [n for n in names if not (np.isfinite(a[n]) and np.isfinite(b[n]))]
    if undefined:  # e.g. recall_rise of a mixture without rise frames
        print(f"{len(undefined)} variants without a {args.metric}: left out")
        names = [n for n in names if n not in undefined]
    if not names:
        raise SystemExit("no mixtures in both runs")
    rows = [("all", names)]
    rows += [(sp, [n for n in names if n.startswith(sp + "/")]) for sp in args.splits]
    labs = [n for n in names if not n.split("/")[0].endswith("-field")]
    fields = [n for n in names if n.split("/")[0].endswith("-field")]
    if labs and fields:
        rows += [("lab", labs), ("field", fields)]
    rows += [(f"group {g}", [n for n in names if group(n) == g]) for g in GROUPS]
    if args.by_scenario:
        scen = sorted({scenario(n) for n in names})
        rows += [(sc, [n for n in names if scenario(n) == sc]) for sc in scen]
    out = []
    for label, sel in rows:
        if sel:
            r = summarise(sel, a, b, args.metric, args.n_boot, args.tie, args.stratify)
            out.append((label, r))
    how = "within scenarios" if args.stratify else "over mixtures"
    print(f"{args.metric}: a = {args.a}, b = {args.b}; b - a, 95 % CI {how}")
    print(
        f"{'':18s} {'mix':>4s} {'var':>4s} {'a':>7s} {'b':>7s} {'b - a':>8s} "
        f"{'95 % CI':>19s} {'width':>7s} {'better':>6s} {'worse':>6s}"
    )
    for label, r in out:
        print(
            f"{label:18s} {r['mixtures']:4d} {r['variants']:4d} {r['a']:7.4f} "
            f"{r['b']:7.4f} {r['diff']:+8.4f} [{r['ci_lo']:+8.4f}, {r['ci_hi']:+8.4f}] "
            f"{r['ci_width']:7.4f} {r['better']:6d} {r['worse']:6d}"
        )
    if args.plot:
        plot(args, names, a, b, out)
    return out


def plot(args, names, a, b, rows):
    import matplotlib.pyplot as plt

    scen = sorted({scenario(n) for n in names})
    fig, ax = plt.subplots(figsize=(7.5, 0.33 * (len(scen) + 4) + 1.2))
    y = {sc: i for i, sc in enumerate(reversed(scen))}
    colors = {"lab": "#2a78d6", "field": "#eb6834"}
    for n in names:
        kind = "field" if n.split("/")[0].endswith("-field") else "lab"
        jitter = 0.12 if kind == "field" else -0.12
        ax.plot(
            b[n] - a[n],
            y[scenario(n)] + jitter,
            "o",
            ms=4,
            color=colors[kind],
            alpha=0.75,
            mec="none",
        )
    top = len(scen) - 0.6
    show = [r for r in rows if r[0] in ("all", "lab", "field")]
    for k, (label, r) in enumerate(show):
        yy = top + 0.8 * (len(show) - k)
        ax.plot([r["ci_lo"], r["ci_hi"]], [yy, yy], color="#0b0b0b", lw=2)
        ax.plot(r["diff"], yy, "D", color="#0b0b0b", ms=5)
        y[f"{label}: {r['diff']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"] = yy
    ax.axvline(0, color="#52514e", lw=0.8)
    ax.set_yticks(list(y.values()), list(y.keys()), fontsize=8)
    ax.set_xlabel(f"{args.metric}: b - a (blue lab, orange field)", fontsize=8)
    how = "within scenarios" if args.stratify else "over mixtures"
    ax.set_title(
        f"a = {args.a}\nb = {args.b}\nmean and 95 % CI {how}", loc="left", fontsize=8
    )
    ax.tick_params(labelsize=8)
    ax.grid(axis="x", color="#e6e5e0", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    stem = args.out or re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        f"compare_{args.metric}_{args.a}_vs_{args.b}_{'-'.join(args.splits)}"
        + ("" if args.truth == "default" else f"_{args.truth}"),
    )
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / f"{stem}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--a", required=True, help="reference: tag or tag:split,...")
    p.add_argument("--b", required=True, help="candidate: tag or tag:split,...")
    p.add_argument(
        "--splits", nargs="+", default=["ldev", "ldev-field", "ldev2", "ldev2-field"]
    )
    p.add_argument("--metric", default="idf1", help="a key of score.json")
    p.add_argument(
        "--truth",
        default="default",
        choices=sorted(TRUTHS),
        help="scores on this truth (rises: score_rises.json)",
    )
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--tie", type=float, default=1e-9, help="|b - a| counted as tie")
    p.add_argument("--by-scenario", action="store_true", help="rows per scenario")
    p.add_argument("--stratify", action="store_true", help="resample within scenarios")
    p.add_argument("--plot", action="store_true", help="figure in figures/")
    p.add_argument("--out", help="figure stem")
    args = p.parse_args(argv)
    compare(args)


if __name__ == "__main__":
    main()
