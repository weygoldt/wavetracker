"""Compare runs per failure mode: recall by SNR band and by distance to the
nearest other line, plus the identity scores.

    python benchmarks/dense/strata.py --tags classic-ldev learned-v1 --splits ldev ldev-field
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA

RUNS = DATA / "runs"
FIG = DATA / "figures"
SNR_BANDS = [
    (-99, 0, "< 0 dB"),
    (0, 6, "0-6 dB"),
    (6, 20, "6-20 dB"),
    (20, 99, "≥ 20 dB"),
]
NN_BANDS = [
    (0, 1.2, "< 1.2 Hz"),
    (1.2, 3, "1.2-3 Hz"),
    (3, 10, "3-10 Hz"),
    (10, 1e9, "≥ 10 Hz"),
]
SCORES = [
    "idf1",
    "purity",
    "recall",
    "precision",
    "false_per_frame",
    "harmonic_per_frame",
    "false_ids_30s",
    "switches_per_fish_min",
    "ids_30s",
]
GROUPS = {  # IDF1 by scenario group
    "crowded": ("crowded",),
    "density": ("density",),
    "atten": ("atten", "ladder"),
    "lowfreq": ("lowfreq", "lowonly"),
}


def frames(tag: str, split: str) -> dict[str, np.ndarray]:
    parts = [
        dict(np.load(p)) for p in sorted((RUNS / tag / split).glob("*/frames.npz"))
    ]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]} if parts else {}


def scores(tag: str, split: str) -> dict[str, float]:
    rows = [
        json.loads(p.read_text())
        for p in sorted((RUNS / tag / split).glob("*/score.json"))
    ]
    out = {k: float(np.mean([r[k] for r in rows])) for k in SCORES} | {"n": len(rows)}
    for g, prefixes in GROUPS.items():
        v = [r["idf1"] for r in rows if r["mixture"].split("/")[1].startswith(prefixes)]
        out[f"idf1 {g}"] = float(np.mean(v)) if v else np.nan
    return out


def strata(fr: dict) -> dict[str, float]:
    out = {}
    for lo, hi, name in SNR_BANDS:
        m = (fr["snr_db"] >= lo) & (fr["snr_db"] < hi)
        out[f"snr {name}"] = float(fr["matched"][m].mean()) if m.any() else np.nan
    for lo, hi, name in NN_BANDS:
        m = (fr["nn_hz"] >= lo) & (fr["nn_hz"] < hi) & (fr["snr_db"] >= 6)
        out[f"nn {name}"] = float(fr["matched"][m].mean()) if m.any() else np.nan
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--splits", nargs="+", default=["ldev", "ldev-field"])
    p.add_argument("--out", default=None, help="figure stem (default: strata_<tags>)")
    args = p.parse_args(argv)
    table = {}
    for sp in args.splits:
        for t in args.tags:
            fr = frames(t, sp)
            if not fr:
                print(f"{t}/{sp}: no runs")
                continue
            table[(t, sp)] = scores(t, sp) | strata(fr)
    keys = (
        SCORES
        + [f"idf1 {g}" for g in GROUPS]
        + [f"snr {b[2]}" for b in SNR_BANDS]
        + [f"nn {b[2]}" for b in NN_BANDS]
    )
    print(
        "recall by SNR band (all frames) and by nearest-line distance (frames with SNR ≥ 6 dB)"
    )
    w = max([len(t + " / " + sp) for t, sp in table] + [28]) + 1
    print(f"{'':{w}s}" + "".join(f"{k[:13]:>14s}" for k in keys))
    for (t, sp), row in table.items():
        print(f"{t + ' / ' + sp:{w}s}" + "".join(f"{row[k]:14.4f}" for k in keys))
    stem = args.out or "strata_" + "_".join(args.tags)
    (FIG / f"{stem}.json").write_text(
        json.dumps({f"{t}/{sp}": r for (t, sp), r in table.items()}, indent=1)
    )

    import matplotlib.pyplot as plt

    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
    groups = [
        ("snr", SNR_BANDS, "recall by SNR"),
        ("nn", NN_BANDS, "recall by distance to nearest line (SNR ≥ 6 dB)"),
    ]
    fig, axs = plt.subplots(
        len(args.splits),
        3,
        figsize=(16, 3.6 * len(args.splits)),
        squeeze=False,
        layout="constrained",
        gridspec_kw={"width_ratios": [4, 4, 3]},
    )
    for si, sp in enumerate(args.splits):
        tags = [t for t in args.tags if (t, sp) in table]
        w = 0.8 / max(len(tags), 1)
        for gi, (pre, bands, title) in enumerate(groups):
            ax = axs[si, gi]
            for ti, t in enumerate(tags):
                v = [table[(t, sp)][f"{pre} {b[2]}"] for b in bands]
                ax.bar(
                    np.arange(len(bands)) + (ti - (len(tags) - 1) / 2) * w,
                    v,
                    w * 0.92,
                    color=colors[ti % len(colors)],
                    label=t,
                )
            ax.set_xticks(range(len(bands)), [b[2] for b in bands])
            ax.set_ylim(0, 1)
            ax.set_title(f"{sp}: {title}", fontsize=10, loc="left")
            ax.grid(axis="y", alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)
        ax = axs[si, 2]
        metrics = ["idf1", "recall", "precision"]
        for ti, t in enumerate(tags):
            v = [table[(t, sp)][m] for m in metrics]
            ax.bar(
                np.arange(len(metrics)) + (ti - (len(tags) - 1) / 2) * w,
                v,
                w * 0.92,
                color=colors[ti % len(colors)],
                label=t,
            )
        ax.set_xticks(range(len(metrics)), ["IDF1", "recall", "precision"])
        ax.set_ylim(0, 1)
        ax.set_title(
            f"{sp}: identity scores (mean of mixtures)", fontsize=10, loc="left"
        )
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        axs[si, 0].legend(frameon=False, fontsize=8, loc="upper left")
    path = FIG / f"{stem}.png"
    fig.savefig(path, dpi=130)
    print(path)


if __name__ == "__main__":
    main()
