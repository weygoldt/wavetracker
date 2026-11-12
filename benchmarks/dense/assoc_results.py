"""Step-B association candidates on ldev / ldev-field: IDF1 change against the
baseline, learned v1 detections and truth detections, one dot per mixture
and the mean.

    python benchmarks/dense/assoc_results.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from run import FIG, RUNS

# (label, learned run, truth-detection run)
ROWS = [
    ("baseline", "B-assoc-base", "B-assoc-base-oracle"),
    ("association rules", "B-assoc-l3-par", "B-assoc-OC-oracle"),
    ("+ identity salience >= 0.75", "B-assoc-C1", None),
    ("motion 2 s", "B-assoc-msh2", "B-assoc-msh2-oracle"),
    ("motion 2 s + retrained linker", "B-assoc-m2L-t0.6", None),
    ("all (truth: motion + rules)", "B-assoc-C2", "B-assoc-OC2-oracle"),
]
SPLITS = ("ldev", "ldev-field")
COLS = [
    ("learned v1, lab", 1, "ldev"),
    ("learned v1, field", 1, "ldev-field"),
    ("truth detections, lab", 2, "ldev"),
    ("truth detections, field", 2, "ldev-field"),
]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def per_mixture(tag: str, split: str) -> dict[str, float]:
    out = {}
    for p in (RUNS / tag / split).glob("*/score.json"):
        r = json.loads(p.read_text())
        out[r["mixture"]] = r["idf1"]
    return out


def main():
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 4, figsize=(16, 4.2), sharey=True, layout="constrained")
    y = np.arange(len(ROWS))[::-1]
    for ax, (title, col, split) in zip(axs, COLS, strict=True):
        base = per_mixture(ROWS[0][col], split)
        for yi, row in zip(y, ROWS, strict=True):
            tag = row[col]
            if tag is None:
                continue
            r = per_mixture(tag, split)
            keys = sorted(set(r) & set(base))
            if len(keys) < len(base):
                continue
            d = np.array([r[k] - base[k] for k in keys])
            jitter = np.linspace(-0.18, 0.18, len(d))
            ax.scatter(d, yi + jitter, s=10, color=BLUE, alpha=0.45, lw=0)
            ax.plot(d.mean(), yi, "D", ms=7, color=ORANGE, mec="#fcfcfb", mew=1.2)
            ax.text(
                0.99,
                yi + 0.28,
                f"{np.mean([r[k] for k in keys]):.3f} ({d.mean():+.3f})",
                transform=ax.get_yaxis_transform(),
                ha="right",
                fontsize=7.5,
                color=MUTED,
            )
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.set_xlabel("IDF1 change against the baseline", fontsize=8, color=MUTED)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8, colors=MUTED)
    axs[0].set_yticks(y, [r[0] for r in ROWS], fontsize=9)
    fig.suptitle(
        "association candidates on ldev (dots: mixtures, diamond: mean; "
        "text: mean IDF1 and change)",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    path = FIG / "assoc_results.png"
    fig.savefig(path, dpi=140)
    print(path)


if __name__ == "__main__":
    main()
