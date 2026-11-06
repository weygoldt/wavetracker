"""Frequency resolution and faint peaks on the dev suite: recall, IDF1,
precision and purity per setting (docs/benchmarks.md, completeness).

    python benchmarks/dense/completeness.py [figure.png]
"""

import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from run import RUNS

R = str(RUNS)
OUT = sys.argv[1] if len(sys.argv) > 1 else "docs/images/completeness_dev.png"
ROWS = [  # (tag, label)
    ("dense", "0.61 Hz, stitching"),
    ("dense-linked", "0.61 Hz, linker (now)"),
    ("proto-fill-v1a", "0.61 Hz, linker + faint peaks"),
    ("dense-nfft16", "0.31 Hz, stitching"),
    ("dense-nfft16-linked", "0.31 Hz, linker (trained at 0.61/1.22 Hz)"),
    ("dense-nfft16-linked-v5", "0.31 Hz, linker retrained"),
    ("proto-fill16v5-v1a", "0.31 Hz, retrained linker + faint peaks"),
]
ROWS = [r for r in ROWS if glob.glob(f"{R}/{r[0]}/dev/*/score.json")]
METRICS = [
    ("recall", "completeness\nframes of a fish detected"),
    ("idf1", "IDF1\none identity per fish"),
    ("precision", "precision\ndetections on a fish"),
    ("purity", "purity\nidentities on one fish"),
]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
SERIES = {"dev": ("#2a78d6", "lab noise"), "dev-field": ("#eb6834", "field noise")}


def mean(tag, split, key):
    d = [
        json.loads(Path(f).read_text())
        for f in sorted(glob.glob(f"{R}/{tag}/{split}/*/score.json"))
    ]
    return float(np.nanmean([x[key] for x in d])) if d else np.nan


fig, axs = plt.subplots(1, 4, figsize=(13, 0.6 * len(ROWS) + 1.4), sharey=True)
y = np.arange(len(ROWS))[::-1]
for ax, (key, title) in zip(axs, METRICS, strict=True):
    for split, (color, label) in SERIES.items():
        v = np.array([mean(t, split, key) for t, _ in ROWS])
        ax.plot(
            v, y, "o", color=color, ms=7, mec="white", mew=1.5, label=label, zorder=3
        )
        ref = mean("dense-linked", split, key)
        ax.axvline(ref, color=color, lw=0.8, ls=":", zorder=1)
        for yy, vv in zip(y, v, strict=True):
            dy = 7 if split == "dev" else -12
            ax.annotate(
                f"{vv:.3f}",
                (vv, yy),
                xytext=(0, dy),
                textcoords="offset points",
                ha="center",
                fontsize=6.5,
                color=MUTED,
            )
    ax.set_title(title, loc="left", fontsize=9, color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, lw=0.6, axis="x")
    ax.set_axisbelow(True)
    ax.margins(x=0.15)
axs[0].set_yticks(y, [lab for _, lab in ROWS], fontsize=8.5, color=INK)
axs[0].set_ylim(-0.6, len(ROWS) - 0.3)
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    fontsize=8.5,
    frameon=False,
    loc="lower center",
    ncol=2,
    bbox_to_anchor=(0.5, -0.04),
)
fig.suptitle(
    "dense settings on the dev suite, means over 17 mixtures (dotted: the current linker)",
    x=0.01,
    ha="left",
    fontsize=10,
    color=INK,
)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(OUT)
