"""Figures of the benchmark mixtures: spectrogram and ground truth by SNR.

Usage::

    python benchmarks/dense/gallery.py                    # dev, seed 0 of every scenario
    python benchmarks/dense/gallery.py --only dev/ladder-30-s0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA
from mixture import MIXTURES, expand_suite

FIG = DATA / "figures"
INK, MUTED = "#0b0b0b", "#52514e"
# sequential blue ramp (dataviz reference palette, steps 150..700)
BLUES = ["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
SNR_RANGE = (-5.0, 55.0)


def draw(name: str, axs, title: bool = True):
    from matplotlib.colors import LinearSegmentedColormap

    d = MIXTURES / name
    truth = np.load(d / "truth.npz")
    ov = np.load(d / "overview.npz")
    meta = json.loads((d / "mixture.json").read_text())
    ff = truth["freqs_filled"]
    fmin = max(0.0, float(np.nanmin(ff)) * 0.9)
    fmax = min(float(np.nanmax(ff)) * 1.06, float(ov["freqs"][-1]))
    m = (ov["freqs"] >= fmin) & (ov["freqs"] <= fmax)
    db = 10 * np.log10(ov["power"][m] + 1e-30)
    ax = axs[0]
    ax.pcolormesh(
        ov["times"],
        ov["freqs"][m],
        db,
        cmap="gray_r",
        vmin=np.percentile(db, 40),
        vmax=np.percentile(db, 99.8),
        shading="auto",
        rasterized=True,
    )
    cmap = LinearSegmentedColormap.from_list("snr", BLUES)
    ax = axs[1]
    t = truth["times"]
    snr = np.stack([np.interp(t, truth["snr_times"], s) for s in truth["snr_db"]])
    known = np.isfinite(truth["freqs"])
    sc = None
    for k in range(len(ff)):
        ax.plot(t, ff[k], color="#e6e5e0", lw=0.8, zorder=1)  # unknown stretches
        sc = ax.scatter(
            t[known[k]],
            truth["freqs"][k][known[k]],
            c=snr[k][known[k]],
            s=0.6,
            cmap=cmap,
            vmin=SNR_RANGE[0],
            vmax=SNR_RANGE[1],
            rasterized=True,
            zorder=2,
        )
    for ax in axs:
        ax.set_ylim(fmin, fmax)
        ax.set_xlim(0, meta["duration"])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=7)
    if title:
        g = meta["design"].get("gain")
        aug = (
            "no augmentation" if not g else ", ".join(f"{k} {v}" for k, v in g.items())
        )
        axs[0].set_title(
            f"{name}: {meta['n_fish']} fish, {aug}", loc="left", fontsize=9, color=INK
        )
    axs[0].set_ylabel("frequency [Hz]", fontsize=8, color=MUTED)
    return sc


def gallery(names: list[str], path: Path):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(
        len(names),
        2,
        figsize=(18, 3.4 * len(names)),
        layout="constrained",
        squeeze=False,
    )
    sc = None
    for row, name in zip(axs, names, strict=True):
        sc = draw(name, row)
        row[1].set_title(
            "ground truth, coloured by SNR", loc="left", fontsize=9, color=MUTED
        )
    axs[-1, 0].set_xlabel("time [s]", fontsize=8, color=MUTED)
    axs[-1, 1].set_xlabel("time [s]", fontsize=8, color=MUTED)
    cb = fig.colorbar(sc, ax=axs[:, 1], shrink=0.3, aspect=30, location="right")
    cb.set_label("SNR of the fundamental [dB]", fontsize=8, color=MUTED)
    cb.ax.tick_params(labelsize=7, colors=MUTED)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--only", nargs="*")
    p.add_argument("--split", default="dev")
    args = p.parse_args(argv)
    FIG.mkdir(parents=True, exist_ok=True)
    base = args.split.removesuffix("-field")
    names = args.only or [
        n.replace(base + "/", args.split + "/", 1)
        for n in expand_suite()
        if n.startswith(base + "/") and n.endswith("-s0")
    ]
    names = [n for n in names if (MIXTURES / n / "mixture.json").exists()]
    # pages of 6
    for i in range(0, len(names), 6):
        path = FIG / f"gallery_{args.split}_{i // 6 + 1}.png"
        gallery(names[i : i + 6], path)
        print(path)


if __name__ == "__main__":
    main()
