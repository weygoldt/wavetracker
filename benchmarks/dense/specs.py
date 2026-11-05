"""Specifications of the benchmark mixtures: table and figure.

The design of every mixture (sources, scales, fish frequencies, gains) is
replayed with the generator's random draws, so it is available before the
mixture is generated; SNRs are added for generated mixtures.

Usage::

    python benchmarks/dense/specs.py   # -> figures/specs.md, specs.csv, specs_*.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA
from mixture import MIXTURES, expand_suite, gains, place_sources

FIG = DATA / "figures"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
BLUES = ["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]


def design_of(name: str, d: dict, library: list[dict]) -> dict:
    """Sources and gains exactly as mixture.generate draws them."""
    split = name.split("/")[0]
    rng = np.random.default_rng([int(d["seed"]), *map(ord, name)])
    pool = [e for e in library if e["ok"] and e["split"] == split]
    sources = place_sources(d, pool, rng)
    gp = gains(d.get("gain"), 2 * len(sources), float(d["duration"]), rng)
    by_name = {e["name"]: e for e in library}
    fish = []
    for s, src in enumerate(sources):
        e = by_name[src["trial"]]
        for k in range(2):
            j = 2 * s + k
            g = gp["common_db"][j]
            fish.append(
                {
                    "source": s,
                    "trial": src["trial"],
                    "scale": src["scale"],
                    "median_hz": e["median_hz"][k] * src["scale"],
                    "lo_hz": e["range_hz"][k][0] * src["scale"],
                    "hi_hz": e["range_hz"][k][1] * src["scale"],
                    "gain_mean_db": float(g.mean()),
                    "gain_min_db": float(g.min()),
                    "gain_max_db": float(g.max()),
                    "modulated": bool(gp["modulated"][j]),
                }
            )
    return {"sources": sources, "fish": fish}


def nearest_lines(f: np.ndarray, max_h: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Per fish: distance to the nearest other fundamental, and to the nearest
    line of another fish including harmonics (1..max_h, both ways) [Hz]."""
    n = len(f)
    nf = np.full(n, np.inf)
    nl = np.full(n, np.inf)
    for k in range(n):
        for j in range(n):
            if j == k:
                continue
            nf[k] = min(nf[k], abs(f[k] - f[j]))
            for h in range(1, max_h + 1):
                nl[k] = min(nl[k], abs(f[k] - h * f[j]), abs(h * f[k] - f[j]) / h)
    return nf, nl


def main():
    library = json.loads((DATA / "library.json").read_text())
    designs = expand_suite()
    rows, fish_rows = [], []
    for name, d in designs.items():
        des = design_of(name, d, library)
        f = np.array([x["median_hz"] for x in des["fish"]])
        nf, nl = nearest_lines(f)
        g = d.get("gain") or {}
        row = {
            "mixture": name,
            "n_fish": d["n_fish"],
            "band_hz": f"{d['band'][0]}-{d['band'][1]}",
            "fish_hz": f"{f.min():.0f}-{f.max():.0f}",
            "below_300_hz": int((f < 300).sum()),
            "neighbour_median_hz": float(np.median(nf)),
            "neighbour_min_hz": float(nf.min()),
            "nearest_line_median_hz": float(np.median(nl)),
            "pairs_within_5_hz": int((nf < 5).sum()),
            "augmentation": ", ".join(f"{k}={v}" for k, v in g.items()) or "none",
            "modulated": int(sum(x["modulated"] for x in des["fish"])) if g else 0,
            "gain_mean_db": float(np.mean([x["gain_mean_db"] for x in des["fish"]])),
            "gain_min_db": float(min(x["gain_min_db"] for x in des["fish"])),
            "sources": "; ".join(
                f"{s['trial']} x{s['scale']:.2f}" for s in des["sources"]
            ),
        }
        truth_file = MIXTURES / name / "truth.npz"
        row["generated"] = truth_file.exists()
        if truth_file.exists():
            tr = np.load(truth_file)
            snr = tr["snr_db"]
            row["snr_median_db"] = float(np.nanmedian(snr))
            row["snr_p10_db"] = float(np.nanpercentile(snr, 10))
            row["frames_below_10_db"] = float(np.nanmean(snr < 10))
            known = np.isfinite(tr["freqs"]).mean()
            row["truth_known"] = float(known)
        rows.append(row)
        for x, a, b in zip(des["fish"], nf, nl, strict=True):
            fish_rows.append(
                {"mixture": name, **x, "neighbour_hz": a, "nearest_line_hz": b}
            )
    table = pd.DataFrame(rows)
    fish = pd.DataFrame(fish_rows)
    FIG.mkdir(parents=True, exist_ok=True)
    table.to_csv(FIG / "specs.csv", index=False)
    fish.to_csv(FIG / "specs_fish.csv", index=False)
    cols = [
        "mixture",
        "n_fish",
        "fish_hz",
        "below_300_hz",
        "neighbour_median_hz",
        "neighbour_min_hz",
        "nearest_line_median_hz",
        "pairs_within_5_hz",
        "modulated",
        "gain_mean_db",
        "gain_min_db",
        "snr_median_db",
        "snr_p10_db",
        "frames_below_10_db",
    ]
    show = table.reindex(columns=cols)
    (FIG / "specs.md").write_text(markdown(show))
    print(show.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    for split in ("dev", "test"):
        p = FIG / f"specs_{split}.png"
        plot(fish[fish.mixture.str.startswith(split + "/")], p, split)
        print(p)


def markdown(df: pd.DataFrame) -> str:
    def fmt(v):
        if isinstance(v, float):
            return "" if np.isnan(v) else f"{v:.1f}"
        return str(v)

    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += [
        "| " + " | ".join(fmt(v) for v in row) + " |"
        for row in df.itertuples(index=False)
    ]
    return "\n".join(lines) + "\n"


def plot(fish: pd.DataFrame, path: Path, split: str):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    names = list(dict.fromkeys(fish.mixture))
    cmap = LinearSegmentedColormap.from_list("gain", BLUES)
    norm = Normalize(-45, 0)
    fig, ax = plt.subplots(figsize=(13, 0.42 * len(names) + 1.6), layout="constrained")
    for i, n in enumerate(names):
        y = len(names) - 1 - i
        sub = fish[fish.mixture == n]
        for _, x in sub.iterrows():
            c = cmap(norm(x.gain_mean_db))
            ax.plot([x.lo_hz, x.hi_hz], [y, y], color=c, lw=1.2, solid_capstyle="round")
            ax.plot(x.median_hz, y, "o", ms=4.5, color=c, mec="#fcfcfb", mew=0.8)
    ax.set_xscale("log")
    ticks = [60, 100, 150, 200, 300, 400, 600, 800, 1000, 1300]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set_yticks(range(len(names)), [n.split("/")[1] for n in names][::-1], fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel(
        "EOD frequency [Hz] (dot: median, line: 1-99 % range)", fontsize=9, color=MUTED
    )
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=25)
    cb.set_label("mean gain over the 10 min [dB]", fontsize=8, color=MUTED)
    cb.ax.tick_params(labelsize=7, colors=MUTED)
    ax.set_title(
        f"benchmark mixtures, {split} split: every fish",
        loc="left",
        fontsize=11,
        color=INK,
    )
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
