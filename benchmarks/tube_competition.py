"""Benchmark on the 2022 tube-competition recordings (two fish per trial).

No ground truth traces exist, so a pseudo ground truth is built from the
detections alone (independent of tracking): in short windows the two
dominant frequency peaks are taken as the two fish, smoothed over time, and
labelled winner/loser via ``meta.csv``. Detections within ``--tol`` Hz of a
trace belong to that fish; rises leave that band and count as unmatched.

Usage::

    wavetracker run <recordings...> -c benchmarks/tube_competition.yaml -o output/benchmark
    for d in output/benchmark/2022-*; do wavetracker cleanup $d -n 2; done
    python benchmarks/tube_competition.py output/benchmark \\
        --meta /mnt/data2/2022_tube_competition/raw/meta.csv
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import find_peaks

from wavetracker.results import Results

WINDOW = 10.0  # s, pseudo-truth resolution
MIN_DETECTIONS = 300  # an identity with fewer detections (~50 s) is a fragment


def pseudo_truth(
    res: Results, window: float = WINDOW, max_jump: float = 15.0
) -> np.ndarray:
    """Frequencies of the lower and upper fish at every frame, shape (2, frames).

    In every window the frequency peaks of the detections are found; each fish
    takes the peak closest to its previous value (within `max_jump` Hz), so
    short-lived interference or missing detections cannot capture a trace.
    Both traces are initialised from the two dominant peaks of the first
    2 minutes.
    """
    t_det = res.times[res.idx_v]
    lo, hi = np.floor(res.fund_v.min()) - 5, np.ceil(res.fund_v.max()) + 5
    bins = np.arange(lo, hi + 0.5, 0.5)  # spans the detections
    dt = res.times[1] - res.times[0]

    def peaks_of(f, min_count):
        h = gaussian_filter1d(np.histogram(f, bins)[0].astype(float), 2)
        p, props = find_peaks(h, distance=10, height=min_count)
        order = np.argsort(props["peak_heights"])[::-1]
        return [np.median(f[np.abs(f - bins[i]) < 2]) for i in p[order]]

    first = t_det < res.times[0] + 120
    last = np.sort(peaks_of(res.fund_v[first], 1)[:2])
    if len(last) < 2:
        raise ValueError("could not find two fish in the first 2 minutes")

    edges = np.arange(res.times[0], res.times[-1] + window, window)
    traces = np.full((2, len(edges) - 1), np.nan)
    which = np.digitize(t_det, edges) - 1
    for w in range(len(edges) - 1):
        f = res.fund_v[which == w]
        if len(f) == 0:
            continue
        cand = np.array(peaks_of(f, 0.05 * window / dt))
        if len(cand) == 0:
            continue
        for k in range(2):
            d = np.abs(cand - last[k])
            if d.min() <= max_jump:
                traces[k, w] = cand[d.argmin()]
                last[k] = traces[k, w]
    centers = 0.5 * (edges[:-1] + edges[1:])
    out = np.empty((2, len(res.times)))
    for k in range(2):
        tr = traces[k]
        ok = ~np.isnan(tr)
        tr = np.interp(centers, centers[ok], tr[ok])
        tr = median_filter(tr, size=5, mode="nearest")
        out[k] = np.interp(res.times, centers, tr)
    return out


def match(fund, idx, truth, tol):
    """Fish index (0 lower, 1 upper) of each detection, -1 if unmatched."""
    ref = truth[:, idx]
    err = np.abs(ref - fund[None, :])
    k = err.argmin(0)
    return np.where(err[k, np.arange(len(k))] <= tol, k, -1)


@dataclass
class TrackScore:
    n_ids: int
    n_ids_substantial: int
    purity: float
    mixed_ids: int
    unmatched: float
    det_coverage: tuple[float, float]
    largest_track_coverage: tuple[float, float]
    ids_for_90: tuple[int, int]


def score(fund, idx, ident, truth, n_frames, tol) -> tuple[TrackScore, np.ndarray]:
    fish = match(fund, idx, truth, tol)
    assigned = ~np.isnan(ident)
    ids, sizes = np.unique(ident[assigned], return_counts=True)
    counts = np.zeros((len(ids), 2), int)  # matched detections per id and fish
    pos = np.searchsorted(ids, ident[assigned])
    for k in range(2):
        np.add.at(counts[:, k], pos[fish[assigned] == k], 1)
    total = counts.sum()
    purity = counts.max(1).sum() / total if total else np.nan
    minority = counts.min(1)
    mixed = int(((minority >= 50) & (minority >= 0.1 * counts.sum(1))).sum())

    det_cov, big_cov, n90 = [], [], []
    for k in range(2):
        det_cov.append(len(np.unique(idx[fish == k])) / n_frames)
        best = 0
        for i in ids[np.argsort(counts[:, k])[::-1][:5]]:
            best = max(best, len(np.unique(idx[(ident == i) & (fish == k)])))
        big_cov.append(best / n_frames)
        c = np.sort(counts[:, k])[::-1]
        n90.append(
            int(np.searchsorted(np.cumsum(c), 0.9 * c.sum()) + 1) if c.sum() else 0
        )
    s = TrackScore(
        n_ids=len(ids),
        n_ids_substantial=int((sizes >= MIN_DETECTIONS).sum()),
        purity=float(purity),
        mixed_ids=mixed,
        unmatched=float((fish < 0).mean()),
        det_coverage=tuple(det_cov),
        largest_track_coverage=tuple(big_cov),
        ids_for_90=tuple(n90),
    )
    return s, fish


def majority_fish(ident, fish):
    """Majority fish of every identity (dict id -> 0/1/-1)."""
    out = {}
    for i in np.unique(ident[~np.isnan(ident)]):
        f = fish[(ident == i) & (fish >= 0)]
        out[i] = int(np.bincount(f, minlength=2).argmax()) if len(f) else -1
    return out


def evaluate_recording(folder: Path, meta: pd.Series | None, tol: float) -> dict:
    res = Results.load(folder)
    truth = pseudo_truth(res)
    n_frames = len(res.times)
    raw, _ = score(res.fund_v, res.idx_v, res.ident_v, truth, n_frames, tol)
    row = {"recording": folder.name, "raw": asdict(raw)}

    # winner = upper or lower fish?
    if meta is not None:
        upper_wins = meta["Winner_pitch"] == "high"
        row["pairing"] = meta["Pairing"]
        row["winner"] = 1 if upper_wins else 0
        ref = (meta["Lose_EODf"], meta["Win_EODf"])
        ref = ref if upper_wins else ref[::-1]
        # how close do the pseudo-truth traces get to the reference EODfs?
        row["truth_vs_meta_hz"] = [
            float(np.min(np.abs(truth[k] - ref[k]))) for k in range(2)
        ]
    row["min_separation_hz"] = float(np.min(np.abs(truth[1] - truth[0])))
    row["truth_range"] = [
        [float(truth[k].min()), float(truth[k].max())] for k in range(2)
    ]

    cleaned = folder / "ident_v_cleaned_n2.npy"
    if cleaned.exists():
        ident = np.load(cleaned)
        fund = np.load(folder / "fund_v_cleaned_n2.npy")
        idx = np.load(folder / "idx_v_cleaned_n2.npy")
        cs, fish = score(fund, idx, ident, truth, n_frames, tol)
        maj = majority_fish(ident, fish)
        # coverage of each fish by the cleanup identity that represents it
        cov = []
        for k in range(2):
            ids_k = [i for i, m in maj.items() if m == k]
            frames = (
                np.unique(idx[np.isin(ident, ids_k) & (fish == k)]) if ids_k else []
            )
            cov.append(len(frames) / n_frames)
        row["cleanup"] = asdict(cs) | {
            "fish_coverage": cov,
            "ids_per_fish": [sum(m == k for m in maj.values()) for k in range(2)],
        }
    row["runtime_s"] = res.meta.get("timings", {})
    return row


# reference palette (dataviz skill): fish identity = categorical slots 1/2;
# successive fragments of one fish alternate between two steps of its hue
FISH_COLORS = (("#2a78d6", "#86b6ef"), ("#eb6834", "#f4a582"))
GRAY = "#9b9a96"
INK = "#52514e"


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
    ax.tick_params(colors=INK, labelsize=8)
    ax.grid(axis="y", color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)


def _plot_ids(ax, t, fund, idx, ident, fish, names):
    """Scatter detections, coloured by the majority fish of their identity."""
    maj = majority_fish(ident, fish)
    un = np.isnan(ident) | np.array([maj.get(i, -1) < 0 for i in ident])
    ax.plot(t[idx[un]], fund[un], ".", ms=0.8, color=GRAY, rasterized=True)
    n_shown = [0, 0]
    for k in range(2):
        ids = [i for i, m in maj.items() if m == k]
        # alternate shades in temporal order so fragment boundaries are visible
        ids.sort(key=lambda i: idx[ident == i].min())
        for j, i in enumerate(ids):
            m = ident == i
            ax.plot(
                t[idx[m]],
                fund[m],
                ".",
                ms=0.8,
                color=FISH_COLORS[k][j % 2],
                rasterized=True,
            )
        n_shown[k] = len(ids)
    return n_shown


def plot_tracks(folder: Path, out: Path, names=("loser", "winner"), tol=4.0):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    res = Results.load(folder)
    truth = pseudo_truth(res)
    th = res.times / 3600
    fish = match(res.fund_v, res.idx_v, truth, tol)
    fig, axs = plt.subplots(
        2, 1, figsize=(12, 6.5), sharex=True, sharey=True, layout="constrained"
    )
    n_raw = _plot_ids(axs[0], th, res.fund_v, res.idx_v, res.ident_v, fish, names)
    axs[0].set_title(
        f"wavetracker: {names[1]} {n_raw[1]}, {names[0]} {n_raw[0]} identities",
        loc="left",
        fontsize=10,
        color="#0b0b0b",
    )
    cleaned = folder / "ident_v_cleaned_n2.npy"
    if cleaned.exists():
        fund = np.load(folder / "fund_v_cleaned_n2.npy")
        idx = np.load(folder / "idx_v_cleaned_n2.npy")
        ident = np.load(cleaned)
        fish_c = match(fund, idx, truth, tol)
        n_c = _plot_ids(axs[1], th, fund, idx, ident, fish_c, names)
        axs[1].set_title(
            f"after cleanup -n 2: {names[1]} {n_c[1]}, {names[0]} {n_c[0]} identities",
            loc="left",
            fontsize=10,
            color="#0b0b0b",
        )
    lo, hi = truth.min() - 15, truth.max() + 45
    for ax in axs:
        _style(ax)
        ax.set_ylim(lo, hi)
        ax.set_ylabel("EOD frequency [Hz]", color=INK, fontsize=9)
    axs[1].set_xlabel("time [h]", color=INK, fontsize=9)
    axs[1].set_xlim(0, th[-1])
    handles = [
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            ms=5,
            color=FISH_COLORS[1][0],
            label=f"{names[1]} (upper fish)",
        ),
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            ms=5,
            color=FISH_COLORS[0][0],
            label=f"{names[0]} (lower fish)",
        ),
        Line2D([], [], ls="", marker="o", ms=5, color=GRAY, label="unassigned"),
    ]
    if names[1] == "loser":
        handles[:2] = handles[1::-1]
    fig.legend(
        handles=handles, loc="outside upper right", ncols=3, frameon=False, fontsize=8
    )
    fig.suptitle(
        f"{folder.name} - successive identities alternate light/dark",
        x=0.01,
        ha="left",
        fontsize=11,
        color="#0b0b0b",
    )
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_summary(rows: list[dict], out: Path):
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda r: r["recording"], reverse=True)
    labels = [f"{r['recording'][:10]} ({r.get('pairing', '')})" for r in rows]
    y = np.arange(len(rows))
    fig, axs = plt.subplots(
        1,
        3,
        figsize=(13, 0.45 * len(rows) + 1.6),
        sharey=True,
        layout="constrained",
        gridspec_kw={"width_ratios": [2, 2, 1.2]},
    )
    series = [
        ("detection (ceiling)", lambda r, k: r["raw"]["det_coverage"][k], GRAY, "o"),
        (
            "largest raw track",
            lambda r, k: r["raw"]["largest_track_coverage"][k],
            "#2a78d6",
            "s",
        ),
        (
            "after cleanup",
            lambda r, k: r["cleanup"]["fish_coverage"][k],
            "#eb6834",
            "D",
        ),
    ]
    for col, who in enumerate(("winner", "loser")):
        ax = axs[col]
        for name, get, color, marker in series:
            vals = []
            for r in rows:
                k = r["winner"] if who == "winner" else 1 - r["winner"]
                vals.append(
                    get(r, k) if "cleanup" in r or "cleanup" not in name else np.nan
                )
            ax.plot(
                np.array(vals) * 100,
                y,
                ls="",
                marker=marker,
                ms=7,
                color=color,
                mec="#fcfcfb",
                mew=1.5,
                label=name,
            )
        ax.set_xlim(0, 100)
        ax.set_title(f"coverage of the {who} [% of frames]", loc="left", fontsize=10)
    axs[0].legend(loc="lower left", fontsize=8, frameon=False)
    ax = axs[2]
    n_sub = [r["raw"]["n_ids_substantial"] for r in rows]
    ax.barh(y, n_sub, height=0.55, color="#2a78d6")
    for yi, v in zip(y, n_sub, strict=True):
        ax.text(v + 0.5, yi, str(v), va="center", fontsize=8, color=INK)
    ax.axvline(2, color=INK, lw=0.8, ls=":")
    ax.set_title("identities >= 300 detections\n(truth: 2)", loc="left", fontsize=10)
    for ax in axs:
        _style(ax)
        ax.grid(axis="x", color="#e6e5e0", lw=0.6)
        ax.grid(axis="y", visible=False)
    axs[0].set_yticks(y, labels, fontsize=8)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("folder", type=Path)
    p.add_argument("--meta", type=Path)
    p.add_argument("--tol", type=float, default=4.0)
    p.add_argument(
        "--plot", nargs="*", default=None, help="recordings to draw track figures for"
    )
    args = p.parse_args(argv)
    meta = pd.read_csv(args.meta, index_col=0) if args.meta else None
    rows = []
    for d in sorted(args.folder.iterdir()):
        if not (d / "fund_v.npy").exists():
            continue
        m = meta.loc[d.name] if meta is not None and d.name in meta.index else None
        rows.append(evaluate_recording(d, m, args.tol))
        print(json.dumps(rows[-1]))
    (args.folder / "benchmark.json").write_text(json.dumps(rows, indent=1))
    plot_summary(rows, args.folder / "summary.png")
    for name in args.plot or []:
        winner = next(r["winner"] for r in rows if r["recording"] == name)
        names = ("loser", "winner") if winner == 1 else ("winner", "loser")
        plot_tracks(args.folder / name, args.folder / f"tracks_{name}.png", names)


if __name__ == "__main__":
    main()
