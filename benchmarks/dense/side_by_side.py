"""Side-by-side spectrograms of two runs on the same mixture window.

Each identity is drawn in its own colour; identities whose detections are
mostly false or harmonics of a fish are drawn dashed in red; detections
without identity are grey dots. Windows are chosen without looking at the
runs (the densest band in the middle of the mixture) unless given.

    python benchmarks/dense/side_by_side.py --tags classic-ldev s5000-t0.6 \\
        --mixtures ldev/density-30-s0 ldev-field/lowfreq-30-s0
    python benchmarks/dense/side_by_side.py ... --worst   # + the window worst for the 2nd tag
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA, MIX_CHANNELS, RATE
from score import classify_unmatched, match, truth_at

from wavetracker.results import Results
from wavetracker.spectrogram import PowerSpectrogram

RUNS = DATA / "runs"
MIXTURES = DATA / "mixtures"
OUT = DATA / "figures" / "side_by_side"
TOL = 2.0  # Hz, as score.py
NFFT, STEP = 2**15, int(2**15 * 0.1)
PALETTE = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#0f8fa3",
    "#b0651a",
    "#7a9a01",
]
LABELS = {
    "classic-ldev": "classic pipeline (dense-linked.yaml)",
    "s5000-t0.6": "learned v1 (threshold 0.6)",
    "F-an-t0.5": "default pipeline (learned-assocnet.yaml, threshold 0.5)",
}


def load(tag: str, name: str):
    """Run results with each detection classified against the truth."""
    from scipy.optimize import linear_sum_assignment

    res = Results.load(RUNS / tag / name)
    truth = dict(np.load(MIXTURES / name / "truth.npz"))
    F = truth_at(truth, res.times)
    Ff = truth_at(truth, res.times, "freqs_filled")
    fish = match(res, F, TOL)
    cls = classify_unmatched(res, F, Ff, fish, TOL)
    ids = np.unique(res.ident_v[np.isfinite(res.ident_v)])
    counts = np.zeros((len(ids), len(F)), int)
    m = (fish >= 0) & np.isfinite(res.ident_v)
    np.add.at(counts, (np.searchsorted(ids, res.ident_v[m]), fish[m]), 1)
    r, c = linear_sum_assignment(-counts)
    own = {int(cc): ids[rr] for rr, cc in zip(r, c, strict=True)}
    on_own = np.array(
        [
            fish[i] >= 0 and own.get(int(fish[i])) == res.ident_v[i]
            for i in range(len(fish))
        ]
    )
    bad = (cls == 2) | (cls == 3)
    score = json.loads((RUNS / tag / name / "score.json").read_text())
    return {"res": res, "on_own": on_own, "bad": bad, "score": score, "truth": truth}


def neutral_window(truth, dur=150.0, width=None):
    """Middle of the mixture, the band of `width` Hz holding most fish."""
    t = truth["times"]
    t0 = 0.5 * (t[-1] - dur)
    sel = (t >= t0) & (t < t0 + dur)
    med = np.nanmedian(truth["freqs"][:, sel], axis=1)
    med = med[np.isfinite(med)]
    if width is None:
        width = 150.0 if np.median(med) < 400 else 220.0
    best = max(
        np.arange(med.min() - 10, med.max(), 5.0),
        key=lambda f0: np.sum((med >= f0) & (med < f0 + width)),
    )
    return t0, t0 + dur, best, best + width


def worst_window(a, b, name, dur=150.0, width=220.0):
    """Window where run b has the most fewer frames on the fish's own identity
    than run a (plus its extra false/harmonic detections)."""
    best, arg = -np.inf, None
    ra, rb = a["res"], b["res"]
    ta, tb = ra.times[ra.idx_v], rb.times[rb.idx_v]
    for t0 in np.arange(0, ra.times[-1] - dur, 25.0):
        for f0 in np.arange(60, 1300 - width, 20.0):

            def n(res, t, mask, t0=t0, f0=f0):
                m = (
                    mask
                    & (t >= t0)
                    & (t < t0 + dur)
                    & (res.fund_v >= f0)
                    & (res.fund_v < f0 + width)
                )
                return int(m.sum())

            d = (n(ra, ta, a["on_own"]) - n(rb, tb, b["on_own"])) + (
                n(rb, tb, b["bad"]) - n(ra, ta, a["bad"])
            )
            if d > best:
                best, arg = d, (t0, t0 + dur, f0, f0 + width)
    return arg, best


def spectrogram(name, t0, t1, f0, f1, dev):
    raw = MIXTURES / name / "traces-grid1.raw"
    n = raw.stat().st_size // 4 // MIX_CHANNELS
    x = np.memmap(raw, dtype=np.float32, mode="r", shape=(n, MIX_CHANNELS))
    a = max(0, int(t0 * RATE) - NFFT // 2)
    b = min(n, int(t1 * RATE) + NFFT // 2)
    spec = PowerSpectrogram(NFFT, STEP, RATE, dev)
    p = spec(torch.from_numpy(np.ascontiguousarray(x[a:b].T)).to(dev)).sum(0)
    freqs = np.fft.rfftfreq(NFFT, 1 / RATE)
    k = (freqs >= f0) & (freqs <= f1)
    db = 10 * np.log10(p[torch.from_numpy(k).to(dev)].cpu().numpy() + 1e-30)
    times = (a + np.arange(db.shape[1]) * STEP + NFFT / 2) / RATE
    return times, freqs[k], db


def panel(ax, run, win, spec, title):
    times, freqs, db = spec
    t0, t1, f0, f1 = win
    ax.pcolormesh(
        times,
        freqs,
        db,
        cmap="gray_r",
        shading="auto",
        rasterized=True,
        vmin=np.percentile(db, 35),
        vmax=np.percentile(db, 99.7),
    )
    res = run["res"]
    t = res.times[res.idx_v]
    inw = (t >= t0) & (t <= t1) & (res.fund_v >= f0) & (res.fund_v <= f1)
    free = inw & ~np.isfinite(res.ident_v)
    ax.plot(t[free], res.fund_v[free], ".", ms=1.5, color="#8a8a8a", rasterized=True)
    ids = np.unique(res.ident_v[inw & np.isfinite(res.ident_v)])
    n_bad_ids = 0
    for j, i in enumerate(ids):
        m = inw & (res.ident_v == i)
        o = np.argsort(t[m])
        tt, ff = t[m][o], res.fund_v[m][o]
        bad = run["bad"][m].mean() > 0.5
        n_bad_ids += bad
        # break lines at gaps longer than 2 s
        gaps = np.flatnonzero(np.diff(tt) > 2.0) + 1
        for seg_t, seg_f in zip(np.split(tt, gaps), np.split(ff, gaps), strict=True):
            if bad:
                ax.plot(seg_t, seg_f, "--", lw=1.3, color="#d03b3b")
            else:
                ax.plot(seg_t, seg_f, "-", lw=1.6, color=PALETTE[j % len(PALETTE)])
    own = (run["on_own"] & inw).sum() / max(inw.sum(), 1)
    ax.set_xlim(t0, t1)
    ax.set_ylim(f0, f1)
    ax.set_title(
        f"{title}\n{len(ids)} identities here ({n_bad_ids} false/harmonic, red), "
        f"{own:.0%} of detections on the fish's own identity; mixture IDF1 "
        f"{run['score']['idf1']:.2f}",
        loc="left",
        fontsize=9,
    )
    ax.set_xlabel("time [s]", fontsize=9)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main(argv=None):
    import matplotlib.pyplot as plt

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tags", nargs=2, required=True)
    p.add_argument("--mixtures", nargs="+", required=True)
    p.add_argument(
        "--worst", action="store_true", help="add the window worst for the 2nd tag"
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", help="figure folder (default: figures/side_by_side)")
    args = p.parse_args(argv)
    out = DATA / "figures" / args.out if args.out else OUT
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    neutral = "window: middle of the mixture, densest band"
    todo = [(m, None, neutral) for m in args.mixtures]
    if args.worst:
        cands = []
        for m in args.mixtures:
            w, d = worst_window(load(args.tags[0], m), load(args.tags[1], m), m)
            cands.append((d, m, w))
        d, m, w = max(cands, key=lambda c: c[0])
        todo.append((m, w, f"worst window for {args.tags[1]} of all shown mixtures"))
    paths = []
    for k, (name, win, kind) in enumerate(todo):
        a, b = load(args.tags[0], name), load(args.tags[1], name)
        win = win or neutral_window(a["truth"])
        spec = spectrogram(name, *win, dev)
        fig, axs = plt.subplots(
            1, 2, figsize=(17, 6.2), sharey=True, layout="constrained"
        )
        for ax, run, tag in zip(axs, (a, b), args.tags, strict=True):
            panel(ax, run, win, spec, LABELS.get(tag, tag))
        axs[0].set_ylabel("frequency [Hz]", fontsize=9)
        fig.suptitle(
            f"{name}  ({kind}: {win[0]:.0f}-{win[1]:.0f} s, {win[2]:.0f}-{win[3]:.0f} Hz)",
            x=0.01,
            ha="left",
            fontsize=11,
        )
        path = out / f"{k + 1:02d}_{name.replace('/', '_')}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths.append(path)
        print(path, flush=True)


if __name__ == "__main__":
    main()
