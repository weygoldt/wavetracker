"""How much do the electrode patterns of the tube fish change over time?

For every kept source (library.json) and each of its two fish: the power
pattern over the source's mixture electrodes (channels_used) in 10 s windows
(the linker's pattern_window), dB minus the mean over electrodes as in
wavetracker.linking. Prints the rms pattern difference between windows as a
function of lag, against the split-half difference within one window, and
draws it with the patterns of the median and the most mobile fish.

    python benchmarks/dense/patterns.py [figure.png]
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from common import DATA

OUT = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else DATA / "figures" / "tube_pattern_motion.png"
)
WIN = 10.0
MIN_DET = 20
LAGS = np.array([10, 20, 30, 60, 120, 300, 600, 900])
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def pattern(p):
    db = 10 * np.log10(np.maximum(p.mean(0), 1e-30))
    return db - db.mean(), 10 * np.log10(max(p.mean(0).sum(), 1e-30))


def rms(d):
    return float(np.sqrt(np.mean(d * d)))


lib = [e for e in json.loads((DATA / "library.json").read_text()) if e.get("ok")]
fish = []
for e in lib:
    d = DATA / "sources" / e["name"]
    sv = np.load(d / "sign_v.npy")[:, e["channels_used"]]
    fv, iv = np.load(d / "fund_v.npy"), np.load(d / "idx_v.npy")
    times = np.load(d / "times.npy")
    fd = np.load(d / "truth.npz")["freqs_detected"]
    for k in range(fd.shape[0]):
        on = np.abs(fv - fd[k, iv]) < 1e-6
        t, p = times[iv[on]], sv[on]
        edges = np.arange(t.min(), t.max() + WIN, WIN)
        w = np.digitize(t, edges) - 1
        pats, levs, noise, tw = [], [], [], []
        for j in range(len(edges) - 1):
            pj = p[w == j]
            if len(pj) < MIN_DET:
                pats.append(np.full(p.shape[1], np.nan))
                levs.append(np.nan)
                continue
            a, lv = pattern(pj)
            pats.append(a)
            levs.append(lv)
            h = len(pj) // 2
            noise.append(rms(pattern(pj[:h])[0] - pattern(pj[h:])[0]))
        pats, levs = np.array(pats), np.array(levs)
        D = []
        for L in LAGS:
            s = round(L / WIN)
            dd = [
                rms(pats[i] - pats[i + s])
                for i in range(len(pats) - s)
                if np.isfinite(pats[i]).all() and np.isfinite(pats[i + s]).all()
            ]
            D.append(np.median(dd) if dd else np.nan)
        med = np.nanmedian(pats, 0)
        dev = np.array([rms(x - med) if np.isfinite(x).all() else np.nan for x in pats])
        fish.append(
            dict(
                name=e["name"],
                dataset=e["dataset"],
                k=k,
                D=np.array(D),
                noise=float(np.median(noise)),
                pats=pats,
                levs=levs,
                dev=dev,
                moving=float(np.nanmean(dev > 6.0)),
                level_range=float(
                    np.nanpercentile(levs, 95) - np.nanpercentile(levs, 5)
                ),
            )
        )

D = np.array([f["D"] for f in fish])
noise = np.array([f["noise"] for f in fish])
mov = np.array([f["moving"] for f in fish])
lr = np.array([f["level_range"] for f in fish])
print(f"{len(fish)} fish in {len(lib)} sources")
print(f"split-half noise of a 10 s pattern [dB]: median {np.median(noise):.2f}")
for L, col in zip(LAGS, D.T, strict=True):
    q = np.nanpercentile(col, [25, 50, 75, 90])
    print(
        f"lag {L:4d} s: pattern rms diff median {q[1]:.2f} dB (IQR {q[0]:.2f}-{q[2]:.2f}, p90 {q[3]:.2f})"
    )
print("fraction of 10 s windows > 6 dB rms from the fish's own median pattern:")
print(
    "  median {:.2f}, p75 {:.2f}, p90 {:.2f}, max {:.2f}".format(
        *tuple(np.percentile(mov, [50, 75, 90, 100]))
    )
)
print(
    "level range p5-p95 over 20 min [dB]: median {:.1f}, p90 {:.1f}".format(
        *tuple(np.percentile(lr, [50, 90]))
    )
)
for ds in sorted({f["dataset"] for f in fish}):
    m = np.array([f["dataset"] == ds for f in fish])
    print(
        f"  {ds}: n={m.sum()}, D(60 s) median {np.nanmedian(D[m, 3]):.2f}, moving {np.median(mov[m]):.2f}"
    )


# augmentation: independent OU per electrode, std s, timescale tau; pattern =
# mean removed over n electrodes -> extra rms difference at lag L
def aug(L, s, tau, n=9):
    return np.sqrt(2 * s * s * (1 - np.exp(-L / tau)) * (1 - 1 / n))


fig, axs = plt.subplots(
    1, 3, figsize=(13, 3.8), gridspec_kw=dict(width_ratios=[1.1, 1.6, 1.6])
)
ax = axs[0]
for f in fish:
    ax.plot(LAGS, f["D"], color=SERIES[0], alpha=0.12, lw=0.8)
ax.plot(LAGS, np.nanmedian(D, 0), color=SERIES[0], lw=2, label="tube fish (median)")
ax.axhline(
    np.median(noise),
    color=MUTED,
    ls=":",
    lw=1.2,
    label="split-half difference within 10 s",
)
Lf = np.geomspace(10, 900, 100)
nat = np.interp(Lf, LAGS, np.nanmedian(D, 0))
ax.plot(
    Lf,
    np.sqrt(nat**2 + aug(Lf, 3, 90) ** 2),
    color=SERIES[1],
    lw=1.4,
    ls="--",
    label="+ suite drift (3 dB, 90 s)",
)
ax.plot(
    Lf,
    np.sqrt(nat**2 + aug(Lf, 12, 25) ** 2),
    color=SERIES[2],
    lw=1.4,
    ls="--",
    label="+ planned drift (12 dB, 25 s)",
)
ax.set_xscale("log")
ax.set_xlabel("lag between 10 s windows [s]", color=INK, fontsize=9)
ax.set_ylabel("rms pattern difference [dB]", color=INK, fontsize=9)
ax.set_title("electrode pattern change with lag", color=INK, fontsize=10, loc="left")
ax.legend(fontsize=7, frameon=False)
style(ax)

order = np.argsort(mov)
for ax, idx, what in (
    (axs[1], order[len(order) // 2], "median fish"),
    (axs[2], order[-1], "most-moving fish"),
):
    f = fish[idx]
    tw = np.arange(len(f["pats"])) * WIN / 60
    im = ax.imshow(
        f["pats"].T,
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
        vmin=-15,
        vmax=15,
        extent=(0, tw[-1] + WIN / 60, -0.5, f["pats"].shape[1] - 0.5),
        interpolation="nearest",
    )
    ax.set_xlabel("time in the 20 min source [min]", color=INK, fontsize=9)
    ax.set_ylabel("electrode", color=INK, fontsize=9)
    ax.set_title(
        f"{what}: {f['name']} #{f['k']}\n{f['moving'] * 100:.0f} % of 10 s windows > 6 dB from its median pattern",
        color=INK,
        fontsize=9,
        loc="left",
    )
    style(ax)
    ax.grid(False)
cb = fig.colorbar(im, ax=axs[1:], fraction=0.02, pad=0.01)
cb.set_label("power re mean of electrodes [dB]", fontsize=8, color=INK)
cb.ax.tick_params(labelsize=7, colors=MUTED)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(OUT)
