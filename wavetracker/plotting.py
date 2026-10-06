"""Quick-look figures of tracking results."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .results import Results, load_sparse_spectrogram


def plot_results(
    folder: str | Path,
    output: str | Path | None = None,
    min_detections: int = 10,
    flim: tuple[float | None, float | None] = (None, None),
) -> None:
    import matplotlib

    if output is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = Path(folder)
    res = Results.load(folder)
    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")

    if (folder / "sparse_spectra.npy").exists():
        spec, freqs, times = load_sparse_spectrogram(folder)
        with np.errstate(divide="ignore"):
            db = 10 * np.log10(spec)
        finite = db[np.isfinite(db)]
        vmin, vmax = np.percentile(finite, [50, 99.9]) if finite.size else (None, None)
        ax.pcolormesh(
            times,
            freqs,
            db,
            cmap="gray_r",
            vmin=vmin,
            vmax=vmax,
            shading="auto",
            rasterized=True,
        )

    t = res.times[res.idx_v]
    unassigned = np.isnan(res.ident_v)
    ax.plot(t[unassigned], res.fund_v[unassigned], ".", color="0.6", ms=1)
    ids = res.ids()
    colors = plt.get_cmap("tab20")
    for k, i in enumerate(ids):
        m = res.ident_v == i
        if m.sum() < min_detections:
            continue
        ax.plot(t[m], res.fund_v[m], ".", ms=3, color=colors(k % 20), label=f"{int(i)}")

    if len(res.fund_v):
        lo, hi = np.percentile(res.fund_v, [0.5, 99.5])
        ax.set_ylim(flim[0] or lo - 20, flim[1] or hi + 20)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("frequency [Hz]")
    ax.set_title(str(res.meta.get("input", folder)))
    if 0 < len(ids) <= 30:
        ax.legend(markerscale=6, fontsize="small", ncols=2, loc="upper right")
    if output is None:
        plt.show()
    else:
        fig.savefig(output, dpi=150)
        plt.close(fig)
