"""Run wavetracker on the dense-mixture suite, score it, draw the report.

A run is named by a tag and a wavetracker config (merged with each
mixture's ``band.yaml``, which sets the fish range). Results go to
``$WT_BENCHMARK_DATA/runs/<tag>/<split>/<mixture>/``.

Usage::

    python benchmarks/dense/run.py track  --tag default
    python benchmarks/dense/run.py track  --tag dense -c benchmarks/dense/configs/dense.yaml
    python benchmarks/dense/run.py track  --tag dense -c ... --split dev-field
    python benchmarks/dense/run.py report --tags default dense   # dev and dev-field
    python benchmarks/dense/run.py report --tags default --splits test test-field
    python benchmarks/dense/run.py track  --tag mine -c ... --split ldev   # learned-detector dev
    python benchmarks/dense/run.py track  --tag mine -c ... --split ldev2  # and its second set
    python benchmarks/dense/run.py retrack --tag mine-dt20 --source mine -c ... --set tracking.max_dt=20 --split ldev
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA
from mixture import (
    LEARNED_DEV2_SUITE,
    LEARNED_DEV_SUITE,
    LEARNED_TRAIN_SUITE,
    MIXTURES,
    TRAIN_SUITE,
    expand_suite,
)
from score import TOL, binned, score

from wavetracker.config import Config
from wavetracker.pipeline import detect, track_results

RUNS = DATA / "runs"
FIG = DATA / "figures"

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # categorical slots 1-4
GRAY = "#9b9a96"
CRITICAL = "#d03b3b"


def merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = (
            merge(out[k], v)
            if isinstance(v, dict) and isinstance(out.get(k), dict)
            else v
        )
    return out


def mixtures(split: str | None, only=None) -> list[str]:
    """Complete mixtures of a split: dev, test, train, ldev, ldev2, ltrain
    (lab noise) or their -field variants (the same fish in field noise, see
    noise.py)."""
    suite = list(expand_suite())
    for path in (
        TRAIN_SUITE,
        LEARNED_DEV_SUITE,
        LEARNED_DEV2_SUITE,
        LEARNED_TRAIN_SUITE,
    ):
        if path.exists():
            suite += list(expand_suite(path))
    names = suite + [n.replace("/", "-field/", 1) for n in suite]
    # band.yaml is written last: the mixture is complete
    names = [n for n in names if (MIXTURES / n / "band.yaml").exists()]
    if split:
        names = [n for n in names if n.startswith(split + "/")]
    if only:
        names = [n for n in names if any(o in n for o in only)]
    return names


def track(args):
    variant = yaml.safe_load(Path(args.config).read_text()) if args.config else {}
    variant = variant or {}
    for name in mixtures(args.split, args.only):
        out = RUNS / args.tag / name
        if (out / "score.json").exists() and not args.overwrite:
            continue
        band = yaml.safe_load((MIXTURES / name / "band.yaml").read_text())
        cfg = Config.from_dict(merge(variant, band) | {})
        t0 = time.perf_counter()
        for attempt in range(30):  # other sessions share the GPU
            try:
                det = detect(MIXTURES / name, out, cfg, device=args.device)
                break
            except RuntimeError as e:
                if "out of memory" not in str(e) or attempt == 29:
                    raise
                print("  GPU out of memory, retrying in 60 s", flush=True)
                time.sleep(60)
        t_det = time.perf_counter() - t0
        res = det.results
        t_track = track_results(res, cfg)
        res.save(out)
        summary, table = score(MIXTURES / name, out)
        summary |= {"tag": args.tag, "detect_s": t_det, "track_s": t_track}
        (out / "score.json").write_text(json.dumps(summary, indent=1))
        np.savez_compressed(out / "frames.npz", **table)
        print(
            f"{args.tag} {name}: recall {summary['recall']:.3f} idf1 {summary['idf1']:.3f} "
            f"precision {summary['precision']:.3f} ids {summary['ids_30s']} "
            f"({t_det:.0f} + {t_track:.0f} s)",
            flush=True,
        )


def overrides(sets) -> dict:
    """Nested config from ``section.key=value`` strings (values parsed as YAML)."""
    out: dict = {}
    for kv in sets or []:
        key, value = kv.split("=", 1)
        *path, last = key.split(".")
        d = out
        for k in path:
            d = d.setdefault(k, {})
        d[last] = yaml.safe_load(value)
    return out


def retrack(args):
    """Tracking (and linking or stitching) again on the detections of another
    run, with a config and overrides; detections, spectra and times stay."""
    from wavetracker.results import Results

    variant = yaml.safe_load(Path(args.config).read_text()) if args.config else {}
    variant = merge(variant or {}, overrides(args.set))
    for name in mixtures(args.split, args.only):
        src = RUNS / args.source / name
        out = RUNS / args.tag / name
        if not (src / "fund_v.npy").exists():
            print(f"{name}: no run {args.source}", flush=True)
            continue
        if (out / "score.json").exists() and not args.overwrite:
            continue
        band = yaml.safe_load((MIXTURES / name / "band.yaml").read_text())
        cfg = Config.from_dict(merge(variant, band))
        res = Results.load(src)
        t_track = track_results(res, cfg)
        res.meta["retracked_from"] = args.source
        res.save(out)
        summary, table = score(MIXTURES / name, out)
        summary |= {"tag": args.tag, "source": args.source, "track_s": t_track}
        (out / "score.json").write_text(json.dumps(summary, indent=1))
        np.savez_compressed(out / "frames.npz", **table)
        print(
            f"{args.tag} {name}: recall {summary['recall']:.3f} idf1 {summary['idf1']:.3f} "
            f"precision {summary['precision']:.3f} ids {summary['ids_30s']} "
            f"({t_track:.0f} s)",
            flush=True,
        )


def rescore(args):
    for name in mixtures(args.split, args.only):
        out = RUNS / args.tag / name
        if not (out / "fund_v.npy").exists():
            continue
        old = (
            json.loads((out / "score.json").read_text())
            if (out / "score.json").exists()
            else {}
        )
        summary, table = score(MIXTURES / name, out)
        summary |= {k: old[k] for k in ("tag", "detect_s", "track_s") if k in old}
        (out / "score.json").write_text(json.dumps(summary, indent=1))
        np.savez_compressed(out / "frames.npz", **table)
        print(f"rescored {name}", flush=True)


# --- report ------------------------------------------------------------------


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def scenario(name: str) -> str:
    return name.split("/")[1].rsplit("-s", 1)[0]


def load_scores(tag: str, split: str) -> list[dict]:
    out = []
    for name in mixtures(split):
        f = RUNS / tag / name / "score.json"
        if f.exists():
            out.append(json.loads(f.read_text()))
    return out


def load_frames(tag: str, names: list[str]) -> dict:
    """(fish, frame) tables of several mixtures; `mix` numbers the mixture."""
    parts = []
    for i, n in enumerate(names):
        p = dict(np.load(RUNS / tag / n / "frames.npz"))
        p["mix"] = np.full(len(p["fish"]), i, np.int16)
        parts.append(p)
    if not parts:
        return {}
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


METRICS = [
    ("recall", "completeness", "frames of a fish detected"),
    ("recall_visible", "completeness, visible", "frames >= 6 dB SNR detected"),
    ("idf1", "IDF1", "one identity per fish"),
    ("coverage", "coverage", "by dominant identity"),
    ("precision", "precision", "detections on a fish"),
]


def plot_summary(series, path):
    """One dot per mixture and metric; `series` is a list of (tag, split)."""
    import matplotlib.pyplot as plt

    rows = {(t, sp): load_scores(t, sp) for t, sp in series}
    scen = []
    for _, sp in series:
        for n in mixtures(sp):
            if scenario(n) not in scen:
                scen.append(scenario(n))
    y = np.arange(len(scen))[::-1]
    fig, axs = plt.subplots(
        1,
        len(METRICS) + 2,
        figsize=(17, 0.42 * len(scen) + 1.8),
        sharey=True,
        layout="constrained",
    )
    for j, (key, title, sub) in enumerate(
        [
            *METRICS,
            ("false_per_frame", "false / frame", "not on any fish or harmonic"),
            ("switches_per_fish_min", "switches", "per fish and minute"),
        ]
    ):
        ax = axs[j]
        for i, (t, sp) in enumerate(series):
            for s_i, sc in enumerate(scen):
                vals = [r[key] for r in rows[(t, sp)] if scenario(r["mixture"]) == sc]
                if not vals:
                    continue
                off = (i - (len(series) - 1) / 2) * 0.18
                ax.plot(
                    vals,
                    [y[s_i] + off] * len(vals),
                    "o",
                    ms=5.5,
                    color=SERIES[i],
                    mec="#fcfcfb",
                    mew=1.2,
                    label=f"{t}, {sp}" if s_i == 0 else None,
                )
        ax.set_title(f"{title}\n{sub}", loc="left", fontsize=9, color=INK)
        if key not in ("false_per_frame", "switches_per_fish_min"):
            ax.set_xlim(0, 1.02)
        else:
            ax.set_xlim(left=0)
        _style(ax)
        ax.grid(axis="y", visible=False)
    axs[0].set_yticks(y, scen, fontsize=8)
    fig.legend(
        *axs[0].get_legend_handles_labels(),
        loc="outside lower center",
        ncols=len(series),
        fontsize=8,
        frameon=False,
    )
    fig.suptitle(
        "dense-mixture benchmark (one dot per mixture)",
        x=0.01,
        ha="left",
        fontsize=11,
        color=INK,
    )
    fig.savefig(path, dpi=140)
    plt.close(fig)


CURVES = [
    ("snr_db", np.arange(-10, 62.5, 2.5), "SNR of the fundamental [dB]"),
    ("gain_db", np.arange(-60, 2.5, 2.5), "attenuation of the fish [dB]"),
    ("freq", None, "EOD frequency [Hz] (one dot per fish)"),
    (
        "nn_hz",
        np.array([0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 25, 30, 40, 60]),
        "distance to the nearest line of another fish [Hz]",
    ),
]
FREQ_TICKS = [50, 100, 200, 400, 700, 1000, 1500]


def plot_curves(tags, split, path, groups=None):
    """Completeness (solid) and frames on the fish's IDF1 identity (dashed)
    against covariates, pooled over the mixtures of each scenario group;
    against frequency one dot per fish (filled: completeness, open: on its
    identity)."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullLocator

    names = mixtures(split)
    groups = groups or {
        "density / crowded": [
            n for n in names if scenario(n).startswith(("density", "crowded"))
        ],
        "low frequencies": [n for n in names if scenario(n).startswith("low")],
        "attenuated": [n for n in names if scenario(n).startswith(("atten", "ladder"))],
    }
    fig, axs = plt.subplots(
        len(groups),
        len(CURVES),
        figsize=(17, 3.3 * len(groups)),
        layout="constrained",
        squeeze=False,
    )
    for gi, (gname, gnames) in enumerate(groups.items()):
        attenuated = False
        for i, t in enumerate(tags):
            fr = load_frames(
                t, [n for n in gnames if (RUNS / t / n / "frames.npz").exists()]
            )
            if not fr:
                continue
            attenuated |= bool((fr["gain_db"] < -1).any())
            for ci, (key, edges, _) in enumerate(CURVES):
                ax = axs[gi, ci]
                if key == "freq":
                    fish_key = fr["mix"].astype(np.int64) * 1000 + fr["fish"]
                    _, inv = np.unique(fish_key, return_inverse=True)
                    cnt = np.bincount(inv)
                    f = np.bincount(inv, fr["freq"]) / cnt
                    for what, face in (("matched", SERIES[i]), ("assigned", "none")):
                        v = np.bincount(inv, fr[what].astype(float)) / cnt
                        ax.scatter(
                            f, v, s=14, facecolors=face, edgecolors=SERIES[i], lw=0.8
                        )
                    continue
                x = fr[key]
                centers = 0.5 * (edges[1:] + edges[:-1])
                for what, ls in (("matched", "-"), ("assigned", "--")):
                    m, n = binned(x, fr[what], edges)
                    ok = n >= 50
                    ax.plot(
                        centers[ok],
                        m[ok],
                        ls,
                        color=SERIES[i],
                        lw=2,
                        label=f"{t}: {'completeness' if what == 'matched' else 'on its identity'}",
                    )
        for ci, (key, _, label) in enumerate(CURVES):
            ax = axs[gi, ci]
            if key == "gain_db" and not attenuated:
                ax.set_axis_off()
                ax.text(
                    0.5,
                    0.5,
                    "no attenuation\nin these scenarios",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=MUTED,
                    transform=ax.transAxes,
                )
                continue
            if key == "freq":
                ax.set_xscale("log")
                ax.xaxis.set_major_locator(FixedLocator(FREQ_TICKS))
                ax.xaxis.set_minor_locator(NullLocator())
                ax.set_xticklabels([str(t) for t in FREQ_TICKS])
            ax.set_ylim(0, 1.02)
            _style(ax)
            ax.set_xlabel(label, fontsize=8, color=MUTED)
            if ci == 0:
                ax.set_ylabel(
                    f"{gname}\nfraction of fish frames", fontsize=9, color=INK
                )
    axs[0, 0].legend(fontsize=7, frameon=False, loc="lower right")
    fig.suptitle(
        f"what limits tracking ({split}): solid / filled = detected (completeness), "
        "dashed / open = on the fish's one identity (IDF1)",
        x=0.01,
        ha="left",
        fontsize=11,
        color=INK,
    )
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_tracks(tag, name, path, tmax=None):
    """Overview spectrogram, truth, and detections coloured by what they are."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from score import classify_unmatched, match, truth_at

    from wavetracker.results import Results

    d = RUNS / tag / name
    res = Results.load(d)
    truth = dict(np.load(MIXTURES / name / "truth.npz"))
    ov = np.load(MIXTURES / name / "overview.npz")
    F = truth_at(truth, res.times)
    Ff = truth_at(truth, res.times, "freqs_filled")
    fish = match(res, F, TOL)
    cls = classify_unmatched(res, F, Ff, fish, TOL)
    sc = json.loads((d / "score.json").read_text())
    # identity assigned to each fish (IDF1): recompute from counts
    from scipy.optimize import linear_sum_assignment

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

    fig, ax = plt.subplots(figsize=(16, 9), layout="constrained")
    fmax = min(float(np.nanmax(truth["freqs_filled"])) * 1.08, ov["freqs"][-1])
    fmin = max(0.0, float(np.nanmin(truth["freqs_filled"])) * 0.85)
    fm = (ov["freqs"] >= fmin) & (ov["freqs"] <= fmax)
    db = 10 * np.log10(ov["power"][fm] + 1e-30)
    ax.pcolormesh(
        ov["times"],
        ov["freqs"][fm],
        db,
        cmap="gray_r",
        vmin=np.percentile(db, 40),
        vmax=np.percentile(db, 99.8),
        shading="auto",
        rasterized=True,
    )
    t = res.times[res.idx_v]
    kinds = [
        (on_own, SERIES[0], "on the fish's own identity"),
        ((cls == 0) & ~on_own, SERIES[1], "on a fish, other identity"),
        (cls == 2, "#eda100", "harmonic of a fish"),
        (cls == 3, CRITICAL, "false"),
    ]
    for mk, color, _ in kinds:
        ax.plot(t[mk], res.fund_v[mk], ".", ms=1.2, color=color, rasterized=True)
    ax.set_ylim(fmin, fmax)
    if tmax:
        ax.set_xlim(0, tmax)
    _style(ax)
    ax.grid(False)
    ax.set_xlabel("time [s]", fontsize=9, color=MUTED)
    ax.set_ylabel("frequency [Hz]", fontsize=9, color=MUTED)
    handles = [
        Line2D([], [], ls="", marker="o", ms=6, color=c_, label=l_)
        for _, c_, l_ in kinds
    ]
    ax.legend(
        handles=handles, loc="upper right", fontsize=8, frameon=True, framealpha=0.9
    )
    ax.set_title(
        f"{name} ({tag}): {sc['n_fish']} fish, recall {sc['recall']:.2f}, IDF1 {sc['idf1']:.2f}, "
        f"precision {sc['precision']:.2f}, {sc['ids_30s']} identities >= 30 s",
        loc="left",
        fontsize=10,
        color=INK,
    )
    fig.savefig(path, dpi=130)
    plt.close(fig)


def report(args):
    FIG.mkdir(parents=True, exist_ok=True)
    tags, splits = args.tags, args.splits
    series = [(t, sp) for sp in splits for t in tags]
    stem = "_".join(tags) + "_" + "_".join(splits)
    out = [FIG / f"summary_{stem}.png"]
    plot_summary(series, out[0])
    for sp in splits:
        p = FIG / f"curves_{'_'.join(tags)}_{sp}.png"
        plot_curves(tags, sp, p)
        out.append(p)
    for name in args.tracks or []:
        for t in tags:
            p = FIG / f"tracks_{t}_{name.replace('/', '_')}.png"
            plot_tracks(t, name, p)
            out.append(p)
    lines = []
    for t, sp in series:
        rows = load_scores(t, sp)
        by = {}
        for r in rows:
            by.setdefault(scenario(r["mixture"]), []).append(r)
        lines.append(f"\n{t} ({sp})")
        lines.append(
            f"{'scenario':16s} {'n':>2s} {'recall':>7s} {'r_vis':>6s} {'idf1':>6s} {'cover':>6s} "
            f"{'prec':>6s} {'false/fr':>8s} {'harm/fr':>7s} {'sw/f/min':>8s} {'ids30':>6s}"
        )
        for sc, rs in by.items():

            def mean(k, rs=rs):
                return np.mean([r[k] for r in rs])

            lines.append(
                f"{sc:16s} {len(rs):2d} {mean('recall'):7.3f} {mean('recall_visible'):6.3f} "
                f"{mean('idf1'):6.3f} "
                f"{mean('coverage'):6.3f} {mean('precision'):6.3f} "
                f"{mean('false_per_frame'):8.2f} {mean('harmonic_per_frame'):7.2f} "
                f"{mean('switches_per_fish_min'):8.2f} {mean('ids_30s'):6.1f}"
            )
    text = "\n".join(lines)
    print(text)
    (FIG / f"table_{stem}.txt").write_text(text)
    for p in out:
        print(p)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("track")
    a.add_argument("--tag", required=True)
    a.add_argument("-c", "--config")
    a.add_argument("--split", default="dev")
    a.add_argument("--only", nargs="*")
    a.add_argument("--device", default="auto")
    a.add_argument("--overwrite", action="store_true")
    a.set_defaults(func=track)
    r = sub.add_parser("retrack", help="track the detections of another run again")
    r.add_argument("--tag", required=True)
    r.add_argument("--source", required=True, help="run tag with the detections")
    r.add_argument("-c", "--config")
    r.add_argument("--set", nargs="*", help="overrides, e.g. tracking.max_dt=20")
    r.add_argument("--split", default="dev")
    r.add_argument("--only", nargs="*")
    r.add_argument("--overwrite", action="store_true")
    r.set_defaults(func=retrack)
    b = sub.add_parser("rescore")
    b.add_argument("--tag", required=True)
    b.add_argument("--split", default="dev")
    b.add_argument("--only", nargs="*")
    b.set_defaults(func=rescore)
    c = sub.add_parser("report")
    c.add_argument("--tags", nargs="+", required=True)
    c.add_argument("--splits", nargs="+", default=["dev", "dev-field"])
    c.add_argument("--tracks", nargs="*", help="mixtures to draw track figures for")
    c.set_defaults(func=report)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
