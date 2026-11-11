"""Where the tracker's fragments mix fish, and what each situation costs.

The tracker (``tracking.track``) is run again on the detections of a run; its
raw fragments (before linking) are compared with the truth. Along every
fragment, two consecutive detections matched to different fish are a
*switch*. Each switch is put into one situation, the first that applies:

* ``crossing``: the two fish's truth lines cross within +-``CROSS_S`` s;
* ``rise``: one of the two fish rises (slope above ``RISE_SLOPE`` Hz/s, or
  more than ``RISE_DF`` Hz above its own baseline, the 10th percentile of the
  surrounding minute);
* ``gap``: the fragment bridges more than ``GAP_S`` s between the two;
* ``parallel <1.2`` / ``parallel 1.2-3`` / ``far >=3``: the distance of the
  two lines at the switch.

The cost of a situation: the IDF1 gained by cutting the fragments at its
switches, (a) with perfect joining (every piece labelled by its majority
fish, all pieces of a fish joined) and (b) with the config's linker. The
bound of (a) with all switches cut is the IDF1 of pure fragments; what is left
to the truth identities is what the tracker leaves unassigned.

    python benchmarks/dense/assoc_diag.py --tag A-oracle-snr0 -c benchmarks/dense/configs/dense-linked.yaml --split ldev
    python benchmarks/dense/assoc_diag.py plot diag_A-v1m-relinked_ldev "learned, lab" ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from run import FIG, MIXTURES, RUNS, merge, mixtures, scenario
from score import TOL, classify_unmatched, match, truth_at

from wavetracker.config import Config
from wavetracker.linking import link
from wavetracker.results import Results
from wavetracker.tracking import track

CROSS_S = 5.0
RISE_SLOPE = 0.3  # Hz/s
RISE_DF = 1.5  # Hz above the baseline
GAP_S = 1.0
SITUATIONS = [
    "crossing",
    "rise",
    "gap",
    "parallel <1.2",
    "parallel 1.2-3",
    "far >=3",
]


def idf1(ident, fish, cls, n_known):
    """IDF1 as in score.score."""
    has = np.isfinite(ident)
    ids, inv = np.unique(ident[has], return_inverse=True)
    if len(ids) == 0:
        return 0.0
    m = fish[has] >= 0
    counts = np.zeros((len(ids), int(fish.max()) + 1 if fish.max() >= 0 else 1), int)
    np.add.at(counts, (inv[m], fish[has][m]), 1)
    from scipy.optimize import linear_sum_assignment

    r, c = linear_sum_assignment(-counts)
    tp = counts[r, c].sum()
    n_pred = int((has & (cls != 1)).sum())
    fp = n_pred - tp
    fn = int(n_known) - tp
    return 2 * tp / max(2 * tp + fp + fn, 1)


def majority(ident, fish):
    """Every fragment labelled by its majority fish (perfect joining);
    fragments without any matched detection keep their own identity."""
    out = np.full(len(ident), np.nan)
    has = np.isfinite(ident)
    ids, inv = np.unique(ident[has], return_inverse=True)
    pos = np.flatnonzero(has)
    nf = int(fish.max()) + 1
    counts = np.zeros((len(ids), nf + 1), int)
    np.add.at(counts, (inv, np.where(fish[pos] >= 0, fish[pos], nf)), 1)
    best = counts[:, :nf].argmax(1)
    any_m = counts[:, :nf].sum(1) > 0
    lab = np.where(any_m, best, nf + 1 + np.arange(len(ids)))
    out[pos] = lab[inv]
    return out


def baselines(F, times, win_s=60.0):
    """10th percentile of each fish's frequency over +-win_s/2 (rolling)."""
    dt = times[1] - times[0]
    h = max(1, int(round(win_s / 2 / dt)))
    out = np.full(F.shape, np.nan)
    step = max(1, int(round(2.0 / dt)))
    for k in range(len(F)):
        f = F[k]
        cs = np.arange(0, F.shape[1], step)
        vals = np.array(
            [
                np.nanpercentile(f[max(0, c - h) : c + h + 1], 10)
                if np.isfinite(f[max(0, c - h) : c + h + 1]).any()
                else np.nan
                for c in cs
            ]
        )
        ok = np.isfinite(vals)
        if ok.sum() >= 2:
            out[k] = np.interp(np.arange(F.shape[1]), cs[ok], vals[ok])
    return out


def switches(raw, fish, res, Ff, base):
    """All switches of the raw fragments, with their situation."""
    times = res.times
    dt = times[1] - times[0]
    cross_n = int(round(CROSS_S / dt))
    slope_n = max(1, int(round(1.0 / dt)))
    nfr = Ff.shape[1]
    rows = []
    has = np.isfinite(raw) & (fish >= 0)
    for i in np.unique(raw[has]):
        m = np.flatnonzero(has & (raw == i))
        m = m[np.argsort(res.idx_v[m], kind="stable")]
        fa = fish[m]
        ch = np.flatnonzero(fa[1:] != fa[:-1])
        for c in ch:
            d0, d1 = m[c], m[c + 1]
            A, B = int(fish[d0]), int(fish[d1])
            j0, j1 = int(res.idx_v[d0]), int(res.idx_v[d1])
            lo, hi = max(0, j0 - cross_n), min(nfr, j1 + cross_n + 1)
            diff = Ff[A, lo:hi] - Ff[B, lo:hi]
            diff = diff[np.isfinite(diff)]
            crossing = len(diff) > 1 and (diff.min() < 0 < diff.max())
            sep = abs(Ff[A, j1] - Ff[B, j1]) if np.isfinite(Ff[A, j1] - Ff[B, j1]) else abs(
                res.fund_v[d0] - res.fund_v[d1]
            )
            rising = False
            for k, j in ((A, j0), (B, j1)):
                a, b = max(0, j - slope_n), min(nfr - 1, j + slope_n)
                if np.isfinite(Ff[k, a]) and np.isfinite(Ff[k, b]):
                    if abs(Ff[k, b] - Ff[k, a]) / ((b - a) * dt) > RISE_SLOPE:
                        rising = True
                if np.isfinite(Ff[k, j]) and np.isfinite(base[k, j]):
                    if Ff[k, j] - base[k, j] > RISE_DF:
                        rising = True
            gap = (j1 - j0) * dt
            if crossing:
                sit = "crossing"
            elif rising:
                sit = "rise"
            elif gap > GAP_S:
                sit = "gap"
            elif sep < 1.2:
                sit = "parallel <1.2"
            elif sep < 3.0:
                sit = "parallel 1.2-3"
            else:
                sit = "far >=3"
            rows.append(
                {
                    "frag": float(i),
                    "pos": int(c + 1),  # cut before this member (in time order)
                    "det": int(d1),
                    "A": A,
                    "B": B,
                    "t": float(times[j1]),
                    "gap": gap,
                    "sep": float(sep),
                    "crossing": bool(crossing),
                    "rising": bool(rising),
                    "situation": sit,
                }
            )
    return rows


BREAKS = [
    "dropout >=10 s",
    "gap 1-10 s",
    "crossing",
    "rise",
    "neighbour <1.2",
    "neighbour 1.2-3",
    "alone",
]


def breaks(raw, final, fish, res, Ff, base, nn, max_dt):
    """Where a fish's consecutive matched detections lie in different raw
    fragments (or the fish is not assigned in between), with the situation
    and whether the final identities join it."""
    times = res.times
    dt = times[1] - times[0]
    cross_n = int(round(CROSS_S / dt))
    slope_n = max(1, int(round(1.0 / dt)))
    nfr = Ff.shape[1]
    rows = []
    has = np.isfinite(raw) & (fish >= 0)
    for k in np.unique(fish[has]):
        m = np.flatnonzero(has & (fish == k))
        m = m[np.argsort(res.idx_v[m], kind="stable")]
        r = raw[m]
        for c in np.flatnonzero(r[1:] != r[:-1]):
            d0, d1 = m[c], m[c + 1]
            j0, j1 = int(res.idx_v[d0]), int(res.idx_v[d1])
            gap = (j1 - j0) * dt
            lo, hi = max(0, j0 - cross_n), min(nfr, j1 + cross_n + 1)
            crossing = False
            for o in range(len(Ff)):
                if o == k:
                    continue
                d = Ff[k, lo:hi] - Ff[o, lo:hi]
                d = d[np.isfinite(d)]
                if len(d) > 1 and d.min() < 0 < d.max():
                    crossing = True
                    break
            rising = False
            for j in (j0, j1):
                a, b = max(0, j - slope_n), min(nfr - 1, j + slope_n)
                if np.isfinite(Ff[k, a]) and np.isfinite(Ff[k, b]):
                    if abs(Ff[k, b] - Ff[k, a]) / ((b - a) * dt) > RISE_SLOPE:
                        rising = True
                if np.isfinite(Ff[k, j]) and np.isfinite(base[k, j]):
                    if Ff[k, j] - base[k, j] > RISE_DF:
                        rising = True
            near = np.nanmin(nn[k, j0 : j1 + 1]) if j1 >= j0 else np.inf
            if gap >= max_dt:
                sit = "dropout >=10 s"
            elif gap > GAP_S:
                sit = "gap 1-10 s"
            elif crossing:
                sit = "crossing"
            elif rising:
                sit = "rise"
            elif near < 1.2:
                sit = "neighbour <1.2"
            elif near < 3.0:
                sit = "neighbour 1.2-3"
            else:
                sit = "alone"
            # does the old fragment go on with another fish / was the new one
            # on another fish before?
            old_on = np.flatnonzero(
                (raw == r[c]) & (res.idx_v > j0) & (res.idx_v <= j0 + cross_n)
            )
            new_before = np.flatnonzero(
                (raw == r[c + 1]) & (res.idx_v < j1) & (res.idx_v >= j1 - cross_n)
            )
            rows.append(
                {
                    "fish": int(k),
                    "a": float(r[c]),
                    "b": float(r[c + 1]),
                    "t": float(times[j1]),
                    "gap": gap,
                    "near": float(near),
                    "situation": sit,
                    "old_continues": bool(len(old_on)),
                    "new_existed": bool(len(new_before)),
                    "joined": bool(final[d0] == final[d1]),
                    "jump": float(res.fund_v[d1] - res.fund_v[d0]),
                    "slope": float(
                        (Ff[k, min(nfr - 1, j1 + slope_n)] - Ff[k, max(0, j1 - slope_n)])
                        / (2 * slope_n * dt)
                    ),
                    "above": float(Ff[k, j1] - base[k, j1]),
                    "fa": float(final[d0]),
                    "fb": float(final[d1]),
                }
            )
    return rows


def force_join(final, rows, keep, fish):
    """Final identities with the unjoined breaks in `keep` joined, where both
    identities belong mostly to the break's fish (an oracle join)."""
    ids = np.unique(final[np.isfinite(final)])
    parent = {i: i for i in ids}
    maj = {}
    has = np.isfinite(final) & (fish >= 0)
    for i in ids:
        f = fish[has & (final == i)]
        maj[i] = int(np.bincount(f).argmax()) if len(f) else -1

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for r in rows:
        if r["situation"] in keep and not r["joined"]:
            if np.isfinite(r["fa"]) and np.isfinite(r["fb"]):
                if maj[r["fa"]] == r["fish"] == maj[r["fb"]]:
                    parent[find(r["fa"])] = find(r["fb"])
    out = final.copy()
    ok = np.isfinite(final)
    out[ok] = [find(i) for i in final[ok]]
    return out


def cut(raw, res, rows, keep):
    """Fragments cut at the switches in `rows` whose situation is in `keep`."""
    out = raw.copy()
    nxt = np.nanmax(raw) + 1 if np.isfinite(raw).any() else 0
    by = {}
    for r in rows:
        if r["situation"] in keep:
            by.setdefault(r["frag"], []).append(r["t"])
    t = res.times[res.idx_v]
    for i, ts in by.items():
        m = np.flatnonzero(raw == i)
        piece = np.searchsorted(np.sort(ts), t[m], side="right")
        # detections at or after a switch time go to the next piece
        for p in range(1, len(ts) + 1):
            out[m[piece == p]] = nxt
            nxt += 1
    return out


def analyse(name, src_tag, cfg, with_linker=True):
    res = Results.load(RUNS / src_tag / name)
    truth = dict(np.load(MIXTURES / name / "truth.npz"))
    F = truth_at(truth, res.times)
    Ff = truth_at(truth, res.times, "freqs_filled")
    fish = match(res, F, TOL)
    cls = classify_unmatched(res, F, Ff, fish, TOL)
    n_known = int(np.isfinite(F).sum())
    raw = track(
        res.fund_v,
        res.idx_v,
        res.sign_v,
        res.times,
        cfg.tracking,
        min_freq=cfg.harmonic_groups.min_freq,
        max_freq=cfg.harmonic_groups.max_freq,
        sal_v=res.sal_v,
    )
    base = baselines(Ff, res.times)
    rows = switches(raw, fish, res, Ff, base)

    def lnk(ident):
        return link(
            res.fund_v, res.idx_v, res.sign_v, res.cplx_v, ident, res.times, cfg.linking
        )

    truth_ids = np.where(fish >= 0, fish, np.nan).astype(float)
    kept = np.where(np.isfinite(raw) & (fish >= 0), fish, np.nan).astype(float)
    out = {
        "mixture": name,
        "n_known": n_known,
        "n_det": len(fish),
        "n_assigned": int(np.isfinite(raw).sum()),
        "n_switches": len(rows),
        "contam": {
            k: int((np.isfinite(raw) & (cls == v)).sum())
            for k, v in (("matched", 0), ("ignored", 1), ("harmonic", 2), ("false", 3))
        },
        "idf1_truth_ids": idf1(truth_ids, fish, cls, n_known),
        "idf1_kept_ids": idf1(kept, fish, cls, n_known),
        "idf1_raw_majority": idf1(majority(raw, fish), fish, cls, n_known),
        "idf1_pure_majority": idf1(
            majority(cut(raw, res, rows, SITUATIONS), fish), fish, cls, n_known
        ),
        "idf1_raw": idf1(raw, fish, cls, n_known),
        "cost_majority": {},
        "cost_linker": {},
        "switches": rows,
    }
    if with_linker:
        out["idf1_linked"] = idf1(lnk(raw), fish, cls, n_known)
        out["idf1_pure_linked"] = idf1(
            lnk(cut(raw, res, rows, SITUATIONS)), fish, cls, n_known
        )
    for s in SITUATIONS:
        c = cut(raw, res, rows, {s})
        out["cost_majority"][s] = idf1(majority(c, fish), fish, cls, n_known) - out[
            "idf1_raw_majority"
        ]
        if with_linker:
            out["cost_linker"][s] = idf1(lnk(c), fish, cls, n_known) - out["idf1_linked"]
    if with_linker:
        from score import nearest_line

        final = lnk(raw)
        brows = breaks(raw, final, fish, res, Ff, base, nearest_line(Ff), cfg.tracking.max_dt)
        out["breaks"] = brows
        out["cost_break"] = {
            s: idf1(force_join(final, brows, {s}, fish), fish, cls, n_known)
            - out["idf1_linked"]
            for s in BREAKS
        }
        out["idf1_all_breaks_joined"] = idf1(
            force_join(final, brows, set(BREAKS), fish), fish, cls, n_known
        )
        # fragments the linker never sees (fewer than min_detections)
        has = np.isfinite(raw)
        _, cnt = np.unique(raw[has], return_counts=True)
        out["short_frag_dets"] = int(cnt[cnt < cfg.linking.min_detections].sum())
    return out


def summarize(results, label):
    n = len(results)

    def mean(k):
        return np.mean([r[k] for r in results])

    print(f"\n== {label}: {n} mixtures")
    for k in (
        "idf1_raw",
        "idf1_linked",
        "idf1_pure_linked",
        "idf1_raw_majority",
        "idf1_pure_majority",
        "idf1_kept_ids",
        "idf1_truth_ids",
    ):
        if k in results[0]:
            print(f"  {k:22s} {mean(k):.4f}")
    allrows = [s for r in results for s in r["switches"]]
    tot = len(allrows)
    print(f"  switches per mixture {tot / n:.1f}")
    print(
        f"  {'situation':16s} {'n/mix':>7s} {'share':>6s} {'cost(perfect join)':>19s} {'cost(linker)':>13s}"
    )
    for s in SITUATIONS:
        k = sum(1 for x in allrows if x["situation"] == s)
        cm = np.mean([r["cost_majority"][s] for r in results])
        cl = np.mean([r["cost_linker"].get(s, np.nan) for r in results])
        print(f"  {s:16s} {k / n:7.1f} {k / max(tot, 1):6.2f} {cm:19.4f} {cl:13.4f}")
    if "breaks" in results[0]:
        b = [x for r in results for x in r["breaks"]]
        print(
            f"  breaks per mixture {len(b) / n:.1f}, joined by the linker "
            f"{np.mean([x['joined'] for x in b]):.2f}; all unjoined joined: "
            f"{np.mean([r['idf1_all_breaks_joined'] for r in results]):.4f}; "
            f"short-fragment detections {np.mean([r['short_frag_dets'] / r['n_assigned'] for r in results]):.3f}"
        )
        print(
            f"  {'break':16s} {'n/mix':>7s} {'joined':>7s} {'old goes on':>11s} "
            f"{'new existed':>11s} {'cost(unjoined)':>14s}"
        )
        for s in BREAKS:
            xs = [x for x in b if x["situation"] == s]
            if not xs:
                continue
            print(
                f"  {s:16s} {len(xs) / n:7.1f} {np.mean([x['joined'] for x in xs]):7.2f} "
                f"{np.mean([x['old_continues'] for x in xs]):11.2f} "
                f"{np.mean([x['new_existed'] for x in xs]):11.2f} "
                f"{np.mean([r['cost_break'][s] for r in results]):14.4f}"
            )
    contam = {
        k: sum(r["contam"][k] for r in results) for k in results[0]["contam"]
    }
    tot_a = sum(contam.values())
    print("  assigned detections: " + ", ".join(f"{k} {v / tot_a:.3f}" for k, v in contam.items()))
    # by scenario group
    groups = {}
    for r in results:
        groups.setdefault(scenario(r["mixture"]).split("-")[0], []).append(r)
    print("  by group: raw-majority -> pure-majority (perfect join), switches/mix")
    for g, rs in groups.items():
        print(
            f"    {g:10s} {np.mean([r['idf1_raw_majority'] for r in rs]):.3f} -> "
            f"{np.mean([r['idf1_pure_majority'] for r in rs]):.3f}  "
            f"{np.mean([r['n_switches'] for r in rs]):6.1f}"
        )


def plot(stems, labels, path):
    """IDF1 cost of each situation, switches and breaks, one bar per run."""
    import matplotlib.pyplot as plt

    colors = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
    data = [json.loads((FIG / "assoc" / f"{s}.json").read_text()) for s in stems]
    panels = [
        ("switches inside fragments\n(cut there, then perfect joining)", SITUATIONS, "cost_majority"),
        ("breaks left open by the linker\n(oracle join of those breaks)", BREAKS, "cost_break"),
    ]
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), layout="constrained")
    for ax, (title, cats, key) in zip(axs, panels, strict=True):
        y = np.arange(len(cats))[::-1]
        h = 0.8 / len(data)
        for i, (res, lab) in enumerate(zip(data, labels, strict=True)):
            v = [np.mean([r[key].get(c, 0.0) for r in res]) for c in cats]
            yy = y + (len(data) - 1) / 2 * h - i * h
            ax.barh(yy, v, h * 0.85, color=colors[i], label=lab)
            for yv, vv in zip(yy, v, strict=True):
                if vv >= 0.002:
                    ax.text(vv + 0.0003, yv, f"{vv:.3f}", va="center", fontsize=7, color="#52514e")
        ax.set_yticks(y, cats, fontsize=9)
        ax.set_xlabel("IDF1 lost (mean of 17 mixtures)", fontsize=9, color="#52514e")
        ax.set_title(title, loc="left", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#e6e5e0", lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlim(left=0)
    axs[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(
        "where association loses IDF1 on ldev (raw tracker fragments vs truth)",
        x=0.01, ha="left", fontsize=11,
    )
    fig.savefig(path, dpi=140)
    print(path)


def main(argv=None):
    if argv is None and len(sys.argv) > 1 and sys.argv[1] == "plot":
        stems = sys.argv[2::2]
        labels = sys.argv[3::2]
        plot(stems, labels, FIG / "assoc_diagnosis.png")
        return
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tag", required=True, help="run with the detections")
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--split", default="ldev")
    p.add_argument("--only", nargs="*")
    p.add_argument("--no-linker", action="store_true")
    p.add_argument("--out", default=None, help="json name in FIG/assoc/")
    args = p.parse_args(argv)
    from run import overrides

    variant = merge(yaml.safe_load(Path(args.config).read_text()) or {}, overrides(args.set))
    results = []
    for name in mixtures(args.split, args.only):
        band = yaml.safe_load((MIXTURES / name / "band.yaml").read_text())
        cfg = Config.from_dict(merge(variant, band))
        r = analyse(name, args.tag, cfg, with_linker=not args.no_linker)
        results.append(r)
        print(
            f"{name}: raw {r['idf1_raw']:.3f} linked {r.get('idf1_linked', np.nan):.3f} "
            f"majority {r['idf1_raw_majority']:.3f} pure {r['idf1_pure_majority']:.3f} "
            f"switches {r['n_switches']}",
            flush=True,
        )
    summarize(results, f"{args.tag} {args.split}")
    out = FIG / "assoc"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.out or 'diag_' + args.tag + '_' + args.split}.json"
    path.write_text(json.dumps(results))
    print(path)


if __name__ == "__main__":
    main()
