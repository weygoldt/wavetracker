"""What happens to a fish's identity at its rises, and what each outcome costs.

A *rise* is a fast upward jump of a fish's frequency (at least ``RISE_MIN`` Hz
within about 1 s) that stays up for a while and decays back over seconds to a
minute. Rises are found in the truth (``find_rises``) from the frequency
slope; chirps (shorter than ``PERSIST_S``) are left out.

For every rise, the fish's matched detections before the onset
(``PRE_S`` s) give its *home* identity (the most common final identity
there); the detections from the onset to the end of the decay are compared
with it. Outcomes, first match wins:

* ``undetected``: fewer than ``MIN_DETS`` matched detections before or during
  the rise;
* ``pre-wrong``: the home identity belongs mostly to another fish (the fish
  was not on its own identity before the rise; not a rise problem);
* ``switched``: during the rise the fish is on an identity that belongs
  mostly to another fish (for at least ``MIN_DETS`` detections);
* ``unjoined``: during the rise the fish is on an identity of its own, but
  not on the home identity (a break never joined);
* ``dropped``: detections of the fish during the rise have no identity;
* ``tracked`` / ``linked``: the fish keeps its home identity throughout; the
  tracker's raw fragments kept it too, or only the linker joined it.

The cost of an outcome: the IDF1 gained by an oracle that repairs the rises
of that outcome only: every identity mostly of the fish that appears during
the rise is joined to the home identity, and the fish's other detections
during the rise are moved to it.

Identity coverage around rises: the fish's known frames within
+-``COVER_S`` s of an onset that the fish's IDF1 identity holds (``assigned``
in ``frames.npz``), against the frames away from rises.

    python benchmarks/dense/rise_diag.py --tag B-rise-base -c benchmarks/dense/configs/learned-b.yaml --split ldev
    python benchmarks/dense/rise_diag.py plot
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from assoc_diag import idf1
from run import FIG, MIXTURES, RUNS, merge, mixtures, overrides, scenario
from score import TOL, classify_unmatched, match, truth_at

from wavetracker.config import Config
from wavetracker.results import Results
from wavetracker.tracking import track

RISE_MIN = 2.0  # Hz, smallest jump within JUMP_S
JUMP_S = 1.0  # s
SMOOTH_S = 0.5  # s, median filter against single-sample spikes
PERSIST_S = 2.0  # s the frequency must stay above half the jump
MAX_DECAY_S = 60.0  # s, longest decay considered part of the rise
PRE_S = 10.0  # s before the onset that give the home identity
MIN_DETS = 3
COVER_S = 30.0  # s, identity coverage around rises
UNREPAIRABLE = {"pre-wrong", "undetected"}  # no home identity of the fish
OUTCOMES = [
    "tracked",
    "linked",
    "unjoined",
    "switched",
    "dropped",
    "pre-wrong",
    "undetected",
]


def _medfilt(f, n):
    """Running median over n samples, NaN-aware (n odd)."""
    h = n // 2
    pad = np.concatenate([np.full(h, np.nan), f, np.full(h, np.nan)])
    win = np.lib.stride_tricks.sliding_window_view(pad, n)
    with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore")
        return np.nanmedian(win, axis=1)


def find_rises(times: np.ndarray, F: np.ndarray) -> list[dict]:
    """Rises of every fish in a truth (fish, samples) at `times`."""
    dt = times[1] - times[0]
    n_s = max(1, round(SMOOTH_S / dt)) | 1
    n_j = max(1, round(JUMP_S / dt))
    n_p = max(1, round(PERSIST_S / dt))
    out = []
    for k, f in enumerate(F):
        fs = _medfilt(f, n_s)
        d = np.full(len(fs), np.nan)
        d[:-n_j] = fs[n_j:] - fs[:-n_j]
        cand = np.flatnonzero(d > RISE_MIN)
        if len(cand) == 0:
            continue
        # runs of candidate samples (gaps up to JUMP_S)
        splits = np.flatnonzero(np.diff(cand) > n_j) + 1
        last_peak = -np.inf
        for run in np.split(cand, splits):
            # onset: the sample before the steepest step of the run
            seg = np.arange(run[0], min(run[-1] + n_j, len(fs) - 1))
            step = fs[seg + 1] - fs[seg]
            if not np.isfinite(step).any():
                continue
            i0 = int(seg[np.nanargmax(step)])
            # the onset proper: back to where the climb starts
            while i0 > 0 and fs[i0] - fs[i0 - 1] > 0.1 and fs[i0 + 1] - fs[i0 - 1] > 0:
                i0 -= 1
            if i0 <= last_peak:
                continue
            pre = fs[max(0, i0 - round(3.0 / dt)) : max(0, i0 - round(0.3 / dt))]
            if not np.isfinite(pre).any():
                continue
            f_pre = float(np.nanmedian(pre))
            w = fs[i0 : i0 + round(3.0 / dt) + 1]
            if not np.isfinite(w).any():
                continue
            ip = i0 + int(np.nanargmax(w))
            size = float(fs[ip] - f_pre)
            if size < RISE_MIN:
                continue
            after = fs[ip : ip + n_p]
            if not (np.nanmedian(after) > f_pre + 0.5 * size):
                continue  # a chirp, not a rise
            # end of the decay: back within max(0.5 Hz, 20 %) of f_pre
            tail = fs[ip : ip + round(MAX_DECAY_S / dt)]
            back = np.flatnonzero(tail <= f_pre + max(0.5, 0.2 * size))
            ie = ip + (int(back[0]) if len(back) else len(tail) - 1)
            half = np.flatnonzero(tail <= f_pre + 0.5 * size)
            out.append(
                {
                    "fish": int(k),
                    "t0": float(times[i0]),
                    "t_peak": float(times[ip]),
                    "t_end": float(times[min(ie, len(times) - 1)]),
                    "f_pre": f_pre,
                    "size": size,
                    "rise_s": float((ip - i0) * dt),
                    "half_s": float(half[0] * dt) if len(half) else np.nan,
                    "decayed": bool(len(back)),
                }
            )
            last_peak = ip
    # a rise ends at the latest at the next onset of the same fish
    by = {}
    for r in out:
        by.setdefault(r["fish"], []).append(r)
    for rs in by.values():
        rs.sort(key=lambda r: r["t0"])
        for a, b in zip(rs[:-1], rs[1:], strict=True):
            a["t_end"] = min(a["t_end"], b["t0"])
    return sorted(out, key=lambda r: (r["fish"], r["t0"]))


def _mode(x):
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    v, c = np.unique(x, return_counts=True)
    return float(v[c.argmax()])


def majority_fish(ident, fish):
    """Majority fish of every identity (-1: no matched detection)."""
    has = np.isfinite(ident)
    ids, inv = np.unique(ident[has], return_inverse=True)
    f = fish[has]
    out = {}
    for j, i in enumerate(ids):
        ff = f[inv == j]
        ff = ff[ff >= 0]
        out[float(i)] = int(np.bincount(ff).argmax()) if len(ff) else -1
    return out


def outcomes(rises, final, raw, fish, td, maj):
    """Outcome of every rise (adds keys to the rise dicts)."""
    for r in rises:
        k = r["fish"]
        mk = fish == k
        pre = mk & (td >= r["t0"] - PRE_S) & (td < r["t0"])
        dur = mk & (td >= r["t0"]) & (td <= max(r["t_end"], r["t0"] + 1.0))
        r["n_pre"], r["n_dur"] = int(pre.sum()), int(dur.sum())
        home = _mode(final[pre])
        r["home"] = home
        if pre.sum() < MIN_DETS or dur.sum() < MIN_DETS:
            r["outcome"] = "undetected"
            continue
        if not np.isfinite(home) or maj.get(home, -1) != k:
            r["outcome"] = "pre-wrong"
            continue
        fd = final[dur]
        on_home = fd == home
        nan = ~np.isfinite(fd)
        others = fd[~on_home & ~nan]
        own = np.array([maj.get(float(i), -1) == k for i in others], bool)
        r["n_home"] = int(on_home.sum())
        r["n_own_other"] = int(own.sum())
        r["n_foreign"] = int((~own).sum())
        r["n_nan"] = int(nan.sum())
        r["other_ids"] = sorted({float(i) for i in others})
        if r["n_foreign"] >= MIN_DETS:
            r["outcome"] = "switched"
        elif r["n_own_other"] >= MIN_DETS:
            r["outcome"] = "unjoined"
        elif r["n_nan"] >= MIN_DETS and r["n_nan"] > r["n_home"] // 4:
            r["outcome"] = "dropped"
        else:
            rh = _mode(raw[pre])
            rd = raw[dur]
            r["outcome"] = (
                "tracked" if np.isfinite(rh) and np.mean(rd == rh) >= 0.9 else "linked"
            )
        # the first detection of the fish off the home identity: the break
        off = np.flatnonzero(dur)[~on_home]
        if len(off):
            first = off[np.argmin(td[off])]
            last_home = np.flatnonzero(mk & (final == home) & (td < td[first]))
            if len(last_home):
                lh = last_home[np.argmax(td[last_home])]
                r["break_t"] = float(td[first])
                r["break_gap"] = float(td[first] - td[lh])
                r["break_jump"] = None  # filled by the caller (needs fund_v)
                r["_break"] = (int(lh), int(first))
    return rises


def repair(final, rises, keep, fish, td, maj):
    """Oracle repair of the rises whose outcome is in `keep`: identities
    mostly of the fish that appear during the rise are joined to its home
    identity, its other detections during the rise are moved there."""
    out = final.copy()
    parent = {}

    def find(i):
        while parent.get(i, i) != i:
            i = parent[i]
        return i

    moves = []
    for r in rises:
        if r["outcome"] not in keep or r["outcome"] in UNREPAIRABLE:
            continue
        k, home = r["fish"], r["home"]
        dur = (fish == k) & (td >= r["t0"]) & (td <= max(r["t_end"], r["t0"] + 1.0))
        for i in np.unique(final[dur & np.isfinite(final)]):
            if maj.get(float(i), -1) == k and find(float(i)) != find(home):
                parent[find(float(i))] = find(home)
        moves.append((np.flatnonzero(dur), home))
    ok = np.isfinite(out)
    out[ok] = [find(float(i)) for i in out[ok]]
    for d, home in moves:
        out[d] = find(home)
    return out


def coverage(table, rises, times, n_fish):
    """Assigned fraction of known frames within +-COVER_S of a rise onset,
    and away from rises."""
    near = np.zeros(len(table["fish"]), bool)
    t = times[table["frame"]]
    for r in rises:
        near |= (table["fish"] == r["fish"]) & (np.abs(t - r["t0"]) <= COVER_S)
    a = table["assigned"]
    return (
        int(a[near].sum()),
        int(near.sum()),
        int(a[~near].sum()),
        int((~near).sum()),
    )


def analyse(job):
    name, tag, variant = job
    run = RUNS / tag / name
    res = Results.load(run)
    band = yaml.safe_load((MIXTURES / name / "band.yaml").read_text())
    cfg = Config.from_dict(merge(variant, band))
    truth = dict(np.load(MIXTURES / name / "truth.npz"))
    F = truth_at(truth, res.times)
    Ff = truth_at(truth, res.times, "freqs_filled")
    fish = match(res, F, TOL)
    cls = classify_unmatched(res, F, Ff, fish, TOL)
    n_known = int(np.isfinite(F).sum())
    rises = find_rises(truth["times"], truth["freqs_filled"])
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
    final = res.ident_v
    td = res.times[res.idx_v]
    maj = majority_fish(final, fish)
    outcomes(rises, final, raw, fish, td, maj)
    for r in rises:
        b = r.pop("_break", None)
        if b is not None:
            r["break_jump"] = float(res.fund_v[b[1]] - res.fund_v[b[0]])
    base = idf1(final, fish, cls, n_known)
    cost = {}
    for o in OUTCOMES + ["all"]:
        keep = set(OUTCOMES) if o == "all" else {o}
        if not any(r["outcome"] in keep for r in rises):
            cost[o] = 0.0
            continue
        cost[o] = idf1(repair(final, rises, keep, fish, td, maj), fish, cls, n_known) - base
    table = dict(np.load(run / "frames.npz"))
    cov = coverage(table, rises, res.times, len(F))
    score = json.loads((run / "score.json").read_text())
    return {
        "mixture": name,
        "idf1": base,
        "idf1_scored": score["idf1"],
        "rises": rises,
        "cost": cost,
        "cover": cov,
        "score": {k: score[k] for k in ("idf1", "purity", "switches_per_fish_min", "false_ids_30s", "ids_30s")},
    }


def summarize(results, label):
    n = len(results)
    rs = [r for x in results for r in x["rises"]]
    print(f"\n== {label}: {n} mixtures, {len(rs)} rises ({len(rs) / n:.1f} per mixture)")
    print(f"  IDF1 {np.mean([x['idf1'] for x in results]):.4f}")
    print(f"  {'outcome':12s} {'n/mix':>6s} {'share':>6s} {'cost':>7s}")
    for o in OUTCOMES + ["all"]:
        c = sum(1 for r in rs if o == "all" or r["outcome"] == o)
        cost = np.mean([x["cost"][o] for x in results])
        print(f"  {o:12s} {c / n:6.1f} {c / max(len(rs), 1):6.2f} {cost:7.4f}")
    a = np.sum([x["cover"] for x in results], axis=0)
    print(
        f"  identity coverage within +-{COVER_S:.0f} s of onsets {a[0] / max(a[1], 1):.3f} "
        f"({a[1]} fish frames), elsewhere {a[2] / max(a[3], 1):.3f}"
    )
    br = [r for r in rs if r["outcome"] in ("unjoined", "switched") and r.get("break_jump") is not None]
    if br:
        j = np.array([r["break_jump"] for r in br])
        g = np.array([r["break_gap"] for r in br])
        bt = np.array([r["break_t"] - r["t0"] for r in br])
        print(
            f"  breaks (unjoined+switched): jump > 5 Hz {np.mean(j > 5):.2f}, 2-5 Hz "
            f"{np.mean((j > 2) & (j <= 5)):.2f}, < 2 Hz {np.mean(np.abs(j) <= 2):.2f}; "
            f"at onset (< 3 s) {np.mean(bt < 3):.2f}; gap median {np.median(g):.2f} s"
        )
    sizes = np.array([r["size"] for r in rs])
    print(
        "  rise size (Hz) quartiles "
        + " ".join(f"{q:.1f}" for q in np.quantile(sizes, [0.25, 0.5, 0.75]))
        + f"; half-decay median {np.nanmedian([r['half_s'] for r in rs]):.1f} s"
    )
    groups = {}
    for x in results:
        groups.setdefault(scenario(x["mixture"]).split("-")[0], []).append(x)
    for g, xs in groups.items():
        r_g = [r for x in xs for r in x["rises"]]
        print(
            f"    {g:9s} rises/mix {len(r_g) / len(xs):5.1f}  cost(all) "
            f"{np.mean([x['cost']['all'] for x in xs]):.4f}  "
            + " ".join(
                f"{o[:5]} {sum(r['outcome'] == o for r in r_g) / len(xs):.1f}"
                for o in OUTCOMES
            )
        )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tag", required=True, help="run (final identities)")
    p.add_argument("-c", "--config", required=True, help="its tracking config")
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--split", default="ldev")
    p.add_argument("--only", nargs="*")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out", default=None, help="json name in FIG/rises/")
    args = p.parse_args(argv)
    variant = merge(
        yaml.safe_load(Path(args.config).read_text()) or {}, overrides(args.set)
    )
    jobs = [(n, args.tag, variant) for n in mixtures(args.split, args.only)]
    with ProcessPoolExecutor(args.workers) as ex:
        results = list(ex.map(analyse, jobs))
    for x in results:
        print(
            f"{x['mixture']}: idf1 {x['idf1']:.3f} rises {len(x['rises'])} cost {x['cost']['all']:.4f}",
            flush=True,
        )
    summarize(results, f"{args.tag} {args.split}")
    out = FIG / "rises"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.out or args.tag + '_' + args.split}.json"
    path.write_text(json.dumps(results, default=float))
    print(path)


if __name__ == "__main__":
    main()
