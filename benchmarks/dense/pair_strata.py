"""Close pairs by each fish's own SNR: recall, identity and switches.

A fish frame is in a pair when another *fundamental* is within 3 Hz; the
pair's stratum is the frequency difference (< 0.6, 0.6-1.2, 1.2-3 Hz) and
the *own* SNR of the weaker of the two fish (``own_snr.py``; the truth's
SNR of a fish in a close pair holds the power of both, so a faint fish next
to a strong one would count as visible). The partner is the nearest other
fundamental. Per stratum, pooled over fish frames:

* ``recall``: the fish is detected;
* ``id``: on its own (IDF1) identity;
* ``both``: both fish of the pair on their own identities (per fish frame,
  so each pair frame counts twice);
* ``switches``: identity changes along the fish's detections where the later
  one is in the stratum, per minute of stratum frames.

``score`` writes ``pair_score.json`` (numerators and denominators per
stratum) into every run mixture, ``report`` pools them, ``compare`` gives
paired differences of two runs with a bootstrap 95 % CI over mixtures (lab
and field variants of a mixture resampled together, as in ``compare.py``).

    python benchmarks/dense/pair_strata.py score --tags B-rise-j3 --splits ldev ldev-field
    python benchmarks/dense/pair_strata.py report --tags B-rise-j3:ldev,ref-learned-rises:ldev2
    python benchmarks/dense/pair_strata.py compare --a B-rise-j3:ldev,ref-learned-rises:ldev2 --b mine
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA
from compare import parse_spec, tag_for, unit
from own_snr import own_snr_at
from score import match, truth_at

from wavetracker.results import Results

RUNS = DATA / "runs"
MIXTURES = DATA / "mixtures"
FIG = DATA / "figures"
DF_BANDS = [
    (0.0, 0.6, "< 0.6 Hz"),
    (0.6, 1.2, "0.6-1.2 Hz"),
    (1.2, 3.0, "1.2-3 Hz"),
    (3.0, np.inf, ">= 3 Hz"),  # away from pairs: for regressions
]
SNR_BANDS = [
    (-99.0, 6.0, "weaker < 6 dB"),
    (6.0, 15.0, "weaker 6-15 dB"),
    (15.0, 99.0, "weaker >= 15 dB"),
    (6.0, 99.0, "weaker >= 6 dB"),
]
METRICS = ("recall", "id", "both", "switches")
VERSION = 3  # of pair_score.json: older files are scored again
SPLITS = ["ldev", "ldev-field", "ldev2", "ldev2-field"]


def stratum_keys():
    return [(d[2], s[2]) for d in DF_BANDS for s in SNR_BANDS]


def episode_keys(visible_only=False):
    out = []
    for name in EPISODE_CLASSES:
        if not visible_only:
            out.append(f"episodes {name}")
        out.append(f"episodes {name} | weaker >= 6 dB")
    return out


def pair_frames(mix_dir: Path, res: Results) -> dict:
    """Per (fish, frame) of the result: the partner, its distance and the
    weaker own SNR of the two (NaN where unknown)."""
    truth = dict(np.load(mix_dir / "truth.npz"))
    F = truth_at(truth, res.times)
    Ff = truth_at(truth, res.times, "freqs_filled")
    own = own_snr_at(mix_dir, res.times)
    n = len(F)
    df = np.full(F.shape, np.inf)
    partner = np.full(F.shape, -1)
    for k in range(n):
        for j in range(n):
            if j == k:
                continue
            d = np.abs(F[k] - Ff[j])
            better = d < df[k]
            df[k, better] = d[better]
            partner[k, better] = j
    cols = np.arange(F.shape[1])
    p = np.where(partner >= 0, partner, 0)
    weaker = np.fmin(own, own[p, cols[None, :]])
    weaker[(partner < 0) | ~np.isfinite(own)] = np.nan
    return {
        "F": F,
        "Ff": Ff,
        "df": df,
        "partner": partner,
        "weaker": weaker,
        "own": own,
    }


EPISODE_CLASSES = {
    "all": lambda e: True,
    "min df < 0.3 Hz": lambda e: e["min_df"] < 0.3,
    "min df >= 0.3 Hz": lambda e: e["min_df"] >= 0.3,
    "shorter than 10 s": lambda e: e["dur"] < 10,
    "10 s or longer": lambda e: e["dur"] >= 10,
    "crossed": lambda e: e["crossed"],
    "not crossed": lambda e: not e["crossed"],
}
OUTCOMES = ("kept", "swapped", "changed")


def _mode(x):
    u, c = np.unique(x, return_counts=True)
    return u[c.argmax()]


def episode_outcomes(pf, fish, res, pair_hz=1.2, merge_s=2.0, min_s=1.0, win_s=10.0):
    """Every truth episode of two fish closer than pair_hz (frames less than
    merge_s apart joined, at least min_s long): for each fish, the identity
    (mode of its detections) over win_s before and after it is the same
    (kept), the partner's from before (swapped) or another (changed); fish
    without 5 identified detections on both sides are left out."""
    Ff, own = pf["Ff"], pf["own"]
    dt = float(res.times[1] - res.times[0])
    ident = res.ident_v
    w, g = round(win_s / dt), round(0.5 / dt)
    n, n_frames = Ff.shape
    rows = []
    for k in range(n):
        mk = (fish == k) & np.isfinite(ident)
        for j in range(k + 1, n):
            fr = np.flatnonzero(np.abs(Ff[k] - Ff[j]) < pair_hz)
            if not len(fr):
                continue
            brk = np.flatnonzero(np.diff(fr) > merge_s / dt)
            mj = (fish == j) & np.isfinite(ident)
            for a, b in zip(
                np.r_[fr[0], fr[brk + 1]], np.r_[fr[brk], fr[-1]], strict=True
            ):
                if (b - a) * dt < min_s:
                    continue
                sides = {}
                for f, m in ((k, mk), (j, mj)):
                    pre = ident[m & (res.idx_v >= a - w) & (res.idx_v < a - g)]
                    post = ident[m & (res.idx_v > b + g) & (res.idx_v <= b + w)]
                    sides[f] = (
                        _mode(pre) if len(pre) >= 5 else None,
                        _mode(post) if len(post) >= 5 else None,
                    )
                a0, b1 = max(a - 1, 0), min(b + 1, n_frames - 1)
                crossed = np.sign(Ff[k, a0] - Ff[j, a0]) != np.sign(
                    Ff[k, b1] - Ff[j, b1]
                )
                with np.errstate(all="ignore"):
                    weaker = np.nanmedian(np.fmin(own[k, a : b + 1], own[j, a : b + 1]))
                min_df = float(np.min(np.abs(Ff[k, a : b + 1] - Ff[j, a : b + 1])))
                for f, o in ((k, j), (j, k)):
                    pre, post = sides[f]
                    if pre is None or post is None:
                        continue
                    if pre == post:
                        outcome = "kept"
                    elif post == sides[o][0]:
                        outcome = "swapped"
                    else:
                        outcome = "changed"
                    rows.append(
                        {
                            "outcome": outcome,
                            "dur": (b - a) * dt,
                            "crossed": bool(crossed),
                            "min_df": min_df,
                            "visible": bool(weaker >= 6),
                        }
                    )
    out = {}
    for name, sel in EPISODE_CLASSES.items():
        for vis in (False, True):
            r = [e for e in rows if sel(e) and (e["visible"] or not vis)]
            key = f"episodes {name}" + (" | weaker >= 6 dB" if vis else "")
            out[key] = {o: sum(e["outcome"] == o for e in r) for o in OUTCOMES}
            out[key]["frames"] = len(r)  # the denominator: fish sides
    return out


def score_mixture(mix_dir: Path, run_dir: Path) -> dict:
    res = Results.load(run_dir)
    pf = pair_frames(mix_dir, res)
    F = pf["F"]
    n_fish, n_frames = F.shape
    fish = match(res, F)
    ident = res.ident_v
    tab = dict(np.load(run_dir / "frames.npz"))
    matched = np.zeros((n_fish, n_frames), bool)
    assigned = np.zeros((n_fish, n_frames), bool)
    matched[tab["fish"], tab["frame"]] = tab["matched"]
    assigned[tab["fish"], tab["frame"]] = tab["assigned"]
    # identity switches, placed at the frame of the later detection
    switch = np.zeros((n_fish, n_frames), bool)
    for k in range(n_fish):
        m = (fish == k) & np.isfinite(ident)
        fr = res.idx_v[m]
        o = np.argsort(fr, kind="stable")
        fr, seq = fr[o], ident[m][o]
        switch[k, fr[1:][seq[1:] != seq[:-1]]] = True
    known = np.isfinite(F)
    cols = np.arange(n_frames)
    p = np.where(pf["partner"] >= 0, pf["partner"], 0)
    both = assigned & assigned[p, cols[None, :]]
    dt = float(res.times[1] - res.times[0])
    out = {"frame_dt": dt}
    for dlo, dhi, dname in DF_BANDS:
        for slo, shi, sname in SNR_BANDS:
            m = (
                known
                & (pf["df"] >= dlo)
                & (pf["df"] < dhi)
                & (pf["weaker"] >= slo)
                & (pf["weaker"] < shi)
            )
            key = f"{dname} | {sname}"
            out[key] = {
                "frames": int(m.sum()),
                "recall": int(matched[m].sum()),
                "id": int(assigned[m].sum()),
                "both": int(both[m].sum()),
                "switches": int(switch[m].sum()),
            }
    # the old labels: the fish's own truth SNR, which holds both fish
    snr = np.stack(
        [
            np.interp(res.times, np.load(mix_dir / "truth.npz")["snr_times"], s)
            for s in np.load(mix_dir / "truth.npz")["snr_db"]
        ]
    )
    m_old = known & (pf["df"] < 1.2) & (snr >= 6)
    m_new = known & (pf["df"] < 1.2) & (pf["weaker"] >= 6)
    out["labels < 1.2 Hz"] = {
        "old_visible": int(m_old.sum()),
        "new_visible": int(m_new.sum()),
        "both": int((m_old & m_new).sum()),
        "old_recall": int(matched[m_old].sum()),
        "new_recall": int(matched[m_new].sum()),
        "old_id": int(assigned[m_old].sum()),
        "new_id": int(assigned[m_new].sum()),
    }
    out |= episode_outcomes(pf, fish, res)
    return out


def run_dirs(spec: dict, splits) -> list[tuple[str, Path, Path]]:
    out = []
    for sp in splits:
        tag = tag_for(spec, sp)
        for f in sorted((RUNS / tag / sp).glob("*/frames.npz")):
            name = f"{sp}/{f.parent.name}"
            out.append((name, MIXTURES / name, f.parent))
    return out


def load_scores(spec: str, splits, overwrite: bool = False) -> dict[str, dict]:
    """{split/mixture: pair_score} of a run spec, computed where missing."""
    out = {}
    for name, mdir, rdir in run_dirs(parse_spec(spec), splits):
        path = rdir / "pair_score.json"
        stale = (
            path.exists()
            and path.stat().st_mtime < (rdir / "ident_v.npy").stat().st_mtime
        )
        old = json.loads(path.read_text()) if path.exists() and not stale else {}
        if overwrite or old.get("version") != VERSION:
            old = score_mixture(mdir, rdir) | {"version": VERSION}
            path.write_text(json.dumps(old, indent=1))
        out[name] = old
    return out


def pooled(scores: dict[str, dict], names, key, metric) -> float:
    num = sum(scores[n][key][metric] for n in names)
    den = sum(scores[n][key]["frames"] for n in names)
    if metric == "switches":
        dt = scores[names[0]]["frame_dt"] if names else 1.0
        return num / max(den * dt / 60.0, 1e-9)
    return num / den if den else np.nan


def score(args):
    for t in args.tags:
        s = load_scores(t, args.splits, overwrite=args.overwrite)
        print(f"{t}: {len(s)} mixtures", flush=True)


def report(args):
    for t in args.tags:
        s = load_scores(t, args.splits)
        names = sorted(s)
        print(f"\n{t} ({len(names)} variants); switches per fish-minute")
        print(
            f"{'stratum':40s} {'frames':>8s} " + " ".join(f"{m:>8s}" for m in METRICS)
        )
        for d, sn in stratum_keys():
            key = f"{d} | {sn}"
            fr = sum(s[n][key]["frames"] for n in names)
            vals = " ".join(f"{pooled(s, names, key, m):8.3f}" for m in METRICS)
            print(f"{key:40s} {fr:8d} {vals}")
        print(
            f"{'pair episodes < 1.2 Hz (fish sides)':40s} {'n':>8s} "
            + " ".join(f"{o:>8s}" for o in OUTCOMES)
        )
        for key in episode_keys():
            tot = sum(s[n][key]["frames"] for n in names)
            vals = " ".join(
                f"{sum(s[n][key][o] for n in names) / max(tot, 1):8.3f}"
                for o in OUTCOMES
            )
            print(f"{key:40s} {tot:8d} {vals}")
        lab = {
            k: sum(s[n]["labels < 1.2 Hz"][k] for n in names)
            for k in s[names[0]]["labels < 1.2 Hz"]
        }
        print(
            f"< 1.2 Hz, visible by the truth's SNR: {lab['old_visible']} frames "
            f"(recall {lab['old_recall'] / max(lab['old_visible'], 1):.3f}, id "
            f"{lab['old_id'] / max(lab['old_visible'], 1):.3f}); by the weaker own SNR: "
            f"{lab['new_visible']} (recall {lab['new_recall'] / max(lab['new_visible'], 1):.3f}, "
            f"id {lab['new_id'] / max(lab['new_visible'], 1):.3f}); in both {lab['both']}"
        )


def bootstrap_diff(sa, sb, names, key, metric, n_boot, rng):
    units = np.array([unit(n) for n in names])
    u, inv = np.unique(units, return_inverse=True)
    dt = sa[names[0]]["frame_dt"]
    scale = (60.0 / dt) if metric == "switches" else 1.0
    na = np.bincount(inv, [sa[n][key][metric] for n in names], len(u))
    nb = np.bincount(inv, [sb[n][key][metric] for n in names], len(u))
    da = np.bincount(inv, [sa[n][key]["frames"] for n in names], len(u))
    db = np.bincount(inv, [sb[n][key]["frames"] for n in names], len(u))
    idx = rng.integers(len(u), size=(n_boot, len(u)))
    with np.errstate(invalid="ignore", divide="ignore"):
        d = scale * (nb[idx].sum(1) / db[idx].sum(1) - na[idx].sum(1) / da[idx].sum(1))
        point = scale * (nb.sum() / db.sum() - na.sum() / da.sum())
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    return scale * na.sum() / da.sum(), scale * nb.sum() / db.sum(), point, lo, hi


def compare(args):
    sa = load_scores(args.a, args.splits)
    sb = load_scores(args.b, args.splits)
    names = sorted(set(sa) & set(sb))
    rng = np.random.default_rng(0)
    print(
        f"a = {args.a}\nb = {args.b}\n{len(names)} variants; b - a, 95 % CI over mixtures"
    )
    print(
        f"{'stratum':40s} {'metric':>8s} {'a':>8s} {'b':>8s} {'b - a':>8s} {'95 % CI':>20s}"
    )
    keys = (
        stratum_keys() if args.all else [k for k in stratum_keys() if k[1] in args.snr]
    )
    rows = {}
    for d, sn in keys:
        key = f"{d} | {sn}"
        for m in args.metrics:
            a, b, diff, lo, hi = bootstrap_diff(sa, sb, names, key, m, args.n_boot, rng)
            rows[(key, m)] = (a, b, diff, lo, hi)
            print(
                f"{key:40s} {m:>8s} {a:8.3f} {b:8.3f} {diff:+8.3f} [{lo:+8.3f}, {hi:+8.3f}]"
            )
    for key in episode_keys(visible_only=not args.all):
        for m in ("kept", "swapped"):
            a, b, diff, lo, hi = bootstrap_diff(sa, sb, names, key, m, args.n_boot, rng)
            rows[(key, m)] = (a, b, diff, lo, hi)
            print(
                f"{key:40s} {m:>8s} {a:8.3f} {b:8.3f} {diff:+8.3f} "
                f"[{lo:+8.3f}, {hi:+8.3f}]"
            )
    if args.json:
        Path(args.json).write_text(
            json.dumps({f"{k} / {m}": v for (k, m), v in rows.items()}, indent=1)
        )
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--tags", nargs="+", required=True)
    s.add_argument("--splits", nargs="+", default=SPLITS)
    s.add_argument("--overwrite", action="store_true")
    r = sub.add_parser("report")
    r.add_argument("--tags", nargs="+", required=True)
    r.add_argument("--splits", nargs="+", default=SPLITS)
    c = sub.add_parser("compare")
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    c.add_argument("--splits", nargs="+", default=SPLITS)
    c.add_argument("--metrics", nargs="+", default=["recall", "id", "both", "switches"])
    c.add_argument("--snr", nargs="+", default=["weaker >= 6 dB"])
    c.add_argument("--all", action="store_true", help="every SNR band")
    c.add_argument("--n-boot", type=int, default=5000)
    c.add_argument("--json")
    args = p.parse_args(argv)
    {"score": score, "report": report, "compare": compare}[args.cmd](args)


if __name__ == "__main__":
    main()
