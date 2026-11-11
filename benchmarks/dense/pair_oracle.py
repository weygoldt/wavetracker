"""Ceiling of a pair head: the detections of a run plus the hidden fish of
every close pair from the truth, through the same tracker and linker.

Where a fish (at least ``--min-snr`` dB) has no detection but another
fundamental within 1.2 Hz has one (the pair made one peak), it gets a
detection at its truth frequency, with the electrode spectra and salience of
its partner's detection (the same peak). A perfect pair head would add exactly
these; ``--keep`` keeps that share of them (at random), a pair head that
finds only some. Tracking runs on CPU, no detector is needed.

    python benchmarks/dense/pair_oracle.py --source B-aug-s0-t0.6 --tag B-pair-oracle-s0 \\
        -c benchmarks/dense/configs/learned-b.yaml --split ldev
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from run import MIXTURES, RUNS, merge, mixtures
from score import match, score, truth_at

from wavetracker.config import Config
from wavetracker.pipeline import track_results
from wavetracker.results import Results

PAIR_HZ = 1.2


def add_hidden(
    res: Results, truth: dict, min_snr: float, keep: float, rng, exact: bool = False
) -> int:
    """Add the hidden fish of close pairs to `res` (in place); returns how many.
    `exact`: the partner's detection moves to its truth frequency as well."""
    F = truth_at(truth, res.times)
    snr = np.stack(
        [np.interp(res.times, truth["snr_times"], s) for s in truth["snr_db"]]
    )
    fish = match(res, F)
    n_fish, n_frames = F.shape
    det_of = np.full((n_fish, n_frames), -1)
    m = fish >= 0
    det_of[fish[m], res.idx_v[m]] = np.flatnonzero(m)
    add_idx, add_f, add_src = [], [], []
    for k in range(n_fish):
        miss = np.isfinite(F[k]) & (det_of[k] < 0) & (snr[k] >= min_snr)
        for j in range(n_fish):
            if j == k:
                continue
            close = miss & (np.abs(F[k] - F[j]) < PAIR_HZ) & (det_of[j] >= 0)
            fr = np.flatnonzero(close)
            if keep < 1.0:
                fr = fr[rng.random(len(fr)) < keep]
            add_idx.append(fr)
            add_f.append(F[k, fr])
            add_src.append(det_of[j, fr])
            if exact:
                res.fund_v[det_of[j, fr]] = F[j, fr]
            miss &= ~close
    idx = np.concatenate(add_idx).astype(res.idx_v.dtype)
    src = np.concatenate(add_src)
    if not len(idx):
        return 0
    res.fund_v = np.concatenate([res.fund_v, np.concatenate(add_f)])
    res.idx_v = np.concatenate([res.idx_v, idx])
    res.sign_v = np.concatenate([res.sign_v, res.sign_v[src]])
    if res.cplx_v is not None:
        res.cplx_v = np.concatenate([res.cplx_v, res.cplx_v[src]])
    if res.sal_v is not None:
        res.sal_v = np.concatenate([res.sal_v, res.sal_v[src]])
    res.ident_v = np.full(len(res.fund_v), np.nan)
    order = np.lexsort((res.fund_v, res.idx_v))
    for name in ("fund_v", "idx_v", "sign_v", "cplx_v", "sal_v", "ident_v"):
        v = getattr(res, name)
        if v is not None:
            setattr(res, name, v[order])
    return len(idx)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", required=True, help="run tag with the detections")
    p.add_argument("--tag", required=True)
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--split", default="ldev")
    p.add_argument("--min-snr", type=float, default=0.0)
    p.add_argument("--keep", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--exact",
        action="store_true",
        help="the partner's detection at its truth frequency too",
    )
    args = p.parse_args(argv)
    variant = yaml.safe_load(Path(args.config).read_text())
    rng = np.random.default_rng(args.seed)
    for name in mixtures(args.split):
        src = RUNS / args.source / name
        out = RUNS / args.tag / name
        if (out / "score.json").exists() or not (src / "fund_v.npy").exists():
            continue
        band = yaml.safe_load((MIXTURES / name / "band.yaml").read_text())
        cfg = Config.from_dict(merge(variant, band))
        res = Results.load(src)
        truth = dict(np.load(MIXTURES / name / "truth.npz"))
        n_add = add_hidden(res, truth, args.min_snr, args.keep, rng, args.exact)
        t_track = track_results(res, cfg)
        res.meta["pair_oracle"] = {"source": args.source, "added": n_add}
        res.save(out)
        summary, table = score(MIXTURES / name, out)
        summary |= {"tag": args.tag, "source": args.source, "added": n_add}
        (out / "score.json").write_text(json.dumps(summary, indent=1))
        np.savez_compressed(out / "frames.npz", **table)
        print(
            f"{args.tag} {name}: +{n_add} detections, idf1 {summary['idf1']:.3f} "
            f"recall {summary['recall']:.3f} ({t_track:.0f} s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
