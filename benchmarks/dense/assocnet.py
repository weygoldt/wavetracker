"""Train the learned association (wavetracker.assocnet, ``tracking.method:
learned``) on the learned detector's detections of ``ltrain``.

The detections of a run (``--source``, per split) are matched to the truth
(``score.match``); an edge between two detections is "same fish" when both
are matched to the same fish. Edges touching a detection where the truth is
unknown (``classify_unmatched`` class 1) and edges between two unmatched
detections carry no weight. With ``--pairs`` the builder also writes a copy of
every mixture with the hidden fish of close pairs added from the truth (as
``pair_oracle.py`` does), so that the network learns to keep two detections of
one peak apart.

    python benchmarks/dense/assocnet.py build --splits ltrain ltrain-field --pairs
    python benchmarks/dense/assocnet.py build --splits ldev ldev-field          # validation
    python benchmarks/dense/assocnet.py train --tag C-an-v0
    python benchmarks/dense/run.py retrack --tag C-an-v0 --source B-aug-s0-t0.6 \\
        -c benchmarks/dense/configs/learned-assocnet.yaml --split ldev

Data go to ``$WT_BENCHMARK_DATA/assocnet/data/<split>/<mixture>[-pairs].npz``,
models to ``$WT_BENCHMARK_DATA/assocnet/<tag>/``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA
from pair_oracle import add_hidden
from run import MIXTURES, RUNS, mixtures
from score import classify_unmatched, match, truth_at

from wavetracker import assocnet
from wavetracker.results import Results

WORK = DATA / "assocnet"
SOURCES = {  # learned-b detections of every split
    "ltrain": "B-rise-lt",
    "ltrain-field": "B-rise-lt",
    "ldev": "B-aug-s0-t0.6",
    "ldev-field": "B-aug-s0-t0.6",
    "ldev2": "ref-learned-b",
    "ldev2-field": "ref-learned-b",
}
MAX_DT = 2.0  # s, candidate edges
MAX_DF = 2.5  # Hz


def build_one(res: Results, truth: dict) -> dict:
    F = truth_at(truth, res.times)
    Ff = truth_at(truth, res.times, "freqs_filled")
    fish = match(res, F)
    cls = classify_unmatched(res, F, Ff, fish)
    dt = res.times[1] - res.times[0]
    src, dst = assocnet.candidate_edges(
        res.idx_v, res.fund_v, max(1, round(MAX_DT / dt)), MAX_DF
    )
    return {
        "fund": res.fund_v,
        "idx": res.idx_v.astype(np.int64),
        "times": res.times,
        "sign": res.sign_v.astype(np.float32),
        "cplx": res.cplx_v.astype(np.complex64),
        "sal": res.sal_v.astype(np.float32),
        "fish": fish.astype(np.int16),
        "cls": cls.astype(np.int8),
        "src": src.astype(np.int32),
        "dst": dst.astype(np.int32),
    }


def build(args):
    rng = np.random.default_rng(0)
    for split in args.splits:
        src_tag = args.source or SOURCES[split]
        for name in mixtures(split, args.only):
            mix = name.split("/", 1)[1]
            out = WORK / "data" / split / f"{mix}.npz"
            out_p = WORK / "data" / split / f"{mix}-pairs.npz"
            todo = [out] + ([out_p] if args.pairs else [])
            if all(o.exists() for o in todo) and not args.overwrite:
                continue
            t0 = time.perf_counter()
            truth = dict(np.load(MIXTURES / name / "truth.npz"))
            out.parent.mkdir(parents=True, exist_ok=True)
            res = Results.load(RUNS / src_tag / name)
            d = build_one(res, truth)
            np.savez(out, **d)
            msg = f"{len(d['fund'])} detections, {len(d['src'])} edges"
            if args.pairs:
                res = Results.load(RUNS / src_tag / name)
                n_add = add_hidden(res, truth, 0.0, 1.0, rng)
                dp = build_one(res, truth)
                np.savez(out_p, **dp)
                msg += f"; pairs +{n_add}"
            print(f"{name}: {msg} ({time.perf_counter() - t0:.0f} s)", flush=True)


# --- training ------------------------------------------------------------------


class Mixture:
    """One mixture on the device: inputs, labels, edges."""

    def __init__(self, path: Path, device):
        import torch

        z = np.load(path)
        self.name = path.stem
        self.idx = z["idx"]
        self.d = assocnet.detection_inputs(
            z["fund"], z["idx"], z["times"], z["sign"], z["cplx"], z["sal"], device
        )
        fish = z["fish"].astype(np.int64)
        cls = z["cls"]
        src, dst = z["src"].astype(np.int64), z["dst"].astype(np.int64)
        fa, fb = fish[src], fish[dst]
        y = (fa == fb) & (fa >= 0)
        w = ((fa >= 0) | (fb >= 0)) & (cls[src] != 1) & (cls[dst] != 1)
        self.src = torch.as_tensor(src, device=device)
        self.dst = torch.as_tensor(dst, device=device)
        self.y = torch.as_tensor(y, dtype=torch.float32, device=device)
        self.w = torch.as_tensor(w, dtype=torch.float32, device=device)
        self.frame_src = torch.as_tensor(self.idx[src], device=device)
        self.n_frames = int(self.idx[-1]) + 1
        self.fs = np.searchsorted(self.idx, np.arange(self.n_frames + 1))

    def crop(self, f0: int, f1: int):
        """Sub-graph of the detections in frames [f0, f1)."""
        import torch

        lo, hi = int(self.fs[f0]), int(self.fs[min(f1, self.n_frames)])
        e0 = int(torch.searchsorted(self.src, torch.tensor(lo, device=self.src.device)))
        e1 = int(torch.searchsorted(self.src, torch.tensor(hi, device=self.src.device)))
        s, t = self.src[e0:e1], self.dst[e0:e1]
        ok = t < hi
        s, t = s[ok] - lo, t[ok] - lo
        nodes = torch.arange(lo, hi, device=self.src.device)
        return (
            self.d.subset(nodes),
            s,
            t,
            self.y[e0:e1][ok],
            self.w[e0:e1][ok],
        )


def concat(parts):
    """Disjoint union of sub-graphs."""
    import torch

    ds, ss, ts, ys, ws = [], [], [], [], []
    off = 0
    for d, s, t, y, w in parts:
        ds.append(d)
        ss.append(s + off)
        ts.append(t + off)
        ys.append(y)
        ws.append(w)
        off += d.node.shape[0]
    d = assocnet.Detections(
        **{
            k: torch.cat([getattr(x, k) for x in ds])
            for k in assocnet.Detections.__dataclass_fields__
        }
    )
    return d, torch.cat(ss), torch.cat(ts), torch.cat(ys), torch.cat(ws)


def load_split(split: str, device, pairs: bool, only=None) -> list[Mixture]:
    files = sorted((WORK / "data" / split).glob("*.npz"))
    files = [f for f in files if f.stem.endswith("-pairs") == pairs]
    if only:
        files = [f for f in files if any(o in f.stem for o in only)]
    return [Mixture(f, device) for f in files]


def evaluate(net, mixes, crop_frames, device):
    """Weighted BCE, AUC-free summaries and link accuracy of the val crops."""
    import torch

    net.eval()
    tot, wsum, err, n = 0.0, 0.0, 0.0, 0.0
    with torch.no_grad():
        for m in mixes:
            for f0 in range(0, m.n_frames, crop_frames):
                d, s, t, y, w = m.crop(f0, f0 + crop_frames)
                if not len(s):
                    continue
                logit = net(d, s, t)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logit, y, weight=w, reduction="sum"
                )
                tot += loss.item()
                wsum += w.sum().item()
                err += (((logit > 0).float() != y).float() * w).sum().item()
                n += w.sum().item()
    net.train()
    return tot / max(wsum, 1), err / max(n, 1)


def train(args):
    import torch
    from torch.utils.tensorboard import SummaryWriter

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    out = WORK / args.tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    mixes = [m for sp in args.train_splits for m in load_split(sp, device, False)]
    pair_mixes = (
        [m for sp in args.train_splits for m in load_split(sp, device, True)]
        if args.pair_frac > 0
        else []
    )
    val = [
        m
        for sp in args.val_splits
        for m in load_split(sp, device, False, args.val_only)
    ]
    n_edges = sum(len(m.src) for m in mixes)
    n_pos = sum(m.y.sum().item() for m in mixes)
    n_w = sum(m.w.sum().item() for m in mixes)
    print(
        f"train: {len(mixes)} mixtures (+{len(pair_mixes)} with pairs), "
        f"{sum(len(m.idx) for m in mixes)} detections, {n_edges} edges "
        f"({n_pos / n_edges:.3f} same fish, {n_w / n_edges:.3f} weighted); "
        f"val {len(val)}",
        flush=True,
    )
    net = assocnet.AssocNet(args.hidden, args.rounds, args.el_hidden).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05
    )
    log = SummaryWriter(out / "tb")
    best = np.inf
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        parts = []
        for _ in range(args.batch):
            pool = pair_mixes if pair_mixes and rng.random() < args.pair_frac else mixes
            m = pool[rng.integers(len(pool))]
            f0 = int(rng.integers(0, max(1, m.n_frames - args.crop_frames)))
            parts.append(m.crop(f0, f0 + args.crop_frames))
        d, s, t, y, w = concat(parts)
        logit = net(d, s, t)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logit, y, weight=w, reduction="sum"
        ) / w.sum().clamp_min(1.0)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0:
            log.add_scalar("train/loss", loss.item(), step)
        if step % args.val_every == 0 or step == args.steps:
            vl, ve = evaluate(net, val, args.crop_frames, device)
            log.add_scalar("val/loss", vl, step)
            log.add_scalar("val/error", ve, step)
            print(
                f"step {step}: train {loss.item():.4f} val {vl:.4f} error {ve:.4f} "
                f"({time.perf_counter() - t0:.0f} s)",
                flush=True,
            )
            model = assocnet.AssocModel(net, MAX_DT, MAX_DF)
            model.save(out / "last.pt")
            if vl < best:
                best = vl
                model.save(out / "best.pt")
    log.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--splits", nargs="+", default=["ltrain", "ltrain-field"])
    b.add_argument("--source", help="run tag (default: SOURCES per split)")
    b.add_argument("--only", nargs="*")
    b.add_argument("--pairs", action="store_true", help="also with close pairs added")
    b.add_argument("--overwrite", action="store_true")
    t = sub.add_parser("train")
    t.add_argument("--tag", required=True)
    t.add_argument("--train-splits", nargs="+", default=["ltrain", "ltrain-field"])
    t.add_argument("--val-splits", nargs="+", default=["ldev", "ldev-field"])
    t.add_argument("--val-only", nargs="*", default=None)
    t.add_argument("--steps", type=int, default=6000)
    t.add_argument("--batch", type=int, default=4, help="crops per step")
    t.add_argument("--crop-frames", type=int, default=400)
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--hidden", type=int, default=64)
    t.add_argument("--rounds", type=int, default=3)
    t.add_argument("--el-hidden", type=int, default=32)
    t.add_argument("--pair-frac", type=float, default=0.0)
    t.add_argument("--val-every", type=int, default=500)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="cuda")
    args = p.parse_args(argv)
    {"build": build, "train": train}[args.cmd](args)


if __name__ == "__main__":
    main()
