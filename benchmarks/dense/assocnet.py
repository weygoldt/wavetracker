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


MERGED_HZ = 0.5  # a fish without a detection this close to another's shares it


def hidden_partner(res: Results, F: np.ndarray, fish: np.ndarray) -> np.ndarray:
    """For every matched detection, the nearest other fish within MERGED_HZ
    that has no detection of its own in that frame (the two fish make one
    peak), else -1."""
    n_fish, n_frames = F.shape
    has = np.zeros((n_fish, n_frames), bool)
    m = fish >= 0
    has[fish[m], res.idx_v[m]] = True
    d = np.abs(F[:, res.idx_v] - res.fund_v[None])  # (fish, det)
    d[has[:, res.idx_v]] = np.inf  # fish with their own detection
    d[~np.isfinite(d)] = np.inf
    alt = d.argmin(0)
    ok = m & (d[alt, np.arange(len(alt))] <= MERGED_HZ)
    return np.where(ok, alt, -1)


def build_one(res: Results, truth: dict) -> dict:
    F = truth_at(truth, res.times)
    Ff = truth_at(truth, res.times, "freqs_filled")
    fish = match(res, F)
    alt = hidden_partner(res, F, fish)
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
        "fish_alt": alt.astype(np.int16),
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
    """One mixture in memory (CPU): inputs, labels, edges; crops go to the
    training device."""

    def __init__(
        self, path: Path, device, hard_weight: float = 1.0, merged: bool = False
    ):
        import torch

        z = np.load(path)
        self.name = path.stem
        self.idx = z["idx"]
        self.d = assocnet.detection_inputs(
            z["fund"], z["idx"], z["times"], z["sign"], z["cplx"], z["sal"], "cpu"
        )
        fish = z["fish"].astype(np.int64)
        cls = z["cls"]
        src, dst = z["src"].astype(np.int64), z["dst"].astype(np.int64)
        fa, fb = fish[src], fish[dst]
        y = (fa == fb) & (fa >= 0)
        if merged:  # a merged peak belongs to both fish
            alt = z["fish_alt"].astype(np.int64)
            aa, ab = alt[src], alt[dst]
            y |= ((aa == fb) & (aa >= 0)) | ((fa == ab) & (fa >= 0))
            y |= (aa == ab) & (aa >= 0)
        w = ((fa >= 0) | (fb >= 0)) & (cls[src] != 1) & (cls[dst] != 1)
        w = w.astype(np.float32)
        if hard_weight != 1.0:  # edges of detections with true and false candidates
            n = len(fish)
            pos, neg = (w > 0) & y, (w > 0) & ~y
            hard_in = (np.bincount(dst[pos], minlength=n) > 0) & (
                np.bincount(dst[neg], minlength=n) > 0
            )
            hard_out = (np.bincount(src[pos], minlength=n) > 0) & (
                np.bincount(src[neg], minlength=n) > 0
            )
            w[hard_in[dst] | hard_out[src]] *= hard_weight
        self.src = torch.as_tensor(src)
        self.dst = torch.as_tensor(dst)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.w = torch.as_tensor(w, dtype=torch.float32)
        self.device = device
        self.n_frames = int(self.idx[-1]) + 1
        self.fs = np.searchsorted(self.idx, np.arange(self.n_frames + 1))

    def crop(self, f0: int, f1: int):
        """Sub-graph of the detections in frames [f0, f1)."""
        import torch

        lo, hi = int(self.fs[f0]), int(self.fs[min(f1, self.n_frames)])
        e0 = int(torch.searchsorted(self.src, torch.tensor(lo)))
        e1 = int(torch.searchsorted(self.src, torch.tensor(hi)))
        s, t = self.src[e0:e1], self.dst[e0:e1]
        ok = t < hi
        s, t = s[ok] - lo, t[ok] - lo
        dev = self.device
        sub = assocnet.Detections(
            **{
                k: getattr(self.d, k)[lo:hi].to(dev)
                for k in assocnet.Detections.__dataclass_fields__
            }
        )
        return (
            sub,
            s.to(dev),
            t.to(dev),
            self.y[e0:e1][ok].to(dev),
            self.w[e0:e1][ok].to(dev),
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


def load_split(
    split: str,
    device,
    pairs: bool,
    only=None,
    hard_weight: float = 1.0,
    merged: bool = False,
) -> list[Mixture]:
    files = sorted((WORK / "data" / split).glob("*.npz"))
    files = [f for f in files if f.stem.endswith("-pairs") == pairs]
    if only:
        files = [f for f in files if any(o in f.stem for o in only)]
    return [Mixture(f, device, hard_weight, merged) for f in files]


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
    torch.set_num_threads(args.threads)  # the CPU only cuts crops
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    out = WORK / args.tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    hw, mg = args.hard_weight, args.merged
    mixes = [
        m
        for sp in args.train_splits
        for m in load_split(sp, device, False, None, hw, mg)
    ]
    pair_mixes = (
        [
            m
            for sp in args.train_splits
            for m in load_split(sp, device, True, None, hw, mg)
        ]
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
        try:
            logit = net(d, s, t)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logit, y, weight=w, reduction="sum"
            ) / w.sum().clamp_min(1.0)
            opt.zero_grad()
            loss.backward()
        except torch.OutOfMemoryError:  # the GPU is shared: skip this batch
            print(f"step {step}: out of memory, skipped", flush=True)
            opt.zero_grad()
            del d, s, t, y, w
            torch.cuda.empty_cache()
            time.sleep(5)
            continue
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


# --- raw fragments ---------------------------------------------------------------


SEP_BANDS = [(0, 0.25), (0.25, 1.2), (1.2, 3.0), (3.0, np.inf)]


def frag_stats(ident, fish, cls, n_known, idx, F=None) -> dict:
    """Quality of a tracker's raw fragments (before linking); with the truth
    F (fish, frames), switches by the two fish's distance at the switch."""
    from assoc_diag import idf1, majority

    has = np.isfinite(ident)
    m = has & (fish >= 0)
    order = np.lexsort((idx[m], ident[m]))
    i_s, f_s, x_s = ident[m][order], fish[m][order], idx[m][order]
    same = i_s[1:] == i_s[:-1]
    sw = same & (f_s[1:] != f_s[:-1])
    switches = int(sw.sum())
    by_sep = {}
    if F is not None:
        j = np.flatnonzero(sw) + 1
        sep = np.abs(F[f_s[j], x_s[j]] - F[f_s[j - 1], x_s[j]])
        sep = np.where(np.isfinite(sep), sep, np.inf)
        for lo, hi in SEP_BANDS:
            by_sep[f"switches {lo}-{hi} Hz"] = int(((sep >= lo) & (sep < hi)).sum())
    _, cnt = np.unique(ident[has], return_counts=True)
    long = cnt >= 10
    return by_sep | {
        "switches": switches,
        "fragments_10": int(long.sum()),
        "dets_in_short": float(cnt[~long].sum() / max(has.sum(), 1)),
        "assigned": float(has.mean()),
        "idf1_raw": idf1(ident, fish, cls, n_known),
        "idf1_majority": idf1(majority(ident, fish), fish, cls, n_known),
    }


def frags(args):
    """Raw fragments of a tracker config on the detections of every split."""
    import yaml
    from run import merge

    from wavetracker.config import Config
    from wavetracker.tracking import track

    variant = yaml.safe_load(Path(args.config).read_text())
    for kv in args.set or []:
        k, v = kv.split("=", 1)
        variant.setdefault("tracking", {})[k] = yaml.safe_load(v)
    rows = []
    for split in args.splits:
        for name in mixtures(split, args.only):
            mix = name.split("/", 1)[1]
            z = np.load(
                WORK / "data" / split / f"{mix}{'-pairs' if args.pairs else ''}.npz"
            )
            band = yaml.safe_load((MIXTURES / name / "band.yaml").read_text())
            cfg = Config.from_dict(merge(variant, band))
            truth = dict(np.load(MIXTURES / name / "truth.npz"))
            n_known = int(np.isfinite(truth_at(truth, z["times"])).sum())  # frames
            t0 = time.perf_counter()
            ident = track(
                z["fund"],
                z["idx"],
                z["sign"],
                z["times"],
                cfg.tracking,
                cfg.harmonic_groups.min_freq,
                cfg.harmonic_groups.max_freq,
                sal_v=z["sal"],
                cplx_v=z["cplx"],
            )
            F = truth_at(truth, z["times"])
            r = frag_stats(ident, z["fish"].astype(int), z["cls"], n_known, z["idx"], F)
            r |= {"mixture": name, "s": time.perf_counter() - t0}
            rows.append(r)
            print(
                f"{name}: switches {r['switches']} fragments {r['fragments_10']} "
                f"raw {r['idf1_raw']:.3f} majority {r['idf1_majority']:.3f} "
                f"({r['s']:.1f} s)",
                flush=True,
            )
    keys = [k for k in rows[0] if k.startswith("switches")] + [
        "fragments_10",
        "dets_in_short",
        "idf1_raw",
        "idf1_majority",
    ]
    print("mean " + " ".join(f"{k} {np.mean([r[k] for r in rows]):.4f}" for k in keys))
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))


# --- figures ---------------------------------------------------------------------

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def plot_switches(runs: list[tuple[str, str]], path: Path):
    """Switches per mixture by the two fish's distance (frags JSONs)."""
    import matplotlib.pyplot as plt

    keys = [f"switches {lo}-{hi} Hz" for lo, hi in SEP_BANDS]
    names = ["< 0.25 Hz\n(one line)", "0.25-1.2 Hz", "1.2-3 Hz", ">= 3 Hz"]
    fig, ax = plt.subplots(figsize=(6.4, 3.2), layout="constrained")
    width = 0.8 / len(runs)
    for i, (label, f) in enumerate(runs):
        rows = json.loads(Path(f).read_text())
        v = [np.mean([r[k] for r in rows]) for k in keys]
        x = np.arange(len(keys)) + (i - (len(runs) - 1) / 2) * width
        ax.bar(x, v, width * 0.92, color=SERIES[i], label=label)
        for xx, vv in zip(x, v, strict=True):
            ax.text(
                xx, vv, f"{vv:.0f}", ha="center", va="bottom", fontsize=7, color=MUTED
            )
    ax.set_xticks(np.arange(len(keys)), names)
    ax.set_ylabel("switches inside fragments\nper mixture", fontsize=8, color=INK)
    ax.set_title(
        "Raw tracker fragments: where they switch fish (ldev, lab noise)",
        fontsize=9,
        color=INK,
        loc="left",
    )
    _style(ax)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=160)
    print(path)


def plot_pair_oracle(rows: list[tuple[str, float, float]], path: Path):
    """IDF1 without and with the pair oracle: rows of (label, plain, oracle)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.6, 3.4), layout="constrained")
    for i, (label, a, b) in enumerate(rows):
        c = SERIES[i]
        ax.plot([0, 1], [a, b], color=c, lw=2, marker="o", ms=7)
        ax.text(
            1.05, b, f"{label}  {a:.3f} -> {b:.3f}", va="center", fontsize=8, color=INK
        )
    ax.set_xticks(
        [0, 1], ["detections\n(learned-b)", "+ hidden fish of\nclose pairs (oracle)"]
    )
    ax.set_xlim(-0.2, 2.6)
    ax.set_ylabel("IDF1 (mean, lab and field)", fontsize=8, color=INK)
    ax.set_title("Does a perfect pair head pay off?", fontsize=9, color=INK, loc="left")
    _style(ax)
    fig.savefig(path, dpi=160)
    print(path)


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
    t.add_argument(
        "--merged", action="store_true", help="merged peaks belong to both fish"
    )
    t.add_argument(
        "--hard-weight",
        type=float,
        default=1.0,
        help="weight of edges at detections with true and false candidates",
    )
    t.add_argument("--val-every", type=int, default=500)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="cuda")
    t.add_argument("--threads", type=int, default=2, help="CPU threads of torch")
    f = sub.add_parser("frags")
    f.add_argument("-c", "--config", required=True)
    f.add_argument("--splits", nargs="+", default=["ldev", "ldev-field"])
    f.add_argument("--only", nargs="*")
    f.add_argument("--set", nargs="*", help="tracking.<key>=<value>")
    f.add_argument("--pairs", action="store_true", help="with the pair oracle's fish")
    f.add_argument("--out")
    args = p.parse_args(argv)
    {"build": build, "train": train, "frags": frags}[args.cmd](args)


if __name__ == "__main__":
    main()
