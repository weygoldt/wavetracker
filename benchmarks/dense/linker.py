"""Train and evaluate the learned fragment linker (wavetracker.linking).

Detections and tracker fragments come from a benchmark run (``--base``, e.g.
the dense settings); the truth of each mixture says which fragments belong to
one fish. Training uses the ``train`` mixtures (train.yaml: dev sources, own
seeds), model selection the dev suite; the test suite is not touched.

Usage::

    python benchmarks/dense/linker.py build --base dense --splits train train-field dev dev-field
    python benchmarks/dense/linker.py train --base dense
    python benchmarks/dense/linker.py apply --base dense --tag dense-linked --splits dev dev-field
    python benchmarks/dense/run.py report --tags dense dense-linked

On the learned detector's fragments (its own training mixtures, ltrain), with
the tracker settings of the run given explicitly::

    python benchmarks/dense/linker.py build --base A-learned -c benchmarks/dense/configs/learned-linked.yaml --splits ltrain ltrain-field ldev ldev-field
    python benchmarks/dense/linker.py train --base A-learned --train-splits ltrain ltrain-field --val-splits ldev ldev-field --out <model.npz>
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA
from run import RUNS, merge, mixtures
from score import TOL, match, score, truth_at

from wavetracker import linking
from wavetracker.config import Config, LinkingConfig
from wavetracker.results import Results
from wavetracker.tracking import track

MIX = DATA / "mixtures"
WORK = DATA / "linker"
CONFIGS = Path(__file__).with_name("configs")


def base_config(base: str, name: str, config: str | None = None) -> Config:
    """The config of the base run: `config` if given, else configs/<base>.yaml."""
    path = Path(config) if config else CONFIGS / f"{base}.yaml"
    if config and not path.exists():
        raise FileNotFoundError(path)
    variant = (yaml.safe_load(path.read_text()) if path.exists() else {}) or {}
    band = yaml.safe_load((MIX / name / "band.yaml").read_text())
    return Config.from_dict(merge(variant, band))


def tracked(base: str, name: str, config: str | None = None):
    """Results of the base run with the tracker's identities before stitching
    (cached), and the fish of every detection. The tracker's settings come
    from `config` (default: configs/<base>.yaml)."""
    res = Results.load(RUNS / base / name)
    cache = WORK / base / name / "track.npz"
    if cache.exists():
        z = np.load(cache)
        return res, z["ident"], z["fish"]
    cfg = base_config(base, name, config)
    hg = cfg.harmonic_groups
    ident = track(
        res.fund_v,
        res.idx_v,
        res.sign_v,
        res.times,
        cfg.tracking,
        hg.min_freq,
        hg.max_freq,
    )
    truth = dict(np.load(MIX / name / "truth.npz"))
    fish = match(res, truth_at(truth, res.times), TOL)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, ident=ident, fish=fish)
    return res, ident, fish


def fragment_fish(fr: linking.Fragments, fish: np.ndarray):
    """Majority fish of every fragment (-1: none matched) and its share."""
    maj = np.full(len(fr), -1)
    share = np.zeros(len(fr))
    for i, k in enumerate(fr.members):
        f = fish[k]
        f = f[f >= 0]
        if len(f):
            c = np.bincount(f)
            maj[i] = c.argmax()
            share[i] = c.max() / len(k)
    return maj, share


def pair_labels(pairs, maj):
    """1 same fish, 0 different (or one of them on no fish); weight 0 if
    neither fragment is on a fish."""
    fa, fb = maj[pairs[:, 0]], maj[pairs[:, 1]]
    y = ((fa == fb) & (fa >= 0)).astype(np.float32)
    w = ((fa >= 0) | (fb >= 0)).astype(np.float32)
    return y, w


def build_one(
    base: str, name: str, cfg: LinkingConfig, config: str | None = None
) -> dict:
    res, ident, fish = tracked(base, name, config)
    fr = linking.fragments(
        res.fund_v, res.idx_v, res.sign_v, res.cplx_v, ident, res.times, cfg
    )
    valid = np.isfinite(ident)
    dets = linking._Detections(
        res.times[res.idx_v[valid]], res.fund_v[valid], ident[valid]
    )
    pairs = linking.candidates(fr, cfg)
    X = linking.features(fr, pairs, dets, cfg)
    maj, share = fragment_fish(fr, fish)
    y, w = pair_labels(pairs, maj)
    # candidate recall: consecutive fragments of one fish (in start order)
    n_true, n_found = 0, 0
    have = set(map(tuple, pairs))
    for k in np.unique(maj[maj >= 0]):
        idx = np.flatnonzero(maj == k)
        idx = idx[np.argsort(fr.start[idx])]
        for a, b in itertools.pairwise(idx):
            n_true += 1
            n_found += (a, b) in have or (b, a) in have
    return {
        "X": X.astype(np.float32),
        "y": y,
        "w": w,
        "pairs": pairs,
        "maj": maj,
        "share": share,
        "n": fr.n,
        "recall_consecutive": n_found / max(n_true, 1),
        "n_consecutive": n_true,
    }


def build(args):
    cfg = LinkingConfig()
    for split in args.splits:
        for name in mixtures(split, args.only):
            out = WORK / args.base / name / "pairs.npz"
            if out.exists() and not args.overwrite:
                continue
            t0 = time.perf_counter()
            d = build_one(args.base, name, cfg, args.config)
            np.savez(out, **d)
            print(
                f"{name}: {len(d['y'])} pairs ({d['y'].sum():.0f} same fish), "
                f"consecutive recall {d['recall_consecutive']:.3f} of "
                f"{d['n_consecutive']} ({time.perf_counter() - t0:.0f} s)",
                flush=True,
            )


def load_pairs(base: str, splits, only=None, weighted=True):
    Xs, ys, ws, groups = [], [], [], []
    names = [n for sp in splits for n in mixtures(sp, only)]
    for gi, name in enumerate(names):
        f = WORK / base / name / "pairs.npz"
        if not f.exists():
            continue
        z = np.load(f)
        Xs.append(z["X"])
        ys.append(z["y"])
        # weight by what a wrong decision costs: the smaller fragment's detections
        n_small = np.minimum(z["n"][z["pairs"][:, 0]], z["n"][z["pairs"][:, 1]])
        ws.append(z["w"] * np.sqrt(n_small / 10.0) if weighted else z["w"])
        groups.append(np.full(len(z["y"]), gi))
    return (
        np.concatenate(Xs),
        np.concatenate(ys),
        np.concatenate(ws),
        np.concatenate(groups),
    )


GROUPS = {
    "pattern": ["pattern_rms", "pattern_corr", "sign_agree", "dlevel"],
    "context": [
        "occupancy",
        "rank_a",
        "rank_b",
        "n_cand_a",
        "n_cand_b",
        "margin_a",
        "margin_b",
    ],
}


def fit_mlp(
    X, y, w, hidden=(64, 32), epochs=60, lr=3e-3, wd=1e-4, seed=0, val=None, drop=()
):
    import torch

    torch.manual_seed(seed)
    mean = X.astype(np.float64).mean(0)
    std = X.astype(np.float64).std(0)
    # features without variance in the training data say nothing, and would
    # explode on data where they vary (e.g. the number of electrodes)
    const = std < 1e-6 * (1 + np.abs(mean))
    mean[const], std[const] = 0.0, 1e12
    for g in drop:  # features the model must not see: standardised to ~0
        for k in GROUPS.get(g, [g]):
            j = linking.FEATURES.index(k)
            mean[j], std[j] = 0.0, 1e12
    Xt = torch.as_tensor((X - mean) / std, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.float32)
    wt = torch.as_tensor(w, dtype=torch.float32)
    layers, d = [], X.shape[1]
    for h in hidden:
        layers += [torch.nn.Linear(d, h), torch.nn.ReLU()]
        d = h
    layers.append(torch.nn.Linear(d, 1))
    net = torch.nn.Sequential(*layers)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = len(Xt)
    bs = 2048
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            j = perm[i : i + bs]
            out = net(Xt[j])[:, 0]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                out, yt[j], weight=wt[j]
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(j)
        sched.step()
        if val is not None and (ep % 10 == 9 or ep == epochs - 1):
            Xv, yv, wv = val
            with torch.no_grad():
                pv = net(torch.as_tensor((Xv - mean) / std, dtype=torch.float32))[:, 0]
                lv = torch.nn.functional.binary_cross_entropy_with_logits(
                    pv,
                    torch.as_tensor(yv),
                    weight=torch.as_tensor(wv),
                ).item()
            print(f"  epoch {ep + 1}: train {tot / n:.4f} val {lv:.4f}", flush=True)
    lin = [m for m in net if isinstance(m, torch.nn.Linear)]
    return linking.LinkModel(
        mean=mean.astype(np.float64),
        std=std.astype(np.float64),
        weights=[m.weight.detach().numpy().T.astype(np.float64) for m in lin],
        biases=[m.bias.detach().numpy().astype(np.float64) for m in lin],
        features=list(linking.FEATURES),
    )


def train(args):
    parts = [load_pairs(b, args.train_splits) for b in [args.base, *args.extra_bases]]
    X, y, w = (np.concatenate([q[i] for q in parts]) for i in range(3))
    Xv, yv, wv, _ = load_pairs(args.base, args.val_splits, weighted=False)
    print(f"train {len(y)} pairs ({y.mean():.3f} same), val {len(yv)}", flush=True)
    model = fit_mlp(
        X,
        y,
        w,
        hidden=tuple(args.hidden),
        epochs=args.epochs,
        val=(Xv, yv, wv),
        drop=args.drop,
        seed=args.seed,
    )
    out = Path(args.out or WORK / args.base / "linker.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    pv = model.prob(Xv)
    sel = wv > 0
    acc = ((pv[sel] > 0.5) == (yv[sel] > 0.5)).mean()
    tp = ((pv > 0.5) & (yv > 0.5) & sel).sum()
    print(
        f"val accuracy {acc:.4f}, precision {tp / max(((pv > 0.5) & sel).sum(), 1):.4f}, "
        f"recall {tp / max(((yv > 0.5) & sel).sum(), 1):.4f} -> {out}",
        flush=True,
    )


class OracleModel:
    """Same fish (from the truth) instead of a prediction: the bound of the
    candidates and the association."""

    def __init__(self, fish):
        self.fish = fish
        self.fr = None

    def bind(self, fr):
        self.fr = fr

    def prob(self, X, pairs=None):
        maj, _ = fragment_fish(self.fr, self.fish)
        y, _ = pair_labels(pairs, maj)
        return y


def apply(args):
    cfg = LinkingConfig(
        threshold=args.threshold,
        max_shared=args.max_shared,
        **{k: float(v) for k, v in (kv.split("=") for kv in args.set or [])},
    )
    model = (
        "oracle"
        if args.oracle
        else linking.LinkModel.load(args.model or WORK / args.base / "linker.npz")
    )
    rows = []
    for split in args.splits:
        for name in mixtures(split, args.only):
            res, ident, fish = tracked(args.base, name, args.config)
            t0 = time.perf_counter()
            res.ident_v = linking.link(
                res.fund_v,
                res.idx_v,
                res.sign_v,
                res.cplx_v,
                ident,
                res.times,
                cfg,
                OracleModel(fish) if model == "oracle" else model,
            )
            t_link = time.perf_counter() - t0
            out = RUNS / args.tag / name
            res.meta["linking"] = vars(cfg) | {"model": str(args.model or "default")}
            res.save(out)
            summary, table = score(MIX / name, out)
            summary |= {"tag": args.tag, "link_s": t_link}
            (out / "score.json").write_text(json.dumps(summary, indent=1))
            np.savez_compressed(out / "frames.npz", **table)
            rows.append(summary)
            print(
                f"{args.tag} {name}: idf1 {summary['idf1']:.3f} purity "
                f"{summary['purity']:.3f} ids {summary['ids_30s']} ({t_link:.1f} s)",
                flush=True,
            )
    if rows:
        print(
            f"mean idf1 {np.mean([r['idf1'] for r in rows]):.4f} "
            f"purity {np.mean([r['purity'] for r in rows]):.4f}"
        )


def id_scores(ident, fish, cls, n_known) -> dict:
    """IDF1 and purity as in score.py, from arrays."""
    from scipy.optimize import linear_sum_assignment

    has = np.isfinite(ident)
    ids = np.unique(ident[has])
    counts = np.zeros((len(ids), len(n_known)), int)
    m = (fish >= 0) & has
    np.add.at(counts, (np.searchsorted(ids, ident[m]), fish[m]), 1)
    r, c = linear_sum_assignment(-counts) if len(ids) else ([], [])
    idtp = int(counts[r, c].sum())
    n_pred = int((has & (cls != 1)).sum())
    return {
        "idf1": 2 * idtp / max(n_pred + int(n_known.sum()), 1),
        "purity": counts.max(1).sum() / max(counts.sum(), 1),
        "id_acc": idtp / max(int(m.sum()), 1),
    }


class Sweep:
    """Everything linking needs per mixture, in memory, to compare models
    and thresholds quickly (dev splits)."""

    def __init__(self, base, splits, only=None, config=None):
        from score import classify_unmatched

        cfg = LinkingConfig()
        self.items = []
        for split in splits:
            for name in mixtures(split, only):
                res, ident, fish = tracked(base, name, config)
                truth = dict(np.load(MIX / name / "truth.npz"))
                F = truth_at(truth, res.times)
                Ff = truth_at(truth, res.times, "freqs_filled")
                cls = classify_unmatched(res, F, Ff, fish, TOL)
                fr = linking.fragments(
                    res.fund_v, res.idx_v, res.sign_v, res.cplx_v, ident, res.times, cfg
                )
                z = np.load(WORK / base / name / "pairs.npz")
                frames = [np.unique(res.idx_v[k]) for k in fr.members]
                self.items.append(
                    dict(
                        name=name,
                        res=res,
                        ident=ident,
                        fish=fish,
                        cls=cls,
                        n_known=np.isfinite(F).sum(1),
                        fr=fr,
                        X=z["X"],
                        pairs=z["pairs"],
                        frames=frames,
                    )
                )

    def run(self, prob_fn, threshold, **overrides):
        from wavetracker.stitching import resolve_duplicates

        cfg = LinkingConfig(threshold=threshold, **overrides)
        out = []
        for it in self.items:
            fr, res = it["fr"], it["res"]
            ident = it["ident"].copy()
            if len(it["pairs"]):
                p = prob_fn(it)
                roots = linking.associate(
                    fr, it["pairs"], p, lambda i, it=it: it["frames"][i], cfg
                )
                for i, r in enumerate(roots):
                    if r != i:
                        ident[fr.members[i]] = fr.ident[r]
                ident = resolve_duplicates(res.fund_v, res.idx_v, ident)
            out.append(
                {"name": it["name"], "n_fish": len(it["n_known"])}
                | id_scores(ident, it["fish"], it["cls"], it["n_known"])
            )
        return out


def summarize(rows) -> str:
    parts = []
    splits = list(dict.fromkeys(x["name"].split("/")[0] + "/" for x in rows))
    for sp in splits:
        r = [x for x in rows if x["name"].startswith(sp)]
        if r:
            parts.append(
                f"{sp[:-1]}: idf1 {np.mean([x['idf1'] for x in r]):.4f} "
                f"purity {np.mean([x['purity'] for x in r]):.4f}"
            )
    return " | ".join(parts)


def sweep(args):
    sw = Sweep(args.base, args.splits, config=args.config)
    base = sw.run(lambda it: np.zeros(len(it["pairs"])), 1.0)
    print(f"{'tracker only':28s} {summarize(base)}")
    oracle = sw.run(
        lambda it: pair_labels(it["pairs"], fragment_fish(it["fr"], it["fish"])[0])[0],
        0.5,
    )
    print(f"{'oracle':28s} {summarize(oracle)}")
    over = {k: float(v) for k, v in (kv.split("=") for kv in args.set or [])}
    for path in args.models:
        model = linking.LinkModel.load(path)
        for th in args.thresholds:
            rows = sw.run(lambda it, m=model: m.prob(it["X"]), th, **over)
            print(f"{Path(path).stem + f' @{th}':28s} {summarize(rows)}", flush=True)


THRESHOLDS = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98]


def tradeoff(args):
    """IDF1 against purity for the tracker alone, stitching, the linker at
    several thresholds and the oracle; json and figure."""
    model = linking.LinkModel.load(args.model)
    over = {k: float(v) for k, v in (kv.split("=") for kv in args.set or [])}
    out = {"model": str(args.model), "set": over, "panels": {}}
    sw = Sweep(args.base, ["dev", "dev-field"], config=args.config)
    for sp in ("dev", "dev-field"):

        def mean(rows, sp=sp):
            r = [x for x in rows if x["name"].startswith(sp + "/")]
            return {k: float(np.mean([x[k] for x in r])) for k in ("idf1", "purity")}

        sub = Sweep.__new__(Sweep)
        sub.items = [it for it in sw.items if it["name"].startswith(sp + "/")]
        panel = {
            "tracker only": mean(sub.run(lambda it: np.zeros(len(it["pairs"])), 1.0)),
            "oracle": mean(
                sub.run(
                    lambda it: pair_labels(
                        it["pairs"], fragment_fish(it["fr"], it["fish"])[0]
                    )[0],
                    0.5,
                    **over,
                )
            ),
            "linker": [
                mean(sub.run(lambda it: model.prob(it["X"]), th, **over))
                | {"threshold": th}
                for th in THRESHOLDS
            ],
        }
        stitched = [
            json.loads((RUNS / args.base / n / "score.json").read_text())
            for n in mixtures(sp)
        ]
        panel["stitching"] = {
            k: float(np.mean([x[k] for x in stitched])) for k in ("idf1", "purity")
        }
        out["panels"][sp] = panel
        print(sp, json.dumps(panel), flush=True)
    path = Path(args.out)
    path.with_suffix(".json").write_text(json.dumps(out, indent=1))
    plot_tradeoff(out, path)
    print(path)


TITLES = {
    "dev": ("Benchmark, lab noise", "dev, 17 mixtures, 4-30 fish"),
    "dev-field": ("Benchmark, field noise", "dev-field, the same fish in field noise"),
}


def plot_tradeoff(out, path):
    import matplotlib.pyplot as plt
    from run import GRAY, INK, MUTED, SERIES, _style

    panels = out["panels"]
    fig, axs = plt.subplots(
        1, len(panels), figsize=(5.2 * len(panels), 4.6), layout="constrained"
    )
    axs = np.atleast_1d(axs)
    for ax, (name, p) in zip(axs, panels.items(), strict=True):
        _style(ax)
        lk = p["linker"]
        ax.plot(
            [q["purity"] for q in lk],
            [q["idf1"] for q in lk],
            "-o",
            color=SERIES[1],
            lw=2,
            ms=6,
            mec="white",
            mew=1.5,
            zorder=3,
        )
        for q in lk:
            if q["threshold"] in (0.5, 0.8, 0.9, 0.98):
                ax.annotate(
                    f"{q['threshold']:g}",
                    (q["purity"], q["idf1"]),
                    xytext=(5, 4),
                    textcoords="offset points",
                    fontsize=7,
                    color=MUTED,
                )
        ax.annotate(
            "learned linker\n(numbers: threshold)",
            (lk[-1]["purity"], lk[-1]["idf1"]),
            xytext=(-12, -2),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=8,
            color=INK,
        )
        marks = [
            ("tracker only", GRAY, "o"),
            ("stitching", SERIES[0], "s"),
            ("oracle", SERIES[2], "*"),
        ]
        for key, color, m in marks:
            if key not in p:
                continue
            ax.plot(
                p[key]["purity"],
                p[key]["idf1"],
                m,
                color=color,
                ms=11 if m == "*" else 8,
                mec="white",
                mew=1.5,
                zorder=4,
            )
            label = {"oracle": "oracle joins", "stitching": "stitching (now)"}.get(
                key, key
            )
            ax.annotate(
                label,
                (p[key]["purity"], p[key]["idf1"]),
                xytext=(6, -10),
                textcoords="offset points",
                fontsize=8,
                color=INK,
            )
        title, sub = TITLES[name]
        ax.set_title(f"{title}\n{sub}", loc="left", fontsize=9, color=INK)
        ax.set_xlabel(
            "purity (matched detections on their identity's fish)",
            fontsize=8,
            color=MUTED,
        )
        ax.set_ylabel("IDF1 (one identity per fish)", fontsize=8, color=MUTED)
        xs = [q["purity"] for q in lk] + [p[k]["purity"] for k, _, _ in marks if k in p]
        ax.set_xlim(min(xs) - 0.004, max(xs) + 0.006)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("build")
    a.add_argument("--base", default="dense")
    a.add_argument(
        "-c", "--config", help="config of the base run (default: configs/<base>.yaml)"
    )
    a.add_argument(
        "--splits", nargs="+", default=["train", "train-field", "dev", "dev-field"]
    )
    a.add_argument("--only", nargs="*")
    a.add_argument("--overwrite", action="store_true")
    a.set_defaults(func=build)
    b = sub.add_parser("train")
    b.add_argument("--base", default="dense")
    b.add_argument("--train-splits", nargs="+", default=["train", "train-field"])
    b.add_argument("--val-splits", nargs="+", default=["dev", "dev-field"])
    b.add_argument("--hidden", nargs="*", type=int, default=[64, 32])
    b.add_argument("--epochs", type=int, default=60)
    b.add_argument("--out")
    b.add_argument("--drop", nargs="*", default=[], help="feature groups or names")
    b.add_argument("--extra-bases", nargs="*", default=[], help="more training runs")
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(func=train)
    c = sub.add_parser("apply")
    c.add_argument("--base", default="dense")
    c.add_argument(
        "-c", "--config", help="config of the base run (default: configs/<base>.yaml)"
    )
    c.add_argument("--tag", required=True)
    c.add_argument("--model")
    c.add_argument("--splits", nargs="+", default=["dev", "dev-field"])
    c.add_argument("--only", nargs="*")
    c.add_argument("--threshold", type=float, default=0.9)
    c.add_argument("--max-shared", type=float, default=0.05)
    c.add_argument("--oracle", action="store_true", help="truth instead of the model")
    c.add_argument("--set", nargs="*", help="LinkingConfig overrides, key=value")
    c.set_defaults(func=apply)
    d = sub.add_parser("sweep")
    d.add_argument("--base", default="dense")
    d.add_argument(
        "-c", "--config", help="config of the base run (default: configs/<base>.yaml)"
    )
    d.add_argument("--splits", nargs="+", default=["dev", "dev-field"])
    d.add_argument("--models", nargs="+", required=True)
    d.add_argument("--thresholds", nargs="+", type=float, default=[0.5, 0.7, 0.9])
    d.add_argument("--set", nargs="*", help="LinkingConfig overrides, key=value")
    d.set_defaults(func=sweep)
    e = sub.add_parser("tradeoff")
    e.add_argument("--base", default="dense")
    e.add_argument(
        "-c", "--config", help="config of the base run (default: configs/<base>.yaml)"
    )
    e.add_argument("--model", required=True)
    e.add_argument("--set", nargs="*", help="LinkingConfig overrides, key=value")
    e.add_argument("--out", default=str(DATA / "figures" / "linker" / "tradeoff.png"))
    e.set_defaults(func=tradeoff)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
