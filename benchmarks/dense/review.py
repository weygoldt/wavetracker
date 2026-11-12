"""Confidence estimates of tracked results: collect evidence and labels, fit
the calibrated models, and judge them by simulated review (see
docs/review-flags.md and wavetracker.review).

``collect`` re-tracks the detections of a run with ``review.enabled`` (CPU,
deterministic; the identities are those of the plain run, which ``--check``
verifies), and stores every item (join, track, missed join) with its
features and its label from the truth: a join is right if both sides belong
mostly to the same fish; a track is right if most of its detections are on a
fish (matched by ``score.match``, not a harmonic); not joining is right
unless both identities belong mostly to the same fish. Files:
``$WT_BENCHMARK_DATA/confidence/<tag>/<split>/<mixture>.pkl``.

``fit`` fits one model per item group on the collected training splits
(``ltrain``, ``ltrain-field``), ``evaluate`` checks calibration and AUC on
the dev splits and simulates review: the items are visited in an order (by
confidence, by time, at random, ...), and where an item is really wrong the
right fix is applied (split the join, delete the track, merge the
identities); IDF1 after every item gives the review-efficiency curve.

Usage::

    python benchmarks/dense/review.py collect --tag E-cf-rises \\
        -c benchmarks/dense/configs/learned-rises.yaml \\
        --source B-rise-lt --splits ltrain ltrain-field
    python benchmarks/dense/review.py collect --tag E-cf-rises -c ... \\
        --source B-aug-s0-t0.6:ldev,ref-learned-b:ldev2 \\
        --check B-rise-j3:ldev,ref-learned-rises:ldev2 \\
        --splits ldev ldev-field ldev2 ldev2-field
    python benchmarks/dense/review.py fit --tag E-cf-rises --out model.npz
    python benchmarks/dense/review.py evaluate --tag E-cf-rises --model model.npz
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA
from compare import parse_spec, tag_for
from mixture import MIXTURES
from run import merge, mixtures
from score import TRUTHS, classify_unmatched, match, truth_at

from wavetracker.config import Config
from wavetracker.results import Results

RUNS = DATA / "runs"
OUT = DATA / "confidence"
FIG = DATA / "figures" / "confidence"
TRAIN = ("ltrain", "ltrain-field")
DEV = ("ldev", "ldev-field", "ldev2", "ldev2-field")


# --- labels --------------------------------------------------------------------


def majority(fish: np.ndarray, cls: np.ndarray, rows: np.ndarray) -> int:
    """The fish most of `rows` are on (-1: most are on none), ignoring
    detections where the truth is unknown."""
    f = fish[rows][cls[rows] != 1]
    if not len(f):
        return -1
    u, c = np.unique(f, return_counts=True)
    return int(u[np.argmax(c)])


def label(item: dict, fish: np.ndarray, cls: np.ndarray) -> int:
    """1 if the result is right at the item (see module doc), else 0."""
    g = item["group"]
    if g == "track":
        return int(majority(fish, cls, item["rows"][0]) >= 0)
    a = majority(fish, cls, item["rows"][0])
    b = majority(fish, cls, item["rows"][1])
    if g.startswith("join"):
        return int(a == b)  # two pieces of clutter joined: not this join's fault
    return int(not (a == b and a >= 0))


# --- collect -------------------------------------------------------------------


def _collect_one(job) -> str:
    name, src, check, variant, out = job
    from wavetracker.pipeline import track_results

    band = yaml.safe_load((MIXTURES / name / "band.yaml").read_text())
    cfg = Config.from_dict(merge(variant, band))
    res = Results.load(RUNS / src / name)
    t0 = time.perf_counter()
    trace_items = _track(res, cfg, track_results)
    dt = time.perf_counter() - t0
    same = None
    if check:
        ref = Results.load(RUNS / check / name)
        same = bool(np.array_equal(ref.ident_v, res.ident_v, equal_nan=True))
    truth = dict(np.load(MIXTURES / name / TRUTHS["default"]))
    F = truth_at(truth, res.times)
    Ff = truth_at(truth, res.times, "freqs_filled")
    fish = match(res, F)
    cls = classify_unmatched(res, F, Ff, fish)
    items, conf = trace_items
    rows = []
    for it, c in zip(items, conf, strict=True):
        d = {
            "group": it.group,
            "t": it.t,
            "f": it.f,
            "ids": it.ids,
            "raw": it.raw,
            "conf0": float(c),
            "x": it.x,
            "impact": it.impact,
            "k": it.k,
            "rows": tuple(np.asarray(r, np.int32) for r in it.rows),
        }
        d["label"] = label(d, fish, cls)
        rows.append(d)
    data = {
        "name": name,
        "items": rows,
        "ident": res.ident_v,
        "fund": res.fund_v.astype(np.float32),
        "idx": res.idx_v.astype(np.int32),
        "times": res.times,
        "fish": fish.astype(np.int16),
        "cls": cls.astype(np.int8),
        "n_known": int(np.isfinite(F).sum()),
        "track_s": dt,
        "identical": same,
        "freq_resolution": res.meta.get("freq_resolution"),
        "tracking_config": res.meta.get("tracking_config"),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return f"{name}: {len(rows)} items, {dt:.0f} s, identical {same}"


def _track(res, cfg, track_results):
    """Track with review on; returns the items and their raw confidences."""
    import wavetracker.pipeline as pl

    keep = {}
    orig = pl.review

    def spy(results, trace, cfg):
        out = orig(results, trace, cfg)
        keep["items"], keep["conf"] = out[1], out[2]
        return out

    pl.review = spy
    try:
        track_results(res, cfg)
    finally:
        pl.review = orig
    return keep["items"], keep["conf"]


def collect(args):
    variant = yaml.safe_load(Path(args.config).read_text()) or {}
    variant = merge(variant, {"review": {"enabled": True, "model": None}})
    src = parse_spec(args.source)
    chk = parse_spec(args.check) if args.check else None
    jobs = []
    for sp in args.splits:
        for name in mixtures(sp, args.only):
            out = OUT / args.tag / f"{name}.pkl"
            if out.exists() and not args.overwrite:
                continue
            check = tag_for(chk, sp) if chk else None
            jobs.append((name, tag_for(src, sp), check, variant, out))
    print(f"{len(jobs)} mixtures", flush=True)
    if args.jobs > 1:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(args.jobs, initializer=_init_worker) as pool:
            for msg in pool.imap_unordered(_collect_one, jobs):
                print(msg, flush=True)
    else:
        for job in jobs:
            print(_collect_one(job), flush=True)


def _init_worker():
    import torch

    torch.set_num_threads(int(os.environ.get("WORKER_THREADS", "2")))


def load_split(tag: str, splits) -> list[dict]:
    out = []
    for sp in splits:
        for f in sorted((OUT / tag / sp).glob("*.pkl")):
            with open(f, "rb") as fh:
                d = pickle.load(fh)
            d["split"] = sp
            out.append(d)
    return out


# --- fit -----------------------------------------------------------------------


def features_of(group: str, items: list[dict]) -> list[str]:
    from wavetracker import review as rv

    names = list(rv.FEATURES[group])
    if group == "track" and items and "assoc_mean" in items[0]["x"]:
        names += rv.ASSOC
    return names


def table(data: list[dict], group: str):
    """Features, labels, raw scores and mixture numbers of one group."""
    items = [
        (m, it) for m, d in enumerate(data) for it in d["items"] if it["group"] == group
    ]
    names = features_of(group, [it for _, it in items])
    X = np.array([[it["x"].get(k, np.nan) for k in names] for _, it in items], float)
    X = X.reshape(len(items), len(names))
    y = np.array([it["label"] for _, it in items], float)
    raw = np.array([it["raw"] for _, it in items], float)
    mix = np.array([m for m, _ in items], int)
    # lab and field variants of a mixture hold the same fish: one unit
    unit = np.array([_unit(data[m]["split"], data[m]["name"]) for m, _ in items])
    return names, X, y, raw, mix, unit


def _unit(split: str, name: str) -> str:
    return f"{split.removesuffix('-field')}/{name.split('/', 1)[1]}"


def _standardise(X):
    mean = np.nanmean(X, 0)
    std = np.nanstd(X, 0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 1e-9), std, 1.0)
    return mean, std


def _z(X, mean, std):
    z = (X - mean) / std
    return np.clip(np.where(np.isfinite(z), z, 0.0), -8, 8)


def fit_logistic(Z, y, lam: float):
    """L2-penalised logistic regression (bias not penalised)."""
    from scipy.optimize import minimize

    d = Z.shape[1]

    def f(wb):
        w, b = wb[:d], wb[d]
        z = Z @ w + b
        p = 1 / (1 + np.exp(-z))
        loss = np.sum(np.logaddexp(0, z) - y * z) + 0.5 * lam * w @ w
        g = p - y
        return loss, np.r_[Z.T @ g + lam * w, g.sum()]

    p0 = np.clip(y.mean(), 1e-3, 1 - 1e-3)
    wb0 = np.r_[np.zeros(d), np.log(p0 / (1 - p0))]
    r = minimize(f, wb0, jac=True, method="L-BFGS-B")
    return r.x[:d].reshape(d, 1), np.array([r.x[d]])


def fit_mlp(Z, y, hidden=16, wd=1e-2, epochs=400, seed=0):
    import torch

    torch.manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(Z.shape[1], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1)
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=wd)
    zt = torch.as_tensor(Z, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(zt)[:, 0], yt)
        loss.backward()
        opt.step()
    w = [
        net[0].weight.detach().numpy().T.copy(),
        net[2].weight.detach().numpy().T.copy(),
    ]
    b = [net[0].bias.detach().numpy().copy(), net[2].bias.detach().numpy().copy()]
    return w, b


def isotonic(x, y):
    """Pool-adjacent-violators: knots (x, y) of a non-decreasing map."""
    o = np.argsort(x, kind="stable")
    xs, ys = x[o], y[o].astype(float)
    blocks = []  # [sum y, n, sum x]
    for xi, yi in zip(xs, ys, strict=True):
        blocks.append([yi, 1, xi])
        while (
            len(blocks) > 1
            and blocks[-2][0] / blocks[-2][1] >= blocks[-1][0] / blocks[-1][1]
        ):
            s, n, sx = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += n
            blocks[-1][2] += sx
    kx = np.array([b[2] / b[1] for b in blocks])
    ky = np.array([b[0] / b[1] for b in blocks])
    return kx, ky


def _predict(w, b, Z):
    h = Z
    for i, (wi, bi) in enumerate(zip(w, b, strict=True)):
        h = h @ wi + bi
        if i < len(w) - 1:
            h = np.maximum(h, 0)
    return 1 / (1 + np.exp(-h[:, 0]))


def logloss(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    i = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    out = 0.0
    for b in range(bins):
        m = i == b
        if m.any():
            out += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return float(out)


def auc(score, y):
    """P(score of a right item > score of a wrong one), ties half."""
    pos, neg = score[y == 1], score[y == 0]
    if not len(pos) or not len(neg):
        return np.nan
    from scipy.stats import rankdata

    r = rankdata(np.r_[pos, neg])
    return float(
        (r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    )


CANDIDATES = {
    "raw-isotonic": None,
    "logistic-l1": ("logistic", 1.0),
    "logistic-l10": ("logistic", 10.0),
    "logistic-l100": ("logistic", 100.0),
    "mlp-16": ("mlp", 1e-2),
}


def _fit_candidate(kind, Z, y, raw):
    if kind is None:
        kx, ky = isotonic(raw, y)
        return ("iso", kx, ky)
    method, reg = kind
    if method == "logistic":
        w, b = fit_logistic(Z, y, reg)
        return ("net", [w], [b])
    return ("net", *fit_mlp(Z, y, wd=reg))


def _apply_candidate(m, Z, raw):
    if m[0] == "iso":
        return np.interp(raw, m[1], m[2])
    return _predict(m[1], m[2], Z)


def crossval(Z, y, raw, unit, folds=5, seed=0):
    """Out-of-fold predictions of every candidate, folds by mixture."""
    units = np.unique(unit)
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(units, rng.permutation(len(units)) % folds, strict=True))
    f = np.array([fold_of[u] for u in unit])
    out = {name: np.zeros(len(y)) for name in CANDIDATES}
    for k in range(folds):
        tr, te = f != k, f == k
        if not te.any():
            continue
        for name, kind in CANDIDATES.items():
            if len(np.unique(y[tr])) < 2:
                out[name][te] = y[tr].mean() if tr.any() else 0.5
                continue
            m = _fit_candidate(kind, Z[tr], y[tr], raw[tr])
            out[name][te] = _apply_candidate(m, Z[te], raw[te])
    return out


MIN_ITEMS = 40  # fewer: a constant (the base rate) or the raw score


def fit(args):
    from wavetracker.review import GROUPS, GroupModel, ReviewModel

    data = load_split(args.tag, args.splits)
    print(f"{len(data)} mixtures of {args.tag} ({', '.join(args.splits)})")
    groups = {}
    summary = {}
    oof_all: dict[str, list] = {}
    oof_items: dict[str, tuple] = {}
    for g in GROUPS:
        names, X, y, raw, _mix, unit = table(data, g)
        n, n_bad = len(y), int((y == 0).sum())
        if n == 0:
            continue
        mean, std = _standardise(X)
        Z = _z(X, mean, std)
        if n < MIN_ITEMS or n_bad < 5 or n - n_bad < 5:
            # too few to learn: the base rate, monotone in nothing
            base = float(np.clip((y.sum() + 1) / (n + 2), 0.01, 0.99))
            groups[g] = GroupModel(
                names,
                mean,
                std,
                [np.zeros((len(names), 1))],
                [np.array([np.log(base / (1 - base))])],
            )
            summary[g] = {"n": n, "wrong": n_bad, "chosen": "base rate", "base": base}
            print(f"{g}: {n} items, {n_bad} wrong: base rate {base:.3f}")
            continue
        oof = crossval(Z, y, raw, unit, seed=args.seed)
        scores = {
            name: {"logloss": logloss(p, y), "ece": ece(p, y), "auc": auc(p, y)}
            for name, p in oof.items()
        }
        scores["raw"] = {
            "logloss": logloss(raw, y),
            "ece": ece(raw, y),
            "auc": auc(raw, y),
        }
        chosen = args.method or min(CANDIDATES, key=lambda k: scores[k]["logloss"])
        m = _fit_candidate(CANDIDATES[chosen], Z, y, raw)
        if m[0] == "iso":
            # the raw score as the only feature, mapped monotonically
            groups[g] = _raw_model(g, names, m[1], m[2])
        else:
            groups[g] = GroupModel(names, mean, std, list(m[1]), list(m[2]))
        if args.isotonic and m[0] != "iso":
            kx, ky = isotonic(oof[chosen], y)
            groups[g].cal_x, groups[g].cal_y = kx, ky
        summary[g] = {
            "n": n,
            "wrong": n_bad,
            "chosen": chosen,
            "cv": scores,
            "oof_bins": _bins(oof[chosen], y),
            "oof_kind": kind_of(g),
        }
        oof_all.setdefault(kind_of(g), []).append((oof[chosen], y))
        oof_items[g] = (oof[chosen], y)
        line = ", ".join(
            f"{k} {v['logloss']:.3f}/{v['auc']:.3f}" for k, v in scores.items()
        )
        print(f"{g}: {n} items, {n_bad} wrong; cv logloss/auc: {line}; chosen {chosen}")
    oof_kinds = {}
    for k, parts in oof_all.items():
        p = np.concatenate([a for a, _ in parts])
        y = np.concatenate([b for _, b in parts])
        oof_kinds[k] = {
            "n": len(y),
            "logloss": logloss(p, y),
            "ece": ece(p, y),
            "auc": auc(p, y),
            "bins": _bins(p, y),
        }
    meta = {
        "tag": args.tag,
        "splits": list(args.splits),
        "fitted": time.strftime("%Y-%m-%d %H:%M"),
        "summary": summary,
        "oof_kinds": oof_kinds,
    }
    ReviewModel(groups, meta).save(args.out)
    np.savez_compressed(
        Path(args.out).with_suffix(".oof.npz"),
        **{f"{kind_of(g)}/{g}/p": v[0] for g, v in oof_items.items()},
        **{f"{kind_of(g)}/{g}/y": v[1] for g, v in oof_items.items()},
    )
    (Path(args.out).with_suffix(".json")).write_text(
        json.dumps(meta, indent=1, default=float)
    )
    print(f"wrote {args.out}")


def _raw_model(group, names, kx, ky):
    """A GroupModel that passes the raw score through (feature ``raw_logit``,
    weight 1) and maps it by the isotonic knots."""
    from wavetracker.review import GroupModel

    return GroupModel(
        ["raw_logit"],
        np.zeros(1),
        np.ones(1),
        [np.ones((1, 1))],
        [np.zeros(1)],
        cal_x=kx,
        cal_y=ky,
    )


def _logit(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


# --- evaluate: calibration ------------------------------------------------------


def model_conf(data: list[dict], model) -> None:
    """Confidence of every item (``conf``) by `model` (None: the raw score)."""
    for d in data:
        for it in d["items"]:
            it["x"].setdefault("raw_logit", float(_logit(it["raw"])))
            it["conf"] = it["raw"]
        if model is None:
            continue
        for g, m in model.groups.items():
            sel = [it for it in d["items"] if it["group"] == g]
            if not sel:
                continue
            X = np.array(
                [[it["x"].get(k, np.nan) for k in m.features] for it in sel], float
            )
            for it, c in zip(
                sel, m.prob(X.reshape(len(sel), len(m.features))), strict=True
            ):
                it["conf"] = float(c)


def _bins(p, y, n=10):
    edges = np.linspace(0, 1, n + 1)
    i = np.clip(np.digitize(p, edges) - 1, 0, n - 1)
    return [
        (float(p[i == b].mean()), float(y[i == b].mean()), int((i == b).sum()))
        for b in range(n)
        if (i == b).any()
    ]


def kind_of(group: str) -> str:
    return group.split("_")[0].replace("missed", "missed_join")


def calibration_table(data: list[dict], key="conf") -> dict:
    """Per kind and group: n, wrong, logloss, ECE, AUC of `key`, and the
    reliability bins."""
    out = {}
    rows = [it for d in data for it in d["items"]]
    for name, sel in (
        [("all", rows)]
        + [
            (k, [it for it in rows if kind_of(it["group"]) == k])
            for k in ("join", "track", "missed_join")
        ]
        + [
            (g, [it for it in rows if it["group"] == g])
            for g in (
                "join_linker",
                "join_rise",
                "track",
                "missed_linker",
                "missed_rise",
            )
        ]
    ):
        if not sel:
            continue
        p = np.array([it[key] for it in sel])
        y = np.array([it["label"] for it in sel], float)
        bins = _bins(p, y)
        out[name] = {
            "n": len(sel),
            "wrong": int((y == 0).sum()),
            "rate_right": float(y.mean()),
            "mean_conf": float(p.mean()),
            "logloss": logloss(p, y),
            "brier": float(np.mean((p - y) ** 2)),
            "ece": ece(p, y),
            "auc": auc(p, y),
            "bins": bins,
        }
    return out


# --- evaluate: simulated review -----------------------------------------------------


class Sim:
    """IDF1 of a mixture's identities as fixes are applied (as score.score)."""

    def __init__(self, d: dict):
        self.fish = d["fish"].astype(np.int64)
        self.cls = d["cls"]
        self.n_known = d["n_known"]
        self.n_fish = int(max(self.fish.max() + 1, 1))
        self.counted = self.cls != 1

    def idf1(self, ident) -> float:
        from scipy.optimize import linear_sum_assignment

        has = np.isfinite(ident)
        n_pred = int((has & self.counted).sum())
        m = has & (self.fish >= 0)
        if not m.any():
            return 0.0
        ids, inv = np.unique(ident[m], return_inverse=True)
        counts = np.bincount(
            inv * self.n_fish + self.fish[m], minlength=len(ids) * self.n_fish
        ).reshape(len(ids), self.n_fish)
        r, c = linear_sum_assignment(-counts)
        idtp = counts[r, c].sum()
        return 2 * idtp / max(n_pred + self.n_known, 1)

    def maj(self, rows) -> int:
        return majority(self.fish, self.cls, rows)


def _mode(ident, rows):
    v = ident[rows]
    v = v[np.isfinite(v)]
    if not len(v):
        return None
    u, c = np.unique(v, return_counts=True)
    return float(u[np.argmax(c)])


def fix(sim: Sim, it: dict, ident: np.ndarray, state: dict) -> bool:
    """Apply the right fix if the result is wrong at the item now (a user
    looks at the current identities); True if something changed."""
    g = it["group"]
    rows = it["rows"]
    if g in ("track", "rule_track"):
        i0 = it["ids"][0]
        sel = rows[0][ident[rows[0]] == i0]
        if not len(sel) or sim.maj(sel) >= 0:
            return False
        ident[sel] = np.nan
        return True
    if g in ("join_linker", "join_rise", "rule_split"):
        ra, rb = rows
        ia, ib = _mode(ident, ra), _mode(ident, rb)
        if ia is None or ia != ib:
            return False
        fa, fb = sim.maj(ra), sim.maj(rb)
        if fa == fb:
            return False
        # keep the side on the identity's own fish, move the other one out
        main = sim.maj(np.flatnonzero(ident == ia))
        if fb == main or (fa != main and fb >= 0 > fa):
            move, fm = ra[ident[ra] == ia], fa
        else:
            move, fm = rb[ident[rb] == ia], fb
        if fm < 0:
            ident[move] = np.nan
            return True
        if state.get("reassign", True):  # to the identity of its fish
            on = np.flatnonzero((sim.fish == fm) & np.isfinite(ident) & (ident != ia))
            target = _mode(ident, on) if len(on) else None
            if target is not None:
                ident[move] = target
                return True
        ident[move] = state["next"]
        state["next"] += 1
        return True
    if g in ("missed_linker", "missed_rise", "rule_merge"):
        rx, ry = rows
        ix, iy = _mode(ident, rx), _mode(ident, ry)
        if ix is None or iy is None or ix == iy:
            return False
        fx, fy = sim.maj(rx), sim.maj(ry)
        if fx != fy or fx < 0:
            return False
        ident[ident == iy] = ix
        return True
    return False  # look-only items


REASSIGN = True  # a split piece goes to its fish's identity (--split-only: new id)


def review_curve(d: dict, items: list[dict], n_max: int) -> np.ndarray:
    sim = Sim(d)
    ident = d["ident"].copy()
    state = {
        "next": float(np.nanmax(ident)) + 1 if np.isfinite(ident).any() else 0.0,
        "reassign": REASSIGN,
    }
    curve = np.empty(n_max + 1)
    curve[0] = sim.idf1(ident)
    for n, it in enumerate(items[:n_max], start=1):
        curve[n] = sim.idf1(ident) if fix(sim, it, ident, state) else curve[n - 1]
    curve[len(items[:n_max]) + 1 :] = curve[len(items[:n_max])]
    return curve


def orders(d: dict, n_random: int = 5) -> dict[str, list[list[dict]]]:
    items = d["items"]
    conf = np.array([it["conf"] for it in items])
    raw = np.array([it["raw"] for it in items])
    t = np.array([it["t"] for it in items])
    impact = np.array([it["impact"] for it in items], float)
    label = np.array([it["label"] for it in items])
    pick = lambda o: [[items[i] for i in o]]  # noqa: E731
    rng = np.random.default_rng(0)
    out = {
        "confidence": pick(np.argsort(conf, kind="stable")),
        "expected gain": pick(np.argsort(-(1 - conf) * impact, kind="stable")),
        "raw score": pick(np.argsort(raw, kind="stable")),
        "time": pick(np.argsort(t, kind="stable")),
        "random": [
            [items[i] for i in rng.permutation(len(items))] for _ in range(n_random)
        ],
        "oracle": pick(np.lexsort((-impact, label))),
    }
    out["rule-based issues"] = [rule_items(d)]
    return out


_HM = None


def _claudian_harmonics():
    """The eodsorter's harmonic finder (claudian, read only)."""
    global _HM
    if _HM is None:
        import importlib.util

        path = (
            Path.home() / "wrk/tools/claudian/src/audian_plugins/eodsorter/harmonics.py"
        )
        spec = importlib.util.spec_from_file_location("eodsorter_harmonics", path)
        _HM = importlib.util.module_from_spec(spec)
        sys.modules["eodsorter_harmonics"] = _HM
        spec.loader.exec_module(_HM)
    return _HM


def rule_items(d: dict) -> list[dict]:
    """The eodsorter's rule-based issues (model.TrackSet._issues) on the
    result, in time order, as simulated items: join (merge), gap (look; a
    split if the two sides are two fish), short (< 10 points: delete),
    crossing (look only), harmonic (delete)."""
    if "_rules" in d:
        return d["_rules"]
    ident, fund, idx, times = d["ident"], d["fund"].astype(float), d["idx"], d["times"]
    tc = d.get("tracking_config") or {}
    max_dt = float(tc.get("max_dt") or 10.0)
    tol = float(tc.get("freq_tolerance") or 2.5)
    has = np.flatnonzero(np.isfinite(ident))
    order = has[np.lexsort((idx[has], ident[has]))]
    ids, first = np.unique(ident[order], return_index=True)
    bounds = np.append(first, len(order))
    rows = {
        float(i): order[a:b]
        for i, a, b in zip(ids, bounds[:-1], bounds[1:], strict=True)
    }
    t = times[idx]
    out = []
    st_first = np.array([t[rows[i][0]] for i in rows])
    st_last = np.array([t[rows[i][-1]] for i in rows])
    k_first = np.array([idx[rows[i][0]] for i in rows])
    k_last = np.array([idx[rows[i][-1]] for i in rows])
    f_start = np.array([fund[rows[i][0]] for i in rows])
    f_end = np.array([fund[rows[i][-1]] for i in rows])
    keys = list(rows)
    o = np.argsort(st_first, kind="stable")
    ts = st_first[o]
    for a in range(len(keys)):  # join
        lo = np.searchsorted(ts, st_last[a], "right")
        hi = np.searchsorted(ts, st_last[a] + max_dt, "right")
        cand = o[lo:hi]
        cand = cand[k_first[cand] > k_last[a]]
        if not len(cand):
            continue
        df = np.abs(f_start[cand] - f_end[a])
        ok = df <= tol
        if not ok.any():
            continue
        b = cand[ok][np.argmin(df[ok])]
        out.append(
            {
                "group": "rule_merge",
                "t": float(st_last[a]),
                "ids": (keys[a], keys[b]),
                "rows": (rows[keys[a]], rows[keys[b]]),
            }
        )
    for i in keys:  # gap
        r = rows[i]
        gaps = np.flatnonzero(np.diff(t[r]) > 0.5)
        for h in gaps:
            out.append(
                {
                    "group": "rule_split",
                    "t": float(t[r[h]]),
                    "ids": (i,),
                    "rows": (r[: h + 1], r[h + 1 :]),
                }
            )
        if len(r) < 10:  # short
            out.append(
                {"group": "rule_track", "t": float(t[r[0]]), "ids": (i,), "rows": (r,)}
            )
    # crossings: two identities within a frequency bin in one frame (look only)
    bin_hz = float(d.get("freq_resolution") or 0.61)
    k, f = idx[has], fund[has]
    oo = np.lexsort((f, k))
    rr, kk, ff = has[oo], k[oo], f[oo]
    ii = ident[rr]
    hit = np.flatnonzero((kk[1:] == kk[:-1]) & (np.abs(ff[1:] - ff[:-1]) <= bin_hz))
    if len(hit):
        lo_, hi_ = np.minimum(ii[hit], ii[hit + 1]), np.maximum(ii[hit], ii[hit + 1])
        kh = kk[hit]
        o2 = np.lexsort((kh, hi_, lo_))
        lo_, hi_, kh = lo_[o2], hi_[o2], kh[o2]
        new = np.r_[
            True, (lo_[1:] != lo_[:-1]) | (hi_[1:] != hi_[:-1]) | (np.diff(kh) > 30)
        ]
        for j in np.flatnonzero(new):
            out.append(
                {
                    "group": "rule_look",
                    "t": float(times[kh[j]]),
                    "ids": (lo_[j], hi_[j]),
                    "rows": (),
                }
            )
    HM = _claudian_harmonics()  # harmonic
    for x in HM.find_harmonics(fund, idx.astype(np.int64), ident, times, bin_hz):
        hid = float(x.harmonic_id)
        out.append(
            {
                "group": "rule_track",
                "t": float(x.t0),
                "ids": (hid,),
                "rows": (rows[hid],),
            }
        )
    out.sort(key=lambda it: it["t"])
    d["_rules"] = out
    return out


def _simulate_one(args):
    d, n_max, reassign = args
    global REASSIGN
    REASSIGN = reassign
    curves = {}
    counts = {}
    for name, lists in orders(d).items():
        cs = [review_curve(d, lst, n_max) for lst in lists]
        curves[name] = np.mean(cs, 0)
        counts[name] = len(lists[0])
    return d["split"] + "/" + d["name"].split("/", 1)[1], curves, counts


def simulate(data: list[dict], n_max: int, jobs: int, reassign: bool = True):
    import multiprocessing as mp

    with mp.get_context("fork").Pool(jobs) as pool:
        res = pool.map(_simulate_one, [(d, n_max, reassign) for d in data], chunksize=1)
    names = [r[0] for r in res]
    curves = {k: np.stack([r[1][k] for r in res]) for k in res[0][1]}
    counts = {k: np.array([r[2][k] for r in res]) for k in res[0][2]}
    return names, curves, counts


def items_to(curves: np.ndarray, gap: np.ndarray, frac: float) -> np.ndarray:
    """Per mixture: items until the gain reaches `frac` of the reachable gap
    (NaN where the gap is ~0 or never reached)."""
    gain = curves - curves[:, :1]
    out = np.full(len(curves), np.nan)
    for m in range(len(curves)):
        if gap[m] <= 1e-4:
            continue
        hit = np.flatnonzero(gain[m] >= frac * gap[m] - 1e-12)
        if len(hit):
            out[m] = hit[0]
    return out


def recalibrate_oof(data: list[dict], folds: int = 5, seed: int = 0) -> None:
    """Replace every item's confidence by an isotonic map of it, per kind,
    fitted on the other folds of `data` (folds by mixture, lab and field
    together): an honest (out-of-fold) recalibration on dev."""
    units = sorted({_unit(d["split"], d["name"]) for d in data})
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(units, rng.permutation(len(units)) % folds, strict=True))
    for kind in ("join", "track", "missed_join"):
        rows = [
            (fold_of[_unit(d["split"], d["name"])], it)
            for d in data
            for it in d["items"]
            if kind_of(it["group"]) == kind
        ]
        f = np.array([r[0] for r in rows])
        p = np.array([r[1]["conf"] for r in rows])
        y = np.array([r[1]["label"] for r in rows], float)
        new = p.copy()
        for k in range(folds):
            tr, te = f != k, f == k
            if te.any() and tr.sum() > 10:
                kx, ky = isotonic(p[tr], y[tr])
                new[te] = np.interp(p[te], kx, ky)
        for (_, it), c in zip(rows, new, strict=True):
            it["conf"] = float(c)


def _item_arrays(data: list[dict]) -> dict:
    """Per item of every kind: confidence, raw score, label (for figures)."""
    out = {}
    for kind in ("join", "track", "missed_join"):
        sel = [it for d in data for it in d["items"] if kind_of(it["group"]) == kind]
        out[f"items/{kind}/conf"] = np.array([it["conf"] for it in sel])
        out[f"items/{kind}/raw"] = np.array([it["raw"] for it in sel])
        out[f"items/{kind}/label"] = np.array([it["label"] for it in sel], np.int8)
    return out


def evaluate(args):
    from wavetracker.review import ReviewModel

    model = ReviewModel.load(args.model) if args.model else None
    dev = load_split(args.tag, args.splits)
    train = load_split(args.tag, TRAIN) if args.train else []
    model_conf(dev, model)
    model_conf(train, model)
    if args.recalibrate:
        recalibrate_oof(dev, seed=args.seed)
    out = {
        "tag": args.tag,
        "model": args.model,
        "splits": args.splits,
        "mixtures": len(dev),
    }
    out["calibration_dev"] = calibration_table(dev)
    out["raw_dev"] = calibration_table(dev, "raw")
    if train:
        out["calibration_train"] = calibration_table(train)
    for name, tab in (
        ("model, dev", out["calibration_dev"]),
        ("raw, dev", out["raw_dev"]),
    ):
        print(name)
        for k, v in tab.items():
            print(
                f"  {k:14s} n {v['n']:6d} wrong {v['wrong']:5d} right {v['rate_right']:.3f} "
                f"conf {v['mean_conf']:.3f} logloss {v['logloss']:.3f} ece {v['ece']:.3f} "
                f"auc {v['auc']:.3f}"
            )
    names, curves, counts = simulate(dev, args.n_max, args.jobs, not args.split_only)
    gap = curves["confidence"][:, -1] - curves["confidence"][:, 0]
    out["review"] = {"names": names}
    hours = len(names) * 600 / 3600  # 10-min mixtures
    summ = {}
    for k, c in curves.items():
        gain = c - c[:, :1]
        n50, n90 = items_to(c, gap, 0.5), items_to(c, gap, 0.9)
        mean_gain = gain.mean(0)
        g_all = gap.mean()

        def first(fr, mean_gain=mean_gain, g_all=g_all):
            hit = np.flatnonzero(mean_gain >= fr * g_all)
            return int(hit[0]) if len(hit) else None

        summ[k] = {
            "items": float(counts[k].mean()),
            "idf1_0": float(c[:, 0].mean()),
            "gain_at": {
                n: float(gain[:, n].mean())
                for n in (5, 10, 20, 50, 100)
                if n <= args.n_max
            },
            "gain_end": float(gain[:, -1].mean()),
            "n50_median": float(np.nanmedian(n50)),
            "n90_median": float(np.nanmedian(n90)),
            "n50_mean": float(np.nanmean(n50)),
            "n90_mean": float(np.nanmean(n90)),
            "n50_reached": int(np.isfinite(n50).sum()),
            "n90_reached": int(np.isfinite(n90).sum()),
            "mean_curve_n50": first(0.5),
            "mean_curve_n90": first(0.9),
        }
    summ["_gap"] = {
        "mean": float(gap.mean()),
        "median": float(np.median(gap)),
        "zero": int((gap <= 1e-4).sum()),
        "hours": hours,
    }
    out["review"]["summary"] = summ
    print(
        f"reachable gap: mean {gap.mean():.4f}, median {np.median(gap):.4f}, "
        f"{(gap <= 1e-4).sum()} mixtures without"
    )
    for k, v in summ.items():
        if k.startswith("_"):
            continue
        print(
            f"  {k:18s} items {v['items']:6.1f} gain@10 {v['gain_at'].get(10, np.nan):.4f} "
            f"@20 {v['gain_at'].get(20, np.nan):.4f} @50 {v['gain_at'].get(50, np.nan):.4f} "
            f"end {v['gain_end']:.4f}  n50 {v['n50_median']:.0f} n90 {v['n90_median']:.0f} "
            f"(mean curve {v['mean_curve_n50']} / {v['mean_curve_n90']})"
        )
    stem = OUT / args.tag / f"eval-{args.name}"
    np.savez_compressed(
        f"{stem}.npz",
        names=np.array(names),
        gap=gap,
        **{f"curve/{k}": v for k, v in curves.items()},
        **{f"count/{k}": v for k, v in counts.items()},
        **_item_arrays(dev),
    )
    Path(f"{stem}.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {stem}.json / .npz")


# --- figures ----------------------------------------------------------------------

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
GRAY = "#9b9a96"
KIND_TITLE = {
    "join": "joins",
    "track": "identities (is it a fish?)",
    "missed_join": "missed joins",
}
ORDER_STYLE = {
    "confidence": (SERIES[0], "-", 2.0),
    "expected gain": (SERIES[1], "-", 1.6),
    "raw score": (SERIES[2], "-", 1.6),
    "rule-based issues": (SERIES[3], "-", 1.6),
    "time": (GRAY, "--", 1.4),
    "random": (GRAY, ":", 1.6),
    "oracle": (INK, (0, (1, 2)), 1.0),
}


def _qbins(p, y, n):
    """Equal-count bins of p: mean p, mean y and count per bin."""
    o = np.argsort(p, kind="stable")
    parts = np.array_split(o, min(n, max(len(p), 1)))
    parts = [q for q in parts if len(q)]
    return (
        np.array([p[q].mean() for q in parts]),
        np.array([y[q].mean() for q in parts]),
        np.array([len(q) for q in parts]),
    )


def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def plot(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    runs = [r.split("=", 1) for r in args.runs]  # label=tag/eval-name
    # reliability: predicted against observed error rate (1 - confidence,
    # 1 - fraction right), equal-count bins, log axes: errors are rare
    fig, axes = plt.subplots(len(runs), 3, figsize=(10, 3.4 * len(runs)), squeeze=False)
    lo = 3e-3
    for row, (label, path) in enumerate(runs):
        ev = json.loads((OUT / f"{path}.json").read_text())
        z = np.load(OUT / f"{path}.npz")
        oof = (
            np.load(Path(ev["model"]).with_suffix(".oof.npz")) if ev["model"] else None
        )
        for col, kind in enumerate(("join", "track", "missed_join")):
            ax = axes[row, col]
            _style(ax)
            ax.plot([lo, 1], [lo, 1], color=MUTED, lw=0.8, ls="--")
            y = z[f"items/{kind}/label"]
            series = [("dev, model", z[f"items/{kind}/conf"], y, SERIES[0], "o")]
            if oof is not None:
                keys = [
                    k
                    for k in oof.files
                    if k.startswith(f"{kind}/") and k.endswith("/p")
                ]
                if keys:
                    p = np.concatenate([oof[k] for k in keys])
                    yy = np.concatenate([oof[k[:-1] + "y"] for k in keys])
                    series.append(("ltrain, out of fold", p, yy, SERIES[1], "s"))
            series.append(("dev, raw score", z[f"items/{kind}/raw"], y, GRAY, "^"))
            for name, conf, lab, color, marker in series:
                err_p, err_o, n = _qbins(1 - conf, 1 - lab, args.bins)
                se = np.sqrt(np.maximum(err_o * (1 - err_o), 1e-6) / n)
                ax.errorbar(
                    np.maximum(err_p, lo),
                    np.maximum(err_o, lo),
                    yerr=[np.minimum(se, np.maximum(err_o, lo) - lo / 2), se],
                    color=color,
                    marker=marker,
                    ms=4,
                    lw=1.2,
                    elinewidth=0.7,
                    capsize=0,
                    mec="white",
                    mew=0.5,
                    label=f"{name}: ECE {ece(conf, lab):.3f}, AUC {auc(conf, lab):.2f}",
                )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(lo, 1)
            ax.set_ylim(lo, 1)
            ax.set_title(
                f"{label}: {KIND_TITLE[kind]} (n {len(y)})",
                fontsize=9,
                color=INK,
                loc="left",
            )
            ax.set_xlabel(
                "predicted P(wrong) = 1 - confidence", fontsize=8, color=MUTED
            )
            if col == 0:
                ax.set_ylabel("observed fraction wrong", fontsize=8, color=MUTED)
            ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.tight_layout()
    out1 = FIG / f"reliability_{args.name}.png"
    fig.savefig(out1, dpi=160)
    plt.close(fig)
    # review efficiency
    fig, axes = plt.subplots(len(runs), 2, figsize=(10, 3.6 * len(runs)), squeeze=False)
    for row, (label, path) in enumerate(runs):
        z = np.load(OUT / f"{path}.npz")
        ev = json.loads((OUT / f"{path}.json").read_text())
        gap = z["gap"]
        has = gap > 1e-4
        n = np.arange(args.n_plot + 1)
        for col in range(2):
            ax = axes[row, col]
            _style(ax)
            for name, (color, ls, lw) in ORDER_STYLE.items():
                key = f"curve/{name}"
                if key not in z.files:
                    continue
                c = z[key][:, : args.n_plot + 1]
                gain = c - c[:, :1]
                y = gain.mean(0) if col == 0 else (gain[has] / gap[has, None]).mean(0)
                ax.plot(n, y, color=color, ls=ls, lw=lw, label=name)
            if col == 0:
                ax.axhline(gap.mean(), color=MUTED, lw=0.6)
                ax.set_ylabel("IDF1 gain per mixture", fontsize=8, color=MUTED)
                s = ev["review"]["summary"]["confidence"]
                ax.set_title(
                    f"{label}: mean IDF1 {s['idf1_0']:.3f}, reachable +{gap.mean():.4f}",
                    fontsize=9,
                    color=INK,
                    loc="left",
                )
            else:
                ax.axhline(1.0, color=MUTED, lw=0.6)
                for fr in (0.5, 0.9):
                    ax.axhline(fr, color=GRID, lw=0.8, zorder=0)
                ax.set_ylabel("fraction of the reachable gap", fontsize=8, color=MUTED)
                ax.set_title(
                    f"{label}: {int(has.sum())} mixtures with a gap",
                    fontsize=9,
                    color=INK,
                    loc="left",
                )
            ax.set_xlim(0, args.n_plot)
            ax.set_xlabel("items reviewed per 10-min mixture", fontsize=8, color=MUTED)
            sec = ax.secondary_xaxis(
                "top", functions=(lambda x: 6 * x, lambda x: x / 6)
            )
            sec.set_xlabel("per recording hour", fontsize=7, color=MUTED)
            sec.tick_params(colors=MUTED, labelsize=7)
            if row == 0 and col == 0:
                ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.tight_layout()
    out2 = FIG / f"review_efficiency_{args.name}.png"
    fig.savefig(out2, dpi=160)
    plt.close(fig)
    print(out1)
    print(out2)


# --- main ----------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--tag", required=True)
    c.add_argument("-c", "--config", required=True)
    c.add_argument("--source", required=True, help="tag or tag:split,...")
    c.add_argument("--check", help="runs whose identities must be reproduced")
    c.add_argument("--splits", nargs="+", required=True)
    c.add_argument("--only", nargs="*")
    c.add_argument("--jobs", type=int, default=1)
    c.add_argument("--overwrite", action="store_true")
    c.set_defaults(func=collect)
    f = sub.add_parser("fit")
    f.add_argument("--tag", required=True)
    f.add_argument("--splits", nargs="+", default=list(TRAIN))
    f.add_argument("--out", required=True)
    f.add_argument(
        "--method", choices=list(CANDIDATES), help="default: best cv logloss"
    )
    f.add_argument("--isotonic", action="store_true", help="isotonic map on top")
    f.add_argument("--seed", type=int, default=0)
    f.set_defaults(func=fit)
    e = sub.add_parser("evaluate")
    e.add_argument("--tag", required=True)
    e.add_argument("--model", help="ReviewModel .npz (default: raw scores)")
    e.add_argument("--name", default="model")
    e.add_argument("--splits", nargs="+", default=list(DEV))
    e.add_argument("--train", action="store_true", help="also calibration on ltrain")
    e.add_argument("--n-max", type=int, default=300)
    e.add_argument("--jobs", type=int, default=16)
    e.add_argument(
        "--split-only", action="store_true", help="a split piece gets a new id"
    )
    e.add_argument(
        "--recalibrate", action="store_true", help="isotonic, out of fold on the splits"
    )
    e.add_argument("--seed", type=int, default=0)
    e.set_defaults(func=evaluate)
    pl = sub.add_parser("plot")
    pl.add_argument("--runs", nargs="+", required=True, help="label=tag/eval-name")
    pl.add_argument("--name", required=True)
    pl.add_argument("--n-plot", type=int, default=100)
    pl.add_argument(
        "--bins", type=int, default=10, help="reliability: equal-count bins"
    )
    pl.set_defaults(func=plot)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
