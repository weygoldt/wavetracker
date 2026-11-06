"""PROTOTYPE (not in wavetracker yet): track-guided acceptance of faint peaks.

pool: detections of a second pass at weaker thresholds (single peaks allowed)
that are not detections of the base run. Every identity of the base run is
extended frame by frame from each of its edges (into gaps and beyond its
ends) through pool candidates near the predicted frequency; it stops after
`max_miss` frames without one. A candidate within `conflict_hz` of another
identity's detection in that frame is not taken.

    python benchmarks/dense/faint_peaks.py <base tag> <out tag> <split> [key=value ...]
    python benchmarks/dense/faint_peaks.py dense-linked proto-fill-v1a dev tol=0.4 max_miss=2

Candidates are only taken after linking (filling the tracker's fragments
before linking gained nothing, pre=1), and not on a harmonic or subharmonic
of another identity in that frame (harm=1). Prints what the added
detections are (matched, harmonic, false) next to the scores.
"""

import json
import sys
from pathlib import Path

import numpy as np
import yaml

WT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WT / "benchmarks/dense"))
from mixture import MIXTURES  # noqa: E402
from run import RUNS, merge, mixtures  # noqa: E402
from score import classify_unmatched, match, score, truth_at  # noqa: E402

from wavetracker.config import Config  # noqa: E402
from wavetracker.pipeline import detect  # noqa: E402
from wavetracker.results import Results  # noqa: E402

base_tag, out_tag, split = sys.argv[1:4]
P = dict(
    low=3.0,
    high=4.0,
    group=1,
    tol=0.8,  # Hz per frame step around the prediction
    max_miss=3,  # frames without a candidate before stopping
    max_ext=30.0,  # s beyond an edge
    conflict_hz=1.5,
    slope_n=8,  # detections for the edge slope
    harm=1,  # reject candidates on harmonics (2..5) / subharmonics of other ids
    harm_tol=1.0,  # Hz (times h)
    pattern_db=0.0,  # > 0: max rms dB difference to the edge pattern
    cfg="dense-linked",
    pre=0,  # 1: fill the tracker's fragments (base_tag's track cache), then link
)
for kv in sys.argv[4:]:
    k, v = kv.split("=")
    P[k] = type(P[k])(v)
POOL = RUNS.parent / "proto-pool" / f"{P['cfg']}-l{P['low']}-h{P['high']}-g{P['group']}"


def pool_for(name, base):
    d = POOL / name
    if not (d / "fund_v.npy").exists():
        variant = yaml.safe_load(
            (WT / f"benchmarks/dense/configs/{P['cfg']}.yaml").read_text()
        )
        band = yaml.safe_load((MIXTURES / name / "band.yaml").read_text())
        cfg = Config.from_dict(merge(variant, band))
        cfg.harmonic_groups.low_thresh_factor = P["low"]
        cfg.harmonic_groups.high_thresh_factor = P["high"]
        cfg.harmonic_groups.min_group_size = P["group"]
        detect(MIXTURES / name, d, cfg, device="auto").results.save(d)
    w = Results.load(d)
    # drop what the base run detected (same frame, within 0.5 Hz)
    key_b = {}
    for f, i in zip(base.fund_v, base.idx_v, strict=True):
        key_b.setdefault(int(i), []).append(f)
    keep = np.ones(len(w.fund_v), bool)
    for j, (f, i) in enumerate(zip(w.fund_v, w.idx_v, strict=True)):
        fb = key_b.get(int(i))
        if fb is not None and np.min(np.abs(np.asarray(fb) - f)) < 0.5:
            keep[j] = False
    return w, keep


def _pat(p):
    db = 10 * np.log10(np.maximum(np.asarray(p, float), 1e-30))
    return db - db.mean(-1, keepdims=True)


def fill(base, w, keep):
    n_frames = len(base.times)
    dt = base.times[1] - base.times[0]
    # pool by frame
    order = np.argsort(w.idx_v[keep], kind="stable")
    pidx = np.flatnonzero(keep)[order]
    pf = w.idx_v[pidx]
    starts = np.searchsorted(pf, np.arange(n_frames + 1))
    taken = np.zeros(len(w.fund_v), bool)
    # occupancy of other identities per frame
    ident = base.ident_v
    ok = np.isfinite(ident)
    occ = {}
    for f, i, d in zip(base.fund_v[ok], base.idx_v[ok], ident[ok], strict=True):
        occ.setdefault(int(i), []).append((f, d))
    new_f, new_i, new_d, new_src = [], [], [], []
    ids, counts = np.unique(ident[ok], return_counts=True)
    for d in ids[np.argsort(-counts)]:
        sel = np.flatnonzero(ident == d)
        fr = base.idx_v[sel]
        o = np.argsort(fr)
        fr, ff = fr[o], base.fund_v[sel][o]
        have = set(fr.tolist())
        for direction in (+1, -1):
            # every edge: a detection whose neighbour frame in `direction` is empty
            edges = [k for k in range(len(fr)) if (fr[k] + direction) not in have]
            for k in edges:
                # slope from the last slope_n detections towards the edge
                lo, hi = (
                    (max(0, k - P["slope_n"] + 1), k + 1)
                    if direction > 0
                    else (k, min(len(fr), k + P["slope_n"]))
                )
                if hi - lo >= 3:
                    sl = np.polyfit(fr[lo:hi], ff[lo:hi], 1)[0]
                    sl = float(np.clip(sl, -2.0, 2.0))
                else:
                    sl = 0.0
                f0, frame, miss = ff[k], fr[k], 0
                if P["pattern_db"] > 0:
                    lo2, hi2 = (
                        (max(0, k - 10), k + 1)
                        if direction > 0
                        else (k, min(len(fr), k + 11))
                    )
                    ref = _pat(base.sign_v[sel][o][lo2:hi2].mean(0))
                n_ext = 0
                while miss < P["max_miss"] and n_ext * dt < P["max_ext"]:
                    frame += direction
                    n_ext += 1
                    if frame < 0 or frame >= n_frames or frame in have:
                        break
                    pred = f0 + sl * direction * (miss + 1)
                    a, b = starts[frame], starts[frame + 1]
                    best, berr = -1, P["tol"] * (miss + 1)
                    for j in pidx[a:b]:
                        if taken[j]:
                            continue
                        e = abs(w.fund_v[j] - pred)
                        if e <= berr:
                            best, berr = j, e
                    if best >= 0:
                        fb = w.fund_v[best]
                        others = [fo for fo, do in occ.get(int(frame), []) if do != d]
                        if any(abs(fo - fb) < P["conflict_hz"] for fo in others) or (
                            P["harm"]
                            and any(
                                abs(h * fo - fb) < P["harm_tol"] * h
                                or abs(fo / h - fb) < P["harm_tol"]
                                for fo in others
                                for h in range(2, 6)
                            )
                        ):
                            best = -1
                        elif P["pattern_db"] > 0:
                            dp = _pat(w.sign_v[best]) - ref
                            if np.sqrt(np.mean(dp * dp)) > P["pattern_db"]:
                                best = -1
                    if best < 0:
                        miss += 1
                        continue
                    taken[best] = True
                    have.add(frame)
                    occ.setdefault(int(frame), []).append((w.fund_v[best], d))
                    new_f.append(w.fund_v[best])
                    new_i.append(frame)
                    new_d.append(d)
                    new_src.append(best)
                    f0, miss = w.fund_v[best], 0
    src = np.asarray(new_src, int)
    fund = np.concatenate([base.fund_v, np.asarray(new_f, float)])
    idx = np.concatenate([base.idx_v, np.asarray(new_i, np.int64)])
    sign = np.concatenate([base.sign_v, w.sign_v[src]]) if len(src) else base.sign_v
    cplx = np.concatenate([base.cplx_v, w.cplx_v[src]]) if len(src) else base.cplx_v
    idv = np.concatenate([base.ident_v, np.asarray(new_d, float)])
    o = np.argsort(idx, kind="stable")
    meta = dict(base.meta)
    meta["proto_fill"] = P | {"added": len(src)}
    r = Results(fund[o], idx[o], sign[o], idv[o], base.times, meta, cplx_v=cplx[o])
    is_new = np.r_[np.zeros(len(base.fund_v), bool), np.ones(len(src), bool)][o]
    return r, len(src), is_new


rows = []
for name in mixtures(split, None):
    base = Results.load(RUNS / base_tag / name)
    if P["pre"]:
        from wavetracker import linking
        from wavetracker.config import LinkingConfig

        base.ident_v = np.load(RUNS.parent / "linker" / base_tag / name / "track.npz")[
            "ident"
        ]
    w, keep = pool_for(name, base)
    r, n_add, is_new = fill(base, w, keep)
    if P["pre"]:
        r.ident_v = linking.link(
            r.fund_v, r.idx_v, r.sign_v, r.cplx_v, r.ident_v, r.times, LinkingConfig()
        )
    out = RUNS / out_tag / name
    r.save(out)
    summary, table = score(MIXTURES / name, out)
    b = json.loads((RUNS / base_tag / name / "score.json").read_text())
    summary |= {"tag": out_tag, "added": n_add}
    (out / "score.json").write_text(json.dumps(summary, indent=1))
    np.savez_compressed(out / "frames.npz", **table)
    truth = dict(np.load(MIXTURES / name / "truth.npz"))
    F = truth_at(truth, r.times)
    Ff = truth_at(truth, r.times, "freqs_filled")
    fish_r = match(r, F)
    cls = classify_unmatched(r, F, Ff, fish_r)
    c_new = cls[is_new]
    # added and matched to the fish of its identity's majority
    ok_id = np.isfinite(r.ident_v) & (fish_r >= 0)
    maj = {}
    for i_, f_ in zip(r.ident_v[ok_id & ~is_new], fish_r[ok_id & ~is_new], strict=True):
        maj.setdefault(i_, []).append(f_)
    maj = {i_: np.bincount(v).argmax() for i_, v in maj.items()}
    right = np.array(
        [
            fish_r[j] >= 0 and maj.get(r.ident_v[j], -2) == fish_r[j]
            for j in np.flatnonzero(is_new)
        ]
    )
    summary["added_cls"] = [int((c_new == c).sum()) for c in range(4)] + [
        int(right.sum()) if len(right) else 0
    ]
    rows.append((name, b, summary))
    print(
        f"{name}: +{n_add} recall {b['recall']:.3f}->{summary['recall']:.3f} "
        f"idf1 {b['idf1']:.3f}->{summary['idf1']:.3f} prec {b['precision']:.3f}->{summary['precision']:.3f} "
        f"purity {b['purity']:.3f}->{summary['purity']:.3f}",
        flush=True,
    )
tot = np.sum([x[2]["added_cls"] for x in rows], 0)
n = max(tot[:4].sum(), 1)
print(
    f"added: {tot[:4].sum()}: matched {tot[0] / n:.3f} (to its identity's fish {tot[4] / n:.3f}), ignored {tot[1] / n:.3f}, harmonic {tot[2] / n:.3f}, false {tot[3] / n:.3f}"
)
keys = [
    "recall",
    "recall_detected",
    "recall_completed",
    "idf1",
    "precision",
    "purity",
    "false_per_frame",
    "harmonic_per_frame",
]
print("mean over mixtures, base -> filled:")
for k in keys:
    a = np.nanmean([x[1].get(k, np.nan) for x in rows])
    c = np.nanmean([x[2].get(k, np.nan) for x in rows])
    print(f"  {k:20s} {a:.4f} -> {c:.4f}  ({c - a:+.4f})")
