"""Score wavetracker results on a dense mixture against its ground truth.

Detections are matched one to one per frame to the fish of the truth
(closest first, within ``TOL`` Hz). Unmatched detections are *ignored* where
the truth of a fish is unknown (a gap in the source's own tracking) and they
lie near its interpolated trace, *harmonic* when they sit on a harmonic or
subharmonic of a fish, and *false* otherwise.

Per fish:

* recall: frames with a matched detection / frames where the fish is known;
* coverage: frames matched by its dominant identity / known frames;
* switches: changes of identity along its matched detections;
* fragments: identities needed for 90 % of its matched detections.

Overall: precision, MOTA = 1 - (misses + false + switches) / truth frames,
and IDF1 (identities assigned one to one to fish to maximise matched
detections; Ristani et al. 2016), which is what a user sees: one identity
per fish, as long as possible.

Every (fish, frame) is also written with its covariates (SNR, gain,
frequency, distance to the nearest line of another fish), from which recall
and coverage curves are drawn.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from wavetracker.results import Results

TOL = 2.0  # Hz
GAP_TOL = 30.0  # Hz around an interpolated trace where the truth is unknown
MAX_H = 5
VISIBLE_DB = 6.0  # a fish frame with this SNR in the mixture is visible


def truth_at(truth: dict, times: np.ndarray, key: str = "freqs") -> np.ndarray:
    """Truth (fish, frames) at result frame times: nearest truth sample."""
    tt = truth["times"]
    j = np.clip(np.round((times - tt[0]) / (tt[1] - tt[0])).astype(int), 0, len(tt) - 1)
    out = truth[key][:, j]
    outside = (times < tt[0]) | (times > tt[-1] + (tt[1] - tt[0]))
    out[:, outside] = np.nan
    return out


def match(res: Results, F: np.ndarray, tol: float = TOL) -> np.ndarray:
    """Fish of every detection (-1: none), one to one per frame."""
    fish = np.full(len(res.fund_v), -1)
    ref = F[:, res.idx_v]  # (fish, det)
    err = np.abs(ref - res.fund_v[None])
    k, d = np.nonzero(err <= tol)
    order = np.argsort(err[k, d], kind="stable")
    k, d = k[order], d[order]
    taken = set()
    frames = res.idx_v[d]
    for kk, dd, fr in zip(k, d, frames, strict=True):
        if fish[dd] >= 0 or (kk, fr) in taken:
            continue
        fish[dd] = kk
        taken.add((kk, fr))
    return fish


def classify_unmatched(res, F, F_filled, fish, tol=TOL) -> np.ndarray:
    """0 matched, 1 ignored (truth unknown there), 2 harmonic, 3 false."""
    cls = np.where(fish >= 0, 0, 3)
    un = np.flatnonzero(fish < 0)
    f = res.fund_v[un]
    idx = res.idx_v[un]
    gap = np.isnan(F[:, idx]) & np.isfinite(F_filled[:, idx])
    near_gap = (np.abs(F_filled[:, idx] - f[None]) <= GAP_TOL) & gap
    cls[un[near_gap.any(0)]] = 1
    Fv = F_filled[:, idx]
    harm = np.zeros(len(un), bool)
    for h in range(2, MAX_H + 1):
        harm |= (np.abs(h * Fv - f[None]) <= tol * h).any(0)  # harmonic
        harm |= (np.abs(Fv / h - f[None]) <= tol).any(0)  # subharmonic
    sel = (cls[un] == 3) & harm
    cls[un[sel]] = 2
    return cls


def nearest_line(F: np.ndarray) -> np.ndarray:
    """Distance [Hz] of each fish (fish, frames) to the closest line of any
    other fish: one of its harmonics (1..MAX_H), or a fish on one of our own
    harmonics (that distance divided by the harmonic number)."""
    n = len(F)
    out = np.full(F.shape, np.inf)
    for k in range(n):
        for j in range(n):
            if j == k:
                continue
            for h in range(1, MAX_H + 1):
                out[k] = np.fmin(out[k], np.abs(F[k] - h * F[j]))
                out[k] = np.fmin(out[k], np.abs(h * F[k] - F[j]) / h)
    return out


def score(mix_dir: Path, res_dir: Path, tol: float = TOL) -> tuple[dict, dict]:
    truth = dict(np.load(mix_dir / "truth.npz"))
    meta = json.loads((mix_dir / "mixture.json").read_text())
    res = Results.load(res_dir)
    F = truth_at(truth, res.times)
    Ff = truth_at(truth, res.times, "freqs_filled")
    n_fish, n_frames = F.shape
    fish = match(res, F, tol)
    cls = classify_unmatched(res, F, Ff, fish, tol)
    known = np.isfinite(F)
    n_known = known.sum(1)
    ident = res.ident_v
    has_id = np.isfinite(ident)
    ids = np.unique(ident[has_id])
    id_pos = {i: p for p, i in enumerate(ids)}
    counts = np.zeros((len(ids), n_fish), int)
    m = (fish >= 0) & has_id
    np.add.at(counts, (np.searchsorted(ids, ident[m]), fish[m]), 1)

    # identities one to one to fish (IDF1)
    if len(ids):
        r, c = linear_sum_assignment(-counts)
        idtp = int(counts[r, c].sum())
    else:
        r = c = np.array([], int)
        idtp = 0
    id_of_fish = {
        int(cc): float(ids[rr])
        for rr, cc in zip(r, c, strict=True)
        if counts[rr, cc] > 0
    }
    n_pred = int((has_id & (cls != 1)).sum())
    idfn = int(n_known.sum()) - idtp
    idfp = n_pred - idtp
    idf1 = 2 * idtp / max(2 * idtp + idfp + idfn, 1)

    per_fish = []
    matched_frame = np.zeros((n_fish, n_frames), bool)
    dominant_frame = np.zeros((n_fish, n_frames), bool)
    assigned_frame = np.zeros((n_fish, n_frames), bool)  # IDF1 identity
    switches_total = 0
    for k in range(n_fish):
        mk = fish == k
        fr = res.idx_v[mk]
        matched_frame[k, fr] = True
        col = counts[:, k]
        top = ids[col.argmax()] if len(ids) and col.max() > 0 else np.nan
        dominant_frame[k, res.idx_v[mk & (ident == top)]] = True
        if k in id_of_fish:
            assigned_frame[k, res.idx_v[mk & (ident == id_of_fish[k])]] = True
        order = np.argsort(fr)
        seq = ident[mk][order]
        seq = seq[np.isfinite(seq)]
        sw = int((seq[1:] != seq[:-1]).sum()) if len(seq) > 1 else 0
        switches_total += sw
        c = np.sort(col)[::-1]
        frag = int(np.searchsorted(np.cumsum(c), 0.9 * c.sum()) + 1) if c.sum() else 0
        nk = max(int(n_known[k]), 1)
        per_fish.append(
            {
                "fish": k,
                "source": int(truth["source"][k]),
                "median_hz": float(np.nanmedian(F[k])),
                "known_frames": int(n_known[k]),
                "recall": float(matched_frame[k][known[k]].mean())
                if n_known[k]
                else np.nan,
                "coverage": float(dominant_frame[k][known[k]].sum() / nk),
                "idf1_coverage": float(assigned_frame[k][known[k]].sum() / nk),
                "switches": sw,
                "fragments": frag,
                "median_snr_db": float(np.nanmedian(truth["snr_db"][k])),
                "median_gain_db": float(np.median(truth["gain_db"][k])),
            }
        )

    n_det = len(fish)
    n_false = int((cls == 3).sum())
    n_harm = int((cls == 2).sum())
    n_match = int((cls == 0).sum())
    gt = int(n_known.sum())
    misses = gt - int(matched_frame[known].sum())
    dt = res.times[1] - res.times[0]
    long_ids = [i for i in ids if (ident == i).sum() * dt >= 30.0]
    false_ids = [
        i
        for i in long_ids
        if counts[id_pos[i]].sum() < 0.5 * ((ident == i) & (cls != 1)).sum()
    ]
    snr = np.stack(
        [
            np.interp(res.times, truth["snr_times"], s, left=np.nan, right=np.nan)
            for s in truth["snr_db"]
        ]
    )
    visible = known & (snr >= VISIBLE_DB)
    summary = {
        "mixture": meta["name"],
        "n_fish": n_fish,
        "frames": n_frames,
        "recall": float(matched_frame[known].mean()),
        "recall_visible": float(matched_frame[visible].mean())
        if visible.any()
        else np.nan,
        "visible": float(visible.sum() / max(known.sum(), 1)),
        "coverage": float(dominant_frame[known].mean()),
        "precision": n_match / max(n_det - int((cls == 1).sum()), 1),
        "harmonic_per_frame": n_harm / n_frames,
        "false_per_frame": n_false / n_frames,
        "idf1": idf1,
        "mota": 1 - (misses + n_false + n_harm + switches_total) / max(gt, 1),
        "switches_per_fish_min": switches_total / n_fish / (n_frames * dt / 60),
        "ids": len(ids),
        "ids_30s": len(long_ids),
        "false_ids_30s": len(false_ids),
        "purity": float(counts.max(1).sum() / max(counts.sum(), 1)),
        "fish": per_fish,
    }

    # per (fish, frame) table with covariates
    gain = np.stack([np.interp(res.times, truth["times"], g) for g in truth["gain_db"]])
    nn = nearest_line(Ff)
    kk, jj = np.nonzero(known)
    table = {
        "fish": kk.astype(np.int16),
        "frame": jj.astype(np.int32),
        "freq": F[kk, jj].astype(np.float32),
        "snr_db": snr[kk, jj].astype(np.float32),
        "gain_db": gain[kk, jj].astype(np.float32),
        "nn_hz": nn[kk, jj].astype(np.float32),
        "matched": matched_frame[kk, jj],
        "dominant": dominant_frame[kk, jj],
        "assigned": assigned_frame[kk, jj],
        "n_fish": np.full(len(kk), n_fish, np.int16),
    }
    return summary, table


def binned(x, y, edges):
    """Mean of y and count in bins of x."""
    i = np.digitize(x, edges) - 1
    ok = (i >= 0) & (i < len(edges) - 1) & np.isfinite(x)
    n = np.bincount(i[ok], minlength=len(edges) - 1)
    s = np.bincount(i[ok], weights=y[ok].astype(float), minlength=len(edges) - 1)
    with np.errstate(invalid="ignore"):
        return s / n, n
