"""Can one fish be told from two in one spectral peak? Cues measured on the
truth of the dev mixtures, before any network is built.

Two fundamentals closer than about 1.2 Hz make one peak in the 1.64 s
spectrogram. Over a few seconds two things still tell them apart:

* the **beat**: the power of the bin pulses at the frequency difference
  (summed spectrogram, what the detector sees today);
* the **spatial pattern**: the field of one fish is quasi-static, so all
  electrodes are in phase or in antiphase and the complex STFT values of
  the bin, over electrodes and frames, span one *real* direction. Two fish
  span two. Measured as the ratio of the two largest eigenvalues of the
  electrode covariance over the window, complex (``R = X^H X / W``) and real
  (``Re R``: also catches two fish with a constant phase difference), and
  per frame as the non-realness of the electrode vector (``lambda2 /
  (lambda1 + lambda2)`` of ``[Re x, Im x]``).

Samples are windows of ``--window`` seconds (non-overlapping) of every truth
pair closer than 1.2 Hz throughout the window (positives) and of every
single fish with no other line (fundamental or harmonic 2-5 of another fish)
within 3 Hz (negatives), at the bin of the pair's mean frequency (of the
fish). Fish pairs are split by their mean difference (< 0.6, 0.6-1.2 Hz) and
by the SNR of the weaker fish; negatives are weighted to the SNR of the
stronger fish of the positives (the peak's power), so a cue cannot score
by SNR alone.

    python benchmarks/dense/pair_cues.py measure --split ldev
    python benchmarks/dense/pair_cues.py measure --split ldev-field
    python benchmarks/dense/pair_cues.py report
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA, MIX_CHANNELS
from salience_train import DF, FT, NFFT, STEP, Mixture, mixture_dirs

from wavetracker.spectrogram import PowerSpectrogram

OUT = DATA / "salience" / "pair-cues"
FIG = DATA / "figures"
PAIR_HZ = 1.2
CLEAR_HZ = 3.0
MAX_H = 5
F_TOP = 2600.0  # bins kept per mixture (fish and their 2nd-3rd harmonics)
DF_BANDS = [(0.0, 0.6, "< 0.6 Hz"), (0.6, 1.2, "0.6-1.2 Hz")]
SNR_BANDS = [
    (-99, 6, "weaker < 6 dB"),
    (6, 15, "weaker 6-15 dB"),
    (15, 99, "weaker >= 15 dB"),
]


def stft_bins(mix: Mixture, gain: float, kmax: int, dev) -> torch.Tensor:
    """Complex STFT (electrodes, bins < kmax, frames) of the whole mixture."""
    spec = PowerSpectrogram(NFFT, STEP, 20000.0, dev)
    out = torch.zeros((MIX_CHANNELS, kmax, mix.n_frames), dtype=torch.complex64)
    block = 300
    for j0 in range(0, mix.n_frames, block):
        n = min(block, mix.n_frames - j0)
        x = mix.traces(j0, n, gain)
        tr = torch.from_numpy(np.ascontiguousarray(x.T)).to(dev)
        X = spec.stft(tr)[:, :kmax, :n]
        out[:, :, j0 : j0 + n] = X.cpu()
        del X, tr
    return out


def other_lines(F: np.ndarray, skip: set[int], fc: np.ndarray) -> np.ndarray:
    """Distance (frames) of frequency fc to the closest fundamental or
    harmonic (1..MAX_H) of any fish not in `skip`."""
    d = np.full(fc.shape, np.inf)
    for m in range(len(F)):
        if m in skip:
            continue
        for h in range(1, MAX_H + 1):
            d = np.fmin(d, np.abs(h * F[m] - fc))
    return d


def clean_snr(mix: Mixture, gain: float, max_gap_s: float = 60.0) -> np.ndarray:
    """SNR (fish, frames) [dB] of every fish on its own. The truth's SNR is
    the mixture's power at the fish's bin, so a close pair gets the power of
    both: where another line is within CLEAR_HZ, a fish's SNR is interpolated
    in time from the frames where it is clear of others (NaN if they are more
    than `max_gap_s` away)."""
    snr = mix.snr_at(gain)
    G = mix.guess
    out = np.full(snr.shape, np.nan)
    t = np.arange(mix.n_frames) * FT
    for i in range(len(G)):
        d = other_lines(G, {i}, G[i])
        ok = (d >= CLEAR_HZ) & np.isfinite(G[i])
        if ok.sum() < 2:
            continue
        v = np.interp(t, t[ok], snr[i, ok])
        # distance to the nearest clear frame
        idx = np.flatnonzero(ok)
        pos = np.clip(np.searchsorted(idx, np.arange(len(t))), 1, len(idx) - 1)
        gap = (
            np.minimum(
                np.abs(idx[pos] - np.arange(len(t))),
                np.abs(idx[pos - 1] - np.arange(len(t))),
            )
            * FT
        )
        out[i] = np.where(gap <= max_gap_s, v, np.nan)
    return out


def lin_db(snr: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(10 * np.log10(np.mean(10 ** (snr / 10))))


def samples(mix: Mixture, gain: float, W: int) -> list[dict]:
    """Windows of pairs (< PAIR_HZ apart throughout) and clean single fish,
    plus fish on another fish's harmonic and bare harmonics."""
    F = mix.freqs
    G = mix.guess  # also where the truth does not know the fish
    snr = clean_snr(mix, gain)
    n = len(F)
    rows = []
    for j0 in range(0, mix.n_frames - W, W):
        sl = slice(j0, j0 + W)
        f, g, s = F[:, sl], G[:, sl], snr[:, sl]
        full = np.isfinite(f).all(1)
        for i in np.flatnonzero(full):
            # single fish
            d = other_lines(g, {i}, f[i])
            dmin = float(np.nanmin(d)) if np.isfinite(d).any() else np.inf
            on_harm = None
            for m in range(n):
                if m == i or not np.isfinite(g[m]).all():
                    continue
                for h in range(2, MAX_H + 1):
                    if np.max(np.abs(h * g[m] - f[i])) < PAIR_HZ:
                        on_harm = (m, h)
            if dmin >= CLEAR_HZ:
                rows.append(
                    dict(
                        kind="single",
                        fc=f[i].copy(),
                        snr1=lin_db(s[i]),
                        snr2=np.nan,
                        dfm=np.nan,
                        j0=j0,
                        fish=(int(i),),
                        third=dmin,
                    )
                )
            elif dmin >= PAIR_HZ:
                rows.append(
                    dict(
                        kind="near",
                        fc=f[i].copy(),
                        snr1=lin_db(s[i]),
                        snr2=np.nan,
                        dfm=dmin,
                        j0=j0,
                        fish=(int(i),),
                        third=dmin,
                    )
                )
            if on_harm is not None:
                m, h = on_harm
                d3 = other_lines(g, {i, m}, f[i])
                rows.append(
                    dict(
                        kind="onharm",
                        fc=f[i].copy(),
                        snr1=lin_db(s[i]),
                        snr2=lin_db(s[m]),
                        dfm=float(np.mean(np.abs(h * g[m] - f[i]))),
                        j0=j0,
                        fish=(int(i), int(m)),
                        third=float(np.nanmin(d3)),
                        h=h,
                    )
                )
            # bare harmonics of fish i
            for h in (2, 3):
                fh = h * f[i]
                if fh.max() > F_TOP - 5:
                    continue
                d = other_lines(g, {i}, fh)
                if np.nanmin(d) >= CLEAR_HZ:
                    rows.append(
                        dict(
                            kind="harm",
                            fc=fh,
                            snr1=lin_db(s[i]),
                            snr2=np.nan,
                            dfm=np.nan,
                            j0=j0,
                            fish=(int(i),),
                            third=float(np.nanmin(d)),
                            h=h,
                        )
                    )
        for i in np.flatnonzero(full):
            for k in np.flatnonzero(full):
                if k <= i:
                    continue
                dd = np.abs(f[i] - f[k])
                if dd.max() >= PAIR_HZ:
                    continue
                fc = 0.5 * (f[i] + f[k])
                d3 = other_lines(g, {i, k}, fc)
                a, b = lin_db(s[i]), lin_db(s[k])
                rows.append(
                    dict(
                        kind="pair",
                        fc=fc,
                        snr1=max(a, b),
                        snr2=min(a, b),
                        dfm=float(dd.mean()),
                        j0=j0,
                        fish=(int(i), int(k)),
                        third=float(np.nanmin(d3)),
                    )
                )
    return rows


def eig_ratio(R: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """lambda2 / lambda1 and lambda2 / lambda3 of Hermitian matrices (S, E, E)."""
    ev = torch.linalg.eigvalsh(R.cpu()).flip(-1).clamp_min(1e-30)  # descending
    l3 = ev[:, 2] if ev.shape[1] > 2 else torch.full_like(ev[:, 1], float("nan"))
    return ev[:, 1] / ev[:, 0], ev[:, 1] / l3


def features(X: torch.Tensor, rows: list[dict], W: int) -> dict[str, np.ndarray]:
    """Cues of every sample window from the complex STFT X (E, bins, frames)."""
    if not rows:
        return {}
    dev = X.device
    E = X.shape[0]
    kc = np.stack([np.round(r["fc"] / DF).astype(np.int64) for r in rows])  # (S, W)
    jj = np.stack([r["j0"] + np.arange(W) for r in rows])
    kc = np.clip(kc, 1, X.shape[1] - 2)
    out = {}
    K = torch.as_tensor(kc, device=dev)
    J = torch.as_tensor(jj, device=dev)
    x = X[:, K, J].permute(1, 2, 0)  # (S, W, E)
    p = x.abs().square().sum(-1).double()  # summed power (S, W)
    # neighbour bins: a pair 0.6-1.2 Hz apart straddles bins
    p3 = sum(X[:, K + d, J].abs().square().sum(0).double() for d in (-1, 0, 1))
    t = torch.arange(W, device=dev, dtype=torch.float64) - (W - 1) / 2

    def cv_detrended(y):
        ym = y.mean(1, keepdim=True)
        slope = (y * t).sum(1, keepdim=True) / (t * t).sum()
        res = y - ym - slope * t
        return (res.std(1) / ym[:, 0]).cpu().numpy()

    out["beat_cv"] = cv_detrended(p)
    out["beat_cv3"] = cv_detrended(p3)
    ld = 10 * torch.log10(p.clamp_min(1e-30))
    out["beat_db_std"] = cv_detrended(ld + 1000.0) * 1000.0  # std of detrended dB
    xd = x.to(torch.complex128)
    Rc = torch.einsum("swe,swf->sef", xd.conj(), xd) / W
    out["cplx_l2l1"], out["cplx_l2l3"] = (v.cpu().numpy() for v in eig_ratio(Rc))
    Rr = Rc.real.to(torch.complex128)
    out["real_l2l1"], out["real_l2l3"] = (v.cpu().numpy() for v in eig_ratio(Rr))
    # with the neighbour bins (3 W snapshots, same real field per fish)
    x3 = torch.cat([X[:, K + d, J].permute(1, 2, 0) for d in (-1, 0, 1)], 1)
    x3 = x3.to(torch.complex128)
    R3 = torch.einsum("swe,swf->sef", x3.conj(), x3).real / (3 * W)
    out["real3_l2l1"], out["real3_l2l3"] = (
        v.cpu().numpy() for v in eig_ratio(R3.to(torch.complex128))
    )
    # per frame: non-realness of the electrode vector
    re, im = xd.real, xd.imag
    a = (re * re).sum(-1)
    b = (im * im).sum(-1)
    c = (re * im).sum(-1)
    tr = a + b
    disc = torch.sqrt(((a - b) / 2) ** 2 + c * c)
    l1, l2 = tr / 2 + disc, (tr / 2 - disc).clamp_min(0)
    out["frame_nonreal"] = (l2 / (l1 + l2).clamp_min(1e-30)).mean(1).cpu().numpy()
    out["E"] = np.full(len(rows), E)
    return out


def measure(args):
    dev = torch.device(args.device)
    W = max(3, round(args.window / FT))
    OUT.mkdir(parents=True, exist_ok=True)
    split = args.split
    lab = split.removesuffix("-field")
    gain = 1.0 if split.endswith("-field") else 0.0
    allrows, feats = [], {}
    for path in mixture_dirs(lab)[: args.limit]:
        mix = Mixture(path)
        rows = samples(mix, gain, W)
        kmax = int(F_TOP / DF) + 3
        X = stft_bins(mix, gain, kmax, dev)
        # random electrode subsets as in training and the field (2-9)
        rng = np.random.default_rng(zlib.crc32(path.name.encode()))
        for sub in args.electrodes:
            if sub == MIX_CHANNELS:
                el = list(range(MIX_CHANNELS))
            else:
                el = sorted(rng.choice(MIX_CHANNELS, sub, replace=False).tolist())
            fe = features(X[el], rows, W)
            for k, v in fe.items():
                feats.setdefault(f"{k}@{sub}", []).append(v)
        del X
        torch.cuda.empty_cache()
        for r in rows:
            r["mixture"] = path.name
        allrows += rows
        kinds = {
            k: sum(r["kind"] == k for r in rows)
            for k in ("pair", "single", "near", "onharm", "harm")
        }
        print(path.name, kinds, flush=True)
    table = {
        "kind": np.array([r["kind"] for r in allrows]),
        "snr1": np.array([r["snr1"] for r in allrows]),
        "snr2": np.array([r["snr2"] for r in allrows]),
        "dfm": np.array([r["dfm"] for r in allrows]),
        "third": np.array([r["third"] for r in allrows]),
        "fmean": np.array([float(np.mean(r["fc"])) for r in allrows]),
        "mixture": np.array([r["mixture"] for r in allrows]),
        "h": np.array([r.get("h", 0) for r in allrows]),
    }
    for k, v in feats.items():
        table[k] = np.concatenate(v)
    np.savez_compressed(OUT / f"{split}-w{args.window:g}.npz", **table)
    print(OUT / f"{split}-w{args.window:g}.npz")


def wauc(pos: np.ndarray, neg: np.ndarray, wneg: np.ndarray) -> float:
    """AUC (pairs score higher) with weighted negatives; ties count half."""
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    order = np.argsort(neg)
    ns, nw = neg[order], wneg[order] / wneg.sum()
    cw = np.concatenate([[0.0], np.cumsum(nw)])
    lo = np.searchsorted(ns, pos, "left")
    hi = np.searchsorted(ns, pos, "right")
    return float(np.mean(cw[lo] + 0.5 * (cw[hi] - cw[lo])))


def snr_weights(target_snr: np.ndarray, neg_snr: np.ndarray, width=3.0) -> np.ndarray:
    """Weights of the negatives so that their SNR histogram matches the
    target's (bins of `width` dB)."""
    edges = np.arange(-30, 80 + width, width)
    ht, _ = np.histogram(target_snr, edges)
    hn, _ = np.histogram(neg_snr, edges)
    i = np.clip(np.digitize(neg_snr, edges) - 1, 0, len(hn) - 1)
    w = np.where(hn[i] > 0, ht[i] / np.maximum(hn[i], 1), 0.0)
    return w


CUES = [
    ("beat_cv", "beat (summed power CV)"),
    ("beat_cv3", "beat, +-1 bin"),
    ("cplx_l2l1", "complex cov l2/l1"),
    ("real_l2l1", "real cov l2/l1"),
    ("real_l2l3", "real cov l2/l3"),
    ("real3_l2l1", "real cov +-1 bin l2/l1"),
    ("frame_nonreal", "per-frame non-realness"),
]


def report(args):
    res = {}
    lines = []
    for path in sorted(OUT.glob("*.npz")):
        z = dict(np.load(path))
        name = path.stem
        kind = z["kind"]
        subs = sorted({int(k.split("@")[1]) for k in z if "@" in k})
        pos_all = (
            (kind == "pair") & (z["third"] >= CLEAR_HZ) & (z["snr2"] >= args.min_snr)
        )
        neg = kind == "single"
        lines.append(
            f"\n## {name}: {pos_all.sum()} isolated pair windows, {neg.sum()} single windows"
        )
        hdr = "| stratum | n pairs | " + " | ".join(c[1] for c in CUES) + " |"
        for sub in subs:
            lines.append(f"\n{sub} electrodes\n")
            lines.append(hdr)
            lines.append("|" + "---|" * (len(CUES) + 2))
            for dlo, dhi, dname in DF_BANDS:
                for slo, shi, sname in SNR_BANDS:
                    pos = (
                        pos_all
                        & (z["dfm"] >= dlo)
                        & (z["dfm"] < dhi)
                        & (z["snr2"] >= slo)
                        & (z["snr2"] < shi)
                    )
                    if pos.sum() < 5:
                        continue
                    w = snr_weights(z["snr1"][pos], z["snr1"][neg])
                    cells = []
                    for c, _ in CUES:
                        v = z[f"{c}@{sub}"]
                        a = wauc(v[pos], v[neg], w)
                        res[f"{name}|{sub}|{dname}|{sname}|{c}"] = a
                        cells.append(f"{a:.3f}")
                    lines.append(
                        f"| {dname}, {sname} | {pos.sum()} | "
                        + " | ".join(cells)
                        + " |"
                    )
            # fish on a harmonic vs bare harmonic
            pos = (kind == "onharm") & (z["snr1"] >= 6)
            hn = kind == "harm"
            if pos.sum() >= 5 and hn.sum() >= 5:
                w = snr_weights(z["snr2"][pos], z["snr1"][hn])  # harmonic owner's SNR
                cells = []
                for c, _ in CUES:
                    v = z[f"{c}@{sub}"]
                    cells.append(f"{wauc(v[pos], v[hn], w):.3f}")
                lines.append(
                    f"| fish (>= 6 dB) on a harmonic vs bare harmonic | {pos.sum()} | "
                    + " | ".join(cells)
                    + " |"
                )
    text = "\n".join(lines)
    print(text)
    (OUT / "report.md").write_text(text)
    (OUT / "auc.json").write_text(json.dumps(res, indent=1))
    plot(res)


def plot(res: dict):
    import matplotlib.pyplot as plt

    names = sorted({k.split("|")[0] for k in res})
    subs = sorted({int(k.split("|")[1]) for k in res})
    cues = [c for c, _ in CUES]
    labels = dict(CUES)
    colors = [
        "#9b9a96",
        "#c9c8c4",
        "#2a78d6",
        "#eb6834",
        "#eda100",
        "#1baf7a",
        "#6250d6",
    ]
    strata = [(d[2], s[2]) for d in DF_BANDS for s in SNR_BANDS]
    fig, axs = plt.subplots(
        len(names),
        len(subs),
        figsize=(7 * len(subs), 3.6 * len(names)),
        squeeze=False,
        layout="constrained",
    )
    for ni, nm in enumerate(names):
        for si, sub in enumerate(subs):
            ax = axs[ni, si]
            st = [s for s in strata if f"{nm}|{sub}|{s[0]}|{s[1]}|{cues[0]}" in res]
            w = 0.8 / len(cues)
            for ci, c in enumerate(cues):
                v = [res[f"{nm}|{sub}|{d}|{s}|{c}"] for d, s in st]
                ax.bar(
                    np.arange(len(st)) + (ci - (len(cues) - 1) / 2) * w,
                    v,
                    w * 0.92,
                    color=colors[ci % len(colors)],
                    label=labels[c],
                )
            ax.axhline(0.5, color="#52514e", lw=0.8)
            ax.set_ylim(0.3, 1.0)
            ax.set_xticks(range(len(st)), [f"{d}\n{s}" for d, s in st], fontsize=7)
            ax.set_title(
                f"{nm}, {sub} electrodes: AUC pair vs single fish",
                fontsize=10,
                loc="left",
            )
            ax.grid(axis="y", alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)
    axs[0, 0].legend(frameon=False, fontsize=7, ncol=2, loc="lower left")
    path = FIG / "B-pair-cues.png"
    fig.savefig(path, dpi=130)
    print(path)


BEAT = ["beat_cv", "beat_cv3", "beat_db_std"]
SPATIAL = [
    "cplx_l2l1",
    "cplx_l2l3",
    "real_l2l1",
    "real_l2l3",
    "real3_l2l1",
    "real3_l2l3",
    "frame_nonreal",
]


def combine(args):
    """How much the electrodes add to the summed stream: logistic regression
    on the log cues (and the SNR of the stronger fish), trained and scored
    across mixtures (leave groups of mixtures out), pairs (weaker fish
    >= 6 dB) against single fish weighted to the pairs' SNR."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    rows = []
    for path in sorted(OUT.glob("*.npz")):
        z = dict(np.load(path))
        kind = z["kind"]
        subs = sorted({int(k.split("@")[1]) for k in z if "@" in k})
        pos = (kind == "pair") & (z["third"] >= CLEAR_HZ) & (z["snr2"] >= 6)
        neg = kind == "single"
        sel = pos | neg
        y = pos[sel].astype(int)
        snr = z["snr1"][sel]
        w = np.ones(sel.sum())
        w[y == 0] = snr_weights(snr[y == 1], snr[y == 0])
        w[y == 1] = 1.0
        w[y == 0] *= (y == 1).sum() / max(w[y == 0].sum(), 1e-9)
        groups = z["mixture"][sel]
        dfm = z["dfm"][sel]
        for sub in subs:

            def feats(names, sub=sub, z=z, sel=sel, snr=snr):
                cols = [
                    np.log(np.clip(z[f"{n}@{sub}"][sel], 1e-9, None)) for n in names
                ]
                cols.append(snr / 10)
                x = np.stack(cols, 1)
                return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=-20.0)

            res = {}
            for name, names in (
                ("beat", BEAT),
                ("spatial", SPATIAL),
                ("both", BEAT + SPATIAL),
            ):
                x = feats(names)
                score = np.zeros(len(y))
                for tr, te in GroupKFold(5).split(x, y, groups):
                    m = LogisticRegression(max_iter=2000, C=1.0)
                    m.fit(x[tr], y[tr], sample_weight=w[tr])
                    score[te] = m.decision_function(x[te])
                for dlo, dhi, dname in [*DF_BANDS, (0, 1.2, "all < 1.2 Hz")]:
                    p_ = (y == 1) & (dfm >= dlo) & (dfm < dhi)
                    n_ = y == 0
                    res[(name, dname)] = wauc(score[p_], score[n_], w[n_])
            line = f"{path.stem:16s} {sub} el: " + "  ".join(
                f"{d}: beat {res[('beat', d)]:.3f} spatial {res[('spatial', d)]:.3f} "
                f"both {res[('both', d)]:.3f}"
                for _, _, d in [*DF_BANDS, (0, 1.2, "all < 1.2 Hz")]
            )
            print(line, flush=True)
            rows.append(line)
    (OUT / "combine.txt").write_text("\n".join(rows))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("measure")
    a.add_argument("--split", default="ldev")
    a.add_argument("--window", type=float, default=3.0, help="seconds")
    a.add_argument("--electrodes", type=int, nargs="+", default=[9, 4, 2])
    a.add_argument("--limit", type=int)
    a.add_argument("--device", default="cuda")
    b = sub.add_parser("report")
    b.add_argument("--min-snr", type=float, default=0.0, help="weaker fish of a pair")
    sub.add_parser("combine")
    args = p.parse_args(argv)
    {"measure": measure, "report": report, "combine": combine}[args.cmd](args)


if __name__ == "__main__":
    main()
