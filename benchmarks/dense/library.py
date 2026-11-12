"""Source library: every usable tube trial analysed alone, with ground truth.

For each trial a segment of ``SEGMENT`` s is run through wavetracker with the
species range of the tube fish (two fish, high SNR: the reliable case). The
detections are assigned to the lower and the upper fish, which gives a
frequency trace per fish and frame; identities carry the assignment through
rises. The library also stores the stationary hum lines of the segment and
the ``MIX_CHANNELS`` electrodes on which the fish are best seen.

Usage::

    python benchmarks/dense/library.py            # all trials (cached)
    python benchmarks/dense/library.py --only 2022-05-10-10_00 --plot
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter, uniform_filter1d
from scipy.signal import find_peaks, welch

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA, MIX_CHANNELS, RATE, SEGMENT, Trial, segment_start, trials

from wavetracker.config import Config
from wavetracker.pipeline import detect, track_results
from wavetracker.results import Results

LIB = DATA / "sources"
WINDOW = 10.0  # s, resolution of the coarse traces
ASSIGN_TOL = 30.0  # Hz, detections farther from both coarse traces are dropped


def source_config() -> Config:
    return Config.from_dict(
        {
            "detector": {"method": "classic"},
            "harmonic_groups": {
                "min_freq": 400.0,
                "max_freq": 1200.0,
                "min_good_peak_power": None,  # gain differs between setups
                # mains_freq stays at 50 Hz: without it hum teeth are detected
                # and can capture a fish's trace; a fish on a mains harmonic
                # is unknown there instead
            },
        }
    )


def _peaks(f, bins, min_count):
    h = gaussian_filter1d(np.histogram(f, bins)[0].astype(float), 2)
    p, props = find_peaks(h, distance=8, height=min_count)
    order = np.argsort(props["peak_heights"])[::-1]
    return [float(np.median(f[np.abs(f - bins[i]) < 2])) for i in p[order]]


def coarse_traces(res: Results, max_jump: float = 15.0) -> np.ndarray:
    """Frequency of the lower and the upper fish at every frame (2, frames),
    from the two dominant detection peaks in windows of ``WINDOW`` s.

    Initialised from the two dominant peaks of the whole segment, then
    followed forward and backward from the window that agrees best with them,
    so a quiet start cannot capture a trace."""
    t_det = res.times[res.idx_v]
    f = res.fund_v
    bins = np.arange(np.floor(f.min()) - 5, np.ceil(f.max()) + 5.5, 0.5)
    glob = _peaks(f, bins, 1)[:2]
    if len(glob) < 2:
        raise ValueError("fewer than two fish")
    glob = np.sort(glob)
    dt = res.times[1] - res.times[0]
    edges = np.arange(res.times[0], res.times[-1] + WINDOW, WINDOW)
    which = np.digitize(t_det, edges) - 1
    cands = []
    for w in range(len(edges) - 1):
        fw = f[which == w]
        cands.append(np.array(_peaks(fw, bins, 0.05 * WINDOW / dt)) if len(fw) else [])

    def misfit(c):
        if len(c) < 2:
            return np.inf
        return sum(np.abs(np.asarray(c) - g).min() for g in glob)

    w0 = int(np.argmin([misfit(c) for c in cands]))
    traces = np.full((2, len(cands)), np.nan)
    for direction in (range(w0, len(cands)), range(w0, -1, -1)):
        last = glob.copy()
        for w in direction:
            c = np.asarray(cands[w])
            if len(c) == 0:
                continue
            for k in range(2):
                d = np.abs(c - last[k])
                if d.min() <= max_jump:
                    traces[k, w] = c[d.argmin()]
                    last[k] = traces[k, w]
    centers = 0.5 * (edges[:-1] + edges[1:])
    out = np.empty((2, len(res.times)))
    for k in range(2):
        ok = np.isfinite(traces[k])
        tr = np.interp(centers, centers[ok], traces[k, ok])
        tr = median_filter(tr, size=5, mode="nearest")
        out[k] = np.interp(res.times, centers, tr)
    return out


def _drop_isolated(f: np.ndarray, half: int, df: float = 2.0, min_n: int = 3):
    """NaN where a frame has fewer than `min_n` other frames within `half`
    frames and `df` Hz (stray detections of hum or noise given to a fish;
    a fish's trace, rises included, is continuous)."""
    n = len(f)
    count = np.zeros(n, int)
    for d in range(-half, half + 1):
        if d == 0:
            continue
        a, b = max(0, -d), min(n, n - d)
        close = np.abs(f[a:b] - f[a + d : b + d]) <= df  # NaN compares False
        count[a:b] += close
    return np.where(count >= min_n, f, np.nan)


def build_truth(res: Results) -> tuple[np.ndarray, dict]:
    """Frequency of each fish at every frame (2, frames), NaN where it was
    not detected, plus QC numbers.

    1. every detection goes to the closer coarse trace (within ``ASSIGN_TOL``);
    2. identities whose detections agree on one fish (>= 90 %) pass that fish
       to all their detections: rises leave the coarse band but stay in their
       identity;
    3. per frame and fish the detection closest to the fish's running median
       is kept."""
    coarse = coarse_traces(res)
    ref = coarse[:, res.idx_v]
    err = np.abs(ref - res.fund_v[None])
    label = err.argmin(0)
    label[err[label, np.arange(len(label))] > ASSIGN_TOL] = -1

    ident = res.ident_v
    n_relabelled = 0
    for i in np.unique(ident[np.isfinite(ident)]):
        m = ident == i
        lab = label[m]
        lab = lab[lab >= 0]
        if len(lab) < 10:
            continue
        counts = np.bincount(lab, minlength=2)
        k = counts.argmax()
        if counts[k] >= 0.9 * len(lab):
            # detections outside both bands (large rises), or given to the
            # other fish although more than 5 Hz off its trace (a rise that
            # came close to it), follow their identity
            off = np.abs(res.fund_v - coarse[1 - k, res.idx_v]) > 5.0
            take = m & ((label < 0) | ((label == 1 - k) & off))
            n_relabelled += int(take.sum())
            label[take] = k

    n = len(res.times)
    dt = res.times[1] - res.times[0]
    truth = np.full((2, n), np.nan)
    for k in range(2):
        m = label == k
        idx, f = res.idx_v[m], res.fund_v[m]
        # first pass: per frame the detection closest to the coarse trace,
        # then the one closest to the running median (~5 s) of those
        ref = coarse[k, idx]
        for _ in range(2):
            d = np.abs(f - ref)
            order = np.argsort(-d)  # the closest is written last
            per_frame = np.full(n, np.nan)
            per_frame[idx[order]] = f[order]
            ok = np.isfinite(per_frame)
            smooth = median_filter(
                np.interp(np.arange(n), np.flatnonzero(ok), per_frame[ok]),
                size=int(5 / dt) | 1,
                mode="nearest",
            )
            ref = smooth[idx]
        truth[k] = _drop_isolated(per_frame, round(1.0 / dt))

    qc = {
        "coverage": [float(np.isfinite(truth[k]).mean()) for k in range(2)],
        "median_hz": [float(np.nanmedian(truth[k])) for k in range(2)],
        "range_hz": [
            [
                float(np.nanpercentile(truth[k], 1)),
                float(np.nanpercentile(truth[k], 99)),
            ]
            for k in range(2)
        ],
        "min_separation_hz": float(np.nanmin(np.abs(truth[1] - truth[0]))),
        "unassigned": float((label < 0).mean()),
        "relabelled_by_identity": n_relabelled,
    }
    return truth, qc


TRUTH_VERSION = 4  # + isolated detections dropped before completion
RIDGE_MIN_DB = 6.0  # a completed frame needs the ridge this far above the floor
RISE_HZ = 40.0  # in gaps the band reaches this far above the interpolated track
BELOW_HZ = 6.0  # ... and this far below
OTHER_HZ = 4.0  # never closer than this to the other fish
JUMP_BINS = 16  # largest step of the ridge between frames (~10 Hz)
JUMP_COST = 0.5  # dB per bin of a step


def ridge_spectrogram(
    trial: Trial, res: Results, fmax: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Electrode-summed power on wavetracker's frames (frames, bins <= fmax)
    in dB above the local floor (median over ~20 Hz), stationary lines
    (more than 6 dB up in 90 % of the frames) removed."""
    import torch

    from wavetracker.spectrogram import step_size

    sc = res.meta["config"]["spectrogram"]
    nfft, step = sc["nfft"], step_size(sc["nfft"], sc["overlap_frac"])
    n = len(res.times)
    s0 = round(res.times[0] * RATE - nfft / 2)
    x = np.asarray(trial.raw()[s0 : s0 + (n - 1) * step + nfft], np.float32)
    freqs = np.fft.rfftfreq(nfft, 1 / RATE)
    nb = int(np.searchsorted(freqs, fmax))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    win = torch.hann_window(nfft, periodic=False, device=dev)
    p = torch.zeros((nb, n), device=dev)
    for c in range(x.shape[1]):
        xc = torch.as_tensor(np.ascontiguousarray(x[:, c]), device=dev)
        z = torch.stft(xc, nfft, step, window=win, center=False, return_complex=True)
        p += (z[:nb].real ** 2 + z[:nb].imag ** 2)[:, :n]
        del z
    del x
    db = 10 * torch.log10(p.T + 1e-30).cpu().numpy()  # (frames, bins)
    del p
    blk = max(3, int(20 / freqs[1]))
    nblk = nb // blk
    med = np.median(db[:, : nblk * blk].reshape(n, nblk, blk), axis=2)
    centers = (np.arange(nblk) + 0.5) * blk
    floor = np.stack([np.interp(np.arange(nb), centers, m) for m in med])
    snr = db - floor
    line = (snr > 6.0).mean(0) > 0.9  # stationary lines, removed per gap
    return snr, freqs[:nb], line


def _ridge_path(score, lo, hi, start, end):
    """Best path through score (T, B) with bins in [lo_t, hi_t], anchored at
    bin `start` before the first and `end` after the last frame (None: free);
    a step of d bins costs JUMP_COST |d|, steps above JUMP_BINS are not
    allowed."""
    T, B = score.shape
    bins = np.arange(B)
    allowed = (bins[None] >= lo[:, None]) & (bins[None] <= hi[:, None])
    sc = np.where(allowed, np.clip(score, -5.0, 25.0), -np.inf)
    prev = np.zeros(B) if start is None else np.full(B, -np.inf)
    if start is not None:
        prev[start] = 0.0
    back = np.zeros((T, B), np.int8)
    offsets = range(-min(JUMP_BINS, B - 1), min(JUMP_BINS, B - 1) + 1)

    def step(prev):
        best = np.full(B, -np.inf)
        arg = np.zeros(B, np.int8)
        for d in offsets:  # from bin j - d to bin j
            cand = np.full(B, -np.inf)
            if d >= 0:
                cand[d:] = prev[: B - d] if d else prev
            else:
                cand[:d] = prev[-d:]
            cand -= JUMP_COST * abs(d)
            better = cand > best
            best[better] = cand[better]
            arg[better] = d
        return best, arg

    for t in range(T):
        best, arg = step(prev)
        prev = best + sc[t]
        back[t] = arg
    if end is not None:
        best, arg = step(prev)
        j = int(end)
        j = int(np.argmax(prev)) if not np.isfinite(best[j]) else j - int(arg[j])
    else:
        j = int(np.argmax(prev))
    path = np.empty(T, int)
    for t in range(T - 1, -1, -1):
        path[t] = j
        j -= int(back[t, j])
    return path


def complete_truth(trial: Trial, res: Results, truth: np.ndarray) -> np.ndarray:
    """Truth on every frame on which a fish's ridge is visible.

    In every gap of a fish's detections the ridge is followed from the last
    to the next detection (dynamic programming) in a band from BELOW_HZ under
    to RISE_HZ over the interpolated track (rises the tracker missed), never
    within OTHER_HZ of the other fish. A frame is kept if the ridge is at
    least RIDGE_MIN_DB above the local floor."""
    fmax = 1.3 * np.nanmax(truth) + RISE_HZ
    snr, freqs, line = ridge_spectrogram(trial, res, fmax)
    df = freqs[1]
    # teeth of the recording's hum combs and of the mains: in a gap the ridge
    # may not run on them, unless the fish itself is close to one
    spacings = [float(sp) for sp in res.meta.get("interference_combs", [])] + [50.0]
    teeth = np.unique(
        np.concatenate([np.arange(1, int(fmax / sp) + 1) * sp for sp in spacings])
    )
    out = truth.copy()
    n = truth.shape[1]
    frames = np.arange(n)
    filled = []
    for k in range(2):
        ok = np.isfinite(truth[k])
        filled.append(np.interp(frames, frames[ok], truth[k][ok]))
    for k in range(2):
        other = filled[1 - k]
        ok = np.isfinite(truth[k])
        idx = np.flatnonzero(~ok)
        if len(idx) == 0:
            continue
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        for run in runs:
            g0, g1 = run[0], run[-1] + 1
            c = filled[k][g0:g1]
            lo_f, hi_f = c - BELOW_HZ, c + RISE_HZ
            o = other[g0:g1]
            above = o > c
            hi_f = np.where(above, np.minimum(hi_f, o - OTHER_HZ), hi_f)
            lo_f = np.where(~above, np.maximum(lo_f, o + OTHER_HZ), lo_f)
            b0 = max(0, int(np.floor(lo_f.min() / df)) - 1)
            b1 = min(len(freqs) - 1, int(np.ceil(hi_f.max() / df)) + 1)
            if b1 <= b0 + 2:
                continue
            lo = np.clip(np.ceil(lo_f / df) - b0, 0, b1 - b0 - 1).astype(int)
            hi = np.clip(np.floor(hi_f / df) - b0, 0, b1 - b0 - 1).astype(int)
            bad = hi < lo
            lo[bad] = hi[bad]
            start = round(truth[k, g0 - 1] / df) - b0 if g0 > 0 else None
            end = round(truth[k, g1] / df) - b0 if g1 < n else None
            if start is not None and not 0 <= start < b1 - b0:
                start = None
            if end is not None and not 0 <= end < b1 - b0:
                end = None
            sc = snr[g0:g1, b0:b1].copy()
            fb = (np.arange(b0, b1)) * df
            # hum teeth, mains harmonics and stationary lines are blocked
            # unless the fish is close to them (a fish on a mains harmonic is
            # exactly what the detector misses)
            near_fish = np.abs(fb[None, :] - c[:, None]).min(0) <= 2.0
            sc[:, line[b0:b1] & ~near_fish] = -5.0
            near = teeth[(teeth > (b0 - 2) * df) & (teeth < (b1 + 2) * df)]
            for t in near:
                if np.abs(c - t).min() > 2.0:
                    sc[:, np.abs(fb - t) <= 0.8] = -5.0
            path = _ridge_path(sc, lo, hi, start, end)
            val = sc[np.arange(len(path)), path]
            # parabolic refinement on the dB values
            j = np.clip(path + b0, 1, len(freqs) - 2)
            a, b, cc = (snr[np.arange(g0, g1), j + d] for d in (-1, 0, 1))
            den = a - 2 * b + cc
            off = np.where(np.abs(den) > 1e-9, 0.5 * (a - cc) / den, 0.0)
            f = (j + np.clip(off, -0.5, 0.5)) * df
            # a ridge is continuous: a path wandering through noise passes the
            # threshold only here and there, and jumps
            # the ridge must be about as strong as the fish next to the gap
            # (within 25 dB of the detections that bound it)
            ends = [g for g in (g0 - 1, g1) if 0 <= g < n]
            anchor = np.mean(
                [snr[g, min(round(truth[k, g] / df), len(freqs) - 1)] for g in ends]
            )
            keep = val >= max(RIDGE_MIN_DB, anchor - 25.0)
            if len(f) >= 3:
                smooth = median_filter(f, size=min(7, len(f) | 1), mode="nearest")
                keep &= np.abs(f - smooth) <= 2.0
                keep &= uniform_filter1d(keep.astype(float), 9, mode="nearest") >= 0.5
            out[k, g0:g1] = np.where(keep, f, np.nan)
    return out


def hum_lines(x: np.ndarray, spacings, fish_bands, fmax: float = 9500.0) -> np.ndarray:
    """Frequencies of stationary comb teeth in `x` (samples, channels): peaks of
    the long-term spectrum within 0.8 Hz of k * spacing, 10 dB above their
    surroundings and not within 3 Hz of a fish band."""
    f, p = welch(x, RATE, nperseg=2**18, axis=0)
    db = 10 * np.log10(p.sum(1) + 1e-30)
    df = f[1]
    lines = []
    for sp in spacings:
        for k in range(1, int(fmax / sp) + 1):
            m = np.abs(f - k * sp) <= 0.8
            if not m.any():
                continue
            i = np.flatnonzero(m)[db[m].argmax()]
            ctx = db[max(0, i - int(5 / df)) : i + int(5 / df)]
            if db[i] - np.median(ctx) < 10.0:
                continue
            a, b, c = db[i - 1 : i + 2]
            fi = f[i] + 0.5 * (a - c) / (a - 2 * b + c) * df
            if any(lo - 3 <= fi <= hi + 3 for lo, hi in fish_bands):
                continue
            lines.append(fi)
    return np.unique(np.round(lines, 3))


def best_channels(res: Results, truth: np.ndarray, n: int) -> list[int]:
    """The `n` electrodes on which either fish is strongest (median power of
    its detections, each fish normalised to its best electrode)."""
    score = np.zeros(res.sign_v.shape[1])
    for k in range(2):
        m = np.isclose(res.fund_v, truth[k, res.idx_v])
        p = np.median(res.sign_v[m], axis=0)
        score = np.maximum(score, p / p.max())
    return sorted(int(c) for c in np.argsort(score)[::-1][:n])


def analyse(
    trial: Trial, device: str = "auto", overwrite: bool = False, retruth: bool = False
) -> dict:
    """Analyse one trial (cached). `retruth`: keep the tracking results and
    rebuild truth, hum lines and QC."""
    start = segment_start(trial)
    out = LIB / trial.name
    entry_file = out / "source.json"
    if entry_file.exists() and not (overwrite or retruth):
        return json.loads(entry_file.read_text())
    if retruth and not overwrite and entry_file.exists():
        done = json.loads(entry_file.read_text())
        if done.get("truth_version") == TRUTH_VERSION:  # already up to date
            return done
    cfg = source_config()
    t0 = time.perf_counter()
    if retruth and (out / "ident_v.npy").exists():
        res = Results.load(out)
    else:
        for attempt in range(30):  # other sessions share the GPU
            try:
                det = detect(
                    trial.path, out, cfg, start=start, duration=SEGMENT, device=device
                )
                break
            except RuntimeError as e:
                if "out of memory" not in str(e) or attempt == 29:
                    raise
                print("  GPU out of memory, retrying in 60 s", flush=True)
                time.sleep(60)
        res = det.results
        track_results(res, cfg)
        res.save(out)
    detected, qc = build_truth(res)
    truth = complete_truth(trial, res, detected)
    np.savez(out / "truth.npz", times=res.times, freqs=truth, freqs_detected=detected)
    qc["coverage_detected"] = qc["coverage"]
    qc["coverage"] = [float(np.isfinite(truth[k]).mean()) for k in range(2)]
    qc["min_separation_hz"] = float(np.nanmin(np.abs(truth[1] - truth[0])))
    both = np.isfinite(truth).all(0)
    qc["close_fraction"] = float((np.abs(truth[1] - truth[0])[both] < 3.0).mean())

    combs = res.meta.get("interference_combs", [])
    hum_file = out / "hum_lines.npy"
    if retruth and hum_file.exists():  # reading the raw data is the slow part
        lines = np.load(hum_file)
    else:
        i0 = int(start * RATE)
        x = np.asarray(trial.raw()[i0 : i0 + int(SEGMENT * RATE)], np.float64)
        bands = [
            (h * qc["range_hz"][k][0] - 5, h * qc["range_hz"][k][1] + 5)
            for k in range(2)
            for h in range(1, 15)
        ]
        lines = hum_lines(x, [float(s) for s in combs], bands)
        np.save(hum_file, lines)

    entry = trial.to_dict() | {
        "truth_version": TRUTH_VERSION,
        "start": start,
        "length": SEGMENT,
        "channels_used": best_channels(res, truth, MIX_CHANNELS),
        "hum_lines": len(lines),
        "interference_combs": combs,
        "n_ids": res.n_ids,
        "seconds": time.perf_counter() - t0,
        **qc,
    }
    # the pair must be the trial's fish: the ratio of their frequencies as in
    # the metadata (whose values are temperature corrected), apart (a rise may
    # come close to the other fish for a moment) and known
    # in at least half of the frames (gaps are unknown, not wrong)
    t_lo, t_hi = sorted(qc["median_hz"])
    m_lo, m_hi = sorted(trial.eodf)
    entry["ratio_error"] = float(abs(np.log((t_hi / t_lo) / (m_hi / m_lo))))
    entry["ok"] = bool(
        entry["ratio_error"] < 0.03
        and qc["close_fraction"] < 0.01  # not two traces on one fish
        # completion bridges gaps of the detections; with less than half of
        # the frames detected it follows hum as readily as the fish
        and min(qc["coverage_detected"]) >= 0.5
        and min(qc["coverage"]) >= 0.5
    )
    entry_file.write_text(json.dumps(entry, indent=1))
    return entry


def plot_source(entry: dict, path: Path, ax=None):
    """Overview spectrogram of the segment with the ground truth."""
    import matplotlib.pyplot as plt

    from wavetracker.results import load_sparse_spectrogram

    d = LIB / entry["name"]
    spec, freqs, times = load_sparse_spectrogram(d)
    tr = np.load(d / "truth.npz")
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(11, 4), layout="constrained")
    lo = min(r[0] for r in entry["range_hz"]) - 25
    hi = max(r[1] for r in entry["range_hz"]) + 40
    fm = (freqs >= lo) & (freqs <= hi)
    db = 10 * np.log10(spec[fm] + 1e-20)
    ax.pcolormesh(
        (times - times[0]) / 60,
        freqs[fm],
        db,
        cmap="gray_r",
        vmin=np.percentile(db, 50),
        vmax=np.percentile(db, 99.7),
        shading="auto",
        rasterized=True,
    )
    det = tr["freqs_detected"] if "freqs_detected" in tr.files else tr["freqs"]
    tm = (tr["times"] - tr["times"][0]) / 60
    for k, color in enumerate(("#2a78d6", "#eb6834")):
        ax.plot(tm, det[k], ".", ms=0.6, color=color, rasterized=True)
        added = np.isfinite(tr["freqs"][k]) & ~np.isfinite(det[k])
        ax.plot(
            tm[added],
            tr["freqs"][k][added],
            ".",
            ms=0.6,
            color="#1baf7a",
            rasterized=True,
        )
    ax.set_ylim(lo, hi)
    cov = ", ".join(f"{c * 100:.0f}" for c in entry["coverage"])
    if "coverage_detected" in entry:
        det_cov = ", ".join(f"{c * 100:.0f}" for c in entry["coverage_detected"])
        cov = f"{cov} (detected {det_cov}; green: completed)"
    flag = "" if entry["ok"] else "  REJECTED"
    ax.set_title(
        f"{entry['name']} ({entry['group']}, {entry['split']}) at "
        f"{entry['start'] / 3600:.2f} h: truth covers {cov} %{flag}",
        loc="left",
        fontsize=9,
    )
    ax.tick_params(labelsize=7)
    if own:
        ax.set_xlabel("time in segment [min]")
        ax.set_ylabel("frequency [Hz]")
        fig.savefig(path, dpi=130)
        plt.close(fig)


def contact_sheet(entries: list[dict], path: Path, per_page: int = 12):
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    pages = [entries[i : i + per_page] for i in range(0, len(entries), per_page)]
    out = []
    for p, page in enumerate(pages):
        rows = (len(page) + 1) // 2
        fig, axs = plt.subplots(
            rows, 2, figsize=(16, 2.6 * rows), layout="constrained", squeeze=False
        )
        for ax, e in zip(axs.flat, page, strict=False):
            plot_source(e, None, ax=ax)
        for ax in axs.flat[len(page) :]:
            ax.set_visible(False)
        f = path.with_name(f"{path.stem}_{p + 1}{path.suffix}")
        fig.savefig(f, dpi=110)
        plt.close(fig)
        out.append(f)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--only", nargs="*", help="trial names")
    p.add_argument("--device", default="auto")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--retruth", action="store_true", help="rebuild truth from cached tracking"
    )
    p.add_argument("--plot", action="store_true", help="contact sheets of the truth")
    args = p.parse_args(argv)
    todo = [t for t in trials() if not args.only or t.name in args.only]
    LIB.mkdir(parents=True, exist_ok=True)
    entries = []
    for i, t in enumerate(todo):
        print(
            f"[{i + 1}/{len(todo)}] {t.name} ({t.channels} ch, {t.split})", flush=True
        )
        e = analyse(t, args.device, args.overwrite, args.retruth)
        print(
            f"  coverage {e['coverage']}, {e['median_hz']} Hz, "
            f"{e['hum_lines']} hum lines, ok={e['ok']}",
            flush=True,
        )
        entries.append(e)
    if not args.only:
        (DATA / "library.json").write_text(json.dumps(entries, indent=1))
    if args.plot:
        for f in contact_sheet(entries, DATA / "figures" / "library.png"):
            print(f)


if __name__ == "__main__":
    main()
