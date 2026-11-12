"""Truth through the steep part of rises.

The source truth (``library.py``) comes from tracking each source recording
alone with the classic wavetracker. At the onset of a rise the frequency
climbs by up to 60 Hz within 0.05-0.5 s, faster than wavetracker's 1.64-s
analysis window can follow: the tracker loses the fish for a few frames and
picks it up again after the steep part. The truth then misses the steep part
(NaN), or the ridge completion and the mixtures' interpolation of short gaps
bridge it along a straight line, and frames whose window straddles the onset
were detected at the old frequency.

Here every rise onset of a source is followed in the source's own recording
by the instantaneous frequency of that fish: the electrodes are band-passed
around the fish (the other fish, its harmonics and sub-harmonics kept out),
and a pulse-pair estimator (phase advance of the analytic signal from sample
to sample, summed over electrodes and a 30-ms window, i.e. power weighted)
gives the frequency every 10 ms; a running median over 0.15 s removes chirps
and keeps steps. Between the last frame before the climb and the first after
it on which the old truth agrees with that frequency (within ``AGREE_HZ``),
the truth is replaced by it, and those frames are flagged. A rise is left as
it was when the other fish (or one of its harmonics) lies in its range, the
trace is noisy (the 30-ms frequency off the smoothed one by more than
``MAX_RESID_HZ`` for longer than a chirp), runs into a band edge or onto a
mains line while the power drops, when no detected frame of the old truth
agrees with it on either side of the climb, or when the old truth around
the rise does not (``MIN_AGREE``).

New files, the old ones stay untouched:

* ``sources/<trial>/truth_rises.npz``: ``times``, ``freqs``,
  ``freqs_detected`` as ``truth.npz`` (NaN on the replaced frames), ``rise``
  (fish, frames) for the replaced frames, and the 10-ms trace through every
  corrected onset (``fine_fish``, ``fine_times``, ``fine_freqs``);
  ``sources/<trial>/rises.json`` lists every onset and what was done.
* ``mixtures/<split>/<name>/truth_rises.npz``: everything ``truth.npz``
  holds, rebuilt with the stored warps from the corrected source truths
  (``mixture.warp_truth``), plus ``rise`` (fish, samples) for the new frames.
  The fish power is that of a 1.64-s window at the truth frequency, which is
  meaningless across a step; on SNR frames that see a corrected onset the
  fish and noise power are interpolated (in dB) from the frames around it
  (a fish's power hardly changes during a rise).

Usage::

    python benchmarks/dense/rise_truth.py sources            # all dev sources
    python benchmarks/dense/rise_truth.py sources --only 2019-11-25-09_59 --plot
    python benchmarks/dense/rise_truth.py mixtures --splits ltrain ldev ldev2
    python benchmarks/dense/rise_truth.py mixtures --check   # old truth reproduced?
    python benchmarks/dense/rise_truth.py report             # validation figures
    python benchmarks/dense/rise_truth.py runs --tags A B --mixtures ldev/density-10-s0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA, RATE, Trial

LIB = DATA / "sources"
FIG = DATA / "figures" / "risetruth"
TRUTH_FILE = "truth_rises.npz"

IF_WIN_S = 0.03  # pulse-pair window [s]
IF_HOP_S = 0.01  # step of the fine trace [s]
IF_MEDIAN_S = 0.15  # running median: removes chirps (10-40 ms), keeps steps
PRE_S, POST_S = 2.0, 4.0  # analysis window: before the onset, after the peak
PAD_S = 2.0  # raw data read beyond the window (band-pass edge effects)
BAND_EDGE_HZ = 2.0  # raised-cosine transition of the band-pass
BELOW_HZ = 5.0  # band below the lowest truth in the window
ABOVE_HZ = 15.0  # ... above the highest (the truth may miss the peak)
WIDEN_HZ = 20.0  # band widened by this when the frequency runs into its top
OTHER_HZ = 4.0  # the band stays this far from the other fish's lines
AGREE_HZ = 1.0  # old truth and instantaneous frequency agree
EDGE_HZ = 1.0  # instantaneous frequency this close to a band edge: unreliable
MAX_RESID_HZ = 1.5  # median deviation of the 30-ms from the smoothed frequency
NOISY_N, NOISY_FRAC = 5, 0.1  # ... allowed in this many samples or this fraction
MIN_POWER_DB = -30.0  # band power relative to the anchors: below, unreliable
LINE_HZ = 0.3  # the frequency sits on a mains or comb line ...
LINE_FRAC = 0.5  # ... in this fraction of the stretch while the band power
LINE_POWER_DB = -10.0  # ... drops below this (the line took over): unreliable
MIN_AGREE = 0.8  # detected frames of the window (outside the stretch) agreeing
MAX_SPAN_S = 6.0  # longest stretch between the anchors
STEP_LAG_S = 0.1  # the climb is found as the largest rise over this lag


# --- instantaneous frequency ----------------------------------------------


def read_segment(trial: Trial, t0: float, t1: float, channels) -> tuple:
    """Raw traces (samples, channels) of [t0, t1] s and the time of the first
    sample, read without a memory map (the address-space limit of capped jobs
    does not fit maps of 15-30 GB files)."""
    i0 = max(0, int(np.floor(t0 * RATE)))
    i1 = int(np.ceil(t1 * RATE))
    path = Path(trial.path) / "traces-grid1.raw"
    x = np.fromfile(
        path,
        np.float32,
        count=(i1 - i0) * trial.channels,
        offset=i0 * trial.channels * 4,
    )
    x = x[: len(x) // trial.channels * trial.channels].reshape(-1, trial.channels)
    return x[:, list(channels)].astype(np.float64), i0 / RATE


def band_analytic(x: np.ndarray, lo: float, hi: float, edge: float = BAND_EDGE_HZ):
    """Analytic signal (samples, channels) of `x` band-passed to [lo, hi] Hz
    with raised-cosine edges of `edge` Hz outside."""
    n = len(x)
    spec = np.fft.rfft(x, axis=0)
    f = np.fft.rfftfreq(n, 1 / RATE)
    m = np.clip((f - (lo - edge)) / edge, 0, 1) * np.clip((hi + edge - f) / edge, 0, 1)
    m = np.sin(0.5 * np.pi * m) ** 2
    z = np.zeros((n, x.shape[1]), complex)
    z[: len(f)] = 2 * spec * m[:, None]
    return np.fft.ifft(z, axis=0)


def pulse_pair(z: np.ndarray, win_s: float = IF_WIN_S, hop_s: float = IF_HOP_S):
    """Instantaneous frequency of the analytic signals `z` (samples, channels):
    the phase advance from sample to sample, summed over channels and a Hann
    window of `win_s` (power weighted). Returns times of the window centres
    [s from the first sample], frequency [Hz] and power."""
    r = (np.conj(z[:-1]) * z[1:]).sum(1)
    p = (np.abs(z[:-1]) ** 2).sum(1)
    w = max(3, int(win_s * RATE))
    h = max(1, int(hop_s * RATE))
    k = np.hanning(w + 2)[1:-1]

    def smooth(a):
        return np.convolve(a, k, "valid")[::h]

    rr = smooth(r.real) + 1j * smooth(r.imag)
    pp = smooth(p)
    t = (np.arange(len(rr)) * h + w / 2) / RATE
    return t, np.angle(rr) * RATE / (2 * np.pi), pp


def running_median(f: np.ndarray, n: int) -> np.ndarray:
    n |= 1
    h = n // 2
    pad = np.concatenate([np.full(h, f[0]), f, np.full(h, f[-1])])
    return np.median(np.lib.stride_tricks.sliding_window_view(pad, n), axis=1)


def _fill(f: np.ndarray) -> np.ndarray:
    ok = np.isfinite(f)
    if ok.sum() < 2:
        return np.full(len(f), np.nan)
    i = np.arange(len(f))
    return np.interp(i, i[ok], f[ok])


# --- one onset ----------------------------------------------------------------


def _band(own: np.ndarray, other: np.ndarray, size: float) -> tuple | None:
    """Pass band [lo, hi] around a fish's truth `own` in the window and the
    highest `hi` allowed, away from the lines of the other fish (fundamental,
    harmonics, sub-harmonics); None if one of them lies within the fish's own
    range."""
    f_lo, f_hi = float(np.nanmin(own)), float(np.nanmax(own))
    lo = f_lo - BELOW_HZ
    hi = f_hi + max(ABOVE_HZ, 0.5 * size)
    cap = f_hi + 3 * WIDEN_HZ
    o_lo, o_hi = float(np.nanmin(other)), float(np.nanmax(other))
    for h in (1 / 3, 1 / 2, 1, 2, 3):
        l_lo, l_hi = h * o_lo, h * o_hi
        if l_hi < lo - OTHER_HZ or l_lo > cap + OTHER_HZ:
            continue
        if l_lo <= f_hi + 1 and l_hi >= f_lo - 1:
            return None
        if l_lo > f_hi:
            cap = min(cap, l_lo - OTHER_HZ)
        else:
            lo = max(lo, l_hi + OTHER_HZ)
    hi = min(hi, cap)
    if hi < f_hi + 1:
        return None
    return lo, hi, cap


def lines_near(lo: float, hi: float, combs) -> np.ndarray:
    """Mains (50 Hz) and interference-comb teeth in [lo, hi]."""
    out = []
    for sp in [50.0, *[float(c) for c in combs]]:
        k = np.arange(np.ceil(lo / sp), np.floor(hi / sp) + 1)
        out.extend(k * sp)
    return np.unique(out)


def instantaneous(trial, channels, w0, w1, lo, hi):
    """Fine trace of [w0, w1]: times, smoothed frequency, 30-ms frequency and
    smoothed power."""
    x, xs = read_segment(trial, w0 - PAD_S, w1 + PAD_S, channels)
    z = band_analytic(x, lo, hi)
    del x
    ti, fi, pw = pulse_pair(z)
    del z
    ti = ti + xs
    keep = (ti >= w0 - 0.5) & (ti <= w1 + 0.5)
    ti, fi, pw = ti[keep], fi[keep], pw[keep]
    n = round(IF_MEDIAN_S / IF_HOP_S)
    return ti, running_median(fi, n), fi, running_median(pw, n)


def follow_onset(
    trial: Trial,
    channels,
    times: np.ndarray,
    truth: np.ndarray,
    detected: np.ndarray,
    rise: dict,
    combs=(),
) -> dict:
    """Follow one rise onset of fish ``rise['fish']`` by its instantaneous
    frequency. Returns the rise record with ``status`` ('corrected',
    'traced' (nothing to correct), 'other-fish', 'unreliable', 'disagrees',
    'no-anchor', 'too-long') and, if corrected, the stretch between the
    anchors and the fine trace through it."""
    k = rise["fish"]
    out = dict(rise)
    dk = np.isfinite(detected[k])
    # the window reaches back and on to detected frames of the fish
    w0 = rise["t0"] - PRE_S
    prev = np.flatnonzero(dk & (times <= rise["t0"] - 0.5))
    if len(prev):
        w0 = max(min(w0, times[prev[-1]] - 1.0), rise["t0"] - MAX_SPAN_S - 2)
    w1 = rise["t_peak"] + POST_S
    nxt = np.flatnonzero(dk & (times >= rise["t_peak"] + 1.0))
    if len(nxt):
        w1 = min(max(w1, times[nxt[0]] + 1.0), rise["t0"] + MAX_SPAN_S + 2)
    win = (times >= w0) & (times <= w1)
    own = _fill(truth[k])[win]
    other = _fill(truth[1 - k])[win]
    band = _band(own, other, rise["size"]) if np.isfinite(own).any() else None
    if band is None:
        out["status"] = "other-fish"
        return out
    lo, hi, cap = band
    while True:
        ti, fm, fi, pm = instantaneous(trial, channels, w0, w1, lo, hi)
        top = fm >= hi - EDGE_HZ
        if not top.any() or hi >= cap:
            break
        hi = min(cap, hi + WIDEN_HZ)
    out["band"] = [lo, hi]
    # frames: instantaneous frequency at the frame time
    fr = np.flatnonzero(win)
    f_fr = np.interp(times[fr], ti, fm)
    d = detected[k, fr]
    agree = np.isfinite(d) & (np.abs(f_fr - d) <= AGREE_HZ)
    if agree.sum() < 3:
        out["status"] = "no-anchor"
        return out
    p_ref = float(np.median(np.interp(times[fr[agree]], ti, pm)))
    resid = running_median(np.abs(fi - fm), round(IF_MEDIAN_S / IF_HOP_S))
    ok_if = (
        (fm >= lo + EDGE_HZ)
        & (fm <= hi - EDGE_HZ)
        & (pm >= p_ref * 10 ** (MIN_POWER_DB / 10))
    )
    # the climb: the largest rise of the fine trace over STEP_LAG_S near the onset
    lag = round(STEP_LAG_S / IF_HOP_S)
    climb = np.full(len(fm), -np.inf)
    climb[:-lag] = fm[lag:] - fm[:-lag]
    near = (ti >= rise["t0"] - 1.0) & (ti <= rise["t_peak"] + 0.5)
    if not near.any():
        out["status"] = "no-anchor"
        return out
    i_step = int(np.flatnonzero(near)[np.argmax(climb[near])])
    t_step = float(ti[i_step] + 0.5 * STEP_LAG_S)
    out["t_step"] = t_step
    before = np.flatnonzero(agree & (times[fr] <= t_step))
    after = np.flatnonzero(agree & (times[fr] >= t_step))
    if len(before) == 0 or len(after) == 0:
        out["status"] = "no-anchor"
        return out
    a, b = int(fr[before[-1]]), int(fr[after[0]])
    t_a, t_b = float(times[a]), float(times[b])
    out["t_a"], out["t_b"] = t_a, t_b
    inside = np.arange(a + 1, b)
    # validation: the old detected truth elsewhere in the window agrees
    outside = np.isfinite(d) & ((times[fr] <= t_a) | (times[fr] >= t_b))
    out["agree_outside"] = float(agree[outside].mean()) if outside.any() else np.nan
    if outside.any():
        dev = np.abs(f_fr - d)[outside]
        out["median_dev_outside"] = float(np.median(dev))
    sel = (ti > t_a + 1e-3) & (ti < t_b - 1e-3)  # strictly between the anchors
    after_b = (ti >= t_a) & (ti <= t_b + 2.0)
    out["if_max"] = float(fm[after_b].max())
    out["size_if"] = float(out["if_max"] - f_fr[before[-1]])
    if len(inside) == 0:
        out["status"] = "traced"
        return out
    if t_b - t_a > MAX_SPAN_S:
        out["status"] = "too-long"
        return out
    if not (out["agree_outside"] >= MIN_AGREE):
        out["status"] = "disagrees"
        return out
    lines = lines_near(lo, hi, combs)
    on_line = np.zeros(sel.sum(), bool)
    for fl in lines:
        on_line |= np.abs(fm[sel] - fl) <= LINE_HZ
    out["min_power_db"] = float(10 * np.log10(pm[sel].min() / p_ref))
    out["max_resid_hz"] = float(resid[sel].max())
    out["on_line"] = float(on_line.mean())
    # a noisy trace is unreliable; short excursions (the climb itself, a
    # chirp) are not
    n_noisy = int((resid[sel] > MAX_RESID_HZ).sum())
    captured = (
        on_line.mean() > LINE_FRAC
        and t_b - t_a > 0.3
        and out["min_power_db"] < LINE_POWER_DB
    )
    if (
        not ok_if[sel].all()
        or n_noisy > max(NOISY_N, NOISY_FRAC * sel.sum())
        or captured
    ):
        out["status"] = "unreliable"
        return out
    new = np.interp(times[inside], ti, fm)
    old = truth[k, inside]
    out["status"] = "corrected"
    out["frames"] = inside.tolist()
    out["freqs"] = new.tolist()
    out["old_unknown"] = int((~np.isfinite(old)).sum())
    out["old_completed"] = int(
        (np.isfinite(old) & ~np.isfinite(detected[k, inside])).sum()
    )
    dev = np.abs(new - old)
    out["old_off_2hz"] = int((np.isfinite(old) & (dev > 2.0)).sum())
    out["max_change"] = float(np.nanmax(dev)) if np.isfinite(dev).any() else np.nan
    out["fine_times"] = ti[sel].tolist()
    out["fine_freqs"] = fm[sel].tolist()
    return out


# --- a source -------------------------------------------------------------------


def onsets(times: np.ndarray, truth: np.ndarray) -> list[dict]:
    """Rises of both fish (``rise_diag.find_rises``) in the source truth,
    gaps filled by interpolation (a gap followed by a higher frequency that
    decays back is a rise too)."""
    from rise_diag import find_rises

    filled = np.stack([_fill(f) for f in truth])
    return find_rises(times, filled)


def complete_source(entry: dict) -> dict:
    """Correct the rise onsets of one library source; writes
    ``truth_rises.npz`` and ``rises.json`` next to its ``truth.npz``."""
    d = LIB / entry["name"]
    trial = Trial(**{k: entry[k] for k in Trial.__dataclass_fields__})
    tr = np.load(d / "truth.npz")
    times, truth, detected = tr["times"], tr["freqs"], tr["freqs_detected"]
    new = truth.copy()
    new_det = detected.copy()
    flag = np.zeros(truth.shape, bool)
    rise_id = np.full(truth.shape, -1, np.int32)
    records = []
    fine = {"fish": [], "times": [], "freqs": [], "rise": []}
    for i, r in enumerate(onsets(times, truth)):
        rec = follow_onset(
            trial,
            entry["channels_used"],
            times,
            truth,
            detected,
            r,
            entry.get("interference_combs", {}),
        )
        rec["id"] = i
        if rec["status"] == "corrected" and any(
            o["fish"] == rec["fish"] and o["t_a"] < rec["t_b"] and rec["t_a"] < o["t_b"]
            for o in records
            if o["status"] == "corrected"
        ):
            rec["status"] = "overlap"  # with an onset corrected before
        if rec["status"] == "corrected":
            k = rec["fish"]
            fr = np.asarray(rec["frames"])
            new[k, fr] = rec["freqs"]
            new_det[k, fr] = np.nan
            flag[k, fr] = True
            rise_id[k, fr] = i
            n = len(rec["fine_times"])
            fine["fish"].append(np.full(n, k, np.int8))
            fine["times"].append(np.asarray(rec["fine_times"]))
            fine["freqs"].append(np.asarray(rec["fine_freqs"]))
            fine["rise"].append(np.full(n, i, np.int32))
        records.append(
            {k: v for k, v in rec.items() if k not in ("fine_times", "fine_freqs")}
        )
    cat = {
        k: np.concatenate(v) if v else np.zeros(0, dt)
        for (k, v), dt in zip(
            fine.items(), (np.int8, float, float, np.int32), strict=True
        )
    }
    np.savez(
        d / TRUTH_FILE,
        times=times,
        freqs=new,
        freqs_detected=new_det,
        rise=flag,
        rise_id=rise_id,
        fine_fish=cat["fish"],
        fine_times=cat["times"],
        fine_freqs=cat["freqs"],
        fine_rise=cat["rise"],
    )
    (d / "rises.json").write_text(json.dumps(records, indent=1))
    status = {}
    for rec in records:
        status[rec["status"]] = status.get(rec["status"], 0) + 1
    return {
        "name": entry["name"],
        "rises": len(records),
        "status": status,
        "frames": int(flag.sum()),
        "fine": len(cat["times"]),
    }


def source_trace(entry: dict, k: int, rises: bool = True):
    """(times, frequency) of fish `k` of a source, with the 10-ms trace
    through its corrected onsets in place of the frames there."""
    d = LIB / entry["name"]
    if not rises:
        tr = np.load(d / "truth.npz")
        return tr["times"], tr["freqs"][k]
    tr = np.load(d / TRUTH_FILE)
    t, f = tr["times"], tr["freqs"][k]
    m = tr["fine_fish"] == k
    if not m.any():
        return t, f
    ft, ff = tr["fine_times"][m], tr["fine_freqs"][m]
    drop = tr["rise"][k]  # the frames between the anchors
    tt = np.concatenate([t[~drop], ft])
    fv = np.concatenate([f[~drop], ff])
    o = np.argsort(tt, kind="stable")
    return tt[o], fv[o]


def rise_intervals(entry: dict, k: int) -> list[tuple[float, float]]:
    """Source-time stretches (t_a, t_b) between the anchors of the corrected
    onsets of fish `k` (the frames inside are new)."""
    recs = json.loads((LIB / entry["name"] / "rises.json").read_text())
    return [
        (r["t_a"], r["t_b"])
        for r in recs
        if r["status"] == "corrected" and r["fish"] == k
    ]


# --- mixtures -------------------------------------------------------------------


def mixture_freqs(meta: dict, library: list[dict], rises: bool = True) -> dict:
    """Frequencies of the truth of a mixture rebuilt from its source truths
    with the stored warps (``mixture.warp_truth``, as ``mixture.generate``
    does): ``freqs``, ``freqs_filled``, ``freqs_detected`` and, with
    `rises`, ``rise`` (fish, samples): the samples whose source time lies
    between the anchors of a corrected onset. Without `rises` this reproduces
    the stored ``truth.npz``."""
    import mixture as mx

    by_name = {e["name"]: e for e in library}
    t_truth = np.arange(0.0, float(meta["design"]["duration"]), 1 / mx.TRUTH_RATE)
    shape = (2 * len(meta["sources"]), len(t_truth))
    out = {
        "freqs": np.full(shape, np.nan),
        "freqs_filled": np.full(shape, np.nan),
        "freqs_detected": np.full(shape, np.nan),
        "rise": np.zeros(shape, bool),
    }
    for s, (src, w) in enumerate(
        zip(meta["sources"], mx.source_warps(meta), strict=True)
    ):
        e = by_name[src["trial"]]
        tq = w.tau(t_truth)
        z = np.load(LIB / e["name"] / (TRUTH_FILE if rises else "truth.npz"))
        for k in range(2):
            j = 2 * s + k
            st, sf = source_trace(e, k, rises)
            out["freqs"][j], out["freqs_filled"][j] = mx.warp_truth(w, st, sf, t_truth)
            if rises:
                new = np.zeros(len(t_truth), bool)
                for t_a, t_b in rise_intervals(e, k):
                    new |= (tq > t_a) & (tq < t_b)
                out["rise"][j] = new & np.isfinite(out["freqs"][j])
            on = mx._detected_at(z["times"], z["freqs_detected"][k], tq)
            out["freqs_detected"][j] = np.where(
                on & ~out["rise"][j], out["freqs"][j], np.nan
            )
    return out


def snr_frames_at(rise: np.ndarray, t_truth: np.ndarray, snr_t: np.ndarray):
    """SNR frames (fish, snr frames) whose 1.64-s window holds a rise sample."""
    import mixture as mx

    half = mx.SNR_NFFT / RATE / 2
    bad = np.zeros((len(rise), len(snr_t)), bool)
    for j in range(len(rise)):
        if rise[j].any():
            tf = t_truth[rise[j]]
            bad[j] = np.searchsorted(tf, snr_t + half) > np.searchsorted(
                tf, snr_t - half
            )
    return bad


def _interp_db(t, v, bad):
    """Power `v` on the `bad` samples interpolated in dB from the others."""
    out = v.astype(np.float64).copy()
    good = ~bad & np.isfinite(v) & (v > 0)
    if bad.any() and good.sum() >= 2:
        out[bad] = 10 ** np.interp(t[bad], t[good], np.log10(v[good]))
    return out


def rebuild_mixture(name: str, library: list[dict]) -> dict:
    """``truth_rises.npz`` of a lab mixture and of its field variant: their
    ``truth.npz`` with the frequencies rebuilt from the corrected source
    truths, the ``rise`` flags, and on SNR frames that see a corrected onset
    (``rise_snr``) the fish and noise power interpolated in dB."""
    import mixture as mx

    split, rest = name.split("/", 1)
    lab = mx.MIXTURES / name
    field = mx.MIXTURES / f"{split}-field" / rest
    meta = json.loads((lab / "mixture.json").read_text())
    new = mixture_freqs(meta, library)
    dirs = [lab] + ([field] if (field / "band.yaml").exists() else [])
    stats = {"name": name, "fish": int(new["rise"].shape[0])}
    for d in dirs:
        old = dict(np.load(d / "truth.npz"))
        if old["freqs"].shape != new["freqs"].shape:
            raise ValueError(f"{d}: truth of another shape")
        bad = snr_frames_at(new["rise"], old["times"], old["snr_times"])
        tr = old | new | {"rise_snr": bad}
        fp, npw = old["fish_power"].copy(), old["noise_power"].copy()
        for j in np.flatnonzero(bad.any(1)):
            fp[j] = _interp_db(old["snr_times"], old["fish_power"][j], bad[j])
            npw[j] = _interp_db(old["snr_times"], old["noise_power"][j], bad[j])
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = 10 * np.log10(fp / npw)
        tr["fish_power"] = fp.astype(np.float32)
        tr["noise_power"] = npw.astype(np.float32)
        tr["snr_db"] = np.where(bad, snr, old["snr_db"])
        tmp = d / (TRUTH_FILE + ".tmp")
        with open(tmp, "wb") as fh:  # atomic: runs may read it meanwhile
            np.savez(fh, **tr)
        tmp.replace(d / TRUTH_FILE)
        if d == lab:
            known_old = np.isfinite(old["freqs"])
            known = np.isfinite(new["freqs"])
            diff = np.abs(new["freqs"] - old["freqs"])
            rise = new["rise"]
            stats |= {
                "known_frames": int(known.sum()),
                "known_frames_old": int(known_old.sum()),
                "rise_frames": int(rise.sum()),
                "rise_was_unknown": int((rise & ~known_old).sum()),
                "rise_off_2hz": int((rise & known_old & (diff > 2.0)).sum()),
                "changed_outside": int(
                    (~rise & ((known != known_old) | (known_old & (diff > 1e-6)))).sum()
                ),
                "snr_frames_interpolated": int(bad.sum()),
                "rise_snr_db_median": float(np.median(tr["snr_db"][bad]))
                if bad.any()
                else np.nan,
            }
    stats["variants"] = len(dirs)
    return stats


def check_rebuild(name: str, library: list[dict]) -> float:
    """Largest difference [Hz] between the stored truth of a mixture and the
    one ``mixture_freqs`` rebuilds from the old source truths (0 expected)."""
    import mixture as mx

    old = np.load(mx.MIXTURES / name / "truth.npz")
    meta = json.loads((mx.MIXTURES / name / "mixture.json").read_text())
    re = mixture_freqs(meta, library, rises=False)
    worst = 0.0
    for key in ("freqs", "freqs_filled", "freqs_detected"):
        a, b = old[key], re[key]
        if not np.array_equal(np.isfinite(a), np.isfinite(b)):
            return np.inf
        if np.isfinite(a).any():
            worst = max(worst, float(np.nanmax(np.abs(a - b))))
    return worst


# --- figures ----------------------------------------------------------------


def _spec(x, t0, nfft, hop):
    from scipy.signal import stft

    p = 0
    for c in range(x.shape[1]):
        f, tt, z = stft(
            x[:, c],
            RATE,
            window="hann",
            nperseg=nfft,
            noverlap=nfft - hop,
            boundary=None,
            padded=False,
        )
        p = p + np.abs(z) ** 2
    return f, tt + t0, 10 * np.log10(p + 1e-30)


def plot_onsets(entry: dict, records: list[dict], path: Path, n: int = 6):
    """Old and new truth over the source's spectrogram (wavetracker's 1.64-s
    window and a 0.2-s window) at its largest corrected onsets."""
    import matplotlib.pyplot as plt

    d = LIB / entry["name"]
    trial = Trial(**{k: entry[k] for k in Trial.__dataclass_fields__})
    old = np.load(d / "truth.npz")
    new = np.load(d / TRUTH_FILE)
    t = old["times"]
    recs = [r for r in records if r["status"] == "corrected"]
    recs = sorted(recs, key=lambda r: -r.get("size_if", r["size"]))[:n]
    if not recs:
        return None
    fig, axs = plt.subplots(
        2, len(recs), figsize=(3.6 * len(recs), 7), squeeze=False, layout="constrained"
    )
    for j, r in enumerate(recs):
        k = r["fish"]
        t0, t1 = r["t_step"] - 2.5, r["t_step"] + 4.5
        x, xs = read_segment(trial, t0 - 1, t1 + 1, entry["channels_used"])
        lo, hi = r["band"]
        lo = max(lo, np.nanmin(new["freqs"][k][(t >= t0) & (t <= t1)]) - 6)
        hi = min(hi, r.get("if_max", hi) + 8)
        for i, (nfft, hop, lab) in enumerate(
            ((2**15, 3276, "1.64-s window"), (2**12, 200, "0.20-s window"))
        ):
            ax = axs[i, j]
            ff, tt, db = _spec(x, xs, nfft, hop)
            m = (ff >= lo) & (ff <= hi)
            tm = (tt >= t0) & (tt <= t1)
            dd = db[m][:, tm]
            ax.pcolormesh(
                tt[tm] - r["t_step"],
                ff[m],
                dd,
                cmap="gray_r",
                vmin=np.percentile(dd, 30),
                vmax=dd.max(),
                shading="auto",
                rasterized=True,
            )
            w = (t >= t0) & (t <= t1)
            fl = new["rise"][k] & w
            mf = new["fine_rise"] == r["id"]
            ax.plot(
                new["fine_times"][mf] - r["t_step"],
                new["fine_freqs"][mf],
                "-",
                lw=0.8,
                color="#d03b3b",
            )
            ax.plot(
                t[w] - r["t_step"],
                old["freqs"][k][w],
                "o",
                ms=3.5,
                mfc="none",
                color="#2a78d6",
                label="old truth",
            )
            ax.plot(
                t[fl] - r["t_step"],
                new["freqs"][k][fl],
                "o",
                ms=3.5,
                color="#d03b3b",
                label="new frames",
            )
            ax.set_ylim(lo, hi)
            ax.set_xlim(-2.5, 4.5)
            ax.set_title(
                f"{entry['name']} fish {k} at {r['t_step']:.1f} s\n"
                f"{lab}; {len(r['frames'])} frames new",
                fontsize=7,
                loc="left",
            )
            ax.tick_params(labelsize=7)
    axs[0, 0].legend(fontsize=7, loc="upper left")
    axs[1, 0].set_xlabel("time from the climb [s]", fontsize=8)
    axs[1, 0].set_ylabel("frequency [Hz]", fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# --- command line -----------------------------------------------------------


def dev_sources(library: list[dict]) -> list[dict]:
    """Library sources of the dev pool (ltrain, ldev, ldev2); test sources
    are not touched."""
    return [e for e in library if e["ok"] and e["split"] == "dev"]


def cmd_sources(args):
    library = json.loads((DATA / "library.json").read_text())
    todo = [e for e in dev_sources(library) if not args.only or e["name"] in args.only]
    rows = []
    for i, e in enumerate(todo):
        row = complete_source(e)
        rows.append(row)
        print(
            f"[{i + 1}/{len(todo)}] {e['name']}: {row['rises']} rises, "
            f"{row['status']}, {row['frames']} frames new",
            flush=True,
        )
        if args.plot:
            recs = json.loads((LIB / e["name"] / "rises.json").read_text())
            p = plot_onsets(e, recs, FIG / f"onsets_{e['name']}.png")
            if p:
                print(p)
    if not args.only:
        (FIG.parent / "risetruth_sources.json").write_text(json.dumps(rows, indent=1))


def cmd_mixtures(args):
    import mixture as mx

    library = json.loads((DATA / "library.json").read_text())
    rows = []
    for split in args.splits:
        if split in ("test", "test-field"):
            raise SystemExit("the test split is not touched")
        for p in sorted((mx.MIXTURES / split).glob("*/band.yaml")):
            name = f"{split}/{p.parent.name}"
            if args.only and not any(o in name for o in args.only):
                continue
            if args.check:
                print(
                    f"{name}: rebuilt old truth differs by {check_rebuild(name, library):.2e} Hz",
                    flush=True,
                )
                continue
            st = rebuild_mixture(name, library)
            rows.append(st)
            print(
                f"{name}: {st['rise_frames']} rise frames ({st['rise_was_unknown']} "
                f"unknown before, {st['rise_off_2hz']} > 2 Hz off), "
                f"changed outside {st['changed_outside']}",
                flush=True,
            )
    if args.check:
        return
    out = FIG.parent / f"risetruth_mixtures_{'_'.join(args.splits)}.json"
    out.write_text(json.dumps(rows, indent=1))
    print(out)


def _records(library: list[dict]) -> list[dict]:
    from mixture import POOL

    out = []
    for e in dev_sources(library):
        split = next(
            (
                sp
                for sp, (_, g) in POOL.items()
                if g and e["group"] in g and sp != "ldev2"
            ),
            "dev",
        )
        for r in json.loads((LIB / e["name"] / "rises.json").read_text()):
            out.append(r | {"source": e["name"], "pool": split})
    return out


def plot_validation(library: list[dict], path: Path):
    """What happened to the rise onsets of the dev sources, how well the
    instantaneous frequency agrees with the old truth where that was
    detected, and what the old truth missed."""
    import matplotlib.pyplot as plt

    recs = _records(library)
    statuses = [
        "corrected",
        "traced",
        "unreliable",
        "no-anchor",
        "too-long",
        "disagrees",
        "other-fish",
        "overlap",
    ]
    fig, axs = plt.subplots(1, 4, figsize=(18, 4), layout="constrained")
    ax = axs[0]
    for i, (pool, color) in enumerate((("ltrain", "#2a78d6"), ("ldev", "#eb6834"))):
        n = [
            sum(r["status"] == st and r["pool"] == pool for r in recs)
            for st in statuses
        ]
        ax.barh(
            np.arange(len(statuses)) + 0.2 * (1 - 2 * i),
            n,
            0.4,
            color=color,
            label=f"{pool} sources ({sum(n)} onsets)",
        )
    ax.set_yticks(range(len(statuses)), statuses)
    ax.invert_yaxis()
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("rise onsets of the dev sources", loc="left", fontsize=10)
    ax = axs[1]
    dev = np.array([r["median_dev_outside"] for r in recs if "median_dev_outside" in r])
    agree = np.array([r["agree_outside"] for r in recs if "median_dev_outside" in r])
    ax.hist(np.clip(dev, 0.005, 10), bins=np.geomspace(0.005, 10, 40), color="#2a78d6")
    ax.set_xscale("log")
    ax.axvline(AGREE_HZ, color="#d03b3b", lw=1)
    ax.set_title(
        "instantaneous frequency - old detected truth,\n"
        f"outside the stretches: median per onset {np.median(dev):.2f} Hz,\n"
        f"frames within {AGREE_HZ:g} Hz {np.mean(agree):.2f}",
        loc="left",
        fontsize=9,
    )
    ax.set_xlabel("|difference| [Hz]")
    ax = axs[2]
    cor = [r for r in recs if r["status"] == "corrected"]
    ax.plot(
        [r["size"] for r in cor],
        [r["size_if"] for r in cor],
        "o",
        ms=3,
        color="#2a78d6",
        alpha=0.7,
    )
    lim = max(max(r["size_if"] for r in cor), max(r["size"] for r in cor)) + 2
    ax.plot([0, lim], [0, lim], color="#52514e", lw=0.8)
    ax.set_xlabel("rise size in the old truth [Hz]")
    ax.set_ylabel("rise size, instantaneous frequency [Hz]")
    ax.set_title(
        "corrected onsets: rise size\n(the old truth misses peaks)",
        loc="left",
        fontsize=9,
    )
    ax = axs[3]
    n_fr = [len(r["frames"]) for r in cor]
    ax.hist(n_fr, bins=np.arange(0.5, max(n_fr) + 1.5), color="#2a78d6")
    ax.set_xlabel("new frames per corrected onset (0.164 s each)")
    ax.set_title(
        f"{len(cor)} onsets corrected, {sum(n_fr)} frames;\nold truth there: "
        f"{sum(r['old_completed'] for r in cor)} completed, "
        f"{sum(r['old_unknown'] for r in cor)} unknown,\n"
        f"{sum(r['old_off_2hz'] for r in cor)} more than 2 Hz off",
        loc="left",
        fontsize=9,
    )
    for a in axs:
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(labelsize=8)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_mixture_onsets(name: str, path: Path, n: int = 4):
    """Old and new truth of a mixture over its own spectrogram (wavetracker's
    resolution) at its largest corrected onsets."""
    import matplotlib.pyplot as plt
    import mixture as mx
    from common import MIX_CHANNELS

    d = mx.MIXTURES / name
    old = np.load(d / "truth.npz")
    new = np.load(d / TRUTH_FILE)
    t = new["times"]
    picks = []
    for j in range(len(new["rise"])):
        r = new["rise"][j]
        if not r.any():
            continue
        idx = np.flatnonzero(r)
        for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            change = np.nanmax(np.abs(new["freqs"][j, run] - old["freqs"][j, run]))
            picks.append((change, j, run))
    picks = sorted(picks, key=lambda p: -p[0])[:n]
    if not picks:
        return None
    raw = np.memmap(d / "traces-grid1.raw", np.float32, mode="r").reshape(
        -1, MIX_CHANNELS
    )
    fig, axs = plt.subplots(
        1, len(picks), figsize=(4 * len(picks), 4), squeeze=False, layout="constrained"
    )
    for ax, (_, j, run) in zip(axs[0], picks, strict=True):
        tc = t[run].mean()
        t0, t1 = tc - 4, tc + 6
        i0 = max(0, int((t0 - 1) * RATE))
        x = np.asarray(raw[i0 : int((t1 + 1) * RATE)], np.float64)
        ff, tt, db = _spec(x, i0 / RATE, 2**15, 2000)
        w = (t >= t0) & (t <= t1)
        f_all = np.concatenate([new["freqs"][j, w], old["freqs"][j, w]])
        lo, hi = np.nanmin(f_all) - 4, np.nanmax(f_all) + 6
        m = (ff >= lo) & (ff <= hi)
        tm = (tt >= t0) & (tt <= t1)
        dd = db[m][:, tm]
        ax.pcolormesh(
            tt[tm] - tc,
            ff[m],
            dd,
            cmap="gray_r",
            vmin=np.percentile(dd, 30),
            vmax=dd.max(),
            shading="auto",
            rasterized=True,
        )
        ax.plot(
            t[w] - tc, old["freqs"][j, w], ".", ms=4, color="#2a78d6", label="old truth"
        )
        fl = w & new["rise"][j]
        ax.plot(
            t[fl] - tc,
            new["freqs"][j, fl],
            ".",
            ms=4,
            color="#d03b3b",
            label="new rise frames",
        )
        ax.set_ylim(lo, hi)
        ax.set_title(f"{name} fish {j}", loc="left", fontsize=8)
        ax.tick_params(labelsize=7)
    axs[0, 0].legend(fontsize=7, loc="upper left")
    axs[0, 0].set_xlabel("time [s]", fontsize=8)
    axs[0, 0].set_ylabel("frequency [Hz]", fontsize=8)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def rise_windows(name: str, n: int, min_change: float = 4.0, seed: int = 0):
    """Up to `n` corrected onsets of a mixture (fish, centre time), chosen
    from those whose truth changed by at least `min_change` Hz, at random
    (not by any run's result)."""
    import mixture as mx

    d = mx.MIXTURES / name
    old = np.load(d / "truth.npz")
    new = np.load(d / TRUTH_FILE)
    t = new["times"]
    cands = []
    for j in range(len(new["rise"])):
        idx = np.flatnonzero(new["rise"][j])
        if len(idx) == 0:
            continue
        for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            change = np.nanmax(np.abs(new["freqs"][j, run] - old["freqs"][j, run]))
            tc = float(t[run].mean())
            if change >= min_change and 5.0 < tc < t[-1] - 6.0:  # whole window
                cands.append((j, tc))
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(cands), size=min(n, len(cands)), replace=False)
    return [cands[i] for i in sorted(pick)]


def plot_rise_runs(name: str, tags: list[str], path: Path, n: int = 4, seed: int = 0):
    """Detections of runs (rows) at corrected onsets of a mixture (columns)
    over its spectrogram, with the corrected truth: identities in colours,
    the fish's own (IDF1) identity in blue, detections that match no fish in
    red, the new rise frames as red rings."""
    import matplotlib.pyplot as plt
    import mixture as mx
    from common import MIX_CHANNELS
    from score import match, truth_at

    from wavetracker.results import Results

    d = mx.MIXTURES / name
    new = dict(np.load(d / TRUTH_FILE))
    wins = rise_windows(name, n, seed=seed)
    if not wins:
        return None
    raw = np.memmap(d / "traces-grid1.raw", np.float32, mode="r").reshape(
        -1, MIX_CHANNELS
    )
    runs = {}
    for tag in tags:
        res = Results.load(DATA / "runs" / tag / name)
        F = truth_at(new, res.times)
        fish = match(res, F)
        ids = np.unique(res.ident_v[np.isfinite(res.ident_v)])
        counts = np.zeros((len(ids), len(F)), int)
        m = (fish >= 0) & np.isfinite(res.ident_v)
        np.add.at(counts, (np.searchsorted(ids, res.ident_v[m]), fish[m]), 1)
        from scipy.optimize import linear_sum_assignment

        r, c = linear_sum_assignment(-counts)
        own = {int(cc): ids[rr] for rr, cc in zip(r, c, strict=True) if counts[rr, cc]}
        runs[tag] = (res, fish, own)
    other = ["#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#52514e"]
    fig, axs = plt.subplots(
        len(tags),
        len(wins),
        figsize=(3.8 * len(wins), 3.4 * len(tags)),
        squeeze=False,
        layout="constrained",
    )
    t = new["times"]
    for col, (j, tc) in enumerate(wins):
        t0, t1 = tc - 4, tc + 5
        i0 = max(0, int((t0 - 1) * RATE))
        x = np.asarray(raw[i0 : int((t1 + 1) * RATE)], np.float64)
        ff, tt, db = _spec(x, i0 / RATE, 2**15, 1000)
        w = (t >= t0) & (t <= t1)
        lo = np.nanmin(new["freqs_filled"][j, w]) - 5
        hi = np.nanmax(new["freqs_filled"][j, w]) + 5
        fm = (ff >= lo) & (ff <= hi)
        tm = (tt >= t0) & (tt <= t1)
        dd = db[fm][:, tm]
        for row, tag in enumerate(tags):
            ax = axs[row, col]
            res, fish, own = runs[tag]
            ax.pcolormesh(
                tt[tm] - tc,
                ff[fm],
                dd,
                cmap="gray_r",
                vmin=np.percentile(dd, 30),
                vmax=dd.max(),
                shading="auto",
                rasterized=True,
            )
            ax.plot(t[w] - tc, new["freqs"][j, w], "-", lw=0.6, color="#0b0b0b")
            fl = w & new["rise"][j]
            ax.plot(
                t[fl] - tc, new["freqs"][j, fl], "o", ms=5, mfc="none", color="#d03b3b"
            )
            td = res.times[res.idx_v]
            sel = (td >= t0) & (td <= t1) & (res.fund_v >= lo) & (res.fund_v <= hi)
            ids = res.ident_v[sel]
            for k, i in enumerate(np.unique(ids[np.isfinite(ids)])):
                mi = sel & (res.ident_v == i)
                color = "#2a78d6" if own.get(j) == i else other[k % len(other)]
                ax.plot(td[mi] - tc, res.fund_v[mi], ".", ms=3.5, color=color)
            un = sel & (fish < 0)
            ax.plot(td[un] - tc, res.fund_v[un], "x", ms=3, color="#d03b3b")
            ax.set_ylim(lo, hi)
            ax.set_xlim(-4, 5)
            fr = res.idx_v[(fish == j) & (td >= t0) & (td <= t1)]
            rt = truth_at(
                {"times": t, "rise": new["rise"].astype(float)}, res.times, "rise"
            )[j]
            hit = int((rt[fr] > 0).sum())
            n_r = int(((rt > 0) & (res.times >= t0) & (res.times <= t1)).sum())
            ax.set_title(
                f"{tag}\n{name} fish {j}, rise frames found {hit}/{n_r}",
                loc="left",
                fontsize=7,
            )
            ax.tick_params(labelsize=7)
    axs[-1, 0].set_xlabel("time [s]", fontsize=8)
    axs[-1, 0].set_ylabel("frequency [Hz]", fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def cmd_runs(args):
    for name in args.mixtures:
        stem = "_vs_".join(args.tags) + "_" + name.replace("/", "_")
        p = plot_rise_runs(name, args.tags, FIG / f"runs_{stem}.png", args.n, args.seed)
        if p:
            print(p)


def cmd_report(args):
    library = json.loads((DATA / "library.json").read_text())
    FIG.mkdir(parents=True, exist_ok=True)
    print(plot_validation(library, FIG / "validation.png"))
    for name in args.mixtures:
        p = plot_mixture_onsets(name, FIG / f"mixture_{name.replace('/', '_')}.png")
        if p:
            print(p)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("sources", help="correct the rise onsets of the dev sources")
    a.add_argument("--only", nargs="*")
    a.add_argument("--plot", action="store_true")
    a.set_defaults(func=cmd_sources)
    b = sub.add_parser("mixtures", help="rebuild truth_rises.npz of mixtures")
    b.add_argument("--splits", nargs="+", default=["ltrain", "ldev", "ldev2"])
    b.add_argument("--only", nargs="*")
    b.add_argument(
        "--check",
        action="store_true",
        help="only check that the old truth.npz is reproduced from the old sources",
    )
    b.set_defaults(func=cmd_mixtures)
    c = sub.add_parser("report", help="validation figures")
    c.add_argument(
        "--mixtures",
        nargs="*",
        default=[
            "ldev/density-10-s0",
            "ldev/lowfreq-30-s1",
            "ldev-field/atten-fish-30-s0",
        ],
    )
    c.set_defaults(func=cmd_report)
    r = sub.add_parser("runs", help="runs at corrected onsets, side by side")
    r.add_argument("--tags", nargs="+", required=True)
    r.add_argument("--mixtures", nargs="+", required=True)
    r.add_argument("--n", type=int, default=4)
    r.add_argument("--seed", type=int, default=0)
    r.set_defaults(func=cmd_runs)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
