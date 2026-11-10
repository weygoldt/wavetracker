"""Classic and learned detector side by side on real field recordings.

There is no truth here and nothing is scored: the figures are for judging by
eye whether the learned detector (trained on lab tube mixtures plus field
noise, see docs/benchmarks.md "Learned detector (v1)") transfers to real field
fish. The two runs of a segment differ only in the `detector` section of the
config, and both panels of a figure show the same spectrogram (the classic
run's stored fine spectrogram; it does not depend on the detector).

Recordings:

* the 16-electrode logger grid iriri01-grid01-dev03 (48 kHz, 5-min WavPack
  files, decoded with ffmpeg; the venv's audioio cannot read them). Its day
  is fish-free; the day-2 hours 10:48-15:38 (files 282-340) were training
  noise of the learned detector and are refused here.
* the 2-electrode boat take DR0000_0209 (48 kHz WAV, about 40 fish in a
  chorus, moving electrodes). DR0000_0208 and its corrections are never used.

Windows of the figures are chosen from the spectrogram alone (the band of
`--band` Hz with the most spectral lines in the window), never from the runs.

    python benchmarks/field_check.py survey     # spectral lines over the night -> figure
    python benchmarks/field_check.py convert    # chosen .wv files -> 16-channel WAV
    python benchmarks/field_check.py run        # detect + track, both detectors
    python benchmarks/field_check.py plot       # side-by-side figures + one PDF
    python benchmarks/field_check.py counts     # counts per segment and detector
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

GRID = Path(
    os.environ.get(
        "WT_FIELD_GRID",
        "/run/media/weygoldt/data1/efish/fielddata/iriri2026/iriri2026_recordings/"
        "iriri01_lago-de-pedro/grid01/iriri01-grid01-dev03-20260905T1118",
    )
)
BOAT = Path(
    os.environ.get(
        "WT_FIELD_BOAT",
        "/run/media/weygoldt/data1/efish/fielddata/iriri2026/experiments/"
        "wavefish_habitatmapping/wav/DR0000_0209.wav",
    )
)
GRID_RATE = 48000.0
GRID_CHANNELS = 16
TRAINING_NOISE = range(282, 341)  # grid files that were training noise: refused
DATA = Path(
    os.environ.get("WT_FIELD_CHECK", "~/wrk/data/wavetracker_benchmark/field_check")
).expanduser()
FIG = Path("~/wrk/data/wavetracker_benchmark/figures/field_side_by_side").expanduser()
DETECTORS = ("classic", "learned")

# Grid files evenly spaced over the night with fish (19:18-05:00 by the survey):
# 20:18, 22:58, 01:38 and 04:18. The boat take is the extra recording.
SEGMENTS = {
    "grid01_f108_2018": {"recording": "grid", "file": 108},
    "grid01_f140_2258": {"recording": "grid", "file": 140},
    "grid01_f172_0138": {"recording": "grid", "file": 172},
    "grid01_f204_0418": {"recording": "grid", "file": 204},
    "boat_DR0000_0209": {"recording": "boat"},
}

# Base configs: docs/field-recordings.md (battery powered: no mains, no
# interference removal; no absolute power limit; two harmonics suffice), the
# learned linker as in benchmarks/dense/configs/dense-linked.yaml (the default,
# it replaces the doc's `stitching` settings). The spectrogram stays at the
# default 0.61 Hz: the learned model needs it, and both detectors get the same.
# Grid: threshold factors 4 and 7 ("a good start for grids"), the electrode
# pattern as min-max (16 electrodes), 20-s blocks for the GPU. Boat take: the
# doc's 3 and 5 and dB patterns for two electrodes.
_COMMON = {
    "interference": {"enabled": False},
    "harmonic_groups": {
        "mains_freq": 0.0,
        "min_group_size": 2,
        "max_harmonics": 10,
        "min_good_peak_power": None,
        "max_missing_harmonics": 0,
    },
    "tracking": {"min_support": 3, "gap_tolerance": 0.5, "gap_tolerance_rate": 0.1},
    "linking": {"enabled": True},
    "output": {"save_fine_spec": True, "fine_spec_max_freq": 2050.0},
}
BASE = {
    "grid": {
        "spectrogram": {"block_duration": 20.0},
        "harmonic_groups": {
            "min_freq": 200.0,
            "max_freq": 1500.0,
            "low_thresh_factor": 4.0,
            "high_thresh_factor": 7.0,
        },
    },
    "boat": {
        "harmonic_groups": {
            "min_freq": 80.0,
            "max_freq": 2000.0,
            "low_thresh_factor": 3.0,
            "high_thresh_factor": 5.0,
        },
        "tracking": {"amplitude_feature": "db"},
    },
}


def config_dict(recording: str, detector: str) -> dict:
    d = json.loads(json.dumps(_COMMON))
    for sec, vals in BASE[recording].items():
        d.setdefault(sec, {}).update(vals)
    d["detector"] = {"method": detector}
    return {k: d[k] for k in sorted(d, key=_SECTION_ORDER.index)}


_SECTION_ORDER = [
    "spectrogram",
    "detector",
    "clicks",
    "interference",
    "harmonic_groups",
    "tracking",
    "stitching",
    "linking",
    "output",
]


def config_path(recording: str, detector: str) -> Path:
    return DATA / "configs" / f"{recording}-{detector}.yaml"


def write_configs() -> None:
    import yaml

    for rec in BASE:
        for det in DETECTORS:
            p = config_path(rec, det)
            p.parent.mkdir(parents=True, exist_ok=True)
            head = (
                f"# field_check: {rec}, {det} detector. The {rec} configs differ "
                "only in `detector`.\n"
            )
            p.write_text(head + yaml.safe_dump(config_dict(rec, det), sort_keys=False))


def segment_input(name: str) -> Path:
    seg = SEGMENTS[name]
    if seg["recording"] == "boat":
        if "0208" in BOAT.name:
            raise ValueError("DR0000_0208 is not to be used")
        return BOAT
    return DATA / "wav" / f"{name}.wav"


def run_dir(name: str, detector: str) -> Path:
    return DATA / "runs" / name / detector


def grid_files() -> list[Path]:
    files = sorted(GRID.glob("*.wv"))
    if not files:
        raise FileNotFoundError(f"no .wv files in {GRID} (set WT_FIELD_GRID)")
    return files


def grid_file(index: int) -> Path:
    if index in TRAINING_NOISE:
        raise ValueError(
            f"grid file {index} was training noise of the learned detector"
        )
    return grid_files()[index]


def clock(path: Path) -> datetime:
    m = re.search(r"(\d{8}T\d{6})", path.name)
    return datetime.strptime(m.group(1), "%Y%m%dT%H%M%S")


def decode(path: Path, start: float = 0.0, duration: float | None = None):
    """(samples, 16) float32 of a WavPack file, through ffmpeg (as noise.py)."""
    cmd = ["ffmpeg", "-v", "error"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    if duration:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-i", str(path), "-f", "f32le", "-acodec", "pcm_f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, GRID_CHANNELS)


# --------------------------------------------------------------------- survey


def line_excess(power: np.ndarray, df: float, width_hz: float = 20.0):
    """dB above the running median over `width_hz` of a spectrum."""
    from scipy.ndimage import median_filter

    db = 10 * np.log10(power + 1e-30)
    k = round(width_hz / df) | 1
    return db - median_filter(db, k, mode="nearest")


def count_lines(excess, freqs, f0, f1, height=3.0, min_sep_hz=1.5):
    from scipy.signal import find_peaks

    band = (freqs >= f0) & (freqs < f1)
    peaks, _ = find_peaks(
        excess[band], height=height, distance=max(1, round(min_sep_hz / freqs[1]))
    )
    return freqs[band][peaks]


def survey(args):
    """Mean spectra of 30 s in the middle of every n-th file of the night:
    how many lines stand above the local floor (fish, harmonics, hum)."""
    import matplotlib.pyplot as plt

    files = grid_files()
    idx = [i for i in range(*args.files) if i not in TRAINING_NOISE and i < len(files)]
    nfft = 78732  # 0.61 Hz at 48 kHz
    win = np.hanning(nfft).astype(np.float32)
    freqs = np.fft.rfftfreq(nfft, 1 / GRID_RATE)
    fmax = np.searchsorted(freqs, 2100)
    rows, counts, labels = [], [], []
    for i in idx:
        x = decode(files[i], start=135.0, duration=30.0)
        n = (len(x) - nfft) // nfft + 1
        p = np.zeros(fmax)
        for c in range(GRID_CHANNELS):
            for j in range(n):
                seg = x[j * nfft : (j + 1) * nfft, c] * win
                p += np.abs(np.fft.rfft(seg)[:fmax]) ** 2
        ex = line_excess(p / n, freqs[1])
        lines = count_lines(ex, freqs[:fmax], *args.band)
        rows.append(ex)
        counts.append(len(lines))
        labels.append(clock(files[i]).strftime("%d. %H:%M"))
        print(
            f"file {i:3d} {labels[-1]}  {len(lines):3d} lines in "
            f"{args.band[0]:.0f}-{args.band[1]:.0f} Hz: "
            + " ".join(f"{f:.0f}" for f in lines[:25]),
            flush=True,
        )
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "survey.json").write_text(
        json.dumps(
            {
                "files": idx,
                "clock": labels,
                "lines": counts,
                "band": list(args.band),
            },
            indent=1,
        )
    )
    fig, axs = plt.subplots(
        1,
        2,
        figsize=(15, 7),
        layout="constrained",
        sharey=True,
        gridspec_kw={"width_ratios": [4, 1]},
    )
    ax = axs[0]
    ax.pcolormesh(
        freqs[:fmax],
        np.arange(len(idx)),
        np.clip(np.array(rows), 0, 12),
        cmap="gray_r",
        shading="nearest",
        rasterized=True,
    )
    ax.set_yticks(
        np.arange(len(idx)),
        [f"{i} ({lab})" for i, lab in zip(idx, labels, strict=True)],
    )
    ax.set_xlim(100, 2100)
    ax.set_ylim(len(idx) - 0.5, -0.5)
    ax.set_xlabel("frequency [Hz]")
    ax.set_title(
        f"{GRID.name}: dB above the 20-Hz running median, electrode sum, "
        "30 s mid-file (0-12 dB)",
        loc="left",
        fontsize=10,
    )
    ax.tick_params(labelsize=7)
    ax = axs[1]
    ax.barh(np.arange(len(idx)), counts, color="#2a78d6")
    ax.set_xlabel(f"lines > 3 dB, {args.band[0]:.0f}-{args.band[1]:.0f} Hz")
    FIG.mkdir(parents=True, exist_ok=True)
    p = FIG / "survey_grid01_night.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(p)


# ---------------------------------------------------------------- convert, run


def convert(args):
    """Grid files -> 16-channel int16 WAV (lossless: the loggers record 16 bit)."""
    out = DATA / "wav"
    out.mkdir(parents=True, exist_ok=True)
    for name in _segments(args):
        seg = SEGMENTS[name]
        if seg["recording"] != "grid":
            continue
        src, dst = grid_file(seg["file"]), segment_input(name)
        if dst.exists():
            continue
        print(f"{src.name} -> {dst}", flush=True)
        tmp = dst.with_suffix(".part.wav")
        subprocess.run(
            [
                *("ffmpeg", "-v", "error", "-y", "-i", str(src)),
                *("-c:a", "pcm_s16le", "-f", "wav", str(tmp)),
            ],
            check=True,
        )
        # the channel order must survive: compare with the raw decode
        from scipy.io import wavfile

        _, w = wavfile.read(tmp, mmap=True)
        ref = decode(src, duration=1.0)
        if not np.allclose(w[: len(ref)] / 32768.0, ref, atol=1e-6):
            raise RuntimeError(f"{tmp}: samples or channel order differ from {src}")
        tmp.rename(dst)
        (dst.with_suffix(".json")).write_text(
            json.dumps(
                {"source": str(src), "file": seg["file"], "start": str(clock(src))}
            )
        )


def run(args):
    """Detect and track every segment with both detectors."""
    import time

    from wavetracker.config import Config
    from wavetracker.pipeline import detect, track_results

    write_configs()
    for name in _segments(args):
        rec = SEGMENTS[name]["recording"]
        for det in args.detectors:
            out = run_dir(name, det)
            if (out / "ident_v.npy").exists() and not args.force:
                continue
            cfg = Config.load(config_path(rec, det))
            t0 = time.perf_counter()
            res = detect(segment_input(name), out, cfg, device=args.device).results
            t1 = time.perf_counter()
            track_results(res, cfg)
            res.save(out)
            cfg.save(out / "config.yaml")
            print(
                f"{name} {det}: {len(res.fund_v)} detections, {res.n_ids} ids; "
                f"detect {t1 - t0:.0f} s, track {time.perf_counter() - t1:.0f} s",
                flush=True,
            )


# ------------------------------------------------------------------- figures

PALETTE = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#0f8fa3",
    "#b0651a",
    "#7a9a01",
]
LABELS = {
    "classic": "classic (peaks + harmonic groups)",
    "learned": "learned v1 (threshold 0.6)",
}
LINE_DB = 6.0  # a spectrogram cell counts as a line this far above its 20-Hz median


def fish_band(name: str) -> tuple[float, float]:
    hc = BASE[SEGMENTS[name]["recording"]]["harmonic_groups"]
    return hc["min_freq"], hc["max_freq"]


def spectrogram(name: str, t0: float, t1: float, f0: float, f1: float):
    """(times, freqs, dB[freq, time]) of the stored fine spectrogram (electrode
    sum), the same for both runs."""
    from wavetracker.results import load_fine_spectrogram

    spec, freqs, times = load_fine_spectrogram(run_dir(name, "classic"))
    it = np.flatnonzero((times >= t0) & (times <= t1))
    jf = np.flatnonzero((freqs >= f0) & (freqs <= f1))
    p = np.asarray(spec[it[0] : it[-1] + 1, jf[0] : jf[-1] + 1], np.float64)
    return times[it], freqs[jf], 10 * np.log10(p.T + 1e-30)


def line_cells(db: np.ndarray, df: float) -> np.ndarray:
    """Per bin, the fraction of frames standing LINE_DB above the 20-Hz median."""
    from scipy.ndimage import median_filter

    k = round(20.0 / df) | 1
    ex = db - median_filter(db, size=(k, 1), mode="nearest")
    return (ex > LINE_DB).mean(1)


def busiest_bands(name, t0, t1, width, n=2, step=5.0):
    """The n non-overlapping bands of `width` Hz within the fish range that
    hold most spectrogram line cells in [t0, t1]; from the spectrogram alone."""
    f_lo, f_hi = fish_band(name)
    _, freqs, db = spectrogram(name, t0, t1, f_lo, f_hi)
    score = np.cumsum(np.r_[0.0, line_cells(db, freqs[1] - freqs[0])])
    starts = np.arange(f_lo, f_hi - width + step / 2, step)
    a = np.searchsorted(freqs, starts)
    b = np.searchsorted(freqs, starts + width)
    s = score[b] - score[a]
    bands = []
    for _ in range(n):
        free = [
            k
            for k in range(len(starts))
            if all(abs(starts[k] - f) >= width for f, _ in bands)
        ]
        if not free:
            break
        k = max(free, key=lambda k: s[k])
        bands.append((float(starts[k]), float(starts[k] + width)))
    return bands


def windows() -> list[tuple[str, tuple[float, float, float, float], str]]:
    """Every figure's segment and window, from the spectrogram only: the
    middle of the segment, its busiest band and the next busiest one that does
    not overlap it; for the boat take also the low band (100-350 Hz) that the
    field doc names as hard (faint fish below ~300 Hz), fixed beforehand."""
    out = []
    for name, seg in SEGMENTS.items():
        if seg["recording"] == "grid":
            dur, width, total, n = 150.0, 300.0, 300.0, 2
        else:
            dur, width, total, n = 175.0, 250.0, 349.0, 2
        t0 = round((total - dur) / 2)
        bands = busiest_bands(name, t0, t0 + dur, width, n=n)
        kinds = ["busiest band", "2nd busiest band"]
        for (f0, f1), kind in zip(bands, kinds, strict=False):
            out.append((name, (t0, t0 + dur, f0, f1), kind))
        if seg["recording"] == "boat":
            out.append((name, (t0, t0 + dur, 100.0, 350.0), "low band, fixed"))
    return out


def _tracks(ax, res, win, lw=1.3, ends=True, max_gap=2.0):
    """Identities as coloured lines (broken at gaps > max_gap, dots at their
    own starts and ends), unassigned detections as grey dots."""
    t0, t1, f0, f1 = win
    t = res.times[res.idx_v]
    inw = (t >= t0) & (t <= t1) & (res.fund_v >= f0) & (res.fund_v <= f1)
    free = inw & ~np.isfinite(res.ident_v)
    ax.plot(t[free], res.fund_v[free], ".", ms=1.6, color="#8a8a8a", rasterized=True)
    ids = np.unique(res.ident_v[inw & np.isfinite(res.ident_v)])
    first = {i: t[res.ident_v == i].min() for i in ids}
    last = {i: t[res.ident_v == i].max() for i in ids}
    for j, i in enumerate(sorted(ids, key=lambda i: first[i])):
        m = inw & (res.ident_v == i)
        o = np.argsort(t[m])
        tt, ff = t[m][o], res.fund_v[m][o]
        c = PALETTE[j % len(PALETTE)]
        gaps = np.flatnonzero(np.diff(tt) > max_gap) + 1
        for st, sf in zip(np.split(tt, gaps), np.split(ff, gaps), strict=True):
            ax.plot(st, sf, "-", lw=lw, color=c, alpha=0.85, solid_capstyle="butt")
        if ends:
            for te in (first[i], last[i]):
                if te == tt[0] or te == tt[-1]:
                    fe = ff[0] if te == tt[0] else ff[-1]
                    ax.plot(te, fe, "o", ms=2.6, color=c, mec="k", mew=0.3)
    return inw, ids, free


def _spec_image(ax, spec, lo=35, hi=99.7):
    times, freqs, db = spec
    ax.imshow(
        db,
        origin="lower",
        aspect="auto",
        cmap="gray_r",
        interpolation="nearest",
        extent=(times[0], times[-1], freqs[0], freqs[-1]),
        vmin=np.percentile(db, lo),
        vmax=np.percentile(db, hi),
        rasterized=True,
    )


def side_by_side(name, win, kind):
    import matplotlib.pyplot as plt

    from wavetracker.results import Results

    spec = spectrogram(name, *win)
    fig, axs = plt.subplots(
        1, 3, figsize=(22, 8.0), sharex=True, sharey=True, layout="constrained"
    )
    for ax, det in zip(axs, DETECTORS, strict=False):
        res = Results.load(run_dir(name, det))
        _spec_image(ax, spec)
        inw, ids, free = _tracks(ax, res, win)
        tt = res.times[res.idx_v][inw]
        long = sum(
            np.sum(inw & (res.ident_v == i)) * (res.times[1] - res.times[0]) >= 30.0
            for i in ids
        )
        ax.set_title(
            f"{LABELS[det]}\n{inw.sum()} detections here, {free.sum()} without "
            f"identity (grey)\n{len(ids)} identities, {long} with >= 30 s here; "
            f"{len(np.unique(tt)) / len(spec[0]):.0%} of frames with a detection",
            loc="left",
            fontsize=9,
        )
    times, freqs, db = spec
    _spec_image(axs[2], (times, freqs, db - np.median(db, axis=0)), lo=50)
    axs[2].set_title(
        "reference: the same spectrogram without tracks, each frame minus its\n"
        "median over the band (removes broadband noise such as clicks and rain)",
        loc="left",
        fontsize=9,
    )
    for ax in axs:
        ax.set_xlim(win[0], win[1])
        ax.set_ylim(win[2], win[3])
        ax.set_xlabel("time [s]", fontsize=9)
        ax.tick_params(labelsize=8)
    axs[0].set_ylabel("frequency [Hz]", fontsize=9)
    fig.suptitle(
        f"{_title(name)}   window: middle of the segment, {kind} "
        f"({win[0]:.0f}-{win[1]:.0f} s, {win[2]:.0f}-{win[3]:.0f} Hz); "
        "dots: an identity's start or end",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    return fig


def _title(name: str) -> str:
    seg = SEGMENTS[name]
    if seg["recording"] == "boat":
        return f"{BOAT.name} (2 electrodes, boat take)"
    src = grid_file(seg["file"])
    return f"{src.name} (grid01, 16 electrodes, file {seg['file']})"


def _short(name: str) -> str:
    seg = SEGMENTS[name]
    if seg["recording"] == "boat":
        return BOAT.stem
    return f"grid01 file {seg['file']} ({clock(grid_file(seg['file'])):%d. %H:%M})"


def overview(names: list[str], title: str):
    import matplotlib.pyplot as plt

    from wavetracker.results import Results

    fig, axs = plt.subplots(
        len(names),
        2,
        figsize=(20, max(4.6 * len(names) + 0.6, 9.0)),
        sharey=True,
        layout="constrained",
        squeeze=False,
    )
    for row, name in zip(axs, names, strict=True):
        f0, f1 = fish_band(name)
        res0 = Results.load(run_dir(name, "classic"))
        win = (0.0, float(res0.times[-1]), f0, f1)
        spec = spectrogram(name, *win)
        for ax, det in zip(row, DETECTORS, strict=True):
            res = res0 if det == "classic" else Results.load(run_dir(name, det))
            _spec_image(ax, spec, lo=40)
            inw, ids, _ = _tracks(ax, res, win, lw=0.7, ends=False)
            ax.set_xlim(win[0], win[1])
            ax.set_ylim(f0, f1)
            t, end = res.times[res.idx_v], res.times[-1]
            near = (t < EDGE_ONLY_S) | (t > end - EDGE_ONLY_S)
            n_edge = sum(near[inw & (res.ident_v == i)].all() for i in ids)
            ax.set_title(
                f"{_short(name)}, {LABELS[det]}: {inw.sum()} detections, "
                f"{len(ids)} identities ({n_edge} only in the first/last "
                f"{EDGE_ONLY_S:.0f} s)",
                loc="left",
                fontsize=9,
            )
            ax.tick_params(labelsize=8)
        row[0].set_ylabel("frequency [Hz]", fontsize=9)
    for ax in axs[-1]:
        ax.set_xlabel("time [s]", fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12)
    return fig


def plot(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    FIG.mkdir(parents=True, exist_ok=True)
    grid = [n for n, s in SEGMENTS.items() if s["recording"] == "grid"]
    boat = [n for n, s in SEGMENTS.items() if s["recording"] == "boat"]
    figs = [
        (
            "overview_grid01_night",
            lambda: overview(
                grid,
                "Overview: grid01, four 5-min files of the night 5./6. Sep "
                "(left classic, right learned; identities coloured, unassigned "
                "grey; fish range 200-1500 Hz). The learned detector bursts in "
                "the first and last ~3 s of every file (edge artifact).",
            ),
        ),
        (
            "overview_boat_DR0000_0209",
            lambda: overview(
                boat,
                "Overview: boat take DR0000_0209 (left classic, right learned; "
                "fish range 80-2000 Hz)",
            ),
        ),
    ]
    for k, (name, win, kind) in enumerate(windows()):
        tag = f"{k + 1:02d}_{name}_{win[2]:.0f}-{win[3]:.0f}Hz"
        figs.append((tag, lambda n=name, w=win, c=kind: side_by_side(n, w, c)))
    survey_png = FIG / "survey_grid01_night.png"
    with PdfPages(FIG / "field_side_by_side.pdf") as pdf:
        if survey_png.exists():  # how the grid files were chosen (`survey`)
            img = plt.imread(survey_png)
            fig = plt.figure(figsize=(img.shape[1] / 100, img.shape[0] / 100))
            fig.figimage(img)
            pdf.savefig(fig, dpi=100)
            plt.close(fig)
        for tag, make in figs:
            fig = make()
            p = FIG / f"{tag}.png"
            fig.savefig(p, dpi=args.dpi)
            pdf.savefig(fig, dpi=args.dpi)
            plt.close(fig)
            print(p, flush=True)
    print(FIG / "field_side_by_side.pdf")


# -------------------------------------------------------------------- counts


EDGE_S = 5.0  # counts leave out this much at both ends of a segment [s]
EDGE_ONLY_S = 3.0  # identities with all detections this close to an end


def counts(args):
    """Counts per segment and detector, over the segment without its first and
    last EDGE_S seconds: the learned detector bursts there (about 16 frames at
    each end of a recording, 2-10 times its usual detections per frame), which
    would otherwise dominate its identity counts. The identities living only
    in the first or last EDGE_ONLY_S seconds are counted apart."""
    from wavetracker.results import Results

    rows = []
    for name in SEGMENTS:
        f0, f1 = fish_band(name)
        for det in DETECTORS:
            res = Results.load(run_dir(name, det))
            dt = res.times[1] - res.times[0]
            t, end = res.times[res.idx_v], res.times[-1]
            inb = (res.fund_v >= f0) & (res.fund_v <= f1)
            inner = inb & (t >= EDGE_S) & (t <= end - EDGE_S)
            tracked = np.isfinite(res.ident_v)
            ids = np.unique(res.ident_v[inner & tracked])
            all_ids, n = np.unique(res.ident_v[inb & tracked], return_counts=True)
            long = set(all_ids[n * dt >= 30.0])
            near = (t < EDGE_ONLY_S) | (t > end - EDGE_ONLY_S)
            edge_only = [i for i in all_ids if near[inb & (res.ident_v == i)].all()]
            frames = (res.times >= EDGE_S) & (res.times <= end - EDGE_S)
            per_frame = np.bincount(res.idx_v[inner], minlength=len(res.times))
            rows.append(
                {
                    "segment": name,
                    "detector": det,
                    "detections": int(inner.sum()),
                    "tracked": float(tracked[inner].mean()),
                    "identities": len(ids),
                    "identities_30s": len(long & set(ids)),
                    "frames_with_detection": float(np.mean(per_frame[frames] > 0)),
                    "median_per_frame": float(np.median(per_frame[frames])),
                    "edge_only_identities": len(edge_only),
                    "fish_band_hz": [f0, f1],
                }
            )
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "counts.json").write_text(json.dumps(rows, indent=1))
    lines = [
        "| segment | detector | detections | with identity | identities "
        "| identities >= 30 s | frames with a detection | detections per frame "
        f"(median) | identities only in the first/last {EDGE_ONLY_S:.0f} s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['segment']} | {r['detector']} | {r['detections']} | "
            f"{r['tracked']:.0%} | {r['identities']} | {r['identities_30s']} | "
            f"{r['frames_with_detection']:.0%} | {r['median_per_frame']:.0f} | "
            f"{r['edge_only_identities']} |"
        )
    text = "\n".join(lines)
    (DATA / "counts.md").write_text(
        text + f"\n\nAll columns but the last: within the fish band of the config, "
        f"without the first and last {EDGE_S:.0f} s of a segment (the learned "
        "detector bursts there). Identities: with a detection there; >= 30 s: at "
        "least 30 s of detected frames over the whole segment (detections x "
        "frame step).\n"
    )
    print(text)


def _segments(args) -> list[str]:
    names = args.segments or list(SEGMENTS)
    bad = set(names) - set(SEGMENTS)
    if bad:
        raise ValueError(f"unknown segments {sorted(bad)}; known: {list(SEGMENTS)}")
    return names


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("survey", help="spectral lines over the night of the grid")
    a.add_argument("--files", type=int, nargs=3, default=[84, 244, 4], metavar="N")
    a.add_argument("--band", type=float, nargs=2, default=[150.0, 2000.0])
    a.set_defaults(func=survey)
    a = sub.add_parser("convert", help="grid files -> WAV")
    a.add_argument("--segments", nargs="+")
    a.set_defaults(func=convert)
    a = sub.add_parser("run", help="detect + track with both detectors")
    a.add_argument("--segments", nargs="+")
    a.add_argument("--detectors", nargs="+", default=list(DETECTORS))
    a.add_argument("--device", default="auto")
    a.add_argument("--force", action="store_true", help="rerun existing runs")
    a.set_defaults(func=run)
    a = sub.add_parser("plot", help="side-by-side figures and one PDF")
    a.add_argument("--dpi", type=int, default=130)
    a.set_defaults(func=plot)
    a = sub.add_parser("counts", help="counts per segment and detector")
    a.set_defaults(func=counts)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
