"""Field-noise variants of the mixtures: the same fish in real field noise.

The lab mixtures keep the noise of the lab recordings, against which the
fish have 40-60 dB. Here every lab mixture gets real fish-free field noise
added: 10 min of an empty day of a 16-electrode logger grid in the Iriri
(iriri01-grid01-dev03, fish arrive at dusk), resampled to 20 kHz, on a 3x3
block of the 4x4 grid (the noise is correlated across electrodes, as it is
in the water), dev and test from different hours of the day.

Calibration: the noise is scaled by the ratio of the EOD amplitudes of a fish
next to an electrode in both systems, so that a lab fish at 0 dB gain is as
far above the field noise as a field fish next to an electrode, and the
attenuation of the mixtures is distance. Amplitudes are wavetracker's power
at the fundamental on the best electrode (same 20 kHz, same resolution): for
the lab fish the median of the library sources, for the field the 95th
percentile of the tracked fish of 20 min of the same deployment's night.

Usage::

    python benchmarks/dense/noise.py calibrate    # -> noise/calibration.json, figure
    python benchmarks/dense/noise.py derive        # every lab mixture -> <split>-field/
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA, MIX_CHANNELS, RATE
from mixture import MIXTURES, SNR_NFFT, TRAIN_SUITE, _device, expand_suite

FIELD = Path(
    os.environ.get(
        "WT_FIELD_NOISE",
        "/run/media/weygoldt/data1/efish/fielddata/iriri2026/iriri2026_recordings/"
        "iriri01/grid01/iriri01-grid01-dev03-20260905T1118",
    )
)
FIELD_RATE = 48000.0
FIELD_CHANNELS = 16
FILE_S = 300.0
NOISE = DATA / "noise"
# files of 5 min from 11:18:37: the empty day, split into dev and test hours
DAY_DEV = (8, 46)  # 11:58-15:08
DAY_TEST = (46, 84)  # 15:08-18:18
DAY_TRAIN = (1, 47)  # 11:23-15:13, the dev hours shifted by one file (see noise_plan)
NIGHT = (96, 4)  # first file, number of files: 19:18-19:38, fish
# 3x3 blocks of the 4x4 grid (row-major channel order assumed)
BLOCKS = [
    [4 * (r0 + i) + (c0 + j) for i in range(3) for j in range(3)]
    for r0 in (0, 1)
    for c0 in (0, 1)
]


def field_files() -> list[str]:
    files = sorted(glob.glob(str(FIELD / "*.wv")))
    if not files:
        raise FileNotFoundError(f"no field recordings in {FIELD} (set WT_FIELD_NOISE)")
    return files


def decode(path: str) -> np.ndarray:
    raw = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            path,
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, FIELD_CHANNELS)


def load(first: int, n_files: int, channels=None) -> np.ndarray:
    """Files [first, first + n_files) at RATE, (samples, channels) float32.

    Decoded into one preallocated array and resampled electrode by electrode
    (a 5 min file of 16 electrodes is 0.9 GB as float32)."""
    files = field_files()[first : first + n_files]
    chans = np.arange(FIELD_CHANNELS) if channels is None else np.asarray(channels)
    per_file = int(FILE_S * FIELD_RATE)
    x = np.empty((per_file * len(files), len(chans)), np.float32)
    n = 0
    for p in files:
        part = decode(p)[:, chans]
        x[n : n + len(part)] = part[: len(x) - n]
        n += min(len(part), len(x) - n)
        del part
    x = x[:n]
    out = np.empty((-(-n * 5 // 12), len(chans)), np.float32)  # 48 -> 20 kHz
    for c in range(len(chans)):
        y = resample_poly(x[:, c], 5, 12)
        out[: len(y), c] = y
    return out[: len(y)]


def _write_grid(folder: Path, x: np.ndarray, cfg_from: Path):
    folder.mkdir(parents=True, exist_ok=True)
    x.astype(np.float32).tofile(folder / "traces-grid1.raw")
    text = (cfg_from / "fishgrid.cfg").read_text()
    import re

    n = x.shape[1]
    cols = 4 if n == 16 else 3
    text = re.sub(r"Columns1(\s*):\s*\d+", rf"Columns1\g<1>: {cols}", text)
    text = re.sub(r"Rows1(\s*):\s*\d+", rf"Rows1\g<1>: {n // cols}", text)
    (folder / "fishgrid.cfg").write_text(text)


def best_channel_power(sign_v: np.ndarray) -> np.ndarray:
    return sign_v.max(1)


def calibrate(args):
    """Ratio of near-electrode fish amplitudes, lab / field."""
    from wavetracker.config import Config
    from wavetracker.pipeline import detect, track_results
    from wavetracker.results import Results

    library = json.loads((DATA / "library.json").read_text())
    grid_cfg = next(Path(e["path"]) for e in library if e["channels"] == 9)
    night = NOISE / "night"
    if not (night / "ident_v.npy").exists():
        print("decoding the night", flush=True)
        _write_grid(night, load(*NIGHT), grid_cfg)
        cfg = Config.from_dict(
            {
                "harmonic_groups": {
                    "min_freq": 300.0,
                    "max_freq": 1500.0,
                    "mains_freq": 0.0,
                    "min_good_peak_power": None,
                    "low_thresh_factor": 4.0,
                    "high_thresh_factor": 7.0,
                    "min_group_size": 2,
                },
                "output": {"save_fine_spec": False},
            }
        )
        det = detect(night, night, cfg)
        track_results(det.results, cfg)
        det.results.save(night)
    res = Results.load(night)
    dt = res.times[1] - res.times[0]
    ids, counts = np.unique(res.ident_v[np.isfinite(res.ident_v)], return_counts=True)
    fish_ids = ids[counts * dt >= 30.0]
    m = np.isin(res.ident_v, fish_ids)
    p_field = best_channel_power(res.sign_v[m])
    per_fish_field = [
        float(np.median(best_channel_power(res.sign_v[res.ident_v == i])))
        for i in fish_ids
    ]

    per_fish_lab = []
    for e in library:
        if not e["ok"]:
            continue
        r = Results.load(DATA / "sources" / e["name"])
        tr = np.load(DATA / "sources" / e["name"] / "truth.npz")["freqs"]
        for k in range(2):
            mk = np.isclose(r.fund_v, tr[k, r.idx_v])
            if mk.sum() > 100:
                per_fish_lab.append(float(np.median(best_channel_power(r.sign_v[mk]))))
    p_lab = float(np.median(per_fish_lab))
    p_near = float(np.percentile(p_field, 95))
    gain = float(np.sqrt(p_lab / p_near))

    # the field floor, for the record: day noise of the dev hours
    day = load(DAY_DEV[0] + 10, 1)
    cal = {
        "field_recording": str(FIELD),
        "night_files": list(NIGHT),
        "night_fish_30s": len(fish_ids),
        "night_fish_median_hz": [
            float(np.median(res.fund_v[res.ident_v == i])) for i in fish_ids
        ],
        "field_near_power": p_near,
        "field_fish_median_power": per_fish_field,
        "lab_fish_median_power": per_fish_lab,
        "lab_power": p_lab,
        "noise_gain": gain,
        "noise_gain_db": float(20 * np.log10(gain)),
        "day_rms_per_channel": [float(v) for v in day.std(0)],
        "definition": "noise_gain = sqrt(median lab fish / 95th pct field fish), "
        "power at the fundamental on the best electrode (wavetracker sign_v)",
    }
    NOISE.mkdir(parents=True, exist_ok=True)
    (NOISE / "calibration.json").write_text(json.dumps(cal, indent=1))
    print(
        json.dumps({k: v for k, v in cal.items() if not isinstance(v, list)}, indent=1)
    )
    plot_calibration(cal, res, fish_ids)


def plot_calibration(cal, res, fish_ids):
    import matplotlib.pyplot as plt

    from wavetracker.results import load_sparse_spectrogram

    fig, axs = plt.subplots(
        1,
        2,
        figsize=(15, 4.5),
        layout="constrained",
        gridspec_kw={"width_ratios": [2, 1]},
    )
    spec, freqs, times = load_sparse_spectrogram(NOISE / "night")
    fm = (freqs > 250) & (freqs < 1500)
    db = 10 * np.log10(spec[fm] + 1e-30)
    ax = axs[0]
    ax.pcolormesh(
        (times - times[0]) / 60,
        freqs[fm],
        db,
        cmap="gray_r",
        vmin=np.percentile(db, 30),
        vmax=np.percentile(db, 99.8),
        shading="auto",
        rasterized=True,
    )
    for i in fish_ids:
        m = res.ident_v == i
        ax.plot(
            (res.times[res.idx_v[m]] - times[0]) / 60,
            res.fund_v[m],
            ".",
            ms=0.6,
            color="#2a78d6",
            rasterized=True,
        )
    ax.set_title(
        f"field night (calibration): {len(fish_ids)} tracks >= 30 s",
        loc="left",
        fontsize=10,
    )
    ax.set_xlabel("time [min]", fontsize=8)
    ax.set_ylabel("frequency [Hz]", fontsize=8)
    ax = axs[1]
    lab = 10 * np.log10(np.array(cal["lab_fish_median_power"]))
    fld = 10 * np.log10(np.array(cal["field_fish_median_power"])) + cal["noise_gain_db"]
    bins = np.arange(min(lab.min(), fld.min()) - 3, max(lab.max(), fld.max()) + 3, 3)
    ax.hist(lab, bins, color="#2a78d6", alpha=0.8, label="lab fish (library)")
    ax.hist(fld, bins, color="#eb6834", alpha=0.8, label="field fish, scaled")
    ax.axvline(10 * np.log10(cal["lab_power"]), color="#2a78d6", lw=1)
    ax.axvline(
        10 * np.log10(cal["field_near_power"]) + cal["noise_gain_db"],
        color="#eb6834",
        lw=1,
        ls="--",
    )
    ax.set_xlabel(
        "power at the fundamental, best electrode [dB, lab units]", fontsize=8
    )
    ax.set_title("lab median = field 95th percentile (dashed)", loc="left", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    for a in axs:
        a.tick_params(labelsize=7)
    p = DATA / "figures" / "field_noise_calibration.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(p)


def noise_plan() -> dict[str, dict]:
    """Field-noise segment (2 files) and electrode block for every mixture;
    dev and test from their own hours, no segment used twice in a split."""
    plan = {}
    # The training mixtures (train.yaml) reuse the dev hours, shifted by one
    # file: the noise is a condition there, not what is evaluated, and the
    # test hours stay unseen.
    suites = {"dev": expand_suite(), "test": expand_suite()}
    if TRAIN_SUITE.exists():
        suites["train"] = expand_suite(TRAIN_SUITE)
    for split, (f0, f1) in (("dev", DAY_DEV), ("test", DAY_TEST), ("train", DAY_TRAIN)):
        if split not in suites:
            continue
        names = sorted(n for n in suites[split] if n.startswith(split + "/"))
        starts = list(range(f0, f1 - 1, 2))
        if len(names) > len(starts):
            raise ValueError(f"{len(names)} mixtures, {len(starts)} noise segments")
        for i, n in enumerate(names):
            rng = np.random.default_rng([7, *map(ord, n)])
            plan[n] = {
                "first_file": starts[i],
                "n_files": 2,
                "block": int(rng.integers(4)),
            }
    return plan


def power_spec(x: np.ndarray, dev, hop_s: float = 0.5) -> np.ndarray:
    """Electrode-summed power spectrogram (bins, frames) as in mixture.snr_db."""
    hop = int(hop_s * RATE)
    win = torch.hann_window(SNR_NFFT, periodic=True, device=dev)
    scale = 1.0 / win.sum() ** 2
    p = 0
    for c in range(x.shape[1]):
        xc = torch.as_tensor(
            np.ascontiguousarray(x[:, c]), dtype=torch.float32, device=dev
        )
        z = torch.stft(xc, SNR_NFFT, hop, window=win, center=False, return_complex=True)
        p = p + (z.real**2 + z.imag**2) * scale
    return p.cpu().numpy()


def derive_one(name: str, plan: dict, gain: float, dev):
    src = MIXTURES / name
    split, rest = name.split("/")
    out = MIXTURES / f"{split}-field" / rest
    if (out / "band.yaml").exists():
        return
    print(f"{name} -> {out.relative_to(MIXTURES)}", flush=True)
    seg = plan[name]
    meta = json.loads((src / "mixture.json").read_text())
    n_out = int(meta["duration"] * RATE)
    noise = load(seg["first_file"], seg["n_files"], BLOCKS[seg["block"]])
    if len(noise) < n_out:  # a file shorter than 5 min: take the next one too
        noise = load(seg["first_file"], seg["n_files"] + 1, BLOCKS[seg["block"]])
    noise = noise[:n_out] * gain
    if len(noise) < n_out:
        raise ValueError(f"field noise segment too short for {name}")
    mix = np.fromfile(src / "traces-grid1.raw", np.float32).reshape(-1, MIX_CHANNELS)
    out.mkdir(parents=True, exist_ok=True)
    (mix + noise).astype(np.float32).tofile(out / "traces-grid1.raw")
    shutil.copy(src / "fishgrid.cfg", out / "fishgrid.cfg")

    truth = dict(np.load(src / "truth.npz"))
    pn = power_spec(noise, dev)
    pn = np.asarray(
        torch.nn.functional.avg_pool1d(torch.as_tensor(pn).T[:, None], 11, 1, 5)[:, 0].T
    )
    freqs = np.fft.rfftfreq(SNR_NFFT, 1 / RATE)
    t = truth["snr_times"]
    cols = np.arange(len(t))
    noise_p = truth["noise_power"].astype(np.float64).copy()
    for k, f in enumerate(truth["freqs_filled"]):
        fk = np.interp(t, truth["times"], f)
        b = np.clip(np.round(fk / freqs[1]).astype(int), 1, len(freqs) - 2)
        noise_p[k] += pn[b, cols]
    truth["noise_power"] = noise_p.astype(np.float32)
    truth["snr_db"] = 10 * np.log10(truth["fish_power"] / noise_p)
    np.savez(out / "truth.npz", **truth)

    ov = dict(np.load(src / "overview.npz"))
    nb = len(ov["freqs"]) * 4
    ov["power"] = (ov["power"] + pn[:nb].reshape(nb // 4, 4, -1).mean(1)).astype(
        np.float32
    )
    np.savez_compressed(out / "overview.npz", **ov)
    meta["name"] = f"{split}-field/{rest}"
    meta["noise"] = {
        "recording": str(FIELD),
        **seg,
        "channels": BLOCKS[seg["block"]],
        "gain": gain,
        "added_to": name,
    }
    (out / "mixture.json").write_text(json.dumps(meta, indent=1))
    shutil.copy(src / "band.yaml", out / "band.yaml")  # last: the variant is complete


def derive(args):
    cal = json.loads((NOISE / "calibration.json").read_text())
    gain = cal["noise_gain"] * 10 ** (args.extra_db / 20)
    plan = noise_plan()
    dev = _device(args.device)
    for name in plan:
        if (MIXTURES / name / "band.yaml").exists():
            derive_one(name, plan, gain, dev)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("calibrate")
    a.set_defaults(func=calibrate)
    b = sub.add_parser("derive")
    b.add_argument("--device", default="auto")
    b.add_argument("--extra-db", type=float, default=0.0, help="raise the noise")
    b.set_defaults(func=derive)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
