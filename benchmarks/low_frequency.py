"""Odd-harmonic low fish in a dense chorus: recall and sub-harmonic ghosts.

Synthesizes `wavetracker.synthetic.low_frequency_chorus` (fish at 120-300 Hz
without a 2nd harmonic, chorus at 350-900 Hz) as a 2-channel 48 kHz
recording and runs detection with the field-recording settings, with and
without ``harmonic_groups.max_missing_harmonics``. Reports per-frame recall
of the low and chorus fish, the fraction of unmatched detections, and
sub-harmonic ghosts: unmatched detections within ``--tol`` Hz of f/2 or f/3
of a chorus fish.

Usage::

    python benchmarks/low_frequency.py --seeds 0 1 2 --duration 60
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from wavetracker.config import Config
from wavetracker.evaluation import match_detections
from wavetracker.pipeline import detect
from wavetracker.synthetic import low_frequency_chorus, save_recording, synthesize

FIELD = {
    "spectrogram": {"nfft": 65536, "overlap_frac": 0.9},
    "interference": {"enabled": False},
    "harmonic_groups": {
        "min_freq": 20.0,
        "max_freq": 2000.0,
        "mains_freq": 0.0,
        "low_thresh_factor": 3.0,
        "high_thresh_factor": 5.0,
        "min_group_size": 2,
        "max_harmonics": 10,
    },
}


def score(res, truth_times, truth_freqs, n_low, tol):
    """Recall of low / chorus fish, unmatched fraction, ghost rates."""
    fish = match_detections(res, truth_times, truth_freqs, tol)
    in_range = (res.times >= truth_times[0]) & (res.times <= truth_times[-1])
    n_frames = int(in_range.sum())
    recall = np.array(
        [
            len(np.unique(res.idx_v[fish == k])) / n_frames
            for k in range(len(truth_freqs))
        ]
    )
    t = res.times[res.idx_v]
    chorus = np.stack([np.interp(t, truth_times, f) for f in truth_freqs[n_low:]])
    unmatched = fish < 0
    ghost = {
        m: unmatched & np.any(np.abs(chorus / m - res.fund_v[None]) <= tol, axis=0)
        for m in (2, 3)
    }
    return {
        "low_recall": recall[:n_low],
        "chorus_recall": recall[n_low:],
        "unmatched": unmatched.sum(),
        "unmatched_low": (unmatched & (res.fund_v < 350)).sum(),
        "unmatched_high": (unmatched & (res.fund_v > 900)).sum(),
        "ghost_2": ghost[2].sum(),
        "ghost_3": ghost[3].sum(),
        "detections": len(fish),
        "frames": n_frames,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--n-low", type=int, default=8)
    ap.add_argument("--n-chorus", type=int, default=30)
    ap.add_argument("--tol", type=float, default=1.0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        fish = low_frequency_chorus(
            args.duration, n_low=args.n_low, n_chorus=args.n_chorus, rng=rng
        )
        rec = synthesize(
            fish, args.duration, rate=48000.0, channels=2, mains=0.0, rng=rng
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rec.wav"
            save_recording(rec, path)
            for missing in (0, 1):
                cfg = Config.from_dict(FIELD)
                cfg.harmonic_groups.max_missing_harmonics = missing
                out = detect(path, Path(tmp) / f"m{missing}", cfg, device=args.device)
                s = score(
                    out.results, rec.truth_times, rec.truth_freqs, args.n_low, args.tol
                )
                low = " ".join(
                    f"{f.base_freq:.0f}:{r:.2f}"
                    for f, r in zip(fish, s["low_recall"], strict=False)
                )
                print(
                    f"seed {seed} max_missing {missing}: "
                    f"low recall {s['low_recall'].mean():.3f} "
                    f"(min {s['low_recall'].min():.2f}), "
                    f"chorus recall {s['chorus_recall'].mean():.3f}, "
                    f"unmatched {s['unmatched']}/{s['detections']} "
                    f"({s['unmatched_low']} < 350 Hz, {s['unmatched_high']} > 900 Hz), "
                    f"ghosts f/2 {s['ghost_2']} f/3 {s['ghost_3']} "
                    f"in {s['frames']} frames\n    low fish {low}"
                )


if __name__ == "__main__":
    main()
