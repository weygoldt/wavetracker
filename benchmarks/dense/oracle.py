"""Ceilings of the dense benchmark: oracle detections and oracle identities.

``detect``: the truth as detections. Every frame on which the truth knows a
fish (``truth["freqs"]``, as the scorer reads it at the result frames) gets one
detection at the fish's truth frequency, optionally only where the fish is at
least ``--min-snr`` dB above the floor of the mixture. The spectrogram is that
of ``pipeline.detect`` (clicks removed, interference cancelled, the same
frames), and every detection gets its per-electrode power and complex value at
the bin of its frequency, as ``detect`` stores them, so that the tracker's
amplitude similarity and the linker's pattern features see the mixture. Then
the unchanged tracker and linker of the config run on them: the ceiling of a
perfect detector with this association.

``ids``: the detections of an existing run, every detection matched to a fish
(``score.match``) gets that fish as its identity, unmatched ones none: the
ceiling of a perfect association with these detections. ``--kept`` only
relabels the detections the run's tracker gave an identity (what the tracker
and linker kept, joined perfectly).

    python benchmarks/dense/oracle.py detect --tag A-oracle-truth -c benchmarks/dense/configs/dense-linked.yaml --split ldev
    python benchmarks/dense/oracle.py detect --tag A-oracle-snr0 --min-snr 0 -c ... --split ldev
    python benchmarks/dense/oracle.py ids --tag A-oracle-ids-learned --source s5000-t0.6 --split ldev
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from run import MIXTURES, RUNS, merge, mixtures
from score import TOL, match, score, truth_at

import wavetracker.pipeline as pipeline
from wavetracker.config import Config
from wavetracker.harmonics import Detections
from wavetracker.io import FrameLayout, open_recording
from wavetracker.results import Results
from wavetracker.spectrogram import step_size


def frame_times(path: Path, cfg: Config) -> np.ndarray:
    """The frame times `pipeline.detect` will use for this recording."""
    sc = cfg.spectrogram
    with open_recording(path, buffersize=sc.block_duration) as data:
        rate = float(data.rate)
        nfft = sc.resolve_nfft(rate)
        return FrameLayout(
            0, len(data), nfft, step_size(nfft, sc.overlap_frac), rate
        ).times()


def oracle_freqs(name: str, times: np.ndarray, min_snr: float | None) -> np.ndarray:
    """Truth frequency (fish, frames) at the result frames, NaN where the
    truth does not know the fish (or it is below `min_snr`, as in score.py)."""
    truth = dict(np.load(MIXTURES / name / "truth.npz"))
    F = truth_at(truth, times)
    if min_snr is not None:
        snr = np.stack(
            [
                np.interp(times, truth["snr_times"], s, left=np.nan, right=np.nan)
                for s in truth["snr_db"]
            ]
        )
        F[~(snr >= min_snr)] = np.nan
    return F


class OracleDetector:
    """Stands in for `harmonics.detect_harmonic_groups` in `pipeline.detect`:
    the truth of the frames of each block, in block order."""

    def __init__(self, F: np.ndarray):
        self.F = F
        self.offset = 0

    def __call__(self, log_spec, freqs, cfg, low, high, band_bins=None) -> Detections:
        n = log_spec.shape[0]
        sub = self.F[:, self.offset : self.offset + n]
        self.offset += n
        frame, fish = np.nonzero(np.isfinite(sub.T))  # sorted by frame
        f = sub[fish, frame]
        b = np.clip(np.round(f / freqs[1]).astype(np.int64), 0, len(freqs) - 1)
        return Detections(frame.astype(np.int64), b, f.astype(np.float64))


def detect_oracle(name: str, out: Path, cfg: Config, min_snr, device: str):
    times = frame_times(MIXTURES / name, cfg)
    F = oracle_freqs(name, times, min_snr)
    oracle = OracleDetector(F)
    orig = pipeline.detect_harmonic_groups, pipeline.merge_tooth_neighbours
    pipeline.detect_harmonic_groups = oracle
    # every truth detection is kept, also next to interference teeth
    pipeline.merge_tooth_neighbours = lambda frame, *a, **k: np.ones(len(frame), bool)
    try:
        det = pipeline.detect(MIXTURES / name, out, cfg, device=device)
    finally:
        pipeline.detect_harmonic_groups, pipeline.merge_tooth_neighbours = orig
    res = det.results
    assert oracle.offset == len(times) and np.allclose(res.times, times)
    assert len(res.fund_v) == int(np.isfinite(F).sum())
    return res


def cmd_detect(args):
    variant = yaml.safe_load(Path(args.config).read_text()) or {}
    for name in mixtures(args.split, args.only):
        out = RUNS / args.tag / name
        if (out / "score.json").exists() and not args.overwrite:
            continue
        band = yaml.safe_load((MIXTURES / name / "band.yaml").read_text())
        cfg = Config.from_dict(merge(variant, band))
        if cfg.detector.method != "classic":
            raise ValueError("the oracle replaces the classic detector's groups")
        t0 = time.perf_counter()
        for attempt in range(30):  # other sessions share the GPU
            try:
                res = detect_oracle(name, out, cfg, args.min_snr, args.device)
                break
            except RuntimeError as e:
                if "out of memory" not in str(e) or attempt == 29:
                    raise
                print("  GPU out of memory, retrying in 60 s", flush=True)
                time.sleep(60)
        t_det = time.perf_counter() - t0
        res.meta["oracle"] = {"detections": "truth", "min_snr": args.min_snr}
        t_track = pipeline.track_results(res, cfg)
        res.save(out)
        summary, table = score(MIXTURES / name, out)
        summary |= {"tag": args.tag, "detect_s": t_det, "track_s": t_track}
        (out / "score.json").write_text(json.dumps(summary, indent=1))
        np.savez_compressed(out / "frames.npz", **table)
        print(
            f"{args.tag} {name}: {len(res.fund_v)} detections, recall "
            f"{summary['recall']:.3f} idf1 {summary['idf1']:.3f} precision "
            f"{summary['precision']:.3f} ids {summary['ids_30s']} "
            f"({t_det:.0f} + {t_track:.0f} s)",
            flush=True,
        )


def cmd_ids(args):
    for name in mixtures(args.split, args.only):
        src = RUNS / args.source / name
        out = RUNS / args.tag / name
        if not (src / "fund_v.npy").exists():
            print(f"{name}: no run {args.source}")
            continue
        if (out / "score.json").exists() and not args.overwrite:
            continue
        res = Results.load(src)
        truth = dict(np.load(MIXTURES / name / "truth.npz"))
        fish = match(res, truth_at(truth, res.times), TOL)
        keep = fish >= 0
        if args.kept:
            keep &= np.isfinite(res.ident_v)
        res.ident_v = np.where(keep, fish, -1).astype(np.float64)
        res.ident_v[~keep] = np.nan
        res.meta["oracle"] = {
            "identities": "truth",
            "source": args.source,
            "kept_only": args.kept,
        }
        res.save(out)
        summary, table = score(MIXTURES / name, out)
        summary |= {"tag": args.tag}
        (out / "score.json").write_text(json.dumps(summary, indent=1))
        np.savez_compressed(out / "frames.npz", **table)
        print(
            f"{args.tag} {name}: recall {summary['recall']:.3f} idf1 "
            f"{summary['idf1']:.3f} precision {summary['precision']:.3f}",
            flush=True,
        )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("detect", help="truth as detections, then track and link")
    a.add_argument("--tag", required=True)
    a.add_argument("-c", "--config", required=True)
    a.add_argument("--split", default="ldev")
    a.add_argument("--only", nargs="*")
    a.add_argument("--min-snr", type=float, default=None, help="dB; default: all")
    a.add_argument("--device", default="auto")
    a.add_argument("--overwrite", action="store_true")
    a.set_defaults(func=cmd_detect)
    b = sub.add_parser("ids", help="truth identities on the detections of a run")
    b.add_argument("--tag", required=True)
    b.add_argument("--source", required=True, help="run tag with the detections")
    b.add_argument("--split", default="ldev")
    b.add_argument("--only", nargs="*")
    b.add_argument(
        "--kept", action="store_true", help="only detections the run's tracker kept"
    )
    b.add_argument("--overwrite", action="store_true")
    b.set_defaults(func=cmd_ids)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
