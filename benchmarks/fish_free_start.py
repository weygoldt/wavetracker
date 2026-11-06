"""Do results for fish depend on what a recording starts with?

Puts real fish-free data in front of a real recording with fish, tracks both, and
compares the fish one to one with those of the recording alone, so that no ground
truth is needed. The fish-free part is a whole number of frame steps, so that the
frames of both line up.

Usage::

    python benchmarks/fish_free_start.py OUTDIR --fish day/*.wav \\
        --prefix "static=air/rec-0.wav:290" --prefix "quiet=water/rec-0.wav:290"

`--prefix NAME=FILE:SECONDS` is repeatable. Any format ffmpeg reads is fine, such as
the WavPack files of field loggers, which need no conversion. `--old` also runs the
settings of earlier versions: no click removal, thresholds frozen after the first
block. Add `-c cfg.yaml` for your configuration.
"""

from __future__ import annotations

import argparse
import glob
import subprocess
from pathlib import Path

import numpy as np

import wavetracker.pipeline as pipeline
from wavetracker.config import Config
from wavetracker.pipeline import detect, track_results
from wavetracker.results import Results
from wavetracker.spectrogram import step_size

MATCH_TOLERANCE = 0.5  # Hz, for one detection to be the same in both runs
LONG = 30.0  # s, a track at least this long does not happen by chance
SPLICE_MARGIN = 4.0  # s, windows reaching across the splice are not compared


def probe(path: str) -> tuple[float, int, float]:
    fields = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=sample_rate,channels,duration",
            "-of",
            "csv=p=0",
            path,
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    rate, channels, duration = fields.split(",")
    return float(rate), int(channels), float(duration)


class Stream:
    """A recording made of parts (file, seconds), decoded by ffmpeg on demand."""

    def __init__(self, parts: list[tuple[str, float | None]]) -> None:
        self.parts = []
        position = 0
        for path, seconds in parts:
            self.rate, self.channels, duration = probe(path)
            use = duration if seconds is None else min(seconds, duration)
            n = round(use * self.rate)
            self.parts.append((position, position + n, path))
            position += n
        self.n = position

    def __len__(self) -> int:
        return self.n

    def __enter__(self) -> Stream:
        return self

    def __exit__(self, *exc) -> None:
        pass

    def __getitem__(self, item: slice) -> np.ndarray:
        out = np.zeros((item.stop - item.start, self.channels), np.float32)
        for first, last, path in self.parts:
            lo, hi = max(item.start, first), min(item.stop, last)
            if lo >= hi:
                continue
            raw = subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-ss",
                    f"{(lo - first) / self.rate:.6f}",
                    "-t",
                    f"{(hi - lo) / self.rate:.6f}",
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
            data = np.frombuffer(raw, np.float32).reshape(-1, self.channels)
            n = min(len(data), hi - lo)
            out[lo - item.start : lo - item.start + n] = data[:n]
        return out


def run(parts, cfg: Config, folder: Path) -> Results:
    if (folder / "ident_v.npy").exists():
        return Results.load(folder)
    pipeline.open_recording = lambda path, buffersize=None: Stream(parts)
    pipeline.resolve_input = lambda path: [str(p) for p, _ in parts]
    out = detect("stream", folder, cfg)
    track_results(out.results, cfg)
    out.results.save(folder)
    return out.results


def _same_detection(idx_a, f_a, idx_b, f_b) -> np.ndarray:
    """For each detection a: index of the same one in b, or -1. b sorted by frame."""
    lo = np.searchsorted(idx_b, idx_a, "left")
    hi = np.searchsorted(idx_b, idx_a, "right")
    found = np.full(len(idx_a), -1)
    for k, (i, j, f) in enumerate(zip(lo, hi, f_a, strict=True)):
        if j > i:
            best = i + np.abs(f_b[i:j] - f).argmin()
            if abs(f_b[best] - f) <= MATCH_TOLERANCE:
                found[k] = best
    return found


def compare(res: Results, reference: Results) -> dict[str, float]:
    offset = len(res.times) - len(reference.times)
    dt = res.times[1] - res.times[0]
    margin = round(SPLICE_MARGIN / dt)
    start = res.idx_v >= offset + margin
    idx = res.idx_v[start] - offset
    fund, ident = res.fund_v[start], res.ident_v[start]
    ref_start = reference.idx_v >= margin
    ref_idx, ref_fund = reference.idx_v[ref_start], reference.fund_v[ref_start]
    ref_ident = reference.ident_v[ref_start]

    found = _same_detection(ref_idx, ref_fund, idx, fund)
    extra = _same_detection(idx, fund, ref_idx, ref_fund) < 0

    # the part of each long track that stays one identity
    kept = total = 0
    ids, counts = np.unique(ref_ident[~np.isnan(ref_ident)], return_counts=True)
    for i in ids[counts >= LONG / dt]:
        match = found[ref_ident == i]
        labels = ident[match[match >= 0]]
        labels = labels[~np.isnan(labels)]
        kept += np.unique(labels, return_counts=True)[1].max() if len(labels) else 0
        total += (ref_ident == i).sum()

    def long_tracks(values):
        _, c = np.unique(values[~np.isnan(values)], return_counts=True)
        return int((c >= LONG / dt).sum())

    return {
        "prefix detections per frame": (res.idx_v < offset).sum() / max(offset, 1),
        "of the fish's detections found": (found >= 0).mean(),
        "detections that are new": extra.mean() if len(extra) else 0.0,
        "of the long tracks kept as one identity": kept / max(total, 1),
        "tracks of 30 s or more": long_tracks(ident),
        "tracks of 30 s or more alone": long_tracks(ref_ident),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("output", type=Path)
    ap.add_argument("--fish", nargs="+", required=True, help="Files of the recording.")
    ap.add_argument("--prefix", action="append", required=True, metavar="NAME=FILE:SEC")
    ap.add_argument("-c", "--config", type=Path)
    ap.add_argument("--old", action="store_true", help="Also the earlier settings.")
    args = ap.parse_args()

    fish = [(f, None) for p in args.fish for f in sorted(glob.glob(p))]
    settings = {"new": {}}
    if args.old:
        settings["old"] = {"clicks": False, "window": None, "band": None}
    for label, change in settings.items():
        cfg = Config.load(args.config)
        if change:
            cfg.clicks.enabled = change["clicks"]
            cfg.harmonic_groups.threshold_window = change["window"]
            cfg.harmonic_groups.threshold_band = change["band"]
        rate = probe(fish[0][0])[0]
        step = step_size(cfg.spectrogram.resolve_nfft(rate), cfg.spectrogram.overlap_frac)
        reference = run(fish, cfg, args.output / label / "alone")
        for spec in args.prefix:
            name, rest = spec.split("=", 1)
            path, seconds = rest.rsplit(":", 1)
            whole = round(float(seconds) * rate / step) * step / rate
            res = run([(path, whole), *fish], cfg, args.output / label / name)
            print(f"\n{label}, {name} first:")
            for key, value in compare(res, reference).items():
                print(f"  {key:42s} {value:8.3f}")


if __name__ == "__main__":
    main()
