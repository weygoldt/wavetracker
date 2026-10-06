"""End-to-end analysis: spectrogram -> harmonic groups -> tracking."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import __version__
from .config import Config
from .harmonics import Detections, detect_harmonic_groups
from .interference import CombCanceller, merge_tooth_neighbours
from .io import FrameLayout, iter_blocks, open_recording, resolve_input
from .results import Results
from .spectrogram import (
    PowerSpectrogram,
    decibel,
    estimate_noise_std,
    frequencies,
    get_device,
    step_size,
)
from .stitching import stitch
from .tracking import track

log = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]
"""Called with (frames done, total frames)."""


@dataclass
class Timings:
    read: float = 0.0
    spectrogram: float = 0.0
    detection: float = 0.0
    tracking: float = 0.0
    total: float = 0.0


class _SparseSpectrogram:
    """Max-pools the summed spectrogram to an overview image while streaming."""

    def __init__(self, freqs, times, max_freq, freq_res, time_bins):
        df = freqs[1] - freqs[0]
        self.fpool = max(1, round(freq_res / df))
        self.nf = int(np.searchsorted(freqs, max_freq)) // self.fpool * self.fpool
        self.tpool = max(1, math.ceil(len(times) / time_bins))
        self.freqs = freqs[: self.nf].reshape(-1, self.fpool).mean(1)
        self.times = times
        self.chunks: list[np.ndarray] = []
        self.rest = np.empty((self.nf // self.fpool, 0), dtype=np.float32)

    def add(self, power: torch.Tensor) -> None:
        """`power` has shape (freqs, frames)."""
        p = power[: self.nf].T.reshape(power.shape[1], -1, self.fpool).amax(-1).T
        p = np.concatenate([self.rest, p.cpu().numpy()], axis=1)
        n = p.shape[1] // self.tpool * self.tpool
        if n:
            self.chunks.append(p[:, :n].reshape(p.shape[0], -1, self.tpool).max(-1))
        self.rest = p[:, n:]

    def finish(self):
        if self.rest.shape[1]:
            self.chunks.append(self.rest.max(1, keepdims=True))
        spec = np.concatenate(self.chunks, axis=1) if self.chunks else self.rest
        t = self.times
        groups = np.arange(len(t)) // self.tpool
        times = np.bincount(groups, weights=t) / np.bincount(groups)
        return spec, self.freqs, times


@dataclass
class DetectionOutput:
    results: Results
    timings: Timings = field(default_factory=Timings)


def detect(
    input_path: str | Path,
    output_dir: str | Path,
    cfg: Config,
    start: float = 0.0,
    duration: float | None = None,
    device: str = "auto",
    progress: ProgressCallback | None = None,
) -> DetectionOutput:
    """Compute spectrograms and detect fish fundamentals in a recording.

    Results (without identities) and spectrograms are written to `output_dir`.
    """
    t_start = time.perf_counter()
    timings = Timings()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dev = get_device(device)
    sc, hc = cfg.spectrogram, cfg.harmonic_groups

    with open_recording(input_path, buffersize=sc.block_duration) as data:
        rate = float(data.rate)
        channels = np.setdiff1d(np.arange(data.channels), sc.exclude_channels)
        if len(channels) == 0:
            raise ValueError("All channels are excluded.")
        n_channels = len(channels)
        n_total = len(data)
        s0 = min(n_total, max(0, round(start * rate)))
        s1 = n_total if duration is None else min(n_total, s0 + round(duration * rate))
        step = step_size(sc.nfft, sc.overlap_frac)
        layout = FrameLayout(s0, s1, sc.nfft, step, rate)
        times = layout.times()
        n_frames = layout.n_frames
        if n_frames < 2:
            raise ValueError("Selected data is shorter than two FFT windows.")
        frames_per_block = max(1, round(sc.block_duration * rate / step))
        freqs = frequencies(sc.nfft, rate)
        log.info(
            "%s: %d channels @ %.0f Hz, %.1f s analysed (%d frames, df=%.3f Hz, "
            "dt=%.3f s) on %s",
            input_path,
            n_channels,
            rate,
            (s1 - s0) / rate,
            n_frames,
            freqs[1],
            step / rate,
            dev,
        )

        spectrogram = PowerSpectrogram(sc.nfft, step, rate, dev)
        oc = cfg.output
        sparse = _SparseSpectrogram(
            freqs,
            times,
            oc.sparse_spec_max_freq,
            oc.sparse_spec_freq_res,
            oc.sparse_spec_time_bins,
        )
        fine = None
        if oc.save_fine_spec:
            nf_fine = int(np.searchsorted(freqs, oc.fine_spec_max_freq, side="right"))
            fine = np.lib.format.open_memmap(
                output_dir / "fine_spec.npy",
                mode="w+",
                dtype=np.float32,
                shape=(n_frames, nf_fine),
            )
            np.save(output_dir / "fine_freqs.npy", freqs[:nf_fine])
            np.save(output_dir / "fine_times.npy", times)

        canceller = (
            CombCanceller(cfg.interference, freqs) if cfg.interference.enabled else None
        )
        comb_log: dict[float, dict] = {}
        low_th, high_th = hc.low_threshold, hc.high_threshold
        noise_std = None
        funds, idxs, signs = [], [], []

        t_read = time.perf_counter()
        for block in iter_blocks(data, layout, frames_per_block, channels):
            t0 = time.perf_counter()
            timings.read += t0 - t_read

            power = spectrogram(torch.from_numpy(block.data).to(dev, non_blocking=True))
            if canceller is not None:
                power = canceller(power)
                for comb in canceller.combs:
                    entry = comb_log.setdefault(
                        round(comb.spacing, 1),
                        {"blocks": set(), "channels": set(), "max_teeth": 0},
                    )
                    entry["blocks"].add(block.first_frame)
                    entry["channels"].add(int(channels[comb.channel]))
                    entry["max_teeth"] = max(entry["max_teeth"], len(comb.teeth))
            summed = power.sum(0)
            log_spec = decibel(summed)
            if low_th is None or high_th is None:
                noise_std = estimate_noise_std(log_spec)
                low_th = noise_std * hc.low_thresh_factor if low_th is None else low_th
                high_th = (
                    noise_std * hc.high_thresh_factor if high_th is None else high_th
                )
                log.info(
                    "Noise std %.2f dB -> thresholds low=%.2f dB, high=%.2f dB",
                    noise_std,
                    low_th,
                    high_th,
                )
            log_np = log_spec.T.contiguous().cpu().numpy()
            sparse.add(summed)
            if fine is not None:
                fine[block.first_frame : block.first_frame + block.n_frames] = (
                    summed[: fine.shape[1]].T.cpu().numpy()
                )
            t1 = time.perf_counter()
            timings.spectrogram += t1 - t0

            det = detect_harmonic_groups(log_np, freqs, hc, low_th, high_th)
            if canceller is not None and len(det.frame):
                keep = merge_tooth_neighbours(
                    det.frame,
                    det.freq,
                    log_np[det.frame, det.bin],
                    canceller.tooth_frequencies(),
                    cfg.interference.neighbour_distance,
                    cfg.interference.neighbour_tolerance,
                    cfg.interference.tooth_neighbour_distance,
                )
                det = Detections(det.frame[keep], det.bin[keep], det.freq[keep])
            if len(det.frame):
                sel = power[
                    :,
                    torch.from_numpy(det.bin).to(dev),
                    torch.from_numpy(det.frame).to(dev),
                ]
                signs.append(sel.T.cpu().numpy())
                funds.append(det.freq)
                idxs.append(det.frame + block.first_frame)
            del power, summed, log_spec
            t_read = time.perf_counter()
            timings.detection += t_read - t1
            if progress:
                progress(block.first_frame + block.n_frames, n_frames)

    fund_v = np.concatenate(funds) if funds else np.empty(0)
    idx_v = np.concatenate(idxs).astype(np.int64) if idxs else np.empty(0, np.int64)
    sign_v = (
        np.concatenate(signs).astype(np.float32)
        if signs
        else np.empty((0, n_channels), np.float32)
    )
    spec, sfreqs, stimes = sparse.finish()
    np.save(output_dir / "sparse_spectra.npy", spec)
    np.save(output_dir / "sparse_freq.npy", sfreqs)
    np.save(output_dir / "sparse_time.npy", stimes)
    if fine is not None:
        fine.flush()
        del fine

    timings.total = time.perf_counter() - t_start
    meta = {
        "version": __version__,
        "input": str(Path(input_path).resolve()),
        "files": resolve_input(input_path),
        "rate": rate,
        "channels": channels.tolist(),
        "start": s0 / rate,
        "duration": (s1 - s0) / rate,
        "device": str(dev),
        "frame_step": step / rate,
        "freq_resolution": float(freqs[1]),
        "noise_std": noise_std,
        "interference_combs": {
            f"{spacing:.1f}": {
                "blocks": len(e["blocks"]),
                "channels": sorted(e["channels"]),
                "max_teeth": e["max_teeth"],
            }
            for spacing, e in sorted(comb_log.items())
        },
        "low_threshold": low_th,
        "high_threshold": high_th,
        "config": cfg.to_dict(),
        "timings": vars(timings),
    }
    results = Results(fund_v, idx_v, sign_v, np.full(len(fund_v), np.nan), times, meta)
    results.save(output_dir)
    return DetectionOutput(results, timings)


def track_results(results: Results, cfg: Config) -> float:
    """Assign identities in place; returns the time it took."""
    t0 = time.perf_counter()
    results.ident_v = track(
        results.fund_v,
        results.idx_v,
        results.sign_v,
        results.times,
        cfg.tracking,
        min_freq=cfg.harmonic_groups.min_freq,
        max_freq=cfg.harmonic_groups.max_freq,
    )
    results.ident_v = stitch(
        results.fund_v,
        results.idx_v,
        results.sign_v,
        results.ident_v,
        results.times,
        cfg.stitching,
    )
    dt = time.perf_counter() - t0
    results.meta.setdefault("timings", {})["tracking"] = dt
    results.meta["tracking_config"] = vars(cfg.tracking)
    results.meta["stitching_config"] = vars(cfg.stitching)
    return dt
