"""End-to-end analysis: spectrogram -> harmonic groups -> tracking."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import __version__
from .clicks import remove_clicks
from .config import Config, HarmonicGroupsConfig
from .harmonics import Detections, absolute_limit_binds, detect_harmonic_groups
from .interference import CombCanceller, merge_tooth_neighbours
from .io import FrameLayout, iter_blocks, open_recording, resolve_input
from .linking import link
from .pairs import fill_partners, resolve_pairs, split_partners
from .results import Results
from .rises import join_rises
from .salience import StreamingSalience
from .spectrogram import (
    PowerSpectrogram,
    band_noise_std,
    decibel,
    estimate_noise_std,
    frame_noise_std,
    frequencies,
    get_device,
    min_noise_std,
    step_size,
)
from .stitching import stitch
from .tracking import _no_salience, drop_weak_identities, track

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


class DetectionThresholds:
    """Peak thresholds of every frame, block by block.

    They follow the noise around each frame (`threshold_window`), so that a
    recording may start long before it has the noise of the water or any fish:
    results for a fish do not depend on what came before it, and over frequency
    (`threshold_band`), so that noise common to all electrodes at low frequencies
    does not make false fish. Values given in the configuration are used as they
    are.
    """

    #: Scale of `band_noise_std`, which reads 1.08 to 1.14 of `frame_noise_std` on
    #: Gaussian noise: bands only raise thresholds, and not where the noise is the
    #: same over frequency. Measured on real fish at night in a field recording,
    #: against what is found without bands: 0.6 finds 94 % of it and leaves 0.03 of
    #: 0.14 other detections per frame; 0.7 finds 88 % and leaves 0.02; 0.85 finds 78 %.
    BAND_SCALE = 0.6

    def __init__(
        self,
        cfg: HarmonicGroupsConfig,
        n_channels: int,
        frame_step: float,
        band_bins: int | None = None,
        lag: int = 1,
    ) -> None:
        self.cfg = cfg
        self.floor = min_noise_std(n_channels)
        adaptive = cfg.threshold_window is not None
        self.band_bins = band_bins if adaptive else None
        self.lag = lag
        self.known_bands: np.ndarray | None = None
        self.width = (
            None
            if cfg.threshold_window is None
            else max(1, round(cfg.threshold_window / frame_step)) | 1
        )
        self.known: float | None = None
        """The latest noise estimate, for blocks without data."""
        self.blocks: list[float | None] = []
        """Median noise of each block (None: no data), for the results."""

    def _smooth(self, std: torch.Tensor) -> torch.Tensor:
        """Median over `width` frames along the last axis, ignoring NaN."""
        half = self.width // 2
        padded = torch.nn.functional.pad(std, (half, half), value=torch.nan)
        return padded.unfold(-1, self.width, 1).nanmedian(-1).values

    def _band_noise(self, log_spec: torch.Tensor) -> np.ndarray:
        """Scaled noise per band and frame, shape (frames, bands), NaN unknown."""
        std = band_noise_std(log_spec, self.lag, self.band_bins)
        std = self.BAND_SCALE * self._smooth(std)
        std = std.T.cpu().numpy()
        known = np.isfinite(std).any(0)
        if known.any():
            self.known_bands = np.nanmedian(std[:, known], axis=0)
            if not known.all():
                full = np.full(std.shape[1], np.nan)
                full[known] = self.known_bands
                self.known_bands = full
        if self.known_bands is not None:
            std = np.where(np.isfinite(std), std, self.known_bands)
        return std

    def _frame_noise(self, log_spec: torch.Tensor) -> np.ndarray:
        n_frames = log_spec.shape[1]
        if self.width is None:  # frozen at the first block with data
            if self.known is None:
                estimate = estimate_noise_std(log_spec)
                self.known = None if math.isnan(estimate) else estimate
            self.blocks.append(self.known)
            return np.full(n_frames, np.nan if self.known is None else self.known)
        std = self._smooth(frame_noise_std(log_spec)).cpu().numpy()
        valid = np.isfinite(std)
        if valid.any():
            self.known = float(np.median(std[valid]))
        self.blocks.append(self.known if valid.any() else None)
        fill = self.floor if self.known is None else self.known
        return np.maximum(np.where(valid, std, fill), self.floor)

    def __call__(self, log_spec: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Low and high threshold of each frame of `log_spec` (freqs, frames),
        with bands of `band_bins` shape (frames, bands), else (frames,)."""
        cfg = self.cfg
        n_frames = log_spec.shape[1]
        if cfg.low_threshold is not None and cfg.high_threshold is not None:
            return (
                np.full(n_frames, float(cfg.low_threshold)),
                np.full(n_frames, float(cfg.high_threshold)),
            )
        std = self._frame_noise(log_spec)
        if self.band_bins is not None:
            bands = self._band_noise(log_spec)
            std = np.fmax(std[:, None], bands)
        low = cfg.low_thresh_factor * std
        high = cfg.high_thresh_factor * std
        if cfg.low_threshold is not None:
            low = np.full(low.shape, float(cfg.low_threshold))
        if cfg.high_threshold is not None:
            high = np.full(high.shape, float(cfg.high_threshold))
        return low, high

    @property
    def noise_std(self) -> float | None:
        """Median noise over the blocks so far (None: no data, or not needed)."""
        known = [b for b in self.blocks if b is not None]
        return float(np.median(known)) if known else None

    def summary(self) -> tuple[float | None, float | None]:
        """One low and high threshold for the recording (None: no data)."""
        cfg, std = self.cfg, self.noise_std
        low, high = cfg.low_threshold, cfg.high_threshold
        if low is None and std is not None:
            low = cfg.low_thresh_factor * max(std, self.floor)
        if high is None and std is not None:
            high = cfg.high_thresh_factor * max(std, self.floor)
        return low, high


def classic_salience(
    log_np: np.ndarray,
    det: Detections,
    high_th: np.ndarray,
    band_bins: int | None,
    fish_bins: slice,
) -> np.ndarray:
    """A salience (0-1) for classic detections, as the learned detector gives.

    The fundamental's level over the local noise floor, `snr` [dB], in units of
    the high threshold of its frame and band (the prominence a "good" peak
    needs): ``1 - 2 ** (-snr / high)``. It is 0 at the floor, 0.5 a high
    threshold above it, 0.75 at twice that, and rises monotonically. The floor
    is the median of the frame's dB spectrum `log_np` (frames, freqs) over the
    detection's threshold band, or over the fish range `fish_bins` where the
    thresholds have no bands (`high_th` of shape (frames,)).
    """
    if len(det.frame) == 0:
        return np.empty(0, np.float32)
    level = log_np[det.frame, det.bin]
    if high_th.ndim == 2:
        band = np.minimum(det.bin // band_bins, high_th.shape[1] - 1)
        cols = np.minimum(
            band[:, None] * band_bins + np.arange(band_bins), log_np.shape[1] - 1
        )
        floor = np.median(log_np[det.frame[:, None], cols], axis=1)
        high = high_th[det.frame, band]
    else:
        frames, inverse = np.unique(det.frame, return_inverse=True)
        floor = np.median(log_np[frames, fish_bins], axis=1)[inverse]
        high = high_th[det.frame]
    snr = np.maximum(level - floor, 0.0)
    return (1.0 - 0.5 ** (snr / np.maximum(high, 1e-9))).astype(np.float32)


@dataclass
class DetectionOutput:
    results: Results
    timings: Timings = field(default_factory=Timings)


def _config_used(cfg: Config, nfft: int) -> dict:
    """The config for wavetracker.json, with the nfft that was used (readers
    of the results need the window length, which depends on the rate)."""
    d = cfg.to_dict()
    d["spectrogram"]["nfft"] = int(nfft)
    return d


def detect(
    input_path: str | Path | Sequence[str | Path],
    output_dir: str | Path,
    cfg: Config,
    start: float = 0.0,
    duration: float | None = None,
    device: str = "auto",
    progress: ProgressCallback | None = None,
) -> DetectionOutput:
    """Compute spectrograms and detect fish fundamentals in a recording.

    Results (without identities) and spectrograms are written to `output_dir`.
    `input_path` is a file, a recording directory, or a sequence of files that
    together make one recording (see `io.resolve_input`).

    Every detection gets a salience (0-1, `Results.sal_v`): the network's with
    the learned detector, and with the classic one a stand-in from the
    fundamental's level over the local noise (see `classic_salience`).
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
        nfft = sc.resolve_nfft(rate)
        step = step_size(nfft, sc.overlap_frac)
        layout = FrameLayout(s0, s1, nfft, step, rate)
        times = layout.times()
        n_frames = layout.n_frames
        if n_frames < 2:
            raise ValueError("Selected data is shorter than two FFT windows.")
        frames_per_block = max(1, round(sc.block_duration * rate / step))
        freqs = frequencies(nfft, rate)
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

        spectrogram = PowerSpectrogram(nfft, step, rate, dev)
        oc = cfg.output
        # stored spectrograms cover the tracked range unless limited explicitly
        auto_max = 1.25 * hc.max_freq
        sparse_max = (
            auto_max if oc.sparse_spec_max_freq is None else oc.sparse_spec_max_freq
        )
        fine_max = auto_max if oc.fine_spec_max_freq is None else oc.fine_spec_max_freq
        sparse = _SparseSpectrogram(
            freqs,
            times,
            sparse_max,
            oc.sparse_spec_freq_res,
            oc.sparse_spec_time_bins,
        )
        fine = None
        if oc.save_fine_spec:
            nf_fine = int(np.searchsorted(freqs, fine_max, side="right"))
            fine = np.lib.format.open_memmap(
                output_dir / "fine_spec.npy",
                mode="w+",
                dtype=np.float32,
                shape=(n_frames, nf_fine),
            )
            np.save(output_dir / "fine_freqs.npy", freqs[:nf_fine])
            np.save(output_dir / "fine_times.npy", times)

        canceller = (
            CombCanceller(cfg.interference, freqs, min_fish_freq=hc.min_freq)
            if cfg.interference.enabled
            else None
        )
        comb_log: dict[float, dict] = {}
        band_bins = (
            None
            if hc.threshold_band is None
            else max(1, round(hc.threshold_band / freqs[1]))
        )
        thresholds = DetectionThresholds(
            hc, n_channels, step / rate, band_bins, max(1, round(nfft / step))
        )
        click_blocks: list[float] = []
        noise_floor = None
        fish_bins = slice(*np.searchsorted(freqs, [hc.min_freq, hc.max_freq]))
        funds, idxs, signs, cplxs, sals = [], [], [], [], []
        streamer = None
        if cfg.detector.method == "learned":
            streamer = StreamingSalience.from_config(
                cfg.detector, freqs, step / rate, hc.min_freq, hc.max_freq, dev
            )
            log.info("Learned detector: %s", streamer.model_info["path"])

        def collect(det, sal, first_frame, log_np, power, stft, teeth):
            """Store detections of one block with their electrode spectra."""
            if teeth is not None and len(det.frame):
                keep = merge_tooth_neighbours(
                    det.frame,
                    det.freq,
                    log_np[det.frame, det.bin],
                    teeth,
                    cfg.interference.neighbour_distance,
                    cfg.interference.neighbour_tolerance,
                    cfg.interference.tooth_neighbour_distance,
                )
                det = Detections(det.frame[keep], det.bin[keep], det.freq[keep])
                sal = None if sal is None else sal[keep]
            if len(det.frame):
                b = torch.from_numpy(det.bin).to(dev)
                fr = torch.from_numpy(det.frame).to(dev)
                signs.append(power[:, b, fr].T.cpu().numpy())
                cplxs.append(stft[:, b, fr].T.cpu().numpy())
                funds.append(det.freq)
                idxs.append(det.frame + first_frame)
                if sal is not None:
                    sals.append(sal)

        t_read = time.perf_counter()
        for block in iter_blocks(data, layout, frames_per_block, channels):
            t0 = time.perf_counter()
            timings.read += t0 - t_read

            traces = torch.from_numpy(block.data).to(dev, non_blocking=True)
            if cfg.clicks.enabled:
                traces, fraction = remove_clicks(traces, rate, cfg.clicks)
                click_blocks.append(round(float(fraction.mean()), 6))
            stft = spectrogram.stft(traces)
            del traces
            power = spectrogram.power(stft)
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
            low_th, high_th = thresholds(log_spec)
            log.debug(
                "Frame %d: noise std %s dB, thresholds low=%.2f dB, high=%.2f dB",
                block.first_frame,
                thresholds.blocks[-1] if thresholds.blocks else None,
                np.median(low_th),
                np.median(high_th),
            )
            log_np = log_spec.T.contiguous().cpu().numpy()
            if noise_floor is None:
                band = (freqs >= hc.min_freq) & (freqs <= hc.max_freq)
                level = log_np[:, band]
                level = level[np.isfinite(level)]
                # Digital silence has no level: use the first block with data.
                if level.size:
                    noise_floor = float(np.median(level))
                    if absolute_limit_binds(noise_floor, np.median(high_th), hc):
                        log.warning(
                            "Median noise floor %.1f dB + high threshold %.1f dB is "
                            "below harmonic_groups.min_good_peak_power (%.1f dB): weak "
                            "fish are removed by this absolute limit. Lower it or set "
                            "it to null.",
                            noise_floor,
                            np.median(high_th),
                            hc.min_good_peak_power,
                        )
            sparse.add(summed)
            if fine is not None:
                fine[block.first_frame : block.first_frame + block.n_frames] = (
                    summed[: fine.shape[1]].T.cpu().numpy()
                )
            t1 = time.perf_counter()
            timings.spectrogram += t1 - t0

            teeth = canceller.tooth_frequencies() if canceller is not None else None
            if streamer is None:
                det = detect_harmonic_groups(
                    log_np, freqs, hc, low_th, high_th, band_bins
                )
                sal = classic_salience(log_np, det, high_th, band_bins, fish_bins)
                collect(det, sal, block.first_frame, log_np, power, stft, teeth)
            else:
                # the block is evaluated once the next one gives its context:
                # keep what the detections need (fundamental bins only)
                k = streamer.k_hi + 1
                payload = (
                    block.first_frame,
                    log_np,
                    power[:, :k].clone(),
                    stft[:, :k].clone(),
                    teeth,
                )
                for (fr, b, f, s), pl in streamer.push(
                    streamer.features(log_spec, stft), payload
                ):
                    collect(Detections(fr, b, f), s, *pl)
            del stft, power, summed, log_spec
            t_read = time.perf_counter()
            timings.detection += t_read - t1
            if progress:
                progress(block.first_frame + block.n_frames, n_frames)
        if streamer is not None:
            for (fr, b, f, s), pl in streamer.finish():
                collect(Detections(fr, b, f), s, *pl)

    fund_v = np.concatenate(funds) if funds else np.empty(0)
    sal_v = np.concatenate(sals) if sals else np.empty(0, np.float32)
    idx_v = np.concatenate(idxs).astype(np.int64) if idxs else np.empty(0, np.int64)
    sign_v = (
        np.concatenate(signs).astype(np.float32)
        if signs
        else np.empty((0, n_channels), np.float32)
    )
    cplx_v = (
        np.concatenate(cplxs).astype(np.complex64)
        if cplxs
        else np.empty((0, n_channels), np.complex64)
    )
    spec, sfreqs, stimes = sparse.finish()
    np.save(output_dir / "sparse_spectra.npy", spec)
    np.save(output_dir / "sparse_freq.npy", sfreqs)
    np.save(output_dir / "sparse_time.npy", stimes)
    if fine is not None:
        fine.flush()
        del fine

    timings.total = time.perf_counter() - t_start
    low_summary, high_summary = thresholds.summary()
    meta = {
        "version": __version__,
        "input": (
            [str(Path(p).resolve()) for p in input_path]
            if isinstance(input_path, (list, tuple))
            else str(Path(input_path).resolve())
        ),
        "files": resolve_input(input_path),
        "rate": rate,
        "channels": channels.tolist(),
        "start": s0 / rate,
        "duration": (s1 - s0) / rate,
        "device": str(dev),
        "frame_step": step / rate,
        "freq_resolution": float(freqs[1]),
        "noise_std": thresholds.noise_std,
        "noise_std_blocks": thresholds.blocks,
        "click_fraction": float(np.mean(click_blocks)) if click_blocks else 0.0,
        "click_fraction_blocks": click_blocks,
        "interference_combs": {
            f"{spacing:.1f}": {
                "blocks": len(e["blocks"]),
                "channels": sorted(e["channels"]),
                "max_teeth": e["max_teeth"],
            }
            for spacing, e in sorted(comb_log.items())
        },
        "low_threshold": low_summary,
        "high_threshold": high_summary,
        "noise_floor": noise_floor,
        "config": _config_used(cfg, nfft),
        "timings": vars(timings),
    }
    if streamer is not None:
        meta["detector_model"] = streamer.model_info
    results = Results(
        fund_v,
        idx_v,
        sign_v,
        np.full(len(fund_v), np.nan),
        times,
        meta,
        cplx_v=cplx_v,
        sal_v=sal_v,
    )
    results.save(output_dir)
    return DetectionOutput(results, timings)


def _subset(results: Results, keep: np.ndarray) -> Results:
    def sel(v):
        return None if v is None else v[keep]

    return Results(
        fund_v=results.fund_v[keep],
        idx_v=results.idx_v[keep],
        sign_v=results.sign_v[keep],
        ident_v=results.ident_v[keep],
        times=results.times,
        meta=results.meta,
        cplx_v=sel(results.cplx_v),
        sal_v=sel(results.sal_v),
    )


def track_results(results: Results, cfg: Config) -> float:
    """Assign identities in place; returns the time it took."""
    t0 = time.perf_counter()
    if cfg.tracking.min_identity_salience is not None and results.sal_v is None:
        raise ValueError(_no_salience("min_identity_salience"))
    if cfg.pairs.enabled and cfg.pairs.partners == "fill":
        # split partners second: everything else is tracked without them
        partner = split_partners(
            results.fund_v, results.idx_v, results.sal_v, cfg.pairs.split_df
        )
        sub = _subset(results, ~partner)
        _identities(sub, cfg)
        ident = np.full(len(results.fund_v), np.nan)
        ident[~partner] = sub.ident_v
        results.ident_v = fill_partners(
            results.fund_v, results.idx_v, ident, partner, results.times, cfg.pairs
        )
    else:
        _identities(results, cfg)
    dt = time.perf_counter() - t0
    results.meta.setdefault("timings", {})["tracking"] = dt
    results.meta["tracking_config"] = vars(cfg.tracking)
    results.meta["stitching_config"] = vars(cfg.stitching)
    results.meta["linking_config"] = vars(cfg.linking)
    if cfg.rises.enabled:
        results.meta["rises_config"] = vars(cfg.rises)
    if cfg.pairs.enabled:
        results.meta["pairs_config"] = vars(cfg.pairs)
    return dt


def _identities(results: Results, cfg: Config) -> None:
    """Tracking, linking (or stitching), the rise joiner, the pair resolver
    and the identity salience filter, in place."""
    results.ident_v = track(
        results.fund_v,
        results.idx_v,
        results.sign_v,
        results.times,
        cfg.tracking,
        min_freq=cfg.harmonic_groups.min_freq,
        max_freq=cfg.harmonic_groups.max_freq,
        sal_v=results.sal_v,
    )

    def pairs(stage):
        if cfg.pairs.enabled and cfg.pairs.resolve and cfg.pairs.stage == stage:
            results.ident_v = resolve_pairs(
                results.fund_v,
                results.idx_v,
                results.sign_v,
                results.ident_v,
                results.times,
                cfg.pairs,
            )

    pairs("tracking")
    if cfg.linking.enabled:
        results.ident_v = link(
            results.fund_v,
            results.idx_v,
            results.sign_v,
            results.cplx_v,
            results.ident_v,
            results.times,
            cfg.linking,
        )
    else:
        results.ident_v = stitch(
            results.fund_v,
            results.idx_v,
            results.sign_v,
            results.ident_v,
            results.times,
            cfg.stitching,
        )
    if cfg.rises.enabled:
        results.ident_v = join_rises(
            results.fund_v,
            results.idx_v,
            results.sign_v,
            results.ident_v,
            results.times,
            cfg.rises,
            sal_v=results.sal_v,
        )
    pairs("linking")
    if cfg.tracking.min_identity_salience is not None:
        results.ident_v = drop_weak_identities(
            results.ident_v, results.sal_v, cfg.tracking.min_identity_salience
        )
