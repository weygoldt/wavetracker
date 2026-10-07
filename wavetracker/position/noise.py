"""Noise floor per channel at a fish's frequency.

The observation model needs the amplitude a channel shows when the fish is
far away. Three sources, in order of preference:

* :meth:`SpectrumNoiseFloor.from_recording`: a low percentile over survey
  frames of each channel's power spectrum, computed from the recording with
  the scaling and frame grid of wavetracker's spectrogram (``sign_v`` is the
  power at the nearest bin of the same spectrum);
* :meth:`SpectrumNoiseFloor.load`: a precomputed ``.npz`` with ``freqs``
  (F,) and ``power`` (F, C);
* :class:`DetectionNoiseFloor`: a low percentile of the detection powers
  near the frequency (no recording needed; overestimates the floor where
  every detection is seen on all channels).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class SpectrumNoiseFloor:
    def __init__(self, freqs: np.ndarray, power: np.ndarray):
        self.freqs = np.asarray(freqs, float)
        self.power = np.asarray(power, float).reshape(len(self.freqs), -1)

    def amplitude(self, f: float) -> np.ndarray:
        """Amplitude (sqrt power) per channel: minimum over +-2 bins."""
        k = int(np.clip(np.searchsorted(self.freqs, f), 0, len(self.freqs) - 1))
        lo, hi = max(k - 2, 0), min(k + 3, len(self.freqs))
        return np.sqrt(self.power[lo:hi].min(0))

    def save(self, path: str | Path) -> None:
        np.savez(path, freqs=self.freqs, power=self.power)

    @classmethod
    def load(cls, path: str | Path) -> SpectrumNoiseFloor:
        d = np.load(path)
        return cls(d["freqs"], d["power"])

    @classmethod
    def from_recording(
        cls,
        path: str | Path,
        frame_times: np.ndarray,
        nfft: int,
        channels: list[int] | None = None,
        quantile: float = 10.0,
        max_freq: float | None = None,
        max_frames: int = 400,
    ) -> SpectrumNoiseFloor:
        """Percentile over frames (centred at `frame_times`) of the power
        spectral density of each channel, scaled as wavetracker's spectrogram."""
        from ..io import open_recording

        frame_times = np.asarray(frame_times, float)
        if len(frame_times) > max_frames:
            pick = np.linspace(0, len(frame_times) - 1, max_frames).round().astype(int)
            frame_times = frame_times[pick]
        win = np.hanning(nfft)  # symmetric, as torch.hann_window(periodic=False)
        with open_recording(path) as data:
            rate = float(data.rate)
            scale = 2.0 / (rate * np.sum(win**2))
            freqs = np.fft.rfftfreq(nfft, 1.0 / rate)
            # None: whole spectrum up to Nyquist
            kmax = (
                len(freqs)
                if max_freq is None
                else int(np.searchsorted(freqs, max_freq)) + 3
            )
            spectra = []
            for t in frame_times:
                s0 = round(t * rate - nfft / 2)
                if s0 < 0 or s0 + nfft > len(data):
                    continue
                seg = np.asarray(data[s0 : s0 + nfft], dtype=np.float64)
                if channels is not None:
                    seg = seg[:, channels]
                seg = seg - seg.mean(0)
                z = np.fft.rfft(seg * win[:, None], axis=0)[:kmax]
                spectra.append(np.abs(z) ** 2 * scale)
        if not spectra:
            raise ValueError("No frames of the recording within the survey")
        power = np.percentile(np.stack(spectra), quantile, axis=0)
        return cls(freqs[:kmax], power)


class DetectionNoiseFloor:
    """Percentile of the detection powers of each channel near a frequency."""

    def __init__(
        self,
        fund_v: np.ndarray,
        sign_v: np.ndarray,
        quantile: float = 10.0,
        band: float = 20.0,
        min_count: int = 200,
    ):
        order = np.argsort(fund_v)
        self.f = np.asarray(fund_v)[order]
        self.p = np.asarray(sign_v)[order]
        self.quantile, self.band, self.min_count = quantile, band, min_count

    def amplitude(self, f: float) -> np.ndarray:
        band = self.band
        while True:
            lo, hi = np.searchsorted(self.f, [f - band, f + band])
            if hi - lo >= self.min_count or hi - lo == len(self.f):
                break
            band *= 2
        return np.sqrt(np.percentile(self.p[lo:hi], self.quantile, axis=0))
