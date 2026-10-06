"""Multi-channel power spectrograms on the GPU (or CPU) with PyTorch."""

from __future__ import annotations

import math

import numpy as np
import torch


def get_device(name: str = "auto") -> torch.device:
    """Resolve 'auto' to the best available device."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def step_size(nfft: int, overlap_frac: float) -> int:
    return max(1, int(nfft * (1.0 - overlap_frac)))


def frequencies(nfft: int, rate: float) -> np.ndarray:
    return np.fft.rfftfreq(nfft, 1.0 / rate)


class PowerSpectrogram:
    """Computes one-sided power spectral densities of frame-aligned blocks.

    Scaling matches ``matplotlib.mlab.specgram(scale_by_freq=True)``, i.e.
    units of signal**2 / Hz.
    """

    def __init__(self, nfft: int, step: int, rate: float, device: torch.device):
        self.nfft = nfft
        self.step = step
        self.rate = rate
        self.device = device
        self.window = torch.hann_window(
            nfft, periodic=False, dtype=torch.float32, device=device
        )
        scale = torch.full((nfft // 2 + 1, 1), 2.0, device=device)
        scale[0] = 1.0
        if nfft % 2 == 0:
            scale[-1] = 1.0
        self.scale = scale / (rate * self.window.pow(2).sum())

    def __call__(self, block: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Power of each channel, shape (channels, freqs, frames)."""
        x = torch.as_tensor(block, device=self.device)
        x = x - x.mean(dim=-1, keepdim=True)
        spec = torch.stft(
            x,
            n_fft=self.nfft,
            hop_length=self.step,
            window=self.window,
            center=False,
            return_complex=True,
        )
        return spec.real.square_().add_(spec.imag.square()).mul_(self.scale)


def decibel(power: torch.Tensor, min_power: float = 1e-20) -> torch.Tensor:
    """10*log10(power); values below `min_power` become -inf."""
    db = 10.0 * torch.log10(power.clamp_min(min_power))
    return db.masked_fill_(power <= min_power, -math.inf)


def estimate_noise_std(
    log_spec: torch.Tensor, chunk: int = 128, nbins: int = 100
) -> float:
    """Noise standard deviation of a dB spectrogram (freqs, frames).

    Uses the upper-middle part of the spectrum (bins n/2..3n/4), which is
    assumed to contain no fish. The spectrum is detrended in chunks of
    `chunk` bins and the std is taken as half the width of the histogram at
    1/sqrt(e) of its maximum, as in the original wavetracker.
    """
    n = log_spec.shape[0]
    seg = log_spec[n // 2 : n * 3 // 4].T  # (frames, bins)
    seg = seg[:, : (seg.shape[1] // chunk) * chunk]
    if seg.shape[1] == 0:
        return float("nan")
    seg = torch.nan_to_num(seg, neginf=float("nan"))
    chunks = seg.reshape(seg.shape[0], -1, chunk)
    detrended = (
        chunks - chunks.nanmean(-1, keepdim=True) + seg.nanmean(-1)[:, None, None]
    )
    detrended = detrended.reshape(seg.shape[0], -1)

    lo = detrended.nan_to_num(nan=float("inf")).amin(-1, keepdim=True)
    hi = detrended.nan_to_num(nan=float("-inf")).amax(-1, keepdim=True)
    width = (hi - lo).clamp_min(1e-12) / nbins
    idx = ((detrended - lo) / width).floor().clamp_(0, nbins - 1)
    idx = idx.nan_to_num(nan=0).long()
    hist = torch.zeros(seg.shape[0], nbins, device=seg.device)
    hist.scatter_add_(1, idx, torch.ones_like(detrended).nan_to_num_(nan=0))
    above = hist > hist.amax(-1, keepdim=True) / math.sqrt(math.e)
    bins = torch.arange(nbins, device=seg.device).expand_as(hist)
    first = torch.where(above, bins, nbins).amin(-1)
    last = torch.where(above, bins, -1).amax(-1)
    std = 0.5 * (last - first + 1) * width.squeeze(-1)
    return float(std.mean())
