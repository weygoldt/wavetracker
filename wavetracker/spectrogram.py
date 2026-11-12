"""Multi-channel power spectrograms on the GPU (or CPU) with PyTorch."""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.special import polygamma


def available(device: str | torch.device) -> bool:
    """True if `device` exists on this machine."""
    d = torch.device(device)
    if d.type == "cuda":
        return torch.cuda.is_available() and (d.index or 0) < torch.cuda.device_count()
    if d.type == "mps":
        return torch.backends.mps.is_available()
    return d.type == "cpu"


def get_device(name: str = "auto", prefer: str | None = None) -> torch.device:
    """Resolve a device name. 'auto' is GPU first: `prefer` (e.g. the device a
    run used) if this machine has it, else CUDA, MPS, CPU."""
    if name != "auto":
        return torch.device(name)
    if prefer is not None and available(prefer):
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def out_of_memory(err: BaseException) -> bool:
    """True if `err` is a GPU running out of memory (CUDA or MPS)."""
    if isinstance(err, torch.OutOfMemoryError):
        return True
    return isinstance(err, RuntimeError) and "out of memory" in str(err).lower()


def free_gpu_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


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

    def stft(self, block: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Complex spectrum of each channel, shape (channels, freqs, frames),
        scaled so that ``abs(X)**2`` is the power spectral density. The phase
        refers to each frame's first sample; phase differences between
        channels (e.g. the relative sign of two electrodes) are meaningful."""
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
        return spec.mul_(self.scale.sqrt())

    @staticmethod
    def power(spec: torch.Tensor) -> torch.Tensor:
        return spec.real.square().add_(spec.imag.square())

    def __call__(self, block: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Power of each channel, shape (channels, freqs, frames)."""
        return self.power(self.stft(block))


def decibel(power: torch.Tensor, min_power: float = 1e-20) -> torch.Tensor:
    """10*log10(power); values below `min_power` become -inf."""
    db = 10.0 * torch.log10(power.clamp_min(min_power))
    return db.masked_fill_(power <= min_power, -math.inf)


def frame_noise_std(
    log_spec: torch.Tensor, chunk: int = 128, nbins: int = 100
) -> torch.Tensor:
    """Noise standard deviation of each frame of a dB spectrogram (freqs, frames),
    NaN for frames without data (digital silence).

    Uses the upper-middle part of the spectrum (bins n/2..3n/4), which is
    assumed to contain no fish. The spectrum is detrended in chunks of
    `chunk` bins and the std is taken as half the width of the histogram at
    1/sqrt(e) of its maximum, as in the original wavetracker.
    """
    n = log_spec.shape[0]
    seg = log_spec[n // 2 : n * 3 // 4].T  # (frames, bins)
    seg = seg[:, : (seg.shape[1] // chunk) * chunk]
    if seg.shape[1] == 0:
        return torch.full((log_spec.shape[1],), torch.nan, device=log_spec.device)
    seg = torch.nan_to_num(seg, neginf=float("nan"))
    has_data = seg.isfinite().any(-1)
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
    return torch.where(has_data, std, torch.nan)


def estimate_noise_std(
    log_spec: torch.Tensor, chunk: int = 128, nbins: int = 100
) -> float:
    """Mean noise standard deviation of the frames with data, see `frame_noise_std`."""
    return float(frame_noise_std(log_spec, chunk, nbins).nanmean())


def band_noise_std(
    log_spec: torch.Tensor, lag: int, band_bins: int, stride: int = 4
) -> torch.Tensor:
    """Noise standard deviation per band of a dB spectrogram (freqs, frames),
    shape (bands, frames), NaN where unknown.

    The median change of a band's bins between windows `lag` frames apart, which
    do not overlap. Fish do not raise it, however many there are, since a fish's
    bins barely change. It reads 1.08 to 1.14 of `frame_noise_std` on Gaussian
    noise.
    """
    n_bins, n_frames = log_spec.shape
    n_bands = n_bins // band_bins
    device = log_spec.device
    if n_frames <= lag or n_bands == 0:
        return torch.full((max(n_bands, 1), n_frames), torch.nan, device=device)
    sub = log_spec[: n_bands * band_bins]
    first = torch.arange(0, n_frames - lag, stride, device=device)
    change = (sub[:, first + lag] - sub[:, first]).abs()
    change = torch.where(change.isfinite(), change, torch.nan)
    per_band = (change.reshape(n_bands, band_bins, len(first)).nanmedian(1).values) / (
        0.6745 * math.sqrt(2.0)
    )
    # back onto all frames: each value belongs to the middle of its two windows
    at = (first + lag // 2).clamp_max(n_frames - 1)
    nearest = torch.searchsorted(at, torch.arange(n_frames, device=device))
    return per_band[:, nearest.clamp_max(len(first) - 1)]


def min_noise_std(n_channels: int) -> float:
    """Lower bound of the noise estimate of a spectrum summed over `n_channels`.

    The power of noise summed over N channels is chi-square with 2N degrees of
    freedom, which fixes the spread of its logarithm, and correlated channels only
    make it wider. `frame_noise_std` reads 0.74 to 0.91 of the analytic value on
    Gaussian noise, so below 0.7 of it the spectrum is not noise: for example one
    click dominating the window, whose spectrum is smooth.
    """
    return 0.7 * 10.0 / math.log(10.0) * math.sqrt(float(polygamma(1, n_channels)))
