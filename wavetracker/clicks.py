"""Removal of impulsive noise ("static") from the raw traces.

Electrodes out of the water pick up static: clicks of a few milliseconds, up to
4 orders of magnitude above the noise, several per second. A spectrogram cannot
deal with them. One click in an FFT window has a smooth spectrum, which makes the
noise estimate collapse, and two make an evenly spaced ripple, which the harmonic
group detector takes for a fish. Weaker crackle in the water triples the noise
estimate and buries weak fish. So clicks are removed before the spectrogram.

Each channel is split by its spectrum over the block:

- Clicks are searched in the high band, which fish do not reach (the band the
  noise estimate relies on too), so that even a chirp of a fish right at an
  electrode does not look like one. Long loud events (a boat, rain) are real
  signals and stay.
- Cutting clicks out is not enough, because a hole in a loud fish is itself a
  click of the fish's amplitude. What stands out of the spectral floor (fish,
  their chirps, hum) is left untouched, and the clicks are bridged by straight
  lines in the rest, which is noise and the clicks themselves. The spectrum does
  not tell when a fish is there, so a hole gets as much of the narrowband part as
  the samples around it hold: nothing of a fish that arrives later in the block.

Channels without any click are returned exactly as they are.
"""

from __future__ import annotations

import math

import scipy.fft
import torch
import torch.nn.functional as F

from .config import ClickConfig


def _spectral_floor(power: torch.Tensor, df: float, cfg: ClickConfig) -> torch.Tensor:
    """Smooth floor of a power spectrum (channels, freqs).

    A low quantile over a wide range, of medians over narrow ranges: a fish can
    fill most of the spectrum around it with chirps and its harmonics, and the
    floor must stay below that however loud the fish is.
    """
    n_bins = power.shape[-1]
    chunk = min(max(8, round(cfg.line_width / df)), n_bins)
    n_chunks = n_bins // chunk
    medians = (
        power[..., : n_chunks * chunk]
        .reshape(power.shape[0], n_chunks, chunk)
        .median(-1)
        .values
    )
    width = max(3, round(cfg.floor_width / (chunk * df))) | 1
    padded = F.pad(medians[:, None], (width // 2, width // 2), mode="replicate")[:, 0]
    rank = max(1, math.ceil(cfg.floor_quantile * width))
    floor = padded.unfold(-1, width, 1).kthvalue(rank, -1).values
    floor = floor.repeat_interleave(chunk, -1)
    return F.pad(floor[:, None], (0, n_bins - floor.shape[-1]), mode="replicate")[:, 0]


def _periodic(x: torch.Tensor, n_fft: int) -> torch.Tensor:
    """`x` extended to `n_fft` samples that lead smoothly back to its start.

    A sine that is not periodic in the FFT window spreads into skirts below any
    margin, which would ring from both ends of the block in the part repaired.
    """
    n = x.shape[-1]
    m = n_fft - n
    if m == 0:
        return x
    # point reflections continue the value and the slope at both ends
    after = 2 * x[..., -1:] - x[..., -(m + 1) : -1].flip(-1)
    before = 2 * x[..., :1] - x[..., 1 : m + 1].flip(-1)
    k = (torch.arange(m, device=x.device, dtype=x.dtype) + 0.5) / m
    fade = torch.cos(0.5 * torch.pi * k).square()
    return torch.cat([x, after * fade + before * (1 - fade)], dim=-1)


def _long_runs(hot: torch.Tensor, length: int) -> torch.Tensor:
    """True in runs of at least `length` consecutive True along the last axis."""
    n = hot.shape[-1]
    if length > n:
        return torch.zeros_like(hot)
    total = F.pad(hot.long().cumsum(-1), (1, 0))
    full = (total[..., length:] - total[..., :-length]) == length
    starts = F.pad(full.long().cumsum(-1), (1, 0))
    i = torch.arange(n, device=hot.device)
    lo = (i - length + 1).clamp_min(0)
    hi = (i + 1).clamp_max(full.shape[-1])
    return (starts[..., hi] - starts[..., lo]) > 0


def _runs_holding(runs: torch.Tensor, seeds: torch.Tensor) -> torch.Tensor:
    """The runs of True in `runs` that hold a True of `seeds`, along the last axis."""
    # every sample of a run has the same count of False before it
    labels = (~runs).long().cumsum(-1)
    held = torch.zeros(
        runs.shape[0], runs.shape[-1] + 1, device=runs.device, dtype=torch.long
    )
    held.scatter_add_(-1, labels, (seeds & runs).long())
    return runs & (held.gather(-1, labels) > 0)


def find_clicks(x: torch.Tensor, rate: float, cfg: ClickConfig) -> torch.Tensor:
    """Boolean mask of the short excursions of `x` (channels, samples), which
    should be free of what is not a click: see `remove_clicks`."""
    n = x.shape[-1]
    size = max(1, round(cfg.resolution * rate))
    n_bins = n // size
    if n_bins == 0:
        return torch.zeros_like(x, dtype=torch.bool)
    magnitude = x.abs()
    # median over what is not digital silence: half a silent block has noise too
    sample = magnitude[..., :: cfg.sample_stride]
    sigma = torch.where(sample > 0, sample, torch.nan).nanmedian(-1, True).values
    # the largest value per bin: fast, and bridges zero crossings within a click
    envelope = (
        magnitude[..., : n_bins * size].reshape(x.shape[0], n_bins, size).amax(-1)
    )
    sigma = sigma / 0.6745
    longest = max(1, round(cfg.max_duration / cfg.resolution))
    # hysteresis: a click must reach the threshold, and then lasts for as long as
    # it stays above the lower one. A long loud event holds no click, and neither
    # does a long event around one.
    trigger = envelope > cfg.threshold * sigma
    trigger &= ~_long_runs(trigger, longest)
    hot = _runs_holding(envelope > cfg.extent_threshold * sigma, trigger)
    hot &= ~_long_runs(hot, longest)
    # The join of the periodic extension shows in the high band. Windows weigh
    # these samples by about -80 dB here, and at full weight in the next block.
    edge = min(round(cfg.edge / cfg.resolution), n_bins // 2)
    if edge:
        hot[..., :edge] = False
        hot[..., -edge:] = False
    guard = round(cfg.guard / cfg.resolution)
    if guard:
        hot = F.max_pool1d(hot.float()[:, None], 2 * guard + 1, 1, guard)[:, 0] > 0
    mask = hot.repeat_interleave(size, -1)
    return F.pad(mask, (0, n - mask.shape[-1]))


def bridge(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Replace the masked samples by lines between their nearest good samples."""
    n = x.shape[-1]
    i = torch.arange(n, device=x.device).expand_as(x)
    before = torch.where(mask, -1, i).cummax(-1).values
    after = torch.where(mask, n, i).flip(-1).cummin(-1).values.flip(-1)
    # no good sample on one side: hold the other (an all-masked channel: its first)
    lo = torch.where(before < 0, after, before).clamp(0, n - 1)
    hi = torch.where(after >= n, lo, after).clamp(0, n - 1)
    weight = (i - lo).to(x.dtype) / (hi - lo).clamp_min(1).to(x.dtype)
    line = x.gather(-1, lo) * (1 - weight) + x.gather(-1, hi) * weight
    return torch.where(mask, line, x)


def _running_sum(x: torch.Tensor, width: int) -> torch.Tensor:
    """Sum over `width` samples around each sample of `x` (channels, samples)."""
    n = x.shape[-1]
    total = F.pad(x.double().cumsum(-1), (1, 0))
    i = torch.arange(n, device=x.device)
    lo = (i - width // 2).clamp_min(0)
    hi = (i + width // 2 + 1).clamp_max(n)
    return total[..., hi] - total[..., lo]


def _local_part(
    x: torch.Tensor, part: torch.Tensor, good: torch.Tensor, width: int
) -> torch.Tensor:
    """`part` scaled to what the good samples of `x` around each sample hold of it.

    The least squares gain, within 0 and 1: 1 where a fish is there and 0 where it
    is not. All part of `part` scale together, and a hole is short: that is enough.
    """
    num = _running_sum(x * part * good, width)
    den = _running_sum(part * part * good, width)
    gain = (num / den.clamp_min(torch.finfo(den.dtype).tiny)).clamp(0, 1)
    return part * gain.to(x.dtype)


def remove_clicks(
    x: torch.Tensor, rate: float, cfg: ClickConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """Repair the clicks in `x` (channels, samples).

    Returns the repaired traces and the fraction of samples repaired on each
    channel. Channels without clicks are returned unchanged, bit for bit.
    """
    n = x.shape[-1]
    if n < 64:  # too short to tell anything, and to make periodic
        return x, torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    n_fft = scipy.fft.next_fast_len(n + min(round(cfg.fade * rate), n // 2), real=True)
    spectrum = torch.fft.rfft(_periodic(x, n_fft))
    power = spectrum.real.square() + spectrum.imag.square()
    df = rate / n_fft
    narrow = power > _spectral_floor(power, df, cfg) * 10 ** (cfg.line_margin / 10)
    freqs = torch.arange(power.shape[-1], device=x.device, dtype=x.dtype) * df
    # a smooth edge: a sharp one would ring around the clicks
    position = ((freqs / rate - cfg.high_band) / cfg.high_band_edge).clamp(0, 1)
    gain = torch.sin(0.5 * torch.pi * position).square()
    high = torch.fft.irfft(spectrum * gain * ~narrow, n=n_fft)[..., :n]
    # round-off of the FFT is not noise: digital silence stays silent
    mask = find_clicks(torch.where(x == 0, 0.0, high), rate, cfg)
    fraction = mask.to(x.dtype).mean(-1)
    hit = mask.any(-1)
    if not hit.any():
        return x, fraction
    # channel by channel: the index arrays of a block would take gigabytes at once
    out = x.clone()
    width = max(1, round(cfg.local_window * rate))
    for c in hit.nonzero()[:, 0]:
        tonal = torch.fft.irfft(spectrum[c] * narrow[c], n=n_fft)[:n]
        tonal = _local_part(x[c], tonal, ~mask[c], width)
        # good samples are x exactly: only the masked ones change
        repaired = tonal + bridge(x[c] - tonal, mask[c])
        out[c] = torch.where(mask[c], repaired, x[c])
    return out, fraction
