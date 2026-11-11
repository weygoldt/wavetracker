"""Learned fundamental detector ("deep salience").

A small convolutional network maps the electrode-summed power spectrogram to a
*salience* per time-frequency bin: the probability that a fish's fundamental
is there. Peaks of the salience along frequency become detections, which the
tracker and linker take over as with the classic detector
(``detector.method: learned``).

Input features, all in physical units so that a model works at any sampling
rate with the same spectrogram resolution:

* the log power of each frame, minus its median and divided by its spread
  in bands of ``band_hz`` (levels and gains do not matter, only how far a bin
  stands out of the local noise);
* a *harmonic stack*: channel ``h`` at candidate bin ``k`` holds the feature
  at bin ``h * k`` (on a linear frequency axis starting at 0 Hz, harmonic h
  of bin k is bin h*k). A max over the few bins an off-centre harmonic can
  move to makes this tolerant to sub-bin frequencies. Channels below 1
  (f/2, f/3) show what is an octave or a twelfth below a candidate: a
  harmonic of a lower fish has its fundamental there.

The network never pools or strides along frequency, so the salience keeps the
full spectrogram resolution; time dilations give it seconds of context, and
an optional temporal stack (kernels along time only, per frequency bin) tens
of seconds more. A second output is the sub-bin offset of the fundamental.

``StreamingSalience`` runs a model over a recording block by block, with the
neighbouring blocks as context, so block boundaries leave no trace.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

FORMAT = "wavetracker-salience"
GRID_TOLERANCE = 0.01  # relative mismatch of resolution / frame step accepted


@dataclass(frozen=True)
class SalienceSpec:
    """Everything needed to rebuild a model and its input features."""

    freq_resolution: float
    """Frequency resolution of the spectrogram the model was trained on [Hz]."""
    frame_step: float
    """Time step between spectrogram frames [s]."""
    harmonics: tuple[float, ...] = (1 / 3, 1 / 2, 1, 2, 3, 4, 5)
    """Channels of the harmonic stack (multiples of the candidate frequency)."""
    band_hz: float = 100.0
    """Width of the bands in which frames are normalised [Hz]."""
    channels: int = 32
    kernel_bins: int = 5
    kernel_frames: int = 3
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 1)
    """Time dilation of each convolution layer (one layer per entry)."""
    temporal_dilations: tuple[int, ...] = ()
    """Time dilation of each layer of the temporal stack after the
    convolutions: kernels along time only (``temporal_kernel`` frames, one
    bin), weights shared across bins, so each frequency bin gets tens of
    seconds of context at little cost. Empty: no temporal stack (v1)."""
    temporal_kernel: int = 3
    temporal_kernel_bins: int = 1
    """Extent of the temporal kernels along frequency (1: per bin). Wider
    kernels let the temporal stack follow lines that drift across bins."""
    temporal_bin_dilations: tuple[int, ...] = ()
    """Frequency dilation of each temporal layer (empty: 1 for all)."""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SalienceSpec:
        valid = {f.name for f in fields(cls)}
        kw = {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in d.items()
            if k in valid
        }
        return cls(**kw)


def normalise(
    log_power: torch.Tensor, df: float, band_hz: float = 100.0
) -> torch.Tensor:
    """Log power (frames, bins) [dB] -> how far each bin stands out of the
    noise of its frame: (L - median) / (1.4826 MAD), with median and MAD
    taken in bands of `band_hz` and interpolated linearly between band
    centres. Non-finite values (digital silence) become 0."""
    T, K = log_power.shape
    nb = max(8, round(band_hz / df))
    n_bands = max(1, math.ceil(K / nb))
    L = log_power.float()
    finite = torch.isfinite(L)
    Lb = torch.where(finite, L, torch.full_like(L, float("nan")))
    Lb = F.pad(Lb, (0, n_bands * nb - K), value=float("nan")).reshape(T, n_bands, nb)
    med = Lb.nanmedian(dim=-1).values
    mad = (Lb - med[..., None]).abs().nanmedian(dim=-1).values
    med = torch.nan_to_num(med, nan=0.0)
    scale = torch.nan_to_num(1.4826 * mad, nan=1.0).clamp_min(1e-3)
    if n_bands > 1:
        size = n_bands * nb
        med = F.interpolate(med[None], size=size, mode="linear", align_corners=False)[0]
        scale = F.interpolate(
            scale[None], size=size, mode="linear", align_corners=False
        )[0]
        med, scale = med[:, :K], scale[:, :K]
    else:
        med, scale = med.expand(T, K), scale.expand(T, K)
    z = (L - med) / scale
    return torch.where(finite, z, torch.zeros_like(z))


def _radius(h: float) -> int:
    """Bins an off-centre harmonic h can move: a fundamental within half a bin
    of the candidate puts harmonic h within h/2 bins of h*k."""
    if h < 1:
        return 1
    if h == 1:
        return 0
    return math.floor(h / 2 + 0.5)


def harmonic_stack(
    z: torch.Tensor, k_lo: int, k_hi: int, harmonics: tuple[float, ...]
) -> torch.Tensor:
    """Features (frames, bins) -> stack (len(harmonics), frames, candidates)
    for the candidate fundamental bins k_lo..k_hi. Harmonics beyond the
    spectrum are filled with the noise level (0)."""
    K = z.shape[1]
    k = torch.arange(k_lo, k_hi + 1, device=z.device, dtype=torch.float64)
    out = []
    for h in harmonics:
        r = _radius(h)
        zp = z if r == 0 else F.max_pool1d(z[None], 2 * r + 1, stride=1, padding=r)[0]
        idx = torch.round(h * k).long()
        valid = idx < K
        ch = zp[:, idx.clamp(max=K - 1)]
        out.append(torch.where(valid[None, :], ch, torch.zeros_like(ch)))
    return torch.stack(out)


class SalienceNet(nn.Module):
    """Fully convolutional over (time, frequency); no pooling or stride along
    frequency. Input (batch, harmonics, frames, bins); output the salience
    logits and the sub-bin offset (in bins, within +-0.5), each
    (batch, frames, bins)."""

    def __init__(self, spec: SalienceSpec):
        super().__init__()
        self.spec = spec
        c, kt, kf = spec.channels, spec.kernel_frames, spec.kernel_bins
        convs, norms = [], []
        n_in = len(spec.harmonics)
        for d in spec.dilations:
            convs.append(
                nn.Conv2d(
                    n_in, c, (kt, kf), padding=(d * (kt // 2), kf // 2), dilation=(d, 1)
                )
            )
            norms.append(nn.BatchNorm2d(c))
            n_in = c
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)
        kt, kb = spec.temporal_kernel, spec.temporal_kernel_bins
        bin_dil = spec.temporal_bin_dilations or (1,) * len(spec.temporal_dilations)
        if len(bin_dil) != len(spec.temporal_dilations):
            raise ValueError("temporal_bin_dilations needs one entry per layer")
        self.temporal = nn.ModuleList(
            nn.Conv2d(
                c,
                c,
                (kt, kb),
                padding=(d * (kt // 2), db * (kb // 2)),
                dilation=(d, db),
            )
            for d, db in zip(spec.temporal_dilations, bin_dil, strict=True)
        )
        self.temporal_norms = nn.ModuleList(
            nn.BatchNorm2d(c) for _ in spec.temporal_dilations
        )
        self.head = nn.Conv2d(c, 2, 1)
        self.checkpoint = False
        """Training only: recompute each layer in the backward pass instead
        of keeping its activations (long crops of the temporal stack)."""

    @property
    def context_frames(self) -> int:
        """Frames on each side that influence an output frame."""
        s = self.spec
        trunk = sum(d * (s.kernel_frames // 2) for d in s.dilations)
        return trunk + sum(d * (s.temporal_kernel // 2) for d in s.temporal_dilations)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        layers = list(zip(self.convs, self.norms, strict=True)) + list(
            zip(self.temporal, self.temporal_norms, strict=True)
        )
        y = None
        for conv, norm in layers:
            inp = x if y is None else y
            if self.checkpoint and self.training and torch.is_grad_enabled():
                h = torch.utils.checkpoint.checkpoint(
                    _layer, conv, norm, inp, use_reentrant=False
                )
            else:
                h = _layer(conv, norm, inp)
            y = h if y is None else y + h
        o = self.head(y)
        return o[:, 0], 0.5 * torch.tanh(o[:, 1])


def _layer(conv: nn.Module, norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return F.relu(norm(conv(x)))


def decode(
    logits: torch.Tensor,
    offset: torch.Tensor,
    threshold: float,
    k_lo: int,
    df: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Salience peaks along frequency above `threshold` -> (frame, bin,
    frequency [Hz], salience). A peak is at least its lower neighbour and
    above its upper one, so plateaus give one detection."""
    p = torch.sigmoid(logits.float())
    left = F.pad(p, (1, 0), value=-1.0)[:, :-1]
    right = F.pad(p, (0, 1), value=-1.0)[:, 1:]
    peak = (p >= threshold) & (p >= left) & (p > right)
    fr, kk = torch.nonzero(peak, as_tuple=True)
    bins = kk + k_lo
    freq = (bins.double() + offset[fr, kk].double()) * df
    return (
        fr.cpu().numpy().astype(np.int64),
        bins.cpu().numpy().astype(np.int64),
        freq.cpu().numpy(),
        p[fr, kk].cpu().numpy().astype(np.float32),
    )


def save_model(
    path: str | Path, model: SalienceNet, spec: SalienceSpec, meta: dict | None = None
):
    torch.save(
        {
            "format": FORMAT,
            "version": 1,
            "spec": asdict(spec),
            "state": model.state_dict(),
            "meta": meta or {},
        },
        path,
    )


def load_model(path: str | Path, device: str | torch.device = "cpu"):
    """-> (model in eval mode, spec, meta)."""
    ck = torch.load(path, map_location=device, weights_only=True)
    if ck.get("format") != FORMAT:
        raise ValueError(f"{path} is not a wavetracker salience model")
    spec = SalienceSpec.from_dict(ck["spec"])
    model = SalienceNet(spec).to(device)
    model.load_state_dict(ck["state"])
    model.eval()
    return model, spec, ck.get("meta", {})


def default_model_path() -> Path:
    return Path(__file__).parent / "models" / "salience.pt"


def _mirror(x: torch.Tensor, left: int, right: int) -> torch.Tensor:
    """Extend frames (frames, bins) by `left` / `right` frames mirrored at
    the ends (without repeating the end frame), repeating the mirror if the
    data are shorter than the padding."""
    while left > 0 or right > 0:
        n = len(x) - 1
        if n < 1:
            reps = [x] * (1 + max(left, 0) + max(right, 0))
            return torch.cat(reps)
        a, b = min(max(left, 0), n), min(max(right, 0), n)
        x = torch.cat([x[1 : a + 1].flip(0), x, x[len(x) - 1 - b : len(x) - 1].flip(0)])
        left, right = left - a, right - b
    return x


class StreamingSalience:
    """Runs a salience model over consecutive spectrogram blocks.

    Each block is evaluated once the next one has arrived, with
    ``context_frames`` of its neighbours on both sides, so the result does not
    depend on where blocks start. ``push`` returns the detections of the
    blocks that are complete, ``finish`` those of the last one; both as a list
    of ((frame, bin, freq, salience), payload) with frames relative to the
    block the payload belongs to.
    """

    def __init__(
        self,
        model: SalienceNet,
        spec: SalienceSpec,
        freqs: np.ndarray,
        frame_step: float,
        min_freq: float,
        max_freq: float,
        threshold: float,
        device: torch.device,
    ):
        df = float(freqs[1])
        if abs(df - spec.freq_resolution) > GRID_TOLERANCE * spec.freq_resolution:
            raise ValueError(
                f"Spectrogram resolution {df:.4f} Hz does not match the model's "
                f"{spec.freq_resolution:.4f} Hz: set spectrogram.freq_resolution "
                f"to {spec.freq_resolution:.4f} (and nfft to null)."
            )
        if abs(frame_step - spec.frame_step) > GRID_TOLERANCE * spec.frame_step:
            raise ValueError(
                f"Frame step {frame_step:.4f} s does not match the model's "
                f"{spec.frame_step:.4f} s: check spectrogram.overlap_frac."
            )
        self.model, self.spec, self.device = model, spec, device
        self.df, self.threshold = df, float(threshold)
        self.k_lo = max(1, math.ceil(min_freq / df))
        self.k_hi = min(len(freqs) - 1, math.floor(max_freq / df))
        if self.k_hi < self.k_lo:
            raise ValueError("Empty frequency range for the learned detector.")
        self.k_need = min(len(freqs), math.ceil(max(spec.harmonics) * self.k_hi) + 4)
        self.ctx = model.context_frames
        self._tail: torch.Tensor | None = None  # last ctx frames before the queue
        self._queue: list[tuple[torch.Tensor, Any]] = []
        self.model_info: dict[str, Any] = {}

    @classmethod
    def from_config(cls, dc, freqs, frame_step, min_freq, max_freq, device):
        path = Path(dc.model).expanduser() if dc.model else default_model_path()
        model, spec, meta = load_model(path, device)
        s = cls(
            model, spec, freqs, frame_step, min_freq, max_freq, dc.threshold, device
        )
        s.model_info = {"path": str(path.resolve()), **meta}
        return s

    def features(self, log_power: torch.Tensor) -> torch.Tensor:
        """Log power (bins, frames) of one block -> normalised features
        (frames, bins needed by the harmonic stack)."""
        k = min(self.k_need, log_power.shape[0])
        return normalise(log_power[:k].T.to(self.device), self.df, self.spec.band_hz)

    def push(self, z: torch.Tensor, payload: Any = None) -> list:
        """Add the features of the next block; returns the blocks that now
        have enough context on both sides."""
        self._queue.append((z, payload))
        out = []
        while self._queue:
            after = sum(len(q[0]) for q in self._queue[1:])
            if after < self.ctx:
                break
            out.append(self._pop())
        return out

    def finish(self) -> list:
        """The remaining blocks (the recording ends after them)."""
        out = []
        while self._queue:
            out.append(self._pop())
        self._tail = None
        return out

    def _pop(self):
        z, payload = self._queue.pop(0)
        following = [q[0] for q in self._queue]
        right = torch.cat(following)[: self.ctx] if following and self.ctx else None
        det = self._run(self._tail, z, right)
        self._advance(z)
        return det, payload

    def _advance(self, z: torch.Tensor) -> None:
        tail = z if self._tail is None else torch.cat([self._tail, z])
        self._tail = tail[-self.ctx :] if self.ctx else tail[:0]

    @torch.no_grad()
    def _run(self, left, z, right):
        parts = [p for p in (left, z, right) if p is not None and len(p)]
        x = torch.cat(parts)
        n_left = 0 if left is None else len(left)
        # the recording starts or ends here: mirror the data into the missing
        # context (the network was trained with real context on both sides)
        pad_l = self.ctx - n_left
        pad_r = self.ctx - (0 if right is None else len(right))
        x = _mirror(x, pad_l, pad_r)
        n_left += max(pad_l, 0)
        stack = harmonic_stack(x, self.k_lo, self.k_hi, self.spec.harmonics)
        logits, offset = self.model(stack[None])
        sl = slice(n_left, n_left + len(z))
        return decode(logits[0, sl], offset[0, sl], self.threshold, self.k_lo, self.df)
