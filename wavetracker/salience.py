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
PAIR_SPACING = 2
"""A salience peak with another peak within this many bins is not split by
the pair head (the salience resolves that pair; two equal tones are about two
bins apart where they start to make two peaks)."""


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
    pair_head: bool = False
    """A second head says, at every bin, whether two fundamentals closer than
    the window resolves (one spectral peak) are there, with the sub-bin
    offsets of both: a salience peak with a high pair probability becomes two
    detections (see `decode`). False: one detection per peak (v1)."""
    pair_reach: float = 2.5
    """Largest offset of a fish of a pair from its bin [bins of the model's
    grid] the pair head can give."""
    pair_dilations: tuple[int, ...] = ()
    """Time dilation of each layer of a branch between the trunk and the
    pair head (residual convolutions as in the trunk; empty: the head reads
    the trunk directly). Their context adds to the trunk's."""
    electrode_channels: int = 0
    """Width of the electrode branch (0: none, the model sees only the
    electrode sum). One encoder with shared weights reads every electrode's
    spatial features at the candidate bins (`electrode_features`); the mean
    and the maximum over electrodes, for any number of electrodes, join the
    network through a gate that starts at zero."""
    electrode_dilations: tuple[int, ...] = (1, 2, 4, 8)
    """Time dilation of each layer of the electrode encoder."""
    electrode_kernel_bins: int = 3
    electrode_into: str = "pair"
    """Where the electrode branch joins: "pair" (the input of the pair
    branch: the salience stays that of the sum) or "trunk" (after the
    convolutions, before the temporal stack)."""

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


ELECTRODE_FEATURES = 3


def electrode_features(stft: torch.Tensor, k_lo: int, k_hi: int) -> torch.Tensor:
    """Complex spectra (electrodes, bins, frames) -> spatial features (frames,
    electrodes, 3, candidate bins k_lo..k_hi) for the electrode branch: the
    log10 share of the electrode in the summed power, and its phase relative
    to the strongest electrode of the bin (cos, sin, scaled by the square root
    of the share). One fish gives every electrode a fixed share and a phase
    near 0 or pi; two fish in one bin make both move with their beat. Levels,
    gains and the common phase do not matter."""
    X = stft[:, k_lo : k_hi + 1]
    P = X.real.square() + X.imag.square()
    share = P / P.sum(0, keepdim=True).clamp_min(1e-30)
    ref = torch.gather(X, 0, P.argmax(0, keepdim=True))
    rel = X * ref.conj()
    rel = rel / rel.abs().clamp_min(1e-30)
    a = share.sqrt()
    f = torch.stack(
        [torch.log10(share.clamp_min(1e-4)), a * rel.real, a * rel.imag], 1
    )  # (E, 3, K, T)
    return f.permute(3, 0, 1, 2).float()


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
        self.pair = nn.Conv2d(c, 3, 1) if spec.pair_head else None
        kt, kf = spec.kernel_frames, spec.kernel_bins
        self.pair_convs = nn.ModuleList(
            nn.Conv2d(c, c, (kt, kf), padding=(d * (kt // 2), kf // 2), dilation=(d, 1))
            for d in (spec.pair_dilations if spec.pair_head else ())
        )
        self.pair_norms = nn.ModuleList(nn.BatchNorm2d(c) for _ in self.pair_convs)
        ce = spec.electrode_channels
        self.el_convs = nn.ModuleList()
        if ce:
            if spec.electrode_into not in ("pair", "trunk"):
                raise ValueError("electrode_into must be 'pair' or 'trunk'")
            if spec.electrode_into == "pair" and not spec.pair_head:
                raise ValueError("electrode_into 'pair' needs the pair head")
            kb = spec.electrode_kernel_bins
            n_in = ELECTRODE_FEATURES
            for d in spec.electrode_dilations:
                self.el_convs.append(
                    nn.Conv2d(
                        n_in,
                        ce,
                        (kt, kb),
                        padding=(d * (kt // 2), kb // 2),
                        dilation=(d, 1),
                    )
                )
                n_in = ce
            self.el_proj = nn.Conv2d(2 * ce, c, 1)
            self.el_gate = nn.Parameter(torch.zeros(c))
        self.checkpoint = False
        """Training only: recompute each layer in the backward pass instead
        of keeping its activations (long crops of the temporal stack)."""

    @property
    def context_frames(self) -> int:
        """Frames on each side that influence an output frame."""
        s = self.spec
        convs = sum(d * (s.kernel_frames // 2) for d in s.dilations)
        el = self.electrode_context
        if s.electrode_channels and s.electrode_into == "trunk":
            convs = max(convs, el)
        trunk = convs + sum(d * (s.temporal_kernel // 2) for d in s.temporal_dilations)
        if s.pair_head:
            if s.electrode_channels and s.electrode_into == "pair":
                trunk = max(trunk, el)
            trunk += sum(d * (s.kernel_frames // 2) for d in s.pair_dilations)
        return trunk

    @property
    def electrode_context(self) -> int:
        """Frames on each side that influence the electrode branch's output."""
        s = self.spec
        if not s.electrode_channels:
            return 0
        return sum(d * (s.kernel_frames // 2) for d in s.electrode_dilations)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Salience logits and sub-bin offset (see `outputs` for all heads)."""
        out = self.outputs(x)
        return out["logits"], out["offset"]

    def outputs(
        self,
        x: torch.Tensor,
        el: torch.Tensor | None = None,
        el_mask: torch.Tensor | None = None,
        el_start: int = 0,
    ) -> dict[str, torch.Tensor]:
        """All heads: ``logits`` and ``offset`` (batch, frames, bins) and,
        with the pair head, ``pair_logits`` (batch, frames, bins) and
        ``pair_offset`` (batch, 2, frames, bins), the offsets of the two
        fish of a pair from the bin in bins (unordered).

        `el` (batch, electrodes, 3, frames, bins): `electrode_features` of
        the same frames and bins, for the electrode branch (None: the branch
        adds nothing); `el_mask` (batch, electrodes) marks the electrodes
        that are there (None: all), so a batch can mix electrode counts.
        Training may pass fewer frames of `el`, starting at frame `el_start`
        of `x` (the branch adds nothing outside them)."""
        e = None
        if self.el_convs and el is not None:
            e = self.electrodes(el, el_mask)
            pad = x.shape[2] - el_start - e.shape[2]
            if el_start or pad:
                e = F.pad(e, (0, 0, el_start, pad))
        into_trunk = e is not None and self.spec.electrode_into == "trunk"
        y = self.trunk(x, e if into_trunk else None)
        o = self.head(y)
        out = {"logits": o[:, 0], "offset": 0.5 * torch.tanh(o[:, 1])}
        if self.pair is not None:
            yp = y if e is None or into_trunk else y + e
            q = self.pair(self.pair_branch(yp))
            out["pair_logits"] = q[:, 0]
            out["pair_offset"] = self.spec.pair_reach * torch.tanh(q[:, 1:3])
        return out

    def pair_branch(self, y: torch.Tensor) -> torch.Tensor:
        """Trunk features -> features of the pair head."""
        for conv, norm in zip(self.pair_convs, self.pair_norms, strict=True):
            y = y + _layer(conv, norm, y)
        return y

    def electrodes(
        self, el: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Electrode features (batch, electrodes, 3, frames, bins) -> gated
        contribution (batch, channels, frames, bins): the shared encoder on
        every electrode, mean and maximum over the electrodes there."""
        B, E = el.shape[:2]
        if mask is None:
            mask = torch.ones((B, E), dtype=torch.bool, device=el.device)
        total = peak = None
        for i in range(E):
            if self.checkpoint and self.training and torch.is_grad_enabled():
                h = torch.utils.checkpoint.checkpoint(
                    self._encode, el[:, i], use_reentrant=False
                )
            else:
                h = self._encode(el[:, i])
            m = mask[:, i].to(h.dtype)[:, None, None, None]
            total = h * m if total is None else total + h * m
            hm = torch.where(m > 0, h, torch.full_like(h, -1e4))
            peak = hm if peak is None else torch.maximum(peak, hm)
        n = mask.sum(1).clamp_min(1).to(total.dtype)[:, None, None, None]
        peak = torch.where(peak > -1e3, peak, torch.zeros_like(peak))
        pooled = torch.cat([total / n, peak], 1)
        return self.el_gate[None, :, None, None] * self.el_proj(pooled)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        h = None
        for conv in self.el_convs:
            z = F.relu(conv(x if h is None else h))
            h = z if h is None else h + z
        return h

    def trunk(self, x: torch.Tensor, e: torch.Tensor | None = None) -> torch.Tensor:
        """Features (batch, channels, frames, bins) before the heads; `e` is
        added after the convolutions (electrode branch into the trunk)."""
        layers = list(zip(self.convs, self.norms, strict=True)) + list(
            zip(self.temporal, self.temporal_norms, strict=True)
        )
        n_convs = len(self.convs)
        y = None
        for i, (conv, norm) in enumerate(layers):
            if e is not None and i == n_convs:
                y = y + e
            inp = x if y is None else y
            if self.checkpoint and self.training and torch.is_grad_enabled():
                h = torch.utils.checkpoint.checkpoint(
                    _layer, conv, norm, inp, use_reentrant=False
                )
            else:
                h = _layer(conv, norm, inp)
            y = h if y is None else y + h
        if e is not None and len(layers) == n_convs:
            y = y + e
        return y


def _layer(conv: nn.Module, norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return F.relu(norm(conv(x)))


def decode(
    logits: torch.Tensor,
    offset: torch.Tensor,
    threshold: float,
    k_lo: int,
    df: float,
    pair_logits: torch.Tensor | None = None,
    pair_offset: torch.Tensor | None = None,
    pair_threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Salience peaks along frequency above `threshold` -> (frame, bin,
    frequency [Hz], salience). A peak is at least its lower neighbour and
    above its upper one, so plateaus give one detection.

    With the pair head (`pair_logits`, `pair_offset` (2, frames, bins)), a
    peak whose pair probability reaches `pair_threshold` (default
    `threshold`) and that has no other peak within ``PAIR_SPACING`` bins
    gets a second detection: at the pair offset farther from the peak's own
    detection, with the bin nearest its frequency and the pair probability
    as its salience (how sure the network is that a second fish is there).
    A peak with a neighbour that close is one fish of a pair the salience
    already resolves."""
    p = torch.sigmoid(logits.float())
    left = F.pad(p, (1, 0), value=-1.0)[:, :-1]
    right = F.pad(p, (0, 1), value=-1.0)[:, 1:]
    peak = (p >= threshold) & (p >= left) & (p > right)
    fr, kk = torch.nonzero(peak, as_tuple=True)
    bins = kk + k_lo
    freq = (bins.double() + offset[fr, kk].double()) * df
    sal = p[fr, kk]
    if pair_logits is not None and pair_offset is not None and len(fr):
        thr = threshold if pair_threshold is None else pair_threshold
        r = PAIR_SPACING
        # peaks within r bins, the peak itself included
        n_near = F.avg_pool1d(
            F.pad(peak.float(), (r, r))[:, None], 2 * r + 1, stride=1
        )[:, 0] * (2 * r + 1)
        q = torch.sigmoid(pair_logits.float()[fr, kk])
        split = (q >= thr) & (n_near[fr, kk] < 1.5)
        if bool(split.any()):
            fs, ks = fr[split], kk[split]
            off = pair_offset.float()[:, fs, ks].double()  # (2, n)
            # the peak's own detection stays; the pair offset farther from
            # it is the second fish
            own = offset[fs, ks].double()
            far = torch.where(
                (off[0] - own).abs() >= (off[1] - own).abs(), off[0], off[1]
            )
            k2 = (ks + k_lo).double() + far
            n_bins = logits.shape[-1]
            b2 = torch.round(k2).long().clamp(k_lo, k_lo + n_bins - 1)
            fr = torch.cat([fr, fs])
            bins = torch.cat([bins, b2])
            freq = torch.cat([freq, k2 * df])
            sal = torch.cat([sal, q[split]])
            order = torch.from_numpy(
                np.lexsort((freq.cpu().numpy(), fr.cpu().numpy()))
            ).to(fr.device)
            fr, bins, freq, sal = fr[order], bins[order], freq[order], sal[order]
    return (
        fr.cpu().numpy().astype(np.int64),
        bins.cpu().numpy().astype(np.int64),
        freq.cpu().numpy(),
        sal.cpu().numpy().astype(np.float32),
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


def _frames(z) -> int:
    """Frames of block features (a tensor, or a tuple with electrodes)."""
    return len(z[0]) if isinstance(z, tuple) else len(z)


def _cat(parts: list):
    if isinstance(parts[0], tuple):
        return tuple(torch.cat(p) for p in zip(*parts, strict=True))
    return torch.cat(parts)


def _slice(z, sl: slice):
    return tuple(t[sl] for t in z) if isinstance(z, tuple) else z[sl]


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
        pair_threshold: float | None = None,
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
        self.pair_threshold = pair_threshold
        """Pair head: pair probability at which a peak becomes two detections
        (None: `threshold`; models without the head ignore it)."""
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
            model,
            spec,
            freqs,
            frame_step,
            min_freq,
            max_freq,
            dc.threshold,
            device,
            pair_threshold=getattr(dc, "pair_threshold", None),
        )
        s.model_info = {"path": str(path.resolve()), **meta}
        return s

    def features(
        self, log_power: torch.Tensor, stft: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Log power (bins, frames) of one block -> normalised features
        (frames, bins needed by the harmonic stack). With an electrode
        branch and the block's complex spectra `stft` (electrodes, bins,
        frames), also its `electrode_features` at the candidate bins."""
        k = min(self.k_need, log_power.shape[0])
        z = normalise(log_power[:k].T.to(self.device), self.df, self.spec.band_hz)
        if not self.spec.electrode_channels or stft is None:
            return z
        el = electrode_features(stft.to(self.device), self.k_lo, self.k_hi)
        return z, el

    def push(self, z, payload: Any = None) -> list:
        """Add the features of the next block; returns the blocks that now
        have enough context on both sides."""
        self._queue.append((z, payload))
        out = []
        while self._queue:
            after = sum(_frames(q[0]) for q in self._queue[1:])
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
        right = (
            _slice(_cat(following), slice(None, self.ctx))
            if following and self.ctx
            else None
        )
        det = self._run(self._tail, z, right)
        self._advance(z)
        return det, payload

    def _advance(self, z) -> None:
        tail = z if self._tail is None else _cat([self._tail, z])
        self._tail = _slice(tail, slice(-self.ctx, None) if self.ctx else slice(0, 0))

    @torch.no_grad()
    def _run(self, left, z, right):
        parts = [p for p in (left, z, right) if p is not None and _frames(p)]
        x = _cat(parts)
        n_left = 0 if left is None else _frames(left)
        # the recording starts or ends here: mirror the data into the missing
        # context (the network was trained with real context on both sides)
        pad_l = self.ctx - n_left
        pad_r = self.ctx - (0 if right is None else _frames(right))
        el = None
        if isinstance(x, tuple):
            x, el = x
            el = _mirror(el, pad_l, pad_r).permute(1, 2, 0, 3)[None]
        x = _mirror(x, pad_l, pad_r)
        n_left += max(pad_l, 0)
        stack = harmonic_stack(x, self.k_lo, self.k_hi, self.spec.harmonics)
        out = self.model.outputs(stack[None], el)
        sl = slice(n_left, n_left + _frames(z))
        pair = {}
        if "pair_logits" in out:
            pair = {
                "pair_logits": out["pair_logits"][0, sl],
                "pair_offset": out["pair_offset"][0, :, sl],
                "pair_threshold": self.pair_threshold,
            }
        return decode(
            out["logits"][0, sl],
            out["offset"][0, sl],
            self.threshold,
            self.k_lo,
            self.df,
            **pair,
        )
