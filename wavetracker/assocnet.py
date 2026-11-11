"""Learned association of detections (``tracking.method: learned``).

The rule-based tracker (``wavetracker.tracking``) links detections by a fixed
error: two thirds the rank of their electrode-pattern distance, one third a
sigmoid of their frequency difference. Where two fish cross or run close
together the two errors tie and it swaps them. This module learns the link
instead, in the manner of tracking-by-detection with a learned association
(Brasó & Leal-Taixé, CVPR 2020; Trackastra, Gallusser & Weigert, ECCV 2024):

1. **Candidate graph**: every detection is linked to the detections of the
   next ``assoc_max_dt`` seconds whose frequency is within ``assoc_max_df``
   Hz (directed, earlier to later).
2. **Edge scores**: a small message-passing network gives the probability
   that the two detections of an edge are the same fish. It sees, for every
   detection, its salience and its neighbours in the same frame; for every
   edge the time and frequency step, the level change, and the two
   electrode patterns compared electrode by electrode (power relative to the
   mean over electrodes, each electrode's share of the power, and the
   relative phase, i.e. sign, from the complex spectra), pooled over
   electrodes with shared weights so that any number of electrodes works.
   A few rounds of message passing between detections and edges let every
   edge see its competitors and the slope of the lines over a few seconds.
3. **Linking**: frame by frame, the detections of a frame are attached to
   tracks that end within ``assoc_max_dt`` before it, most probable first,
   down to ``assoc_threshold``; one successor per detection. Tracks shorter
   than ``assoc_min_detections`` stay unassigned.

Hours-long recordings are scored in chunks of frames with enough context on
both sides for the network's receptive field, so the scores do not depend on
the chunking, and the linking runs once over the whole recording, so
identities are consistent across chunks. The fragment linker, the rise
joiner and the salience filter then run as with the rule-based tracker.

The weights are not packaged: train them with
``benchmarks/dense/assocnet.py`` and give their path as
``tracking.assoc_model``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numba import njit
from torch import nn

EPS = 1e-30
DB_CLIP = (-40.0, 20.0)
NODE_FEATURES = [
    "salience",
    "df_below",  # Hz to the next lower detection in the frame (/5, clipped 1)
    "df_above",  # Hz to the next higher detection in the frame (/5, clipped 1)
    "n_close",  # other detections within 1.2 Hz in the frame
    "n_near",  # other detections within 3 Hz in the frame
    "level",  # dB of the summed power relative to the frame's median detection (/10)
]
EDGE_FEATURES = [
    "dt",  # s
    "df",  # Hz, later - earlier
    "abs_df",
    "rate",  # df / dt [Hz/s] (/10, clipped +-1)
    "dlevel",  # dB, later - earlier (/10)
    "sal_a",
    "sal_b",
    "coherence",  # |<u_a, u_b>| of the unit complex spectra (0-1)
    "pattern_rms",  # dB rms difference of the patterns (/10)
    "pattern_corr",
]
ELECTRODE_FEATURES = 6  # per electrode and edge: p_a, p_b, p_b - p_a, cos, w_a, w_b
NODE_ELECTRODE_FEATURES = 2  # per electrode and detection: p, w


# --- candidate graph ----------------------------------------------------------


@njit(cache=True)
def _candidates(idx, fund, max_frames, max_df):
    n = idx.shape[0]
    n_frames = idx[-1] + max_frames + 2
    fs = np.searchsorted(idx, np.arange(n_frames + 1))
    count = 0
    for a in range(n):
        for b in range(fs[idx[a] + 1], fs[idx[a] + max_frames + 1]):
            if abs(fund[b] - fund[a]) <= max_df:
                count += 1
    src = np.empty(count, dtype=np.int64)
    dst = np.empty(count, dtype=np.int64)
    k = 0
    for a in range(n):
        for b in range(fs[idx[a] + 1], fs[idx[a] + max_frames + 1]):
            if abs(fund[b] - fund[a]) <= max_df:
                src[k] = a
                dst[k] = b
                k += 1
    return src, dst


def candidate_edges(
    idx_v: np.ndarray, fund_v: np.ndarray, max_frames: int, max_df: float
) -> tuple[np.ndarray, np.ndarray]:
    """Directed edges (a, b): b 1 to `max_frames` frames after a, frequencies
    within `max_df` Hz. `idx_v` must be sorted. Sorted by a, then b."""
    if len(idx_v) < 2:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return _candidates(
        np.ascontiguousarray(idx_v, dtype=np.int64),
        np.ascontiguousarray(fund_v, dtype=np.float64),
        int(max_frames),
        float(max_df),
    )


@njit(cache=True)
def _frame_context(idx, fund, level):
    """Per detection: Hz to the next lower / higher detection of its frame
    (inf without), detections within 1.2 and 3 Hz, level over the frame's
    median."""
    n = idx.shape[0]
    out = np.zeros((n, 5))
    s = 0
    while s < n:
        e = s
        while e < n and idx[e] == idx[s]:
            e += 1
        med = np.median(level[s:e])
        for a in range(s, e):
            below = np.inf
            above = np.inf
            c1 = 0
            c3 = 0
            for b in range(s, e):
                if b == a:
                    continue
                d = fund[b] - fund[a]
                if d < 0:
                    below = min(below, -d)
                else:
                    above = min(above, d)
                if abs(d) <= 1.2:
                    c1 += 1
                if abs(d) <= 3.0:
                    c3 += 1
            out[a, 0] = below
            out[a, 1] = above
            out[a, 2] = c1
            out[a, 3] = c3
            out[a, 4] = level[a] - med
        s = e
    return out


# --- inputs -----------------------------------------------------------------


@dataclass
class Detections:
    """Per-detection inputs of the network (tensors on one device)."""

    t: torch.Tensor  # s
    f: torch.Tensor  # Hz
    sal: torch.Tensor
    level: torch.Tensor  # dB of the summed power
    pattern: torch.Tensor  # (n, ch) dB relative to the mean over electrodes
    share: torch.Tensor  # (n, ch) share of the power times ch (mean 1)
    unit: torch.Tensor  # (n, ch) complex spectra scaled to unit norm
    node: torch.Tensor  # (n, len(NODE_FEATURES))
    node_el: torch.Tensor  # (n, ch, NODE_ELECTRODE_FEATURES)

    def subset(self, sel: torch.Tensor) -> Detections:
        return Detections(
            **{k: getattr(self, k)[sel] for k in self.__dataclass_fields__}
        )


def detection_inputs(
    fund_v, idx_v, times, sign_v, cplx_v, sal_v, device="cpu"
) -> Detections:
    """Network inputs of every detection (`idx_v` sorted)."""
    sign = np.maximum(np.asarray(sign_v, dtype=np.float64), EPS)
    n_ch = sign.shape[1]
    db = 10.0 * np.log10(sign)
    pattern = np.clip(db - db.mean(1, keepdims=True), *DB_CLIP)
    total = sign.sum(1)
    share = sign / total[:, None] * n_ch
    level = 10.0 * np.log10(total)
    if cplx_v is None:  # no phases: all electrodes in phase
        cplx = np.sqrt(sign).astype(np.complex128)
    else:
        cplx = np.asarray(cplx_v, dtype=np.complex128)
    norm = np.sqrt((np.abs(cplx) ** 2).sum(1, keepdims=True))
    unit = cplx / np.maximum(norm, EPS)
    ctx = _frame_context(
        np.ascontiguousarray(idx_v, dtype=np.int64),
        np.ascontiguousarray(fund_v, dtype=np.float64),
        level,
    )
    sal = np.ones(len(fund_v)) if sal_v is None else np.asarray(sal_v, np.float64)
    node = np.column_stack(
        [
            sal,
            np.minimum(ctx[:, 0] / 5.0, 1.0),
            np.minimum(ctx[:, 1] / 5.0, 1.0),
            np.minimum(ctx[:, 2], 5.0),
            np.minimum(ctx[:, 3], 10.0) / 2.0,
            np.clip(ctx[:, 4] / 10.0, -3.0, 3.0),
        ]
    )
    node_el = np.stack([pattern / 10.0, share], axis=-1)

    def tt(x, dtype=torch.float32):
        return torch.as_tensor(np.ascontiguousarray(x), dtype=dtype, device=device)

    return Detections(
        t=tt(np.asarray(times)[idx_v], torch.float64),
        f=tt(fund_v, torch.float64),
        sal=tt(sal),
        level=tt(level),
        pattern=tt(pattern),
        share=tt(share),
        unit=tt(unit, torch.complex64),
        node=tt(node),
        node_el=tt(node_el),
    )


def edge_inputs(d: Detections, src: torch.Tensor, dst: torch.Tensor):
    """(edge features (E, len(EDGE_FEATURES)), electrode features (E, ch, 6))."""
    dt = (d.t[dst] - d.t[src]).float()
    df = (d.f[dst] - d.f[src]).float()
    pa, pb = d.pattern[src], d.pattern[dst]
    ua, ub = d.unit[src], d.unit[dst]
    z = ua * ub.conj()
    zs = z.sum(1)
    coh = zs.abs()
    align = (zs / coh.clamp_min(1e-12)).conj()
    cos = (z * align[:, None]).real / z.abs().clamp_min(1e-12)
    dp = pb - pa
    rms = dp.pow(2).mean(1).sqrt()
    ca, cb = pa - pa.mean(1, keepdim=True), pb - pb.mean(1, keepdim=True)
    corr = (ca * cb).sum(1) / (ca.norm(dim=1) * cb.norm(dim=1)).clamp_min(1e-6)
    scal = torch.stack(
        [
            dt,
            df,
            df.abs(),
            (df / dt.clamp_min(1e-6) / 10.0).clamp(-1.0, 1.0),
            (d.level[dst] - d.level[src]) / 10.0,
            d.sal[src],
            d.sal[dst],
            coh.clamp(0.0, 1.0),
            rms / 10.0,
            corr,
        ],
        dim=1,
    )
    el = torch.stack(
        [pa / 10.0, pb / 10.0, dp / 10.0, cos, d.share[src], d.share[dst]], dim=-1
    )
    return scal, el


# --- network ----------------------------------------------------------------


def _mlp(*sizes: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def _scatter_mean_max(x: torch.Tensor, index: torch.Tensor, n: int):
    s = torch.zeros(n, x.shape[1], dtype=x.dtype, device=x.device)
    s.index_add_(0, index, x)
    c = torch.zeros(n, dtype=x.dtype, device=x.device)
    c.index_add_(0, index, torch.ones_like(index, dtype=x.dtype))
    mean = s / c.clamp_min(1.0)[:, None]
    mx = torch.zeros(n, x.shape[1], dtype=x.dtype, device=x.device)
    mx = mx.scatter_reduce(
        0, index[:, None].expand_as(x), x, reduce="amax", include_self=False
    )
    return mean, mx


class AssocNet(nn.Module):
    """Edge classifier: message passing between detections and their
    candidate links, with direction-aware aggregation (past and future edges
    apart)."""

    def __init__(self, hidden: int = 64, rounds: int = 3, el_hidden: int = 32):
        super().__init__()
        self.hparams = {"hidden": hidden, "rounds": rounds, "el_hidden": el_hidden}
        h, e = hidden, el_hidden
        self.edge_el = _mlp(ELECTRODE_FEATURES, e, e)
        self.node_el = _mlp(NODE_ELECTRODE_FEATURES, e, e)
        self.node_in = _mlp(len(NODE_FEATURES) + 2 * e, h, h)
        self.edge_in = _mlp(len(EDGE_FEATURES) + 2 * e, h, h)
        self.edge_up = nn.ModuleList([_mlp(3 * h, h, h) for _ in range(rounds)])
        self.node_up = nn.ModuleList([_mlp(5 * h, h, h) for _ in range(rounds)])
        self.head = _mlp(3 * h, h, 1)

    @staticmethod
    def _pool(x: torch.Tensor, mlp: nn.Module) -> torch.Tensor:
        y = mlp(x)  # (n, ch, e)
        return torch.cat([y.mean(1), y.amax(1)], dim=1)

    def forward(self, d: Detections, src: torch.Tensor, dst: torch.Tensor):
        n = d.node.shape[0]
        scal, el = edge_inputs(d, src, dst)
        h = self.node_in(torch.cat([d.node, self._pool(d.node_el, self.node_el)], 1))
        e = self.edge_in(torch.cat([scal, self._pool(el, self.edge_el)], 1))
        for eu, nu in zip(self.edge_up, self.node_up, strict=True):
            e = e + eu(torch.cat([e, h[src], h[dst]], 1))
            mf, xf = _scatter_mean_max(e, src, n)  # a node's future edges
            mp, xp = _scatter_mean_max(e, dst, n)  # its past edges
            h = h + nu(torch.cat([h, mf, xf, mp, xp], 1))
        return self.head(torch.cat([e, h[src], h[dst]], 1))[:, 0]


@dataclass
class AssocModel:
    """A trained network and the candidate rule it was trained with."""

    net: AssocNet
    max_dt: float  # s
    max_df: float  # Hz

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "hparams": self.net.hparams,
                "state": self.net.state_dict(),
                "max_dt": self.max_dt,
                "max_df": self.max_df,
                "node_features": NODE_FEATURES,
                "edge_features": EDGE_FEATURES,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device="cpu") -> AssocModel:
        z = torch.load(path, map_location=device, weights_only=True)
        if z["node_features"] != NODE_FEATURES or z["edge_features"] != EDGE_FEATURES:
            raise ValueError(f"{path}: trained on other features than this version's")
        net = AssocNet(**z["hparams"]).to(device)
        net.load_state_dict(z["state"])
        net.eval()
        return cls(net, float(z["max_dt"]), float(z["max_df"]))


def receptive_frames(model: AssocModel, frame_dt: float) -> int:
    """Frames of context an edge score depends on, on either side."""
    k = max(1, round(model.max_dt / frame_dt))
    return (model.net.hparams["rounds"] + 2) * k


@torch.no_grad()
def edge_probabilities(
    model: AssocModel,
    d: Detections,
    idx_v: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    frame_dt: float,
    chunk_frames: int = 3000,
) -> np.ndarray:
    """Link probability of every edge, in chunks of `chunk_frames` frames
    (by the edge's earlier detection) with the receptive field as context on
    both sides, so the chunking does not change the scores."""
    prob = np.zeros(len(src), dtype=np.float32)
    if not len(src):
        return prob
    device = d.node.device
    margin = receptive_frames(model, frame_dt)
    f_src = idx_v[src]
    first, last = int(idx_v[0]), int(idx_v[-1])
    for c0 in range(first, last + 1, chunk_frames):
        c1 = c0 + chunk_frames
        keep = np.flatnonzero((f_src >= c0) & (f_src < c1))
        if not len(keep):
            continue
        lo = np.searchsorted(idx_v, c0 - margin)
        hi = np.searchsorted(idx_v, c1 + margin)
        sel_e = np.flatnonzero((src >= lo) & (src < hi) & (dst >= lo) & (dst < hi))
        nodes = torch.arange(lo, hi, device=device)
        sub = d.subset(nodes)
        s = torch.as_tensor(src[sel_e] - lo, device=device)
        t = torch.as_tensor(dst[sel_e] - lo, device=device)
        logit = model.net(sub, s, t)
        p = torch.sigmoid(logit).cpu().numpy()
        pos = np.searchsorted(sel_e, keep)  # kept edges are inside sel_e
        prob[keep] = p[pos]
    return prob


# --- linking ----------------------------------------------------------------


@njit(cache=True)
def _link(idx, src, dst, prob, order, threshold, n):
    """Frame by frame (by the later detection), most probable edge first
    (`order`): attach a detection to the track of an earlier detection that
    has no successor yet. Returns the track of every detection."""
    track = np.full(n, -1, dtype=np.int64)
    has_succ = np.zeros(n, dtype=np.bool_)
    next_track = 0
    k = 0
    m = order.shape[0]
    b_first = 0  # detections of earlier frames get their tracks first
    while k < m:
        frame = idx[dst[order[k]]]
        while b_first < n and idx[b_first] < frame:
            if track[b_first] < 0:
                track[b_first] = next_track
                next_track += 1
            b_first += 1
        e = k
        while e < m and idx[dst[order[e]]] == frame:
            e += 1
        for q in range(k, e):
            j = order[q]
            if prob[j] < threshold:
                break
            a = src[j]
            b = dst[j]
            if has_succ[a] or track[b] >= 0:
                continue
            track[b] = track[a]
            has_succ[a] = True
        k = e
    for b in range(b_first, n):
        if track[b] < 0:
            track[b] = next_track
            next_track += 1
    return track


def link_edges(
    idx_v: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    prob: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Track of every detection (0, 1, ...) from scored edges."""
    n = len(idx_v)
    if n == 0:
        return np.zeros(0, np.int64)
    idx = np.ascontiguousarray(idx_v, dtype=np.int64)
    prob = np.ascontiguousarray(prob, dtype=np.float64)
    order = np.lexsort((-prob, idx[dst])).astype(np.int64)
    return _link(
        idx,
        np.ascontiguousarray(src, dtype=np.int64),
        np.ascontiguousarray(dst, dtype=np.int64),
        prob,
        order,
        float(threshold),
        n,
    )


_MODELS: dict[tuple[str, str], AssocModel] = {}


def load_model(path: str | Path, device="cpu") -> AssocModel:
    key = (str(Path(path).resolve()), str(device))
    if key not in _MODELS:
        _MODELS[key] = AssocModel.load(path, device)
    return _MODELS[key]


def track_learned(
    fund_v: np.ndarray,
    idx_v: np.ndarray,
    sign_v: np.ndarray,
    cplx_v: np.ndarray | None,
    sal_v: np.ndarray | None,
    times: np.ndarray,
    model: AssocModel,
    threshold: float = 0.5,
    min_detections: int = 1,
    device="cpu",
    chunk_frames: int = 3000,
) -> np.ndarray:
    """Identity of every detection (NaN: none) from the learned association.
    `idx_v` must be sorted."""
    n = len(fund_v)
    ident = np.full(n, np.nan)
    if n < 2 or len(times) < 2:
        return ident
    frame_dt = float(times[1] - times[0])
    k = max(1, round(model.max_dt / frame_dt))
    src, dst = candidate_edges(idx_v, fund_v, k, model.max_df)
    d = detection_inputs(fund_v, idx_v, times, sign_v, cplx_v, sal_v, device)
    prob = edge_probabilities(model, d, idx_v, src, dst, frame_dt, chunk_frames)
    tr = link_edges(idx_v, src, dst, prob, threshold)
    size = np.bincount(tr)
    ok = size[tr] >= min_detections
    ident[ok] = tr[ok]
    return ident
