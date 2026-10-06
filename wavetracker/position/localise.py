"""Position of a stationary fish from its detections by moving electrodes.

Observation model (residuals of one fish, parameters ``x, y, depth, heading,
log strength``; ``V`` the predicted channel signal, ``n`` the channel's noise
floor amplitude at the fish's frequency, ``A`` the detected amplitude):

detected frames
    ``log sqrt(V**2 + n**2) - log A`` for every channel;
relative sign
    ``sign_weight * (1 - s V_a V_b / |V_a V_b|)`` for channel pairs that are
    both ``strong_snr`` above the noise floor (``a`` the strongest channel of
    the frame, ``s`` = +1 / -1 for same / opposite sign from ``cplx_v``);
censored frames
    survey frames where the fish was not detected must not be predicted above
    its lowest detected level ``c`` (5th percentile of ``A``):
    ``censor_weight * max(0, log sqrt(W**2 + n**2) - log c)``;
detection range
    ``bound_weight * max(0, d - detection_range)``, ``d`` the 3-D distance to
    the nearest electrode-pair midpoint at the fish's detections.

The robust (soft-L1) least-squares fit starts from a grid over position,
heading and depth on a subsampled problem, refines the best distinct basins
on all data and reports a competing basin (ambiguity). Standard errors come
from a delete-a-block jackknife over blocks of the fish's detection times.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import pairwise

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from ..config import PositionMergingConfig
from .efield import SourceModel
from .electrodes import ChannelMap

LEVEL_PERCENTILE = 5.0
"""Percentile of a fish's detected amplitudes defining its censoring level."""
N_TOP = 40
"""Number of strongest detections defining the centre of the start grid."""


def source_model(cfg: PositionMergingConfig) -> SourceModel:
    return SourceModel(
        kind=cfg.model,
        body_length=cfg.body_length,
        n_poles=cfg.n_poles,
        water_depth=cfg.water_depth,
        n_images=cfg.n_images,
    )


def max_depth(cfg: PositionMergingConfig) -> float:
    if cfg.max_depth is not None:
        return cfg.max_depth
    if cfg.water_depth is not None:
        return cfg.water_depth - 0.05
    return cfg.detection_range


@dataclass
class Observations:
    """Detections of one fish (candidate) and the frames it was not seen."""

    t_det: np.ndarray
    """Times of the detections [s] (nd,)."""
    amp: np.ndarray
    """Amplitude (sqrt of power) per channel (nd, C)."""
    cplx: np.ndarray | None
    """Complex spectrum per channel (nd, C), for the relative signs."""
    noise: np.ndarray
    """Noise floor amplitude per channel at the fish's frequency (C,)."""
    pos_det: np.ndarray
    """Electrode positions at the detections (nd, E, 3)."""
    t_cens: np.ndarray
    """Times of censored survey frames (nc,)."""
    pos_cens: np.ndarray
    """Electrode positions at the censored frames (nc, E, 3)."""
    channels: ChannelMap
    seg: np.ndarray | None = None
    """Segment of each detection (nd,), for per-segment residuals."""

    def __post_init__(self) -> None:
        self.noise = np.maximum(np.asarray(self.noise, float), 1e-30)
        self.amp = np.maximum(np.asarray(self.amp, float), 1e-30)
        self.level = np.percentile(self.amp, LEVEL_PERCENTILE, axis=0)


def select_censored(
    pos_det: np.ndarray,
    pos_cens: np.ndarray,
    channels: ChannelMap,
    max_distance: float,
) -> np.ndarray:
    """Censored frames with a channel midpoint within `max_distance` of the
    detection midpoints (farther frames cannot constrain the fish)."""
    if len(pos_cens) == 0 or len(pos_det) == 0:
        return np.zeros(len(pos_cens), bool)
    tree = cKDTree(channels.midpoints(pos_det).reshape(-1, 3))
    mid = channels.midpoints(pos_cens)
    d, _ = tree.query(mid.reshape(-1, 3), distance_upper_bound=max_distance)
    return np.isfinite(d.reshape(mid.shape[:2])).any(axis=1)


def relative_signs(
    amp: np.ndarray, cplx: np.ndarray | None, noise: np.ndarray, strong_snr: float
):
    """Frame, channel a (strongest), channel b and sign of strong pairs."""
    empty = (np.zeros(0, int),) * 3 + (np.zeros(0),)
    if cplx is None or amp.shape[1] < 2:
        return empty
    snr = amp / noise
    anchor = snr.argmax(1)
    rows, a, b = [], [], []
    for c in range(amp.shape[1]):
        m = (anchor != c) & (snr[:, c] > strong_snr) & (snr.max(1) > strong_snr)
        idx = np.flatnonzero(m)
        rows.append(idx)
        a.append(anchor[idx])
        b.append(np.full(len(idx), c))
    rows, a, b = np.concatenate(rows), np.concatenate(a), np.concatenate(b)
    s = np.sign(np.real(cplx[rows, b] * np.conj(cplx[rows, a])))
    s[s == 0] = 1.0
    return rows, a, b, s


class Problem:
    """Residuals of one fish's observations (optionally a subset)."""

    def __init__(
        self,
        obs: Observations,
        model: SourceModel,
        cfg: PositionMergingConfig,
        keep_det: np.ndarray | None = None,
        keep_cens: np.ndarray | None = None,
    ):
        self.obs, self.model, self.cfg = obs, model, cfg
        kd = np.ones(len(obs.t_det), bool) if keep_det is None else keep_det
        kc = np.ones(len(obs.t_cens), bool) if keep_cens is None else keep_cens
        self.A = obs.amp[kd]
        self.logA = np.log(self.A)
        cplx = None if obs.cplx is None else obs.cplx[kd]
        self.rows, self.a, self.b, self.s = relative_signs(
            self.A, cplx, obs.noise, cfg.strong_snr
        )
        pos_d, pos_c = obs.pos_det[kd], obs.pos_cens[kc]
        self.nd, self.nc = len(pos_d), len(pos_c)
        self.points = np.ascontiguousarray(
            np.concatenate([pos_d, pos_c]).reshape(-1, 3)
        )
        self.n_el = obs.pos_det.shape[1]
        self.n2 = obs.noise**2
        self.log_level = np.log(obs.level)
        self.mids = obs.channels.midpoints(pos_d)  # (nd, C, 3)
        mids = self.mids.reshape(-1, 3)
        self.tree = cKDTree(mids)
        r = cfg.detection_range
        self.bounds = (
            [
                mids[:, 0].min() - r,
                mids[:, 1].min() - r,
                cfg.min_depth,
                -np.inf,
                -np.inf,
            ],
            [
                mids[:, 0].max() + r,
                mids[:, 1].max() + r,
                max_depth(cfg),
                np.inf,
                np.inf,
            ],
        )

    def voltages(self, q) -> tuple[np.ndarray, np.ndarray]:
        """Predicted channel signals at detections (nd, C) and censored frames."""
        x, y, d, th, logp = q
        pot = self.model.potential(self.points, x, y, d, th).reshape(-1, self.n_el)
        v = self.obs.channels.values(pot) * np.exp(logp)
        return v[: self.nd], v[self.nd :]

    def dist_to_pairs(self, x: float, y: float, depth: float) -> float:
        d, _ = self.tree.query([x, y, -depth])
        return float(d)

    def jac(self, q) -> np.ndarray:
        """Analytic Jacobian of `resid` (the sign term is piecewise constant)."""
        cfg = self.cfg
        x, y, d, th, logp = q
        pot, g = self.model.potential_and_gradient(self.points, x, y, d, th)
        scale = np.exp(logp)
        ch = self.obs.channels
        v = ch.values(pot.reshape(-1, self.n_el)) * scale  # (n, C)
        dv = ch.values(g.reshape(-1, self.n_el, 4).transpose(0, 2, 1)) * scale
        dv = np.concatenate([dv.transpose(0, 2, 1), v[..., None]], axis=-1)
        dlog = (v / (v**2 + self.n2))[..., None] * dv  # d/dq 0.5 log(v^2 + n^2)
        j_amp = dlog[: self.nd].reshape(-1, 5)
        w = v[self.nd :]
        active = 0.5 * np.log(w**2 + self.n2) > self.log_level
        j_cens = (cfg.censor_weight * active[..., None] * dlog[self.nd :]).reshape(
            -1, 5
        )
        j_sign = np.zeros((len(self.rows), 5))
        j_bound = np.zeros((1, 5))
        dist, i = self.tree.query([x, y, -d])
        if dist > cfg.detection_range and dist > 0:
            diff = (np.array([x, y, -d]) - self.tree.data[i]) / dist
            j_bound[0, :3] = cfg.bound_weight * diff * [1, 1, -1]
        return np.concatenate([j_amp, j_sign, j_cens, j_bound])

    def resid(self, q) -> np.ndarray:
        cfg = self.cfg
        v, w = self.voltages(q)
        r_amp = 0.5 * np.log(v**2 + self.n2) - self.logA
        va, vb = v[self.rows, self.a], v[self.rows, self.b]
        r_sign = cfg.sign_weight * (1 - self.s * va * vb / (np.abs(va * vb) + 1e-300))
        r_cens = cfg.censor_weight * np.maximum(
            0.0, 0.5 * np.log(w**2 + self.n2) - self.log_level
        )
        excess = self.dist_to_pairs(q[0], q[1], q[2]) - cfg.detection_range
        r_bound = cfg.bound_weight * max(0.0, excess)
        return np.concatenate([r_amp.ravel(), r_sign, r_cens.ravel(), [r_bound]])

    def scale_logp(self, q) -> float:
        """Log strength matching the strongest predicted to observed amplitudes."""
        v, _ = self.voltages([*q[:4], 0.0])
        pv = max(np.percentile(np.abs(v), 99), 1e-300)
        return float(np.log(np.percentile(self.A, 99) / pv))

    def lsq(self, x0, tol: float = 1e-8):
        lo, hi = self.bounds
        x0 = np.array(x0, float)
        x0[:3] = np.clip(x0[:3], np.add(lo[:3], 1e-3), np.subtract(hi[:3], 1e-3))
        return least_squares(
            self.resid,
            x0,
            jac=self.jac,
            loss="soft_l1",
            f_scale=1.0,
            bounds=self.bounds,
            ftol=tol,
            xtol=tol,
        )


@dataclass
class FitResult:
    x: float
    y: float
    depth: float
    heading: float
    """Head direction [rad]."""
    log_strength: float
    cost: float
    r_fit: float
    """Correlation of observed and predicted log amplitudes (strong channels)."""
    n_det: int
    n_cens: int
    dist_to_pair: float
    """3-D distance to the nearest electrode-pair midpoint at a detection [m]."""
    at_bound: bool
    at_depth_cap: bool
    alt_dcost: float = np.inf
    """Cost of the best competing basin minus the best cost."""
    alt_x: float = np.nan
    alt_y: float = np.nan
    ambiguous: bool = False
    basin_radius: float = 0.0
    """Largest distance of a refined basin within ambiguity_dcost of the best
    cost [m] (0: unique)."""
    se_major: float = np.nan
    """Jackknife standard error along the major axis of the error ellipse [m]."""
    se_minor: float = np.nan
    se_angle: float = np.nan
    """Direction of the major axis [deg]."""
    se_x: float = np.nan
    se_y: float = np.nan
    jk_blocks: int = 0

    @property
    def params(self) -> np.ndarray:
        return np.array([self.x, self.y, self.depth, self.heading, self.log_strength])

    def to_dict(self) -> dict:
        return asdict(self)


def summarise(pb: Problem, x: np.ndarray, cost: float) -> FitResult:
    v, _ = pb.voltages(x)
    pred = 0.5 * np.log(v**2 + pb.n2)
    hi = np.sqrt(10.0) * pb.obs.noise < pb.A
    r_fit = float(np.corrcoef(pb.logA[hi], pred[hi])[0, 1]) if hi.sum() > 5 else np.nan
    d_pair = pb.dist_to_pairs(x[0], x[1], x[2])
    return FitResult(
        x=float(x[0]),
        y=float(x[1]),
        depth=float(x[2]),
        heading=float(np.mod(x[3], 2 * np.pi)),
        log_strength=float(x[4]),
        cost=float(cost),
        r_fit=r_fit,
        n_det=pb.nd,
        n_cens=pb.nc,
        dist_to_pair=d_pair,
        at_bound=bool(d_pair >= pb.cfg.detection_range - 0.02),
        at_depth_cap=bool(x[2] >= pb.bounds[1][2] - 0.01),
    )


def _subsample(n: int, keep: np.ndarray | None, stride: int) -> np.ndarray:
    k = np.ones(n, bool) if keep is None else keep.copy()
    idx = np.flatnonzero(k)
    k[idx[np.arange(len(idx)) % stride != 0]] = False
    return k


def start_points(pb: Problem, cfg: PositionMergingConfig) -> list[list[float]]:
    snr = (pb.A / pb.obs.noise).max(1)
    top = np.argsort(snr)[-N_TOP:]
    centre = pb.mids[top].reshape(-1, 3).mean(0)
    offs = np.linspace(-cfg.start_half_width, cfg.start_half_width, cfg.start_grid)
    headings = np.linspace(0, 2 * np.pi, cfg.start_headings, endpoint=False)
    lo, hi = pb.bounds[0][2] + 1e-3, pb.bounds[1][2] - 1e-3
    depths = sorted({float(np.clip(d, lo, hi)) for d in cfg.start_depths})
    return [
        [centre[0] + dx, centre[1] + dy, d, th, 0.0]
        for dx in offs
        for dy in offs
        for th in headings
        for d in depths
    ]


def fit(
    obs: Observations,
    model: SourceModel,
    cfg: PositionMergingConfig,
    keep_det: np.ndarray | None = None,
    keep_cens: np.ndarray | None = None,
) -> FitResult:
    """Two-stage multi-start fit with ambiguity check."""
    pb = Problem(obs, model, cfg, keep_det, keep_cens)
    # stage 1: all starts on a subsampled problem
    pb1 = Problem(
        obs,
        model,
        cfg,
        _subsample(len(obs.t_det), keep_det, cfg.stage1_stride),
        _subsample(len(obs.t_cens), keep_cens, cfg.stage1_stride),
    )
    sols = []
    for q in start_points(pb, cfg):
        q[4] = pb1.scale_logp(q)
        res = pb1.lsq(q, tol=1e-4)
        sols.append((res.cost, res.x))
    sols.sort(key=lambda cx: cx[0])
    # stage 2: refine the best distinct basins on all data
    picks: list[np.ndarray] = []
    for _, x in sols:
        if all(np.hypot(*(x[:2] - p[:2])) > cfg.alt_min_distance for p in picks):
            picks.append(x)
        if len(picks) == cfg.n_refine:
            break
    refined = sorted((pb.lsq(x) for x in picks), key=lambda r: r.cost)
    best = refined[0]
    out = summarise(pb, best.x, best.cost)
    alt = [
        r
        for r in refined[1:]
        if np.hypot(*(r.x[:2] - best.x[:2])) > cfg.alt_min_distance
    ]
    near = [
        np.hypot(*(r.x[:2] - best.x[:2]))
        for r in refined[1:]
        if r.cost - best.cost < cfg.ambiguity_dcost
    ]
    out.basin_radius = float(max(near, default=0.0))
    if alt:
        out.alt_dcost = float(alt[0].cost - best.cost)
        out.alt_x, out.alt_y = float(alt[0].x[0]), float(alt[0].x[1])
        out.ambiguous = bool(out.alt_dcost < cfg.ambiguity_dcost)
    return out


def refit(
    obs: Observations,
    model: SourceModel,
    cfg: PositionMergingConfig,
    inits: list[np.ndarray],
    keep_det: np.ndarray | None = None,
    keep_cens: np.ndarray | None = None,
) -> FitResult:
    """Local fits from given starting points (heading +-45 deg), best kept."""
    pb = Problem(obs, model, cfg, keep_det, keep_cens)
    best = None
    for init in inits:
        for dth in (0.0, np.pi / 4, -np.pi / 4):
            x0 = np.array(init, float)
            x0[3] += dth
            res = pb.lsq(x0)
            if best is None or res.cost < best.cost:
                best = res
    return summarise(pb, best.x, best.cost)


def jackknife(
    obs: Observations,
    model: SourceModel,
    cfg: PositionMergingConfig,
    result: FitResult,
    min_detections: int = 10,
) -> FitResult:
    """Delete-a-block jackknife standard errors of the position (in place).

    Blocks are `cfg.jackknife_blocks` quantiles of the detection times; a
    deleted block removes the detections and censored frames in its span.
    """
    est = []
    edges = np.quantile(obs.t_det, np.linspace(0, 1, cfg.jackknife_blocks + 1))
    edges[-1] += 1e-9
    for a, b in pairwise(edges):
        kd = ~((obs.t_det >= a) & (obs.t_det < b))
        kc = ~((obs.t_cens >= a) & (obs.t_cens < b))
        if kd.sum() < min_detections or kd.all():
            continue
        r = refit(obs, model, cfg, [result.params], kd, kc)
        est.append([r.x, r.y])
    result.jk_blocks = len(est)
    if len(est) < 3:
        return result
    est = np.array(est)
    n = len(est)
    dev = est - est.mean(0)
    cov = (n - 1) / n * dev.T @ dev
    ev, evec = np.linalg.eigh(cov)
    ev = np.clip(ev, 0, None)
    result.se_major, result.se_minor = float(np.sqrt(ev[1])), float(np.sqrt(ev[0]))
    result.se_angle = float(np.degrees(np.arctan2(evec[1, 1], evec[0, 1])) % 180)
    result.se_x, result.se_y = (float(s) for s in np.sqrt(np.diag(cov)))
    return result


def segment_check(
    obs: Observations, model: SourceModel, cfg: PositionMergingConfig, params
) -> dict[int, tuple[float, float]]:
    """Per segment under the given fish: mean |log-amplitude residual| and
    3-D distance of the fish to the nearest electrode-pair midpoint at the
    segment's detections.

    The residual is averaged over the channels `cfg.strong_snr` above the
    noise floor (all channels if a segment has none): near the noise floor
    residuals are compressed and do not reveal a wrongly assigned weak
    segment; the distance does (beyond `detection_range` the fish could not
    have been detected).
    """
    pb = Problem(obs, model, cfg)
    v, _ = pb.voltages(params)
    r = np.abs(0.5 * np.log(v**2 + pb.n2) - pb.logA)
    strong = cfg.strong_snr * obs.noise < pb.A
    fish = np.array([params[0], params[1], -params[2]])
    dist = np.linalg.norm(pb.mids - fish, axis=-1).min(1)
    seg = np.zeros(len(r), int) if obs.seg is None else np.asarray(obs.seg)
    out = {}
    for s in np.unique(seg):
        m = seg == s
        res = r[m][strong[m]].mean() if strong[m].any() else r[m].mean()
        out[int(s)] = (float(res), float(dist[m].min()))
    return out
