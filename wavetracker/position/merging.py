"""Group track segments into fish by frequency and source position.

With moving electrodes each fish is in range only during passes, so
identities are track segments, and a neighbouring fish is often only a few Hz
away. Fish resting during the survey keep their position, which identifies
them across gaps. Steps (:func:`merge_by_position`):

1. segments: identities within the survey split at gaps > ``split_gap``;
   segments shorter than ``min_segment_duration`` or with fewer than
   ``min_segment_detections`` detections are clutter;
2. candidates: segments, strongest first, join the candidate with the
   closest median frequency within ``freq_tolerance`` that has no segment
   overlapping in time by more than ``max_overlap``;
3. each candidate is localised (:mod:`wavetracker.position.localise`);
4. position check: segments whose mean |log residual| under the joint fit
   exceeds ``max(resid_abs, resid_rel * median)`` are split off, regrouped by
   frequency and fitted as new candidates; reduced candidates are refitted;
5. jackknife standard errors;
6. merge: candidates with segments whose median frequencies agree within
   ``merge_freq_tolerance`` (fish drift slowly; no temporal overlap) and whose positions agree within
   ``merge_se_factor`` combined standard errors are merged if the joint fit
   explains all their segments (step 4 criterion). This is what
   frequency-only stitching cannot do.

The result is a fish identity per detection (``fish_v``, NaN for clutter and
detections outside the survey) and a table of fish.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import PositionMergingConfig
from ..results import Results
from .electrodes import ElectrodeTrack
from .localise import (
    FitResult,
    Observations,
    fit,
    jackknife,
    refit,
    segment_check,
    select_censored,
    source_model,
)

log = logging.getLogger(__name__)

FISH_FILE = "fish_v.npy"
TABLE_FILE = "fish.csv"
SEGMENTS_FILE = "fish_segments.csv"
META_FILE = "position_merging.json"

Progress = Callable[[str, int, int], None]
"""Called with (stage, done, total)."""


# --- segments and frequency grouping ------------------------------------------------


def make_segments(
    results: Results, in_survey: np.ndarray, cfg: PositionMergingConfig
) -> tuple[pd.DataFrame, np.ndarray]:
    """Split identities at gaps; returns the segment table and the segment of
    each detection (-1: none)."""
    t_all = results.times[results.idx_v]
    seg_of = np.full(len(t_all), -1)
    rows = []
    tracked = in_survey & ~np.isnan(results.ident_v)
    power = results.sign_v.sum(1)
    for ident in np.unique(results.ident_v[tracked]):
        idx = np.flatnonzero(tracked & (results.ident_v == ident))
        idx = idx[np.argsort(t_all[idx], kind="stable")]
        breaks = np.flatnonzero(np.diff(t_all[idx]) > cfg.split_gap) + 1
        for part in np.split(idx, breaks):
            tt = t_all[part]
            clutter = (tt[-1] - tt[0] < cfg.min_segment_duration) or (
                len(part) < cfg.min_segment_detections
            )
            seg_of[part] = len(rows)
            rows.append(
                {
                    "segment": len(rows),
                    "ident": int(ident),
                    "t0": float(tt[0]),
                    "t1": float(tt[-1]),
                    "n": len(part),
                    "freq": float(np.median(results.fund_v[part])),
                    "power": float(power[part].sum()),
                    "clutter": bool(clutter),
                }
            )
    columns = ["segment", "ident", "t0", "t1", "n", "freq", "power", "clutter"]
    return pd.DataFrame(rows, columns=columns), seg_of


def _overlaps(spans_a, spans_b, max_overlap: float) -> bool:
    return any(
        min(b1, a1) - max(b0, a0) > max_overlap
        for a0, a1 in spans_a
        for b0, b1 in spans_b
    )


def group_by_frequency(
    segs: pd.DataFrame, tol: float, max_overlap: float
) -> list[list[int]]:
    """Greedy grouping, strongest segments first; candidate frequency is the
    power-weighted median of its segments' frequencies."""
    groups: list[dict] = []
    for s in segs.sort_values("power", ascending=False).itertuples():
        best = None
        for g in groups:
            df = abs(s.freq - g["freq"])
            if df > tol or _overlaps([(s.t0, s.t1)], g["spans"], max_overlap):
                continue
            if best is None or df < abs(s.freq - best["freq"]):
                best = g
        if best is None:
            groups.append(
                {
                    "freq": s.freq,
                    "segs": [s.segment],
                    "spans": [(s.t0, s.t1)],
                    "w": [s.power],
                    "f": [s.freq],
                }
            )
            continue
        best["segs"].append(s.segment)
        best["spans"].append((s.t0, s.t1))
        best["w"].append(s.power)
        best["f"].append(s.freq)
        o = np.argsort(best["f"])
        cw = np.cumsum(np.array(best["w"])[o])
        best["freq"] = float(np.array(best["f"])[o][np.searchsorted(cw, cw[-1] / 2)])
    return [[int(x) for x in g["segs"]] for g in groups]


# --- observations ----------------------------------------------------------------


@dataclass
class _Context:
    """Everything needed to build a candidate's observations."""

    results: Results
    track: ElectrodeTrack
    cfg: PositionMergingConfig
    noise_floor: object
    seg_of: np.ndarray
    survey_frames: np.ndarray
    frame_pos: np.ndarray
    """Electrode positions at all frames (frames, E, 3)."""

    def observations(self, seg_ids) -> tuple[Observations, float]:
        r, cfg = self.results, self.cfg
        idx = np.flatnonzero(np.isin(self.seg_of, seg_ids))
        idx = idx[np.argsort(r.idx_v[idx], kind="stable")]
        f0 = float(np.median(r.fund_v[idx]))
        frames = r.idx_v[idx]
        near = np.abs(r.fund_v - f0) <= cfg.censor_freq_tolerance
        seen = np.union1d(r.idx_v[near], frames)
        cens = np.setdiff1d(self.survey_frames, seen)[:: max(cfg.censor_stride, 1)]
        pos_det = self.frame_pos[frames]
        pos_cens = self.frame_pos[cens]
        keep = select_censored(
            pos_det,
            pos_cens,
            self.track.channels,
            cfg.censor_range_factor * cfg.detection_range,
        )
        obs = Observations(
            t_det=r.times[frames],
            amp=np.sqrt(r.sign_v[idx]),
            cplx=None if r.cplx_v is None else r.cplx_v[idx],
            noise=self.noise_floor.amplitude(f0),
            pos_det=pos_det,
            t_cens=r.times[cens[keep]],
            pos_cens=pos_cens[keep],
            channels=self.track.channels,
            seg=self.seg_of[idx],
        )
        return obs, f0


# --- parallel tasks ----------------------------------------------------------------


def _init_worker() -> None:
    import numba

    numba.set_num_threads(1)


def _fit_task(args):
    obs, cfg = args
    model = source_model(cfg)
    res = fit(obs, model, cfg)
    return res, segment_check(obs, model, cfg, res.params)


def _jackknife_task(args):
    obs, cfg, res = args
    return jackknife(obs, source_model(cfg), cfg, res)


def _merge_task(args):
    obs, cfg, inits = args
    model = source_model(cfg)
    res = refit(obs, model, cfg, inits)
    return res, segment_check(obs, model, cfg, res.params)


def _map(fn, items: list, n_jobs: int, stage: str, progress: Progress | None):
    out = []
    if progress:
        progress(stage, 0, len(items))
    if n_jobs == 1 or len(items) <= 1:
        for k, item in enumerate(items):
            out.append(fn(item))
            if progress:
                progress(stage, k + 1, len(items))
        return out
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    with ProcessPoolExecutor(
        max_workers=min(n_jobs, len(items)),
        mp_context=get_context("spawn"),
        initializer=_init_worker,
    ) as ex:
        for k, res in enumerate(ex.map(fn, items)):
            out.append(res)
            if progress:
                progress(stage, k + 1, len(items))
    return out


# --- main ------------------------------------------------------------------------


@dataclass
class Candidate:
    segs: list[int]
    freq: float
    spans: list[tuple[float, float]]
    fit: FitResult
    seg_check: dict[int, tuple[float, float]] = field(default_factory=dict)
    """Residual and distance per segment (see localise.segment_check)."""

    def positions(self) -> list[np.ndarray]:
        """Best position and, if ambiguous, the competing one."""
        p = [np.array([self.fit.x, self.fit.y])]
        if self.fit.ambiguous:
            p.append(np.array([self.fit.alt_x, self.fit.alt_y]))
        return p

    def inits(self) -> list[np.ndarray]:
        out = [self.fit.params]
        if self.fit.ambiguous:
            q = self.fit.params.copy()
            q[:2] = [self.fit.alt_x, self.fit.alt_y]
            out.append(q)
        return out


@dataclass
class MergeOutput:
    fish_v: np.ndarray
    """Fish of each detection (NaN: clutter, outside the survey, untracked)."""
    fish: pd.DataFrame
    """One row per fish."""
    segments: pd.DataFrame
    """One row per segment, with its fish (-1: clutter) and residual."""
    stats: dict

    def save(self, folder: str | Path) -> None:
        folder = Path(folder)
        np.save(folder / FISH_FILE, self.fish_v)
        self.fish.to_csv(folder / TABLE_FILE, index=False)
        self.segments.to_csv(folder / SEGMENTS_FILE, index=False)
        (folder / META_FILE).write_text(json.dumps(self.stats, indent=2, default=str))


def _bad_segments(check: dict[int, tuple[float, float]], cfg) -> list[int]:
    """Segments the fish does not explain: residual above max(resid_abs,
    resid_rel * median) or fish beyond the detection range."""
    if len(check) < 2:
        return []
    resid = np.array([r for r, _ in check.values()])
    thr = max(cfg.resid_abs, cfg.resid_rel * float(np.median(resid)))
    return [
        s for s, (r, d) in check.items() if r > thr or d > cfg.detection_range + 0.05
    ]


def _se(c: Candidate, cfg: PositionMergingConfig) -> float:
    """Position uncertainty for merging: jackknife SE (major axis), at least
    `merge_min_se`, and for ambiguous fits at least the distance to the
    farthest competing basin within `ambiguity_dcost` (the jackknife, which
    refits from the best basin, does not see them)."""
    se = c.fit.se_major if np.isfinite(c.fit.se_major) else 0.0
    if c.fit.ambiguous:
        se = max(se, c.fit.basin_radius)
    return max(se, cfg.merge_min_se)


def merge_by_position(
    results: Results,
    track: ElectrodeTrack,
    cfg: PositionMergingConfig,
    noise_floor=None,
    n_jobs: int | None = None,
    progress: Progress | None = None,
) -> MergeOutput:
    """Group the identities of `results` into fish (see module docstring).

    `noise_floor` has an ``amplitude(freq) -> (channels,)`` method (see
    :mod:`wavetracker.position.noise`); default: from the detections.
    """
    from .noise import DetectionNoiseFloor

    t_start = time.perf_counter()
    n_jobs = n_jobs or os.cpu_count() or 1
    if results.sign_v.shape[1] != track.n_channels:
        raise ValueError(
            f"sign_v has {results.sign_v.shape[1]} channels but the electrode "
            f"geometry defines {track.n_channels}; set reference or channel_pairs"
        )
    if results.cplx_v is None:
        log.warning("No cplx_v in the results: relative signs are not used")
    if noise_floor is None:
        noise_floor = DetectionNoiseFloor(
            results.fund_v, results.sign_v, cfg.noise_quantile
        )

    # survey frames: valid geometry within [t_start, t_end]
    times = results.times
    frame_pos = track.at(times)
    survey = np.isfinite(frame_pos).all(axis=(1, 2))
    if cfg.t_start is not None:
        survey &= times >= cfg.t_start
    if cfg.t_end is not None:
        survey &= times <= cfg.t_end
    survey_frames = np.flatnonzero(survey)
    if len(survey_frames) == 0:
        raise ValueError("No frames with valid electrode positions in the survey")
    in_survey = survey[results.idx_v]

    segs, seg_of = make_segments(results, in_survey, cfg)
    good = segs[~segs.clutter]
    ctx = _Context(results, track, cfg, noise_floor, seg_of, survey_frames, frame_pos)
    seg_freq = dict(zip(segs.segment, segs.freq, strict=True))
    seg_span = {int(s.segment): (float(s.t0), float(s.t1)) for s in segs.itertuples()}
    log.info(
        "%d segments, %d kept (%.0f%% of survey detections), %d clutter",
        len(segs),
        len(good),
        100 * good.n.sum() / max(in_survey.sum(), 1),
        segs.clutter.sum(),
    )

    def fit_groups(groups: list[list[int]], stage: str) -> list[Candidate]:
        obs = [ctx.observations(g) for g in groups]
        out = _map(_fit_task, [(o, cfg) for o, _ in obs], n_jobs, stage, progress)
        return [
            Candidate(g, f0, [seg_span[s] for s in g], res, sr)
            for g, (_, f0), (res, sr) in zip(groups, obs, out, strict=True)
        ]

    timings = {}
    t0 = time.perf_counter()
    groups = group_by_frequency(good, cfg.freq_tolerance, cfg.max_overlap)
    cands = fit_groups(groups, "fit")
    n_initial = len(cands)
    timings["fit"] = time.perf_counter() - t0

    # position check: split off segments the joint fit does not explain
    t0 = time.perf_counter()
    kept, changed, evicted = [], [], []
    for c in cands:
        bad = _bad_segments(c.seg_check, cfg)
        if bad and len(bad) < len(c.segs):
            evicted += bad
            changed.append([s for s in c.segs if s not in bad])
        else:
            kept.append(c)
    regroup = (
        group_by_frequency(
            good[good.segment.isin(evicted)], cfg.freq_tolerance, cfg.max_overlap
        )
        if evicted
        else []
    )
    log.info(
        "position check: %d segments split off from %d candidates -> %d new",
        len(evicted),
        len(changed),
        len(regroup),
    )
    cands = kept + fit_groups(changed + regroup, "refit")
    timings["position_check"] = time.perf_counter() - t0

    t0 = time.perf_counter()

    def add_se(cs: list[Candidate], stage: str) -> None:
        items = [(ctx.observations(c.segs)[0], cfg, c.fit) for c in cs]
        for c, res in zip(
            cs, _map(_jackknife_task, items, n_jobs, stage, progress), strict=True
        ):
            c.fit = res

    add_se(cands, "jackknife")
    timings["jackknife"] = time.perf_counter() - t0

    # merge candidates of one fish
    t0 = time.perf_counter()
    rejected: set[frozenset] = set()
    n_merged = 0
    while True:
        pairs = []
        for i, a in enumerate(cands):
            for j in range(i + 1, len(cands)):
                b = cands[j]
                key = frozenset([tuple(a.segs), tuple(b.segs)])
                if (
                    min(abs(seg_freq[x] - seg_freq[y]) for x in a.segs for y in b.segs)
                    > cfg.merge_freq_tolerance
                    or key in rejected
                    or _overlaps(a.spans, b.spans, cfg.max_overlap)
                ):
                    continue
                dist = min(
                    float(np.hypot(*(pa - pb)))
                    for pa in a.positions()
                    for pb in b.positions()
                )
                thr = cfg.merge_se_factor * np.hypot(_se(a, cfg), _se(b, cfg))
                if dist <= thr:
                    pairs.append((dist / thr, i, j, key))
        if not pairs:
            break
        pairs.sort(key=lambda p: p[0])
        used: set[int] = set()
        batch = []
        for _, i, j, key in pairs:
            if i in used or j in used:
                continue
            used |= {i, j}
            batch.append((i, j, key))
        items = []
        for i, j, _ in batch:
            obs, _ = ctx.observations(cands[i].segs + cands[j].segs)
            items.append((obs, cfg, cands[i].inits() + cands[j].inits()))
        outs = _map(_merge_task, items, n_jobs, "merge", progress)
        merged, drop = [], set()
        for (i, j, key), (res, sr) in zip(batch, outs, strict=True):
            bad = _bad_segments(sr, cfg)
            log.debug(
                "merge %.1f Hz + %.1f Hz: %s",
                cands[i].freq,
                cands[j].freq,
                f"rejected (segments {bad})" if bad else "accepted",
            )
            if bad:
                rejected.add(key)
                continue
            a, b = cands[i], cands[j]
            segs_ab = a.segs + b.segs
            merged.append(
                Candidate(
                    segs_ab,
                    float(np.median(results.fund_v[np.isin(seg_of, segs_ab)])),
                    a.spans + b.spans,
                    res,
                    sr,
                )
            )
            drop |= {i, j}
        if not merged:
            continue
        add_se(merged, "merge jackknife")
        n_merged += len(merged)
        cands = [c for k, c in enumerate(cands) if k not in drop] + merged
    timings["merge"] = time.perf_counter() - t0
    log.info("merged %d pairs of candidates", n_merged)

    # outputs
    cands.sort(key=lambda c: c.freq)
    fish_v = np.full(len(results.fund_v), np.nan)
    seg_fish = np.full(len(segs), -1)
    seg_res = np.full(len(segs), np.nan)
    seg_dist = np.full(len(segs), np.nan)
    rows = []
    for k, c in enumerate(cands):
        fish_v[np.isin(seg_of, c.segs)] = k
        seg_fish[c.segs] = k
        for s, (v, d) in c.seg_check.items():
            seg_res[s] = v
            seg_dist[s] = d
        span = np.array(c.spans)
        d = c.fit.to_dict()
        d["heading"] = float(np.degrees(d["heading"]))
        rows.append(
            {
                "fish": k,
                "freq": c.freq,
                **d,
                "n_segments": len(c.segs),
                "t_first": float(span[:, 0].min()),
                "t_last": float(span[:, 1].max()),
                "segments": " ".join(map(str, sorted(c.segs))),
            }
        )
    table = pd.DataFrame(rows)
    segs = segs.assign(fish=seg_fish, resid=seg_res, dist=seg_dist)
    timings["total"] = time.perf_counter() - t_start
    stats = {
        "segments": len(segs),
        "clutter_segments": int(segs.clutter.sum()),
        "initial_candidates": n_initial,
        "segments_split_off": len(evicted),
        "merges": n_merged,
        "fish": len(cands),
        "ambiguous": int(table.ambiguous.sum()) if len(table) else 0,
        "survey_detections": int(in_survey.sum()),
        "assigned_detections": int(np.isfinite(fish_v).sum()),
        "timings": timings,
        "config": asdict(cfg),
    }
    return MergeOutput(fish_v, table, segs, stats)


def load_fish(folder: str | Path) -> tuple[np.ndarray, pd.DataFrame]:
    """Fish identity per detection and the fish table of a results directory."""
    folder = Path(folder)
    return np.load(folder / FISH_FILE), pd.read_csv(folder / TABLE_FILE)
