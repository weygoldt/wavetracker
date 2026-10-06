"""Scoring tracking results against ground truth frequency traces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .results import Results


@dataclass
class FishScore:
    fish: int
    recall: float
    """Fraction of frames in which the fish was detected (any identity)."""
    coverage: float
    """Fraction of frames covered by the fish's dominant identity."""
    n_ids: int
    """Number of identities that cover at least 5 % of this fish."""
    freq_error: float
    """Median absolute frequency error of matched detections [Hz]."""


@dataclass
class Score:
    fish: list[FishScore]
    precision: float
    """Fraction of detections that match a fish."""
    purity: float
    """Fraction of assigned detections whose identity's dominant fish is right."""

    @property
    def recall(self) -> float:
        return float(np.mean([f.recall for f in self.fish]))

    @property
    def coverage(self) -> float:
        return float(np.mean([f.coverage for f in self.fish]))

    def __str__(self) -> str:
        lines = [
            f"precision {self.precision:.3f}  recall {self.recall:.3f}  "
            f"coverage {self.coverage:.3f}  purity {self.purity:.3f}"
        ]
        lines += [
            f"  fish {f.fish}: recall {f.recall:.3f} coverage {f.coverage:.3f} "
            f"ids {f.n_ids} |df| {f.freq_error:.2f} Hz"
            for f in self.fish
        ]
        return "\n".join(lines)


def match_detections(
    results: Results, truth_times: np.ndarray, truth_freqs: np.ndarray, tol: float = 1.5
) -> np.ndarray:
    """Index of the ground-truth fish of each detection (-1 if none within `tol`)."""
    t = results.times[results.idx_v]
    ref = np.stack([np.interp(t, truth_times, f) for f in truth_freqs])  # (fish, det)
    err = np.abs(ref - results.fund_v[None, :])
    best = err.argmin(0)
    return np.where(err[best, np.arange(len(best))] <= tol, best, -1)


def evaluate(
    results: Results, truth_times: np.ndarray, truth_freqs: np.ndarray, tol: float = 1.5
) -> Score:
    fish = match_detections(results, truth_times, truth_freqs, tol)
    ident = results.ident_v
    t = results.times[results.idx_v]
    in_range = (results.times >= truth_times[0]) & (results.times <= truth_times[-1])
    n_frames = int(in_range.sum())

    # dominant fish of every identity
    dominant = {}
    for i in np.unique(ident[~np.isnan(ident)]):
        f = fish[ident == i]
        f = f[f >= 0]
        dominant[i] = np.bincount(f).argmax() if len(f) else -1
    assigned = ~np.isnan(ident)
    correct = np.array(
        [dominant[i] == f for i, f in zip(ident[assigned], fish[assigned], strict=True)]
    )
    purity = float(correct.mean()) if len(correct) else 0.0

    scores = []
    for k in range(len(truth_freqs)):
        m = fish == k
        frames = np.unique(results.idx_v[m])
        ids, counts = np.unique(ident[m & assigned], return_counts=True)
        top = ids[counts.argmax()] if len(ids) else np.nan
        cover = len(np.unique(results.idx_v[m & (ident == top)])) if len(ids) else 0
        ref = np.interp(t[m], truth_times, truth_freqs[k])
        scores.append(
            FishScore(
                fish=k,
                recall=len(frames) / max(n_frames, 1),
                coverage=cover / max(n_frames, 1),
                n_ids=int((counts >= 0.05 * max(m.sum(), 1)).sum()),
                freq_error=float(np.median(np.abs(results.fund_v[m] - ref)))
                if m.any()
                else np.nan,
            )
        )
    precision = float((fish >= 0).mean()) if len(fish) else 0.0
    return Score(scores, precision, purity)
