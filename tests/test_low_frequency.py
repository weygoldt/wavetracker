"""Odd-harmonic low fish in a dense chorus (harmonic groups with gaps)."""

import numpy as np
import pytest

from wavetracker.config import Config
from wavetracker.evaluation import match_detections
from wavetracker.pipeline import detect
from wavetracker.synthetic import low_frequency_chorus, save_recording, synthesize

N_LOW = 8
FIELD = {
    "spectrogram": {"nfft": 65536, "overlap_frac": 0.9},
    "interference": {"enabled": False},
    "harmonic_groups": {
        "min_freq": 20.0,
        "max_freq": 2000.0,
        "mains_freq": 0.0,
        "low_thresh_factor": 3.0,
        "high_thresh_factor": 5.0,
        "min_group_size": 2,
    },
}


@pytest.fixture(scope="module")
def chorus(tmp_path_factory):
    rng = np.random.default_rng(0)
    fish = low_frequency_chorus(15.0, n_low=N_LOW, n_chorus=30, rng=rng)
    rec = synthesize(fish, 15.0, rate=48000.0, channels=2, mains=0.0, rng=rng)
    path = tmp_path_factory.mktemp("chorus") / "rec.wav"
    save_recording(rec, path)
    return path, rec


def _score(path, rec, tmp_path, missing):
    cfg = Config.from_dict(FIELD)
    cfg.harmonic_groups.max_missing_harmonics = missing
    res = detect(path, tmp_path / f"m{missing}", cfg, device="cpu").results
    fish = match_detections(res, rec.truth_times, rec.truth_freqs, tol=1.0)
    n = len(res.times)
    recall = np.array(
        [len(np.unique(res.idx_v[fish == k])) / n for k in range(len(rec.truth_freqs))]
    )
    t = res.times[res.idx_v]
    sub = np.stack(
        [np.interp(t, rec.truth_times, f) / m for f in rec.truth_freqs for m in (2, 3)]
    )
    ghosts = (fish < 0) & np.any(np.abs(sub - res.fund_v) <= 1.0, axis=0)
    return recall[:N_LOW].mean(), recall[N_LOW:].mean(), ghosts.sum() / n


def test_gaps_find_low_fish_without_ghosts(chorus, tmp_path):
    path, rec = chorus
    low0, chorus0, ghosts0 = _score(path, rec, tmp_path, 0)
    low1, chorus1, ghosts1 = _score(path, rec, tmp_path, 1)
    assert low0 < 0.4  # missing 2nd harmonic: mostly undetected
    assert low1 > 0.7
    assert chorus1 >= chorus0 - 0.01
    assert ghosts1 <= ghosts0  # sub-harmonic ghosts per frame
