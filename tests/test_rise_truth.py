"""Truth through the steep part of rises (benchmarks/dense/rise_truth.py) on
synthetic recordings: the instantaneous frequency, the onset follower, the
warped mixture truth and the rise-onset stratum of the scoring."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).parents[1] / "benchmarks" / "dense"))

import mixture
import rise_truth
import score
from common import Trial, Warp

RATE = 20000.0
FRAME = 0.1638  # wavetracker's frame step at 20 kHz


def _rise(t, t0=10.0, f_pre=600.0, size=20.0, climb=0.12, tau=4.0):
    """A rise: a climb of `size` Hz within about `climb` s at `t0`, then an
    exponential decay."""
    up = 1 / (1 + np.exp(-(t - t0) / (climb / 8)))
    decay = np.exp(-np.clip(t - t0, 0, None) / tau)
    return f_pre + size * up * decay


def _recording(tmp_path, duration=20.0, noise=0.01, seed=0):
    """Two fish on two electrodes: fish 0 rises at 10 s, fish 1 at 700 Hz."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * RATE)) / RATE
    f0 = _rise(t)
    phase0 = 2 * np.pi * np.cumsum(f0) / RATE
    phase1 = 2 * np.pi * 700.0 * t
    x = noise * rng.standard_normal((len(t), 2))
    for ph, g in ((phase0, (1.0, 0.4)), (phase1, (0.3, 1.0))):
        w = np.sin(ph) + 0.4 * np.sin(2 * ph)
        x += np.outer(w, g)
    (tmp_path / "traces-grid1.raw").write_bytes(x.astype(np.float32).tobytes())
    trial = Trial(
        "synthetic", "2022", str(tmp_path), 2, duration, (600, 700), "g", "dev"
    )
    return trial


def _source_truth(gap=(9.9, 10.6)):
    """Old truth on wavetracker's frames: detected except across the onset,
    where the ridge completion bridged it along a straight line."""
    times = 0.82 + np.arange(int(18.0 / FRAME)) * FRAME
    f0 = _rise(times)
    det0 = np.where((times > gap[0]) & (times < gap[1]), np.nan, f0)
    ok = np.isfinite(det0)
    bridged = np.interp(times, times[ok], det0[ok])
    truth = np.stack([bridged, np.full(len(times), 700.0)])
    detected = np.stack([det0, np.full(len(times), 700.0)])
    return times, truth, detected


def test_pulse_pair_follows_a_steep_climb():
    t = np.arange(int(3.0 * RATE)) / RATE
    f = _rise(t, t0=1.5, size=30.0, climb=0.1)
    x = np.sin(2 * np.pi * np.cumsum(f) / RATE)[:, None]
    z = rise_truth.band_analytic(x, 590.0, 640.0)
    ti, fi, _ = rise_truth.pulse_pair(z)
    inner = (ti > 0.5) & (ti < 2.5)
    err = np.abs(fi - np.interp(ti, t, f))[inner]
    assert np.median(err) < 0.05
    assert err.max() < 2.0  # 30 Hz within 0.1 s, followed within a 30-ms window


def test_follow_onset_replaces_the_bridge_by_the_climb(tmp_path):
    trial = _recording(tmp_path)
    times, truth, detected = _source_truth()
    rises = rise_truth.onsets(times, truth)
    assert len(rises) == 1 and rises[0]["fish"] == 0
    rec = rise_truth.follow_onset(trial, [0, 1], times, truth, detected, rises[0])
    assert rec["status"] == "corrected"
    fr = np.asarray(rec["frames"])
    # the new frames are the frames between the anchors, the old gap
    assert np.all((times[fr] > 9.9 - FRAME) & (times[fr] < 10.6 + FRAME))
    true = _rise(times[fr])
    np.testing.assert_allclose(rec["freqs"], true, atol=1.0)
    assert np.abs(truth[0, fr] - true).max() > 5.0  # the bridge was off
    ft = np.asarray(rec["fine_times"])
    assert ft.min() > rec["t_a"] and ft.max() < rec["t_b"]
    np.testing.assert_allclose(rec["fine_freqs"], _rise(ft), atol=1.5)


def test_follow_onset_keeps_out_of_the_other_fish(tmp_path):
    trial = _recording(tmp_path)
    times, truth, detected = _source_truth()
    truth[1] = 612.0  # the other fish in the range of the rise
    detected[1] = 612.0
    rises = rise_truth.onsets(times, truth)
    rec = rise_truth.follow_onset(trial, [0, 1], times, truth, detected, rises[0])
    assert rec["status"] == "other-fish"


def test_warp_truth_interpolates_short_gaps_only():
    grid_t = np.arange(0.0, 50.0)
    w = Warp(0.5, 100.0, grid_t, np.zeros(50), np.zeros(50))
    st = 100.0 + np.arange(0, 20.0, 0.1)
    sf = np.full(len(st), 600.0)
    sf[30:35] = np.nan  # 0.6 s: bridged
    sf[100:120] = np.nan  # 2.1 s: unknown
    t_truth = np.arange(0.0, 30.0, 0.1)
    truth, filled = mixture.warp_truth(w, st, sf, t_truth)
    np.testing.assert_allclose(filled, 300.0)  # frequency times the scale
    tq = w.tau(t_truth)
    assert np.isfinite(truth[(tq > 103.0) & (tq < 103.5)]).all()
    assert np.isnan(truth[(tq > 110.5) & (tq < 111.5)]).all()


def test_score_has_a_rise_onset_stratum(tmp_path):
    from wavetracker.results import Results

    times = np.arange(10) * 0.1
    freqs = np.full((1, 10), 500.0)
    rise = np.zeros((1, 10), bool)
    rise[0, 4:8] = True
    mix = tmp_path / "mix"
    mix.mkdir()
    common = {
        "times": times,
        "freqs": freqs,
        "freqs_filled": freqs,
        "freqs_detected": np.where(rise, np.nan, freqs),
        "source": np.array([0]),
        "gain_db": np.zeros((1, 10)),
        "snr_times": times,
        "snr_db": np.full((1, 10), 30.0),
    }
    np.savez(mix / "truth.npz", **common)
    np.savez(mix / "truth_rises.npz", **common, rise=rise)
    (mix / "mixture.json").write_text(json.dumps({"name": "ldev/x"}))
    idx = np.array([0, 1, 2, 4, 5])
    res = Results(np.full(5, 500.0), idx, np.ones((5, 1)), np.zeros(5), times)
    res.save(tmp_path / "res")
    default, table = score.score(mix, tmp_path / "res")
    assert default["rise_frames"] == 0 and "rise" not in table
    summary, table = score.score(mix, tmp_path / "res", truth="rises")
    assert summary["rise_frames"] == 4
    assert summary["recall_rise"] == pytest.approx(0.5)
    assert table["rise"].sum() == 4
    assert score.score_file("default") == "score.json"
    assert score.score_file("rises", "frames", ".npz") == "frames_rises.npz"
