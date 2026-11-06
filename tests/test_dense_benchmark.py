"""The dense-mixture benchmark (benchmarks/dense) on synthetic signals: the
split of a source into fish and residual, the gain processes, the matching."""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).parents[1] / "benchmarks" / "dense"))

import mixture
import score

RATE = 20000.0


def _source(duration=20.0, f=(500.0, 620.0), noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * RATE)) / RATE
    x = noise * rng.standard_normal((len(t), 2))
    gains = ((1.0, 0.3), (0.4, 1.0))  # fish x electrode
    for k, fk in enumerate(f):
        w = sum(
            a * np.sin(2 * np.pi * h * fk * t)
            for h, a in ((1, 1.0), (2, 0.5), (3, 0.25))
        )
        x += np.outer(w, gains[k])
    tt = np.arange(0, duration, 0.1)
    traces = [(tt, np.full(len(tt), fk), np.zeros(len(tt), bool)) for fk in f]
    return x.astype(np.float32), traces


def _band_db(y, f0, width=3.0):
    spec = np.abs(np.fft.rfft(y.sum(1) * np.hanning(len(y)))) ** 2
    freqs = np.fft.rfftfreq(len(y), 1 / RATE)
    return 10 * np.log10(spec[np.abs(freqs - f0) <= width].mean())


def test_split_is_exact_and_separates_the_fish():
    x, traces = _source()
    f0, f1, r = mixture.split_source(
        x, traces, np.array([]), 1.0, np.random.default_rng(1), device="cpu"
    )
    assert mixture.split_source.last_qc["line_bins"] == [0, 0]
    np.testing.assert_allclose(f0 + f1 + r, x, atol=1e-4 * np.abs(x).max())
    for h in (1, 2, 3):
        assert _band_db(f0, 500 * h) - _band_db(f0, 620 * h) > 40
        assert _band_db(f1, 620 * h) - _band_db(f1, 500 * h) > 40
    # the residual is noise, also under the fish
    noise = _band_db(r, 560.0, 20.0)
    assert abs(_band_db(r, 500.0) - noise) < 3
    assert abs(_band_db(r, 620.0) - noise) < 3


def test_attenuated_fish_leaves_the_noise_floor():
    x, traces = _source()
    f0, f1, r = mixture.split_source(
        x, traces, np.array([]), 1.0, np.random.default_rng(1), device="cpu"
    )
    y = f1 + r  # fish 0 removed: noise remains, no hole
    floor = _band_db(y, 560.0, 20.0)
    assert abs(_band_db(y, 500.0) - floor) < 2
    assert abs(_band_db(y, 1000.0) - _band_db(y, 1060.0, 20.0)) < 2
    # attenuated by 40 dB: 40 dB less fish
    gain = _band_db(1e-2 * f0 + f1 + r, 1500.0, 0.5) - _band_db(x, 1500.0, 0.5)
    assert -41 < gain < -39


def test_split_removes_what_would_alias():
    x, traces = _source(duration=5.0)
    x[:, 0] += np.sin(2 * np.pi * 9000 * np.arange(len(x)) / RATE).astype(np.float32)
    f0, f1, r = mixture.split_source(
        x, traces, np.array([]), 1.5, np.random.default_rng(1), device="cpu"
    )
    # content above 0.9 * Nyquist / 1.5 = 6 kHz is gone
    assert _band_db(f0 + f1 + r, 9000.0) < _band_db(x, 9000.0) - 60


def test_gains_ramp_and_dyads():
    rng = np.random.default_rng(0)
    g = mixture.gains({"ramp_db": 60.0}, 6, 600.0, rng)
    span = g["common_db"].max(1) - g["common_db"].min(1)
    np.testing.assert_allclose(span, 60.0, atol=0.2)
    assert np.all(g["common_db"].max(1) <= 1e-9)
    g = mixture.gains(
        {"per": "dyad", "static_db": [-40, 0], "fraction": 0.5}, 6, 600.0, rng
    )
    c = g["common_db"]
    np.testing.assert_array_equal(c[0::2], c[1::2])  # both fish of a source
    assert g["modulated"].sum() == 4  # half of 3 dyads, rounded to 2: 4 fish
    assert np.all(g["channel_db"] == 0)
    g = mixture.gains({"channel_db": 3.0}, 2, 600.0, rng)
    assert 1.5 < g["channel_db"].std() < 4.5
    assert np.all(g["common_db"] == 0)


def test_match_is_one_to_one_per_frame():
    from wavetracker.results import Results

    times = np.arange(3) * 0.2
    F = np.array([[500.0, 500.0, 500.0], [501.0, 501.0, 501.0]])
    # frame 0: two detections between the fish; frame 1: two on fish 0
    fund = np.array([500.4, 500.6, 500.0, 500.1, 700.0])
    idx = np.array([0, 0, 1, 1, 2])
    res = Results(fund, idx, np.ones((5, 1)), np.zeros(5), times)
    fish = score.match(res, F, tol=2.0)
    assert sorted(fish[:2]) == [0, 1]
    assert sorted(fish[2:4]) == [0, 1]  # second one goes to fish 1 (0.9 Hz)
    assert fish[4] == -1


def test_noise_in_fish_masks_stays_in_the_residual():
    """Only what stands out of the noise goes to a fish; in a mask widened
    for a gap of the truth, noise-only bins must not (they would carry the
    source's unscaled noise into the mixture)."""
    x, traces = _source(noise=0.2)  # ~50 dB per bin, as the tube fish
    traces = [(t, f, np.ones(len(t), bool)) for t, f, _ in traces]  # all gap
    f0, _, r = mixture.split_source(
        x, traces, np.array([]), 1.0, np.random.default_rng(1), device="cpu"
    )
    # 505-512 Hz: inside fish 0's widened mask (up to +15 Hz), noise only
    assert _band_db(f0, 508.5, 3.5) < _band_db(x, 508.5, 3.5) - 10
    assert abs(_band_db(r, 508.5, 3.5) - _band_db(x, 508.5, 3.5)) < 2


def test_ridge_path_follows_a_rise_and_avoids_the_band_edge():
    import library

    T, B = 60, 70
    ridge = np.full(T, 10)
    ridge[20:] = 10 + np.round(25 * np.exp(-(np.arange(T - 20)) / 8)).astype(int)
    ridge[20] = 22  # fast onset: 12 bins in one frame
    score = np.zeros((T, B))
    score[np.arange(T), ridge] = 20.0
    score[:, 60] = 30.0  # a strong line outside the allowed band
    lo = np.zeros(T, int)
    hi = np.full(T, 50)
    path = library._ridge_path(score, lo, hi, start=10, end=int(ridge[-1]))
    np.testing.assert_array_equal(path, ridge)


def test_ridge_path_in_a_band_narrower_than_a_step():
    import library

    score = np.zeros((5, 4))
    score[:, 2] = 10.0
    path = library._ridge_path(score, np.zeros(5, int), np.full(5, 3), None, None)
    np.testing.assert_array_equal(path, 2)


def test_dyads_too_wide_for_their_stratum_do_not_stack():
    """In a narrow band most strata cannot hold a wide dyad; clipping them to
    the edge of what fits stacked several dyads on the same frequencies."""
    pool = [
        {
            "name": f"t{i}",
            "median_hz": [640.0 + i, 864.0 + i],
            "range_hz": [[635.0 + i, 645.0 + i], [859.0 + i, 869.0 + i]],
            "length": 1200.0,
            "start": 1000.0,
            "channels_used": list(range(9)),
        }
        for i in range(20)
    ]
    design = {"n_fish": 30, "band": [550, 850], "duration": 600.0}
    src = mixture.place_sources(design, pool, np.random.default_rng(0))
    fish = np.sort(np.concatenate([s["fish_hz"] for s in src]))
    assert len(fish) == 30
    # chance may bring two fish close, stacking puts many on one spot
    in_window = [np.sum(np.abs(fish - f) <= 1.0) for f in fish]
    assert max(in_window) <= 3
    assert fish.min() >= 550 - 10 and fish.max() <= 850 + 10


def test_detected_flags_follow_the_nearest_source_frame():
    st = np.arange(11) * 1.0
    sd = np.where(st < 5, 500.0, np.nan)
    tq = np.array([-0.3, 0.4, 4.4, 4.6, 7.0, 10.2])
    assert mixture._detected_at(st, sd, tq).tolist() == [
        True,
        True,
        True,
        False,
        False,
        False,
    ]


def test_recall_is_split_into_detected_and_completed_frames(tmp_path):
    import json

    from wavetracker.results import Results

    times = np.arange(10) * 0.1
    freqs = np.full((1, 10), 500.0)
    mix = tmp_path / "mix"
    mix.mkdir()
    np.savez(
        mix / "truth.npz",
        times=times,
        freqs=freqs,
        freqs_filled=freqs,
        freqs_detected=np.where(np.arange(10) < 5, freqs, np.nan),
        source=np.array([0]),
        gain_db=np.zeros((1, 10)),
        snr_times=times,
        snr_db=np.full((1, 10), 30.0),
    )
    (mix / "mixture.json").write_text(json.dumps({"name": "dev/x"}))
    idx = np.array([0, 1, 2, 7])
    res = Results(np.full(4, 500.0), idx, np.ones((4, 1)), np.zeros(4), times)
    res.save(tmp_path / "res")
    summary, table = score.score(mix, tmp_path / "res")
    assert summary["recall"] == pytest.approx(0.4)
    assert summary["recall_detected"] == pytest.approx(0.6)
    assert summary["recall_completed"] == pytest.approx(0.2)
    assert summary["completed"] == pytest.approx(0.5)
    assert table["detected"].tolist() == [True] * 5 + [False] * 5
