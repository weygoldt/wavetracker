import numpy as np
import pytest

from wavetracker.config import Config, StitchingConfig
from wavetracker.evaluation import evaluate
from wavetracker.pipeline import detect, track_results
from wavetracker.stitching import resolve_duplicates, stitch
from wavetracker.synthetic import Fish, save_recording, synthesize

DT = 0.164


def _fragment(t0, t1, f, ident, rise=None, n_ch=4, pattern=None):
    t = np.arange(t0, t1, DT)
    freq = np.full(len(t), f, dtype=float)
    if rise is not None:  # (size, tau): decaying excursion at the fragment start
        freq += rise[0] * np.exp(-(t - t0) / rise[1])
    pattern = np.ones(n_ch) if pattern is None else pattern
    return t, freq, np.full(len(t), ident, dtype=float), np.tile(pattern, (len(t), 1))


def _assemble(*frags):
    t = np.concatenate([f[0] for f in frags])
    order = np.argsort(t, kind="stable")
    times = np.arange(0, t.max() + 1, DT)
    idx = np.round(t[order] / DT).astype(int)
    return (
        np.concatenate([f[1] for f in frags])[order],
        idx,
        np.concatenate([f[3] for f in frags])[order],
        np.concatenate([f[2] for f in frags])[order],
        times,
    )


def test_joins_across_rise():
    fund, idx, sign, ident, times = _assemble(
        _fragment(0, 100, 700.0, 0), _fragment(100.3, 300, 700.5, 1, rise=(20, 8))
    )
    out = stitch(fund, idx, sign, ident, times, StitchingConfig())
    assert len(np.unique(out)) == 1


def test_does_not_join_different_baselines():
    fund, idx, sign, ident, times = _assemble(
        _fragment(0, 100, 700.0, 0), _fragment(100.3, 300, 715.0, 1)
    )
    out = stitch(fund, idx, sign, ident, times, StitchingConfig())
    assert len(np.unique(out)) == 2


def test_dropout_joined_only_without_competitor():
    a = _fragment(0, 100, 700.0, 0)
    b = _fragment(400, 600, 700.4, 1)
    fund, idx, sign, ident, times = _assemble(a, b)
    out = stitch(fund, idx, sign, ident, times, StitchingConfig())
    assert len(np.unique(out)) == 1
    # another fish at that frequency, overlapping both fragments: no join
    c = _fragment(50, 450, 700.2, 2)
    fund, idx, sign, ident, times = _assemble(a, b, c)
    out = stitch(fund, idx, sign, ident, times, StitchingConfig())
    assert out[ident == 0][0] != out[ident == 1][0]


def test_greedy_prefers_matching_continuation():
    a = _fragment(0, 100, 700.0, 0)
    good = _fragment(100.3, 200, 700.2, 1)
    bad = _fragment(100.5, 200, 702.6, 2)
    fund, idx, sign, ident, times = _assemble(a, good, bad)
    out = stitch(fund, idx, sign, ident, times, StitchingConfig())
    assert out[ident == 1][0] == out[ident == 0][0]
    assert out[ident == 2][0] != out[ident == 0][0]


def test_resolve_duplicates_keeps_closest():
    fund = np.array([700.0, 700.1, 703.0, 700.2, 700.1])
    idx = np.array([0, 1, 1, 2, 3])
    ident = np.zeros(5)
    out = resolve_duplicates(fund, idx, ident)
    assert np.isnan(out[2]) and not np.isnan(out[1])


@pytest.mark.slow
def test_stitching_reduces_fragmentation_without_mixing(tmp_path):
    rng = np.random.default_rng(5)
    rises = [
        (float(t), float(rng.uniform(15, 40)), float(rng.uniform(3, 15)))
        for t in np.arange(20, 280, 25)
    ]
    fish = [
        Fish(640.3, drift=1.0, position=0.3, rises=rises),
        Fish(720.9, drift=1.0, position=0.7),
    ]
    rec = synthesize(fish, 300.0, channels=6, rng=rng)
    path = tmp_path / "rises.wav"
    save_recording(rec, path)
    res = detect(path, tmp_path / "out", Config()).results
    scores = {}
    for enabled in (False, True):
        cfg = Config()
        cfg.stitching.enabled = enabled
        track_results(res, cfg)
        scores[enabled] = evaluate(res, rec.truth_times, rec.truth_freqs, tol=60)
    assert scores[True].purity > 0.99
    assert scores[True].fish[0].coverage > scores[False].fish[0].coverage + 0.2
