import numpy as np
import pytest

from wavetracker.config import Config, TrackingConfig
from wavetracker.pipeline import detect
from wavetracker.tracking import track

from .legacy_tracking import freq_tracking_v6


@pytest.fixture(scope="module")
def detections(synthetic_wav, tmp_path_factory):
    path, _ = synthetic_wav
    return detect(path, tmp_path_factory.mktemp("det"), Config(), device="cpu").results


def _same(a, b):
    return bool(np.all((np.isnan(a) & np.isnan(b)) | (a == b)))


@pytest.mark.slow
def test_matches_original_implementation(detections):
    r = detections
    cfg = TrackingConfig()
    new = track(r.fund_v, r.idx_v, r.sign_v, r.times, cfg, 400, 1200, v6_compat=True)
    old = freq_tracking_v6(
        r.fund_v,
        r.idx_v,
        r.sign_v.astype(float),
        r.times,
        freq_tolerance=cfg.freq_tolerance,
        max_dt=cfg.max_dt,
        min_freq=400,
        max_freq=1200,
    )
    assert _same(new, old)


@pytest.mark.slow
def test_matches_original_with_noise_detections(detections):
    """Random spurious detections stress the tie-breaking and merge rules."""
    r = detections
    rng = np.random.default_rng(3)
    n = len(r.fund_v) // 3
    idx = np.concatenate([r.idx_v, rng.integers(0, r.idx_v.max(), n)])
    fund = np.concatenate(
        [r.fund_v, r.fund_v[rng.integers(0, len(r.fund_v), n)] + rng.normal(0, 2, n)]
    )
    sign = np.concatenate([r.sign_v, rng.random((n, r.sign_v.shape[1]))])
    order = np.argsort(idx, kind="stable")
    idx, fund, sign = idx[order], fund[order], sign[order].astype(float)
    cfg = TrackingConfig()
    new = track(fund, idx, sign, r.times, cfg, 400, 1200, v6_compat=True)
    old = freq_tracking_v6(
        fund,
        idx,
        sign,
        r.times,
        freq_tolerance=cfg.freq_tolerance,
        max_dt=cfg.max_dt,
        min_freq=400,
        max_freq=1200,
    )
    assert _same(new, old)


def test_out_of_range_detections_are_not_tracked(detections):
    r = detections
    ident = track(r.fund_v, r.idx_v, r.sign_v, r.times, TrackingConfig(), 600, 1200)
    assert np.all(np.isnan(ident[r.fund_v < 600]))
    assert np.any(~np.isnan(ident[r.fund_v >= 600]))


def test_empty_input():
    ident = track(
        np.empty(0),
        np.empty(0, int),
        np.empty((0, 3)),
        np.arange(10.0),
        TrackingConfig(),
    )
    assert ident.shape == (0,)


def test_support_counts_neighbours():
    from wavetracker.tracking import support

    times = np.arange(100) * 0.1
    # a line at 600 Hz in every frame, plus an isolated noise detection
    idx = np.concatenate([np.arange(50), [25]])
    fund = np.concatenate([np.full(50, 600.0), [603.0]])
    order = np.argsort(idx, kind="stable")
    s = support(fund[order], idx[order], times, window=0.5, df=1.0)
    s = s[np.argsort(order)]
    assert s[-1] == 0  # isolated
    assert s[25] >= 5


def test_support_filter_blocks_noise_bridges():
    """Two fish 4 Hz apart, a gap in fish A, and noise detections bridging
    A to B: with the support filter A is not continued on B."""
    dt = 0.1
    times = np.arange(400) * dt
    rows = [(k, 600.0) for k in range(0, 150)] + [(k, 600.0) for k in range(250, 400)]
    rows += [(k, 604.0) for k in range(150, 400)]
    rows += [(155, 601.5), (165, 602.5), (175, 603.5)]  # stepping stones
    rows.sort()
    idx = np.array([r[0] for r in rows])
    fund = np.array([r[1] for r in rows])
    sign = np.ones((len(fund), 2))
    cfg = TrackingConfig(freq_tolerance=2.5, max_dt=5.0, min_support=3)
    ident = track(fund, idx, sign, times, cfg)
    early_a = ident[(fund == 600.0) & (idx < 150)]
    fish_b = ident[fund == 604.0]
    assert not np.isin(np.unique(early_a), np.unique(fish_b)).any()


def test_db_amplitude_feature():
    from wavetracker.tracking import normalize_signatures

    sign = np.array([[1.0, 10.0], [2.0, 2.0]])
    db = normalize_signatures(sign, "db")
    np.testing.assert_allclose(db, [[-5.0, 5.0], [0.0, 0.0]], atol=1e-9)
    mm = normalize_signatures(sign[:1], "minmax")
    np.testing.assert_allclose(mm, [[0.0, 1.0]])
