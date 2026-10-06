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
