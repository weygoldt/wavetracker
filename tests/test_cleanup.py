import numpy as np

from wavetracker.postprocessing.cleanup import assign_leftovers

DT = 0.164


def _track(t0, t1, f, ident):
    t = np.arange(t0, t1, DT)
    return (
        np.round(t / DT).astype(int),
        np.full(len(t), f),
        np.full(len(t), float(ident)),
    )


def _join(*tracks):
    idx = np.concatenate([t[0] for t in tracks])
    order = np.argsort(idx, kind="stable")
    return (
        np.concatenate([t[1] for t in tracks])[order],
        idx[order],
        np.concatenate([t[2] for t in tracks])[order],
        np.arange(0, idx.max() + 1) * DT,
    )


def test_continuation_is_assigned():
    fund, idx, ident, times = _join(
        _track(0, 600, 700.0, 0), _track(0, 1200, 640.0, 1), _track(610, 1200, 700.5, 2)
    )
    out = assign_leftovers(fund, idx, ident, times, kept_ids=[0, 1])
    assert np.all(out[ident == 2] == 0)


def test_ambiguous_or_conflicting_leftovers_stay():
    # leftover overlaps fish 0 in time -> frame conflict; fish 1 is far in frequency
    fund, idx, ident, times = _join(
        _track(0, 600, 700.0, 0), _track(0, 1200, 640.0, 1), _track(300, 900, 700.2, 2)
    )
    out = assign_leftovers(fund, idx, ident, times, kept_ids=[0, 1])
    assert np.all(out[ident == 2] == 2)
    # two kept fish that both continue at the leftover's frequency -> ambiguous
    fund, idx, ident, times = _join(
        _track(0, 600, 700.0, 0),
        _track(1300, 1800, 700.3, 1),
        _track(610, 1200, 700.1, 2),
    )
    out = assign_leftovers(fund, idx, ident, times, kept_ids=[0, 1])
    assert np.all(out[ident == 2] == 2)


def test_small_leftovers_are_ignored():
    fund, idx, ident, times = _join(
        _track(0, 600, 700.0, 0), _track(605, 630, 700.0, 2)
    )
    out = assign_leftovers(fund, idx, ident, times, kept_ids=[0])
    assert np.all(out[ident == 2] == 2)


def test_frequency_kde_matches_dense_gaussians():
    from wavetracker.postprocessing.cleanup import frequency_kde, gauss

    rng = np.random.default_rng(0)
    ff = np.concatenate([rng.normal(640, 1, 300), rng.normal(790, 1, 400)])
    grid = np.arange(np.floor(ff.min() - 25), np.ceil(ff.max() + 25), 0.1)
    dense = gauss(grid, ff, sigma=5.0, size=1, norm=True)
    kde, g_max = frequency_kde(ff, grid, 5.0)
    np.testing.assert_allclose(kde, dense.sum(0), atol=1e-3 * dense.sum(0).max())
    assert abs(g_max - dense.max()) < 1e-6


def test_frequency_kde_memory_is_linear():
    """143k detections over 20-2000 Hz (the Iriri case): the dense version
    needed ~46 GB; this must stay small."""
    import tracemalloc

    from wavetracker.postprocessing.cleanup import frequency_kde

    ff = np.random.default_rng(1).uniform(20, 2000, 143_531)
    grid = np.arange(0.0, 2025.0, 0.1)
    tracemalloc.start()
    frequency_kde(ff, grid, 5.0)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert peak < 50e6


def test_gauss_rows_outside_grid_are_zero_not_nan():
    from wavetracker.postprocessing.cleanup import gauss

    g = gauss(np.arange(400, 1200, 0.1), np.array([600.0, 1900.0]), 5.0, 1, norm=True)
    assert np.all(np.isfinite(g)) and g[1].sum() == 0


def test_window_without_detections_returns_empty_table():
    from wavetracker.postprocessing.cleanup import get_valid_ids_by_freq_dist

    times = np.arange(1000) * 0.1
    idx = np.arange(100)  # detections only in the first 10 s
    ident = np.zeros(100)
    fund = np.full(100, 700.0)
    kde_th, valid = get_valid_ids_by_freq_dist(
        times, idx, ident, fund, np.zeros(100), np.array([0.0]), 50.0, 10.0, 2.5, 3.0
    )
    assert kde_th == 3.0 and valid.shape == (0, 3)
