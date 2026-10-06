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
