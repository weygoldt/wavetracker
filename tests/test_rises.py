import numpy as np
import pytest

from wavetracker.config import Config, RisesConfig
from wavetracker.rises import join_rises

DT = 0.164
P1 = np.array([1.0, 2.0, 4.0, 8.0])  # electrode powers of one fish
P2 = P1[::-1]  # another fish elsewhere on the grid


def _piece(t0, t1, ident, f0, pattern=P1, rise=None, seed=0, sal=0.9):
    """Detections every frame from t0 to t1 at f0, optionally with a rise
    (size, tau) decaying from t0, and a slightly noisy electrode pattern."""
    rng = np.random.default_rng(seed)
    t = np.arange(np.ceil(t0 / DT), np.floor(t1 / DT) + 1) * DT
    f = np.full(len(t), f0) + rng.normal(0, 0.05, len(t))
    if rise is not None:
        f += rise[0] * np.exp(-(t - t0) / rise[1])
    sign = np.asarray(pattern, float) * 10 ** (rng.normal(0, 0.1, (len(t), 4)) / 10)
    return t, f, np.full(len(t), float(ident)), sign, np.full(len(t), sal)


def _assemble(*pieces, with_salience=False):
    t = np.concatenate([p[0] for p in pieces])
    order = np.argsort(t, kind="stable")
    times = np.arange(0, t.max() + 1, DT)
    idx = np.round(t[order] / DT).astype(np.int64)
    out = (
        np.concatenate([p[1] for p in pieces])[order],
        idx,
        np.concatenate([p[3] for p in pieces])[order],
        np.concatenate([p[2] for p in pieces])[order],
        times,
    )
    if with_salience:
        return (*out, np.concatenate([p[4] for p in pieces])[order])
    return out


ON = RisesConfig(enabled=True, min_salience=None)


def _n_ids(ident):
    return len(np.unique(ident[np.isfinite(ident)]))


def test_joins_the_fragments_of_a_rise():
    fund, idx, sign, ident, times = _assemble(
        _piece(0, 100, 0, 700.0),
        _piece(100.5, 200, 1, 701.0, rise=(12.0, 8.0)),
    )
    out = join_rises(fund, idx, sign, ident, times, ON)
    assert _n_ids(out) == 1
    assert np.isfinite(out).all()


def test_off_by_default_and_unchanged_when_off():
    assert Config.classic().rises.enabled is False
    fund, idx, sign, ident, times = _assemble(
        _piece(0, 100, 0, 700.0),
        _piece(100.5, 200, 1, 701.0, rise=(12.0, 8.0)),
    )
    out = join_rises(fund, idx, sign, ident, times, RisesConfig())
    np.testing.assert_array_equal(out, ident)


def test_keeps_a_fish_with_another_electrode_pattern_apart():
    fund, idx, sign, ident, times = _assemble(
        _piece(0, 100, 0, 700.0),
        _piece(100.5, 200, 1, 701.0, pattern=P2, rise=(12.0, 8.0)),
    )
    assert _n_ids(join_rises(fund, idx, sign, ident, times, ON)) == 2


@pytest.mark.parametrize(
    ("start", "f_after", "why"),
    [
        (100.5, 690.0, "downward jump"),
        (100.5, 760.0, "jump above max_jump"),
        (100.5, 700.5, "jump below min_jump"),
        (112.0, 708.0, "gap above max_gap"),
    ],
)
def test_only_upward_jumps_of_rise_size_across_short_gaps(start, f_after, why):
    fund, idx, sign, ident, times = _assemble(
        _piece(0, 100, 0, 700.0), _piece(start, 200, 1, f_after)
    )
    assert _n_ids(join_rises(fund, idx, sign, ident, times, ON)) == 2, why


def test_the_earlier_identity_must_stop():
    # the lower line goes on: the higher one is another fish
    fund, idx, sign, ident, times = _assemble(
        _piece(0, 150, 0, 700.0),
        _piece(100.5, 200, 1, 708.0, seed=1),
    )
    assert _n_ids(join_rises(fund, idx, sign, ident, times, ON)) == 2


def test_short_later_piece_is_not_joined():
    fund, idx, sign, ident, times = _assemble(
        _piece(0, 100, 0, 700.0), _piece(100.5, 101.0, 1, 708.0)
    )
    assert _n_ids(join_rises(fund, idx, sign, ident, times, ON)) == 2


def test_joins_identities_that_alternate_at_rises():
    # the linker left a fish on two identities that take turns at its rises
    fund, idx, sign, ident, times = _assemble(
        _piece(0, 100, 0, 700.0),
        _piece(100.5, 200, 1, 701.0, rise=(10.0, 8.0), seed=1),
        _piece(200.5, 300, 0, 702.0, rise=(10.0, 8.0), seed=2),
        _piece(300.5, 400, 1, 703.0, rise=(10.0, 8.0), seed=3),
    )
    assert _n_ids(join_rises(fund, idx, sign, ident, times, ON)) == 1


def test_joins_across_a_ghost_tail_and_drops_it():
    # the detector reports the old frequency for 3 s after the onset
    fund, idx, sign, ident, times = _assemble(
        _piece(0, 103, 0, 700.0),
        _piece(100.2, 200, 1, 701.0, rise=(12.0, 8.0), seed=1),
    )
    out = join_rises(fund, idx, sign, ident, times, ON)
    assert _n_ids(out) == 1
    tail = (ident == 0) & (times[idx] >= 100.2)
    assert tail.sum() > 10
    assert np.isnan(out[tail]).all()  # one fish, one frequency
    assert np.isfinite(out[~tail]).all()


def test_identities_that_share_frames_are_not_joined():
    fund, idx, sign, ident, times = _assemble(
        _piece(0, 100, 0, 700.0),
        _piece(100.5, 200, 1, 701.0, rise=(12.0, 8.0)),
        _piece(150, 200, 0, 690.0, seed=4),  # the first identity again, below
    )
    assert _n_ids(join_rises(fund, idx, sign, ident, times, ON)) == 2


def test_track_results_joins_rises_when_enabled():
    from wavetracker.pipeline import track_results
    from wavetracker.results import Results

    fund, idx, sign, _, times = _assemble(
        _piece(0, 100, 0, 700.0),
        _piece(100.5, 200, 1, 701.0, rise=(12.0, 8.0)),
    )

    def run(rises):
        res = Results(
            fund_v=fund.copy(),
            idx_v=idx.copy(),
            sign_v=sign.copy(),
            ident_v=np.full(len(fund), np.nan),
            times=times,
            meta={},
        )
        cfg = Config.from_dict(
            {
                "detector": {"method": "classic"},
                "linking": {"enabled": False},
                "stitching": {"enabled": False},
            }
            | ({"rises": {"enabled": True, "min_salience": None}} if rises else {})
        )
        track_results(res, cfg)
        return res.ident_v

    assert _n_ids(run(False)) == 2
    assert _n_ids(run(True)) == 1


def test_faint_fish_are_not_joined_by_their_noisy_patterns():
    cfg = RisesConfig(enabled=True, min_salience=0.4)
    for sal, n in ((0.9, 1), (0.2, 2)):
        fund, idx, sign, ident, times, sal_v = _assemble(
            _piece(0, 100, 0, 700.0, sal=sal),
            _piece(100.5, 200, 1, 701.0, rise=(12.0, 8.0), sal=sal),
            with_salience=True,
        )
        out = join_rises(fund, idx, sign, ident, times, cfg, sal_v=sal_v)
        assert _n_ids(out) == n


def test_salience_check_needs_salience():
    fund, idx, sign, ident, times = _assemble(
        _piece(0, 100, 0, 700.0), _piece(100.5, 200, 1, 701.0, rise=(12.0, 8.0))
    )
    with pytest.raises(ValueError, match="salience"):
        join_rises(fund, idx, sign, ident, times, RisesConfig(enabled=True))
