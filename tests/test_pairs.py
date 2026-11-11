import numpy as np
import pytest

from wavetracker.config import Config, PairsConfig
from wavetracker.pairs import episodes, resolve_pairs

DT = 0.164
PA = np.array([1.0, 2.0, 4.0, 8.0, 4.0, 2.0])  # electrode powers of fish A
PB = np.array([8.0, 4.0, 2.0, 1.0, 2.0, 4.0])  # fish B, elsewhere on the grid
ON = PairsConfig(enabled=True)


def _scene(fa, fb, seed=0, peak_hz=1.2):
    """Two fish with frequencies fa(t), fb(t), detected in every frame; where
    they are closer than `peak_hz` both detections carry the spectra of the
    one peak they share (the sum of the two)."""
    rng = np.random.default_rng(seed)
    times = np.arange(0.0, 100.0, DT)
    n = len(times)
    f = np.stack([fa(times), fb(times)]) + rng.normal(0, 0.03, (2, n))
    noise = 10 ** (rng.normal(0, 0.1, (2, n, len(PA))) / 10)
    sign = np.stack([PA * noise[0], PB * noise[1]])
    shared = np.abs(f[0] - f[1]) < peak_hz
    sign[:, shared] = (PA + PB) * noise[0, shared]
    fish = np.repeat([[0], [1]], n, axis=1)
    idx = np.tile(np.arange(n), 2)
    order = np.lexsort((f.ravel(), idx))
    return {
        "fund": f.ravel()[order],
        "idx": idx[order],
        "sign": sign.reshape(-1, len(PA))[order],
        "fish": fish.ravel()[order],
        "times": times,
    }


def _flat_pair(t):
    """A at 600 Hz; B at 603 Hz, within 0.4 Hz of A from 42 to 58 s."""
    fb = np.interp(t, [0, 40, 42, 58, 60, 100], [603, 603, 600.4, 600.4, 603, 603])
    return np.full(len(t), 600.0), fb


def _crossing(t):
    """A climbs from 590 to 610 Hz through B at 600 Hz (crossing at 50 s)."""
    return 590.0 + 0.2 * t, np.full(len(t), 600.0)


def _resolve(s, ident, cfg=ON):
    return resolve_pairs(s["fund"], s["idx"], s["sign"], ident, s["times"], cfg)


def _purity(ident, fish):
    """Fraction of detections on the identity holding most of their fish."""
    ok = 0
    for k in np.unique(fish):
        v = ident[fish == k]
        _, c = np.unique(v[np.isfinite(v)], return_counts=True)
        ok += c.max() if len(c) else 0
    return ok / len(fish)


def test_off_by_default_and_unchanged_when_off():
    assert Config().pairs.enabled is False
    s = _scene(*[lambda t, i=i: _flat_pair(t)[i] for i in (0, 1)])
    ident = s["fish"].astype(float)
    ident[(s["times"][s["idx"]] > 50)] = 1 - s["fish"][s["times"][s["idx"]] > 50]
    out = _resolve(s, ident, PairsConfig())
    np.testing.assert_array_equal(out, ident)


def test_finds_the_episode_of_two_close_identities():
    s = _scene(*[lambda t, i=i: _flat_pair(t)[i] for i in (0, 1)])
    eps = episodes(s["fund"], s["idx"], s["fish"].astype(float), ON.max_df, 2.0 / DT)
    assert len(eps) == 1
    t0, t1 = s["times"][eps[0].first], s["times"][eps[0].last]
    assert 40 < t0 < 42 and 58 < t1 < 60


def test_swaps_back_two_fish_exchanged_in_a_shared_peak():
    s = _scene(*[lambda t, i=i: _flat_pair(t)[i] for i in (0, 1)])
    t = s["times"][s["idx"]]
    ident = np.where(t > 50, 1 - s["fish"], s["fish"]).astype(float)
    assert _purity(ident, s["fish"]) < 0.6
    out = _resolve(s, ident)
    assert _purity(out, s["fish"]) > 0.9
    # the fish before the episode keep their identities
    np.testing.assert_array_equal(out[t < 40], ident[t < 40])


def test_keeps_two_fish_that_left_on_their_own_identities():
    s = _scene(*[lambda t, i=i: _flat_pair(t)[i] for i in (0, 1)])
    ident = s["fish"].astype(float)
    np.testing.assert_array_equal(_resolve(s, ident), ident)


def test_crossing_fish_keep_their_identities_through_the_crossing():
    s = _scene(*[lambda t, i=i: _crossing(t)[i] for i in (0, 1)])
    # a tracker that kept the frequency order swapped them at the crossing
    lower = np.zeros(len(s["fund"]), bool)
    for f in np.unique(s["idx"]):
        m = np.flatnonzero(s["idx"] == f)
        lower[m[np.argmin(s["fund"][m])]] = True
    ident = np.where(lower, 0.0, 1.0)
    assert _purity(ident, s["fish"]) < 0.6
    out = _resolve(s, ident)
    assert _purity(out, s["fish"]) > 0.9


def test_order_untangles_an_episode_that_alternates_between_the_fish():
    s = _scene(*[lambda t, i=i: _flat_pair(t)[i] for i in (0, 1)])
    t = s["times"][s["idx"]]
    rng = np.random.default_rng(1)
    flip = np.repeat(rng.random(len(s["times"])) < 0.5, 1)[s["idx"]]
    inside = (t > 43) & (t < 57)
    ident = np.where(inside & flip, 1 - s["fish"], s["fish"]).astype(float)
    without = _resolve(s, ident)
    with_order = _resolve(s, ident, PairsConfig(enabled=True, order=True))
    assert _purity(without, s["fish"]) < 0.95
    assert _purity(with_order, s["fish"]) > 0.99


def test_no_decision_without_clear_patterns_on_both_sides():
    s = _scene(*[lambda t, i=i: _flat_pair(t)[i] for i in (0, 1)])
    t = s["times"][s["idx"]]
    ident = np.where(t > 50, 1 - s["fish"], s["fish"]).astype(float)
    ident[(s["fish"] == 1) & (t > 58)] = np.nan  # B is not followed after it
    out = _resolve(s, ident)
    np.testing.assert_array_equal(out, ident)


@pytest.mark.parametrize("stage", ["tracking", "linking"])
def test_pairs_in_the_pipeline(stage):
    from wavetracker.pipeline import track_results
    from wavetracker.results import Results

    s = _scene(*[lambda t, i=i: _flat_pair(t)[i] for i in (0, 1)])
    res = Results(
        fund_v=s["fund"],
        idx_v=s["idx"],
        sign_v=s["sign"],
        ident_v=np.full(len(s["fund"]), np.nan),
        times=s["times"],
        meta={},
    )
    cfg = Config.from_dict(
        {"pairs": {"enabled": True, "stage": stage}, "linking": {"enabled": False}}
    )
    track_results(res, cfg)
    assert res.meta["pairs_config"]["stage"] == stage
    assert np.isfinite(res.ident_v).mean() > 0.9


def test_stage_must_be_known():
    with pytest.raises(ValueError, match="stage"):
        PairsConfig(stage="somewhere")
