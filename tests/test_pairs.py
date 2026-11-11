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


def test_the_weaker_detection_of_a_split_peak_is_its_partner():
    from wavetracker.pairs import split_partners

    fund = np.array([600.0, 600.4, 610.0, 600.0, 601.5])
    idx = np.array([0, 0, 0, 1, 1])
    sal = np.array([0.9, 0.4, 0.8, 0.9, 0.5])
    np.testing.assert_array_equal(
        split_partners(fund, idx, sal, 1.2), [False, True, False, False, False]
    )
    # equal salience: the later of the two in the arrays
    sal[1] = 0.9
    assert split_partners(fund, idx, sal, 1.2)[1]


def _gap_scene(df=0.05):
    """A at 600 Hz throughout; B at 600.5 Hz, its own detections missing from
    40 to 60 s, where split partners of A's peak run along B's path (off it
    by `df`)."""
    times = np.arange(0.0, 100.0, DT)
    n = len(times)
    t_b = (times < 40) | (times > 60)
    n_p = (~t_b).sum()
    fund = np.r_[np.full(n, 600.0), np.full(t_b.sum(), 600.5), np.full(n_p, 600.5 - df)]
    idx = np.r_[np.arange(n), np.flatnonzero(t_b), np.flatnonzero(~t_b)]
    ident = np.r_[np.zeros(n), np.ones(t_b.sum()), np.full(n_p, np.nan)]
    partner = np.r_[np.zeros(n + t_b.sum(), bool), np.ones(n_p, bool)]
    o = np.lexsort((fund, idx))
    return fund[o], idx[o], ident[o], partner[o], times


def test_partners_fill_the_gap_of_the_identity_whose_path_they_are_on():
    from wavetracker.pairs import fill_partners

    fund, idx, ident, partner, times = _gap_scene()
    out = fill_partners(fund, idx, ident, partner, times, ON)
    np.testing.assert_array_equal(out[~partner], ident[~partner])
    assert (out[partner] == 1).all()


def test_partners_off_the_path_or_after_a_gap_stay_unassigned():
    from wavetracker.pairs import fill_partners

    fund, idx, ident, partner, times = _gap_scene(df=0.6)
    out = fill_partners(fund, idx, ident, partner, times, ON)
    assert not np.isfinite(out[partner]).any()
    # partners only from 45 s on: more than fill_gap after B's last detection
    fund, idx, ident, partner, times = _gap_scene()
    late = partner & (times[idx] < 45)
    keep = ~late
    out = fill_partners(fund[keep], idx[keep], ident[keep], partner[keep], times, ON)
    t = times[idx[keep]]
    p = partner[keep]
    # reached from B's next detection at 60 s, backward
    assert (out[p] == 1).all()
    out = fill_partners(
        fund[keep],
        idx[keep],
        ident[keep],
        partner[keep],
        times,
        PairsConfig(fill_gap=1.0),
    )
    assert (out[p] == 1).all()  # frame by frame, each step is short
    assert t[p].min() >= 45


def test_a_partner_does_not_take_an_identity_present_in_its_frame():
    from wavetracker.pairs import fill_partners

    fund, idx, ident, partner, times = _gap_scene()
    ident[partner] = np.nan
    ident[~partner & (ident == 0)] = 1.0  # B's identity holds A's peak too
    ident[~partner & (fund == 600.5)] = 1.0
    out = fill_partners(fund, idx, ident, partner, times, ON)
    assert not np.isfinite(out[partner]).any()


def test_split_partners_do_not_change_the_other_identities():
    from wavetracker.pipeline import track_results
    from wavetracker.results import Results

    s = _scene(*[lambda t, i=i: _flat_pair(t)[i] for i in (0, 1)])
    sal = np.full(len(s["fund"]), 0.9)
    t = s["times"][s["idx"]]
    # in the shared peak, B's detection is a weaker split partner
    in_peak = (s["fish"] == 1) & (np.abs(s["fund"] - 600.0) < 1.2)
    sal[in_peak] = 0.5

    def run(keep, partners):
        res = Results(
            fund_v=s["fund"][keep],
            idx_v=s["idx"][keep],
            sign_v=s["sign"][keep],
            ident_v=np.full(keep.sum(), np.nan),
            times=s["times"],
            meta={},
            sal_v=sal[keep],
        )
        cfg = Config.from_dict(
            {
                "pairs": {"enabled": True, "resolve": False, "partners": partners},
                "linking": {"enabled": False},
            }
        )
        track_results(res, cfg)
        return res.ident_v

    alone = run(~in_peak, "track")
    filled = run(np.ones(len(sal), bool), "fill")
    np.testing.assert_array_equal(filled[~in_peak], alone)
    assert np.isfinite(filled[in_peak]).mean() > 0.9
    assert t[in_peak].min() > 40


def test_partners_must_be_known():
    with pytest.raises(ValueError, match="partners"):
        PairsConfig(partners="somewhere")
