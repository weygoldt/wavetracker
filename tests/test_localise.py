import numpy as np
import pytest
from scipy.optimize import approx_fprime

from wavetracker.config import PositionMergingConfig
from wavetracker.position.electrodes import ElectrodeTrack, channel_map
from wavetracker.position.localise import (
    Observations,
    Problem,
    fit,
    jackknife,
    segment_check,
    select_censored,
    source_model,
)
from wavetracker.synthetic import StationaryFish, boat_survey, simulate_survey


def test_channel_map():
    plus, minus = channel_map(3, reference=2)
    assert plus.tolist() == [0, 1] and minus.tolist() == [2, 2]
    plus, minus = channel_map(3, reference=0)
    assert plus.tolist() == [1, 2] and minus.tolist() == [0, 0]
    plus, minus = channel_map(4)
    assert plus.tolist() == [0, 1, 2, 3] and (minus == -1).all()
    plus, minus = channel_map(4, channel_pairs=[[0, 1], [2, 3]])
    assert plus.tolist() == [0, 2] and minus.tolist() == [1, 3]
    with pytest.raises(ValueError):
        channel_map(3, channel_pairs=[[0, 0]])


def test_electrode_track_interpolation_and_files(tmp_path):
    t = np.array([0.0, 1.0, 2.0, 3.0])
    pos = np.zeros((4, 3, 3))
    pos[:, :, 0] = t[:, None] + np.arange(3)
    pos[2] = np.nan  # geometry missing at t = 2
    track = ElectrodeTrack.from_arrays(t, pos, reference=2)
    p = track.at(np.array([0.5, 1.5, 2.5, 3.5]))
    assert p[0, 0, 0] == pytest.approx(0.5)
    assert track.valid(np.array([0.5, 1.5, 2.5, 3.5])).tolist() == [
        True,
        False,
        False,
        False,
    ]
    v = np.array([[1.0, 2.0, 5.0]])
    assert track.channels.values(v).tolist() == [[-4.0, -3.0]]
    mid = track.channels.midpoints(pos[:1])
    assert mid[0, 0, 0] == pytest.approx(1.0)  # (0 + 2) / 2

    np.savez(tmp_path / "e.npz", time=t, positions=pos)
    cols = {"time": t}
    for e in range(3):
        for a, name in enumerate("xyz"):
            cols[f"{name}{e}"] = pos[:, e, a]
    import pandas as pd

    pd.DataFrame(cols).to_csv(tmp_path / "e.csv", index=False)
    for f in ("e.npz", "e.csv"):
        loaded = ElectrodeTrack.load(tmp_path / f, reference=2)
        np.testing.assert_array_equal(loaded.positions, pos)
        assert loaded.n_channels == 2


@pytest.fixture(scope="module")
def survey():
    rng = np.random.default_rng(5)
    t, pos = boat_survey(length=6.0, width=3.0, lane_spacing=1.0)
    fish = [StationaryFish(600.0, 3.2, 1.6, 0.4, 0.8)]
    sd = simulate_survey(fish, t, pos, rng=rng)
    track = ElectrodeTrack.from_arrays(sd.time, sd.positions, reference=2)
    return sd, track, fish[0]


@pytest.fixture
def cfg():
    return PositionMergingConfig(
        reference=2,
        water_depth=1.2,
        start_grid=4,
        start_headings=4,
        start_depths=[0.4],
        n_refine=4,
    )


def observations(sd, track, cfg, with_sign=True):
    r = sd.results
    m = sd.fish == 0
    frames = r.idx_v[m]
    cens = np.setdiff1d(np.arange(len(r.times)), frames)[:: cfg.censor_stride]
    pd_, pc = track.at(r.times[frames]), track.at(r.times[cens])
    keep = select_censored(pd_, pc, track.channels, 3 * cfg.detection_range)
    return Observations(
        t_det=r.times[frames],
        amp=np.sqrt(r.sign_v[m]),
        cplx=r.cplx_v[m] if with_sign else None,
        noise=np.full(2, 0.03),
        pos_det=pd_,
        t_cens=r.times[cens][keep],
        pos_cens=pc[keep],
        channels=track.channels,
        seg=(r.times[frames] > r.times[frames].mean()).astype(int),
    )


@pytest.mark.parametrize("kind", ["monopoles", "dipole"])
def test_jacobian_matches_finite_differences(survey, cfg, kind):
    sd, track, _ = survey
    cfg.model = kind
    pb = Problem(observations(sd, track, cfg), source_model(cfg), cfg)
    # a point outside the detection range activates the bound term
    for q in ([3.0, 1.0, 0.5, 0.3, 0.2], [3.0, 8.0, 0.5, 2.0, 1.0]):
        q = np.array(q)
        jac = pb.jac(q)
        sel = np.r_[np.arange(len(jac) - 1)[::37], len(jac) - 1]
        num = np.array(
            [approx_fprime(q, lambda x, i=i: pb.resid(x)[i], 1e-7) for i in sel]
        )
        np.testing.assert_allclose(jac[sel], num, rtol=1e-3, atol=1e-4)


def test_fit_recovers_position(survey, cfg):
    sd, track, fish = survey
    obs = observations(sd, track, cfg)
    model = source_model(cfg)
    res = fit(obs, model, cfg)
    assert np.hypot(res.x - fish.x, res.y - fish.y) < 0.1
    assert abs(res.depth - fish.depth) < 0.1
    assert not res.ambiguous and res.r_fit > 0.9
    jackknife(obs, model, cfg, res)
    assert res.jk_blocks >= 6 and 0 < res.se_major < 0.1
    check = segment_check(obs, model, cfg, res.params)
    assert set(check) == {0, 1}
    assert all(r < 0.5 and d < cfg.detection_range for r, d in check.values())
    # a fish far away explains neither segment
    far = res.params.copy()
    far[:2] += [0.0, 5.0]
    assert all(
        d > cfg.detection_range for _, d in segment_check(obs, model, cfg, far).values()
    )


def test_fit_without_relative_signs(survey, cfg):
    sd, track, fish = survey
    res = fit(observations(sd, track, cfg, with_sign=False), source_model(cfg), cfg)
    # amplitudes alone leave the side of the boat path ambiguous in general;
    # here the censored frames of the neighbouring lanes still resolve it
    assert np.hypot(res.x - fish.x, res.y - fish.y) < 0.3
