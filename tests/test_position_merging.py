import numpy as np
import pytest

from wavetracker.config import PositionMergingConfig
from wavetracker.position.electrodes import ElectrodeTrack
from wavetracker.position.merging import (
    group_by_frequency,
    load_fish,
    make_segments,
    merge_by_position,
)
from wavetracker.position.noise import DetectionNoiseFloor, SpectrumNoiseFloor
from wavetracker.results import Results
from wavetracker.synthetic import StationaryFish, boat_survey, simulate_survey


def _results(spec, step=0.2):
    """Detections from (ident, t0, t1, freq) tuples, one per frame."""
    rows = []
    for ident, t0, t1, f in spec:
        for t in np.arange(t0, t1 + 1e-9, step):
            rows.append((round(t / step), f, ident))
    rows.sort()
    n = max(r[0] for r in rows) + 1
    return Results(
        fund_v=np.array([r[1] for r in rows], float),
        idx_v=np.array([r[0] for r in rows]),
        sign_v=np.ones((len(rows), 2)),
        ident_v=np.array([r[2] for r in rows], float),
        times=np.arange(n) * step,
    )


def test_segments_and_frequency_grouping():
    cfg = PositionMergingConfig()
    r = _results(
        [
            (1, 0.0, 10.0, 600.0),  # identity 1 continues after a 10 s gap ...
            (1, 20.0, 30.0, 603.0),  # ... on another fish
            (2, 40.0, 50.0, 600.4),  # same frequency as the first, later
            (3, 5.0, 15.0, 600.2),  # overlaps the first: another fish
            (4, 60.0, 61.0, 700.0),  # clutter (too short)
        ]
    )
    segs, seg_of = make_segments(r, np.ones(len(r.fund_v), bool), cfg)
    assert len(segs) == 5 and segs.clutter.tolist() == [False] * 4 + [True]
    assert (seg_of >= 0).all()
    groups = group_by_frequency(segs[~segs.clutter], 1.0, 1.0)
    as_sets = sorted(sorted(g) for g in groups)
    # 600.0 and 600.4 join; 600.2 overlaps in time; 603 is apart
    assert as_sets == [[0, 2], [1], [3]]


def test_noise_floor_from_detections_and_recording(tmp_path):
    from audioio import write_audio

    rng = np.random.default_rng(0)
    f = rng.uniform(500, 700, 5000)
    p = rng.exponential(1.0, (5000, 2)) * [1.0, 4.0]
    floor = DetectionNoiseFloor(f, p, quantile=50.0).amplitude(600.0)
    np.testing.assert_allclose(floor**2, np.log(2) * np.array([1.0, 4.0]), rtol=0.2)

    rate, sd = 8000.0, np.array([0.01, 0.03])
    x = rng.standard_normal((int(30 * rate), 2)) * sd
    write_audio(str(tmp_path / "noise.wav"), x, rate)
    nfft = 4096
    times = np.arange(1.0, 29.0, 0.25)
    nf = SpectrumNoiseFloor.from_recording(
        tmp_path / "noise.wav", times, nfft, quantile=50.0, max_freq=2000.0
    )
    # white noise: mean PSD 2 sd^2 / rate, exponentially distributed; the
    # minimum over 5 bins of the median lies somewhat below the median
    expect = np.sqrt(np.log(2) * 2 * sd**2 / rate)
    amp = nf.amplitude(1000.0)
    assert np.all(amp < 1.05 * expect) and np.all(amp > 0.6 * expect)
    nf.save(tmp_path / "floor.npz")
    np.testing.assert_array_equal(
        SpectrumNoiseFloor.load(tmp_path / "floor.npz").amplitude(1000.0), amp
    )


FISH = [
    StationaryFish(600.0, 2.0, 1.5, 0.4, 0.3),
    StationaryFish(603.0, 7.5, 4.5, 0.5, 2.0),  # 3 Hz from fish 0
    StationaryFish(700.0, 3.0, 4.2, 0.3, 1.0, drift=2.5),  # drifts across passes
    StationaryFish(800.0, 2.5, 2.5, 0.6, 4.0),
    StationaryFish(800.5, 8.0, 3.5, 0.4, 5.0),  # 0.5 Hz from fish 3
    StationaryFish(900.0, 5.0, 3.0, 0.3, 0.5, strength=0.7),
]


@pytest.fixture(scope="module")
def merged(tmp_path_factory):
    rng = np.random.default_rng(3)
    t, pos = boat_survey(length=10.0, width=6.0, lane_spacing=1.0)
    sd = simulate_survey(FISH, t, pos, n_clutter=20, rng=rng)
    r = sd.results
    # the tracker joined a pass of fish 0 with a later pass of fish 1
    tt = r.times[r.idx_v]
    a = np.unique(r.ident_v[sd.fish == 0])[0]
    b = next(
        i
        for i in np.unique(r.ident_v[sd.fish == 1])
        if tt[r.ident_v == i].min() > tt[r.ident_v == a].max() + 5
    )
    r.ident_v[r.ident_v == b] = a
    cfg = PositionMergingConfig(
        reference=2,
        water_depth=1.2,
        freq_tolerance=0.5,  # splits the drifting fish into several candidates
        start_grid=5,
        start_headings=4,
        start_depths=[0.4],
    )
    track = ElectrodeTrack.from_arrays(sd.time, sd.positions, reference=2)
    out = merge_by_position(r, track, cfg, n_jobs=4)
    folder = tmp_path_factory.mktemp("merged")
    out.save(folder)
    return sd, out, a, folder


@pytest.mark.slow
def test_segments_are_merged_into_the_right_fish(merged):
    sd, out, joined, _ = merged
    assert out.stats["merges"] >= 1
    assert out.stats["segments_split_off"] >= 1
    assert len(out.fish) == len(FISH)
    labels = []
    for k, fi in enumerate(FISH):
        lab = out.fish_v[sd.fish == k]
        lab = lab[np.isfinite(lab)]
        v, c = np.unique(lab, return_counts=True)
        assert len(v) == 1, f"fish {k} split into {dict(zip(v, c, strict=True))}"
        assert len(lab) > 0.8 * (sd.fish == k).sum()
        row = out.fish.iloc[int(v[0])]
        assert np.hypot(row.x - fi.x, row.y - fi.y) < 0.15
        assert abs(row.depth - fi.depth) < 0.1
        assert not row.ambiguous
        labels.append(int(v[0]))
    assert len(set(labels)) == len(FISH)  # no two fish merged
    # the identity joining fish 0 and fish 1 (3 Hz apart) is split again
    m = sd.results.ident_v == joined
    assert set(sd.fish[m]) == {0, 1}
    for k in (0, 1):
        assert set(out.fish_v[m & (sd.fish == k)]) == {labels[k]}
    # clutter stays unassigned
    assert np.isnan(out.fish_v[sd.fish == -1]).all()


@pytest.mark.slow
def test_outputs_are_written(merged):
    _, out, _, folder = merged
    fish_v, table = load_fish(folder)
    np.testing.assert_array_equal(fish_v, out.fish_v)
    assert {"freq", "x", "y", "depth", "se_major", "ambiguous", "r_fit"} <= set(
        table.columns
    )
    assert (folder / "fish_segments.csv").exists()
    assert (folder / "position_merging.json").exists()
    assert not (folder / "ident_v.npy").exists()  # identities are left alone
