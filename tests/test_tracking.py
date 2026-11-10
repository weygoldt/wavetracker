import numpy as np
import pytest

from wavetracker.config import Config, TrackingConfig
from wavetracker.pipeline import detect, track_results
from wavetracker.results import Results
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


def test_classic_detections_have_a_salience(detections):
    """The classic detector scores its detections like the learned one, from
    the fundamental's level over the local noise: the weaker fish scores lower."""
    r = detections
    assert r.sal_v is not None and r.sal_v.shape == r.fund_v.shape
    assert np.all((r.sal_v >= 0) & (r.sal_v <= 1))
    weak = np.abs(r.fund_v - 910) < 20  # amplitude 0.5, the others 1
    strong = np.abs(r.fund_v - 520) < 20
    assert np.median(r.sal_v[weak]) < np.median(r.sal_v[strong])


@pytest.mark.parametrize("bands", [True, False])
def test_classic_salience_is_the_level_over_the_noise(bands):
    """0.5 one high threshold above the floor, 0.75 two, 0 at or below it."""
    from wavetracker.harmonics import Detections
    from wavetracker.pipeline import classic_salience

    log_np = np.zeros((3, 400), dtype=np.float32)  # floor at 0 dB
    log_np[0, 50], log_np[1, 250], log_np[2, 120] = 12.0, 24.0, -3.0
    det = Detections(np.array([0, 1, 2]), np.array([50, 250, 120]), np.zeros(3))
    high = np.full((3, 4), 12.0) if bands else np.full(3, 12.0)
    sal = classic_salience(log_np, det, high, 100, slice(0, 400))
    np.testing.assert_allclose(sal, [0.5, 0.75, 0.0], atol=1e-6)


@pytest.mark.parametrize("v6_compat", [False, True])
def test_start_threshold_that_every_detection_passes_changes_nothing(
    detections, v6_compat
):
    """start_salience: None is the tracker as before (the equivalence tests
    above run with it); a threshold that every temporary identity passes must
    not change a single assignment either, also with spurious detections."""
    r = detections
    rng = np.random.default_rng(3)
    n = len(r.fund_v) // 3
    idx = np.concatenate([r.idx_v, rng.integers(0, r.idx_v.max(), n)])
    fund = np.concatenate(
        [r.fund_v, r.fund_v[rng.integers(0, len(r.fund_v), n)] + rng.normal(0, 2, n)]
    )
    sign = np.concatenate([r.sign_v, rng.random((n, r.sign_v.shape[1]))])
    sal = rng.random(len(fund))
    order = np.argsort(idx, kind="stable")
    idx, fund, sign, sal = idx[order], fund[order], sign[order], sal[order]
    off = track(
        fund, idx, sign, r.times, TrackingConfig(), 400, 1200, v6_compat=v6_compat
    )
    cfg = TrackingConfig(start_salience=0.0, start_evidence=0.0)
    on = track(fund, idx, sign, r.times, cfg, 400, 1200, v6_compat=v6_compat, sal_v=sal)
    assert _same(on, off)


def _salience_scene():
    """120 s at 0.2 s per frame: a strong fish at 600 Hz, confident in every
    frame; a faint fish at 700 Hz from 30 s on, detected in 70 % of the frames
    but confident only in a burst of three at 40-42 s; and doubtful clutter
    scattered over 800-1000 Hz. Returns detections, times and labels."""
    rng = np.random.default_rng(1)
    dt, n_frames = 0.2, 600
    times = np.arange(n_frames) * dt
    patterns = {0: np.array([1.0, 0.5, 0.2, 0.1]), 1: np.array([0.1, 0.3, 1.0, 0.4])}
    rows = []  # frame, freq, salience, label (0 strong, 1 faint, 2 clutter)
    for k in range(n_frames):
        rows.append((k, 600.0 + 0.5 * np.sin(k / 50), rng.uniform(0.8, 1.0), 0))
    for k in range(150, n_frames):
        confident = k in (200, 205, 210)
        if confident or rng.random() < 0.7:
            sal = rng.uniform(0.6, 0.8) if confident else rng.uniform(0.1, 0.4)
            rows.append((k, 700.0 + 0.02 * rng.normal(), sal, 1))
    for _ in range(400):
        rows.append(
            (rng.integers(n_frames), rng.uniform(800, 1000), rng.uniform(0.05, 0.4), 2)
        )
    rows.sort(key=lambda r: r[0])
    idx = np.array([r[0] for r in rows])
    fund = np.array([r[1] for r in rows])
    sal = np.array([r[2] for r in rows])
    label = np.array([r[3] for r in rows])
    sign = np.empty((len(rows), 4))
    for i, lab in enumerate(label):
        sign[i] = (
            rng.random(4) if lab == 2 else patterns[lab] * rng.uniform(0.9, 1.1, 4)
        )
    return fund, idx, sign, sal, times, label


def test_only_confident_detections_start_identities():
    """ByteTrack-style association: with a start threshold, doubtful
    detections join identities but do not start them. The faint fish is
    followed once its three confident detections start it; the clutter, which
    links into short tracks of its own, makes no identities."""
    fund, idx, sign, sal, times, label = _salience_scene()

    # one low threshold: all detections are alike, the clutter makes identities
    ident = track(fund, idx, sign, times, TrackingConfig(), sal_v=sal)
    assert len(np.unique(ident[(label == 2) & np.isfinite(ident)])) >= 5

    cfg = TrackingConfig(start_salience=0.5, start_evidence=0.6)  # 3 detections
    ident = track(fund, idx, sign, times, cfg, sal_v=sal)
    assert np.all(np.isnan(ident[label == 2]))
    # (a track's first detection is never linked, as in the original)
    strong = ident[label == 0][1:]
    assert np.all(strong == strong[0])
    faint = ident[(label == 1) & (idx >= 200)]  # from its first confident one
    assert np.all(faint == faint[0]) and faint[0] != strong[0]
    # it is also followed before the confident burst, within one window
    assert np.isfinite(ident[(label == 1) & (idx < 200)]).mean() > 0.5


def test_start_threshold_needs_a_salience():
    fund, idx, sign, sal, times, _ = _salience_scene()
    cfg = Config()
    cfg.tracking.start_salience = 0.5
    res = Results(fund, idx, sign, np.full(len(fund), np.nan), times)
    with pytest.raises(ValueError, match="sal_v"):
        track_results(res, cfg)
    res.sal_v = sal
    track_results(res, cfg)
    assert np.all(np.isnan(res.ident_v[sal < 0.5][fund[sal < 0.5] > 800]))
