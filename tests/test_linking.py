import numpy as np
import pytest

from wavetracker import linking
from wavetracker.config import LinkingConfig
from wavetracker.linking import FEATURES, LinkModel, link

DT = 0.164


def _fragment(t0, t1, f, ident, n_ch=4, pattern=None, rise=None):
    t = np.arange(t0, t1, DT)
    freq = np.full(len(t), f, dtype=float)
    if rise is not None:  # (size, tau): decaying excursion at the start
        freq += rise[0] * np.exp(-(t - t0) / rise[1])
    pattern = np.ones(n_ch) if pattern is None else np.asarray(pattern, float)
    return t, freq, np.full(len(t), ident, dtype=float), np.tile(pattern, (len(t), 1))


def _assemble(*frags):
    t = np.concatenate([f[0] for f in frags])
    order = np.argsort(t, kind="stable")
    times = np.arange(0, t.max() + 1, DT)
    idx = np.round(t[order] / DT).astype(int)
    sign = np.concatenate([f[3] for f in frags])[order]
    return (
        np.concatenate([f[1] for f in frags])[order],
        idx,
        sign,
        np.sqrt(sign).astype(complex),
        np.concatenate([f[2] for f in frags])[order],
        times,
    )


def _rule_model(**weights):
    """A linear 'model': logit = bias + sum(w * feature), on raw features."""
    n = len(FEATURES)
    w = np.zeros((n, 1))
    bias = weights.pop("bias", 0.0)
    for k, v in weights.items():
        w[FEATURES.index(k), 0] = v
    return LinkModel(
        mean=np.zeros(n),
        std=np.ones(n),
        weights=[w],
        biases=[np.array([bias])],
        features=list(FEATURES),
    )


# same fish if the baselines agree within ~1 Hz and the gap is short
RULE = _rule_model(bias=8.0, abs_dbase=-4.0, gap=-0.05)


def test_joins_close_continuation():
    fund, idx, sign, cplx, ident, times = _assemble(
        _fragment(0, 100, 700.0, 0), _fragment(112, 300, 700.3, 1)
    )
    out = link(fund, idx, sign, cplx, ident, times, LinkingConfig(), RULE)
    assert len(np.unique(out)) == 1


def test_keeps_different_fish_apart():
    fund, idx, sign, cplx, ident, times = _assemble(
        _fragment(0, 100, 700.0, 0), _fragment(112, 300, 706.0, 1)
    )
    out = link(fund, idx, sign, cplx, ident, times, LinkingConfig(), RULE)
    assert len(np.unique(out)) == 2


def test_best_pair_first_and_each_end_once():
    a = _fragment(0, 100, 700.0, 0)
    good = _fragment(110, 200, 700.1, 1)
    bad = _fragment(105, 200, 701.0, 2)  # overlaps `good`: cannot be the same fish
    fund, idx, sign, cplx, ident, times = _assemble(a, good, bad)
    out = link(fund, idx, sign, cplx, ident, times, LinkingConfig(), RULE)
    assert out[ident == 1][0] == out[ident == 0][0]
    assert out[ident == 2][0] != out[ident == 0][0]


def test_identities_never_share_many_frames():
    # two parallel fragments at the same frequency (two fish within 0.2 Hz):
    # the rule says join, the frame conflict forbids it
    fund, idx, sign, cplx, ident, times = _assemble(
        _fragment(0, 100, 700.0, 0), _fragment(20, 120, 700.2, 1)
    )
    out = link(fund, idx, sign, cplx, ident, times, LinkingConfig(), RULE)
    assert len(np.unique(out)) == 2


def test_contained_alternating_fragment_is_absorbed():
    # one fish whose detections alternate between two identities for a while
    a = _fragment(0, 200, 700.0, 0)
    b = _fragment(50, 80, 700.05, 1)
    keep = np.ones(len(a[0]), bool)
    in_b = (a[0] >= 50) & (a[0] < 80)
    keep[np.flatnonzero(in_b)[::2]] = False  # A gives up every other frame
    a = tuple(x[keep] for x in a)
    b_keep = np.isin(np.round(b[0] / DT), np.round(a[0] / DT), invert=True)
    b = tuple(x[b_keep] for x in b)
    fund, idx, sign, cplx, ident, times = _assemble(a, b)
    model = _rule_model(bias=4.0, abs_dmed=-4.0, df_overlap=-4.0)
    out = link(fund, idx, sign, cplx, ident, times, LinkingConfig(), model)
    assert len(np.unique(out)) == 1


def test_features_are_finite_and_named():
    fund, idx, sign, cplx, ident, times = _assemble(
        _fragment(0, 100, 700.0, 0, pattern=[1, 0.5, 0.1, 0.01]),
        _fragment(101, 200, 700.3, 1, rise=(12, 5), pattern=[1, 0.4, 0.1, 0.02]),
        _fragment(60, 160, 703.0, 2, pattern=[0.01, 0.1, 0.5, 1]),
    )
    cfg = LinkingConfig()
    fr = linking.fragments(fund, idx, sign, cplx, ident, times, cfg)
    pairs = linking.candidates(fr, cfg)
    valid = np.isfinite(ident)
    dets = linking._Detections(times[idx[valid]], fund[valid], ident[valid])
    X = linking.features(fr, pairs, dets, cfg)
    assert X.shape == (len(pairs), len(FEATURES))
    assert np.isfinite(X).all()
    col = {k: i for i, k in enumerate(FEATURES)}
    p01 = np.flatnonzero((pairs[:, 0] == 0) & (pairs[:, 1] == 1))
    assert len(p01) == 1
    row = X[p01[0]]
    assert row[col["pattern_rms"]] < 1.5  # similar patterns
    assert row[col["df_edge"]] > 5  # the rise onset
    assert abs(row[col["dbase"]]) < 1.5  # ... but the same baseline


def test_model_roundtrip(tmp_path):
    m = _rule_model(bias=1.0, gap=-0.5)
    m.save(tmp_path / "m.npz")
    m2 = LinkModel.load(tmp_path / "m.npz")
    X = np.random.default_rng(0).normal(size=(5, len(FEATURES)))
    assert np.allclose(m.prob(X), m2.prob(X))


def test_shipped_model_joins_simple_break():
    if not linking.DEFAULT_MODEL.exists():
        pytest.skip("no shipped linker model")
    fund, idx, sign, cplx, ident, times = _assemble(
        _fragment(0, 100, 700.0, 0, pattern=[1, 0.5, 0.2, 0.1]),
        _fragment(115, 300, 700.2, 1, pattern=[1, 0.5, 0.2, 0.1]),
        _fragment(0, 300, 730.0, 2, pattern=[0.1, 0.2, 0.5, 1]),
    )
    out = link(fund, idx, sign, cplx, ident, times)
    assert out[ident == 0][0] == out[ident == 1][0]
    assert out[ident == 2][0] != out[ident == 0][0]


def test_pipeline_uses_linker_when_enabled(tmp_path):
    from wavetracker.config import Config
    from wavetracker.pipeline import track_results
    from wavetracker.results import Results

    fund, idx, sign, cplx, _ident, times = _assemble(
        _fragment(0, 100, 700.0, 0), _fragment(112, 300, 700.3, 1)
    )
    model = tmp_path / "rule.npz"
    RULE.save(model)
    res = Results(fund, idx, sign, np.full(len(fund), np.nan), times, cplx_v=cplx)
    cfg = Config.from_dict({"linking": {"enabled": True, "model": str(model)}})
    cfg.tracking.max_dt = 5.0  # the 12 s gap breaks the track
    track_results(res, cfg)
    assert len(np.unique(res.ident_v[np.isfinite(res.ident_v)])) == 1
    assert res.meta["linking_config"]["enabled"] is True
    cfg.linking.enabled = False
    cfg.stitching.enabled = False
    track_results(res, cfg)
    assert len(np.unique(res.ident_v[np.isfinite(res.ident_v)])) == 2


def test_parallel_fish_are_not_joined():
    # two fish 3.5 Hz apart at the same time; the weaker is detected in
    # pieces, on every other frame, the stronger on the frames between: no
    # frame is shared, but a fish is never at two frequencies at once
    strong = _fragment(0, 200, 128.5, 0)
    pieces = [
        _fragment(t, t + 6, 125.0, 1 + i) for i, t in enumerate(range(10, 190, 30))
    ]
    pieces = [tuple(x[::2] for x in p) for p in pieces]
    occupied = np.concatenate([np.round(p[0] / DT).astype(int) for p in pieces])
    keep = ~np.isin(np.round(strong[0] / DT).astype(int), occupied)
    strong = tuple(x[keep] for x in strong)
    fund, idx, sign, cplx, ident, times = _assemble(strong, *pieces)
    model = _rule_model(bias=6.0, gap=-0.01)  # would join anything in the window
    out = link(fund, idx, sign, cplx, ident, times, LinkingConfig(), model)
    assert np.all(out[ident == 0] != out[ident == 1][0])
    # without the check they would be joined
    cfg = LinkingConfig(max_parallel=1.1)
    out = link(fund, idx, sign, cplx, ident, times, cfg, model)
    assert out[ident == 0][0] == out[ident == 1][0]
