import json

import numpy as np
import pytest

from wavetracker.assocnet import link_evidence
from wavetracker.config import Config, LinkingConfig, RisesConfig
from wavetracker.linking import FEATURES, LinkModel, link
from wavetracker.results import REVIEW_FILE, Results
from wavetracker.review import (
    GroupModel,
    Item,
    ReviewModel,
    Trace,
    build_items,
    confidences,
    report,
)
from wavetracker.rises import join_rises

DT = 0.164
P1 = (1.0, 2.0, 4.0, 8.0)  # electrode powers of one fish
P2 = P1[::-1]  # another fish elsewhere on the grid


def _fragment(t0, t1, f, ident, pattern=P1, sal=0.9, rise=None, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(np.ceil(t0 / DT), np.floor(t1 / DT) + 1) * DT
    freq = np.full(len(t), f) + rng.normal(0, 0.02, len(t))
    if rise is not None:  # (size, tau): decaying excursion at the start
        freq += rise[0] * np.exp(-(t - t0) / rise[1])
    sign = np.tile(np.asarray(pattern, float), (len(t), 1))
    sign *= 10 ** (rng.normal(0, 0.05, sign.shape) / 10)
    return t, freq, np.full(len(t), float(ident)), sign, np.full(len(t), sal)


def _assemble(*frags):
    t = np.concatenate([f[0] for f in frags])
    order = np.argsort(t, kind="stable")
    times = np.arange(0, t.max() + 1, DT)
    idx = np.round(t[order] / DT).astype(np.int64)
    sign = np.concatenate([f[3] for f in frags])[order]
    return (
        np.concatenate([f[1] for f in frags])[order],
        idx,
        sign,
        np.sqrt(sign).astype(complex),
        np.concatenate([f[2] for f in frags])[order],
        np.concatenate([f[4] for f in frags])[order],
        times,
    )


def _rule_model(**weights):
    """A linear linker 'model' on raw features (as in test_linking)."""
    n = len(FEATURES)
    w = np.zeros((n, 1))
    bias = weights.pop("bias", 0.0)
    for k, v in weights.items():
        w[FEATURES.index(k), 0] = v
    return LinkModel(np.zeros(n), np.ones(n), [w], [np.array([bias])], list(FEATURES))


RULE = _rule_model(bias=8.0, abs_dbase=-4.0, gap=-0.05)


def _linked(*frags, cfg=None):
    """Fragments joined by the rule linker, with the trace."""
    fund, idx, sign, cplx, frag, sal, times = _assemble(*frags)
    trace = Trace(frag_v=frag.copy())
    ident = link(
        fund,
        idx,
        sign,
        cplx,
        frag,
        times,
        cfg or LinkingConfig(),
        RULE,
        trace=trace.link,
    )
    trace.linked_v = ident.copy()
    trace.joined_v = ident.copy()
    return fund, idx, sign, sal, ident, times, trace, frag


def _by_group(items, group):
    return [it for it in items if it.group == group]


def test_off_by_default():
    cfg = Config.classic()
    assert cfg.review.enabled is False
    assert cfg.review.model is None
    cfg = Config.from_dict({"review": {"enabled": True, "max_confidence": 0.9}})
    assert cfg.review.max_confidence == 0.9


def test_link_trace_does_not_change_the_result():
    frags = (_fragment(0, 100, 700.0, 0), _fragment(112, 300, 700.3, 1))
    fund, idx, sign, cplx, frag, _, times = _assemble(*frags)
    plain = link(fund, idx, sign, cplx, frag, times, LinkingConfig(), RULE)
    trace = {}
    traced = link(
        fund, idx, sign, cplx, frag, times, LinkingConfig(), RULE, trace=trace
    )
    np.testing.assert_array_equal(plain, traced)
    assert trace["accepted"].tolist() == [True]
    assert (trace["a"][0], trace["b"][0]) == (0.0, 1.0)
    assert trace["X"].shape == (1, len(FEATURES))
    assert trace["prob"][0] > 0.99


def test_join_item_has_the_two_fragments_as_sides():
    fund, idx, sign, sal, ident, times, trace, frag = _linked(
        _fragment(0, 100, 700.0, 0), _fragment(112, 300, 700.3, 1)
    )
    items = build_items(fund, idx, sign, sal, ident, times, trace, Config.classic())
    (join,) = _by_group(items, "join_linker")
    assert join.kind == "join" and join.action == "split"
    np.testing.assert_array_equal(np.sort(join.rows[0]), np.flatnonzero(frag == 0))
    np.testing.assert_array_equal(np.sort(join.rows[1]), np.flatnonzero(frag == 1))
    assert join.k == idx[frag == 1].min()
    assert join.fragments == (1.0,)
    assert join.t == pytest.approx(times[join.k])
    assert join.raw > 0.99  # the linker's probability
    assert join.x["logit_p"] > 5
    assert join.x["side_dmed"] == pytest.approx(0.3, abs=0.05)
    assert join.impact == min((frag == 0).sum(), (frag == 1).sum())
    (track,) = _by_group(items, "track")
    assert track.x["log_units"] == pytest.approx(np.log1p(2))
    assert track.x["min_join"] == pytest.approx(join.x["logit_p"])
    assert not _by_group(items, "missed_linker")


def test_a_join_cuts_a_chain_into_before_and_after():
    fund, idx, sign, sal, ident, times, trace, _frag = _linked(
        _fragment(0, 100, 700.0, 0),
        _fragment(110, 200, 700.1, 1),
        _fragment(210, 300, 700.2, 2),
    )
    items = _by_group(
        build_items(fund, idx, sign, sal, ident, times, trace, Config.classic()),
        "join_linker",
    )
    assert len(items) == 2
    for it in items:
        cut = times[it.k]
        a, b = it.rows
        assert len(a) + len(b) == len(fund)
        assert times[idx[a]].max() < cut <= times[idx[b]].min()
        np.testing.assert_array_equal(
            np.sort(b), np.flatnonzero(np.isin(_frag, it.fragments))
        )


def _every(frag, step, offset):
    """Every `step`-th detection of a fragment from `offset`."""
    return tuple(x[offset::step] for x in frag)


def test_a_split_names_the_fragments_of_the_later_part():
    # fragment 2 takes the frames fragment 0 misses for 10 s: the joins of
    # the identity are not a chain in time, and a cut at one frame would
    # move the wrong detections
    fund, idx, sign, sal, ident, times, trace, frag = _linked(
        _every(_fragment(0, 100, 700.0, 0), 2, 0),
        _fragment(101, 200, 700.1, 1),
        _every(_fragment(60, 70, 700.05, 2), 2, 1),
    )
    assert len(np.unique(ident)) == 1
    items = _by_group(
        build_items(fund, idx, sign, sal, ident, times, trace, Config.classic()),
        "join_linker",
    )
    assert len(items) == 2
    for it in items:
        a, b = it.rows
        np.testing.assert_array_equal(
            np.sort(b), np.flatnonzero(np.isin(frag, it.fragments))
        )
        assert not np.isin(frag[a], it.fragments).any()
    cut_in_time = [it for it in items if (idx[it.rows[0]] >= it.k).any()]
    assert cut_in_time  # the join of fragment 2: a time cut would also move 0


def test_rejected_candidate_is_a_missed_join():
    # fragment 2 overlaps fragment 1, which joins fragment 0 first: the pair
    # (2, 1) is probable but would share frames
    fund, idx, sign, sal, ident, times, trace, frag = _linked(
        _fragment(0, 100, 700.0, 0),
        _fragment(112, 300, 700.3, 1),
        _fragment(105, 250, 700.8, 2, pattern=P2),
    )
    assert ident[frag == 2][0] != ident[frag == 0][0]
    items = build_items(fund, idx, sign, sal, ident, times, trace, Config.classic())
    (missed,) = _by_group(items, "missed_linker")
    assert missed.kind == "missed_join" and missed.action == "merge"
    assert set(missed.ids) == {ident[frag == 0][0], ident[frag == 2][0]}
    assert missed.raw < 0.1  # 1 - the linker's probability
    assert missed.x["side_shared"] > 0.5
    cfg = Config.from_dict(
        {"detector": {"method": "classic"}, "review": {"missed_joins": False}}
    )
    items = build_items(fund, idx, sign, sal, ident, times, trace, cfg)
    assert not _by_group(items, "missed_linker")


def test_track_items_see_harmonics():
    frags = (_fragment(0, 100, 400.0, 0), _fragment(0, 100, 800.0, 1, sal=0.6))
    fund, idx, sign, _, ident, sal, times = _assemble(*frags)
    items = build_items(fund, idx, sign, sal, ident, times, Trace(), Config.classic())
    tracks = {it.ids[0]: it for it in _by_group(items, "track")}
    assert set(tracks) == {0.0, 1.0}
    assert tracks[1.0].x["harm_low"] == pytest.approx(1.0)
    assert tracks[1.0].x["harm_high"] == 0.0
    assert tracks[0.0].x["harm_high"] == pytest.approx(1.0)
    assert tracks[1.0].raw == pytest.approx(0.6)  # mean salience, uncalibrated
    assert tracks[1.0].x["nn_df"] == 20.0  # capped: nothing near 800 Hz
    assert tracks[0.0].action == "delete"
    np.testing.assert_array_equal(
        np.sort(tracks[1.0].rows[0]), np.flatnonzero(ident == 1)
    )


def test_rise_join_item():
    frags = (
        _fragment(0, 100, 700.0, 0),
        _fragment(100.5, 200, 701.0, 1, rise=(12.0, 8.0)),
    )
    fund, idx, sign, _, frag, sal, times = _assemble(*frags)
    trace = Trace(frag_v=frag.copy(), linked_v=frag.copy())
    cfg = Config.from_dict(
        {
            "detector": {"method": "classic"},
            "rises": {"enabled": True, "min_salience": None},
        }
    )
    ident = join_rises(fund, idx, sign, frag, times, cfg.rises, trace=trace.rises)
    assert trace.rises["accepted"].tolist() == [True]
    trace.joined_v = ident.copy()
    items = build_items(fund, idx, sign, sal, ident, times, trace, cfg)
    (join,) = _by_group(items, "join_rise")
    assert join.x["rise_jump"] == pytest.approx(12.0, abs=2.0)
    assert join.x["rise_corr"] > 0.9
    assert join.x["rise_rms"] < 1.0
    assert 0.9 < join.raw <= 1.0
    assert join.k == idx[frag == 1].min()
    assert len(join.rows[0]) == (frag == 0).sum()
    (track,) = _by_group(items, "track")
    assert track.x["n_rise"] == 1


def test_rise_trace_does_not_change_the_result():
    frags = (
        _fragment(0, 100, 700.0, 0),
        _fragment(100.5, 200, 701.0, 1, rise=(12.0, 8.0)),
        _fragment(99, 200, 712.0, 2, pattern=P2),
    )
    fund, idx, sign, _, frag, _, times = _assemble(*frags)
    cfg = RisesConfig(enabled=True, min_salience=None)
    trace = {}
    np.testing.assert_array_equal(
        join_rises(fund, idx, sign, frag, times, cfg),
        join_rises(fund, idx, sign, frag, times, cfg, trace=trace),
    )
    assert len(trace["a"]) == len(trace["accepted"]) == len(trace["rms"])


def test_link_evidence():
    # tracks 0: detections 0, 2, 4; 1: detections 1, 3
    track = np.array([0, 1, 0, 1, 0])
    src = np.array([0, 0, 1, 2, 2, 3])
    dst = np.array([2, 3, 3, 4, 3, 4])
    prob = np.array([0.9, 0.4, 0.8, 0.7, 0.3, 0.6])
    link_p, alt_p = link_evidence(track, src, dst, prob)
    np.testing.assert_allclose(link_p, [np.nan, np.nan, 0.9, 0.8, 0.7])
    np.testing.assert_allclose(alt_p, [0, 0, 0, 0.4, 0.6])


def test_trace_expand():
    t = Trace(frag_v=np.array([1.0, 2.0]), track={"alt_p": np.array([0.5, 0.2])})
    e = t.expand(np.array([True, False, True]))
    np.testing.assert_array_equal(e.frag_v, [1.0, np.nan, 2.0])
    np.testing.assert_array_equal(e.track["alt_p"], [0.5, 0.0, 0.2])
    assert e.linked_v is None


def _model():
    track = GroupModel(
        features=["a", "b"],
        mean=np.array([1.0, 0.0]),
        std=np.array([2.0, 1.0]),
        weights=[np.array([[2.0], [-1.0]])],
        biases=[np.array([0.5])],
    )
    join = GroupModel(
        features=["a"],
        mean=np.zeros(1),
        std=np.ones(1),
        weights=[np.ones((1, 3)), np.ones((3, 1))],
        biases=[np.zeros(3), np.array([-1.0])],
        cal_x=np.array([0.0, 0.5, 1.0]),
        cal_y=np.array([0.0, 0.2, 1.0]),
    )
    return ReviewModel({"track": track, "join_linker": join}, {"trained_on": "test"})


def _item(group, raw=0.5, **x):
    return Item(group, 1.0, 500.0, 0.0, 2.0, (3.0,), raw, x, "why", 10, ())


def test_model_probabilities_missing_features_and_calibration():
    m = _model()
    items = [
        _item("track", a=3.0, b=0.5),
        _item("track", a=3.0),  # b missing: the training mean
        _item("join_linker", a=1.0),
        _item("join_rise", raw=0.7),  # no model: the raw score
    ]
    conf = confidences(items, m)
    sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
    assert conf[0] == pytest.approx(sig(2.0 * 1.0 - 0.5 + 0.5))
    assert conf[1] == pytest.approx(sig(2.0 * 1.0 + 0.5))
    p = sig(3.0 - 1.0)  # hidden ReLU(1) * 3 - 1
    assert conf[2] == pytest.approx(np.interp(p, [0, 0.5, 1], [0, 0.2, 1]))
    assert conf[3] == 0.7
    np.testing.assert_array_equal(confidences(items, None), [0.5, 0.5, 0.5, 0.7])


def test_model_roundtrip(tmp_path):
    m = _model()
    m.save(tmp_path / "m.npz")
    back = ReviewModel.load(tmp_path / "m.npz")
    assert back.meta == {"trained_on": "test"}
    items = [_item("track", a=0.3, b=2.0), _item("join_linker", a=-1.0)]
    np.testing.assert_allclose(confidences(items, back), confidences(items, m))
    assert back.groups["track"].cal_x is None


def test_report_lowest_confidence_first():
    items = [_item("track"), _item("join_linker"), _item("missed_linker")]
    items[1].k = 7
    out = report(items, np.array([0.9, 0.2, 0.5]), "m.npz", Config.classic())
    assert out["version"] == 1 and out["calibration"] == "m.npz"
    assert [r["kind"] for r in out["items"]] == ["join", "missed_join", "track"]
    assert [r["action"] for r in out["items"]] == ["split", "merge", "delete"]
    assert out["items"][0]["k"] == 7 and "k" not in out["items"][1]
    for key in ("confidence", "t", "f", "t0", "t1", "ids", "impact", "why", "source"):
        assert key in out["items"][0]
    cfg = Config.from_dict(
        {"detector": {"method": "classic"}, "review": {"max_confidence": 0.6}}
    )
    assert len(report(items, np.array([0.9, 0.2, 0.5]), None, cfg)["items"]) == 2


def _scene(tmp_path):
    fund, idx, sign, cplx, _frag, sal, times = _assemble(
        _fragment(0, 100, 700.0, 0),
        _fragment(112, 300, 700.3, 1),
        _fragment(0, 300, 640.0, 2, pattern=P2),
    )
    model = tmp_path / "rule.npz"
    RULE.save(model)
    cfg = Config.from_dict(
        {
            "detector": {"method": "classic"},
            "linking": {"enabled": True, "model": str(model)},
        }
    )
    cfg.tracking.max_dt = 5.0  # the 12 s gap breaks the track
    res = Results(
        fund, idx, sign, np.full(len(fund), np.nan), times, cplx_v=cplx, sal_v=sal
    )
    return res, cfg


def test_pipeline_writes_review_json_and_keeps_identities(tmp_path):
    from wavetracker.pipeline import track_results

    res, cfg = _scene(tmp_path)
    track_results(res, cfg)
    assert res.review is None
    res.save(tmp_path / "off")
    assert not (tmp_path / "off" / REVIEW_FILE).exists()
    plain = res.ident_v.copy()

    cfg.review.enabled = True
    res.ident_v[:] = np.nan
    track_results(res, cfg)
    np.testing.assert_array_equal(res.ident_v, plain)
    res.save(tmp_path / "on")
    out = json.loads((tmp_path / "on" / REVIEW_FILE).read_text())
    back = Results.load(tmp_path / "on")
    assert back.frag_v is not None and len(back.frag_v) == len(res.fund_v)
    (join,) = [r for r in out["items"] if r["kind"] == "join"]
    ident = back.ident_v == join["ids"][0]
    moved = np.isin(back.frag_v, join["fragments"]) & ident
    assert moved.sum() + join["impact"] == ident.sum()  # the smaller side stays
    assert out["calibration"] is None
    kinds = sorted(r["kind"] for r in out["items"])
    assert kinds == ["join", "track", "track"]
    conf = [r["confidence"] for r in out["items"]]
    assert conf == sorted(conf)
    assert res.meta["review_config"]["enabled"] is True
    assert "review" in res.meta["timings"]


def test_tracking_again_without_review_drops_the_old_flags(tmp_path):
    from wavetracker.pipeline import track_results

    res, cfg = _scene(tmp_path)
    cfg.review.enabled = True
    track_results(res, cfg)
    res.save(tmp_path / "a")
    back = Results.load(tmp_path / "a")
    assert back.frag_v is not None
    cfg.review.enabled = False
    track_results(back, cfg)
    assert back.review is None and back.frag_v is None
    assert "review_config" not in back.meta
    back.save(tmp_path / "b")
    assert not (tmp_path / "b" / "frag_v.npy").exists()


def test_pipeline_uses_the_review_model(tmp_path):
    from wavetracker.pipeline import track_results

    res, cfg = _scene(tmp_path)
    m = ReviewModel(
        {
            "join_linker": GroupModel(
                ["logit_p"],
                np.zeros(1),
                np.ones(1),
                [np.zeros((1, 1))],
                [np.array([2.0])],
            )
        }
    )
    m.save(tmp_path / "review.npz")
    cfg.review.enabled = True
    cfg.review.model = str(tmp_path / "review.npz")
    track_results(res, cfg)
    assert res.review["calibration"] == str(tmp_path / "review.npz")
    (join,) = [r for r in res.review["items"] if r["kind"] == "join"]
    assert join["confidence"] == pytest.approx(1 / (1 + np.exp(-2.0)), abs=1e-4)
