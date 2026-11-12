from typing import ClassVar

import numpy as np
import pytest
import torch

from wavetracker import assocnet
from wavetracker.config import Config, TrackingConfig
from wavetracker.tracking import track

DT = 0.164


def _lines(n_frames=200, freqs=(500.0, 503.0, 620.0), n_ch=4, seed=0, drop=0.1):
    """Detections of a few flat fish with their own electrode patterns."""
    rng = np.random.default_rng(seed)
    idx, fund, sign, fish = [], [], [], []
    patterns = rng.random((len(freqs), n_ch)) + 0.05
    for k, f in enumerate(freqs):
        keep = rng.random(n_frames) > drop
        fr = np.flatnonzero(keep)
        idx.append(fr)
        fund.append(f + rng.normal(0, 0.05, len(fr)))
        sign.append(patterns[k] * rng.uniform(0.8, 1.2, (len(fr), n_ch)))
        fish.append(np.full(len(fr), k))
    idx, fund, sign, fish = map(np.concatenate, (idx, fund, sign, fish))
    order = np.lexsort((fund, idx))
    times = np.arange(n_frames) * DT
    sign = sign[order]
    cplx = np.sqrt(sign).astype(np.complex64) * np.exp(
        1j * rng.uniform(0, 6, (len(order), 1))
    )
    sal = rng.uniform(0.6, 1.0, len(order)).astype(np.float32)
    return fund[order], idx[order], sign, cplx, sal, times, fish[order]


def _model(rounds=2, seed=0):
    torch.manual_seed(seed)
    return assocnet.AssocModel(assocnet.AssocNet(16, rounds, 8).eval(), 1.0, 2.5)


def test_candidate_edges_follow_time_and_frequency():
    idx = np.array([0, 0, 1, 2, 5, 9])
    fund = np.array([500.0, 510.0, 501.0, 505.0, 500.5, 500.0])
    src, dst = assocnet.candidate_edges(idx, fund, max_frames=4, max_df=2.5)
    got = set(zip(src.tolist(), dst.tolist(), strict=True))
    # 0 -> 2 (1 Hz), 0 -> 4 (5 frames: too late), 2 -> 4 and 4 -> 5 (4 frames)
    assert got == {(0, 2), (2, 4), (4, 5)}
    assert np.all(idx[dst] > idx[src])


def test_link_edges_one_successor_most_probable_first():
    # frame 0: a0, a1; frame 1: b0, b1; edges crosswise less probable
    idx = np.array([0, 0, 1, 1, 2])
    src = np.array([0, 0, 1, 1, 2, 3])
    dst = np.array([2, 3, 2, 3, 4, 4])
    prob = np.array([0.9, 0.2, 0.3, 0.8, 0.4, 0.7])
    tr = assocnet.link_edges(idx, src, dst, prob, threshold=0.5)
    assert tr[0] == tr[2] and tr[1] == tr[3] and tr[0] != tr[1]
    assert tr[4] == tr[3]  # 3 -> 4 (0.7) wins over 2 -> 4 (0.4 < threshold)
    tr = assocnet.link_edges(idx, src, dst, prob, threshold=0.95)
    assert len(np.unique(tr)) == 5  # nothing linked


def test_link_edges_skips_frames():
    idx = np.array([0, 3, 4])
    src, dst = np.array([0, 0, 1]), np.array([1, 2, 2])
    tr = assocnet.link_edges(idx, src, dst, np.array([0.9, 0.95, 0.6]), 0.5)
    # 1 takes 0's only successor slot; 2 then continues 1's track
    assert tr[0] == tr[1] == tr[2]


def test_chunked_scores_equal_whole_recording():
    fund, idx, sign, cplx, sal, times, _ = _lines(n_frames=300)
    model = _model()
    k = round(model.max_dt / DT)
    src, dst = assocnet.candidate_edges(idx, fund, k, model.max_df)
    d = assocnet.detection_inputs(fund, idx, times, sign, cplx, sal)
    whole = assocnet.edge_probabilities(
        model, d, idx, src, dst, DT, chunk_frames=10_000
    )
    for chunk in (37, 100):
        part = assocnet.edge_probabilities(
            model, d, idx, src, dst, DT, chunk_frames=chunk
        )
        np.testing.assert_allclose(part, whole, atol=1e-5)


def test_edge_inputs_ignore_common_phase_and_electrode_order():
    fund, idx, sign, cplx, sal, times, _ = _lines(n_frames=50)
    model = _model()
    src, dst = assocnet.candidate_edges(idx, fund, 6, 2.5)
    d = assocnet.detection_inputs(fund, idx, times, sign, cplx, sal)
    rot = cplx * np.exp(1j * np.random.default_rng(1).uniform(0, 6, (len(cplx), 1)))
    perm = np.array([2, 0, 3, 1])
    d2 = assocnet.detection_inputs(fund, idx, times, sign[:, perm], rot[:, perm], sal)
    with torch.no_grad():
        a = model.net(d, torch.as_tensor(src), torch.as_tensor(dst))
        b = model.net(d2, torch.as_tensor(src), torch.as_tensor(dst))
    np.testing.assert_allclose(a.numpy(), b.numpy(), atol=1e-4)


def test_any_number_of_electrodes():
    model = _model()
    for n_ch in (2, 9):
        fund, idx, sign, cplx, sal, times, _ = _lines(n_frames=40, n_ch=n_ch)
        ident = assocnet.track_learned(fund, idx, sign, cplx, sal, times, model)
        assert ident.shape == fund.shape


def test_save_load_roundtrip(tmp_path):
    fund, idx, sign, cplx, sal, times, _ = _lines(n_frames=60)
    model = _model()
    model.save(tmp_path / "m.pt")
    again = assocnet.AssocModel.load(tmp_path / "m.pt")
    assert again.max_dt == model.max_dt and again.max_df == model.max_df
    a = assocnet.track_learned(fund, idx, sign, cplx, sal, times, model, 0.3)
    b = assocnet.track_learned(fund, idx, sign, cplx, sal, times, again, 0.3)
    np.testing.assert_array_equal(a, b)


class _RuleNet(torch.nn.Module):
    """Stands in for a trained network: same fish if the frequencies are close."""

    hparams: ClassVar[dict] = {"rounds": 1}

    def forward(self, d, src, dst):
        return 4.0 - 8.0 * (d.f[dst] - d.f[src]).abs().float()


def test_track_learned_follows_lines_and_drops_short_tracks():
    fund, idx, sign, cplx, sal, times, fish = _lines(n_frames=200)
    # a lone stray detection
    fund = np.append(fund, 700.0)
    idx = np.append(idx, 100)
    sign = np.vstack([sign, sign[:1]])
    cplx = np.vstack([cplx, cplx[:1]])
    sal = np.append(sal, 0.7)
    fish = np.append(fish, -1)
    order = np.lexsort((fund, idx))
    fund, idx, sign, cplx, sal, fish = (
        x[order] for x in (fund, idx, sign, cplx, sal, fish)
    )
    model = assocnet.AssocModel(_RuleNet(), 1.0, 2.5)
    ident = assocnet.track_learned(
        fund, idx, sign, cplx, sal, times, model, threshold=0.5, min_detections=3
    )
    assert np.isnan(ident[fish < 0]).all()
    for k in range(3):
        ids = ident[fish == k]
        assert np.isfinite(ids).all() and len(np.unique(ids)) == 1
    assert len(np.unique(ident[np.isfinite(ident)])) == 3


def test_tracking_method_learned_through_track(tmp_path):
    fund, idx, sign, cplx, sal, times, _ = _lines(n_frames=120)
    _model().save(tmp_path / "m.pt")
    cfg = TrackingConfig(method="learned", assoc_model=str(tmp_path / "m.pt"))
    ident = track(fund, idx, sign, times, cfg, 550, 700, sal_v=sal, cplx_v=cplx)
    assert np.isnan(ident[fund < 550]).all()  # outside the fish range
    assert np.isfinite(ident[fund >= 550]).all()


def test_tracking_method_config():
    assert TrackingConfig().method == "classic"
    assert TrackingConfig(method="learned").assoc_model is None  # the packaged one
    with pytest.raises(ValueError, match="method"):
        TrackingConfig(method="hungarian")
    cfg = Config.from_dict({"tracking": {"method": "learned", "assoc_model": "x.pt"}})
    assert cfg.tracking.assoc_threshold == 0.7  # the default pipeline's
    cfg = Config.from_dict(
        {"detector": {"method": "classic"}, "tracking": {"method": "learned"}}
    )
    assert cfg.tracking.assoc_threshold == 0.5


def test_packaged_model_loads():
    model = assocnet.load_model(None)
    assert model.max_dt > 0 and model.max_df > 0


def test_cost_mode_keeps_the_rule_based_tracker_with_learned_errors(tmp_path):
    fund, idx, sign, cplx, sal, times, fish = _lines(n_frames=300, drop=0.3)
    model = assocnet.AssocModel(_RuleNet(), 1.0, 2.5)
    key = (str((tmp_path / "rule.pt").resolve()), "cpu")
    assocnet._MODELS[key] = model  # a stand-in for a trained network
    cfg = TrackingConfig(
        method="learned",
        assoc_mode="cost",
        assoc_model=str(tmp_path / "rule.pt"),
        assoc_device="cpu",  # the stand-in is cached for the CPU
    )
    try:
        ident = track(fund, idx, sign, times, cfg, sal_v=sal, cplx_v=cplx)
    finally:
        del assocnet._MODELS[key]
    for k in range(3):  # the rule-based tracker never assigns the first frame
        ids = ident[(fish == k) & (idx > idx[0])]
        assert np.isfinite(ids).all() and len(np.unique(ids)) == 1
    assert len(np.unique(ident[np.isfinite(ident)])) == 3


def test_assoc_mode_config():
    assert TrackingConfig().assoc_mode == "link"
    with pytest.raises(ValueError, match="assoc_mode"):
        TrackingConfig(assoc_mode="both")


def _oom_unless_cpu(real):
    def fake(*args, device="cpu", **kwargs):
        if device != "cpu":
            raise torch.OutOfMemoryError("CUDA out of memory (test)")
        return real(*args, device=device, **kwargs)

    return fake


def test_learned_association_falls_back_to_the_cpu(tmp_path, monkeypatch, caplog):
    fund, idx, sign, cplx, sal, times, _ = _lines(n_frames=120)
    _model().save(tmp_path / "m.pt")
    cfg = TrackingConfig(method="learned", assoc_model=str(tmp_path / "m.pt"))
    want = track(fund, idx, sign, times, cfg, sal_v=sal, cplx_v=cplx)
    monkeypatch.setattr(
        assocnet, "track_learned", _oom_unless_cpu(assocnet.track_learned)
    )
    load = assocnet.load_model
    monkeypatch.setattr(assocnet, "load_model", lambda path, device: load(path, "cpu"))
    cfg.assoc_device = "cuda"  # need not exist: the fakes never use it
    with caplog.at_level("WARNING"):
        got = track(fund, idx, sign, times, cfg, sal_v=sal, cplx_v=cplx)
    assert np.array_equal(got, want, equal_nan=True)
    assert "out of memory" in caplog.text


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_learned_association_is_the_same_on_the_gpu(tmp_path):
    fund, idx, sign, cplx, sal, times, _ = _lines(n_frames=300, drop=0.3)
    _model().save(tmp_path / "m.pt")
    cfg = TrackingConfig(
        method="learned", assoc_model=str(tmp_path / "m.pt"), assoc_device="cpu"
    )
    cpu = track(fund, idx, sign, times, cfg, sal_v=sal, cplx_v=cplx)
    cfg.assoc_device = "cuda"
    gpu = track(fund, idx, sign, times, cfg, sal_v=sal, cplx_v=cplx)
    assert np.array_equal(cpu, gpu, equal_nan=True)
