"""Learned (salience) detector: features, network, decoding, streaming, config."""

import numpy as np
import pytest
import torch

from wavetracker.config import Config, DetectorConfig
from wavetracker.salience import (
    SalienceNet,
    SalienceSpec,
    StreamingSalience,
    decode,
    harmonic_stack,
    load_model,
    normalise,
    save_model,
)

DF = 20000.0 / 2**15
STEP = 3276 / 20000.0


def small_spec(**kw) -> SalienceSpec:
    d = dict(
        freq_resolution=DF,
        frame_step=STEP,
        channels=4,
        dilations=(1, 2, 1),
    )
    d.update(kw)
    return SalienceSpec(**d)


def test_normalise_removes_level_and_scale():
    rng = np.random.default_rng(0)
    power = rng.exponential(1.0, (50, 2000)) * np.linspace(1, 100, 2000)
    log = torch.as_tensor(10 * np.log10(power), dtype=torch.float32)
    z = normalise(log, DF, band_hz=100.0)
    assert z.shape == log.shape
    assert abs(float(z.median())) < 0.1
    # a gain (constant dB offset) does not change the features
    z2 = normalise(log + 37.0, DF, band_hz=100.0)
    torch.testing.assert_close(z, z2, atol=1e-4, rtol=0)
    # -inf (digital silence) becomes finite
    log[:, :10] = -torch.inf
    assert torch.isfinite(normalise(log, DF, band_hz=100.0)).all()


def test_harmonic_stack_aligns_harmonics_on_the_candidate():
    T, K = 3, 4000
    z = torch.zeros(T, K)
    k0 = 700  # fundamental bin
    for h in (1, 2, 3, 4, 5):
        z[:, h * k0] = float(h)
    z[:, k0 // 2] = 9.0  # something an octave below
    harm = (1 / 3, 1 / 2, 1, 2, 3, 4, 5)
    st = harmonic_stack(z, 600, 800, harm)
    assert st.shape == (len(harm), T, 201)
    col = st[:, 0, k0 - 600]
    torch.testing.assert_close(col[2:], torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert col[1] == 9.0  # f/2 channel sees the line an octave below
    # h = 1 is exact: the neighbour bin does not see the fundamental
    assert st[2, 0, k0 - 600 + 1] == 0.0
    # harmonics beyond the spectrum are filled with the noise level (0)
    st2 = harmonic_stack(z, 600, 800, (1, 10))
    assert (st2[1] == 0).all()


def test_harmonic_stack_tolerates_off_bin_harmonics():
    z = torch.zeros(1, 4000)
    z[0, 3 * 700 + 1] = 1.0  # 3rd harmonic of a fish half a bin above k=700
    st = harmonic_stack(z, 690, 710, (1, 3))
    assert st[1, 0, 700 - 690] == 1.0


def test_decode_finds_close_peaks_and_applies_offsets():
    T, F = 2, 50
    logits = torch.full((T, F), -8.0)
    logits[0, 10] = 3.0
    logits[0, 12] = 2.0  # 2 bins (1.2 Hz) away: a separate fish
    logits[0, 11] = 0.0  # the dip between them
    logits[1, 30] = -1.0  # below threshold
    offset = torch.zeros(T, F)
    offset[0, 10] = 0.25
    frames, bins, freqs, sal = decode(logits, offset, threshold=0.5, k_lo=100, df=DF)
    assert list(frames) == [0, 0]
    assert list(bins) == [110, 112]
    np.testing.assert_allclose(freqs, [(110.25) * DF, 112 * DF])
    assert sal[0] > sal[1] > 0.5


def test_net_keeps_shape_and_receptive_field():
    spec = small_spec()
    net = SalienceNet(spec).eval()
    assert net.context_frames == (1 + 2 + 1) * 1  # kernel 3 frames: +-1 per dilation
    x = torch.zeros(1, len(spec.harmonics), 40, 60)
    with torch.no_grad():
        a, off = net(x)
        assert a.shape == (1, 40, 60) and off.shape == (1, 40, 60)
        assert float(off.abs().max()) <= 0.5
        x2 = x.clone()
        x2[0, :, 20, 30] = 5.0
        b, _ = net(x2)
    changed = (a - b).abs()[0].amax(dim=1) > 0
    rows = torch.nonzero(changed).flatten()
    assert (
        rows.min() >= 20 - net.context_frames and rows.max() <= 20 + net.context_frames
    )


def test_model_roundtrip_and_grid_check(tmp_path):
    spec = small_spec()
    net = SalienceNet(spec)
    path = tmp_path / "m.pt"
    save_model(path, net, spec, meta={"note": "test"})
    net2, spec2, meta = load_model(path, "cpu")
    assert spec2 == spec and meta["note"] == "test"
    for p1, p2 in zip(
        net.state_dict().values(), net2.state_dict().values(), strict=True
    ):
        torch.testing.assert_close(p1, p2)
    freqs = np.fft.rfftfreq(2**15, 1 / 20000.0)
    StreamingSalience(net2, spec2, freqs, STEP, 400, 1200, 0.5, torch.device("cpu"))
    coarse = np.fft.rfftfreq(2**14, 1 / 20000.0)
    with pytest.raises(ValueError, match="resolution"):
        StreamingSalience(
            net2, spec2, coarse, STEP, 400, 1200, 0.5, torch.device("cpu")
        )


def test_streaming_equals_one_pass():
    """Block boundaries are invisible: blocks with context give the same
    detections as the whole recording at once."""
    torch.manual_seed(0)
    spec = small_spec(dilations=(1, 4, 2))
    net = SalienceNet(spec).eval()
    freqs = np.fft.rfftfreq(2**15, 1 / 20000.0)
    rng = np.random.default_rng(1)
    K = int(np.searchsorted(freqs, 5000))
    log = torch.as_tensor(
        10 * np.log10(rng.exponential(1.0, (K, 70))), dtype=torch.float32
    )
    for k in (1310, 1312):  # two lines at 800 Hz, 1.2 Hz apart
        log[k] += 15
    log[2 * 1310] += 10

    def run(block_sizes):
        s = StreamingSalience(
            net, spec, freqs, STEP, 400, 1200, 0.3, torch.device("cpu")
        )
        out, f0 = [], 0
        for n in block_sizes:
            for det, payload in s.push(s.features(log[:, f0 : f0 + n]), f0):
                out.append((det[0] + payload, det[2], det[3]))
            f0 += n
        for det, payload in s.finish():
            out.append((det[0] + payload, det[2], det[3]))
        fr = np.concatenate([o[0] for o in out])
        fq = np.concatenate([o[1] for o in out])
        sl = np.concatenate([o[2] for o in out])
        order = np.lexsort((fq, fr))
        return fr[order], fq[order], sl[order]

    whole = run([70])
    blocks = run([25, 3, 30, 12])
    assert len(whole[0]) > 0
    for a, b in zip(whole, blocks, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)


def test_detector_config():
    assert Config().detector.method == "classic"
    cfg = Config.from_dict({"detector": {"method": "learned", "threshold": 0.3}})
    assert cfg.detector.method == "learned" and cfg.detector.threshold == 0.3
    with pytest.raises(ValueError, match="method"):
        DetectorConfig(method="deterministic")


def test_pipeline_learned_detector_is_block_independent(synthetic_wav, tmp_path):
    """The learned detector runs inside `detect`, stores a salience per
    detection, and gives the same detections whatever the block length."""
    from wavetracker.pipeline import detect
    from wavetracker.results import Results

    path, _ = synthetic_wav
    torch.manual_seed(3)
    spec = small_spec(dilations=(1, 8, 2))
    model = tmp_path / "m.pt"
    save_model(model, SalienceNet(spec).eval(), spec)
    runs = []
    for block in (60.0, 7.0):
        cfg = Config()
        cfg.detector.method = "learned"
        cfg.detector.model = str(model)
        cfg.detector.threshold = 0.2
        cfg.harmonic_groups.min_freq, cfg.harmonic_groups.max_freq = 400.0, 1000.0
        cfg.spectrogram.block_duration = block
        # both adapt per block; switched off so only the detector is compared
        cfg.interference.enabled = False
        cfg.clicks.enabled = False
        out = detect(path, tmp_path / f"b{block}", cfg, device="cpu")
        r = out.results
        assert r.sal_v is not None and len(r.sal_v) == len(r.fund_v) == len(r.sign_v)
        assert np.all((r.fund_v >= 400 - 1) & (r.fund_v <= 1000 + 1))
        assert np.all(r.sal_v >= 0.2)
        assert r.meta["detector_model"]["path"].endswith("m.pt")
        loaded = Results.load(tmp_path / f"b{block}")
        np.testing.assert_array_equal(loaded.sal_v, r.sal_v)
        order = np.lexsort((r.fund_v, r.idx_v))
        runs.append((r.idx_v[order], r.fund_v[order], r.sal_v[order]))
    assert len(runs[0][0]) > 0

    # FFTs of different block sizes differ in the last float bits, which can
    # flip peaks right at the threshold of this untrained (noisy) model
    def unmatched(x, y):
        n = 0
        for i in np.unique(x[0]):
            fx, fy = x[1][x[0] == i], y[1][y[0] == i]
            d = (
                np.abs(fx[:, None] - fy[None, :]).min(1)
                if len(fy)
                else np.full(len(fx), np.inf)
            )
            n += int(np.sum(d > 1e-3))
        return n

    assert unmatched(runs[0], runs[1]) + unmatched(runs[1], runs[0]) <= 1e-4 * len(
        runs[0][0]
    )
