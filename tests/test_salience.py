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
    electrode_features,
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


def test_recording_edges_look_like_the_middle():
    """Where a recording starts or ends there is no context: it is mirrored
    from the data, so a recording that does not change over time gives the
    same salience in its first and last frames as in the middle (zeros
    there gave bursts of false detections in field recordings)."""
    torch.manual_seed(5)
    spec = small_spec(dilations=(1, 4, 2))
    net = SalienceNet(spec).eval()
    freqs = np.fft.rfftfreq(2**15, 1 / 20000.0)
    rng = np.random.default_rng(2)
    K = int(np.searchsorted(freqs, 5000))
    frame = 10 * np.log10(rng.exponential(1.0, K))
    frame[1310] += 15
    log = torch.as_tensor(np.repeat(frame[:, None], 40, axis=1), dtype=torch.float32)
    s = StreamingSalience(net, spec, freqs, STEP, 400, 1200, 0.05, torch.device("cpu"))
    ((det, _),) = s.push(s.features(log)) + s.finish()
    fr, _, fq, sal = det
    per_frame = [sorted(np.round(fq[fr == j], 6)) for j in range(40)]
    assert all(p == per_frame[20] for p in per_frame)
    np.testing.assert_allclose(sal[fr == 0], sal[fr == 20], atol=1e-5)


def test_temporal_stack_extends_the_receptive_field_exactly():
    """Per-bin temporal layers (kernel along time only) add their dilations
    to the context, and nothing outside it reaches an output frame; they
    do not widen the frequency context."""
    torch.manual_seed(1)
    spec = small_spec(temporal_dilations=(1, 4, 2))
    net = SalienceNet(spec).eval()
    assert net.context_frames == (1 + 2 + 1) + (1 + 4 + 2)
    T, K = 60, 40
    x = torch.zeros(1, len(spec.harmonics), T, K)
    with torch.no_grad():
        a, _ = net(x)
        x2 = x.clone()
        x2[0, :, 30, 20] = 5.0
        b, _ = net(x2)
    diff = (a - b).abs()[0]
    rows = torch.nonzero(diff.amax(dim=1) > 0).flatten()
    assert rows.min() == 30 - net.context_frames
    assert rows.max() == 30 + net.context_frames
    cols = torch.nonzero(diff.amax(dim=0) > 0).flatten()
    trunk = SalienceNet(small_spec())  # same frequency context as without
    assert cols.min() >= 20 - 3 * (spec.kernel_bins // 2)
    assert cols.max() <= 20 + 3 * (spec.kernel_bins // 2)
    assert len(trunk.temporal) == 0


def test_spec_without_temporal_keys_loads_as_before(tmp_path):
    """Models saved before the temporal stack existed rebuild unchanged."""
    spec = small_spec()
    d = {k: v for k, v in spec.__dict__.items() if not k.startswith("temporal")}
    old = SalienceSpec.from_dict(d)
    assert old.temporal_dilations == ()
    net = SalienceNet(old)
    assert not any(k.startswith("temporal") for k in net.state_dict())


def test_streaming_with_temporal_stack_equals_one_pass():
    torch.manual_seed(2)
    spec = small_spec(dilations=(1, 2), temporal_dilations=(1, 8, 4))
    net = SalienceNet(spec).eval()
    freqs = np.fft.rfftfreq(2**15, 1 / 20000.0)
    rng = np.random.default_rng(3)
    K = int(np.searchsorted(freqs, 5000))
    log = torch.as_tensor(
        10 * np.log10(rng.exponential(1.0, (K, 90))), dtype=torch.float32
    )
    log[1310] += 15

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

    whole = run([90])
    blocks = run([7, 30, 2, 40, 11])
    assert len(whole[0]) > 0
    for a, b in zip(whole, blocks, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)


def test_temporal_stack_can_follow_drifting_lines():
    """Temporal kernels with a frequency extent (and their own dilation along
    frequency) widen the frequency context by their reach, and leave the time
    context and streaming exactness as they are."""
    torch.manual_seed(4)
    spec = small_spec(
        temporal_dilations=(1, 4, 8),
        temporal_kernel_bins=3,
        temporal_bin_dilations=(1, 1, 2),
    )
    net = SalienceNet(spec).eval()
    assert net.context_frames == (1 + 2 + 1) + (1 + 4 + 8)
    T, K = 60, 60
    x = torch.zeros(1, len(spec.harmonics), T, K)
    with torch.no_grad():
        a, _ = net(x)
        x2 = x.clone()
        x2[0, :, 30, 30] = 5.0
        b, _ = net(x2)
    diff = (a - b).abs()[0]
    rows = torch.nonzero(diff.amax(dim=1) > 0).flatten()
    cols = torch.nonzero(diff.amax(dim=0) > 0).flatten()
    assert rows.min() == 30 - net.context_frames
    assert rows.max() == 30 + net.context_frames
    reach = 3 * (spec.kernel_bins // 2) + (1 + 1 + 2)
    assert cols.min() == 30 - reach and cols.max() == 30 + reach


# --- pair head: two fish in one spectral peak --------------------------------


def test_models_without_pair_head_load_and_run_as_before(tmp_path):
    """Specs saved before the pair head existed rebuild without it: the same
    parameters, the same two outputs."""
    spec = small_spec()
    d = {k: v for k, v in spec.__dict__.items() if not k.startswith("pair")}
    old = SalienceSpec.from_dict(d)
    assert old.pair_head is False
    net = SalienceNet(old).eval()
    assert not any(k.startswith("pair") for k in net.state_dict())
    x = torch.randn(1, len(spec.harmonics), 20, 30)
    with torch.no_grad():
        out = net(x)
        full = net.outputs(x)
    assert len(out) == 2 and set(full) == {"logits", "offset"}
    torch.testing.assert_close(out[0], full["logits"])


def test_pair_head_outputs():
    spec = small_spec(pair_head=True, pair_reach=2.5)
    net = SalienceNet(spec).eval()
    x = torch.randn(2, len(spec.harmonics), 20, 30) * 3
    with torch.no_grad():
        out = net.outputs(x)
        logits, _ = net(x)
    torch.testing.assert_close(logits, out["logits"])
    assert out["pair_logits"].shape == (2, 20, 30)
    assert out["pair_offset"].shape == (2, 2, 20, 30)
    assert float(out["pair_offset"].abs().max()) <= 2.5


def _peaks(T=1, F=40):
    logits = torch.full((T, F), -8.0)
    offset = torch.zeros(T, F)
    pair = torch.full((T, F), -8.0)
    poff = torch.zeros(2, T, F)
    return logits, offset, pair, poff


def test_decode_splits_a_pair_peak_into_two_detections():
    logits, offset, pair, poff = _peaks()
    logits[0, 10] = 3.0
    offset[0, 10] = 0.2
    pair[0, 10] = 2.0
    poff[:, 0, 10] = torch.tensor([0.9, -0.7])  # unordered
    logits[0, 25] = 3.0  # a single fish
    offset[0, 25] = -0.1
    poff[:, 0, 25] = torch.tensor([1.0, -1.0])  # ignored: pair probability low
    fr, b, f, s = decode(logits, offset, 0.5, 100, DF, pair, poff, 0.5)
    # the peak's own detection stays, the pair offset farther from it is added
    np.testing.assert_allclose(f, np.array([109.3, 110.2, 124.9]) * DF)
    assert list(fr) == [0, 0, 0]
    assert list(b) == [109, 110, 125]  # the bin nearest each frequency
    # the second fish's salience is the pair probability
    np.testing.assert_allclose(s[0], torch.sigmoid(torch.tensor(2.0)), rtol=1e-6)
    assert s[1] > 0.9
    # without the pair outputs (or below the pair threshold): one per peak
    one = decode(logits, offset, 0.5, 100, DF)
    np.testing.assert_allclose(one[2], np.array([110.2, 124.9]) * DF)
    high = decode(logits, offset, 0.5, 100, DF, pair, poff, 0.95)
    np.testing.assert_allclose(high[2], one[2])


def test_decode_does_not_split_a_peak_with_a_close_neighbour():
    """Two fish the salience already shows as two peaks (2 bins apart) stay
    two detections, even where the pair head flags both."""
    logits, offset, pair, poff = _peaks()
    logits[0, 10], logits[0, 11], logits[0, 12] = 3.0, 0.0, 2.5
    pair[0, 10] = pair[0, 12] = 3.0
    poff[:, 0, 10] = torch.tensor([0.0, 2.0])
    poff[:, 0, 12] = torch.tensor([-2.0, 0.0])
    _, _, f, _ = decode(logits, offset, 0.5, 100, DF, pair, poff, 0.5)
    np.testing.assert_allclose(f, np.array([110, 112]) * DF)


def test_decode_keeps_pair_detections_inside_the_window():
    logits, offset, pair, poff = _peaks(F=10)
    logits[0, 0] = 3.0
    pair[0, 0] = 3.0
    poff[:, 0, 0] = torch.tensor([-1.4, 0.3])
    _, b, f, _ = decode(logits, offset, 0.5, 100, DF, pair, poff, 0.5)
    np.testing.assert_allclose(f, np.array([98.6, 100.0]) * DF)
    assert list(b) == [100, 100]  # clamped to the candidate bins


def test_streaming_with_pair_head_equals_one_pass():
    torch.manual_seed(6)
    spec = small_spec(
        dilations=(1, 2), temporal_dilations=(1, 4), pair_head=True, pair_dilations=(2,)
    )
    net = SalienceNet(spec).eval()
    with torch.no_grad():  # make the pair head fire often
        net.pair.bias[0] = 3.0
    freqs = np.fft.rfftfreq(2**15, 1 / 20000.0)
    rng = np.random.default_rng(7)
    K = int(np.searchsorted(freqs, 5000))
    log = torch.as_tensor(
        10 * np.log10(rng.exponential(1.0, (K, 60))), dtype=torch.float32
    )
    log[1310] += 15

    def run(block_sizes):
        s = StreamingSalience(
            net,
            spec,
            freqs,
            STEP,
            400,
            1200,
            0.3,
            torch.device("cpu"),
            pair_threshold=0.5,
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

    whole = run([60])
    blocks = run([7, 20, 3, 30])
    no_pair = StreamingSalience(
        net, spec, freqs, STEP, 400, 1200, 0.3, torch.device("cpu"), pair_threshold=1.0
    )
    single = sum(
        len(d[0]) for d, _ in no_pair.push(no_pair.features(log)) + no_pair.finish()
    )
    assert len(whole[0]) > single  # pairs were split
    for a, b in zip(whole, blocks, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)


def test_detector_config_pair_threshold():
    assert DetectorConfig().pair_threshold is None
    cfg = Config.from_dict(
        {"detector": {"method": "learned", "threshold": 0.6, "pair_threshold": 0.4}}
    )
    assert cfg.detector.pair_threshold == 0.4
    with pytest.raises(ValueError, match="pair_threshold"):
        DetectorConfig(pair_threshold=1.5)


def test_pair_branch_adds_its_context():
    """Layers between the trunk and the pair head add their time context; the
    salience itself does not depend on them."""
    torch.manual_seed(8)
    spec = small_spec(temporal_dilations=(1, 2), pair_head=True, pair_dilations=(1, 4))
    net = SalienceNet(spec).eval()
    assert net.context_frames == (1 + 2 + 1) + (1 + 2) + (1 + 4)
    T, K = 60, 30
    x = torch.zeros(1, len(spec.harmonics), T, K)
    with torch.no_grad():
        a = net.outputs(x)
        x2 = x.clone()
        x2[0, :, 30, 15] = 5.0
        b = net.outputs(x2)
    rows = torch.nonzero((a["pair_logits"] - b["pair_logits"]).abs()[0].amax(1) > 0)
    assert rows.min() == 30 - net.context_frames
    assert rows.max() == 30 + net.context_frames
    sal = torch.nonzero((a["logits"] - b["logits"]).abs()[0].amax(1) > 0)
    assert sal.min() == 30 - (net.context_frames - 5)


# --- electrode branch ----------------------------------------------------------


def _two_fish_stft(E=5, K=8, T=40, seed=0, second=True):
    """Complex spectra (electrodes, bins, frames): one fish at bin 3 with a
    real (signed) pattern, optionally a second one 0.3 Hz higher."""
    rng = np.random.default_rng(seed)
    t = np.arange(T) * STEP
    a1 = rng.normal(size=E)
    a2 = rng.normal(size=E)
    X = np.zeros((E, K, T), complex)
    X[:, 3] = a1[:, None] * np.exp(2j * np.pi * 0.0 * t)[None] * np.exp(0.7j)
    if second:
        X[:, 3] += 0.8 * a2[:, None] * np.exp(2j * np.pi * 0.3 * t)[None]
    X += 1e-4 * (rng.normal(size=X.shape) + 1j * rng.normal(size=X.shape))
    return torch.as_tensor(X, dtype=torch.complex64)


def test_electrode_features_tell_one_fish_from_two():
    one = electrode_features(_two_fish_stft(second=False), 0, 7)
    two = electrode_features(_two_fish_stft(), 0, 7)
    assert one.shape == (40, 5, 3, 8)
    # one fish: fixed shares, phases 0 or pi (sin ~ 0); two: both move
    assert float(one[:, :, 2, 3].abs().max()) < 1e-3
    assert float(one[:, :, 0, 3].std(0).max()) < 1e-3
    assert float(two[:, :, 2, 3].abs().max()) > 0.1
    assert float(two[:, :, 0, 3].std(0).max()) > 0.05
    # gains and the common phase do not matter
    g = electrode_features(_two_fish_stft() * 37.0 * np.exp(1j * 1.3), 0, 7)
    torch.testing.assert_close(g, two, atol=1e-4, rtol=1e-4)
    # silence stays finite
    assert torch.isfinite(
        electrode_features(torch.zeros(3, 8, 5, dtype=torch.complex64), 0, 7)
    ).all()


def _el_spec(**kw):
    d = dict(
        temporal_dilations=(1, 2),
        pair_head=True,
        pair_dilations=(1,),
        electrode_channels=4,
        electrode_dilations=(1, 2),
    )
    d.update(kw)
    return small_spec(**d)


def test_electrode_branch_starts_at_zero_and_any_count():
    torch.manual_seed(9)
    spec = _el_spec()
    net = SalienceNet(spec).eval()
    x = torch.randn(1, len(spec.harmonics), 30, 16)
    el = electrode_features(_two_fish_stft(E=6, K=16, T=30), 0, 15).permute(1, 2, 0, 3)[
        None
    ]
    with torch.no_grad():
        a = net.outputs(x)
        b = net.outputs(x, el)
        for k in a:  # gate 0: the electrodes change nothing yet
            torch.testing.assert_close(a[k], b[k])
        net.el_gate.fill_(1.0)
        c = net.outputs(x, el)
        assert float((c["pair_logits"] - a["pair_logits"]).abs().max()) > 0
        torch.testing.assert_close(c["logits"], a["logits"])  # into "pair"
        # any order, and padded electrodes that are masked out
        perm = net.outputs(x, el[:, [3, 0, 5, 1, 4, 2]])
        torch.testing.assert_close(perm["pair_logits"], c["pair_logits"])
        pad = torch.cat([el, torch.randn(1, 3, 3, 30, 16)], 1)
        mask = torch.tensor([[True] * 6 + [False] * 3])
        padded = net.outputs(x, pad, mask)
        torch.testing.assert_close(padded["pair_logits"], c["pair_logits"])
        # two electrodes work as well
        two = net.outputs(x, el[:, :2])
        assert torch.isfinite(two["pair_logits"]).all()


def test_electrode_branch_into_trunk_changes_the_salience_and_context():
    torch.manual_seed(10)
    spec = _el_spec(
        electrode_into="trunk", dilations=(1,), electrode_dilations=(1, 2, 4)
    )
    net = SalienceNet(spec).eval()
    # electrode context (7) exceeds the convolutions' (1) before the temporal stack
    assert net.context_frames == 7 + (1 + 2) + 1
    with torch.no_grad():
        net.el_gate.fill_(1.0)
        x = torch.zeros(1, len(spec.harmonics), 50, 16)
        el = torch.zeros(1, 3, 3, 50, 16)
        a = net.outputs(x, el)
        el2 = el.clone()
        el2[0, :, :, 25, 8] = 1.0
        b = net.outputs(x, el2)
    rows = torch.nonzero((a["logits"] - b["logits"]).abs()[0].amax(1) > 0).flatten()
    assert (
        rows.min() >= 25 - net.context_frames and rows.max() <= 25 + net.context_frames
    )
    assert len(rows) > 0


def test_streaming_with_electrodes_equals_one_pass():
    torch.manual_seed(11)
    for into in ("pair", "trunk"):
        spec = _el_spec(electrode_into=into)
        net = SalienceNet(spec).eval()
        with torch.no_grad():
            net.el_gate.fill_(2.0)
            net.pair.bias[0] = 2.0
        freqs = np.fft.rfftfreq(2**15, 1 / 20000.0)
        rng = np.random.default_rng(12)
        K = int(np.searchsorted(freqs, 5000))
        T, E = 50, 4
        stft = torch.as_tensor(
            rng.normal(size=(E, K, T)) + 1j * rng.normal(size=(E, K, T)),
            dtype=torch.complex64,
        )
        stft[:, 1310] *= 30
        log = 10 * torch.log10(stft.abs().square().sum(0))

        def run(block_sizes, net=net, spec=spec, freqs=freqs, log=log, stft=stft):
            s = StreamingSalience(
                net,
                spec,
                freqs,
                STEP,
                400,
                1200,
                0.3,
                torch.device("cpu"),
                pair_threshold=0.5,
            )
            out, f0 = [], 0
            for n in block_sizes:
                sl = slice(f0, f0 + n)
                for det, payload in s.push(s.features(log[:, sl], stft[:, :, sl]), f0):
                    out.append((det[0] + payload, det[2], det[3]))
                f0 += n
            for det, payload in s.finish():
                out.append((det[0] + payload, det[2], det[3]))
            fr = np.concatenate([o[0] for o in out])
            fq = np.concatenate([o[1] for o in out])
            sl_ = np.concatenate([o[2] for o in out])
            order = np.lexsort((fq, fr))
            return fr[order], fq[order], sl_[order]

        whole = run([T])
        blocks = run([9, 20, 2, 19])
        assert len(whole[0]) > 0
        for a, b in zip(whole, blocks, strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)


def test_pipeline_learned_detector_with_electrodes(synthetic_wav, tmp_path):
    """A model with an electrode branch reads the electrodes of the recording
    inside `detect`, whatever its count, block by block."""
    from wavetracker.pipeline import detect

    path, _ = synthetic_wav
    torch.manual_seed(13)
    spec = _el_spec(electrode_into="trunk", pair_head=False, pair_dilations=())
    net = SalienceNet(spec).eval()
    with torch.no_grad():
        net.el_gate.fill_(1.0)
    model = tmp_path / "m.pt"
    save_model(model, net, spec)
    runs = []
    for block in (60.0, 7.0):
        cfg = Config()
        cfg.detector.method = "learned"
        cfg.detector.model = str(model)
        cfg.detector.threshold = 0.2
        cfg.harmonic_groups.min_freq, cfg.harmonic_groups.max_freq = 400.0, 1000.0
        cfg.spectrogram.block_duration = block
        cfg.interference.enabled = False
        cfg.clicks.enabled = False
        r = detect(path, tmp_path / f"b{block}", cfg, device="cpu").results
        order = np.lexsort((r.fund_v, r.idx_v))
        runs.append((r.idx_v[order], r.fund_v[order]))
    assert len(runs[0][0]) > 0
    n = min(len(runs[0][0]), len(runs[1][0]))
    assert abs(len(runs[0][0]) - len(runs[1][0])) <= 1e-4 * len(runs[0][0]) + 1
    np.testing.assert_allclose(runs[0][1][:n], runs[1][1][:n], atol=1e-3)
