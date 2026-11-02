"""Recordings that start long before there is a fish in them.

Field loggers start on the boat: first static with the electrodes out of the
water, then quiet water, and only later a fish, after seconds to hours. Results
for the fish must not depend on what came before.
"""

import numpy as np
import pytest
import torch
from audioio import write_audio

from wavetracker.config import Config, HarmonicGroupsConfig, TrackingConfig
from wavetracker.evaluation import evaluate
from wavetracker.harmonics import detect_harmonic_groups
from wavetracker.pipeline import DetectionThresholds, detect, track_results
from wavetracker.spectrogram import (
    band_noise_std,
    decibel,
    frame_noise_std,
    min_noise_std,
)
from wavetracker.synthetic import Fish, add_clicks, synthesize
from wavetracker.tracking import track

RATE = 20000.0
CHANNELS = 6
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
FISH_SECONDS = 60.0


def _segment(kind, seconds, seed):
    """Fish-free data at the level of `synthesize`'s default noise (0.05)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * RATE)
    if kind == "silence":
        return np.zeros((n, CHANNELS), np.float32)
    noise = (0.05 * rng.standard_normal((n, CHANNELS))).astype(np.float32)
    if kind == "quiet water":
        return noise
    if kind == "static":
        return add_clicks(noise, RATE, per_second=10.0, amplitude=50.0, rng=rng)
    raise ValueError(kind)


@pytest.fixture(scope="module")
def fish_recording():
    rng = np.random.default_rng(42)
    fish = [
        Fish(520.0, drift=1.0, position=0.1),
        Fish(715.0, drift=2.0, position=0.5),
        Fish(910.0, drift=1.5, amplitude=0.5, position=0.9),
    ]
    return synthesize(fish, FISH_SECONDS, rate=RATE, channels=CHANNELS, rng=rng)


def _run(tmp_path, name, data, device="cpu"):
    """Detect and track `data`, written the way a logger would."""
    wav = tmp_path / f"{name}.wav"
    write_audio(str(wav), 0.9 * data / np.abs(data).max(), RATE)
    cfg = Config()
    out = detect(wav, tmp_path / name, cfg, device=device)
    track_results(out.results, cfg)
    return out.results


@pytest.fixture(scope="module")
def clean_run(fish_recording, tmp_path_factory):
    return _run(tmp_path_factory.mktemp("clean"), "clean", fish_recording.data)


@pytest.mark.parametrize(
    "prefix",
    [
        ["silence"],
        ["static"],
        ["quiet water"],
        ["static", "quiet water"],
        ["silence", "static", "quiet water"],
    ],
    ids=" + ".join,
)
def test_fish_are_found_as_without_the_fish_free_start(
    prefix, fish_recording, clean_run, tmp_path
):
    seconds = 40.0
    start = seconds * len(prefix)
    parts = [_segment(kind, seconds, i) for i, kind in enumerate(prefix)]
    res = _run(tmp_path, "rec", np.concatenate([*parts, fish_recording.data]))

    truth_t = fish_recording.truth_times + start
    score = evaluate(res, truth_t, fish_recording.truth_freqs)
    clean = evaluate(clean_run, fish_recording.truth_times, fish_recording.truth_freqs)
    assert score.recall > clean.recall - 0.01
    assert score.precision > 0.98
    assert score.purity > 0.98

    # no fish before the fish: at most stray detections (windows reaching into
    # the fish's start are not "before")
    t = res.times[res.idx_v]
    early = t < start - 2.0
    assert early.sum() < 0.02 * (res.times < start - 2.0).sum()
    # the thresholds in the fish part are those of a recording without the start
    assert res.meta["noise_std_blocks"][-1] == pytest.approx(
        clean_run.meta["noise_std_blocks"][-1], rel=0.1
    )
    # every fish stays one identity
    assert res.n_ids <= clean_run.n_ids + 1


@pytest.mark.parametrize("device", DEVICES)
def test_hours_of_nothing_do_not_matter(device, fish_recording, clean_run, tmp_path):
    """Hours are many blocks of the same: long enough to fill every history."""
    seconds = 7 * Config().spectrogram.block_duration
    start = seconds
    data = np.concatenate([_segment("quiet water", seconds, 5), fish_recording.data])
    res = _run(tmp_path, "rec", data, device)
    score = evaluate(
        res, fish_recording.truth_times + start, fish_recording.truth_freqs
    )
    clean = evaluate(clean_run, fish_recording.truth_times, fish_recording.truth_freqs)
    assert score.recall > clean.recall - 0.01
    assert score.precision > 0.98


def test_results_report_what_happened_over_time(fish_recording, tmp_path):
    parts = [_segment("static", 130.0, 0), fish_recording.data]
    res = _run(tmp_path, "rec", np.concatenate(parts))
    blocks = res.meta["click_fraction_blocks"]
    assert len(blocks) == len(res.meta["noise_std_blocks"]) >= 3
    assert blocks[0] > 0 and blocks[-1] == 0
    assert 0 < res.meta["click_fraction"] < 1


# --- thresholds ---------------------------------------------------------------


def _noise_log_spec(frames, channels, bins=2**14 + 1, seed=0):
    """dB spectrogram (freqs, frames) of Gaussian noise summed over `channels`."""
    g = torch.Generator().manual_seed(seed)
    power = torch.randn(channels, 2, bins, frames, generator=g).square().sum((0, 1))
    return decibel(power)


def _thresholds(frames_per_second=6.0, n_channels=16, **hc):
    cfg = HarmonicGroupsConfig(**hc)
    return DetectionThresholds(cfg, n_channels, 1.0 / frames_per_second)


def test_thresholds_follow_the_noise_within_a_block():
    """Electrodes going into the water in the middle of a block."""
    calm = _noise_log_spec(180, 16)
    wild = _noise_log_spec(180, 1, seed=1)  # common-mode noise: as one channel
    low, high = _thresholds()(torch.cat([calm, wild], dim=1))
    assert low.shape == high.shape == (360,)
    assert np.median(low[:120]) == pytest.approx(6.0 * 1.0, rel=0.15)
    assert np.median(low[240:]) == pytest.approx(6.0 * 4.1, rel=0.15)
    np.testing.assert_allclose(high, low * 10.0 / 6.0, rtol=1e-5)


def test_silent_frames_do_not_lower_the_thresholds():
    log = _noise_log_spec(360, 16)
    full, _ = _thresholds()(log.clone())
    log[:, :180] = -np.inf
    low, _ = _thresholds()(log)
    assert np.isfinite(low).all()
    assert np.median(low[200:]) == pytest.approx(np.median(full[200:]), rel=0.1)
    # and fully silent blocks use what is known, with no NaN
    silent = torch.full((2**14 + 1, 100), -np.inf)
    t = _thresholds()
    t(_noise_log_spec(100, 16))
    assert np.median(t(silent)[0]) == pytest.approx(6.0 * 1.0, rel=0.15)


@pytest.mark.parametrize("channels", [1, 4, 16, 64])
def test_noise_floor_never_binds_on_real_noise(channels):
    """The floor catches a collapsed estimate, it must not limit real noise."""
    t = _thresholds(n_channels=channels)
    low, _ = t(_noise_log_spec(200, channels))
    assert low.min() > 6.0 * min_noise_std(channels)


def test_smooth_spectra_cannot_collapse_the_thresholds():
    """One click in a window: an almost noise-free spectrum, estimate near 0."""
    log = torch.full((2**14 + 1, 100), -60.0) + 0.01 * torch.randn(2**14 + 1, 100)
    low, high = _thresholds(n_channels=16)(log)
    np.testing.assert_allclose(low, 6.0 * min_noise_std(16))
    np.testing.assert_allclose(high, 10.0 * min_noise_std(16))


def test_explicit_thresholds_are_kept():
    low, high = _thresholds(low_threshold=12.0, high_threshold=20.0)(
        _noise_log_spec(50, 16)
    )
    assert (low == 12.0).all() and (high == 20.0).all()
    low, high = _thresholds(low_threshold=12.0)(_noise_log_spec(50, 16))
    assert (low == 12.0).all()
    assert np.median(high) == pytest.approx(10.0, rel=0.15)


def test_thresholds_can_be_frozen_as_in_the_original():
    t = _thresholds(threshold_window=None)
    first, _ = t(_noise_log_spec(100, 16))
    later, _ = t(_noise_log_spec(100, 1, seed=1))
    assert len(np.unique(first)) == 1
    np.testing.assert_array_equal(first[:1], later[:1])


# --- thresholds over frequency ---------------------------------------------------

BINS = 2**14 + 1
BAND_BINS = 128
LAG = 10


def _band_thresholds(n_channels=16, **hc):
    cfg = HarmonicGroupsConfig(**hc)
    return DetectionThresholds(cfg, n_channels, 1.0 / 6.0, BAND_BINS, LAG)


def _common_mode_log_spec(frames, channels=16, up_to=2048, seed=0):
    """Noise that all electrodes share in the lower bins: as noisy as one channel."""
    g = torch.Generator().manual_seed(seed)
    power = torch.randn(channels, 2, BINS, frames, generator=g).square().sum((0, 1))
    common = torch.randn(2, up_to, frames, generator=g).square().sum(0)
    power[:up_to] += 100.0 * common
    return decibel(power)


def test_band_noise_reads_the_frame_noise_on_gaussian_noise():
    log = _noise_log_spec(100, 16)
    bands = band_noise_std(log, LAG, BAND_BINS)
    ratio = bands.nanmedian() / frame_noise_std(log).nanmedian()
    assert 1.0 < ratio < 1.25


def test_band_thresholds_follow_common_mode_noise():
    low, high = _band_thresholds()(_common_mode_log_spec(120))
    assert low.shape == high.shape == (120, BINS // BAND_BINS)
    common = np.median(low[:, : 2048 // BAND_BINS - 1])
    above = np.median(low[:, 2048 // BAND_BINS + 1 :])
    assert above == pytest.approx(6.0 * 1.0, rel=0.15)
    assert common > 2.0 * above


def test_bands_leave_uniform_noise_as_it_is():
    """Recordings without such noise keep their thresholds exactly."""
    log = _noise_log_spec(120, 16)
    plain, _ = _thresholds()(log)
    with_bands, _ = _band_thresholds()(log)
    np.testing.assert_array_equal(
        with_bands, np.broadcast_to(plain[:, None], with_bands.shape)
    )


def test_fish_do_not_raise_the_band_noise():
    """A fish is stable over time, whatever fills the band."""
    log = _noise_log_spec(120, 16)
    reference = band_noise_std(log, LAG, BAND_BINS).nanmedian(1).values
    # 20 stable lines in a band of 128 bins: a dense chorus
    for b in range(4, 124, 6):
        log[256 + b] = 20.0 + 0.3 * torch.randn(120)
    filled = band_noise_std(log, LAG, BAND_BINS).nanmedian(1).values
    band = 256 // BAND_BINS
    assert filled[band] < 1.25 * reference[band]


def test_thresholds_per_band_decide_per_band():
    """A band asking for more than its peak has takes the peak out."""
    freqs = np.arange(BINS) * 0.5
    log = np.full((3, BINS), -100.0)
    # a fish at 300 Hz; the last peak of a spectrum stays unconfirmed
    for harmonic in (1, 2, 3, 4, 5):
        log[:, 600 * harmonic] = -60.0
    cfg = HarmonicGroupsConfig(
        min_good_peak_power=None, mains_freq=0.0, refine_frequency=False
    )
    n_bands = BINS // BAND_BINS
    low, high = np.full((3, n_bands), 6.0), np.full((3, n_bands), 10.0)
    det = detect_harmonic_groups(log, freqs, cfg, low, high, BAND_BINS)
    assert np.array_equal(np.unique(det.frame), [0, 1, 2])
    assert np.all(det.freq == 300.0)
    # in frame 1 the fundamental's band needs a fall of more than the peak's height
    low[1, 600 // BAND_BINS] = 50.0
    det = detect_harmonic_groups(log, freqs, cfg, low, high, BAND_BINS)
    assert np.array_equal(np.unique(det.frame[det.freq == 300.0]), [0, 2])


def _common_mode_recording(seconds, rate=RATE, channels=CHANNELS, seed=7):
    """Quiet independent noise, and much stronger noise below 1 kHz on all channels,
    as electrodes in water have it: the high band does not show it."""
    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    data = 0.01 * rng.standard_normal((n, channels))
    spectrum = np.fft.rfft(rng.standard_normal(n))
    spectrum[np.fft.rfftfreq(n, 1.0 / rate) > 1000.0] = 0.0
    return (data + np.fft.irfft(spectrum, n)[:, None]).astype(np.float32)


def test_common_mode_noise_makes_no_false_fish(tmp_path):
    """False fish in the water, with no click anywhere and no fish."""
    wav = tmp_path / "rec.wav"
    data = _common_mode_recording(60.0)
    write_audio(str(wav), 0.9 * data / np.abs(data).max(), RATE)
    cfg = Config()
    cfg.harmonic_groups.min_good_peak_power = None
    cfg.harmonic_groups.threshold_band = None
    without = detect(wav, tmp_path / "a", cfg, device="cpu").results
    cfg.harmonic_groups.threshold_band = 100.0
    with_bands = detect(wav, tmp_path / "b", cfg, device="cpu").results
    frames = len(with_bands.times)
    assert len(without.fund_v) > frames  # the case is a real one
    assert len(with_bands.fund_v) < 0.05 * len(without.fund_v)


# --- tracking -----------------------------------------------------------------

DT = 0.164


def _two_fish_crossing(frame_offset, n=3000, separation=0.6):
    """Two fish circling each other within the frequency tolerance, with distinct
    electrode patterns: only the amplitude error can tell them apart."""
    rng = np.random.default_rng(0)
    frames = np.arange(n)
    idx = np.repeat(frames, 2) + frame_offset
    a = 700.0 + separation * np.sin(frames / 40.0)
    b = 700.0 - separation * np.sin(frames / 40.0)
    fund = np.column_stack([a, b]).ravel() + rng.normal(0, 0.02, 2 * n)
    sign = np.empty((2 * n, 4))
    sign[::2] = np.array([1.0, 0.5, 0.1, 0.05]) * rng.uniform(0.9, 1.1, (n, 4))
    sign[1::2] = np.array([0.05, 0.1, 0.5, 1.0]) * rng.uniform(0.9, 1.1, (n, 4))
    return fund, idx, sign


def _agreement(ident):
    """Smallest fraction of one fish's detections carrying its main identity."""
    fractions = []
    for fish in (ident[::2], ident[1::2]):
        tracked = fish[~np.isnan(fish)]
        fractions.append(np.unique(tracked, return_counts=True)[1].max() / len(fish))
    return min(fractions)


STRAYS = {
    "one": ([100], [610.0]),
    "far apart": ([100, 5000], [610.0, 900.0]),
    "in one window": ([100, 110], [610.0, 611.0]),
    "several": (
        [100, 110, 120, 400, 800, 900],
        [610.0, 611.0, 610.5, 900.0, 1100.0, 1100.5],
    ),
}


def _track_with_strays(strays, **kw):
    offset = 20000  # about an hour before the fish
    times = np.arange(offset + 3100) * DT
    fund, idx, sign = _two_fish_crossing(offset)
    stray_idx, stray_f = (np.array(v) for v in STRAYS[strays])
    stray_s = np.random.default_rng(1).random((len(stray_f), 4))
    ident = track(
        np.concatenate([stray_f, fund]),
        np.concatenate([stray_idx, idx]),
        np.concatenate([stray_s, sign]),
        times,
        TrackingConfig(),
        **kw,
    )
    return ident[len(stray_f) :]


@pytest.mark.parametrize("strays", STRAYS)
def test_stray_detections_long_before_the_fish_do_not_change_identities(strays):
    """A stray detection without a partner used to leave the amplitude error
    distribution empty: for the whole recording only the frequency linked
    detections, fish swapped identities at every crossing (agreement 0.49)."""
    ident = _track_with_strays(strays)
    assert _agreement(ident) > 0.99
    assert len(np.unique(ident[~np.isnan(ident)])) == 2


def test_v6_compat_keeps_the_distribution_from_the_first_detection():
    """The verification of the port against the original relies on this, down
    to the original's failure on a single stray detection."""
    assert _agreement(_track_with_strays("one", v6_compat=True)) < 0.6


def test_error_distribution_of_a_recording_starting_with_fish_is_unchanged():
    """Only a nearly empty first window is passed over."""
    from wavetracker.tracking import _error_distribution_start

    _, idx, _ = _two_fish_crossing(0)
    frame_start = np.searchsorted(idx, np.arange(idx[-1] + 4 * 60 + 3))
    assert _error_distribution_start(frame_start, idx[0], idx[-1], 60) == idx[0]
