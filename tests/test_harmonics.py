import numpy as np
import pytest
import torch

from wavetracker.config import HarmonicGroupsConfig
from wavetracker.harmonics import detect_harmonic_groups
from wavetracker.spectrogram import (
    PowerSpectrogram,
    decibel,
    estimate_noise_std,
    frequencies,
)

RATE = 20000.0
NFFT = 2**15


def _spectrum(freqs_hz, seconds=3.0, noise=0.01, mains=0.0):
    """Fish at `freqs_hz`; an item may be (frequency, harmonic amplitudes)."""
    rng = np.random.default_rng(0)
    t = np.arange(int(seconds * RATE)) / RATE
    x = noise * rng.standard_normal(len(t))
    for f in freqs_hz:
        f, amps = f if isinstance(f, tuple) else (f, (1.0, 0.5, 0.3, 0.2))
        for h, a in enumerate(amps, start=1):
            x += a * np.sin(2 * np.pi * h * f * t)
    for h in range(1, 10):
        x += mains * np.sin(2 * np.pi * 50 * h * t)
    power = PowerSpectrogram(NFFT, 3276, RATE, torch.device("cpu"))(
        x[None].astype(np.float32)
    )[0]
    log = decibel(power)
    std = estimate_noise_std(log)
    return log.T.contiguous().numpy(), std


def test_detects_fish_and_ignores_mains():
    cfg = HarmonicGroupsConfig()
    log, std = _spectrum([523.3, 811.7], mains=0.3)
    det = detect_harmonic_groups(
        log,
        frequencies(NFFT, RATE),
        cfg,
        cfg.low_thresh_factor * std,
        cfg.high_thresh_factor * std,
    )
    for frame in np.unique(det.frame):
        f = np.sort(det.freq[det.frame == frame])
        np.testing.assert_allclose(f, [523.3, 811.7], atol=0.1)
    assert len(np.unique(det.frame)) == log.shape[0]


def test_harmonic_is_not_a_second_fish():
    cfg = HarmonicGroupsConfig()
    log, std = _spectrum([450.0])  # 2nd harmonic at 900 Hz is inside the range
    det = detect_harmonic_groups(
        log,
        frequencies(NFFT, RATE),
        cfg,
        cfg.low_thresh_factor * std,
        cfg.high_thresh_factor * std,
    )
    np.testing.assert_allclose(det.freq, 450.0, atol=0.1)


def test_frequency_range():
    cfg = HarmonicGroupsConfig(min_freq=600.0, max_freq=1000.0)
    log, std = _spectrum([523.3, 811.7])
    det = detect_harmonic_groups(
        log,
        frequencies(NFFT, RATE),
        cfg,
        cfg.low_thresh_factor * std,
        cfg.high_thresh_factor * std,
    )
    np.testing.assert_allclose(det.freq, 811.7, atol=0.1)


def test_max_harmonics_caps_wide_ranges():
    from wavetracker.harmonics import n_harmonics

    # original formula for the original range: max_freq * group // min_freq - 1
    assert n_harmonics(HarmonicGroupsConfig(min_freq=400.0, max_freq=1200.0)) == 8
    # default range 80-2400 Hz would need 89; capped at 10
    assert n_harmonics(HarmonicGroupsConfig()) == 10
    wide = HarmonicGroupsConfig(min_freq=20.0, max_freq=2000.0, min_group_size=2)
    wide.max_harmonics = None
    assert n_harmonics(wide) == 199


ODD = (1.0, 0.0, 0.5, 0.0, 0.3)  # odd-harmonic waveform: no 2nd harmonic
LOW = 163.7  # 3rd harmonic 491.1 Hz, clear of 50 Hz mains harmonics


def _detect(fish, **kw):
    cfg = HarmonicGroupsConfig(min_group_size=2, **kw)
    log, std = _spectrum(fish)
    det = detect_harmonic_groups(
        log,
        frequencies(NFFT, RATE),
        cfg,
        cfg.low_thresh_factor * std,
        cfg.high_thresh_factor * std,
    )
    return det, log.shape[0]


def test_missing_second_harmonic_needs_option():
    fish = [(LOW, ODD), 811.7]
    det, n_frames = _detect(fish)
    assert not np.any(np.abs(det.freq - LOW) < 1.0)

    det, n_frames = _detect(fish, max_missing_harmonics=1)
    # (min_group_size 2 also yields a few noise pairs in this clean spectrum)
    for frame in range(n_frames):
        f = det.freq[det.frame == frame]
        for target in (LOW, 811.7):
            assert np.sum(np.abs(f - target) < 0.1) == 1


@pytest.mark.parametrize("amps", [(1.0, 0.5, 0.3, 0.2), ODD])
def test_no_subharmonic_ghost(amps):
    # a weak pure tone at f/3 and the fish's own peak at f form a group with
    # harmonics 1 and 3 present; the fish has to keep its peak
    fish = [(3 * LOW, amps), (LOW, (0.05,))]
    det, n_frames = _detect(fish, max_missing_harmonics=1)
    assert not np.any(np.abs(det.freq - LOW) < 1.0)
    assert np.sum(np.abs(det.freq - 3 * LOW) < 0.1) == n_frames


def test_missing_fundamental_is_not_a_fish():
    # harmonics 3, 5, 7 of LOW without the fundamental
    det, _ = _detect(
        [(LOW, (0.0, 0.0, 1.0, 0.0, 0.6, 0.0, 0.4))], max_missing_harmonics=1
    )
    assert not np.any(np.abs(det.freq - LOW) < 1.0)


def test_group_window_fits_into_collected_harmonics():
    from wavetracker.harmonics import n_harmonics

    cfg = HarmonicGroupsConfig(
        min_group_size=3, max_missing_harmonics=2, max_harmonics=3
    )
    assert n_harmonics(cfg) == 5


@pytest.mark.parametrize("tone", [6 * LOW, LOW / 2])
def test_fundamental_keeps_its_peaks(tone):
    # a weak tone at 6f makes the fish's 3rd harmonic a complete group
    # (3f, 6f); one at f/2 makes a complete group (f/2, f). Both take the
    # fish's peaks without gaps allowed, but not with them.
    fish = [(LOW, ODD), (tone, (0.05,))]
    det, n_frames = _detect(fish)
    assert np.sum(np.abs(det.freq - LOW) < 0.1) < n_frames

    det, n_frames = _detect(fish, max_missing_harmonics=1)
    assert np.sum(np.abs(det.freq - LOW) < 0.1) == n_frames
    assert not np.any(np.abs(det.freq - 3 * LOW) < 1.0)
    assert not np.any(np.abs(det.freq - LOW / 2) < 1.0)


def test_peak_of_another_fish_counts_as_gap():
    # a stronger fish at 2f claims the odd fish's 2nd-harmonic slot
    fish = [(LOW, ODD), (2 * LOW, (2.0, 1.0, 0.6))]
    det, n_frames = _detect(fish, max_missing_harmonics=1)
    for f in (LOW, 2 * LOW):
        assert np.sum(np.abs(det.freq - f) < 0.1) == n_frames
    det, n_frames = _detect(fish, max_missing_harmonics=1, exclusive_harmonics="all")
    assert not np.any(np.abs(det.freq - LOW) < 0.1)
