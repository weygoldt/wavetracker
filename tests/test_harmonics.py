import numpy as np
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
    rng = np.random.default_rng(0)
    t = np.arange(int(seconds * RATE)) / RATE
    x = noise * rng.standard_normal(len(t))
    for f in freqs_hz:
        for h, a in enumerate((1.0, 0.5, 0.3, 0.2), start=1):
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
