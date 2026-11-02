"""Removal of impulsive noise ("static") from the raw traces."""

import numpy as np
import pytest
import torch

from wavetracker.clicks import find_clicks, remove_clicks
from wavetracker.config import ClickConfig
from wavetracker.spectrogram import PowerSpectrogram, decibel, frequencies, step_size
from wavetracker.synthetic import add_clicks

RATE = 20000.0
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
NOISE = 1e-3


def _noise(seconds, channels=2, seed=0):
    rng = np.random.default_rng(seed)
    n = int(seconds * RATE)
    return (NOISE * rng.standard_normal((channels, n))).astype(np.float32)


def _fish(seconds, freq=653.0, amplitude=1.0, seed=1):
    """A drifting fish with harmonics, as real fish are: never a pure tone."""
    rng = np.random.default_rng(seed)
    n = int(seconds * RATE)
    walk = np.cumsum(rng.standard_normal(n // 1000 + 2))
    walk = (walk - walk.mean()) / max(walk.std(), 1e-9) * 0.5
    f = freq + np.interp(np.arange(n), np.arange(len(walk)) * 1000, walk)
    phase = 2 * np.pi * np.cumsum(f) / RATE
    wave = sum(r * np.sin((h + 1) * phase) for h, r in enumerate((1.0, 0.5, 0.3, 0.2)))
    return (amplitude * wave).astype(np.float32)


def _static(data, per_second=10.0, amplitude=1000 * NOISE, seed=2):
    """Clicks on the (channels, samples) array `data`."""
    return add_clicks(
        data.T,
        RATE,
        per_second=per_second,
        amplitude=amplitude,
        rng=np.random.default_rng(seed),
    ).T


def _repair(x, device="cpu", **kw):
    t = torch.from_numpy(x).to(device)
    out, fraction = remove_clicks(t, RATE, ClickConfig(**kw))
    return out.cpu().numpy(), fraction.cpu().numpy()


def _kurtosis(x):
    return np.mean(x**4, axis=-1) / np.mean(x**2, axis=-1) ** 2


def _log_spectrum(x, nfft=2**13):
    spec = PowerSpectrogram(nfft, step_size(nfft, 0.5), RATE, torch.device("cpu"))
    return decibel(spec(torch.from_numpy(x)).sum(0)).numpy(), frequencies(nfft, RATE)


@pytest.mark.parametrize("device", DEVICES)
def test_clean_recording_is_returned_untouched(device):
    x = _noise(20) + _fish(20)
    out, fraction = _repair(x, device)
    np.testing.assert_array_equal(out, x)
    assert fraction.tolist() == [0.0, 0.0]


@pytest.mark.parametrize("device", DEVICES)
def test_clicks_are_removed_down_to_the_noise(device):
    noise = _noise(20)
    assert _kurtosis(_static(noise)).min() > 100  # real static: 2000 to 600000
    out, fraction = _repair(_static(noise), device)
    # the very ends of a block are not repaired, see `ClickConfig.edge`
    np.testing.assert_allclose(_kurtosis(out[..., 200:-200]), 3.0, atol=1.0)
    np.testing.assert_allclose(out.std(-1), NOISE, rtol=0.1)
    assert fraction.min() > 0 and fraction.max() < 0.15


def test_channels_without_clicks_stay_exactly_as_they_are():
    x = _noise(20)
    x[0] = _static(x)[0]
    out, fraction = _repair(x)
    np.testing.assert_array_equal(out[1], x[1])
    assert fraction[0] > 0 and fraction[1] == 0


@pytest.mark.parametrize("level", [1e3, 1e4])
def test_repair_cuts_no_holes_into_a_strong_fish(level):
    """A hole in a loud sine is itself a click of the fish's amplitude: it
    raised false detections up to 100-fold before repairs left the fish alone."""
    noise = _noise(20)
    fish = _fish(20, amplitude=level * NOISE)
    reference, _ = _log_spectrum(noise + fish)
    out, _ = _repair(_static(noise + fish))
    repaired, freqs = _log_spectrum(out)
    away = np.abs(freqs % 653.0 - 653.0 / 2) < 200
    # the spectrum away from the fish is as low as without any clicks
    assert np.median(repaired[away]) < np.median(reference[away]) + 3.0
    # and the fish itself keeps its power
    peak = np.argmin(np.abs(freqs - 653.0))
    band = slice(peak - 8, peak + 9)
    np.testing.assert_allclose(
        repaired[band].max(0).mean(), reference[band].max(0).mean(), atol=0.5
    )


def test_fish_passing_an_electrode_is_not_a_click():
    """Seconds long and far above the noise of the rest of the block."""
    x = _noise(30)
    n = int(3 * RATE)
    i = int(10 * RATE)
    env = np.hanning(n).astype(np.float32)
    x[:, i : i + n] += _fish(3, amplitude=3000 * NOISE)[:n] * env
    out, fraction = _repair(x)
    np.testing.assert_array_equal(out, x)
    assert fraction.max() == 0


def test_long_broadband_events_are_not_clicks():
    """A boat or rain for some seconds is a real broadband signal."""
    x = _noise(30)
    i = int(10 * RATE)
    x[:, i : i + int(2 * RATE)] *= 300
    assert not find_clicks(torch.from_numpy(x), RATE, ClickConfig()).any()


def test_digital_silence_is_handled():
    x = np.zeros((2, int(20 * RATE)), np.float32)
    out, fraction = _repair(x)
    np.testing.assert_array_equal(out, x)
    assert fraction.max() == 0
    # half of the block silent: the noise of the other half counts
    y = _noise(20)
    y[:, : y.shape[1] // 2] = 0
    out, fraction = _repair(y)
    np.testing.assert_array_equal(out, y)
    assert fraction.max() == 0


def test_short_blocks_work():
    x = _static(_noise(2))
    out, _ = _repair(x)
    assert out.shape == x.shape and np.isfinite(out).all()
    assert _kurtosis(out).max() < 5


def _chirping_fish(seconds, amplitude, rate=RATE):
    """A fish with chirps: 100 ms excursions of 100 Hz, as loud as the fish."""
    n = int(seconds * rate)
    t = np.arange(n) / rate
    f = np.full(n, 653.0)
    for t0 in np.arange(1.0, seconds - 1.0, 1.7):
        f += 100.0 * np.exp(-(((t - t0) / 0.05) ** 2))
    phase = 2 * np.pi * np.cumsum(f) / rate
    wave = sum(r * np.sin((h + 1) * phase) for h, r in enumerate((1.0, 0.5, 0.3, 0.2)))
    return (amplitude * wave).astype(np.float32)


@pytest.mark.parametrize("level", [10, 100, 1000, 10000])
def test_chirps_are_not_clicks(level):
    """Short and loud as clicks are, but they are the signal under study."""
    x = _noise(30) + _chirping_fish(30, level * NOISE)
    out, fraction = _repair(x)
    assert fraction.max() == 0
    np.testing.assert_array_equal(out, x)


@pytest.mark.parametrize("level", [100, 10000])
def test_chirps_survive_static(level):
    noise = _noise(30)
    fish = _chirping_fish(30, level * NOISE)
    x = _static(noise, amplitude=max(1000, 10 * level) * NOISE) + fish
    out, _ = _repair(x)
    # repaired samples aside, a chirp is unchanged: compare the band it sweeps
    nfft = 2**11  # 100 ms windows resolve a chirp
    spec = PowerSpectrogram(nfft, nfft // 2, RATE, torch.device("cpu"))
    freqs = frequencies(nfft, RATE)
    band = (freqs > 700) & (freqs < 760)
    t0 = 1.0
    frame = round((t0 * RATE - nfft / 2) / (nfft // 2))
    clean = decibel(spec(torch.from_numpy(noise + fish))[0])[band, frame]
    repaired = decibel(spec(torch.from_numpy(out))[0])[band, frame]
    assert clean.max() - repaired.max() < 3.0


def test_no_ghost_of_a_fish_before_it_arrives():
    """A fish in the last part of a block is spread over all of it by the
    spectral split: repairs must not write it into the time before it came."""
    seconds = 40
    arrival = int(25 * RATE)
    noise = _noise(seconds)
    fish = _fish(seconds, amplitude=100 * NOISE)
    fish[:arrival] = 0
    x = _static(noise + fish, amplitude=1000 * NOISE)
    out, fraction = _repair(x)
    assert fraction.min() > 0.01

    before = out[:, : arrival - int(2 * RATE)]
    reference = noise[:, : arrival - int(2 * RATE)]
    power, freqs = _log_spectrum(before)
    expected, _ = _log_spectrum(reference)
    for harmonic in (653.0, 1306.0):
        band = np.abs(freqs - harmonic) < 8
        assert power[band].max() < expected[band].max() + 6.0
    # after its arrival the fish is as loud as without any static
    after, _ = _log_spectrum(out[:, arrival + int(2 * RATE) :])
    target, _ = _log_spectrum((noise + fish)[:, arrival + int(2 * RATE) :])
    band = np.abs(freqs - 653.0) < 8
    assert np.mean(after[band].max(0)) == pytest.approx(
        np.mean(target[band].max(0)), abs=0.5
    )
