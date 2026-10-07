import numpy as np
import pytest

from wavetracker.config import Config
from wavetracker.evaluation import evaluate, match_detections
from wavetracker.interference import find_combs
from wavetracker.pipeline import detect
from wavetracker.synthetic import Fish, Hum, save_recording, synthesize

KW = {"min_spacing": 20.0, "max_spacing": 300.0, "tol": 0.3, "min_run": 4}


def test_finds_hum_comb_among_fish_lines():
    hum = 95.43 * np.arange(4, 20)
    fish = np.concatenate([523.7 * np.arange(1, 6), 807.9 * np.arange(1, 4)])
    combs = find_combs(np.sort(np.concatenate([hum, fish])), **KW)
    assert len(combs) == 1
    spacing, idx, run = combs[0]
    assert spacing == pytest.approx(95.43, abs=0.01)
    assert len(idx) == len(hum) and run == len(hum)


@pytest.mark.parametrize("f0", [400.0, 523.7, 623.3, 1187.9])
def test_harmonic_series_of_a_fish_is_not_a_comb(f0):
    assert find_combs(f0 * np.arange(1, 16), **KW) == []


def test_several_resting_fish_are_not_a_comb():
    rng = np.random.default_rng(0)
    for _ in range(50):
        f0 = rng.uniform(400, 1200, size=4)
        lines = np.sort((f0[:, None] * np.arange(1, 8)[None, :]).ravel())
        assert find_combs(lines, **KW) == []


def test_two_combs():
    lines = np.sort(np.concatenate([55.55 * np.arange(2, 9), 95.43 * np.arange(4, 12)]))
    spacings = sorted(c[0] for c in find_combs(lines, **KW))
    assert spacings == pytest.approx([55.55, 95.43], abs=0.02)


@pytest.mark.slow
def test_hum_removed_resting_fish_on_single_electrode_kept(tmp_path):
    """A resting fish seen on one electrode only, on the electrode where the
    hum is strongest, must survive while the hum is removed."""
    channels = 6
    gains = [0.3] * channels
    gains[4] = 1.0
    fish = [
        Fish(623.3, drift=0.02, amplitude=0.3, position=0.8, movement=0.0, spread=0.03),
        Fish(512.7, drift=2.0, position=0.2),
    ]
    rng = np.random.default_rng(1)
    rec = synthesize(
        fish,
        240.0,
        channels=channels,
        rng=rng,
        hum=[Hum(95.4, (4, 30), amplitude=0.15, channel_gains=gains)],
    )
    path = tmp_path / "hum.wav"
    save_recording(rec, path)

    cfg = Config()
    # species range: lets the filter remove combs up to 300 Hz spacing
    cfg.harmonic_groups.min_freq, cfg.harmonic_groups.max_freq = 400.0, 1200.0
    results = {}
    for enabled in (False, True):
        cfg.interference.enabled = enabled
        results[enabled] = detect(path, tmp_path / str(enabled), cfg).results

    def hum_detections(r):
        m = match_detections(r, rec.truth_times, rec.truth_freqs)
        return int((m < 0).sum())

    assert hum_detections(results[False]) > len(
        results[False].times
    )  # hum is a problem
    assert hum_detections(results[True]) < 0.01 * len(results[True].times)
    score = evaluate(results[True], rec.truth_times, rec.truth_freqs)
    assert score.fish[0].recall > 0.98  # resting single-electrode fish
    assert score.fish[1].recall > 0.98
    assert results[True].meta["interference_combs"]


def test_merge_tooth_neighbours():
    from wavetracker.interference import merge_tooth_neighbours

    teeth = np.array([763.9, 859.4])
    frame = np.array([0, 0, 1, 1, 2, 3, 3])
    freq = np.array([762.2, 764.6, 763.9, 761.5, 763.95, 700.0, 702.0])
    power = np.array([-60.0, -70.0, -80.0, -60.0, -65.0, -60.0, -61.0])
    keep = merge_tooth_neighbours(frame, freq, power, teeth, 4.0, 0.4)
    # frame 0: split around the tooth -> keep stronger; frame 1: tooth next to
    # fish -> drop the tooth; frame 2: lone detection on a tooth -> kept;
    # frame 3: two fish without a tooth between them -> both kept
    assert keep.tolist() == [True, False, False, True, True, True, True]


def test_tooth_lifted_by_nearby_fish_is_dropped():
    from wavetracker.interference import merge_tooth_neighbours

    teeth = np.array([668.4])
    frame = np.array([0, 0, 1, 1])
    freq = np.array([662.5, 668.4, 662.5, 668.4])
    power = np.array([-50.0, -80.0, -90.0, -80.0])
    keep = merge_tooth_neighbours(frame, freq, power, teeth, 4.0, 0.4, tooth_sep=8.0)
    # frame 0: strong fish 6 Hz away -> tooth dropped; frame 1: the
    # non-tooth detection is weaker -> both kept
    assert keep.tolist() == [True, False, True, True]


def test_line_search_reaches_nyquist_by_default():
    import torch

    from wavetracker.config import InterferenceConfig
    from wavetracker.interference import CombCanceller

    rate, nfft = 48000.0, 4096
    freqs = np.fft.rfftfreq(nfft, 1 / rate)
    t = np.arange(int(4 * rate)) / rate
    # a comb only above 10 kHz: 12 teeth of a 120 Hz comb at 15-16.4 kHz
    x = sum(np.sin(2 * np.pi * 120 * k * t) for k in range(125, 137))
    x = x + 0.01 * np.random.default_rng(0).standard_normal(len(t))
    from wavetracker.spectrogram import PowerSpectrogram

    power = PowerSpectrogram(nfft, 410, rate, torch.device("cpu"))(
        x[None].astype(np.float32)
    )
    cc = CombCanceller(InterferenceConfig(search_max_freq=20000.0), freqs)
    cc(power)
    assert any(abs(c.spacing - 120) < 0.5 for c in cc.combs)


@pytest.mark.slow
def test_resting_low_frequency_fish_is_not_a_comb_by_default(tmp_path):
    """A resting 103.7 Hz fish has harmonics 103.7, 207.4, ... - a comb.
    With the default fish range (from 80 Hz) the comb spacing limit is 72 Hz,
    so the fish survives; the old fixed 300 Hz limit removed it."""
    from wavetracker.synthetic import Fish, save_recording, synthesize

    rng = np.random.default_rng(3)
    fish = [Fish(103.7, drift=0.02, amplitude=0.5, position=0.5, movement=0.0)]
    rec = synthesize(fish, 240.0, channels=4, rng=rng, mains=0.0)
    path = tmp_path / "low.wav"
    save_recording(rec, path)
    recall = {}
    for max_spacing in (None, 300.0):
        cfg = Config()
        cfg.interference.max_spacing = max_spacing
        r = detect(path, tmp_path / str(max_spacing), cfg).results
        recall[max_spacing] = (
            evaluate(r, rec.truth_times, rec.truth_freqs).fish[0].recall
        )
    assert recall[None] > 0.95
    assert recall[300.0] < 0.5
