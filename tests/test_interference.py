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
