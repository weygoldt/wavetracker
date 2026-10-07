import json

import numpy as np
import pytest
import torch
from typer.testing import CliRunner

from wavetracker.cli import app
from wavetracker.config import Config
from wavetracker.evaluation import evaluate
from wavetracker.pipeline import detect, track_results
from wavetracker.results import Results

from .conftest import REAL_RECORDING

runner = CliRunner()
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_synthetic_end_to_end(synthetic_wav, tmp_path, device):
    path, truth = synthetic_wav
    cfg = Config()
    cfg.output.save_fine_spec = True
    out = detect(path, tmp_path, cfg, device=device)
    track_results(out.results, cfg)
    score = evaluate(out.results, truth["times"], truth["freqs"])
    assert score.precision > 0.98
    assert score.recall > 0.95
    assert score.purity > 0.98
    assert all(f.freq_error < 0.2 for f in score.fish)

    r = out.results
    assert r.cplx_v is not None and r.cplx_v.shape == r.sign_v.shape
    # |cplx|^2 is the stored power (no interference in synthetic data)
    np.testing.assert_allclose(np.abs(r.cplx_v) ** 2, r.sign_v, rtol=1e-3)
    assert Results.load(tmp_path).cplx_v.dtype == np.complex64

    fine = np.load(tmp_path / "fine_spec.npy", mmap_mode="r")
    assert fine.shape == (
        len(out.results.times),
        len(np.load(tmp_path / "fine_freqs.npy")),
    )
    assert (tmp_path / "sparse_spectra.npy").exists()


def test_time_window(synthetic_wav, tmp_path):
    path, _ = synthetic_wav
    res = detect(
        path, tmp_path, Config(), start=10.0, duration=20.0, device="cpu"
    ).results
    assert res.times[0] >= 10.0 and res.times[-1] <= 30.0
    assert res.meta["start"] == 10.0


def test_cli(synthetic_wav, tmp_path):
    path, _ = synthetic_wav
    out = tmp_path / "out"
    r = runner.invoke(
        app, ["run", str(path), "-o", str(out), "--duration", "20", "--device", "cpu"]
    )
    assert r.exit_code == 0, r.output
    meta = json.loads((out / "wavetracker.json").read_text())
    assert meta["duration"] == 20.0
    n_ids = Results.load(out).n_ids

    r = runner.invoke(app, ["run", str(path), "-o", str(out)])
    assert "Skipping" in r.output

    cfg = tmp_path / "cfg.yaml"
    assert runner.invoke(app, ["config", str(cfg)]).exit_code == 0
    r = runner.invoke(app, ["track", str(out), "-c", str(cfg)])
    assert r.exit_code == 0, r.output
    assert Results.load(out).n_ids == n_ids

    assert runner.invoke(app, ["summary", str(out)]).exit_code == 0
    assert (
        runner.invoke(app, ["plot", str(out), "-o", str(tmp_path / "p.png")]).exit_code
        == 0
    )
    assert (tmp_path / "p.png").exists()


@pytest.mark.data
@pytest.mark.skipif(not REAL_RECORDING.exists(), reason="recordings not available")
def test_real_recording_two_fish(tmp_path):
    """The tube-competition recordings contain exactly two fish."""
    out = detect(REAL_RECORDING, tmp_path, Config(), start=3600, duration=120)
    track_results(out.results, Config())
    r = out.results
    sizes = sorted(((r.ident_v == i).sum() for i in r.ids()), reverse=True)
    frames = len(r.times)
    # two dominant identities, each present most of the time
    assert sizes[1] > 0.3 * frames
    assert sum(sizes[:2]) > 0.8 * len(r.fund_v)


def test_exclude_channels(synthetic_wav, tmp_path):
    path, _ = synthetic_wav
    cfg = Config()
    cfg.spectrogram.exclude_channels = [0, 3]
    res = detect(path, tmp_path, cfg, duration=10.0, device="cpu").results
    assert res.sign_v.shape[1] == 4
    assert res.meta["channels"] == [1, 2, 4, 5]


def test_stored_spectrograms_follow_tracked_range(synthetic_wav, tmp_path):
    """No fixed frequency limit: by default the stored spectrograms cover
    1.25 x harmonic_groups.max_freq; an explicit limit is respected."""
    path, _ = synthetic_wav
    cfg = Config()
    cfg.output.save_fine_spec = True
    cfg.harmonic_groups.max_freq = 1600.0
    detect(path, tmp_path / "auto", cfg, duration=10.0, device="cpu")
    for name in ("sparse_freq", "fine_freqs"):
        f = np.load(tmp_path / "auto" / f"{name}.npy")
        assert 1950 < f.max() <= 2000.0 + 2  # 1.25 * 1600
    cfg.output.sparse_spec_max_freq = 900.0
    cfg.output.fine_spec_max_freq = 900.0
    detect(path, tmp_path / "explicit", cfg, duration=10.0, device="cpu")
    assert np.load(tmp_path / "explicit" / "fine_freqs.npy").max() <= 900.0
    assert np.load(tmp_path / "explicit" / "sparse_freq.npy").max() <= 900.0


def test_cli_reminds_to_set_frequency_range(synthetic_wav, tmp_path):
    path, _ = synthetic_wav
    args = ["run", str(path), "--duration", "5", "--device", "cpu", "--no-track"]
    r = runner.invoke(app, [*args, "-o", str(tmp_path / "a")])
    assert r.exit_code == 0 and "default fish range" in r.output
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("harmonic_groups:\n  min_freq: 400.0\n  max_freq: 1200.0\n")
    r = runner.invoke(app, [*args, "-o", str(tmp_path / "b"), "-c", str(cfg)])
    assert r.exit_code == 0 and "default fish range" not in r.output


def test_warns_when_absolute_power_limit_binds(synthetic_wav, tmp_path, caplog):
    path, _ = synthetic_wav
    cfg = Config()
    cfg.harmonic_groups.min_good_peak_power = 0.0
    with caplog.at_level("WARNING", logger="wavetracker"):
        detect(path, tmp_path, cfg, duration=5.0, device="cpu")
    assert "min_good_peak_power" in caplog.text
