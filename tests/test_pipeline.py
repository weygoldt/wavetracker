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
