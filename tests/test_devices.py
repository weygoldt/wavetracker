import numpy as np
import pytest
import torch

from wavetracker import pipeline
from wavetracker.config import Config
from wavetracker.results import Results
from wavetracker.spectrogram import available, get_device, out_of_memory


def test_auto_is_gpu_first():
    best = get_device("auto")
    if torch.cuda.is_available():
        assert best.type == "cuda"
    assert get_device("cpu") == torch.device("cpu")
    assert get_device("auto", prefer="cpu") == torch.device("cpu")
    # a device this machine does not have is not preferred
    assert not available("cuda:99")
    assert get_device("auto", prefer="cuda:99") == best


def test_out_of_memory_is_recognised():
    assert out_of_memory(torch.OutOfMemoryError("CUDA out of memory"))
    assert out_of_memory(RuntimeError("MPS backend out of memory"))
    assert not out_of_memory(RuntimeError("shape mismatch"))
    assert not out_of_memory(ValueError("out of memory"))


def test_detect_falls_back_to_the_cpu(monkeypatch, caplog, tmp_path):
    used = []

    def fake(input_path, output_dir, cfg, start, duration, dev, progress):
        used.append(dev.type)
        if dev.type != "cpu":
            raise torch.OutOfMemoryError("CUDA out of memory (test)")
        return "done"

    monkeypatch.setattr(pipeline, "_detect", fake)
    with caplog.at_level("WARNING"):
        assert pipeline.detect("x", tmp_path, Config(), device="cuda") == "done"
    assert used == ["cuda", "cpu"]
    assert "detecting on the CPU" in caplog.text


def test_detect_does_not_hide_other_errors(monkeypatch, tmp_path):
    def fake(*args):
        raise RuntimeError("something else")

    monkeypatch.setattr(pipeline, "_detect", fake)
    with pytest.raises(RuntimeError, match="something else"):
        pipeline.detect("x", tmp_path, Config(), device="cuda")


@pytest.mark.parametrize("meta", [{"device": "cpu"}, {"device": "cuda:99"}, {}])
def test_association_follows_the_detections_device(monkeypatch, meta):
    seen = []

    def fake_identities(results, cfg, trace):
        seen.append(cfg.tracking.assoc_device)
        results.ident_v = np.full(len(results.fund_v), np.nan)

    monkeypatch.setattr(pipeline, "_identities", fake_identities)
    n = 4
    res = Results(
        fund_v=np.full(n, 600.0),
        idx_v=np.arange(n),
        sign_v=np.ones((n, 2)),
        ident_v=np.full(n, np.nan),
        times=np.arange(n) * 0.164,
        meta=dict(meta),
        sal_v=np.ones(n),
    )
    cfg = Config()
    cfg.review.enabled = False
    pipeline.track_results(res, cfg)
    want = "cpu" if meta.get("device") == "cpu" else str(get_device("auto"))
    assert seen == [want]
    assert cfg.tracking.assoc_device == "auto"  # the caller's config is unchanged
