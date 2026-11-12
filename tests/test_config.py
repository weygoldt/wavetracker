import json
from pathlib import Path

import pytest
import yaml

from wavetracker.config import MODELS_DIR, Config, model_file

BENCH = Path(__file__).parents[1] / "benchmarks" / "dense" / "configs"


def test_yaml_roundtrip(tmp_path):
    cfg = Config.classic()
    cfg.tracking.max_dt = 5.0
    cfg.harmonic_groups.low_threshold = 12.0
    cfg.save(tmp_path / "cfg.yaml")
    loaded = Config.load(tmp_path / "cfg.yaml")
    assert loaded == cfg


def test_partial_config_uses_defaults():
    cfg = Config.from_dict({"tracking": {"freq_tolerance": 1.0}})
    assert cfg.tracking.freq_tolerance == 1.0
    assert cfg.tracking.max_dt == Config().tracking.max_dt
    assert cfg.tracking.method == "learned"  # the default pipeline


def _without_models(d):
    d = json.loads(json.dumps(d))
    d["detector"]["model"] = d["tracking"]["assoc_model"] = None
    d["linking"]["model"] = d["review"]["model"] = None
    return d


def test_default_is_the_learned_pipeline():
    cfg = Config()
    assert cfg == Config.load() == Config.from_dict({}) == Config.from_dict(None)
    assert cfg.detector.method == "learned" and cfg.tracking.method == "learned"
    assert cfg.rises.enabled and cfg.review.enabled and cfg.linking.enabled
    # the benchmarked pipeline (docs/benchmarks.md), apart from where its
    # models live
    bench = yaml.safe_load((BENCH / "learned-assocnet.yaml").read_text())
    assert _without_models(Config.from_dict(bench).to_dict()) == _without_models(
        cfg.to_dict()
    )


def test_classic_pipeline_keeps_the_former_defaults():
    cfg = Config.classic()
    assert cfg == Config.from_dict({"detector": {"method": "classic"}})
    assert cfg.detector.method == "classic" and cfg.tracking.method == "classic"
    assert not cfg.rises.enabled and not cfg.review.enabled
    assert cfg.linking.model is None and cfg.linking.threshold == 0.9
    hc = cfg.harmonic_groups
    assert (hc.low_thresh_factor, hc.high_thresh_factor, hc.min_group_size) == (
        6.0,
        10.0,
        3,
    )
    # keys a classic config does not set come from the classic pipeline
    part = Config.from_dict({"detector": {"method": "classic"}, "rises": {}})
    assert part == cfg


def test_packaged_models_resolve(tmp_path):
    names = [
        model_file(None, "salience.pt").name,
        Config().tracking.assoc_model or "assocnet.pt",
        Config().linking.model,
        Config().review.model,
    ]
    for name in names:
        assert model_file(name) == MODELS_DIR / name and (MODELS_DIR / name).exists()
    own = tmp_path / "assocnet.pt"
    own.write_bytes(b"")
    assert model_file(own) == own  # an existing path is used as is
    assert model_file(None, "linker.npz") == MODELS_DIR / "linker.npz"
    with pytest.raises(ValueError):
        model_file(None)


@pytest.mark.parametrize("data", [{"trackign": {}}, {"tracking": {"max_dtt": 3}}])
def test_unknown_keys_are_rejected(data):
    with pytest.raises(ValueError, match="Unknown"):
        Config.from_dict(data)


def _smooth(n):
    """True if n has no prime factor above 5 (a fast FFT size)."""
    for p in (2, 3, 5):
        while n % p == 0:
            n //= p
    return n == 1


@pytest.mark.parametrize("rate", [20000.0, 22050.0, 44100.0, 48000.0, 96000.0])
def test_default_resolution_is_the_same_at_any_rate(rate):
    sc = Config.classic().spectrogram
    nfft = sc.resolve_nfft(rate)
    assert nfft % 2 == 0 and _smooth(nfft)
    assert rate / nfft == pytest.approx(sc.freq_resolution, rel=0.01)


def test_default_resolution_keeps_nfft_2_15_at_20_khz():
    # the benchmarks (20 kHz grids) keep their numbers
    assert Config.classic().spectrogram.resolve_nfft(20000.0) == 2**15


def test_explicit_nfft_overrides_the_resolution():
    cfg = Config.from_dict({"spectrogram": {"nfft": 65536}})
    assert cfg.spectrogram.resolve_nfft(48000.0) == 65536
