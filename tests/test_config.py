import pytest

from wavetracker.config import Config


def test_yaml_roundtrip(tmp_path):
    cfg = Config()
    cfg.tracking.max_dt = 5.0
    cfg.harmonic_groups.low_threshold = 12.0
    cfg.save(tmp_path / "cfg.yaml")
    loaded = Config.load(tmp_path / "cfg.yaml")
    assert loaded == cfg


def test_partial_config_uses_defaults():
    cfg = Config.from_dict({"tracking": {"freq_tolerance": 1.0}})
    assert cfg.tracking.freq_tolerance == 1.0
    assert cfg.tracking.max_dt == Config().tracking.max_dt


@pytest.mark.parametrize("data", [{"trackign": {}}, {"tracking": {"max_dtt": 3}}])
def test_unknown_keys_are_rejected(data):
    with pytest.raises(ValueError, match="Unknown"):
        Config.from_dict(data)
