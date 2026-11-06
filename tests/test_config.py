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


def _smooth(n):
    """True if n has no prime factor above 5 (a fast FFT size)."""
    for p in (2, 3, 5):
        while n % p == 0:
            n //= p
    return n == 1


@pytest.mark.parametrize("rate", [20000.0, 22050.0, 44100.0, 48000.0, 96000.0])
def test_default_resolution_is_the_same_at_any_rate(rate):
    sc = Config().spectrogram
    nfft = sc.resolve_nfft(rate)
    assert nfft % 2 == 0 and _smooth(nfft)
    assert rate / nfft == pytest.approx(sc.freq_resolution, rel=0.01)


def test_default_resolution_keeps_nfft_2_15_at_20_khz():
    # the benchmarks (20 kHz grids) keep their numbers
    assert Config().spectrogram.resolve_nfft(20000.0) == 2**15


def test_explicit_nfft_overrides_the_resolution():
    cfg = Config.from_dict({"spectrogram": {"nfft": 65536}})
    assert cfg.spectrogram.resolve_nfft(48000.0) == 65536
