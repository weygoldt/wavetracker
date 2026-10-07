import numpy as np

from wavetracker.comodulation import ComodulationConfig, score_pairs
from wavetracker.results import Results


def _results(seed=0):
    """Fish 1 (300 Hz), its 3rd harmonic tracked as identity 2, and fish 3
    near 3:1 (902 Hz) that shares only the slow (temperature) drift."""
    rng = np.random.default_rng(seed)
    times = np.arange(0, 120, 0.1)
    slow = 1.0 + 0.004 * np.sin(2 * np.pi * times / 200)  # common Q10-like factor

    def fast():
        x = np.convolve(rng.standard_normal(len(times)), np.hanning(20), "same")
        return 0.05 * x / x.std()

    f1 = 300 * slow + fast()
    f3 = 902 * slow + 3 * fast()
    amp1 = (
        1
        + 0.3
        * np.convolve(rng.standard_normal(len(times)), np.hanning(30), "same")
        / 10
    )
    pattern1, pattern3 = np.array([1.0, 0.5, 0.1]), np.array([0.1, 0.6, 1.0])
    rows = [
        (f1, 1.0, amp1[:, None] * pattern1),
        (3 * f1, 2.0, 0.1 * amp1[:, None] * pattern1),
        (f3, 3.0, np.ones((len(times), 1)) * pattern3),
    ]
    fund, idx, ident, power = [], [], [], []
    for f, i, a in rows:
        noise = 0.01 * rng.standard_normal(len(times))
        fund.append(f + noise)
        idx.append(np.arange(len(times)))
        ident.append(np.full(len(times), i))
        power.append(a**2)
    return Results(
        fund_v=np.concatenate(fund),
        idx_v=np.concatenate(idx),
        sign_v=np.concatenate(power),
        ident_v=np.concatenate(ident),
        times=times,
    )


def test_harmonic_comodulates_neighbour_does_not():
    pairs = score_pairs(_results(), ComodulationConfig(timescale=5.0, max_offset=None))
    pairs = pairs.set_index(["low", "high"])
    harm, other = pairs.loc[(1.0, 2.0)], pairs.loc[(1.0, 3.0)]
    assert harm.h == 3 and abs(harm.offset) < 0.05
    assert harm.freq_corr > 0.9 and harm.freq_explained > 0.8
    assert harm.amp_corr > 0.9 and harm.pattern > 0.99
    assert abs(other.offset) > 1.0
    assert abs(other.freq_corr) < 0.2 and other.freq_explained < 0.2
    assert other.pattern < 0.7


def test_slow_drift_alone_looks_comodulated():
    # without removing slow modulations the temperature drift dominates
    pairs = score_pairs(_results(), ComodulationConfig(timescale=1e4, max_offset=None))
    other = pairs.set_index(["low", "high"]).loc[(1.0, 3.0)]
    assert other.freq_corr > 0.5


def test_max_offset_limits_candidates():
    pairs = score_pairs(_results(), ComodulationConfig(max_offset=1.0))
    assert list(zip(pairs.low, pairs.high, strict=True)) == [(1.0, 2.0)]


def test_find_harmonics_flags_only_the_harmonic():
    from wavetracker.comodulation import find_harmonics

    found = find_harmonics(_results(), ComodulationConfig(timescale=5.0))
    assert found[["low", "high", "h"]].values.tolist() == [[1.0, 2.0, 3]]
    assert "frequency" in found.evidence[0]


def test_cli_harmonics(tmp_path):
    from typer.testing import CliRunner

    from wavetracker.cli import app

    _results().save(tmp_path)
    r = CliRunner().invoke(
        app, ["harmonics", str(tmp_path), "--timescale", "5", "--remove"]
    )
    assert r.exit_code == 0, r.output
    assert (tmp_path / "harmonics.csv").exists()
    assert not np.any(Results.load(tmp_path).ident_v == 2.0)
