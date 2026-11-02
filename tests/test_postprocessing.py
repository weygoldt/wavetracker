"""The post-processing commands, end to end through the CLI."""

import json

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from wavetracker.cli import app

DT = 0.5
N_FRAMES = 7200  # one hour


def write_results(folder, freqs=(600.0, 800.0), ident_offset=0.0, t0=0.0):
    """Results of fish at constant frequencies, one detection per frame each."""
    folder.mkdir(parents=True)
    times = t0 + np.arange(N_FRAMES) * DT
    idx = np.repeat(np.arange(N_FRAMES), len(freqs))
    fund = np.tile(freqs, N_FRAMES)
    ident = np.tile(np.arange(len(freqs), dtype=float), N_FRAMES) + ident_offset
    ident[::100] = np.nan  # some detections stay untracked
    np.save(folder / "fund_v.npy", fund)
    np.save(folder / "idx_v.npy", idx)
    np.save(folder / "ident_v.npy", ident)
    np.save(folder / "sign_v.npy", np.full((len(fund), 4), 1e-6))
    np.save(folder / "times.npy", times)
    return fund, idx, ident, times


def invoke(*args):
    result = CliRunner().invoke(app, [str(a) for a in args])
    assert result.exit_code == 0, result.output
    return result


def test_concat_joins_results_into_one_time_base(tmp_path):
    fund, idx, ident, _ = write_results(tmp_path / "day1")
    write_results(tmp_path / "day2", freqs=(610.0, 790.0))

    invoke("concat", tmp_path / "day1", tmp_path / "day2", "-o", tmp_path / "out")

    out = tmp_path / "out"
    merged_times = np.load(out / "times.npy")
    merged_idx = np.load(out / "idx_v.npy")
    merged_ident = np.load(out / "ident_v.npy")
    n = len(fund)
    assert len(merged_times) == 2 * N_FRAMES
    np.testing.assert_allclose(np.diff(merged_times), DT)  # continuous across days
    np.testing.assert_array_equal(merged_idx[n:], idx + N_FRAMES)
    np.testing.assert_array_equal(
        np.load(out / "fund_v.npy")[n:], np.tile([610.0, 790.0], N_FRAMES)
    )
    # identities of the second day do not collide with those of the first
    assert set(np.unique(merged_ident[:n][~np.isnan(ident)])) == {0.0, 1.0}
    assert set(np.unique(merged_ident[n:][~np.isnan(ident)])) == {2.0, 3.0}
    assert np.array_equal(np.isnan(merged_ident), np.tile(np.isnan(ident), 2))
    meta = json.loads((out / "concatenation_metadata.json").read_text())
    assert meta["n_days"] == 2 and meta["n_preliminary_tracks"] == 4


def test_concat_inserts_gap(tmp_path):
    write_results(tmp_path / "day1")
    write_results(tmp_path / "day2")

    invoke(
        "concat",
        tmp_path / "day1",
        tmp_path / "day2",
        "-o",
        tmp_path / "out",
        "--gap",
        60,
    )

    times = np.load(tmp_path / "out" / "times.npy")
    assert times[N_FRAMES] - times[N_FRAMES - 1] == pytest.approx(DT + 60.0)


def test_concat_refuses_incomplete_results(tmp_path):
    write_results(tmp_path / "day1")
    write_results(tmp_path / "day2")
    (tmp_path / "day2" / "ident_v.npy").unlink()

    result = CliRunner().invoke(
        app,
        [
            "concat",
            str(tmp_path / "day1"),
            str(tmp_path / "day2"),
            "-o",
            str(tmp_path / "out"),
        ],
    )

    assert isinstance(result.exception, FileNotFoundError)
    assert "ident_v.npy" in str(result.exception)
    assert not (tmp_path / "out").exists()


def test_freq_analysis_connects_fish_across_timepoints(tmp_path):
    # Folder names give the start of the recording; the fish drift between days.
    write_results(tmp_path / "2025-07-31_10-00", freqs=(600.0, 800.0))
    write_results(tmp_path / "2025-08-01_10-00", freqs=(603.0, 804.0))

    invoke(
        "freq-analysis",
        tmp_path / "2025-*",
        "-t", "10:30",
        "--window", 600,
        "-n", 2,
    )  # fmt: skip

    extracted = pd.read_csv(tmp_path / "extracted_frequencies.csv")
    assert sorted(extracted.median_freq_hz) == [600.0, 603.0, 800.0, 804.0]
    assert sorted(extracted["rank"]) == [1, 1, 2, 2]

    out = tmp_path / "frequency_analysis"
    traces = pd.read_csv(out / "temporal_traces.csv")
    by_fish = traces.groupby("fish_id").median_freq_hz.apply(sorted).to_list()
    assert sorted(by_fish) == [[600.0, 603.0], [800.0, 804.0]]
    assert (out / "temporal_traces.png").stat().st_size > 0
    assert len(pd.read_csv(out / "fish_summary.csv")) == 2


def test_freq_analysis_options_override_the_config(tmp_path):
    folder = tmp_path / "2025-07-31_10-00"
    write_results(folder, freqs=(500.0, 700.0, 900.0))
    config = tmp_path / "custom.cfg"
    config.write_text(
        "[extraction]\nn_fish = 3\nwindow_duration = 10min\ntimepoints = 10:30\n"
        "[analysis]\nrun_analysis = true\n"
    )
    csv = tmp_path / "picked.csv"

    invoke(
        "freq-analysis", folder, "-c", config, "--no-analyze", "--min-freq", 600,
        "-o", csv,
    )  # fmt: skip

    assert sorted(pd.read_csv(csv).median_freq_hz) == [700.0, 900.0]
    assert not (tmp_path / "frequency_analysis").exists()


def test_cleanup_keeps_the_most_prominent_fish(tmp_path):
    folder = tmp_path / "res"
    fund, _, ident, _ = write_results(folder)
    # a third, short track
    fund = np.append(fund, np.full(100, 700.0))
    np.save(folder / "fund_v.npy", fund)
    np.save(
        folder / "idx_v.npy", np.append(np.load(folder / "idx_v.npy"), np.arange(100))
    )
    np.save(folder / "ident_v.npy", np.append(ident, np.full(100, 2.0)))
    np.save(
        folder / "sign_v.npy",
        np.vstack([np.load(folder / "sign_v.npy"), np.full((100, 4), 1e-6)]),
    )

    invoke("cleanup", folder, "-n", 2)

    cleaned = np.load(folder / "ident_v_cleaned_n2.npy")
    assert set(np.unique(cleaned[~np.isnan(cleaned)])) == {0.0, 1.0}
    assert np.all(np.isnan(cleaned[-100:]))
