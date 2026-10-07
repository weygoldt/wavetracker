"""List input: a recording split over several files, read as one."""

import numpy as np
import pytest
from audioio import write_audio

from wavetracker import io
from wavetracker.config import Config
from wavetracker.io import open_recording, recording_info, resolve_input
from wavetracker.pipeline import detect

RATE = 20000.0


def _write(path, data, stamp):
    write_audio(str(path), data, RATE, metadata={"INFO": {"DateTimeOriginal": stamp}})
    return path


@pytest.fixture
def split_files(tmp_path):
    """Two files whose timestamps are an hour apart (not continuous)."""
    rng = np.random.default_rng(1)
    a = (0.1 * rng.standard_normal((40000, 2))).astype(np.float32)
    b = (0.1 * rng.standard_normal((30000, 2))).astype(np.float32)
    pa = _write(tmp_path / "a.wav", a, "2022-05-10T10:00:00")
    pb = _write(tmp_path / "b.wav", b, "2022-05-10T11:00:00")
    return pa, pb, a, b


def test_feature_flag():
    assert io.MULTI_INPUT is True


def test_resolve_single_element_list(split_files):
    pa = split_files[0]
    assert resolve_input([pa]) == resolve_input(pa) == str(pa)
    assert resolve_input((str(pa),)) == str(pa)


def test_resolve_list_keeps_order(split_files):
    pa, pb = split_files[:2]
    assert resolve_input([pb, pa]) == [str(pb), str(pa)]


def test_resolve_list_missing_file(split_files, tmp_path):
    pa = split_files[0]
    with pytest.raises(FileNotFoundError, match=r"nope\.wav"):
        resolve_input([pa, tmp_path / "nope.wav"])
    with pytest.raises(FileNotFoundError):
        resolve_input([])
    with pytest.raises(FileNotFoundError):
        resolve_input([tmp_path])  # a directory is not a file


def test_open_joined_ignores_timestamps(split_files):
    pa, pb, a, b = split_files
    joined = np.vstack([a, b])
    with open_recording([pa, pb], buffersize=1.0) as data:
        assert len(data) == len(a) + len(b)
        assert data.channels == 2 and data.rate == RATE
        np.testing.assert_allclose(data[39990:40010], joined[39990:40010], atol=1e-4)
        np.testing.assert_allclose(data[:], joined, atol=1e-4)
        np.testing.assert_allclose(data[100:50000, 1], joined[100:50000, 1], atol=1e-4)
        np.testing.assert_allclose(data[-1], joined[-1], atol=1e-4)
    info = recording_info([pa, pb])
    assert info.frames == len(a) + len(b) and info.channels == 2


def test_open_joined_keeps_few_files_open(tmp_path):
    """Memory and handles must not grow with the number of files."""
    rng = np.random.default_rng(2)
    parts = [
        (0.1 * rng.standard_normal((5000, 2))).astype(np.float32) for _ in range(6)
    ]
    paths = [
        _write(tmp_path / f"p{i}.wav", d, f"2022-05-10T1{i}:00:00")
        for i, d in enumerate(parts)
    ]
    joined = np.vstack(parts)
    with open_recording(paths, buffersize=1.0) as data:
        assert len(data._open) == 0  # probing closes every file again
        for k in range(0, len(joined), 2500):
            np.testing.assert_allclose(
                data[k : k + 3000], joined[k : k + 3000], atol=1e-4
            )
            assert len(data._open) <= data.max_open
    assert len(data._open) == 0


def test_open_joined_empty_slice(split_files):
    pa, pb = split_files[:2]
    with open_recording([pa, pb], buffersize=1.0) as data:
        assert data[5:5].shape == (0, 2)
        assert data[5:5, 0].shape == (0,)


def test_open_joined_rejects_mismatched_files(split_files, tmp_path):
    pa = split_files[0]
    pc = tmp_path / "c.wav"
    write_audio(str(pc), np.zeros((1000, 3), np.float32), RATE)
    with pytest.raises(ValueError, match="channels"):
        open_recording([pa, pc])


def test_directory_and_single_file_unchanged(split_files):
    pa = split_files[0]
    with open_recording(pa, buffersize=1.0) as data:
        assert len(data) == 40000
    with open_recording([pa], buffersize=1.0) as data:
        assert len(data) == 40000


def test_detect_split_equals_concatenated(synthetic_wav, tmp_path):
    """Detecting on a recording split into two files gives exactly the
    detections of the concatenated file."""
    from audioio import load_audio

    path, _ = synthetic_wav
    data, rate = load_audio(str(path))
    data = data[: int(20 * rate)].astype(np.float32)
    cut = int(7.3 * rate)
    whole = _write(tmp_path / "whole.wav", data, "2022-05-10T10:00:00")
    pa = _write(tmp_path / "p1.wav", data[:cut], "2022-05-10T10:00:00")
    pb = _write(tmp_path / "p2.wav", data[cut:], "2022-05-10T12:00:00")
    cfg = Config()
    ref = detect(whole, tmp_path / "ref", cfg, device="cpu").results
    res = detect([pa, pb], tmp_path / "split", cfg, device="cpu").results
    np.testing.assert_array_equal(res.fund_v, ref.fund_v)
    np.testing.assert_array_equal(res.idx_v, ref.idx_v)
    np.testing.assert_array_equal(res.times, ref.times)
    assert res.meta["files"] == [str(pa), str(pb)]
    assert res.meta["input"] == [str(pa.resolve()), str(pb.resolve())]
