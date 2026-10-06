from pathlib import Path

import numpy as np
import pytest

from wavetracker.synthetic import Fish, save_recording, synthesize

REAL_RECORDING = Path("/mnt/data2/2022_tube_competition/raw/2022-05-10-10_00")


@pytest.fixture(scope="session")
def synthetic_wav(tmp_path_factory):
    """60 s, 6 channels, 3 fish (one with a rise) with ground truth."""
    rng = np.random.default_rng(42)
    fish = [
        Fish(520.0, drift=1.0, position=0.1),
        Fish(715.0, drift=2.0, position=0.5, rises=[(30.0, 10.0, 4.0)]),
        Fish(910.0, drift=1.5, amplitude=0.5, position=0.9),
    ]
    rec = synthesize(fish, 60.0, rate=20000.0, channels=6, rng=rng)
    path = tmp_path_factory.mktemp("synthetic") / "rec.wav"
    truth = save_recording(rec, path)
    return path, np.load(truth)
