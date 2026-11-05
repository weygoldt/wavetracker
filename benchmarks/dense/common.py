"""Shared pieces of the dense-mixture benchmark: source trials, paths, time warps.

The sources are the two-fish tube-competition recordings of 2019/20 (3x3 and
5x3 grids) and 2022 (1x11 array), all fishgrid ``traces-grid1.raw`` files at
20 kHz. Trials are split by fish group so that no individual of the test pool
is used while developing.
"""

from __future__ import annotations

import csv
import os
import re
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

RATE = 20000.0
ROOT_2019 = Path("/mnt/data2/2019_tube_competition")
ROOT_2022 = Path("/mnt/data2/2022_tube_competition/raw")
DATA = Path(os.environ.get("WT_BENCHMARK_DATA", "~/wrk/data/wavetracker_benchmark"))
DATA = DATA.expanduser()

TEST_GROUPS = {"2019-g5", "2019-g6", "2022-p5", "2022-p6"}
MIN_DEODF = 20.0  # Hz; closer pairs make the two-fish ground truth ambiguous
MIX_CHANNELS = 9  # channels of every mixture (the smallest grid)
SEGMENT = 1200.0  # s of source time analysed per trial


@dataclass
class Trial:
    name: str
    dataset: str  # "2019" or "2022"
    path: str
    channels: int
    duration: float  # s
    eodf: tuple[float, float]  # winner, loser (metadata, whole trial)
    group: str  # fish group: same individuals
    split: str  # "dev" or "test"

    def raw(self) -> np.memmap:
        path = Path(self.path) / "traces-grid1.raw"
        n = path.stat().st_size // 4 // self.channels  # some files end mid-frame
        return np.memmap(path, dtype=np.float32, mode="r", shape=(n, self.channels))

    def to_dict(self) -> dict:
        return asdict(self)


def _grid_channels(cfg: Path) -> int:
    text = cfg.read_text()
    cols = int(re.search(r"Columns1\s*:\s*(\d+)", text).group(1))
    rows = int(re.search(r"Rows1\s*:\s*(\d+)", text).group(1))
    return cols * rows


def _float(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def trials(min_deodf: float = MIN_DEODF, min_duration: float = 3600.0) -> list[Trial]:
    """All usable two-fish trials of both data sets."""
    out = []
    with open(ROOT_2019 / "trial_summary.csv") as f:
        for r in csv.DictReader(f):
            d = ROOT_2019 / r["recording"]
            raw = d / "traces-grid1.raw"
            if not raw.exists():
                continue
            ch = _grid_channels(d / "fishgrid.cfg")
            eodf = (_float(r["EODf_win"]), _float(r["EODf_lose"]))
            group = f"2019-g{r['group']}"
            out.append(
                Trial(
                    r["recording"],
                    "2019",
                    str(d),
                    ch,
                    raw.stat().st_size / 4 / ch / RATE,
                    eodf,
                    group,
                    "test" if group in TEST_GROUPS else "dev",
                )
            )
    with open(ROOT_2022 / "meta.csv") as f:
        for r in csv.DictReader(f):
            name = r[""]
            d = ROOT_2022 / name
            raw = d / "traces-grid1.raw"
            if not raw.exists():
                continue
            ch = _grid_channels(d / "fishgrid.cfg")
            group = f"2022-p{r['Pairing'][0]}"
            out.append(
                Trial(
                    name,
                    "2022",
                    str(d),
                    ch,
                    raw.stat().st_size / 4 / ch / RATE,
                    (_float(r["Win_EODf"]), _float(r["Lose_EODf"])),
                    group,
                    "test" if group in TEST_GROUPS else "dev",
                )
            )
    return [
        t
        for t in out
        if np.all(np.isfinite(t.eodf))
        and abs(t.eodf[0] - t.eodf[1]) >= min_deodf
        and t.duration >= min_duration
    ]


def segment_start(trial: Trial, length: float = SEGMENT) -> float:
    """Reproducible random start of the analysed segment of a trial (avoids
    the first 10 min, when the fish were just put in)."""
    rng = np.random.default_rng(zlib.crc32(trial.name.encode()))
    return float(np.round(rng.uniform(600.0, trial.duration - length - 300.0)))


# --- time warps -------------------------------------------------------------


def common_drift(duration: float, amplitude: float, timescale: float, seed: int):
    """eps(t) on a 1 s grid: smooth random process with std `amplitude`, and
    its integral D(t). All sources share it (temperature, same Q10)."""
    rng = np.random.default_rng(seed)
    n = int(duration) + 2
    x = np.cumsum(rng.standard_normal(n + int(timescale)))
    k = np.hanning(max(int(timescale), 3))
    x = np.convolve(x, k / k.sum(), mode="valid")[:n]
    x -= x.mean()
    eps = x / x.std() * amplitude if amplitude > 0 else np.zeros(n)
    t = np.arange(n, dtype=float)
    d = np.concatenate([[0.0], np.cumsum(0.5 * (eps[1:] + eps[:-1]))])
    return t, eps, d


@dataclass
class Warp:
    """tau(t) = start + a * (t + D(t)): mixture time t -> source time tau.
    Frequencies are scaled by a * (1 + eps(t))."""

    a: float
    start: float
    grid_t: np.ndarray
    grid_eps: np.ndarray
    grid_d: np.ndarray

    def tau(self, t):
        return self.start + self.a * (t + np.interp(t, self.grid_t, self.grid_d))

    def rate(self, t):
        """d tau / d t: factor applied to all frequencies."""
        return self.a * (1.0 + np.interp(t, self.grid_t, self.grid_eps))

    def t_of_tau(self, tau, duration: float):
        """Inverse warp (tau is monotonic in t)."""
        tt = np.linspace(-5.0, duration + 5.0, int(duration + 10) * 20 + 1)
        return np.interp(tau, self.tau(tt), tt)
