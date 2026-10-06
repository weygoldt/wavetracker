"""On-disk format of wavetracker results.

A result directory contains plain ``.npy`` files (compatible with the
post-processing tools and the EOD sorter GUI):

=================== ==========================================================
``fund_v.npy``      fundamental frequency of each detection [Hz]
``idx_v.npy``       frame index of each detection (into ``times``)
``sign_v.npy``      power of each detection on every electrode (n, channels)
``ident_v.npy``     identity of each detection (NaN = unassigned)
``times.npy``       time of each frame relative to recording start [s]
``sparse_*.npy``    downsampled overview spectrogram (freq x time, power)
``fine_*.npy``      optional full-resolution spectrogram (time x freq, power)
``wavetracker.json`` metadata: input, config, thresholds, timings, version
=================== ==========================================================
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

META_FILE = "wavetracker.json"


@dataclass
class Results:
    fund_v: np.ndarray
    idx_v: np.ndarray
    sign_v: np.ndarray
    ident_v: np.ndarray
    times: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_ids(self) -> int:
        return int(np.unique(self.ident_v[~np.isnan(self.ident_v)]).size)

    def ids(self) -> np.ndarray:
        return np.unique(self.ident_v[~np.isnan(self.ident_v)])

    def save(self, folder: str | Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        for name in ("fund_v", "idx_v", "sign_v", "ident_v", "times"):
            np.save(folder / f"{name}.npy", getattr(self, name))
        (folder / META_FILE).write_text(json.dumps(self.meta, indent=2, default=str))

    @classmethod
    def load(cls, folder: str | Path) -> Results:
        folder = Path(folder)
        if not (folder / "fund_v.npy").exists():
            raise FileNotFoundError(f"No wavetracker results in {folder}")
        arrays = {
            name: np.load(folder / f"{name}.npy", allow_pickle=False)
            for name in ("fund_v", "idx_v", "sign_v", "times")
        }
        ident = folder / "ident_v.npy"
        arrays["ident_v"] = (
            np.load(ident) if ident.exists() else np.full(len(arrays["fund_v"]), np.nan)
        )
        meta_file = folder / META_FILE
        meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        return cls(**arrays, meta=meta)


def load_sparse_spectrogram(folder: str | Path):
    """Return (power[freq, time], freqs, times) of the overview spectrogram."""
    folder = Path(folder)
    return (
        np.load(folder / "sparse_spectra.npy"),
        np.load(folder / "sparse_freq.npy"),
        np.load(folder / "sparse_time.npy"),
    )


def load_fine_spectrogram(folder: str | Path):
    """Return (memmapped power[time, freq], freqs, times)."""
    folder = Path(folder)
    return (
        np.load(folder / "fine_spec.npy", mmap_mode="r"),
        np.load(folder / "fine_freqs.npy"),
        np.load(folder / "fine_times.npy"),
    )
