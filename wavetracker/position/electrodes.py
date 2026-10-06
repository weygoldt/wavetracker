"""Time-varying electrode positions and the channels they form.

Electrode positions are supplied by the user, on the recording's time base:

``.npz``
    ``time`` (T,) [s] and either ``positions`` (T, E, 3) or ``x``, ``y``,
    ``z`` (T, E each) [m].
``.csv``
    a ``time`` column and ``x0, y0, z0, x1, y1, z1, ...`` per electrode.

``z`` is the height relative to the water surface (``z <= 0`` under water).
Rows with NaN positions mark times without a valid geometry (e.g. the boat
out of view); frames there are not part of the survey.

Each recorded channel (column of ``sign_v``) is the potential difference of
two electrodes, ``plus - minus``, or of one electrode against a distant
ground (``minus = -1``). The usual setups:

* ``reference=k``: channel i = electrode i - electrode k, electrodes in
  order with k skipped (e.g. two tips and a hull reference: ``reference=2``);
* ``channel_pairs=[[0, 2], [1, 2]]``: the same, explicitly;
* neither: channel i = electrode i against a distant ground.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def channel_map(
    n_electrodes: int,
    reference: int | None = None,
    channel_pairs: list[list[int]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(plus, minus) electrode index of each channel (minus -1: ground)."""
    if channel_pairs is not None:
        pairs = np.asarray(channel_pairs, dtype=int).reshape(-1, 2)
        plus, minus = pairs[:, 0], pairs[:, 1]
    elif reference is not None:
        if not 0 <= reference < n_electrodes:
            raise ValueError(f"reference {reference} out of range")
        plus = np.array([e for e in range(n_electrodes) if e != reference])
        minus = np.full(len(plus), reference)
    else:
        plus = np.arange(n_electrodes)
        minus = np.full(n_electrodes, -1)
    if plus.min() < 0 or max(plus.max(), minus.max()) >= n_electrodes:
        raise ValueError("channel electrode index out of range")
    if np.any(plus == minus):
        raise ValueError("a channel needs two different electrodes")
    return plus, minus


@dataclass(frozen=True)
class ChannelMap:
    """Electrode pair of each recorded channel."""

    plus: np.ndarray
    """Electrode of each channel (C,)."""
    minus: np.ndarray
    """Reference electrode of each channel (C,), -1 for a distant ground."""

    def __len__(self) -> int:
        return len(self.plus)

    def values(self, v: np.ndarray) -> np.ndarray:
        """Channel signals (..., C) from electrode potentials (..., E)."""
        out = v[..., self.plus]
        grounded = self.minus < 0
        return out - np.where(grounded, 0.0, v[..., np.maximum(self.minus, 0)])

    def midpoints(self, pos: np.ndarray) -> np.ndarray:
        """Midpoint of each channel's electrode pair (..., C, 3) from (..., E, 3)."""
        a = pos[..., self.plus, :]
        b = np.where(
            (self.minus < 0)[:, None], a, pos[..., np.maximum(self.minus, 0), :]
        )
        return (a + b) / 2


@dataclass
class ElectrodeTrack:
    time: np.ndarray
    """Sample times [s], increasing."""
    positions: np.ndarray
    """Electrode positions (T, E, 3) [m]."""
    channels: ChannelMap

    @classmethod
    def from_arrays(
        cls,
        time: np.ndarray,
        positions: np.ndarray,
        reference: int | None = None,
        channel_pairs: list[list[int]] | None = None,
    ) -> ElectrodeTrack:
        time = np.asarray(time, dtype=float)
        positions = np.asarray(positions, dtype=float)
        if positions.ndim != 3 or positions.shape[2] != 3:
            raise ValueError("positions must have shape (time, electrodes, 3)")
        if len(time) != len(positions):
            raise ValueError("time and positions differ in length")
        order = np.argsort(time, kind="stable")
        plus, minus = channel_map(positions.shape[1], reference, channel_pairs)
        return cls(time[order], positions[order], ChannelMap(plus, minus))

    @classmethod
    def load(
        cls,
        path: str | Path,
        reference: int | None = None,
        channel_pairs: list[list[int]] | None = None,
    ) -> ElectrodeTrack:
        path = Path(path)
        if path.suffix == ".npz":
            d = np.load(path)
            if "positions" in d:
                pos = d["positions"]
            elif all(k in d for k in "xyz"):
                pos = np.stack([d["x"], d["y"], d["z"]], axis=-1)
            else:
                raise ValueError(f"{path}: needs 'positions' or 'x', 'y', 'z'")
            return cls.from_arrays(d["time"], pos, reference, channel_pairs)
        if path.suffix == ".csv":
            import pandas as pd

            df = pd.read_csv(path)
            n = 0
            while f"x{n}" in df:
                n += 1
            if n == 0 or "time" not in df:
                raise ValueError(f"{path}: needs columns time, x0, y0, z0, ...")
            pos = np.stack(
                [df[[f"x{e}", f"y{e}", f"z{e}"]].to_numpy(float) for e in range(n)],
                axis=1,
            )
            return cls.from_arrays(df["time"].to_numpy(), pos, reference, channel_pairs)
        raise ValueError(f"Unsupported electrode file {path} (.npz or .csv)")

    @property
    def n_electrodes(self) -> int:
        return self.positions.shape[1]

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    def at(self, t: np.ndarray) -> np.ndarray:
        """Linearly interpolated positions (n, E, 3); NaN where invalid."""
        t = np.asarray(t, dtype=float)
        i = np.clip(np.searchsorted(self.time, t), 1, len(self.time) - 1)
        t0, t1 = self.time[i - 1], self.time[i]
        w = np.where(t1 > t0, (t - t0) / np.where(t1 > t0, t1 - t0, 1.0), 0.0)
        p = (1 - w)[:, None, None] * self.positions[i - 1] + w[
            :, None, None
        ] * self.positions[i]
        outside = (t < self.time[0]) | (t > self.time[-1])
        p[outside] = np.nan
        return p

    def valid(self, t: np.ndarray) -> np.ndarray:
        """Times with a valid geometry."""
        return np.isfinite(self.at(t)).all(axis=(1, 2))
