"""Forward models of a wave fish's EOD potential in shallow water.

Two source models (:class:`SourceModel`):

``monopoles`` (default)
    The fish as a horizontal line of monopoles (Chen, House, Krahe & Nelson
    2005, J Comp Physiol A 191:331; as in ``thunderfish.efield.efish_monopoles``):
    ``n_poles`` poles evenly spaced over ``body_length``, the tail pole carrying
    a negative charge equal to the sum of the positive unit charges of the
    others (net charge zero). With 10 poles over 0.2 m the dipole moment is 1
    (unit charge x m), the same as that of the point dipole.
``dipole``
    A horizontal point dipole of unit moment along the heading.

Medium: homogeneous water below an insulating surface (``z = 0``) and,
optionally, above an insulating bottom at ``z = -water_depth``. Method of
images: a source at ``z_s`` has images at ``z_s + 2kH`` and ``-z_s + 2kH``
(``k`` in Z), all with the same charge since both boundaries are insulating.
The series is truncated at ``|k| <= n_images``; it converges quickly because
the fish's net charge is zero. Without a bottom only the surface image is
used (half-space).

Coordinates are metres, ``z <= 0`` below the surface. A fish is described by
``(x, y, depth, heading)``: the centre of its body at ``(x, y, -depth)``, its
head pointing along ``(cos heading, sin heading, 0)``. Potentials are in
arbitrary units (scaled by a fitted strength).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True, fastmath=True)
def _monopoles(px, py, pz, xf, yf, zf, ct, st, s, q, h, n_img, soft):
    n = px.shape[0]
    out = np.empty(n)
    soft2 = soft * soft
    for i in prange(n):
        acc = 0.0
        for j in range(s.shape[0]):
            dx = px[i] - (xf + s[j] * ct)
            dy = py[i] - (yf + s[j] * st)
            r2h = dx * dx + dy * dy + soft2
            for k in range(-n_img, n_img + 1):
                d1 = pz[i] - (zf + 2.0 * k * h)
                d2 = pz[i] - (-zf + 2.0 * k * h)
                acc += q[j] * (
                    1.0 / np.sqrt(r2h + d1 * d1) + 1.0 / np.sqrt(r2h + d2 * d2)
                )
        out[i] = acc
    return out


@njit(cache=True, parallel=True, fastmath=True)
def _dipole(px, py, pz, xf, yf, zf, ct, st, h, n_img, soft):
    n = px.shape[0]
    out = np.empty(n)
    soft2 = soft * soft
    for i in prange(n):
        dx = px[i] - xf
        dy = py[i] - yf
        r2h = dx * dx + dy * dy + soft2
        proj = dx * ct + dy * st  # horizontal moment: all images share it
        acc = 0.0
        for k in range(-n_img, n_img + 1):
            d1 = pz[i] - (zf + 2.0 * k * h)
            d2 = pz[i] - (-zf + 2.0 * k * h)
            r1 = r2h + d1 * d1
            r2 = r2h + d2 * d2
            acc += proj * (1.0 / (r1 * np.sqrt(r1)) + 1.0 / (r2 * np.sqrt(r2)))
        out[i] = acc
    return out


@dataclass(frozen=True)
class SourceModel:
    """EOD source in a water layer with insulating boundaries."""

    kind: str = "monopoles"
    """"monopoles" (line of monopoles along the body) or "dipole"."""
    body_length: float = 0.2
    """Length of the monopole line [m]."""
    n_poles: int = 10
    """Number of monopoles."""
    water_depth: float | None = None
    """Depth of the insulating bottom [m]; None: no bottom (half-space)."""
    n_images: int = 2
    """Truncation of the image series (|k| <= n_images)."""
    softening: float = 0.01
    """Softening length avoiding the 1/r singularity at an electrode [m]."""

    def __post_init__(self) -> None:
        if self.kind not in ("monopoles", "dipole"):
            raise ValueError(f"Unknown source model {self.kind!r}")
        if self.kind == "monopoles" and self.n_poles < 2:
            raise ValueError("n_poles must be >= 2")
        if self.water_depth is not None and self.water_depth <= 0:
            raise ValueError("water_depth must be positive")

    def poles(self) -> tuple[np.ndarray, np.ndarray]:
        """Positions along the body axis (tail ... head) and charges."""
        s = np.linspace(-self.body_length / 2, self.body_length / 2, self.n_poles)
        q = np.ones(self.n_poles)
        q[0] = -(self.n_poles - 1.0)
        return s, q

    def potential(
        self,
        points: np.ndarray,
        x: float,
        y: float,
        depth: float,
        heading: float,
    ) -> np.ndarray:
        """Potential at `points` (..., 3) of a fish centred at (x, y, -depth)."""
        points = np.asarray(points, dtype=np.float64)
        p = np.ascontiguousarray(points.reshape(-1, 3))
        h, n_img = (
            (0.0, 0) if self.water_depth is None else (self.water_depth, self.n_images)
        )
        args = (
            p[:, 0],
            p[:, 1],
            p[:, 2],
            float(x),
            float(y),
            -float(depth),
            float(np.cos(heading)),
            float(np.sin(heading)),
        )
        if self.kind == "dipole":
            out = _dipole(*args, float(h), int(n_img), float(self.softening))
        else:
            s, q = self.poles()
            out = _monopoles(*args, s, q, float(h), int(n_img), float(self.softening))
        return out.reshape(points.shape[:-1])
