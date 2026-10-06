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

from dataclasses import dataclass, field

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True, fastmath=True)
def _monopoles(p, xf, yf, zf, ct, st, s, q, h, n_img, soft, grad):
    """Potential (and its derivatives by x, y, depth, heading) at points p."""
    n = p.shape[0]
    out = np.zeros((n, 5 if grad else 1))
    soft2 = soft * soft
    for i in prange(n):
        v = 0.0
        gx = 0.0
        gy = 0.0
        gd = 0.0
        gt = 0.0
        for j in range(s.shape[0]):
            dx = p[i, 0] - (xf + s[j] * ct)
            dy = p[i, 1] - (yf + s[j] * st)
            r2h = dx * dx + dy * dy + soft2
            a3 = 0.0  # sum of 1/R^3 over images
            b3 = 0.0  # depth derivative
            for k in range(-n_img, n_img + 1):
                d1 = p[i, 2] - (zf + 2.0 * k * h)
                d2 = p[i, 2] - (-zf + 2.0 * k * h)
                i1 = 1.0 / np.sqrt(r2h + d1 * d1)
                i2 = 1.0 / np.sqrt(r2h + d2 * d2)
                v += q[j] * (i1 + i2)
                if grad:
                    c1 = i1 * i1 * i1
                    c2 = i2 * i2 * i2
                    a3 += c1 + c2
                    b3 += d2 * c2 - d1 * c1
            if grad:
                gx += q[j] * dx * a3
                gy += q[j] * dy * a3
                gd += q[j] * b3
                gt += q[j] * s[j] * (dy * ct - dx * st) * a3
        out[i, 0] = v
        if grad:
            out[i, 1] = gx
            out[i, 2] = gy
            out[i, 3] = gd
            out[i, 4] = gt
    return out


@njit(cache=True, parallel=True, fastmath=True)
def _dipole(p, xf, yf, zf, ct, st, h, n_img, soft, grad):
    n = p.shape[0]
    out = np.zeros((n, 5 if grad else 1))
    soft2 = soft * soft
    for i in prange(n):
        dx = p[i, 0] - xf
        dy = p[i, 1] - yf
        r2h = dx * dx + dy * dy + soft2
        proj = dx * ct + dy * st  # horizontal moment: all images share it
        a3 = 0.0  # sum of 1/R^3
        a5 = 0.0  # sum of 1/R^5
        b5 = 0.0  # depth derivative of a3 / 3
        for k in range(-n_img, n_img + 1):
            d1 = p[i, 2] - (zf + 2.0 * k * h)
            d2 = p[i, 2] - (-zf + 2.0 * k * h)
            i1 = 1.0 / np.sqrt(r2h + d1 * d1)
            i2 = 1.0 / np.sqrt(r2h + d2 * d2)
            c1 = i1 * i1 * i1
            c2 = i2 * i2 * i2
            a3 += c1 + c2
            if grad:
                e1 = c1 * i1 * i1
                e2 = c2 * i2 * i2
                a5 += e1 + e2
                b5 += d2 * e2 - d1 * e1
        out[i, 0] = proj * a3
        if grad:
            out[i, 1] = -ct * a3 + 3.0 * proj * dx * a5
            out[i, 2] = -st * a3 + 3.0 * proj * dy * a5
            out[i, 3] = 3.0 * proj * b5
            out[i, 4] = (dy * ct - dx * st) * a3
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
    _poles: tuple = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.kind not in ("monopoles", "dipole"):
            raise ValueError(f"Unknown source model {self.kind!r}")
        if self.kind == "monopoles" and self.n_poles < 2:
            raise ValueError("n_poles must be >= 2")
        if self.water_depth is not None and self.water_depth <= 0:
            raise ValueError("water_depth must be positive")
        s = np.linspace(-self.body_length / 2, self.body_length / 2, self.n_poles)
        q = np.ones(self.n_poles)
        q[0] = -(self.n_poles - 1.0)
        object.__setattr__(self, "_poles", (s, q))

    def poles(self) -> tuple[np.ndarray, np.ndarray]:
        """Positions along the body axis (tail ... head) and charges."""
        return self._poles

    def _eval(self, points, x, y, depth, heading, grad):
        p = np.ascontiguousarray(np.asarray(points, dtype=np.float64).reshape(-1, 3))
        h, n_img = (
            (0.0, 0) if self.water_depth is None else (self.water_depth, self.n_images)
        )
        args = (
            p,
            float(x),
            float(y),
            -float(depth),
            float(np.cos(heading)),
            float(np.sin(heading)),
        )
        tail = (float(h), int(n_img), float(self.softening), grad)
        if self.kind == "dipole":
            return _dipole(*args, *tail)
        return _monopoles(*args, *self._poles, *tail)

    def potential(
        self,
        points: np.ndarray,
        x: float,
        y: float,
        depth: float,
        heading: float,
    ) -> np.ndarray:
        """Potential at `points` (..., 3) of a fish centred at (x, y, -depth)."""
        shape = np.shape(points)[:-1]
        return self._eval(points, x, y, depth, heading, False)[:, 0].reshape(shape)

    def potential_and_gradient(
        self,
        points: np.ndarray,
        x: float,
        y: float,
        depth: float,
        heading: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Potential (n,) and its derivatives by (x, y, depth, heading) (n, 4)
        at points (n, 3)."""
        out = self._eval(points, x, y, depth, heading, True)
        return out[:, 0], out[:, 1:]
