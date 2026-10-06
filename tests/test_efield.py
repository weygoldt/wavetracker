import numpy as np
import pytest

from wavetracker.position.efield import SourceModel


def _reference(model, points, x, y, depth, heading):
    """Plain numpy implementation of the image series."""
    zs = -depth
    if model.water_depth is None:
        zimg = np.array([zs, -zs])
    else:
        k = np.arange(-model.n_images, model.n_images + 1)
        zimg = np.r_[zs + 2 * k * model.water_depth, -zs + 2 * k * model.water_depth]
    u = np.array([np.cos(heading), np.sin(heading)])
    out = np.zeros(len(points))
    if model.kind == "dipole":
        for z in zimg:
            d = points - [x, y, z]
            r2 = (d**2).sum(1) + model.softening**2
            out += (d[:, :2] @ u) / r2**1.5
        return out
    s, q = model.poles()
    for sj, qj in zip(s, q, strict=True):
        for z in zimg:
            d = points - [x + sj * u[0], y + sj * u[1], z]
            out += qj / np.sqrt((d**2).sum(1) + model.softening**2)
    return out


@pytest.fixture
def points():
    rng = np.random.default_rng(0)
    p = rng.uniform(-2, 2, (500, 3))
    p[:, 2] = -rng.uniform(0, 1.0, 500)
    return p


@pytest.mark.parametrize("kind", ["monopoles", "dipole"])
@pytest.mark.parametrize("water_depth", [None, 1.2])
def test_matches_numpy_reference(points, kind, water_depth):
    m = SourceModel(kind=kind, water_depth=water_depth)
    args = (0.3, -0.2, 0.4, 0.7)
    np.testing.assert_allclose(
        m.potential(points, *args), _reference(m, points, *args), rtol=1e-9
    )


@pytest.mark.parametrize("kind", ["monopoles", "dipole"])
@pytest.mark.parametrize("water_depth", [None, 1.2])
def test_gradient_matches_finite_differences(points, kind, water_depth):
    m = SourceModel(kind=kind, water_depth=water_depth)
    q = np.array([0.3, -0.2, 0.4, 0.7])
    v, g = m.potential_and_gradient(points, *q)
    np.testing.assert_allclose(v, m.potential(points, *q), rtol=1e-12)
    eps = 1e-6
    for k in range(4):
        dq = np.zeros(4)
        dq[k] = eps
        fd = (m.potential(points, *(q + dq)) - m.potential(points, *(q - dq))) / (
            2 * eps
        )
        np.testing.assert_allclose(g[:, k], fd, rtol=1e-4, atol=1e-6 * np.abs(fd).max())


def test_far_field_of_monopole_line_is_dipole():
    m = SourceModel(water_depth=None)
    d = SourceModel(kind="dipole", water_depth=None)
    ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    for r in (1.0, 3.0):
        p = np.c_[r * np.cos(ang), r * np.sin(ang), np.full(16, -0.5)]
        vm, vd = m.potential(p, 0, 0, 0.5, 0.3), d.potential(p, 0, 0, 0.5, 0.3)
        err = np.abs(vm - vd).max() / np.abs(vd).max()
        assert err < (0.1 if r == 1.0 else 0.03)


def test_insulating_boundaries():
    """Normal component of the field vanishes at the surface and the bottom."""
    h, eps = 1.0, 1e-5
    xy = np.random.default_rng(1).uniform(-1.5, 1.5, (50, 2))

    def dvdz(model, z):
        up = model.potential(np.c_[xy, np.full(50, z + eps)], 0, 0, 0.4, 1.0)
        lo = model.potential(np.c_[xy, np.full(50, z - eps)], 0, 0, 0.4, 1.0)
        return np.abs(up - lo).max() / (2 * eps)

    for n_img in (2, 6):
        m = SourceModel(water_depth=h, n_images=n_img)
        scale = np.abs(m.potential(np.c_[xy, np.full(50, -0.2)], 0, 0, 0.4, 1.0)).max()
        assert dvdz(m, 0.0) < 1e-4 * scale  # exact by symmetry
    # the bottom is only approximately insulating; the error falls with n_images
    errs = [dvdz(SourceModel(water_depth=h, n_images=n), -h) for n in (0, 2, 6)]
    assert errs[0] > errs[1] > errs[2]


def test_rotation_and_translation_invariance(points):
    m = SourceModel(water_depth=1.2)
    a = 0.9
    rot = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    v1 = m.potential(points, 0.0, 0.0, 0.5, 0.2)
    v2 = m.potential(points @ rot.T + [1.0, 2.0, 0.0], 1.0, 2.0, 0.5, 0.2 + a)
    np.testing.assert_allclose(v1, v2, rtol=1e-9, atol=1e-12)


def test_shape_and_validation():
    m = SourceModel()
    assert m.potential(np.zeros((4, 3, 3)) - [0, 0, 0.1], 1, 0, 0.5, 0).shape == (4, 3)
    with pytest.raises(ValueError):
        SourceModel(kind="quadrupole")
