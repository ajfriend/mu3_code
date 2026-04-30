import numpy as np
import pytest

from mu3 import icosahedron
from mu3.projection import AlphaOnlySlerp, AlphaSlerp, IVEAProjection

PROJECTION_CLASSES = [AlphaSlerp, AlphaOnlySlerp, IVEAProjection]


def _proj_for_face(cls, face: int):
    V = icosahedron.vertices()
    F = icosahedron.faces()
    return cls(V[F[face, 0]], V[F[face, 1]], V[F[face, 2]])


@pytest.mark.parametrize("cls", PROJECTION_CLASSES)
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_roundtrip_random_interior(cls, seed):
    proj = _proj_for_face(cls, 0)
    rng = np.random.default_rng(seed)
    betas = rng.dirichlet([2.0, 2.0, 2.0], size=200)
    for b in betas:
        p = proj.to_sphere(b)
        assert np.isclose(np.linalg.norm(p), 1.0, atol=1e-12)
        b_back = proj.to_bary(p)
        assert np.allclose(b_back, b, atol=1e-10)


@pytest.mark.parametrize("cls", PROJECTION_CLASSES)
@pytest.mark.parametrize("corner", [
    np.array([1.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0]),
    np.array([0.0, 0.0, 1.0]),
])
def test_corners_are_fixed_points(cls, corner):
    proj = _proj_for_face(cls, 0)
    p = proj.to_sphere(corner)
    # Forward at a corner must land on the corresponding face vertex
    assert np.allclose(p, proj.V[int(np.argmax(corner))], atol=1e-14)
    b_back = proj.to_bary(p)
    assert np.allclose(b_back, corner, atol=1e-12)


@pytest.mark.parametrize("cls", PROJECTION_CLASSES)
@pytest.mark.parametrize("beta", [
    np.array([0.5, 0.5, 0.0]),
    np.array([0.5, 0.0, 0.5]),
    np.array([0.0, 0.5, 0.5]),
])
def test_edge_midpoints_roundtrip(cls, beta):
    proj = _proj_for_face(cls, 0)
    p = proj.to_sphere(beta)
    b_back = proj.to_bary(p)
    assert np.allclose(b_back, beta, atol=1e-10)


@pytest.mark.parametrize("cls", PROJECTION_CLASSES)
def test_centroid_roundtrip(cls):
    proj = _proj_for_face(cls, 0)
    beta = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    p = proj.to_sphere(beta)
    b_back = proj.to_bary(p)
    assert np.allclose(b_back, beta, atol=1e-12)


@pytest.mark.parametrize("cls", PROJECTION_CLASSES)
@pytest.mark.parametrize("face", list(range(20)))
def test_per_face_roundtrip(cls, face):
    proj = _proj_for_face(cls, face)
    rng = np.random.default_rng(1000 + face)
    betas = rng.dirichlet([2.0, 2.0, 2.0], size=20)
    for b in betas:
        p = proj.to_sphere(b)
        b_back = proj.to_bary(p)
        assert np.allclose(b_back, b, atol=1e-10)


@pytest.mark.parametrize("cls", PROJECTION_CLASSES)
@pytest.mark.parametrize("eps", [1e-3, 1e-5, 1e-7, 1e-10])
def test_near_corner_roundtrip_no_singularity(cls, eps):
    """Near-corner inputs must invert without raising and must roundtrip.

    Regression for the singular-Jacobian failure mode the inherited
    2D FD-Newton in AlphaSlerp hits when η=κ=0 (i.e., for AlphaOnlySlerp
    before the 1D fast-path override). All 20 faces × 3 corners × 4 ε.
    """
    near_corners = [
        np.array([1.0 - 2 * eps, eps, eps]),
        np.array([eps, 1.0 - 2 * eps, eps]),
        np.array([eps, eps, 1.0 - 2 * eps]),
    ]
    for face in range(20):
        proj = _proj_for_face(cls, face)
        for b in near_corners:
            p = proj.to_sphere(b)
            b_back = proj.to_bary(p)  # must not raise
            assert np.allclose(b_back, b, atol=1e-9), (face, eps, b, b_back)
