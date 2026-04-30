import numpy as np
import pytest

from mu3 import icosahedron
from mu3.projection import AlphaSlerp


def _slerp_for_face(face: int) -> AlphaSlerp:
    V = icosahedron.vertices()
    F = icosahedron.faces()
    return AlphaSlerp(V[F[face, 0]], V[F[face, 1]], V[F[face, 2]])


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_roundtrip_random_interior(seed):
    slerp = _slerp_for_face(0)
    rng = np.random.default_rng(seed)
    betas = rng.dirichlet([2.0, 2.0, 2.0], size=200)
    for b in betas:
        p = slerp.to_sphere(b)
        assert np.isclose(np.linalg.norm(p), 1.0, atol=1e-12)
        b_back = slerp.to_bary(p)
        assert np.allclose(b_back, b, atol=1e-10)


@pytest.mark.parametrize("corner", [
    np.array([1.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0]),
    np.array([0.0, 0.0, 1.0]),
])
def test_corners_are_fixed_points(corner):
    slerp = _slerp_for_face(0)
    p = slerp.to_sphere(corner)
    # Forward at a corner must land on the corresponding face vertex
    assert np.allclose(p, slerp.V[int(np.argmax(corner))], atol=1e-14)
    b_back = slerp.to_bary(p)
    assert np.allclose(b_back, corner, atol=1e-12)


@pytest.mark.parametrize("beta", [
    np.array([0.5, 0.5, 0.0]),
    np.array([0.5, 0.0, 0.5]),
    np.array([0.0, 0.5, 0.5]),
])
def test_edge_midpoints_roundtrip(beta):
    slerp = _slerp_for_face(0)
    p = slerp.to_sphere(beta)
    b_back = slerp.to_bary(p)
    assert np.allclose(b_back, beta, atol=1e-10)


def test_centroid_roundtrip():
    slerp = _slerp_for_face(0)
    beta = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    p = slerp.to_sphere(beta)
    b_back = slerp.to_bary(p)
    assert np.allclose(b_back, beta, atol=1e-12)


@pytest.mark.parametrize("face", list(range(20)))
def test_per_face_roundtrip(face):
    slerp = _slerp_for_face(face)
    rng = np.random.default_rng(1000 + face)
    betas = rng.dirichlet([2.0, 2.0, 2.0], size=20)
    for b in betas:
        p = slerp.to_sphere(b)
        b_back = slerp.to_bary(p)
        assert np.allclose(b_back, b, atol=1e-10)
