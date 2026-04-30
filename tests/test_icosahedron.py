import numpy as np

from mu3 import icosahedron
from mu3.projection import Gnomonic


def test_vertices_on_unit_sphere():
    V = icosahedron.vertices()
    assert V.shape == (12, 3)
    assert np.allclose(np.linalg.norm(V, axis=1), 1.0)


def test_face_table_shape():
    F = icosahedron.faces()
    assert F.shape == (20, 3)
    # every vertex appears in exactly 5 faces
    counts = np.bincount(F.ravel(), minlength=12)
    assert np.all(counts == 5)


def test_gnomonic_roundtrip():
    V = icosahedron.vertices()
    F = icosahedron.faces()
    g = Gnomonic(V[F[0, 0]], V[F[0, 1]], V[F[0, 2]])
    betas = np.array([
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        [0.6, 0.3, 0.1],
        [0.1, 0.7, 0.2],
        [0.2, 0.2, 0.6],
        [0.5, 0.5, 0.0],
    ])
    for beta in betas:
        p = g.to_sphere(beta)
        assert np.isclose(np.linalg.norm(p), 1.0, atol=1e-12)
        b_back = g.to_bary(p)
        assert np.allclose(b_back, beta, atol=1e-12)
