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
    g = Gnomonic(center=V[0])
    xy = np.array([[0.0, 0.0], [0.1, -0.2], [-0.3, 0.4]])
    back = g.inverse(g.forward(xy))
    assert np.allclose(xy, back)
