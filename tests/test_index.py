import numpy as np

import mu3
from mu3 import icosahedron
from mu3.index import _latlng_to_cell_detailed, latlng_to_vec


def _vec_to_latlng_deg(v: np.ndarray) -> tuple[float, float]:
    x, y, z = v
    lat = np.rad2deg(np.arcsin(z))
    lng = np.rad2deg(np.arctan2(y, x))
    return float(lat), float(lng)


def test_vertex_positions_self_identify():
    V = icosahedron.vertices()
    for i, v in enumerate(V):
        lat, lng = _vec_to_latlng_deg(v)
        assert mu3.latlng_to_cell(lat, lng) == i


def test_face_centers_return_valid_id():
    for c in icosahedron.face_centers():
        lat, lng = _vec_to_latlng_deg(c)
        cell = mu3.latlng_to_cell(lat, lng)
        assert 0 <= cell < 12


def test_oracle_agreement():
    rng = np.random.default_rng(20260421)
    p = rng.standard_normal((2000, 3))
    p /= np.linalg.norm(p, axis=1, keepdims=True)

    V = icosahedron.vertices()
    oracle = np.argmax(p @ V.T, axis=1)

    final, _ = _latlng_to_cell_detailed(p)
    assert np.array_equal(final, oracle)


def test_polish_is_noop_at_res0():
    rng = np.random.default_rng(20260421)
    p = rng.standard_normal((2000, 3))
    p /= np.linalg.norm(p, axis=1, keepdims=True)

    final, pre_polish = _latlng_to_cell_detailed(p)
    assert np.array_equal(final, pre_polish)


def test_batched_entrypoint():
    lats = np.array([90.0, -90.0, 0.0])
    lngs = np.array([0.0, 0.0, 0.0])
    out = mu3.latlng_to_cell(lats, lngs)
    assert out.shape == (3,)
    assert out.dtype == np.int64


def test_latlng_to_vec_unit_norm():
    v = latlng_to_vec(np.array([0.0, 30.0, -45.0, 89.0]), np.array([0.0, 45.0, 180.0, -120.0]))
    assert np.allclose(np.linalg.norm(v, axis=-1), 1.0)
