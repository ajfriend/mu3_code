import math

import numpy as np
import pytest

from mu3 import cell_boundary, cell_center, icosahedron


# ---------- sanity: identity / pentagon-center behavior ----------


def test_empty_digits_returns_vertex():
    V = icosahedron.vertices()
    for b in range(12):
        got = cell_center(b, ())
        assert np.allclose(got, V[b])


def test_all_zero_digits_returns_vertex():
    V = icosahedron.vertices()
    for b in range(12):
        for N in (1, 2, 5, 10):
            got = cell_center(b, (0,) * N)
            assert np.allclose(got, V[b])


# ---------- sanity: first-digit lands on the expected face ----------


def test_res1_digit_lands_on_expected_face():
    """cell_center(b, (d,)) must lie geometrically on pentagon_face_table[b, d-2]."""
    pft = icosahedron.pentagon_face_table()
    centers = icosahedron.face_centers()
    for b in range(12):
        for d in (2, 3, 4, 5, 6):
            p = cell_center(b, (d,))
            assert abs(np.linalg.norm(p) - 1.0) < 1e-10
            # The closest face center should be pft[b, d-2].
            nearest_face = int(np.argmax(centers @ p))
            expected_face = int(pft[b, d - 2])
            assert nearest_face == expected_face, (
                f"base={b}, digit={d}: landed on face {nearest_face}, "
                f"expected {expected_face}"
            )


# ---------- sanity: res-1 hex child is at correct great-circle distance ----------


def test_res1_hex_child_distance_from_vbase():
    """Res-1 hex center should be a reasonable angular distance (~20-30°) from V[b].

    (Precise direction depends on Class III rotation at odd resolutions; we check
    the magnitude is in a sensible range rather than predicting the exact direction.)
    """
    V = icosahedron.vertices()
    for b in range(12):
        for d in (2, 3, 4, 5, 6):
            p = cell_center(b, (d,))
            dist_deg = math.degrees(math.acos(float(np.clip(p @ V[b], -1.0, 1.0))))
            assert 15 < dist_deg < 35, f"b={b}, d={d}: dist={dist_deg}°"


# ---------- pentagon boundary at res 0 ----------


def test_res0_pentagon_boundary_is_incident_face_centers():
    """cell_boundary(b, ()) must be exactly the 5 incident face centers."""
    F = icosahedron.faces()
    centers = icosahedron.face_centers()
    for b in range(12):
        bnd = cell_boundary(b, (), closed=False)
        assert bnd.shape == (5, 3)
        # Every boundary vertex is some face center incident to b.
        for v in bnd:
            # find closest face center
            idx = int(np.argmax(centers @ v))
            assert np.allclose(centers[idx], v, atol=1e-10)
            assert b in F[idx]


def test_res0_pentagon_boundary_closed():
    b = 0
    bnd = cell_boundary(b, (), closed=True)
    assert bnd.shape == (6, 3)
    assert np.allclose(bnd[0], bnd[-1])


# ---------- hex boundary at res 1 ----------


def test_res1_hex_boundary_shape_and_unit_norm():
    for b in range(12):
        for d in (2, 3, 4, 5, 6):
            bnd = cell_boundary(b, (d,), closed=False)
            assert bnd.shape == (6, 3)
            norms = np.linalg.norm(bnd, axis=1)
            assert np.allclose(norms, 1.0, atol=1e-10)


def test_res1_hex_boundary_roughly_equilateral():
    """Res-1 hex on the sphere shouldn't be wildly distorted."""
    for b in range(12):
        for d in (2, 3, 4, 5, 6):
            bnd = cell_boundary(b, (d,), closed=False)
            # Consecutive-vertex distances should be roughly equal.
            edges = np.linalg.norm(np.roll(bnd, -1, axis=0) - bnd, axis=1)
            ratio = edges.max() / edges.min()
            assert ratio < 2.0, f"hex very distorted at b={b}, d={d}: {ratio=}"


# ---------- deeper resolutions: smoke tests ----------


def test_deep_resolution_unit_norm():
    rng = np.random.default_rng(20260421)
    for _ in range(200):
        b = int(rng.integers(12))
        N = int(rng.integers(1, 6))
        digits = []
        for k in range(N):
            if k == 0 or all(x == 0 for x in digits):
                # first nonzero cannot be 1
                choices = (0, 2, 3, 4, 5, 6)
            else:
                choices = (0, 1, 2, 3, 4, 5, 6)
            digits.append(int(choices[int(rng.integers(len(choices)))]))
        digits = tuple(digits)

        c = cell_center(b, digits)
        assert abs(np.linalg.norm(c) - 1.0) < 1e-10

        bnd = cell_boundary(b, digits, closed=False)
        if all(x == 0 for x in digits):
            assert bnd.shape == (5, 3)
        else:
            assert bnd.shape == (6, 3)
        norms = np.linalg.norm(bnd, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-10)


def test_pentagon_center_at_depth_shrinks():
    """The pentagon-center cell at higher res has a smaller boundary radius."""
    b = 0
    r_prev = None
    for N in range(1, 6):
        bnd = cell_boundary(b, (0,) * N, closed=False)
        V = icosahedron.vertices()[b]
        # angular distances from V to each boundary vertex
        dists = np.arccos(np.clip(bnd @ V, -1.0, 1.0))
        r_curr = dists.mean()
        if r_prev is not None:
            # each level should shrink by about sqrt(7)
            ratio = r_prev / r_curr
            assert abs(ratio - math.sqrt(7)) < 0.1, (
                f"res {N}: shrink ratio {ratio}, expected ~{math.sqrt(7)}"
            )
        r_prev = r_curr
