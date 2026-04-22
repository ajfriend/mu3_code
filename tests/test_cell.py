import math

import numpy as np
import pytest

from mu3 import cell_boundary, cell_center, icosahedron, is_valid_cell


# ---------- sanity: identity / pentagon-center behavior ----------


def test_empty_digits_returns_vertex():
    V = icosahedron.vertices()
    for b in range(12):
        got = cell_center((b,))
        assert np.allclose(got, V[b])


def test_all_zero_digits_returns_vertex():
    V = icosahedron.vertices()
    for b in range(12):
        for N in (1, 2, 5, 10):
            got = cell_center((b, *(0,) * N))
            assert np.allclose(got, V[b])


# ---------- sanity: first-digit cell belongs to its base pentagon ----------


def test_res1_digit_center_closest_to_own_pentagon():
    """cell_center((b, d)) belongs to pentagon b — its nearest icosa vertex is V[b]."""
    V = icosahedron.vertices()
    for b in range(12):
        for d in (2, 3, 4, 5, 6):
            c = cell_center((b, d))
            assert abs(np.linalg.norm(c) - 1.0) < 1e-10
            nearest = int(np.argmax(V @ c))
            assert nearest == b, (
                f"b={b}, d={d}: cell center closer to V[{nearest}] than to V[{b}]"
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
            p = cell_center((b, d))
            dist_deg = math.degrees(math.acos(float(np.clip(p @ V[b], -1.0, 1.0))))
            assert 15 < dist_deg < 35, f"b={b}, d={d}: dist={dist_deg}°"


# ---------- pentagon boundary at res 0 ----------


def test_res0_pentagon_boundary_is_incident_face_centers():
    """cell_boundary((b,)) must be exactly the 5 incident face centers."""
    F = icosahedron.faces()
    centers = icosahedron.face_centers()
    for b in range(12):
        bnd = cell_boundary((b,), closed=False)
        assert bnd.shape == (5, 3)
        # Every boundary vertex is some face center incident to b.
        for v in bnd:
            # find closest face center
            idx = int(np.argmax(centers @ v))
            assert np.allclose(centers[idx], v, atol=1e-10)
            assert b in F[idx]


def test_res0_pentagon_boundary_closed():
    b = 0
    bnd = cell_boundary((b,), closed=True)
    assert bnd.shape == (6, 3)
    assert np.allclose(bnd[0], bnd[-1])


# ---------- hex boundary at res 1 ----------


def test_res1_hex_boundary_shape_and_unit_norm():
    for b in range(12):
        for d in (2, 3, 4, 5, 6):
            bnd = cell_boundary((b, d), closed=False)
            assert bnd.shape == (6, 3)
            norms = np.linalg.norm(bnd, axis=1)
            assert np.allclose(norms, 1.0, atol=1e-10)


def test_res1_hex_boundary_roughly_equilateral():
    """Res-1 hex on the sphere shouldn't be wildly distorted."""
    for b in range(12):
        for d in (2, 3, 4, 5, 6):
            bnd = cell_boundary((b, d), closed=False)
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
        cell = (b, *digits)

        c = cell_center(cell)
        assert abs(np.linalg.norm(c) - 1.0) < 1e-10

        bnd = cell_boundary(cell, closed=False)
        if all(x == 0 for x in digits):
            assert bnd.shape == (5, 3)
        else:
            assert bnd.shape == (6, 3)
        norms = np.linalg.norm(bnd, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-10)


def test_pentagon_center_at_depth_shrinks():
    """Pentagon-center cells shrink with increasing resolution.

    In Eisenstein coords the shrink is exactly sqrt(7) per res. On the sphere
    the Class III twist at odd res and gnomonic curvature at low res make
    the per-step sphere-angle ratio fluctuate around sqrt(7); the 2-step
    ratio converges cleanly to 7 = sqrt(7)².
    """
    b = 0
    V = icosahedron.vertices()[b]
    radii = []
    for N in range(1, 6):
        bnd = cell_boundary((b, *(0,) * N), closed=False)
        assert bnd.shape == (5, 3)
        dists = np.arccos(np.clip(bnd @ V, -1.0, 1.0))
        radii.append(dists.mean())
    # strictly decreasing
    for a, b in zip(radii, radii[1:]):
        assert b < a
    # 2-step ratio converges to 7 (tighter tolerance at deeper levels)
    assert abs(radii[-3] / radii[-1] - 7.0) < 0.2


# ---------- is_valid_cell ----------


def test_is_valid_cell_res0():
    for b in range(12):
        assert is_valid_cell((b,))


def test_is_valid_cell_hex_paths():
    # single nonzero child digit (not 1)
    for b in range(12):
        for d in (2, 3, 4, 5, 6):
            assert is_valid_cell((b, d))
    # mixed deeper paths — digit 1 is fine AFTER the first nonzero
    assert is_valid_cell((3, 2, 1, 0, 1, 6))
    assert is_valid_cell((0, 0, 0, 5, 1, 1, 1))
    assert is_valid_cell((11, 6, 3, 4, 2, 5))


def test_is_valid_cell_all_zero_digits():
    # pentagon-center cells at any depth
    for b in range(12):
        for N in range(0, 10):
            assert is_valid_cell((b, *(0,) * N))


def test_is_valid_cell_rejects_empty():
    assert not is_valid_cell(())


def test_is_valid_cell_rejects_base_out_of_range():
    assert not is_valid_cell((-1,))
    assert not is_valid_cell((12,))
    assert not is_valid_cell((100,))
    assert not is_valid_cell((-1, 2, 3))


def test_is_valid_cell_rejects_digit_out_of_range():
    assert not is_valid_cell((0, 7))
    assert not is_valid_cell((0, -1))
    assert not is_valid_cell((0, 3, 99))


def test_is_valid_cell_rejects_deleted_direction():
    # first nonzero child = 1
    assert not is_valid_cell((0, 1))
    assert not is_valid_cell((5, 0, 1))
    assert not is_valid_cell((11, 0, 0, 0, 1, 2, 3))


def test_is_valid_cell_rejects_non_sequences_and_non_ints():
    assert not is_valid_cell(None)
    assert not is_valid_cell(5)
    assert not is_valid_cell("hello")
    assert not is_valid_cell((0, 2.0))       # float digit
    assert not is_valid_cell((0.5, 2))       # float base
    assert not is_valid_cell((0, "a"))       # string digit


def test_is_valid_cell_accepts_numpy_ints():
    arr = np.array([3, 2, 5, 6], dtype=np.int64)
    assert is_valid_cell(tuple(int(x) for x in arr))
    # raw numpy scalars in a tuple also OK
    assert is_valid_cell(tuple(arr))
