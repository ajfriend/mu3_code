import math

import numpy as np
import pytest

from mu3 import (
    cell_area,
    cell_boundary,
    cell_center,
    cell_resolution,
    cells_at_res,
    icosahedron,
    is_pentagon,
    is_valid_cell,
)


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


# ---------- cells_at_res ----------


def _valid_count(res: int) -> int:
    """12 * (7^res - number of paths whose first nonzero digit is 1)."""
    if res == 0:
        return 12
    return 12 * (7 ** res - (7 ** res - 1) // 6)


@pytest.mark.parametrize("res,expected", [(0, 12), (1, 72), (2, 492), (3, 3432)])
def test_cells_at_res_count(res, expected):
    assert sum(1 for _ in cells_at_res(res)) == expected
    # formula agrees with the hand-derived counts
    assert _valid_count(res) == expected


@pytest.mark.parametrize("res", [0, 1, 2, 3])
def test_cells_at_res_all_valid(res):
    for cell in cells_at_res(res):
        assert is_valid_cell(cell), f"invalid cell yielded: {cell}"
        assert cell_resolution(cell) == res


@pytest.mark.parametrize("res", [0, 1, 2, 3])
def test_cells_at_res_no_duplicates(res):
    seen = set()
    for cell in cells_at_res(res):
        assert cell not in seen, f"duplicate: {cell}"
        seen.add(cell)


def test_cells_at_res_negative_raises():
    with pytest.raises(ValueError):
        list(cells_at_res(-1))


def test_cells_at_res_is_an_iterator():
    import itertools as _it
    it = cells_at_res(2)
    # should be lazy (an iterator / generator), not a list
    assert hasattr(it, "__next__")
    # first 5 cells without exhausting the generator
    first_five = list(_it.islice(it, 5))
    assert len(first_five) == 5
    # remaining items are still accessible
    rest = list(it)
    assert len(first_five) + len(rest) == 492


# ---------- cell_area ----------


def test_cell_area_res0_equal_and_sum_to_sphere():
    """Res 0 tiles the sphere with 12 pentagons; by icosa symmetry they
    all have the same area, which sums to 4π."""
    areas = [cell_area((b,)) for b in range(12)]
    for a in areas:
        assert a > 0
    # all 12 equal
    assert max(areas) - min(areas) < 1e-10
    assert abs(sum(areas) - 4.0 * math.pi) < 1e-10
    # each ≈ π/3
    assert abs(areas[0] - math.pi / 3) < 1e-10


def test_cell_area_returns_python_float():
    # callers shouldn't have to deal with numpy scalars leaking out
    assert isinstance(cell_area((0,)), float)
    assert isinstance(cell_area((0, 3, 5, 2)), float)


# ---------- cell_area at high resolution / tiny polygons ----------
#
# These tests guard the precision floor of `_spherical_polygon_area`.
# Earlier implementations (per-fan VOS, per-fan Tuynman, lat/lng
# Cagnoli) all underflow at very small cell sizes — Cagnoli, the H3
# production formula, returns 0 for arc-radii ≲ 1e-9 rad. The current
# implementation (per-edge VOS with chord identities and P=V[0]) stays
# exact at all relevant scales. If someone "simplifies" the formula
# back to a pole-anchored variant, these tests will catch it.


def test_spherical_polygon_area_tiny_hex_at_north_pole():
    """A regular spherical hex of arc-radius r centered at the north
    pole has area ≈ (3√3/2) r² in the small-r limit. Verify the area
    function gives this answer correctly even at r = 1e-9 rad, where
    H3's Cagnoli formula underflows to zero."""
    from mu3.cell import _spherical_polygon_area

    for r in (1e-3, 1e-6, 1e-9, 1e-12):
        pts = []
        for k in range(6):
            theta = 2.0 * math.pi * k / 6.0
            pts.append((
                math.sin(r) * math.cos(theta),
                math.sin(r) * math.sin(theta),
                math.cos(r),
            ))
        V = np.array(pts)
        analytic = 1.5 * math.sqrt(3.0) * r * r
        area = _spherical_polygon_area(V)
        assert area > 0, f"r={r}: area is non-positive ({area})"
        rel_err = abs(area - analytic) / analytic
        # Allow generous slack at r=1e-3 where the small-r approximation
        # for the analytic area starts to deviate from the actual
        # spherical hex area; tighter tolerance at smaller r.
        tol = 1e-4 if r >= 1e-3 else 1e-10
        assert rel_err < tol, (
            f"r={r}: rel error {rel_err:.2e} (area={area}, analytic={analytic})"
        )


def test_spherical_polygon_area_tiny_hex_at_south_pole():
    """Same as north-pole test but at the south pole. With a fixed
    south-pole fan reference this would fail catastrophically; with
    P = V[0] (the polygon's first vertex) it works because V[0] is
    near the south pole, *not* near its antipode."""
    from mu3.cell import _spherical_polygon_area

    for r in (1e-3, 1e-6, 1e-9):
        pts = []
        # CCW from outside (looking up at the south pole).
        for k in range(6):
            theta = -2.0 * math.pi * k / 6.0  # negative for CCW-from-outside
            pts.append((
                math.sin(r) * math.cos(theta),
                math.sin(r) * math.sin(theta),
                -math.cos(r),
            ))
        V = np.array(pts)
        analytic = 1.5 * math.sqrt(3.0) * r * r
        area = _spherical_polygon_area(V)
        assert area > 0, f"r={r}: area is non-positive ({area})"
        rel_err = abs(area - analytic) / analytic
        tol = 1e-4 if r >= 1e-3 else 1e-10
        assert rel_err < tol, (
            f"r={r}: rel error {rel_err:.2e} (area={area}, analytic={analytic})"
        )


def test_cell_area_high_res_positive_and_small():
    """At res 18 a typical hex cell is on the order of 1e-15 sr (~25 cm
    on Earth). cell_area must return a positive, finite, small value
    for these — earlier implementations gave 0 or negative garbage."""
    # Pick a few cells at res 18 (hex paths around different bases).
    test_cells = [
        (0, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 2, 3, 4),
        (5, 6, 5, 4, 3, 2, 6, 5, 4, 3, 2, 6, 5, 4, 3, 2, 6, 5, 4),
        (11, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ]
    for cell in test_cells:
        if not is_valid_cell(cell):
            continue
        if is_pentagon(cell):
            continue
        a = cell_area(cell)
        assert math.isfinite(a), f"cell {cell}: area not finite ({a})"
        assert a > 0, f"cell {cell}: area not positive ({a})"
        # Sanity bounds: hex cell area at res 18 is ~10^-15 sr
        # (reasonable range 1e-17 .. 1e-13 covers any orientation).
        assert 1e-17 < a < 1e-13, f"cell {cell}: area {a} out of expected res-18 range"


# ---------- is_pentagon ----------


def test_is_pentagon_res0_all_true():
    for b in range(12):
        assert is_pentagon((b,))


def test_is_pentagon_all_zero_digits_at_each_res():
    for res in range(1, 4):
        for b in range(12):
            assert is_pentagon((b, *([0] * res)))


def test_is_pentagon_false_for_hex_cells():
    assert not is_pentagon((0, 2))
    assert not is_pentagon((0, 0, 3))
    assert not is_pentagon((0, 2, 5, 4))
    assert not is_pentagon((0, 0, 0, 1))   # phantom-form, but still has nonzero


def test_is_pentagon_count_matches_expected_at_each_res():
    """Exactly 12 pentagon-center cells (one per base) at every resolution."""
    for res in range(0, 4):
        n = sum(1 for c in cells_at_res(res) if is_pentagon(c))
        assert n == 12, (res, n)


def test_is_pentagon_invalid_input_returns_false():
    assert is_pentagon(None) is False
    assert is_pentagon(42) is False  # int, no [1:] slicing


# ---------- cell_resolution ----------


def test_cell_resolution_simple():
    assert cell_resolution((0,)) == 0
    assert cell_resolution((11,)) == 0
    assert cell_resolution((0, 2)) == 1
    assert cell_resolution((0, 0, 0)) == 2
    assert cell_resolution((5, 2, 6, 3, 4)) == 4


def test_cell_resolution_matches_cells_at_res():
    for res in range(0, 4):
        for cell in cells_at_res(res):
            assert cell_resolution(cell) == res
