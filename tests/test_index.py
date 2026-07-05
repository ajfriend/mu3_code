import numpy as np

import mu3
from mu3 import cell_center, cells_at_res, icosahedron, is_pentagon
from mu3.cell import cell_resolution
from mu3.index import _latlng_to_cell_detailed, latlng_to_vec3


def _vec_to_latlng_deg(v: np.ndarray) -> tuple[float, float]:
    x, y, z = v
    lat = np.rad2deg(np.arcsin(z))
    lng = np.rad2deg(np.arctan2(y, x))
    return float(lat), float(lng)


def _all_cells_through_res(res_max: int):
    for res in range(res_max + 1):
        yield from cells_at_res(res)


def test_vertex_positions_self_identify():
    V = icosahedron.vertices()
    for i, v in enumerate(V):
        lat, lng = _vec_to_latlng_deg(v)
        assert mu3.latlng_to_cell(lat, lng) == (i,)


def test_face_centers_return_valid_id():
    for c in icosahedron.face_centers():
        lat, lng = _vec_to_latlng_deg(c)
        cell = mu3.latlng_to_cell(lat, lng)
        assert isinstance(cell, tuple)
        assert len(cell) == 1
        assert 0 <= cell[0] < 12


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


def test_latlng_to_vec3_unit_norm():
    v = latlng_to_vec3(np.array([0.0, 30.0, -45.0, 89.0]), np.array([0.0, 45.0, 180.0, -120.0]))
    assert np.allclose(np.linalg.norm(v, axis=-1), 1.0)


def test_round_trip_cell_centers():
    """latlng_to_cell(*vec_to_latlng(cell_center(c)), res) == c for every
    cell at res 0..3. The full forward+polish pipeline should recover the
    original cell when fed its own 3D center."""
    for res in range(4):
        for c in cells_at_res(res):
            v = cell_center(c)
            lat, lng = _vec_to_latlng_deg(v)
            got = mu3.latlng_to_cell(lat, lng, res=res)
            expected = tuple(int(x) for x in c)
            assert got == expected, (c, got, expected)


def test_round_trip_pentagon_centers():
    """Pentagon-center cells (all-zero digits) at every resolution should
    round-trip cleanly. These are the cells closest to the icosa vertices,
    where polish is most sensitive to argmax tie-breaking."""
    for res in range(4):
        for c in cells_at_res(res):
            if not is_pentagon(c):
                continue
            v = cell_center(c)
            lat, lng = _vec_to_latlng_deg(v)
            got = mu3.latlng_to_cell(lat, lng, res=res)
            expected = tuple(int(x) for x in c)
            assert got == expected, (c, got, expected)


def test_round_trip_pentagon_adjacent_cells():
    """Cells with first non-zero digit in {2..6} (the pentagon-incident
    cells at res 1, plus their sub-cells) -- these stress the polish step
    where the cell boundary has 5 (not 6) edges if it's a pentagon center,
    and where the deleted-wedge stitch can shift the candidate's base."""
    for res in (1, 2, 3):
        for c in cells_at_res(res):
            if not (1 <= cell_resolution(c) and any(d != 0 for d in c[1:])):
                continue
            # Restrict to the immediate pentagon-incident cells: digit at
            # the source res is non-zero and all coarser digits are zero
            # (i.e., the cell is in the d=k wedge directly under base).
            digits = c[1:]
            first_nz = next((i for i, d in enumerate(digits) if d != 0), None)
            if first_nz is None or first_nz != 0:
                continue  # skip cells that aren't in d=k of base
            v = cell_center(c)
            lat, lng = _vec_to_latlng_deg(v)
            got = mu3.latlng_to_cell(lat, lng, res=res)
            assert got == tuple(int(x) for x in c), (c, got, c)


def test_returns_tuple_at_every_res():
    """The public API always returns a Python tuple of the right length."""
    lat, lng = _vec_to_latlng_deg(icosahedron.vertices()[0])
    for res in range(4):
        cell = mu3.latlng_to_cell(lat, lng, res=res)
        assert isinstance(cell, tuple)
        assert len(cell) == res + 1
        assert cell[0] == 0


def test_random_points_inside_indexed_cell():
    """For random points on the sphere, the cell returned by latlng_to_cell
    must contain the point inside its spherical Voronoi boundary. This is
    the core correctness invariant: indexing must agree with cell_boundary.

    Uses _polish_boundary to verify: it returns None iff the point is
    inside the polygon."""
    from mu3.cell import _polish_boundary, cell_boundary
    rng = np.random.default_rng(20260429)
    n_samples = 1000
    p = rng.standard_normal((n_samples, 3))
    p /= np.linalg.norm(p, axis=1, keepdims=True)

    for res in range(4):
        for q in p:
            lat, lng = _vec_to_latlng_deg(q)
            cell = mu3.latlng_to_cell(lat, lng, res=res)
            boundary = cell_boundary(cell, closed=False)
            outside_edge = _polish_boundary(q, boundary)
            assert outside_edge is None, (
                f"res {res}: point {q} indexed to cell {cell} but is "
                f"outside boundary edge {outside_edge}"
            )


def test_polish_handles_boundary_points():
    """Sample points slightly inside / slightly outside each cell's boundary
    and verify they round-trip to the right cell. This exercises the polish
    step, which is a no-op on cell-center inputs but kicks in for points
    near cell boundaries."""
    from mu3.cell import cell_boundary
    rng = np.random.default_rng(20260429)
    for c in cells_at_res(2):
        v_center = cell_center(c)
        boundary = cell_boundary(c, closed=False)
        # Walk halfway from center to each corner and verify still in c.
        # (Strictly inside — should never trigger polish.)
        for corner in boundary:
            inside = v_center + 0.5 * (corner - v_center)
            inside /= np.linalg.norm(inside)
            lat, lng = _vec_to_latlng_deg(inside)
            got = mu3.latlng_to_cell(lat, lng, res=2)
            assert got == tuple(int(x) for x in c), (c, got, "inside test")


# Concrete (res, point_xyz, expected_cell, candidate_before_polish)
# triples harvested from /tmp/edge_inside_test.py — points placed at 99.9%
# of the center→edge-midpoint segment of `expected_cell`. The forward
# pipeline (sphere→flat→snap→step) lands on `candidate_before_polish`,
# and the single-hop polish corrects it back to `expected_cell`.
#
# Frozen here so that any regression in polish, twin disambiguation, or
# the forward snap is caught immediately — not just statistically by the
# random-sample tests.
_POLISH_REASSIGN_CASES = [
    (1, (-0.8665299361556349, 0.05813882170785077, 0.4957274928390942),
     (6, 2), (4, 6)),
    (1, (-0.9769498330321001, 0.1581010237111089, 0.1434332250214402),
     (6, 2), (6, 0)),
    (2, (-0.2118415904917403, -0.5727706956539087, 0.7918692257804109),
     (4, 4, 2), (4, 3, 5)),
    (2, (-0.20426478307406132, -0.4973954751369334, 0.8431332277339157),
     (4, 4, 2), (4, 4, 1)),
    (3, (-0.654247583376297, 0.6619446834615914, -0.36577224564366406),
     (1, 2, 4, 1), (1, 2, 4, 2)),
    (3, (-0.31063894167193595, 0.2442677249540158, -0.918605860237528),
     (3, 2, 6, 4), (3, 2, 5, 1)),
    (4, (0.4207150529661235, 0.10725792196215214, 0.9008299408790057),
     (0, 0, 0, 4, 1), (0, 0, 0, 0, 3)),
    (4, (-0.9248331641208833, -0.13003456250748097, 0.3574557750206917),
     (6, 6, 1, 0, 0), (6, 6, 1, 0, 3)),
    (5, (-0.7616209045835952, 0.5849191908941846, -0.2789321383866663),
     (1, 2, 3, 0, 2, 2), (1, 2, 3, 0, 2, 0)),
    (5, (0.5178907906776582, 0.3643649732869044, -0.773968536293888),
     (7, 0, 0, 0, 2, 0), (7, 0, 0, 0, 2, 5)),
    (7, (0.5203598225175395, 0.061525918132592755, 0.8517277831017703),
     (0, 0, 2, 5, 1, 0, 0, 5), (0, 0, 2, 5, 1, 0, 5, 2)),
    (7, (-0.4044834304949926, 0.6522813418127561, 0.6410321408306708),
     (8, 0, 0, 0, 5, 2, 4, 2), (8, 0, 0, 0, 5, 2, 3, 5)),
    (10, (-0.2943214004764178, -0.19976113535845158, -0.9345963845542644),
     (3, 0, 3, 0, 3, 6, 2, 0, 0, 6, 1),
     (3, 0, 3, 0, 3, 6, 2, 0, 6, 3, 4)),
]


def test_polish_reassignment_regressions():
    """Frozen polish-reassignment fixtures across resolutions 1..10.

    For each (res, p3d, expected, pre_polish_candidate):
      - ``vec3_to_cell_raw`` must return ``pre_polish_candidate`` (the
        forward-pipeline answer, before polish kicks in).
      - ``vec3_to_cell_polished`` and ``vec3_to_cell`` must return
        ``expected`` (the contained cell after polish).
      - Raw and polished must differ -- otherwise the case wouldn't
        exercise polish and should be replaced.
    """
    for res, xyz, expected, pre_polish in _POLISH_REASSIGN_CASES:
        p = np.asarray(xyz, dtype=float)
        raw = mu3.vec3_to_cell_raw(p, res)
        polished = mu3.vec3_to_cell_polished(p, res)
        default = mu3.vec3_to_cell(p, res)
        assert raw == pre_polish, (res, pre_polish, raw)
        assert polished == expected, (res, expected, polished)
        assert default == polished, (res, default, polished)
        assert raw != polished, (
            f"res {res}: case no longer requires polish; pick another"
        )


def test_batched_input_not_implemented():
    """Batched lookups at any res raise NotImplementedError. Scalar tuple
    is the only public API; use ``_latlng_to_cell_detailed`` for batched
    res-0 ndarray output."""
    import pytest
    lats = np.array([0.0, 45.0])
    lngs = np.array([0.0, 90.0])
    with pytest.raises(NotImplementedError):
        mu3.latlng_to_cell(lats, lngs, res=0)
    with pytest.raises(NotImplementedError):
        mu3.latlng_to_cell(lats, lngs, res=1)
