"""Tests for the geometric ring-1 neighbor walk.

Most checks are integer / combinatorial. The sphere-geometry check at the
end confirms the walk's output is also geometrically consistent -- ring-1
neighbor centers sit at roughly a single resolution-dependent edge length
on the unit sphere.
"""

import numpy as np

from mu3 import (
    cell_center,
    cell_resolution,
    cell_ring1,
    cells_at_res,
    dodec,
    is_pentagon,
    is_valid_cell,
)
from mu3.cell import _eisenstein_center
from mu3.face_lattice import digit_offset, get_rot
from mu3.neighbor import _has_leading_zero_d1, _step_to_cell


def _test_cells():
    for res in [0,1,2,3]:
        yield from cells_at_res(res)


def test_ring1_size():
    for c in _test_cells():
        N = len(set(cell_ring1(c)))

        if is_pentagon(c):
            assert N == 5
        else:
            assert N == 6


def test_all_neighbors_are_valid():
    for c in _test_cells():
        for nb in cell_ring1(c):
            assert is_valid_cell(nb), (c, nb)
            assert cell_resolution(nb) == cell_resolution(c), (c, nb)


def test_ring1_symmetry():
    """c' ∈ ring1(c) ⇒ c ∈ ring1(c')."""
    for c in _test_cells():
        for nb in cell_ring1(c):
            assert c in cell_ring1(nb), (c, nb)


def test_ring1_excludes_self():
    for c in _test_cells():
        assert c not in cell_ring1(c), c


def _check_neighbor_lengths(cell):
    c = cell_center(cell)
    lens = [
        np.linalg.norm(c - cell_center(nb))
        for nb in cell_ring1(cell)
    ]

    assert min(lens) > 0
    assert max(lens)/min(lens) < 1.3


def test_ring1_sphere_distance():
    for c in _test_cells():
        _check_neighbor_lengths(c)


def _check_neighbors_ccw(cell):
    """Neighbors should be ordered CCW around the source cell on the
    sphere AND span exactly one full revolution.

    Project each neighbor's 3D center onto the source's tangent plane
    and unit-normalize. Each consecutive pair (a, b) -- including the
    wrap-around -- contributes a signed angle ``atan2((a × b)·c, a·b)``.
    Each angle must be positive (neighbors rotate CCW); the sum must
    equal 2π (the loop closes around the source exactly once).
    """
    c = cell_center(cell)
    proj = []
    for nb in cell_ring1(cell):
        n = cell_center(nb)
        n = n - (n @ c) * c
        n = n / np.linalg.norm(n)
        proj.append(n)

    angles = []
    n = len(proj)
    for i in range(n):
        a = proj[i]
        b = proj[(i + 1) % n]

        angles.append(
            np.arctan2(
                np.cross(a, b) @ c,
                a @ b,
            )
        )

    assert all(a > 0 for a in angles)
    assert abs(sum(angles) - 2 * np.pi) < 1e-9
    assert max(angles)/min(angles) < 1.5


def test_ring1_ccw_order():
    for c in _test_cells():
        _check_neighbors_ccw(c)


def test_ring1_ends_at_primary_direction():
    """The last neighbor in cell_ring1's output is the primary-direction
    neighbor (the cell reached by walking D=6 at res >= 1, or the
    primary-direction pentagon ``vertex_neighbors[base][0]`` at res 0)."""
    for c in _test_cells():
        ring = cell_ring1(c)
        last = ring[-1]
        if cell_resolution(c) == 0:
            primary = (int(dodec.neighbors[c[0]][0]),)
            assert last == primary, (c, last, primary)
        else:
            # Compute what the D=6 walk produces. If it's a non-phantom
            # non-self cell, it must equal `last`.
            z_C = _eisenstein_center(c[1:])
            rot_N = get_rot(cell_resolution(c))
            z_n = z_C + digit_offset[6] / rot_N
            nb = _step_to_cell(z_n, c[0], cell_resolution(c))
            cell_t = tuple(int(x) for x in c)
            if not _has_leading_zero_d1(nb) and nb != cell_t:
                assert last == nb, (c, last, nb)
