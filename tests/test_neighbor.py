"""Tests for the geometric ring-1 neighbor walk.

Most checks are integer / combinatorial. The sphere-geometry check at the
end confirms the walk's output is also geometrically consistent -- ring-1
neighbor centers sit at roughly a single resolution-dependent edge length
on the unit sphere.
"""

import numpy as np
import pytest

from mu3 import cell_center, cell_resolution, cell_ring1, cells_at_res, is_pentagon, is_valid_cell


@pytest.mark.parametrize('res', [0, 1, 2, 3])
def test_ring1_size(res):
    for c in cells_at_res(res):
        N = len(set(cell_ring1(c)))

        if is_pentagon(c):
            assert N == 5
        else:
            assert N == 6


@pytest.mark.parametrize('res', [0, 1, 2, 3])
def test_all_neighbors_are_valid(res):
    for cell in cells_at_res(res):
        for nb in cell_ring1(cell):
            assert is_valid_cell(nb)
            assert cell_resolution(nb) == cell_resolution(cell)


@pytest.mark.parametrize('res', [0, 1, 2, 3])
def test_ring1_symmetry(res):
    """c' ∈ ring1(c) ⇒ c ∈ ring1(c')."""
    for cell in cells_at_res(res):
        for nb in cell_ring1(cell):
            back = cell_ring1(nb)
            assert cell in back


@pytest.mark.parametrize('res', [0, 1, 2, 3])
def test_ring1_excludes_self(res):
    for cell in cells_at_res(res):
        assert cell not in cell_ring1(cell)


def _check_neighbor_lengths(cell):
    c = cell_center(cell)
    lens = [
        np.linalg.norm(c - cell_center(nb))
        for nb in cell_ring1(cell)
    ]

    assert min(lens) > 0
    assert max(lens)/min(lens) < 1.5


@pytest.mark.parametrize("res", [0, 1, 2, 3])
def test_ring1_sphere_distance(res):
    for cell in cells_at_res(res):
        _check_neighbor_lengths(cell)


def _check_neighbors_ccw(cell):
    """Neighbors should be ordered CCW around the source cell on the sphere.

    Project each neighbor's 3D center onto the tangent plane at the source
    center, then verify that consecutive projected vectors rotate CCW
    (the cross-product, dotted with the outward normal at the source,
    is positive for every consecutive pair, including the wrap-around).
    """
    c = cell_center(cell)
    proj = []
    for nb in cell_ring1(cell):
        n = cell_center(nb)
        proj.append(n - (n @ c) * c)

    n = len(proj)
    for i in range(n):
        a = proj[i]
        b = proj[(i + 1) % n]
        assert np.cross(a, b) @ c > 0, (cell, i, cell_ring1(cell)[i], cell_ring1(cell)[(i + 1) % n])


@pytest.mark.parametrize("res", [0, 1, 2, 3])
def test_ring1_ccw_order(res):
    for cell in cells_at_res(res):
        _check_neighbors_ccw(cell)
