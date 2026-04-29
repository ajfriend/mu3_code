"""Tests for the geometric ring-1 neighbor walk.

Most checks are integer / combinatorial. The sphere-geometry check at the
end confirms the walk's output is also geometrically consistent -- ring-1
neighbor centers sit at roughly a single resolution-dependent edge length
on the unit sphere.
"""

from __future__ import annotations

import numpy as np
import pytest

from mu3 import cell_center, cell_ring1, cells_at_res, is_pentagon, is_valid_cell
from mu3.cell import _eisenstein_center


# --- Ring size --------------------------------------------------------------

@pytest.mark.parametrize("res", [0, 1, 2, 3])
def test_ring1_size(res):
    """Pentagons have exactly 5 ring-1 neighbors (the deleted-direction
    walk collapses); every other hex cell has exactly 6."""
    for cell in cells_at_res(res):
        n = len(cell_ring1(cell))
        expected = 5 if is_pentagon(cell) else 6
        assert n == expected, (cell, n, expected)


# --- Validity ---------------------------------------------------------------

@pytest.mark.parametrize("res", [0, 1, 2, 3])
def test_all_neighbors_are_valid(res):
    for cell in cells_at_res(res):
        for nb in cell_ring1(cell):
            assert is_valid_cell(nb), (cell, nb)
            assert len(nb) == len(cell), (cell, nb)


# --- Symmetry ---------------------------------------------------------------

@pytest.mark.parametrize("res", [0, 1, 2, 3])
def test_ring1_symmetry(res):
    """c' ∈ ring1(c) ⇒ c ∈ ring1(c')."""
    for cell in cells_at_res(res):
        for nb in cell_ring1(cell):
            back = cell_ring1(nb)
            assert cell in back, (cell, nb, back)


# --- Self-exclusion ---------------------------------------------------------

@pytest.mark.parametrize("res", [0, 1, 2, 3])
def test_ring1_excludes_self(res):
    for cell in cells_at_res(res):
        assert cell not in cell_ring1(cell)


# --- Sphere geometry sanity -------------------------------------------------

def _expected_edge_length(res: int) -> float:
    """Expected Eisenstein unit-step magnitude at resolution ``res``.

    |offset[d]| = 1 at res 0; per-resolution scale is 1/sqrt(7).
    """
    return 1.0 / (7 ** (res / 2))


# Removed: ``test_ring1_eisenstein_step_length`` checked that same-base
# ring-1 neighbors are at flat-Eisenstein distance ``1/sqrt(7)^res``, but
# the +60 deg intra-pentagon stitch identifies cells across the deleted
# wedge, so canonical-form flat distances can be stretched arbitrarily
# (e.g. ``(0, 0, 2)`` and ``(0, 0, 6)`` are 3D-adjacent but at flat
# distance ``sqrt(3) × unit_step``). The right invariant is sphere
# distance, checked below.


@pytest.mark.parametrize("res", [1, 2, 3])
def test_ring1_sphere_distance(res):
    """All ring-1 neighbor centers sit near a single per-resolution edge
    length on the unit sphere. The spread is bounded by icosahedral
    distortion — we check the coarse envelope, not a tight value."""
    edge_lens: list[float] = []
    for cell in cells_at_res(res):
        c = cell_center(cell)
        for nb in cell_ring1(cell):
            nc = cell_center(nb)
            # Great-circle (chord) distance on the unit sphere.
            edge_lens.append(float(np.linalg.norm(c - nc)))

    arr = np.array(edge_lens)
    # At res N, chord should scale as ~1/sqrt(7)^N times the res-0 icosa edge.
    # Res-0 icosa edge chord ≈ 2*sin(arctan(2)/2) ≈ 1.0515.
    # Fairly loose bounds that catch runaway errors without overfitting.
    icosa_edge = 2.0 * np.sin(np.arctan(2.0) / 2.0)
    expected = icosa_edge / (7 ** (res / 2))
    ratio = arr / expected
    assert ratio.min() > 0.5, (ratio.min(), expected)
    assert ratio.max() < 1.5, (ratio.max(), expected)
