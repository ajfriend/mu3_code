"""Tests for the spherical point-in-polygon polish primitive.

Phase 1 of the projection-independent latlng_to_cell plan. These
tests drive
``_polish_cell_sphere`` directly since it is an internal API.
"""

import numpy as np
import pytest

from mu3 import cell_boundary, cell_center, cells_at_res
from mu3.cell import _polish_boundary, _polish_cell_sphere


@pytest.mark.parametrize("res", [0, 1, 2, 3])
def test_cell_center_is_inside_its_cell(res):
    """_polish_cell_sphere on a cell's own center returns None."""
    for cell in cells_at_res(res):
        q = cell_center(cell)
        assert _polish_cell_sphere(q, cell) is None, (
            f"res={res}, cell={cell}: center reported outside (edge "
            f"{_polish_cell_sphere(q, cell)})"
        )


def _random_unit_vectors(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.mark.parametrize("res", [0, 1, 2, 3])
def test_tiling_exactly_one_cell_contains_each_point(res):
    """For each random sphere point, exactly one cell at `res` contains it.

    Strong tiling invariant: cells cover the sphere with no overlap, no
    gap. Catches orientation flips, edge-normal sign errors, and any
    cell-definition mismatch.

    Vectorized rather than calling :func:`_polish_boundary` per (cell,
    point) pair — the primitive's correctness is covered by the other
    tests; here we exercise the geometric property at scale.
    """
    points = _random_unit_vectors(n=200, seed=42)
    cells = list(cells_at_res(res))

    # Stack every cell's edge normals into (n_cells, 6, 3). Pentagons have
    # only 5 edges; pad the 6th slot with a duplicate of an existing normal
    # so every cell's row has identical semantics (duplicate dot matches its
    # twin, which won't change the sign-of-min result).
    N = np.zeros((len(cells), 6, 3))
    for i, cell in enumerate(cells):
        V = cell_boundary(cell, closed=False)
        n = len(V)
        edge_normals = np.cross(V, np.roll(V, -1, axis=0))
        N[i, :n] = edge_normals
        N[i, n:] = edge_normals[-1]

    # dots[p, c, k] = N[c, k] · points[p]
    dots = np.einsum("pd,ckd->pck", points, N)
    inside = (dots >= 0).all(axis=-1)           # (n_points, n_cells)
    inside_count = inside.sum(axis=-1)          # (n_points,)
    bad = np.where(inside_count != 1)[0]
    assert bad.size == 0, (
        f"res={res}: {bad.size} points inside != 1 cell; "
        f"first bad: point index {bad[0]}, inside {inside_count[bad[0]]} cells"
    )


def test_nudge_past_edge_returns_that_edge():
    """Jittering an edge midpoint outward by 1e-4 rad returns that edge index.

    Verifies the returned edge index matches the edge the query was
    nudged across. Runs across all cells at res 2 and all their edges.
    """
    step = 1e-4  # radians, far larger than the 1e-12 boundary precision
    for cell in cells_at_res(2):
        V = cell_boundary(cell, closed=False)
        n = len(V)
        for k in range(n):
            a = V[k]
            b = V[(k + 1) % n]
            midpoint = a + b
            midpoint = midpoint / np.linalg.norm(midpoint)
            outward = -np.cross(a, b)
            outward_tangent = outward - (outward @ midpoint) * midpoint
            outward_tangent = outward_tangent / np.linalg.norm(outward_tangent)
            q = np.cos(step) * midpoint + np.sin(step) * outward_tangent

            got = _polish_cell_sphere(q, cell)
            assert got == k, (
                f"cell={cell}, edge k={k}: nudge outward returned {got} "
                f"(expected {k})"
            )
