"""Tiling invariants for mu3 cells on the sphere.

At every resolution the cells from ``mu3.cell_boundary`` should form a
proper tiling of the sphere:

  1. Every unique 3D vertex is incident to exactly 3 cells.
  2. Every undirected edge appears in exactly 2 cells.
  3. The summed signed spherical area of all cells equals 4π.
  4. Every cell's ring is CCW (viewed from outside the sphere).
  5. V - E + F = 2 (sphere Euler characteristic — implied by (1) and (2)).

Tests drive the library directly via ``cells_at_res`` and ``cell_boundary``.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from mu3 import cell_area, cell_boundary, cells_at_res


# Vertices from different cells that share a corner agree to well below
# 1e-12 via the per-face similarity + gnomonic pipeline; rounding to 10
# decimals is plenty for equality and tight enough not to conflate nearby
# distinct corners at deeper resolutions.
ROUND = 10


@pytest.mark.parametrize("res", [0, 1, 2, 3])
def test_vertex_incidence(res):
    """Every unique 3D corner is shared by exactly 3 cells."""
    all_verts: list[tuple] = []
    for cell in cells_at_res(res):
        for v in cell_boundary(cell, closed=False):
            all_verts.append(tuple(np.round(v, ROUND)))
    counts = Counter(all_verts)
    wrong = [(v, c) for v, c in counts.items() if c != 3]
    assert not wrong, (
        f"res={res}: {len(wrong)} vertex/ices with wrong incidence; "
        f"first: {wrong[0][0]} shared by {wrong[0][1]} cells"
    )


@pytest.mark.parametrize("res", [0, 1, 2, 3])
def test_edge_sharing(res):
    """Every undirected edge appears in exactly 2 cells."""
    edges: list[tuple] = []
    for cell in cells_at_res(res):
        V = cell_boundary(cell, closed=False)
        n = len(V)
        for i in range(n):
            a = tuple(np.round(V[i], ROUND))
            b = tuple(np.round(V[(i + 1) % n], ROUND))
            edges.append(tuple(sorted([a, b])))
    counts = Counter(edges)
    wrong = [(e, c) for e, c in counts.items() if c != 2]
    assert not wrong, (
        f"res={res}: {len(wrong)} edge(s) with wrong sharing count; "
        f"first count = {wrong[0][1]}"
    )


@pytest.mark.parametrize("res", [0, 1, 2, 3])
def test_full_coverage(res):
    """Every cell has positive area, and they sum to the area of the sphere.

    Catches gaps (sum < 4π), overlaps (sum > 4π), and any cell that came
    out CW instead of CCW (would contribute negatively to the sum).
    """
    total = 0.0
    for cell in cells_at_res(res):
        area = cell_area(cell)
        assert area > 0, f"res={res}, cell={cell}: area {area:.3e} not positive"
        total += area
    assert abs(total - 4.0 * np.pi) < 1e-9, (
        f"res={res}: summed area = {total:.12f}, expected 4π = {4*np.pi:.12f}, "
        f"diff = {total - 4*np.pi:.3e}"
    )


@pytest.mark.parametrize("res", [0, 1, 2, 3])
def test_euler_formula(res):
    """V - E + F = 2 (sphere Euler characteristic)."""
    verts: set = set()
    edges: set = set()
    F = 0
    for cell in cells_at_res(res):
        V = cell_boundary(cell, closed=False)
        F += 1
        for i, v in enumerate(V):
            verts.add(tuple(np.round(v, ROUND)))
            a = tuple(np.round(V[i], ROUND))
            b = tuple(np.round(V[(i + 1) % len(V)], ROUND))
            edges.add(tuple(sorted([a, b])))
    assert len(verts) - len(edges) + F == 2, (
        f"res={res}: V={len(verts)}, E={len(edges)}, F={F}, "
        f"V-E+F={len(verts) - len(edges) + F} (expected 2)"
    )
