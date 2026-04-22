"""Tiling invariants for mu3 cells on the sphere.

Checks that the cells produced by scripts/plot_pentagon_globe.py form a
proper tiling at each resolution:

  1. Every unique 3D vertex is incident to exactly 3 cells.
  2. Every undirected edge appears in exactly 2 cells.
  3. The summed (signed) spherical area of all cells equals 4π.

(1) and (2) together imply Euler's formula V - E + F = 2; (3) rules out
overlaps and gaps that happen to preserve the corner/edge counts.
"""

from __future__ import annotations

import itertools
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

# plot_pentagon_globe lives in scripts/; add it to the path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from plot_pentagon_globe import (  # noqa: E402
    build_similarity_maps,
    cell_ring,
    first_nonzero_digit,
)


# Floating-point rounding for vertex-equality checks. The cell-corner pipeline
# is a chain of similarity transforms + gnomonic forward, so adjacent cells'
# shared corners agree to well below 1e-12; rounding to 10 decimals is plenty.
ROUND = 10


def _lnglat_to_unit(lng: float, lat: float) -> np.ndarray:
    lng_r, lat_r = np.radians(lng), np.radians(lat)
    return np.array([
        np.cos(lat_r) * np.cos(lng_r),
        np.cos(lat_r) * np.sin(lng_r),
        np.sin(lat_r),
    ])


def _cell_unique_corners(p: int, digits: tuple, maps) -> list[np.ndarray]:
    """Unique 3D unit vectors at the corners of cell (p, digits).

    The pentagon cell's ring has 6 entries of which two (stitched k=3 and k=4)
    coincide; deduping yields 5. Hex cells yield 6.
    """
    ring = cell_ring(p, digits, maps)
    seen: set[tuple] = set()
    out: list[np.ndarray] = []
    for lng, lat in ring[:-1]:  # drop closing duplicate
        v = _lnglat_to_unit(lng, lat)
        key = tuple(np.round(v, ROUND))
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


def _enumerate_cells(res: int):
    """(base_pentagon, digits) for every valid mu3 cell at resolution res."""
    for p in range(12):
        if res == 0:
            yield (p, ())
            continue
        for digits in itertools.product(range(7), repeat=res):
            if first_nonzero_digit(digits) == 1:
                continue
            yield (p, digits)


def _cached_maps():
    cache: dict[int, dict] = {}

    def get(p: int):
        if p not in cache:
            cache[p] = build_similarity_maps(p)
        return cache[p]

    return get


def _spherical_polygon_area(V: list[np.ndarray]) -> float:
    """Signed spherical polygon area (unit sphere), CCW → positive.

    Van Oosterom–Strackee formula summed over a triangle fan from V[0].
    """
    n = len(V)
    total = 0.0
    v0 = V[0]
    for i in range(1, n - 1):
        a, b = V[i], V[i + 1]
        num = float(np.dot(v0, np.cross(a, b)))
        den = 1.0 + float(np.dot(v0, a)) + float(np.dot(a, b)) + float(np.dot(b, v0))
        total += 2.0 * np.arctan2(num, den)
    return total


@pytest.mark.parametrize("res", [0, 1, 2, 3])
def test_vertex_incidence(res):
    """Every unique 3D corner is shared by exactly 3 cells."""
    get = _cached_maps()
    all_verts: list[tuple] = []
    for p, digits in _enumerate_cells(res):
        for v in _cell_unique_corners(p, digits, get(p)):
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
    get = _cached_maps()
    edges: list[tuple] = []
    for p, digits in _enumerate_cells(res):
        V = _cell_unique_corners(p, digits, get(p))
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
    """Sum of all cell areas equals 4π (no gaps, no overlaps)."""
    get = _cached_maps()
    total = 0.0
    for p, digits in _enumerate_cells(res):
        V = _cell_unique_corners(p, digits, get(p))
        total += _spherical_polygon_area(V)
    assert abs(total - 4.0 * np.pi) < 1e-9, (
        f"res={res}: summed area = {total:.12f}, expected 4π = {4*np.pi:.12f}, "
        f"diff = {total - 4*np.pi:.3e}"
    )


@pytest.mark.parametrize("res", [0, 1, 2, 3])
def test_euler_formula(res):
    """V - E + F = 2 (sphere Euler characteristic)."""
    get = _cached_maps()
    verts: set = set()
    edges: set = set()
    F = 0
    for p, digits in _enumerate_cells(res):
        V = _cell_unique_corners(p, digits, get(p))
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
