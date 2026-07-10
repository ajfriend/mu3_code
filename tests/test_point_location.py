"""Point location via the exact holonomy entry point, judged by the
implementation-agnostic ground truth: the returned cell CONTAINS the
query point (spherical point-in-polygon against the cell's own
geodesic boundary). No comparison to any other implementation.
"""

import cmath
import math

import numpy as np
import pytest

from mu3 import icosahedron, vec3_to_cell
from mu3.cell import (
    _pick_wedge,
    _polish_cell_sphere,
    _project,
    _projection,
    cell_boundary,
    cell_center,
    cells_at_res,
)
from mu3.eisenstein import first_nonzero_digit
from mu3.index import _polish, vec3_to_cell_raw

_locate = vec3_to_cell   # the production pipeline: exact raw + polish


def _assert_contains(cell: tuple, q: np.ndarray):
    assert _polish_cell_sphere(q, cell) is None, \
        f'{cell} does not contain query {q}'


@pytest.mark.parametrize('res', range(4))
def test_cell_centers_round_trip(res):
    for cell in cells_at_res(res):
        assert _locate(cell_center(cell), res) == cell


@pytest.mark.parametrize('res', range(1, 5))
def test_uniform_random_containment(res):
    rng = np.random.default_rng(res)
    v = rng.normal(size=(500, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    for q in v:
        _assert_contains(_locate(q, res), q)


def test_corner_perturbed_containment():
    """Cell corners are 3-cell junctions (including phantom corners) —
    the hardest point-location inputs. Perturb each res-2 corner
    slightly and require containment."""
    rng = np.random.default_rng(0)
    for cell in cells_at_res(2):
        for v in cell_boundary(cell, closed=False):
            q = v + 1e-4 * rng.normal(size=3)
            q /= np.linalg.norm(q)
            _assert_contains(_locate(q, 2), q)


def _assert_picked_wedge_contains(q, eps=1e-9):
    """The exactness invariant behind the search-free ``_sphere_to_flat``:
    the wedge ``_pick_wedge`` selects genuinely contains ``q`` (its
    barycentric is non-negative within ``eps``), so the single inverse on it
    succeeds without trying the other four triangles."""
    base = int(np.argmax(icosahedron.vertices() @ q))
    d = _pick_wedge(q, base)
    beta = _projection(base, d).to_bary(q)
    assert float(beta.min()) >= -eps, \
        f'picked wedge d={d} for base {base} does not contain {q}: beta={beta}'


def test_pick_wedge_contains_uniform():
    rng = np.random.default_rng(42)
    v = rng.normal(size=(5000, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    for q in v:
        _assert_picked_wedge_contains(q)


def test_pick_wedge_spoke_and_center_degeneracies():
    """The only inputs where the dot-product wedge pick could tie: points
    straddling a base→neighbor spoke (a wedge boundary) and points at the
    pentagon centre (all five wedges meet). The picked wedge must still
    contain the point."""
    V = icosahedron.vertices()
    nbrs = icosahedron.vertex_neighbors()
    rng = np.random.default_rng(1)
    for base in range(12):
        vb = V[base]
        for nb in nbrs[base]:
            for t in (0.05, 0.2, 0.4, 0.49):
                mid = (1.0 - t) * vb + t * V[nb]
                for _ in range(5):
                    q = mid + 1e-6 * rng.normal(size=3)
                    q /= np.linalg.norm(q)
                    _assert_picked_wedge_contains(q)
        for _ in range(20):
            q = vb + 1e-7 * rng.normal(size=3)
            q /= np.linalg.norm(q)
            _assert_picked_wedge_contains(q)


@pytest.mark.parametrize('res', range(1, 5))
def test_cut_line_containment(res):
    """Queries on and beside each pentagon's deleted-wedge cut ray
    (flat angle 60 deg) and primary ray (0 deg) — the seam-side
    boundaries of the witness sign test."""
    for base in range(12):
        for theta_deg in (0.0, 1e-3, -1e-3, 60.0, 60.0 + 1e-3,
                          60.0 - 1e-3):
            for r in (0.1, 0.25, 0.4, 0.55):
                z = r * cmath.exp(1j * math.radians(theta_deg))
                q = _project(z, base)
                _assert_contains(_locate(q, res), q)


@pytest.mark.parametrize('res', range(1, 5))
def test_banded_polish_matches_full(res):
    """The banded polish must equal the full-boundary reference polish
    exactly. Uniform random points plus boundary-jittered points (the
    band interior, where the two paths could plausibly diverge)."""
    rng = np.random.default_rng(res)
    pts = list(rng.normal(size=(400, 3)))
    for _ in range(120):
        digits = rng.integers(0, 7, size=res)
        if first_nonzero_digit(map(int, digits)) == 1:
            continue
        cell = (int(rng.integers(0, 12)), *map(int, digits))
        B = cell_boundary(cell, closed=False)
        k = int(rng.integers(0, len(B)))
        t = float(rng.uniform(0.0, 1.0))
        q = (1 - t) * B[k] + t * B[(k + 1) % len(B)]
        scale = float(np.linalg.norm(B[k] - B[(k + 1) % len(B)]))
        pts.append(q + rng.normal(size=3) * scale * rng.uniform(0.0, 0.4))
    for q in pts:
        q = q / np.linalg.norm(q)
        raw = vec3_to_cell_raw(q, res)
        assert _locate(q, res) == _polish(q, raw)
