"""Point location via the exact holonomy entry point, judged by the
implementation-agnostic ground truth: the returned cell CONTAINS the
query point (spherical point-in-polygon against the cell's own
geodesic boundary). No comparison to any other implementation.
"""

import cmath
import math

import numpy as np
import pytest

from mu3 import vec3_to_cell
from mu3.cell import (
    _polish_cell_sphere,
    _project,
    cell_boundary,
    cell_center,
    cells_at_res,
)

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
