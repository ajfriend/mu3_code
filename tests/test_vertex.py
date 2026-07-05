"""Vertices: Z/3 orbit laws, cardinality, Euler, position agreement.

The orbit tests are what pin the orbit-step offset ``rot + 2``
(``rho^3 == id`` plus 3-distinct-cells fails for every other offset —
swept during stage 3), the same method that pinned ``step``'s arrow
signs via the edge round-trips. Law sweeps run res 0-2 plus the res-3
zoo families; cardinality runs res 0-3.
"""

import cmath

import numpy as np
import pytest

from adversarial import cut_end_corner_names, law_sweep_cells
from mu3 import cell_boundary, cells_at_res, is_pentagon
from mu3.cell import _project
from mu3.edge import undirected_edges_of_cell
from mu3.eisenstein import scaled_corner
from mu3.vertex import (
    Vertex,
    _normalize,
    _orbit,
    _orbit_step,
    vertex_directions,
    vertex_to_vec3,
    vertices_of_cell,
)


def test_orbit_closes_with_three_distinct_cells():
    """rho^3 == id and every corner is shared by 3 DISTINCT cells —
    the law pair that pins the rot+2 offset uniquely."""
    for c in law_sweep_cells():
        for d in vertex_directions(c):
            r1, r2, r3 = _orbit(_normalize(c, d))
            assert _orbit_step(*r3) == r1, (c, d)
            assert len({r1[0], r2[0], r3[0]}) == 3, (c, d)


def test_reps_canonicalize_identically():
    """All three orbit representatives (each a normalized name — never
    a pentagon d=1) construct the same canonical Vertex."""
    for c in law_sweep_cells():
        for d in vertex_directions(c):
            v = Vertex(c, d)
            for rc, rd in _orbit(_normalize(c, d)):
                assert rd in vertex_directions(rc), (c, d, rc, rd)
                assert Vertex(rc, rd) == v, (c, d, rc, rd)


@pytest.mark.parametrize('res', [0, 1, 2, 3])
def test_vertex_cardinality(res):
    """V = 2C - 4; 5 distinct vertices per pentagon, 6 per hex."""
    verts = set()
    n_cells = 0
    for c in cells_at_res(res):
        n_cells += 1
        vs = vertices_of_cell(c)
        assert len(set(vs)) == len(vs) == (5 if is_pentagon(c) else 6), c
        verts.update(vs)
    assert len(verts) == 2 * n_cells - 4


@pytest.mark.parametrize('res', [0, 1, 2])
def test_euler(res):
    """C - E + V == 2 from the actual canonical sets (res 3 is covered
    by the cardinality tests: 3C-6 in test_edge, 2C-4 above)."""
    cells = list(cells_at_res(res))
    edges = {e for c in cells for e in undirected_edges_of_cell(c)}
    verts = {v for c in cells for v in vertices_of_cell(c)}
    assert len(cells) - len(edges) + len(verts) == 2


def test_positions_agree_across_reps():
    """The corner developed in each incident cell's own chart projects
    to the same 3D point — exact arithmetic upstream, float only at
    projection, so agreement is at projection-noise scale."""
    for c in law_sweep_cells():
        for d in vertex_directions(c):
            p1, p2, p3 = (
                vertex_to_vec3(*r) for r in _orbit(_normalize(c, d))
            )
            assert np.linalg.norm(p1 - p2) < 1e-12, (c, d)
            assert np.linalg.norm(p1 - p3) < 1e-12, (c, d)


def test_positions_match_cell_boundary():
    """vertex_directions/vertices_of_cell follow cell_boundary row
    order — the alignment that lets exact vertex keys stand in for
    boundary-row float keys."""
    for c in law_sweep_cells():
        B = cell_boundary(c, closed=False)
        ds = vertex_directions(c)
        assert len(B) == len(ds), c
        for row, d in zip(B, ds):
            assert np.linalg.norm(row - vertex_to_vec3(c, d)) < 1e-12, (c, d)


@pytest.mark.parametrize('b', range(12))
def test_interior_corner_exact_identity(b):
    """Flat-interior corners: all three reps develop in the same chart
    with trivial arrows, so their exact S3-scaled corners agree by
    integer equality (the same-frame case of the consistency spec)."""
    cell = (b, 3, 0)
    for d in vertex_directions(cell):
        reps = _orbit(_normalize(cell, d))
        assert {rc[0] for rc, _ in reps} == {b}
        assert len({scaled_corner(rc[1:], rd) for rc, rd in reps}) == 1, \
            (cell, d)


@pytest.mark.parametrize('res', [0, 1, 2, 3])
def test_pentagon_cut_corner_alias(res):
    """(pent, 1) — the post-stitch name of the cut corner — folds onto
    the canonical (pent, 6)."""
    for b in range(12):
        pent = (b,) + (0,) * res
        assert Vertex(pent, 1) == Vertex(pent, 6)


def test_validation():
    with pytest.raises(ValueError):
        Vertex((0, 1), 2)      # invalid cell (leading d=1)
    with pytest.raises(ValueError):
        Vertex((0, 2), 0)      # not a corner name
    with pytest.raises(ValueError):
        Vertex((0, 2), 7)
    # vertex_to_vec3 enforces the same contract: d=0 must not
    # silently project the cell CENTER, d=7 must not KeyError.
    with pytest.raises(ValueError):
        vertex_to_vec3((0, 2), 0)
    with pytest.raises(ValueError):
        vertex_to_vec3((0, 2), 7)
    with pytest.raises(ValueError):
        vertex_to_vec3((0, 1), 2)


@pytest.mark.parametrize('res', [1, 2, 3])
def test_zoo_cut_end_corners(res):
    """The seam-end vertex: its position is the post-stitch image of
    the deleted icosa-face center, and its three incident cells live
    in three different base charts."""
    face_center = (1 + cmath.exp(1j * cmath.pi / 3)) / 3
    for c, d in cut_end_corner_names(res):
        v = Vertex(c, d)
        assert len({b for b, *_ in v.incident_cells()}) == 3, (c, d)
        target = _project(face_center, c[0])
        assert np.linalg.norm(vertex_to_vec3(c, d) - target) < 1e-12, (c, d)
