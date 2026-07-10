"""Directed edges: round-trip laws, ring alignment, counts.

The round-trip laws are the tests that pin ``step``'s arrow sign
conventions (deliberately not derived on paper): ``dest(reverse(e)) ==
origin(e)`` fails for ANY wrong rotation, since all of a cell's
outgoing directions lead to distinct neighbors. They are also the
tests that would have caught the April note's naive ``-d`` reverse,
which breaks at every base-cell seam.
"""

import random

import numpy as np
import pytest

from mu3 import cell_ring1, cells_at_res, is_pentagon
from mu3.cell import cell_boundary
from mu3.edge import (
    DirectedEdge,
    UndirectedEdge,
    corner_leaving_edge,
    directed_edges_of_cell,
    edge_corner_digits,
    edge_to_boundary,
    opposite,
    outgoing_directions,
    undirected_edges_of_cell,
)
from mu3.neighbor import step
from conftest import random_valid_cells
from mu3.vertex import (
    edge_to_vertices,
    edge_vertices,
    vertex_to_vec3,
    vertices_of_cell,
)


def _all_cells():
    for res in [0, 1, 2, 3]:
        yield from cells_at_res(res)


def _all_edges():
    for c in _all_cells():
        yield from directed_edges_of_cell(c)


def test_edge_counts():
    """5 outgoing at pentagons, 6 at hexes; 6C - 12 in total."""
    for res in [0, 1, 2, 3]:
        total = 0
        n_cells = 0
        for c in cells_at_res(res):
            n_cells += 1
            edges = directed_edges_of_cell(c)
            assert len(edges) == (5 if is_pentagon(c) else 6), c
            total += len(edges)
        assert total == 6 * n_cells - 12


def test_dest_matches_ring_order():
    """The i-th outgoing edge's dest is cell_ring1's i-th neighbor —
    the boundary-edge/ring alignment index._polish relies on."""
    for c in _all_cells():
        assert [e.dest() for e in directed_edges_of_cell(c)] \
            == cell_ring1(c), c


def test_reverse_points_back():
    """dest(reverse(e)) == origin(e) — the arrow-transport law.
    (origin(reverse(e)) == dest(e) holds by construction: reverse's
    cell IS the step's destination.)"""
    for e in _all_edges():
        r = e.reverse()
        assert r.dest() == e.cell, (e, r)


def test_reverse_roundtrip():
    for e in _all_edges():
        assert e.reverse().reverse() == e, e


def test_rotate_ccw_cycles():
    """Single CCW rotations enumerate all outgoing edges exactly once
    before returning to the start (period 6 at hexes, 5 at pentagons)."""
    for c in _all_cells():
        edges = directed_edges_of_cell(c)
        e = edges[0]
        seen = [e]
        while True:
            e = e.rotate_ccw()
            if e == seen[0]:
                break
            seen.append(e)
        assert sorted(x.d for x in seen) == sorted(x.d for x in edges), c


def test_validation():
    with pytest.raises(ValueError):
        DirectedEdge((0,), 1)          # deleted direction, res-0 pentagon
    with pytest.raises(ValueError):
        DirectedEdge((0, 0), 1)        # deleted direction, pentagon center
    with pytest.raises(ValueError):
        DirectedEdge((0, 2), 0)        # not a direction
    with pytest.raises(ValueError):
        DirectedEdge((0, 2), 7)
    with pytest.raises(ValueError):
        DirectedEdge((0, 1), 2)        # invalid cell (leading d=1)
    DirectedEdge((0, 2), 1)            # hex d=1 is a real direction


def test_opposite():
    assert [opposite(d) for d in (1, 2, 3, 4, 5, 6)] == [4, 5, 6, 1, 2, 3]


@pytest.mark.parametrize('b', range(12))
def test_interior_steps_have_zero_rot(b):
    """A cell well inside its pentagon's territory, away from the cut:
    every step stays same-base with rot == 0 (the arrow is trivial in
    the flat interior)."""
    cell = (b, 3, 0)
    for d in (1, 2, 3, 4, 5, 6):
        dest, rot = step(cell, d)
        assert dest[0] == b, (cell, d, dest)
        assert rot == 0, (cell, d, rot)


# --- undirected edges (stage 2) ----------------------------------------


def test_undirected_orientation_independence():
    """Both directed orientations construct the SAME canonical
    undirected edge — the Z/2 quotient in one equality."""
    for e in _all_edges():
        r = e.reverse()
        assert UndirectedEdge(e.cell, e.d) == UndirectedEdge(r.cell, r.d)


def test_undirected_lex_min_invariant():
    for c in _all_cells():
        for u in undirected_edges_of_cell(c):
            origin, dest = u.endpoints()
            assert origin < dest, u


def test_undirected_cardinality():
    """E = 3C - 6 at every resolution."""
    for res in [0, 1, 2, 3]:
        n_cells = 0
        edges = set()
        for c in cells_at_res(res):
            n_cells += 1
            edges.update(undirected_edges_of_cell(c))
        assert len(edges) == 3 * n_cells - 6


def test_directed_orientations():
    for c in cells_at_res(1):
        for u in undirected_edges_of_cell(c):
            fwd, back = u.directed_orientations()
            assert (fwd.cell, fwd.d) == (u.cell, u.d)
            assert back.reverse() == fwd
            assert set(u.endpoints()) == {fwd.cell, back.cell}


@pytest.mark.parametrize('b', range(12))
def test_seam_reverse_regression(b):
    """Cross-base and stitch-crossing edges — the populations where the
    naive -d reverse is wrong. (b, 6) -> (q0, 2) crosses a hop-then-
    stitch seam (the post-hop-wedge killer family)."""
    q0_2 = DirectedEdge((b, 6), 6).dest()
    for e in directed_edges_of_cell((b, 6)):
        r = e.reverse()
        assert r.dest() == (b, 6), (e, r)
    assert q0_2 in cell_ring1((b, 6))


# --- wire-pair fast paths: edge_to_boundary / edge_to_vertices ---------
#
# The geometry tier (edge_to_boundary: no identity objects, one exact
# center) must agree with the boundary rows and with the identity tier
# (edge_to_vertices -> vertex_to_vec3). Agreement is to float tolerance,
# not bitwise: canonical vertex names can develop through a different
# chart (other cell, or the pentagon d=1 alias), and cross-chart corner
# projections agree exactly in the algebra but to ~1e-15 in floats.


def _assert_close(a, b):
    assert np.allclose(a, b, rtol=0.0, atol=1e-12), (a, b)


def test_edge_to_boundary_matches_cell_boundary_rows():
    """Each edge's [tail, head] == the matching cell_boundary rows,
    located by canonical vertex identity (vertices_of_cell is aligned
    with boundary row order)."""
    for res in [0, 1, 2]:
        for c in cells_at_res(res):
            B = cell_boundary(c, closed=False)
            verts = vertices_of_cell(c)
            for d in outgoing_directions(c):
                seg = edge_to_boundary(c, d)
                tail, head = edge_to_vertices(c, d)
                _assert_close(seg[0], B[verts.index(tail)])
                _assert_close(seg[1], B[verts.index(head)])


def test_edge_to_boundary_matches_identity_tier():
    """Geometry tier == identity tier positions, on every res-1 edge
    plus seeded random cells at res 3-4."""
    rng = random.Random(0)
    cells = list(cells_at_res(1))
    for res in (3, 4):
        cells.extend(random_valid_cells(rng, res, 40))
    for c in cells:
        for d in outgoing_directions(c):
            seg = edge_to_boundary(c, d)
            tail, head = edge_to_vertices(c, d)
            _assert_close(seg[0], vertex_to_vec3(tail.cell, tail.d))
            _assert_close(seg[1], vertex_to_vec3(head.cell, head.d))


def test_edge_to_vertices_matches_edge_vertices():
    for c in cells_at_res(1):
        for e in directed_edges_of_cell(c):
            assert edge_to_vertices(e.cell, e.d) == edge_vertices(e)


def test_wire_pair_validation():
    for fn in (edge_to_boundary, edge_to_vertices):
        with pytest.raises(ValueError):
            fn((0, 0), 1)        # pentagon: d=1 is the deleted direction
        with pytest.raises(ValueError):
            fn((0, 2), 0)        # no direction 0
        with pytest.raises(ValueError):
            fn((0, 2), 7)        # no direction 7
        with pytest.raises(ValueError):
            fn((0, 1), 2)        # invalid cell (leading digit 1)


def test_corner_leaving_edge_inverts_tail():
    """corner_leaving_edge is the tail-map inverse, total over ALL
    corner names: for every outgoing edge, the tail corner's leaving
    edge is that edge back; at pentagon cut corners both same-cell
    names (canonical 6, alias 1) answer the d=2 edge — the stitch
    making the inverse total is what lets a single pentagon's
    boundary ring close."""
    for res in [0, 1, 2]:
        for c in cells_at_res(res):
            for d in outgoing_directions(c):
                tail, _ = edge_corner_digits(d)
                assert corner_leaving_edge(c, tail) == (c, d)
            if is_pentagon(c):
                assert corner_leaving_edge(c, 6) == (c, 2)
                assert corner_leaving_edge(c, 1) == (c, 2)
