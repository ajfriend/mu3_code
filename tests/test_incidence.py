"""Cross-object incidence laws: cells x edges x vertices cohere as
one combinatorial map.

Six overlapping layers, so a defect in any one relation is caught
from several independent directions:

1. elementwise formula laws (per directed edge): twin symmetry,
   endpoint incidence;
2. vertex-star laws (per vertex): in/out heads and tails, the twin
   bijection, pairwise-adjacency of the incident cells;
3. closure laws: the face cycle (``rotate_ccw``) and the vertex cycle
   (``DirectedEdge.next_around_vertex``, period 3) — the
   combinatorial-map coherence that ties reverse, rotation, and the
   Z/3 orbit together;
4. global partition/counting laws: head/tail triples partition all
   ``6C - 12`` directed edges; the 3V == 2E handshake from actual
   sets (res-3 cardinalities are pinned in test_edge/test_vertex);
5. geometric cross-checks: edge corners lie on BOTH endpoint cells'
   drawn boundaries, as cyclically consecutive rows in CCW order;
6. named zoo stars: the pentagon cut corner (where the d=1 alias
   normalization is what closes the star across the seam), cut-end
   corners, phantom corners.

Sweeps use ``adversarial.law_sweep_cells`` (res 0-2 full + res-3 zoo).
"""

from functools import lru_cache

import numpy as np
import pytest

from adversarial import (
    cut_end_corner_names,
    law_sweep_cells,
    phantom_corner_cells,
)
from mu3 import cell_boundary, cells_at_res
from mu3.edge import DirectedEdge, corner_leaving_edge, directed_edges_of_cell
from mu3.eisenstein import DIGIT_OFFSET, ZETA
from mu3.vertex import (
    Vertex,
    edge_vertices,
    vertex_to_vec3,
    vertices_of_cell,
)


@lru_cache(maxsize=None)
def _sweep_vertices() -> tuple:
    """All distinct vertices of the law sweep, computed once."""
    return tuple({v for c in law_sweep_cells() for v in vertices_of_cell(c)})


@lru_cache(maxsize=None)
def _ev(e: DirectedEdge) -> tuple:
    """Cached edge_vertices — the sweep layers revisit the same edges
    (4 walks per uncached call)."""
    return edge_vertices(e)


# --- layer 1: elementwise edge laws -------------------------------------


def test_edge_vertices_twin_law():
    """An edge's two corners are distinct corners of its origin cell,
    and the reverse edge sees the SAME pair, swapped."""
    for c in law_sweep_cells():
        vs = set(vertices_of_cell(c))
        for e in directed_edges_of_cell(c):
            tail, head = _ev(e)
            assert tail != head, e
            assert tail in vs and head in vs, e
            assert _ev(e.reverse()) == (head, tail), e


def test_edge_endpoint_incidence():
    """The edge's two cells are exactly the SHARED incident cells of
    its endpoint corners; four cells ring a segment."""
    for c in law_sweep_cells():
        for e in directed_edges_of_cell(c):
            tail, head = _ev(e)
            a = set(tail.incident_cells())
            b = set(head.incident_cells())
            assert a & b == {e.cell, e.dest()}, e
            assert len(a | b) == 4, e


# --- layer 2: vertex-star laws ------------------------------------------


def test_vertex_edge_stars():
    """edges_in heads and edges_out tails are the vertex itself;
    reverse is a bijection between the two stars; origin cells of the
    entering star are the incident cells."""
    for v in _sweep_vertices():
        ein, eout = v.edges_in(), v.edges_out()
        assert all(_ev(e)[1] == v for e in ein), v
        assert all(_ev(e)[0] == v for e in eout), v
        assert {e.reverse() for e in ein} == set(eout), v
        assert {e.cell for e in ein} == set(v.incident_cells()), v


def test_vertex_undirected_edges_are_pairwise_adjacencies():
    """The 3 segments at a corner connect exactly the consecutive
    pairs of its 3 incident cells."""
    for v in _sweep_vertices():
        cells = v.incident_cells()
        pairs = {frozenset((cells[i], cells[(i + 1) % 3])) for i in range(3)}
        got = {frozenset(u.endpoints()) for u in v.undirected_edges()}
        assert got == pairs, v


def test_adjacent_vertices():
    """3 distinct neighbors, none self; each shares exactly one
    segment (= exactly 2 incident cells) with v; adjacency is
    symmetric."""
    for v in _sweep_vertices():
        adj = v.adjacent_vertices()
        assert len(set(adj)) == 3 and v not in adj, v
        vic = set(v.incident_cells())
        for w in adj:
            assert len(vic & set(w.incident_cells())) == 2, (v, w)
            assert v in w.adjacent_vertices(), (v, w)


# --- layer 3: closure laws ----------------------------------------------


def test_face_cycle():
    """Iterating rotate_ccw walks the cell's boundary: the heads are
    exactly vertices_of_cell, all distinct, and the cycle closes with
    the boundary period (6 at hexes, 5 at pentagons)."""
    for c in law_sweep_cells():
        edges = directed_edges_of_cell(c)
        e = edges[0]
        heads = []
        for _ in range(len(edges)):
            heads.append(_ev(e)[1])
            e = e.rotate_ccw()
        assert e == edges[0], c
        assert len(set(heads)) == len(edges), c
        assert set(heads) == set(vertices_of_cell(c)), c


def test_vertex_cycle_sigma():
    """sigma^3 == id around every corner, and the sigma-orbit of any
    entering edge is exactly the entering star — the coherence law
    binding reverse, rotate_ccw, and the Z/3 vertex orbit."""
    for v in _sweep_vertices():
        ein = v.edges_in()
        e1 = ein[0]
        e2 = e1.next_around_vertex()
        e3 = e2.next_around_vertex()
        assert e3.next_around_vertex() == e1, v
        assert {e1, e2, e3} == set(ein), v


# --- layer 4: global partition / counting laws ---------------------------


@pytest.mark.parametrize('res', [0, 1, 2])
def test_directed_edge_head_partition(res):
    """e -> head(e) and e -> tail(e) each partition ALL 6C - 12
    directed edges into triples, and the triples are exactly the
    vertex stars."""
    heads: dict = {}
    tails: dict = {}
    n_cells = 0
    for c in cells_at_res(res):
        n_cells += 1
        for e in directed_edges_of_cell(c):
            tail, head = _ev(e)
            heads.setdefault(head, set()).add(e)
            tails.setdefault(tail, set()).add(e)
    assert sum(len(s) for s in heads.values()) == 6 * n_cells - 12
    assert all(len(s) == 3 for s in heads.values())
    assert all(len(s) == 3 for s in tails.values())
    assert heads.keys() == tails.keys()
    for v, s in heads.items():
        assert s == set(v.edges_in()), v
        assert tails[v] == set(v.edges_out()), v


@pytest.mark.parametrize('res', [0, 1, 2])
def test_handshake_from_sets(res):
    """Every undirected edge is reachable through some vertex's star
    (no orphan segments), and the counts satisfy the 3V == 2E
    handshake alongside E = 3C - 6, V = 2C - 4."""
    cells = list(cells_at_res(res))
    verts = {v for c in cells for v in vertices_of_cell(c)}
    edges = {u for v in verts for u in v.undirected_edges()}
    assert len(edges) == 3 * len(cells) - 6
    assert len(verts) == 2 * len(cells) - 4
    assert 3 * len(verts) == 2 * len(edges)


# --- layer 5: geometric cross-checks ------------------------------------


@lru_cache(maxsize=None)
def _bnd(cell: tuple) -> np.ndarray:
    return cell_boundary(cell, closed=False)


def _row_of(p: np.ndarray, B: np.ndarray) -> int:
    dists = np.linalg.norm(B - p, axis=1)
    k = int(np.argmin(dists))
    assert dists[k] < 1e-12
    return k


def test_edge_corners_are_consecutive_boundary_rows():
    """An edge's corners are cyclically CONSECUTIVE boundary rows of
    its origin cell in CCW order — tail immediately before head —
    checked by EXACT vertex identity against vertices_of_cell's row
    alignment. Geometrically, both corners also lie on the DEST
    cell's drawn boundary (a float check by necessity: the canonical
    positions may come from a third chart)."""
    for c in law_sweep_cells():
        vs = vertices_of_cell(c)   # cell_boundary row order, exact
        n = len(vs)
        for e in directed_edges_of_cell(c):
            tail, head = _ev(e)
            assert (vs.index(tail) + 1) % n == vs.index(head), e
            B2 = _bnd(e.dest())
            _row_of(vertex_to_vec3(tail.cell, tail.d), B2)
            _row_of(vertex_to_vec3(head.cell, head.d), B2)


# --- layer 6: named zoo stars -------------------------------------------


@pytest.mark.parametrize('res', [0, 1, 2, 3])
def test_zoo_pentagon_cut_corner_star(res):
    """The cut corner's star closes across the seam: the d=1 -> d=6
    alias normalization is the load-bearing mechanism — without it,
    sigma and the twin bijection would produce the deleted direction
    at the pentagon."""
    for b in range(12):
        v = Vertex((b,) + (0,) * res, 6)
        ein = v.edges_in()
        assert all(_ev(e)[1] == v for e in ein), v
        assert {e.reverse() for e in ein} == set(v.edges_out()), v
        e = ein[0]
        sig3 = e.next_around_vertex().next_around_vertex() \
                .next_around_vertex()
        assert sig3 == e, v


def test_zoo_cut_end_and_phantom_corner_stars():
    """Stars whose three incident cells live in three different base
    charts (cut-end corners) and the 3-hex phantom corners: sigma^3
    and the twin bijection hold through cross-pentagon arrows."""
    verts = {Vertex(c, d) for c, d in cut_end_corner_names(res=3)}
    verts |= {v for c in phantom_corner_cells() for v in vertices_of_cell(c)}
    for v in verts:
        ein = v.edges_in()
        sig3 = ein[0].next_around_vertex().next_around_vertex() \
                     .next_around_vertex()
        assert sig3 == ein[0], v
        assert {e.reverse() for e in ein} == set(v.edges_out()), v


def test_stitch_zeta_unifies_the_folds():
    """The two pentagon alias folds are one fact: the stitch is
    multiplication by ζ, identifying each pentagon object with its
    ζ-image. Exactly: ζ·offset(6) == offset(1) (the cut corner's two
    names) and ζ·offset(1) == offset(2) (the deleted direction's
    image). The code folds realize those identifications — corner
    names 1 → 6 (``vertex._normalize``, via the Vertex constructor)
    and directions 1 → 2 (``edge.corner_leaving_edge``); both are
    pure functions of pentagon-ness, so one pentagon suffices (the
    folds are pinned exhaustively in
    ``test_edge.test_corner_leaving_edge_inverts_tail``)."""
    assert ZETA * DIGIT_OFFSET[6] == DIGIT_OFFSET[1]
    assert ZETA * DIGIT_OFFSET[1] == DIGIT_OFFSET[2]
    pent = (0, 0)
    assert Vertex(pent, 1) == Vertex(pent, 6)
    assert corner_leaving_edge(pent, 1) == corner_leaving_edge(pent, 6) \
        == (pent, 2)
