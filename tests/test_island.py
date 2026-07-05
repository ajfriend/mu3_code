"""Gosper island boundary iterator: set-completeness against a
brute-force oracle + cycle laws, zoo parents.

Two facts pin the fitted constants (_TRIG/_START) completely, with no
walk-based oracle needed:

- SET equality vs brute force (every descendant x 6 directions,
  keep edges whose walked destination leaves the subtree) — the
  output is exactly the boundary set, including any hypothetical
  second component a boundary WALK could never reach;
- the chaining law (test_chaining_law): boundary vertices have
  exactly two incident boundary segments (any in/out split of a
  vertex's 3 cells gives exactly 2 mixed pairs — pinches are
  impossible), so a closed, distinct, chained sequence has a UNIQUE
  successor at every step: the order is fully rigid given the set.
"""

import itertools

import pytest

from adversarial import (
    cut_end_corner_names,
    pentagon_cells,
    phantom_corner_cells,
)
from mu3 import cells_at_res, is_valid_cell
from mu3.edge import DirectedEdge, directed_edges_of_cell
from mu3.island import island_edges
from mu3.neighbor import step
from mu3.vertex import edge_vertices


def _is_desc(c, ancestor):
    return c[:len(ancestor)] == ancestor


def _boundary_set(ancestor, res):
    """Brute-force oracle: every boundary edge, by enumeration."""
    out = set()
    for tail in itertools.product(range(7), repeat=res - len(ancestor) + 1):
        c = ancestor + tail
        if not is_valid_cell(c):
            continue
        for d in (1, 2, 3, 4, 5, 6):
            dest, _ = step(c, d)
            if not _is_desc(dest, ancestor):
                out.add((c, d))
    return out


@pytest.mark.parametrize('res', [0, 1, 2])
def test_boundary_set_exhaustive(res):
    """Fast walk yields exactly the brute-force boundary set (each
    edge once) for every parent cell, deltas 1..3 at coarse parents,
    1..2 at res 2 — hexes and pentagons."""
    max_delta = 3 if res < 2 else 2
    for parent in cells_at_res(res):
        for delta in range(1, max_delta + 1):
            got = list(island_edges(parent, res + delta))
            assert len(set(got)) == len(got), (parent, delta)
            assert set(got) == _boundary_set(parent, res + delta), \
                (parent, delta)


def test_zoo_parents_boundary_set():
    """Adversarial parents at res 3: phantom-corner cells, cut-end
    owners, pentagon centers — islands whose edges cross seams."""
    parents = list(phantom_corner_cells())
    parents += [c for c, _ in cut_end_corner_names(res=3)]
    parents += pentagon_cells(res=3)
    for parent in parents:
        got = list(island_edges(parent, 5))
        assert len(set(got)) == len(got), parent
        assert set(got) == _boundary_set(parent, 5), parent


def test_count_law():
    """6 * 3^delta edges for hex parents, 5 * 3^delta for pentagons —
    the Gosper perimeter law, fast iterator only (no walks)."""
    for parent, sides in [((0, 3), 6), ((0, 0), 5), ((7,), 5)]:
        for delta in range(5):
            res = len(parent) - 1 + delta
            edges = list(island_edges(parent, res))
            assert len(edges) == sides * 3 ** delta, (parent, delta)
            assert len(set(edges)) == len(edges), (parent, delta)


def test_chaining_law():
    """Consecutive edges chain head -> tail as EXACT Vertex identities,
    around the full loop (wrap included) — ties the island walk to the
    stage-4 combinatorial map. Includes a seam-heavy pentagon parent
    and a phantom-corner parent."""
    for parent in [(0, 3), (0, 0), (0, 2, 6, 6)]:
        res = len(parent) + 1
        edges = [DirectedEdge(c, d) for c, d in island_edges(parent, res)]
        for e1, e2 in zip(edges, edges[1:] + edges[:1]):
            assert edge_vertices(e1)[1] == edge_vertices(e2)[0], (e1, e2)


def test_delta_zero_is_own_edges():
    for cell in [(0, 3), (4, 0), (9,)]:
        got = list(island_edges(cell, len(cell) - 1))
        want = [(e.cell, e.d) for e in directed_edges_of_cell(cell)]
        assert got == want, cell


def test_validation():
    with pytest.raises(ValueError):
        list(island_edges((0, 1), 3))    # invalid cell
    with pytest.raises(ValueError):
        list(island_edges((0, 3, 2), 1))  # res above parent
