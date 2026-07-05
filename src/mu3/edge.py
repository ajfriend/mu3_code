"""Directed and undirected edges: first-class ``(cell, d)`` pairs.

A directed
edge names a cell together with one of its outgoing digit directions
(``d ∈ {1..6}``; ``{2..6}`` at pentagons, where ``d=1`` is the deleted
direction). An undirected edge is the orbit of a directed edge under
``reverse``, represented canonically with the lex-smaller endpoint as
origin. ``reverse`` is where the holonomy machinery earns its
keep: direction indices are frame-relative, so the reverse direction
is ``opposite(d)`` transported by the walk's arrow rotation —
``rotate_digit_ccw(opposite(d), rot)`` with ``rot`` from
:func:`mu3.neighbor.step`. The naive ``-d`` flip is wrong at every
base-cell seam (see the note's "Corrections" section).

Edge iteration order aligns with :func:`mu3.neighbor.cell_ring1` and
with boundary-edge indices: ``directed_edges_of_cell(c)[i].dest() ==
cell_ring1(c)[i]``, which is the alignment ``index._polish`` relies
on.
"""

from dataclasses import dataclass
from typing import Sequence

from .cell import is_pentagon, is_valid_cell
from .face_lattice import rotate_digit_ccw
from .neighbor import step


def opposite(d: int) -> int:
    """The 180-degree-opposite digit direction (same frame)."""
    return rotate_digit_ccw(d, 3)


def outgoing_directions(cell: Sequence[int]) -> tuple[int, ...]:
    """The digit directions with an edge out of ``cell``, in CCW walk
    order: pentagons lack the deleted ``d=1``. The single home of that
    rule inside this module."""
    return (2, 3, 4, 5, 6) if is_pentagon(cell) else (1, 2, 3, 4, 5, 6)


@dataclass(frozen=True, slots=True)
class DirectedEdge:
    """``cell`` -> its ring-1 neighbor in digit direction ``d``."""

    cell: tuple
    d: int

    def __post_init__(self):
        cell_t = tuple(int(x) for x in self.cell)
        object.__setattr__(self, 'cell', cell_t)
        object.__setattr__(self, 'd', int(self.d))
        if not is_valid_cell(cell_t):
            raise ValueError(f'DirectedEdge: invalid cell {cell_t}')
        if self.d not in outgoing_directions(cell_t):
            raise ValueError(
                f'DirectedEdge: no direction {self.d} out of {cell_t}'
            )

    def dest(self) -> tuple:
        c, _ = step(self.cell, self.d)
        return c

    def reverse(self) -> 'DirectedEdge':
        """The same undirected edge, traversed the other way. One walk;
        the direction is opposite(d) transported by the arrow."""
        c, rot = step(self.cell, self.d)
        return DirectedEdge(c, rotate_digit_ccw(opposite(self.d), rot))

    def rotate_ccw(self, steps: int = 1) -> 'DirectedEdge':
        """The next outgoing edge(s) CCW around the origin cell —
        period 6 at hexes, 5 at pentagons (the deleted direction is
        not in the cycle)."""
        ds = outgoing_directions(self.cell)
        i = ds.index(self.d)
        return DirectedEdge(self.cell, ds[(i + steps) % len(ds)])

    def next_around_vertex(self) -> 'DirectedEdge':
        """The next edge entering the same head vertex (the corner
        this edge runs INTO — see ``mu3.vertex.edge_vertices``): twin,
        then one step CW around the new origin cell. Period 3 at every
        corner, pentagon cut corners included — the vertex permutation
        that, with ``reverse`` (twin) and ``rotate_ccw`` (face), makes
        the grid a combinatorial map. Laws pinned in
        ``tests/test_incidence.py``."""
        return self.reverse().rotate_ccw(-1)


def directed_edges_of_cell(cell: Sequence[int]) -> list[DirectedEdge]:
    """The 6 (hex) or 5 (pentagon) outgoing directed edges, in walk
    order — ``[e.dest() for e in ...] == cell_ring1(cell)``."""
    cell_t = tuple(int(x) for x in cell)
    return [DirectedEdge(cell_t, d) for d in outgoing_directions(cell_t)]


@dataclass(frozen=True, slots=True)
class UndirectedEdge:
    """The unordered cell pair a directed edge connects.

    Canonical form: the lex-smaller endpoint is the origin (lex-min is
    frame-independent and pentagon-uniform, unlike any fixed
    half-direction section — see the design note). The constructor
    accepts EITHER orientation and canonicalizes, at the cost of one
    walk — so ``UndirectedEdge(e.cell, e.d) ==
    UndirectedEdge(*reverse_of_e)`` by construction, and equality/hash
    are plain field compares.
    """

    cell: tuple
    d: int

    def __post_init__(self):
        e = DirectedEdge(self.cell, self.d)   # validates cell + d
        r = e.reverse()                       # one walk
        if r.cell < e.cell:
            e = r
        object.__setattr__(self, 'cell', e.cell)
        object.__setattr__(self, 'd', e.d)

    def directed_orientations(self) -> tuple[DirectedEdge, DirectedEdge]:
        """The two directed representatives: canonical first."""
        e = DirectedEdge(self.cell, self.d)
        return (e, e.reverse())

    def endpoints(self) -> tuple[tuple, tuple]:
        """The two cells, canonical origin first."""
        fwd, back = self.directed_orientations()
        return (fwd.cell, back.cell)


def undirected_edges_of_cell(cell: Sequence[int]) -> list[UndirectedEdge]:
    """The 6 (hex) or 5 (pentagon) undirected edges incident to
    ``cell``, in walk order of their outgoing directed forms."""
    cell_t = tuple(int(x) for x in cell)
    return [UndirectedEdge(cell_t, d) for d in outgoing_directions(cell_t)]
