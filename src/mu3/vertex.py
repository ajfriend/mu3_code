"""Vertices: first-class ``(cell, d)`` corner names.

``(cell, d)``
names the corner of ``cell`` at flat position ``center + eps/s3``
(``eps`` the unit at digit ``d``'s angle, ``s3 = 1 - omega``) — the
corner between the neighbor directions ``d`` and ``d`` + 60 degrees.

A name is an INSTRUCTION — a starting cell plus a direction — not a
label. Every instruction is valid and denotes a point; several
instructions reach the same point, and canonicalization
(:class:`Vertex`) evaluates an instruction to the distinguished name
of its endpoint. Name-consuming operations either canonicalize or are
alias-aware (``edge.corner_leaving_edge``); nothing rejects a name
for being non-canonical. (The same discipline as phantom cell paths:
digit strings are instructions too, and some points admit several.)

Every corner is 3-valent (pentagon corners included), so a vertex is
the ``Z/3`` orbit ``{(c, eps), (c + eps, eps*omega),
(c + eps*(1 + omega), eps*omega**2)}`` — one instruction per incident
cell; the canonical representative is the minimum ``(cell, d)`` pair
(lex-min cell, smallest-d tie-break). The orbit step is one
arrow-transported walk — ``(dest, rotate_digit_ccw(d, rot + 2))``
with ``(dest, rot) = step(cell, d)`` — the same shape as
``DirectedEdge.reverse`` (offset 3); the offset 2 is
``omega = zeta**2`` read through the arrow, and is pinned by the
orbit tests (``rho^3 == id`` fails for every other offset).

Pentagon cut-corner alias: at a pentagon(-center) cell the corner
names ``d=6`` (pre-stitch, in the deleted wedge) and ``d=1``
(post-stitch) develop to the same surface corner — the one corner
where two instructions share a starting cell (the +60-degree stitch
identifies them), which is exactly why per-incident-cell enumeration
(the orbit) needs the alias fold. ``d=6`` is canonical, keeping
"pentagons have no ``d=1``" uniform with ``mu3.edge``; constructors
accept and normalize the ``d=1`` form.

Note the pair ``(cell, d)`` also serves as an EDGE instruction (a
full step to a neighbor), which is NOT total: the deleted ``d=1`` has
no neighbor on a pentagon (``DirectedEdge`` rejects what ``Vertex``
accepts). Corner instructions are total, edge instructions are not —
the two readings diverge exactly at the cut.

Corners live on the aperture-3 refinement of the cell lattice:
``eisenstein.scaled_corner`` (``s3*Z_c + eps``) is an exact ``Eis``,
so vertex identity and dedup are integer equality — floats appear only
at the final ``_project`` to 3D (the float-quarantine invariant, see
``mu3.traversal``). ``vertices_of_cell`` follows ``cell_boundary`` row
order (corner ``k`` at angle ``k * 60`` degrees <-> digit
``UNIT_DIGITS[k]``), so its exact vertex keys can replace float dedup
keys wherever boundary rows are the unit of work
(``tests/test_one_hop_contract.py`` does exactly that).

Stage 4 — the edge<->vertex incidence block — completes the system
into a combinatorial map (a DCEL): ``DirectedEdge`` is the half-edge,
``reverse`` its twin, ``rotate_ccw`` the next half-edge around the
face (origin cell), and traversed CCW around its origin cell the edge
``(c, d)`` runs from corner ``rotate_digit_ccw(d, 5)`` to corner ``d``
(:func:`edge_vertices`). The induced vertex permutation
``DirectedEdge.next_around_vertex`` cycles the 3 entering edges of
every vertex with period 3 — across the pentagon cut too, where the
``d=1`` alias normalization is what makes it close. The cross-object
laws are pinned in ``tests/test_incidence.py``.

Tier note: ``Vertex`` is the IDENTITY tier (the tier rule lives in
the ``mu3.edge`` module docstring).
"""

from dataclasses import dataclass
from typing import Sequence

from .cell import _project_corner, cell_resolution, is_pentagon, is_valid_cell
from .edge import (
    DirectedEdge,
    UndirectedEdge,
    _require_outgoing,
    edge_corner_digits,
)
from .eisenstein import UNIT_DIGITS, scaled_corner
from .face_lattice import rotate_digit_ccw
from .neighbor import step
from .projection import Vec3


def vertex_directions(cell: Sequence[int]) -> tuple[int, ...]:
    """The digit names of ``cell``'s distinct corners, in
    ``cell_boundary`` row order; pentagons drop the ``d=1`` alias of
    the ``d=6`` cut corner. The single home of that rule."""
    return (6, 2, 3, 4, 5) if is_pentagon(cell) else UNIT_DIGITS


def _normalize(cell: tuple, d: int) -> tuple[tuple, int]:
    """Canonical name of a corner within its own cell: the pentagon
    cut corner's post-stitch name ``d=1`` folds onto ``d=6``."""
    if d == 1 and is_pentagon(cell):
        return cell, 6
    return cell, d


def _orbit_step(cell: tuple, d: int) -> tuple[tuple, int]:
    """``rho(c, eps) = (c + eps, eps*omega)``: the next incident cell
    CCW around the corner, with its name for the same corner —
    ``omega`` transported by the walk's arrow."""
    dest, rot = step(cell, d)
    return _normalize(dest, rotate_digit_ccw(d, rot + 2))


def _orbit(rep: tuple[tuple, int]):
    r2 = _orbit_step(*rep)
    return (rep, r2, _orbit_step(*r2))


@dataclass(frozen=True, slots=True)
class Vertex:
    """A grid vertex (cell corner), canonically named.

    The constructor accepts ANY name of the corner — each of the three
    orbit representatives, plus the pentagon ``d=1`` alias — and
    canonicalizes (two walks), so equality/hash are plain field
    compares.
    """

    cell: tuple
    d: int

    def __post_init__(self):
        cell_t = tuple(int(x) for x in self.cell)
        d = int(self.d)
        if not is_valid_cell(cell_t):
            raise ValueError(f'Vertex: invalid cell {cell_t}')
        if not 1 <= d <= 6:
            raise ValueError(f'Vertex: no corner {d} of {cell_t}')
        c, d = min(_orbit(_normalize(cell_t, d)))
        object.__setattr__(self, 'cell', c)
        object.__setattr__(self, 'd', d)

    def representatives(self) -> tuple[tuple[tuple, int], ...]:
        """The three ``(cell, d)`` orbit representatives — canonical
        first, then CCW around the corner. Names are always normalized
        (never a pentagon ``d=1``), so each is also a valid outgoing
        edge direction — ``edges_in``/``undirected_edges`` rely on
        that."""
        return _orbit((self.cell, self.d))

    def incident_cells(self) -> tuple[tuple, ...]:
        """The 3 cells sharing this corner (exactly 3 everywhere —
        pentagon corners too), canonical cell first."""
        return tuple(c for c, _ in self.representatives())

    def edges_in(self) -> tuple[DirectedEdge, ...]:
        """The 3 directed edges whose head is this vertex — the orbit
        representatives read as directed edges, one per incident cell,
        CCW around the corner. Rep i's edge connects incident cell i
        to incident cell i+1."""
        return tuple(DirectedEdge(c, d) for c, d in self.representatives())

    def edges_out(self) -> tuple[DirectedEdge, ...]:
        """The 3 directed edges whose tail is this vertex — the next
        outgoing edge CCW after each entering one. Equals
        ``{e.reverse() for e in edges_in()}`` as a set (the twin
        bijection, pinned in test_incidence)."""
        return tuple(e.rotate_ccw() for e in self.edges_in())

    def undirected_edges(self) -> tuple[UndirectedEdge, ...]:
        """The 3 boundary segments meeting at this corner — exactly
        the pairwise adjacencies of the 3 incident cells."""
        return tuple(UndirectedEdge(c, d) for c, d in self.representatives())

    def adjacent_vertices(self) -> tuple['Vertex', ...]:
        """The 3 corners one boundary segment away — the far endpoint
        (tail) of each entering edge, matching ``edges_in()`` order.
        Each is the tail digit of ``edge_corner_digits``; the head
        would just be this vertex again."""
        return tuple(
            Vertex(c, edge_corner_digits(d)[0])
            for c, d in self.representatives()
        )


def edge_to_vertices(cell: Sequence[int], d: int) -> tuple[Vertex, Vertex]:
    """Wire-pair form of :func:`edge_vertices`: the two corners of
    edge ``(cell, d)`` as canonical vertices, no ``DirectedEdge`` lift
    required.

    IDENTITY tier — each ``Vertex`` canonicalizes (two arrow walks);
    positions-only callers want ``mu3.edge.edge_to_boundary``.
    """
    cell_t = tuple(int(x) for x in cell)
    d = int(d)
    _require_outgoing(cell_t, d, 'edge_to_vertices')
    tail, head = edge_corner_digits(d)
    return (Vertex(cell_t, tail), Vertex(cell_t, head))


def edge_vertices(edge: DirectedEdge) -> tuple[Vertex, Vertex]:
    """The two corners of an edge's boundary segment, as canonical
    vertices: ``(tail, head)`` in CCW-traversal order around
    ``edge.cell`` — equivalently (interior on the left): facing the
    destination cell from the origin cell's center, ``tail`` is on
    the right and ``head`` on the left. ``reverse`` swaps the pair —
    the twin law; the ordering laws are pinned in
    ``tests/test_incidence.py``. At a pentagon's ``d=2`` edge the
    tail name is the ``d=1`` alias, absorbed by the ``Vertex``
    constructor — no seam special case here. (For an
    ``UndirectedEdge``, which erases orientation, take
    ``edge_vertices(u.directed_orientations()[0])`` — the pair is
    the same set either way.)"""
    return edge_to_vertices(edge.cell, edge.d)


def vertex_to_vec3(cell: Sequence[int], d: int) -> Vec3:
    """Unit 3-vector of the corner named ``(cell, d)``: exact flat
    corner (``eisenstein.scaled_corner``), floats only inside
    ``cell._project_corner`` (whose stitch sends every name of a
    corner to the same 3D point). Same path as ``cell_boundary``
    rows. Validates like :class:`Vertex` and accepts any corner name,
    including the pentagon ``d=1`` alias."""
    cell_t = tuple(int(x) for x in cell)
    d = int(d)
    if not is_valid_cell(cell_t):
        raise ValueError(f'vertex_to_vec3: invalid cell {cell_t}')
    if not 1 <= d <= 6:
        raise ValueError(f'vertex_to_vec3: no corner {d} of {cell_t}')
    return _project_corner(
        scaled_corner(cell_t[1:], d), cell_t[0], cell_resolution(cell_t)
    )


def vertices_of_cell(cell: Sequence[int]) -> list[Vertex]:
    """The 5 (pentagon) or 6 (hex) distinct vertices of ``cell``, in
    ``cell_boundary`` row order: ``[vertex_to_vec3(cell, d) for d in
    vertex_directions(cell)]`` matches ``cell_boundary(cell,
    closed=False)`` row for row (to projection float noise)."""
    cell_t = tuple(int(x) for x in cell)
    return [Vertex(cell_t, d) for d in vertex_directions(cell_t)]
