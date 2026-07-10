"""``cells_to_multipolygon``: the boundary of a cell set, as rings.

Everything is combinatorial until the final corner projections:

- One sweep classifies every (cell, direction) pair of the set: the
  walk destination either leaves the set (a boundary edge, kept as
  the wire pair ``(cell, d)``) or grows the connected component.
- Ring chaining is walk-based, no vertex canonicalization: every grid
  corner is 3-valent and its three incident cells are MUTUALLY
  adjacent, so a boundary can pass through a corner at most once —
  the next boundary edge out of a corner is unique.
- Rings are traversed with the set's interior on the LEFT (outer
  rings CCW viewed from outside the sphere, holes CW), and
  component-first decomposition assigns each hole to its outer by
  construction. The outer is the ring enclosing the SMALLEST area
  under the right-hand rule (``_enclosed_area``) — for sub-hemisphere
  components that is exactly the sign of the excess; for global
  components (sphere minus several disks) it makes a valid,
  necessarily arbitrary choice instead of failing.
- Each ring corner is projected once (the edge HEADS — N projections
  for an N-edge ring), through a per-cell ``_CellFrame`` reused
  across consecutive same-cell edges.

Scope: all cells must share one resolution — render a compacted set
by uncompacting first (mixed-resolution edges don't chain). A set
covering the whole sphere has no boundary and returns ``[]``.
"""

import math

import numpy as np

from .cell import (
    _CellFrame,
    _signed_spherical_excess,
    cell_resolution,
    is_valid_cell,
)
from .edge import corner_leaving_edge, outgoing_directions
from .neighbor import _step_fast
from .vertex import orbit_step


def _component_boundaries(cell_set: set, res: int) -> list[set]:
    """The boundary-edge set of each connected component of
    ``cell_set``: one sweep over all (cell, direction) pairs, each
    classified — destination outside the set is a boundary edge,
    destination still unvisited grows the component. Components are
    maximal, so each boundary set lands with its component."""
    todo = set(cell_set)
    out = []
    while todo:
        frontier = [todo.pop()]
        boundary = set()
        while frontier:
            c = frontier.pop()
            for d in outgoing_directions(c):
                nb, _ = _step_fast(c, res, d)
                if nb not in cell_set:
                    boundary.add((c, d))
                elif nb in todo:
                    todo.remove(nb)
                    frontier.append(nb)
        out.append(boundary)
    return out


def _next_boundary_edge(edge: tuple, boundary: set) -> tuple:
    """The unique boundary edge out of ``edge``'s head corner.

    The head corner of ``(c, d)`` is corner ``d`` of ``c``; its Z/3
    orbit (``vertex.orbit_step``) names it once per incident cell, and
    ``edge.corner_leaving_edge`` reads each name's leaving edge
    (alias-aware: the stitch makes it total, pentagon cut corners
    included). Only two of the three names can answer: the first
    orbit step lands on the EXTERIOR cell across ``edge`` — the cell
    whose absence made ``edge`` a boundary edge — so its leaving edge
    is never in ``boundary``. The continuation is on ``c`` itself
    (no walk) or on the third cell (two orbit walks; a derived
    inverse orbit step could make it one — port-relevant). Exactly
    one of the two candidates is in ``boundary`` (a corner's three
    cells are mutually adjacent: no pinch points).
    """
    same_cell = corner_leaving_edge(*edge)
    if same_cell in boundary:
        return same_cell
    third = corner_leaving_edge(*orbit_step(*orbit_step(*edge)))
    if third in boundary:
        return third
    raise AssertionError(f'no boundary continuation at head of {edge}')


def _trace_ring(start: tuple, boundary: set) -> np.ndarray:
    """Walk one ring from ``start``; rows are the edge head corners in
    traversal order. An edge leaves ``boundary`` when chosen as the
    continuation; ``start``, chosen last, closes the ring."""
    pts = []
    frame = _CellFrame(start[0])
    edge = start
    while True:
        c, d = edge
        if frame.cell != c:
            frame = _CellFrame(c)
        pts.append(frame.corner_vec3(d))
        edge = _next_boundary_edge(edge, boundary)
        boundary.remove(edge)
        if edge == start:
            return np.stack(pts, axis=0)


def cells_to_multipolygon(cells) -> list[list[np.ndarray]]:
    """Boundary of a set of same-resolution cells: a list of polygons,
    one per connected component, each ``[outer, hole, ...]`` — rings
    as (N, 3) arrays of unit 3-vectors, outer CCW, holes CW, not
    closed (first row is not repeated).
    """
    cell_set = {tuple(int(x) for x in c) for c in cells}
    if not cell_set:
        return []
    resolutions = {cell_resolution(c) for c in cell_set}
    if len(resolutions) != 1:
        raise ValueError(
            f'cells_to_multipolygon: mixed resolutions {sorted(resolutions)};'
            f' uncompact to one resolution first'
        )
    for c in cell_set:
        if not is_valid_cell(c):
            raise ValueError(f'cells_to_multipolygon: invalid cell {c}')
    res = resolutions.pop()

    polygons = []
    for boundary in _component_boundaries(cell_set, res):
        rings = []
        while boundary:
            rings.append(_trace_ring(next(iter(boundary)), boundary))
        if not rings:
            continue   # component covers the sphere: no boundary
        rings.sort(key=_enclosed_area)
        polygons.append(rings)
    return polygons


def _enclosed_area(ring: np.ndarray) -> float:
    """Area enclosed by the ring under the right-hand rule (interior
    on the left), in [0, 4π): the signed excess, normalized. The
    outer-loop selection key: the outer is the ring enclosing the
    SMALLEST area — identical to the sign of the excess for
    sub-hemisphere components, and a valid (necessarily arbitrary)
    choice for components without a natural outer, e.g. the sphere
    minus two separated disks, where every ring's excess is negative.
    """
    a = _signed_spherical_excess(ring)
    return a if a >= 0.0 else a + 4.0 * math.pi
