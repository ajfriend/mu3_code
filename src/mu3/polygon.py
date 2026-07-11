"""``cells_to_multipolygon``: the boundary of a cell set, as rings.

Everything is combinatorial until the final corner projections:

- One sweep classifies every (cell, direction) pair of the set: a
  walk destination outside the set is a boundary edge (kept as the
  wire pair ``(cell, d)``); one still unvisited grows the component.
- Ring chaining is walk-based, no vertex canonicalization: every grid
  corner is 3-valent and its three incident cells are MUTUALLY
  adjacent, so a boundary can pass through a corner at most once —
  the next boundary edge out of a corner is unique.
- Rings are traversed with the set's interior on the LEFT (outer
  rings CCW viewed from outside the sphere, holes CW), and
  component-first decomposition assigns each hole to its outer by
  construction. The outer is the ring enclosing the SMALLEST area
  under the right-hand rule — for sub-hemisphere
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

import numpy as np

from .cell import (
    _CellFrame,
    _spherical_polygon_area,
    cell_resolution,
    is_valid_cell,
)
from .edge import corner_leaving_edge, edge_reverse, outgoing_directions
from .neighbor import _step_fast


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
    """The unique boundary edge out of ``edge``'s head corner — the
    textbook DCEL boundary walk, evaluated on demand: rotate within
    the cell (``corner_leaving_edge``, pure arithmetic, alias-aware at
    pentagon cut corners); if that edge is interior, hop its twin
    (``edge_reverse``, one walk) and rotate again on the cell across
    it. A 3-valent corner terminates the walk in at most one hop: the
    only other incident cell is the EXTERIOR one across ``edge``,
    which can't own a boundary edge — so the twin-hop result IS the
    continuation, no test needed (``_trace_ring``'s removal raises
    loudly if that ever broke). This computes what loop-surgery
    designs cache in linked next-pointers. The walk's turning obeys a
    discrete Gauss–Bonnet law, pinned in
    ``test_polygon.test_turning_law``.
    """
    same_cell = corner_leaving_edge(*edge)
    if same_cell in boundary:
        return same_cell
    # The twin's head is this same corner (the twin law), named by the
    # twin's own direction digit.
    nb, dr = edge_reverse(*same_cell)
    return corner_leaving_edge(nb, dr)


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
        # Outer first: smallest right-hand-rule enclosed area (see the
        # module docstring; sphere-minus-two-disks is why not sign).
        rings.sort(key=_spherical_polygon_area)
        polygons.append(rings)
    return polygons
