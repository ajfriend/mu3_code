"""``cells_to_multipolygon``: the boundary of a cell set, as rings.

The multipolygon of a set of same-resolution cells: one polygon per
connected component, each a list of rings ``[outer, hole, ...]`` of
corner positions (unit 3-vectors, row per corner, not closed). Rings
are traversed with the set's interior on the LEFT, so outer rings are
CCW (viewed from outside the sphere) and holes are CW — the
orientation IS the classification (``cell._signed_spherical_excess``).

Everything is combinatorial until the final corner projections:

- Boundary edges are the wire pairs ``(cell, d)`` whose ``step``
  destination is outside the set.
- Ring chaining is walk-based, no vertex canonicalization: every grid
  corner is 3-valent and its three incident cells are MUTUALLY
  adjacent, so a boundary can pass through a corner at most once —
  the next boundary edge out of a corner is unique. The candidates
  are read off the corner's Z/3 orbit (``vertex._orbit_step``): the
  edge leaving corner ``(c, d)`` around cell ``c`` is
  ``(c, rotate_digit_ccw(d, 1))``.
- Each ring corner is projected once (the edge HEADS — N projections
  for an N-edge ring), through a per-cell ``_CellFrame`` reused
  across consecutive same-cell edges.

Scope: all cells must share one resolution — render a compacted set
by uncompacting first (mixed-resolution edges don't chain). A set
covering the whole sphere has no boundary and returns ``[]``.
Components spanning more than a hemisphere are untested territory
for the orientation classification.
"""

import numpy as np

from .cell import (
    _CellFrame,
    _signed_spherical_excess,
    is_pentagon,
    is_valid_cell,
)
from .edge import outgoing_directions
from .face_lattice import rotate_digit_ccw
from .neighbor import step
from .vertex import _orbit_step


def _connected_components(cells: set) -> list[set]:
    todo = set(cells)
    out = []
    while todo:
        seed = todo.pop()
        comp = {seed}
        frontier = [seed]
        while frontier:
            c = frontier.pop()
            for d in outgoing_directions(c):
                nb, _ = step(c, d)
                if nb in todo:
                    todo.remove(nb)
                    comp.add(nb)
                    frontier.append(nb)
        out.append(comp)
    return out


def _next_boundary_edge(edge: tuple, boundary: set) -> tuple:
    """The unique boundary edge out of ``edge``'s head corner.

    The head corner of ``(c, d)`` is corner ``d`` of ``c``; its Z/3
    orbit names it once per incident cell, and the edge leaving it
    CCW around incident cell ``ci`` is ``(ci, rotate_digit_ccw(di, 1))``.
    At a pentagon cut corner the canonical name ``(pent, 6)`` has the
    alias ``(pent, 1)`` on the SAME cell, whose leaving edge is
    ``(pent, 2)`` — the one extra candidate (the ring of a single
    pentagon closes through it). Exactly one candidate is in
    ``boundary`` (a corner's three cells are mutually adjacent: no
    pinch points).
    """
    rep = edge
    for _ in range(3):
        ci, di = rep
        candidate = (ci, rotate_digit_ccw(di, 1))
        if candidate in boundary:
            return candidate
        if di == 6 and is_pentagon(ci) and (ci, 2) in boundary:
            return (ci, 2)
        rep = _orbit_step(ci, di)
    raise AssertionError(f'no boundary continuation at head of {edge}')


def _trace_ring(start: tuple, boundary: set) -> np.ndarray:
    """Walk one ring from ``start``, removing its edges from
    ``boundary``; rows are the edge head corners in traversal order.

    ``start`` stays in ``boundary`` until the ring closes so the final
    ``_next_boundary_edge`` can find it; traversed edges are never
    continuation candidates (an edge leaves its tail corner, the
    search is over its head corner's leaving edges).
    """
    pts = []
    frame = None
    edge = start
    while True:
        c, d = edge
        if frame is None or frame.cell != c:
            frame = _CellFrame(c)
        pts.append(frame.corner_vec3(d))
        nxt = _next_boundary_edge(edge, boundary)
        if edge != start:
            boundary.remove(edge)
        if nxt == start:
            boundary.remove(start)
            return np.stack(pts, axis=0)
        edge = nxt


def cells_to_multipolygon(cells) -> list[list[np.ndarray]]:
    """Boundary of a set of same-resolution cells: a list of polygons,
    one per connected component, each ``[outer, hole, ...]`` — rings
    as (N, 3) arrays of unit 3-vectors, outer CCW, holes CW, not
    closed (first row is not repeated).
    """
    cell_set = {tuple(int(x) for x in c) for c in cells}
    if not cell_set:
        return []
    resolutions = {len(c) - 1 for c in cell_set}
    if len(resolutions) != 1:
        raise ValueError(
            f'cells_to_multipolygon: mixed resolutions {sorted(resolutions)};'
            f' uncompact to one resolution first'
        )
    for c in cell_set:
        if not is_valid_cell(c):
            raise ValueError(f'cells_to_multipolygon: invalid cell {c}')

    polygons = []
    for comp in _connected_components(cell_set):
        boundary = {
            (c, d)
            for c in comp
            for d in outgoing_directions(c)
            if step(c, d)[0] not in cell_set
        }
        rings = []
        while boundary:
            rings.append(_trace_ring(next(iter(boundary)), boundary))
        if not rings:
            continue   # component covers the sphere: no boundary
        outers = [r for r in rings if _signed_spherical_excess(r) > 0.0]
        holes = [r for r in rings if _signed_spherical_excess(r) <= 0.0]
        if len(outers) != 1:
            raise AssertionError(
                f'component produced {len(outers)} outer rings'
            )
        polygons.append([outers[0], *holes])
    return polygons
