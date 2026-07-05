"""Gosper island boundary: the directed edges around a cell's
descendant set.

``island_edges(cell, res)`` yields the directed edges — ``(cell, d)``
wire-format pairs, liftable via ``DirectedEdge(*pair)`` — whose origin
is a res-``res`` descendant of ``cell`` and whose destination is not.
They form the closed CCW boundary loop of the descendant region (the
Gosper island): ``6 * 3**delta`` edges for a hex parent, ``5 *
3**delta`` for a pentagon. The starting edge is deliberately
unspecified — any cyclic rotation of the loop is a valid output.

Port of the H3 iterator design (uber/h3#1138): iterator state is a
per-level cyclic walk position and nothing else — advancing is pure
sequence manipulation, no neighbor walks, no geometry, no membership
tests. mu3's digit conventions collapse most of the H3 version's
tables and state:

- Walk position ``p`` (period 18: six sides x three fractal segments)
  encodes BOTH the child digit and the outgoing edge direction in
  closed form: ``digit = p // 3 + 1`` (digits are sequential CCW) and
  ``dir = rotate_digit_ccw(digit, p % 3 - 1)`` — the boundary child on
  side ``c`` emits its three edges in directions ``(c-1, c, c+1)``.
  H3's separate ``walk_digit``/``edge_dir`` tables and the extra
  edge-position state (with its own shift constant) do not exist here.
- Level interaction, identical 18-cycle arithmetic to H3: when a
  level's position crosses a side transition, the coarser level
  advances first, and if its CELL changed the finer position shifts
  back 6 (one full side). The transition point alternates with the
  Class II/III resolution parity: position ``% 3 == 0`` at odd levels,
  ``== 1`` at even levels.
- Every level starts at the SAME position, a middle-of-side segment:
  side midpoints are fixed points of the Gosper subdivision, so the
  all-middle-segments configuration is a valid boundary state at
  every depth (any ``s % 3 == 1`` works — the sweep found exactly
  those six; H3's per-parity 0/14/16 starts were a different valid
  phase). The phase freedom is the unspecified-start contract.

Constant provenance — deliberately NOT the carry-table discipline:
``_TRIG``/``_START`` are literals, originally fitted against a
DCEL pivot-walk oracle (see git history), and pinned ONLY by the
tests in ``tests/test_island.py``: exhaustive SET equality against
brute-force boundary enumeration (completeness — even against a
hypothetical second boundary component no walk would reach) plus the
chaining law (order — boundary vertices have exactly two incident
segments, so a closed distinct chained cycle is fully rigid). A wrong
edit fails the suite, not the import; deriving at import was rejected
on purpose (it would pull walk machinery into the module that exists
to avoid walks).

Pentagon parents ride the hexagon walk and skip the deleted
first-level digit-1 side, H3-style; deeper digits are never 0 on the
boundary, so no other invalid forms can arise.

Future work (module-local ideas, deliberately not a planning doc):

- **Derive ``_TRIG`` from the carry tables.** The side-transition
  point is a statement about when child-digit steps carry into the
  parent — exactly what ``eisenstein.CARRY`` encodes, as import-cheap
  pure arithmetic (no walk machinery, so the layering objection above
  doesn't apply). If the derivation works out, the fitted literals
  reach the full derived-and-import-asserted discipline. Do this one
  first; it likely produces the algebra the next item needs.
- **Random access by boundary index.** The loop is naturally indexed
  by ``k in [0, sides * 3**delta)``: side = ``k // 3**delta``,
  position within the side = the base-3 digits of the remainder. If
  the shift-back-6 accumulation has a closed form in those digits (it
  looks like a prefix sum of side-transition counts), an
  ``island_edge_at(cell, res, k)`` in O(delta) follows — binary
  search along a boundary, parallel chunking of the loop in a
  compiled port, and iteration with a single-integer state.
- **Pentagon skip as a phase choice** (minor): the skipped digit-1
  edges form one contiguous block per loop, so a pentagon-aware
  starting phase could drop the per-yield ``is_pent`` check and the
  ``yielded`` counter. The current guard short-circuits to ~nothing
  for hexes; only worth it if the loop shape ever bothers a port.

(The obvious-looking simplification of dropping the ``digits`` mirror
was tried and measured 70% slower — see the comment at its
definition.)
"""

from typing import Iterator, Sequence

from .cell import cell_resolution, is_pentagon, is_valid_cell
from .edge import outgoing_directions
from .face_lattice import rotate_digit_ccw

# Side-transition point of the 18-position walk cycle, by level parity
# (index r % 2): advance the coarser level when pos % 3 equals this.
_TRIG = (1, 0)

# The shared starting position for every level: the middle segment of
# side 2 (any s % 3 == 1 is valid — see module docstring; side 2 keeps
# pentagon starts off the deleted side).
_START = 4

_CYCLE = 18

# Per-position closed forms, tabulated at load: side digit and edge
# direction at walk position p — digit = p // 3 + 1, and the side-c
# child emits its three edges in directions (c-1, c, c+1).
_SIDE_DIGIT = tuple(p // 3 + 1 for p in range(_CYCLE))
_EDGE_DIR = tuple(
    rotate_digit_ccw(p // 3 + 1, p % 3 - 1) for p in range(_CYCLE)
)


def island_edges(
    cell: Sequence[int], res: int
) -> Iterator[tuple[tuple, int]]:
    """Yield the boundary edges of ``cell``'s res-``res`` descendant
    region as ``(cell, d)`` pairs, one full CCW loop (starting edge
    unspecified). ``res == cell_resolution(cell)`` yields the cell's
    own edges. O(1) amortized sequence work per edge; no walks."""
    cell_t = tuple(int(x) for x in cell)
    if not is_valid_cell(cell_t):
        raise ValueError(f'island_edges: invalid cell {cell_t}')
    parent_res = cell_resolution(cell_t)
    if res < parent_res:
        raise ValueError(
            f'island_edges: res {res} < cell resolution {parent_res}'
        )

    if res == parent_res:
        for d in outgoing_directions(cell_t):
            yield cell_t, d
        return

    delta = res - parent_res
    is_pent = is_pentagon(cell_t)
    first = parent_res + 1        # coarsest child level (index 0)
    n = len(outgoing_directions(cell_t)) * 3 ** delta
    fine = delta - 1

    # Walk state: one cyclic position per child level (index i =
    # level first + i). digits mirrors pos through _SIDE_DIGIT, kept
    # incrementally: tuple(digits) per yield is a C-level copy where
    # deriving per yield measurably is not.
    pos = [_START] * delta
    digits = [_SIDE_DIGIT[_START]] * delta

    def advance() -> None:
        # Find how far the advancing chain reaches: finest inward
        # while each level sits on its transition point.
        i = fine
        while i > 0 and pos[i] % 3 == _TRIG[(first + i) % 2]:
            i -= 1
        # Advance coarsest-first; a cell change at the coarser level
        # shifts the next finer position back one full side (6).
        changed = False
        for j in range(i, delta):
            if changed:
                pos[j] -= 6
            pos[j] = (pos[j] + _CYCLE + 1) % _CYCLE
            new = _SIDE_DIGIT[pos[j]]
            changed = new != digits[j]
            digits[j] = new

    yielded = 0
    while yielded < n:
        # digit 1 at the first child level is the pentagon-deleted
        # side (the rule's home: edge.outgoing_directions).
        if not (is_pent and digits[0] == 1):
            yield cell_t + tuple(digits), _EDGE_DIR[pos[fine]]
            yielded += 1
        advance()
