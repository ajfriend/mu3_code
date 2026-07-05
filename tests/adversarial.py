"""The adversarial zoo: named families of cells and pairs that broke
real designs on this branch.

Every entry records WHAT DESIGN THE FAMILY KILLED, plus a generator of members. The week's working
method — and the reason these are centralized — is that a generic
sweep plus a NAMED adversarial family is what makes failures
diagnosable: new traversal/indexing work (vertices, ring ordering, a
future seam-region disk tier, the Zig port) should run against every
family here before trusting its own sweeps.

Surfaces already consuming each family are noted so a new session can
see both how to aim a family at new code and where the precedent
tests live.
"""

from mu3 import cells_at_res, dodec

# ---------------------------------------------------------------------
# PHANTOM_CORNER — cells whose deleted-direction walks hit 3-hex
# phantom corners (one ghost position + two real cells).
# Killed: the original NEIGHBOR_TRANS algebraic walk (fixed carry
# constant); resolved by the carried arrow.
# Consumed by: test_neighbor.test_phantom_corner_family (ring-1 band +
# CCW laws); subsumed exhaustively by the edge round-trip laws.
# Also the prospective adversarial population for VERTICES (cut-end
# corners — where a pentagon's seam terminates).


def phantom_corner_cells():
    return [(b, 2, 6, 6) for b in range(12)]


# ---------------------------------------------------------------------
# SEAM_CROSSING_CW — same-base ring-1 adjacencies whose walk crosses
# the cut clockwise; the cells are 3D-adjacent but flat-far (the
# CLAUDE.md stitch gotcha).
# Killed: the "endpoint-side" witness rule (decide the stitch from
# where the walk landed) — 36-cell counterexample family.
# Logs: neighbor.py docstring.
# Consumed by: test_neighbor.test_seam_crossing_cw.


def seam_crossing_pairs():
    return [((b, 2), (b, 6)) for b in range(12)]


# ---------------------------------------------------------------------
# POST_HOP_WEDGE — phantoms reached after a cross-pentagon hop.
# Killed: the "untransported-source" witness rule (decide the stitch
# from the source in its original frame) — 192-cell family; proof the
# arrow must be carried through hops.
# Logs: neighbor.py docstring.
# Consumed by: test_neighbor.test_post_hop_wedge_ccw,
# test_edge.test_seam_reverse_regression.


def post_hop_pairs():
    return [((b, 6), (int(dodec.neighbors[b][0]), 2)) for b in range(12)]


# ---------------------------------------------------------------------
# ON_RAY_CENTERS — cells whose centers sit EXACTLY on the primary ray
# (scaled position exactly real; renders exactly onto the cut).
# Killed: point-location tie handling (the witness sign test at
# sp == 0); forced the closed-cut-belongs-to-CW convention with the
# 1e-9 boundary nudge.
# Logs: neighbor.resolve_position docstring.
# Consumed by: test_point_location.test_cut_line_containment and
# test_cell_centers_round_trip (which sweeps these among all cells).


def on_ray_centers():
    return [(b, 6, 1) for b in range(12)]


# ---------------------------------------------------------------------
# INTERLOPER_DISK — (cell, k) disk queries whose shortest routes cross
# a cut FAR from any cone and land on Gosper-INTERLOPER positions
# (canonical digit strings inside the deleted wedge, invisible to the
# seam-side witness).
# Killed: tier-1's original "no cone within k" guard (264 res-3
# mismatches); motivated the cut-RAY clearance guard. Also the reason
# all three seam-region disk fast tiers were refuted.
# Logs: traversal.py module docstring.
# Consumed by: test_traversal.test_zoo_interloper_disks (via the
# PUBLIC dispatch, so the pin holds however the guard evolves).
# Members are the historical base-0 failures at their failing k,
# replicated across all bases by icosahedral symmetry.


def interloper_disk_cases():
    return [(cell, k)
            for b in range(12)
            for cell, k in [((b, 2, 1, 5), 8),
                            ((b, 2, 1, 6), 8),
                            ((b, 2, 5, 1), 5)]]


# ---------------------------------------------------------------------
# TWO_SEAM_PAIR — pairs whose shortest route crosses the near seam and
# then reaches a SECOND seam that is flat-far from the source.
# Killed: the pair fast path's "one near ray" guard measured from the
# source only; motivated the single-seam-SYSTEM lesson (for disks) and
# the BFS routing of multi-seam pairs.
# Consumed by: test_traversal.test_zoo_pairs. Historical instances
# (base-0 concrete; the crossing structure depends on the specific
# neighbor layout, so these are kept verbatim rather than
# symmetrized).

TWO_SEAM_PAIRS = [
    ((0, 2, 1, 1), (2, 2, 6, 1)),
    ((0, 2, 1, 2), (2, 2, 6, 1)),
    ((0, 2, 1, 1), (2, 2, 6, 6)),
]

# ---------------------------------------------------------------------
# LENS_LONG_PAIR — long pairs whose shortcut seam sits far from the
# straight corridor but inside the lens of near-optimal routes (the
# lens fattens with distance).
# Killed: the pair guard's segment-proximity involvement test (272
# res-3 errors); motivated the min-detour lens test.
# Consumed by: test_traversal.test_zoo_pairs.

LENS_PAIRS = [
    ((0, 2, 1, 3), (0, 4, 4, 4)),
    ((0, 2, 1, 3), (8, 0, 6, 0)),
    ((0, 2, 1, 3), (8, 6, 0, 3)),
]

# ---------------------------------------------------------------------
# CUT_END_CORNER — (cell, d) vertex names at the far end of a
# pentagon's deleted-wedge seam: the post-stitch image of the wedge's
# territory corner (the deleted icosa-face center). The three incident
# cells live in THREE different base charts (e.g. bases 0, 2, 5 for
# the base-0 member), so every orbit step is a cross-pentagon hop —
# the corner where cut, hop, and territory boundary all meet.
# Killed: nothing yet — added with stage-3 vertices as the predicted
# adversarial population.
# Consumed by: test_vertex.test_zoo_cut_end_corners; the member cells
# also feed test_vertex's general law sweeps.
# Members found empirically (2026-07-03): the base-b name at each res;
# the digit tails differ per res because of the Class II/III rotation
# alternation.


def cut_end_corner_names(res=3):
    tail = {1: (2,), 2: (2, 1), 3: (2, 1, 2)}[res]
    return [((b, *tail), 1) for b in range(12)]


# ---------------------------------------------------------------------
# PENTAGONS — the 12 pentagon cells themselves: 5 neighbors, 5k rings,
# the deleted direction, 5-fold everything.
# Killed: nothing directly (by now), but every surface must handle
# them and their counts differ from hexes everywhere.
# Consumed by: test_traversal.test_pentagon_ring_counts,
# test_pair_path_laws_near_pentagons (pentagon-ADJACENT sources),
# test_neighbor ring laws, edge counts.


def pentagon_cells(res=3):
    return [(b,) + (0,) * res for b in range(12)]


# ---------------------------------------------------------------------
# LAW_SWEEP — the standard population for per-object law sweeps:
# every cell at res 0-2, plus the res-3 zoo families above (phantom
# corners, pentagons, cut-end corner owners). Small enough to run
# elementwise laws over every edge/vertex, adversarial enough to hit
# every seam mechanism.
# Consumed by: test_vertex (orbit/position laws), test_incidence
# (cross-object laws).


def law_sweep_cells():
    for res in (0, 1, 2):
        yield from cells_at_res(res)
    yield from phantom_corner_cells()
    yield from pentagon_cells(res=3)
    yield from (c for c, _ in cut_end_corner_names(res=3))
