"""The SINGLE-EDGE contract and its projection hypothesis, discharged.

``vec3_to_cell_raw``'s contract is: the raw cell is the containing
cell or the neighbor across exactly ONE violated edge — so ``_polish``
corrects by that one edge and is done. "One hop" is the consequence;
SINGLE EDGE is the invariant, and it is an invariant of the
architecture, not a radius to tune: cell corners agree exactly between
the flat and spherical descriptions (they are projections of the same
exact lattice points), so the two descriptions can disagree only
inside per-edge bow lenses pinned at shared corners, each straddled by
the raw cell and that one edge-neighbor. The *sagitta* — how far the
pulled-back geodesic edge bows from the straight flat edge between the
same corners — measures the lens; the invariant holds while the worst
sagitta stays under half the lattice spacing, and every admissible
projection MUST keep it there. If a projection erodes the margin, the
projection (or its fitted constants) is what gets fixed or rejected —
"check 2-3 hops instead" is never the answer: a bow escaping the
edge-neighbor would mean flat adjacency no longer models spherical
adjacency, voiding the snap witness, the edge→ring-1 correction, and
the banded gate all at once.

This test discharges the hypothesis for the active projection by
measuring the worst sagitta directly, and exercises the contract
itself on the adversarial inputs (cell corners and geodesic edge
midpoints — points ON the boundaries). The margin shrinks second-order
in edge length, so coarse resolutions bind; res 1 is the global worst.
Shared corners/edges are deduped — by exact canonical Vertex identity
(``mu3.vertex``, aligned with boundary rows) — so each unique point is
inverse-projected and contract-checked once (the inverse projection is
the costly primitive here).

Measured for AlphaSlerp (2026-07-02): max sagitta 0.250 at res 1,
0.144 at res 2, ~0.12 at res 3, shrinking thereafter. The assert
threshold 0.35 is an early-warning line between the measured 0.25 and
the theorem's 0.5: a projection swap that trips it but stays under 0.5
technically still satisfies the invariant — treat that as a projection
on its last margin, to be fixed or rejected, never as license to widen
the search.

The banded polish (``mu3.index._polish_banded``) sharpens the same
hypothesis into a per-res, per-edge-position bow envelope with its own
projection-coupled constants (``mu3.index._BOW_COEFFS``); the
``test_bow_*`` tests at the bottom pin those for the active projection.
"""

import random
from functools import lru_cache

import numpy as np
import pytest

from mu3 import cell_ring1, cells_at_res, icosahedron
from mu3.cell import (
    _sphere_to_flat,
    active_projection_name,
    cell_boundary,
)
from mu3.eisenstein import first_nonzero_digit
from mu3.face_lattice import get_rot
from mu3.index import _BOW_COEFFS, _cell_near_stitch, vec3_to_cell_raw
from mu3.neighbor import resolve_position
from mu3.traversal import disk_k
from mu3.vertex import vertices_of_cell

_SAGITTA_MAX = 0.35

# Early-warning margin for the banded-polish bow envelope: the shipped
# _BOW_COEFFS operating values are fitted at 2x the measured envelope
# (scripts/measure_polish_band.py), so a fresh sampled measurement must
# stay under table/1.4 — trip this and the fit needs redoing (projection
# or parameter drift), well before correctness (measured > table) is at
# risk.
_BOW_MARGIN = 1.4


def _pentagon_adjacent(res):
    out = []
    for b in range(12):
        pent = (b,) + (0,) * res
        out.append(pent)
        out.extend(cell_ring1(pent))
    return out


@lru_cache(maxsize=None)
def _bnd(cell):
    return cell_boundary(cell, closed=False)


def _contains(p3d, cell, tol=1e-9):
    """Tolerant spherical containment: the test inputs are exact
    corners and edge midpoints, which sit ON boundaries shared by 2-3
    cells; containment there is a coin flip at machine epsilon, and
    every incident cell is a correct answer. The tolerance admits
    exactly that on-boundary set."""
    V = _bnd(cell)
    n = len(V)
    return all(
        np.cross(V[k], V[(k + 1) % n]) @ p3d > -tol for k in range(n)
    )


def _check_contract(p3d, base, w, res):
    """Raw resolve of a boundary point must be a containing cell or
    one edge crossing from one (checked as ring-1 membership; reuses
    the caller's pulled-back position)."""
    raw = resolve_position(base, w, res)
    if _contains(p3d, raw):
        return
    assert any(_contains(p3d, nb) for nb in cell_ring1(raw)), (p3d, raw)


_SCOPES = {
    'res1-all': (1, lambda: cells_at_res(1)),
    'res2-all': (2, lambda: cells_at_res(2)),
    'res3-pent-adj': (3, lambda: _pentagon_adjacent(3)),
}


@pytest.mark.parametrize('scope', sorted(_SCOPES))
def test_one_hop_margin_and_contract(scope):
    res, cells = _SCOPES[scope]
    rot_N = get_rot(res)
    worst = 0.0
    seen_edges = set()
    checked_corners = set()
    pullback = {}   # (base, vertex key) -> scaled flat position

    def flat(v, vkey, base):
        key = (base, vkey)
        if key not in pullback:
            pullback[key] = _sphere_to_flat(v, base) * rot_N
        return pullback[key]

    for cell in cells():
        base = cell[0]
        V = _bnd(cell)
        m = len(V)
        keys = vertices_of_cell(cell)
        for k in range(m):
            v1, v2 = V[k], V[(k + 1) % m]
            k1, k2 = keys[k], keys[(k + 1) % m]
            ekey = frozenset((k1, k2))
            if ekey in seen_edges:
                continue
            seen_edges.add(ekey)

            gm = v1 + v2
            gm = gm / np.linalg.norm(gm)
            z1 = flat(v1, k1, base)
            z2 = flat(v2, k2, base)
            zm = _sphere_to_flat(gm, base) * rot_N

            # Sagitta: deviation of the pulled-back geodesic midpoint
            # from the straight flat segment, lattice units.
            d = z2 - z1
            t = ((zm - z1) * d.conjugate()).real / (d * d.conjugate()).real
            t = min(1.0, max(0.0, t))
            worst = max(worst, abs(zm - (z1 + t * d)))

            # The contract itself, on the boundary points: each unique
            # corner once, each unique edge midpoint once.
            if k1 not in checked_corners:
                checked_corners.add(k1)
                _check_contract(v1, base, z1, res)
            _check_contract(gm, base, zm, res)

    assert worst < _SAGITTA_MAX, \
        f'{scope}: worst sagitta {worst:.4f} — single-edge margin ' \
        f'eroding; fix the projection, never the hop count ' \
        f'(theorem fails at 0.5; see module docstring)'


# --- banded-polish bow envelope --------------------------------------
#
# The banded polish (mu3.index._polish_banded) side-tests only edges
# whose flat chord distance is under the allowance 4·c_res·t(1−t).
# Correctness needs the true bow of every non-stitch-guarded edge to
# stay under that allowance; these tests pin the shipped c_res table
# for the ACTIVE projection against a fresh sampled measurement, and
# pin the tail-stability assumption behind the beyond-table rule.


def _measured_envelope(res, cells):
    """Worst bow(t) / (4·t·(1−t)) over the cells' non-guarded edges,
    lattice units, t sampled off-center and at the peak."""
    rot = get_rot(res)
    worst = 0.0
    for cell in cells:
        if _cell_near_stitch(cell):
            continue
        base = cell[0]
        B = _bnd(cell)
        m = len(B)
        flats = [_sphere_to_flat(v, base) * rot for v in B]
        for k in range(m):
            v1, v2 = B[k], B[(k + 1) % m]
            z1, z2 = flats[k], flats[(k + 1) % m]
            chord = z2 - z1
            L2 = (chord * chord.conjugate()).real
            if L2 < 1e-18:
                continue
            for t in (1 / 6, 1 / 2, 5 / 6):
                gm = (1 - t) * v1 + t * v2
                gm = gm / np.linalg.norm(gm)
                zm = _sphere_to_flat(gm, base) * rot
                th = ((zm - z1) * chord.conjugate()).real / L2
                th = min(0.98, max(0.02, th))
                bow = abs(zm - (z1 + th * chord))
                worst = max(worst, bow / (4.0 * th * (1.0 - th)))
    return worst


def _envelope_sample(res):
    """Pentagon 2-disks (spoke straddlers, the worst-bow population),
    cells along every icosa edge (chart seams), and a random sample."""
    rng = random.Random(0)
    V = icosahedron.vertices()
    nbrs = icosahedron.vertex_neighbors()
    cells = set()
    for b in range(12):
        cells |= set(disk_k((b,) + (0,) * res, 2))
    for i in range(12):
        for j in nbrs[i]:
            if j < i:
                continue
            q = V[i] + V[j]
            c = vec3_to_cell_raw(q / np.linalg.norm(q), res)
            cells.add(c)
            cells.update(cell_ring1(c))
    n_random = 20
    while n_random:
        digits = [rng.randrange(7) for _ in range(res)]
        if first_nonzero_digit(digits) != 1:
            cells.add((rng.randrange(12), *digits))
            n_random -= 1
    return cells


@pytest.mark.parametrize('res', [2, 9])
def test_bow_envelope_headroom(res):
    table = _BOW_COEFFS[active_projection_name()]
    cells = cells_at_res(res) if res <= 2 else _envelope_sample(res)
    measured = _measured_envelope(res, cells)
    assert measured * _BOW_MARGIN < table[res], (
        f'res {res}: measured bow envelope {measured:.4f} within '
        f'{_BOW_MARGIN}x of operating coefficient {table[res]:.4f} — '
        f'refit with scripts/measure_polish_band.py'
    )


def test_bow_table_tail_stable():
    """Beyond the table, _bow_coeff continues with the max of the two
    top entries. That is sound because the envelope stabilizes with res
    (worst edges cross chart seams, where relative bow is scale-free)
    — pin that the tail is flat-or-falling per parity."""
    table = _BOW_COEFFS[active_projection_name()]
    top = max(table)
    assert top >= 8, 'table too short to judge tail stability'
    for r in range(top - 3, top + 1):
        assert table[r] <= 1.05 * max(table[r - 1], table[r - 2]), (
            f'coefficient still growing at res {r}; beyond-table '
            f'continuation in _bow_coeff is unsafe'
        )
