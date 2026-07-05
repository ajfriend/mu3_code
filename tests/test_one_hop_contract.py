"""The one-hop contract and its projection hypothesis, discharged.

``vec3_to_cell_raw``'s contract is: the raw cell is the containing cell
or one ring-1 hop from it, so ``_polish``'s single hop suffices. The
grid half of that is exact (nearest-lattice snap + exact resolve). The
geometric half is a THEOREM WITH A HYPOTHESIS about the active
projection: cell corners agree exactly between the flat and spherical
descriptions (they are projections of exact lattice points), so the
flat-vs-spherical mismatch is the *sagitta* — how far the pulled-back
geodesic edge bows away from the straight flat edge between the same
corners. One hop is guaranteed while the worst sagitta stays under
half the lattice spacing.

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
still satisfies one-hop — revisit the threshold consciously if that
ever happens.
"""

from functools import lru_cache

import numpy as np
import pytest

from mu3 import cell_ring1, cells_at_res
from mu3.cell import _sphere_to_flat, cell_boundary
from mu3.face_lattice import get_rot
from mu3.neighbor import resolve_position
from mu3.vertex import vertices_of_cell

_SAGITTA_MAX = 0.35


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
    one ring-1 hop from one (reuses the caller's pulled-back
    position)."""
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
        f'{scope}: worst sagitta {worst:.4f} — one-hop margin eroding ' \
        f'(theorem fails at 0.5; see module docstring)'
