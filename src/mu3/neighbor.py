"""Ring-1 cell neighbors via integer Eisenstein digit arithmetic.

.. warning::

   **Tentative — paused mid-implementation.** The walk is correct on
   the bulk of each face but mis-routes ring-1 neighbours for hex
   cells adjacent to a pentagon (the deleted-wedge / pentagon-distortion
   case). H3's pentagon-rotation table doesn't port directly because
   mu3 uses the ``s7b`` aperture-7 rotation while H3 uses ``s7a``
   (Galois conjugates → mirror-image Eisenstein conventions). This
   module is committed for provenance only; it will likely be
   superseded by a rebuild on a dodecahedron-face base. See the
   2026-04-27 alpha-slerp report and the WIP discussion in the
   April 2026 sessions for context.

Algebraic / graph-theoretic — no projection, no sphere math. Every ring-1
step is a bottom-up carry over the digit sequence using the transition
table ``face_lattice.NEIGHBOR_TRANS``, plus a base-cell rewrap when the
carry reaches the root.

Tables:

- ``NEIGHBOR_TRANS[parity][d][D]`` (see :mod:`mu3.face_lattice`) — the
  Class II / Class III digit carry, derived once from the Eisenstein identity
  ``offset[d] + offset[D] = offset[d_new] + offset[D_carry] * ratio``
  with ``ratio = s7b`` (odd res) or ``ratio = s7a`` (even res).

- :data:`BASE_NEXT`, :data:`BASE_ROT` — analogues of H3's
  ``baseCellNeighbors`` / ``baseCellNeighbor60CCWRots``. ``BASE_NEXT[b][D]``
  is the neighbor pentagon reached by a root-level step in direction
  ``D ∈ {2,3,4,5,6}`` from base pentagon ``b``; ``BASE_ROT[b][D]`` is the
  number of 60° CCW rotations to apply to every digit of the result when
  re-expressing it in the new base's Eisenstein frame.

Pentagon handling (the K-skip):

- If the root-level carry is ``D = 1`` (the pentagon-deleted direction in
  the source base), apply the +60° CCW stitch: rotate every accumulated
  digit CCW once, and treat the root direction as ``5``.

- After a base hop, if the result's first non-zero digit is ``1``, the cell
  has landed in the new base's deleted wedge; rotate every digit CCW until
  the first non-zero digit is valid.
"""

from __future__ import annotations

import cmath
import math
from functools import lru_cache
from typing import Sequence

from . import icosahedron
from .face_lattice import (
    NEIGHBOR_TRANS,
    digit_offset,
    rotate_digit_ccw,
)


# --- Direction / digit geometry ---

# Directions to enumerate for ring-1 (digit 0 is center, skip).
_RING1_DIRECTIONS: tuple[int, ...] = (1, 2, 3, 4, 5, 6)

# Pentagon-wedge CCW digit sequence (digits march strictly CCW around the
# parent hex, with d=1 deleted); shared with :mod:`mu3.cell` and
# :mod:`mu3.icosahedron`.
_CCW_CYCLE: tuple[int, ...] = (2, 3, 4, 5, 6)

# For each face-digit ``d``, the digit whose Eisenstein ray sits at the
# CCW (upper-angle) boundary of ``d``'s wedge. With sequential digits this
# is the identity on 2..6.
_UPPER_DIGIT: dict[int, int] = {2: 2, 3: 3, 4: 4, 5: 5, 6: 6}


def _neighbor_vertex_of_digit(p: int, d: int) -> int:
    """Icosa vertex that pentagon ``p``'s digit-``d`` ray points to.

    Same derivation as ``mu3.cell._neighbor_vertex_of_digit``, inlined here
    to keep the neighbor module import-cheap.
    """
    F = icosahedron.faces()
    pft = icosahedron.pentagon_face_table()
    for i, d_curr in enumerate(_CCW_CYCLE):
        if _UPPER_DIGIT[d_curr] != d:
            continue
        d_next = _CCW_CYCLE[(i + 1) % 5]
        f_curr = int(pft[p, d_curr - 2])
        f_next = int(pft[p, d_next - 2])
        shared = (set(int(x) for x in F[f_curr]) & set(int(x) for x in F[f_next])) - {p}
        assert len(shared) == 1, (p, d, shared)
        return shared.pop()
    raise ValueError(f"no face has upper-boundary digit {d}")


# --- Base tables ---

def _build_base_tables() -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    base_next = [[-1] * 7 for _ in range(12)]
    base_rot = [[-1] * 7 for _ in range(12)]
    for b in range(12):
        for D in range(2, 7):
            b_prime = _neighbor_vertex_of_digit(b, D)
            # Direction in b''s frame pointing back to b.
            D_prime = -1
            for cand in range(2, 7):
                if _neighbor_vertex_of_digit(b_prime, cand) == b:
                    D_prime = cand
                    break
            if D_prime < 0:
                raise RuntimeError(f"no reverse direction for (b={b}, D={D})")
            # exp(-i*theta) where theta is the CCW rotation of b''s frame
            # relative to b's frame; r = # of CCW 60° steps to apply to every
            # old-frame digit to get the new-frame digit.
            exp_minus_theta = digit_offset[D_prime] / (-digit_offset[D])
            r = int(round(cmath.phase(exp_minus_theta) / (math.pi / 3))) % 6
            base_next[b][D] = b_prime
            base_rot[b][D] = r
    return (
        tuple(tuple(row) for row in base_next),
        tuple(tuple(row) for row in base_rot),
    )


BASE_NEXT, BASE_ROT = _build_base_tables()


# --- Ring-1 walk ---

def _first_nonzero(digits: Sequence[int]) -> int:
    for d in digits:
        if d != 0:
            return d
    return 0


def _canonicalize(base: int, digits: list[int]) -> tuple[int, list[int]]:
    """Apply CCW stitch until the first non-zero digit is not the deleted 1.

    Mu3's pentagon has a deleted wedge at digit 1; after a base hop the
    walk's raw output can land in that wedge, in which case the cell must
    be rotated into the stitched (digit-5) wedge.
    """
    while _first_nonzero(digits) == 1:
        digits = [rotate_digit_ccw(d, 1) for d in digits]
    return base, digits


def _rotate_digits(digits: list[int], rot: int) -> list[int]:
    rot %= 6
    if rot == 0:
        return digits
    return [rotate_digit_ccw(d, rot) for d in digits]


def _step(cell: tuple[int, ...], D: int) -> tuple[int, ...]:
    """Ring-1 neighbor of ``cell`` in direction ``D ∈ {1,...,6}``.

    Bottom-up Eisenstein carry; root carry triggers a base hop (and the
    pentagon-stitch special case when the root carry is the deleted digit).
    Digits after the base hop are rotated by ``BASE_ROT`` and stitched as
    needed so the first non-zero digit is valid.
    """
    base = cell[0]
    digits = list(cell[1:])
    N = len(digits)
    if N == 0:
        # Res 0: the "digit sequence" is empty; the direction is applied
        # directly at root.
        return _finish_root(base, [], D)

    # Bottom-up carry from the finest resolution upward.
    carry = D
    for k in range(N, 0, -1):
        if carry == 0:
            break
        parity = k % 2
        d_old = digits[k - 1]
        d_new, carry = NEIGHBOR_TRANS[parity][d_old][carry]
        digits[k - 1] = d_new

    return _finish_root(base, digits, carry)


def _finish_root(base: int, digits: list[int], root_carry: int) -> tuple[int, ...]:
    if root_carry == 0:
        new_base = base
        new_digits = digits
    else:
        # Root-carry of 1 is the deleted direction: the physical step lands
        # in the source base's deleted wedge. Stitch (CCW on the tail) and
        # treat the root direction as 5.
        if root_carry == 1:
            digits = [rotate_digit_ccw(d, 1) for d in digits]
            root_carry = 5
        new_base = BASE_NEXT[base][root_carry]
        rot = BASE_ROT[base][root_carry]
        new_digits = _rotate_digits(digits, rot)

    new_base, new_digits = _canonicalize(new_base, new_digits)
    return (new_base, *new_digits)


@lru_cache(maxsize=None)
def _res0_neighbors(base: int) -> tuple[tuple[int, ...], ...]:
    return tuple((int(b),) for b in icosahedron.vertex_neighbors()[base])


def cell_ring1(cell: Sequence[int]) -> list[tuple[int, ...]]:
    """All ring-1 neighbors of ``cell`` at the same resolution.

    Hex cells return 6 distinct neighbors; pentagon-center cells (and hex
    cells whose Eisenstein position lies on the pentagon's deleted-wedge
    seam) return 5 — duplicates are collapsed in the returned order.

    At res 0 the neighbors are the 5 icosa-vertex pentagons adjacent to
    ``cell[0]``.
    """
    cell_t = tuple(int(x) for x in cell)
    if len(cell_t) == 1:
        return list(_res0_neighbors(cell_t[0]))

    seen: set[tuple[int, ...]] = set()
    ring: list[tuple[int, ...]] = []
    for D in _RING1_DIRECTIONS:
        nb = _step(cell_t, D)
        if nb == cell_t:
            continue
        if nb in seen:
            continue
        seen.add(nb)
        ring.append(nb)
    return ring
