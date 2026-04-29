"""Cross-pentagon re-rooting and digit-string recovery.

Two peer primitives, both pure integer arithmetic in the Eisenstein
ring ``Z[omega]``:

- :func:`canonicalize` -- given an Eisenstein point ``z`` in pentagon
  ``p``'s flat lattice frame, decide which pentagon's territory ``z``
  actually sits in, re-root accordingly, and apply the +60 deg
  intra-stitch if the result lands in the canonical pentagon's deleted
  wedge.
- :func:`z_to_cell` -- inverse of :func:`mu3.cell._eisenstein_center`:
  recover a digit string at a given resolution by iterated
  ``divmod_ei`` from finest to coarsest level.

Cross-pentagon table (precomputed at module load):

For each pentagon ``p`` (0..11) and CCW-from-primary neighbor index
``k`` (0..4), ``CROSS_PENTAGON[p][k] = (q, q_center_idx, rot_idx, j)``
where:

- ``q = dodec.neighbors[p][k]`` is the neighbor pentagon.
- ``q_center_idx`` indexes :data:`EISENSTEIN_UNITS` to give Q's center
  in P's flat frame: one of the 5 unit positions
  ``{0 deg, 120 deg, 180 deg, 240 deg, 300 deg}``.
- ``rot_idx`` indexes :data:`EISENSTEIN_UNITS` to give the rotation
  that maps Q's local frame into the joint (P-flat) frame:
  ``alpha_p + 180 deg - alpha_q (mod 360 deg)``.
- ``j = neighbors[q].index(p)`` for the inverse lookup.

The fact that both shift and rotation are Eisenstein units (sixth
roots of unity) means the re-rooting transform is purely
integer-valued in the lattice -- no floats, no rationalization.
"""

from __future__ import annotations

import math

from . import dodec
from .face_lattice import (
    digit_for_offset,
    digit_offset,
    divmod_ei,
    get_rot,
    omega,
    s7a,
    s7b,
)


# Six Eisenstein units, ordered by angle (multiples of 60 deg CCW from +x).
EISENSTEIN_UNITS: tuple[complex, ...] = (
    1,           # idx 0: 0 deg
    1 + omega,   # idx 1: 60 deg
    omega,       # idx 2: 120 deg
    -1,          # idx 3: 180 deg
    -1 - omega,  # idx 4: 240 deg
    -omega,      # idx 5: 300 deg
)

# Neighbor index k (0..4) -> EISENSTEIN_UNITS idx for the 5 corner
# angles {0, 120, 180, 240, 300} deg in the flat-Eisenstein layout.
# (Index 1 = 60 deg is the deleted-wedge direction; not a neighbor.)
NEIGHBOR_ANGLE_IDX: tuple[int, ...] = (0, 2, 3, 4, 5)


def _build_cross_pentagon_table() -> tuple:
    out = []
    for p in range(12):
        row = []
        for k in range(5):
            q = dodec.neighbors[p][k]
            j = list(dodec.neighbors[q]).index(p)
            ap = NEIGHBOR_ANGLE_IDX[k]
            aq = NEIGHBOR_ANGLE_IDX[j]
            rot_idx = (ap + 3 - aq) % 6
            row.append((q, ap, rot_idx, j))
        out.append(tuple(row))
    return tuple(out)


CROSS_PENTAGON = _build_cross_pentagon_table()


_STITCH_ROT = EISENSTEIN_UNITS[1]   # +60 deg CCW, the intra-pentagon stitch
_DELETED_HI_RAD = math.pi / 3


def _maybe_stitch(z: complex) -> complex:
    """Rotate by +60 deg CCW if ``z`` is in the deleted wedge ``[0, 60) deg``."""
    if z == 0j:
        return z
    a = math.atan2(z.imag, z.real) % (2 * math.pi)
    if 0.0 <= a < _DELETED_HI_RAD:
        return z * _STITCH_ROT
    return z


def canonicalize(z: complex, p: int) -> tuple[int, complex]:
    """``(p_canonical, z_canonical)`` for an Eisenstein point ``z`` in ``p``'s frame.

    Composes:

    1. Territory check across ``p``'s 5 neighbors -- re-root if ``z`` is in a
       neighbor's territory (shift+rotate, both Eisenstein units).
    2. Intra-stitch: +60 deg if the result lands in ``[0, 60) deg``.

    Idempotent on cell-center positions:
    ``canonicalize(canonicalize(z, p)) == canonicalize(z, p)``.
    """
    # Territory check: project z onto each unit-direction-to-neighbor.
    # Threshold 0.5 = perpendicular bisector to that neighbor. Strict
    # ``>``: a cell center never lies exactly on the bisector, so a
    # projection of exactly 0.5 indicates a phantom corner on the icosa
    # edge (handled by the caller's un-fold retry, not by tiebreaking).
    best_proj = 0.5
    best_k = -1
    for k in range(5):
        q_center = EISENSTEIN_UNITS[NEIGHBOR_ANGLE_IDX[k]]
        proj = (z * q_center.conjugate()).real    # |q_center| = 1
        if proj > best_proj:
            best_proj = proj
            best_k = k

    if best_k < 0:
        return p, _maybe_stitch(z)

    q, q_center_idx, rot_idx, _ = CROSS_PENTAGON[p][best_k]
    z_local = (z - EISENSTEIN_UNITS[q_center_idx]) / EISENSTEIN_UNITS[rot_idx]
    return q, _maybe_stitch(z_local)


def z_to_cell(p: int, z: complex, res: int) -> tuple:
    """Inverse of :func:`mu3.cell._eisenstein_center`: digit-string recovery.

    At each level k from ``res`` down to 1, divmod by the level's ratio
    (``s7b`` for odd k, ``s7a`` for even k). The remainder is
    ``digit_offset[d_k]``; the quotient is the residual lifted to level
    k-1. After ``res`` iterations the residue must be 0 (z lay cleanly
    in p's frame at this resolution).

    Returns the cell tuple ``(p, d_1, ..., d_res)``.
    """
    if abs(z) < 1e-12:
        return (p,) + (0,) * res

    digits = [0] * res
    cur = z * get_rot(res)
    for k in range(res, 0, -1):
        ratio = s7b if (k % 2) == 1 else s7a
        cur, r = divmod_ei(cur, ratio)
        d = digit_for_offset(r)
        if d < 0:
            raise RuntimeError(
                f"z_to_cell: bad remainder at level {k}: {r}"
            )
        digits[k - 1] = d
    if abs(cur) > 1e-9:
        raise RuntimeError(
            f"z_to_cell: residue {cur} at root; "
            f"z does not lie cleanly in {p}'s frame at res {res}"
        )
    return (p, *digits)
