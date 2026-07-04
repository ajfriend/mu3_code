"""The cross-pentagon transition table and its holonomy cocycle.

Primitives:

- :data:`CROSS_PENTAGON` / :data:`TAU` -- the pentagon-to-pentagon
  frame transitions, as index tuples and as exact :class:`~mu3.p6.P6`
  group elements respectively. The cocycle consistency laws (and the
  derived +60 deg stitch) are asserted at import by
  :func:`_check_cocycle`.
- :func:`z_to_cell` -- inverse of :func:`mu3.cell._eisenstein_center`:
  recover a digit string at a given resolution by iterated
  ``divmod_ei`` from finest to coarsest level. (Float; used by
  plotting scripts. The exact runtime path is
  ``eisenstein.extract_digits``.)

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

from . import dodec
from .eisenstein import (
    UNITS as EIS_UNITS,
    ZERO as EIS_ZERO,
    ZETA as EIS_ZETA,
    Eis,
    get_rot_eis,
)
from .face_lattice import (
    digit_for_offset,
    divmod_ei,
    get_rot,
    s7a,
    s7b,
)
from .p6 import IDENTITY, P6, STITCH, rotation_about


# Six Eisenstein units, ordered by angle (multiples of 60 deg CCW from
# +x); the float view of eisenstein.UNITS, derived so the exact table
# is the single source of truth.
EISENSTEIN_UNITS: tuple[complex, ...] = tuple(
    u.to_complex() for u in EIS_UNITS
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


# ---------------------------------------------------------------------------
# The holonomy cocycle: CROSS_PENTAGON retyped as p6 group elements.
#
# TAU[p][k] is the frame transition for hopping from pentagon p to its
# k-th neighbor q = dodec.neighbors[p][k], as an exact P6 element:
#
#     z_q = zeta^{-rot_idx} * z_p + zeta^{aq}
#
# (the translation part is p's center as seen from q). This is the
# existing table retyped, not new data; canonicalize's float hop
# ``(z - EISENSTEIN_UNITS[qci]) / EISENSTEIN_UNITS[ri]`` is the same map.
#
# BRANCH CUTS — why the consistency laws below look the way they do.
# The idealized cocycle condition would read "product of the 5
# transitions around a pentagon == the stitch". That
# is NOT literally true for the stored taus: each chart carries a
# deleted-wedge cut ending at its k=0 corner (the corner between the
# primary-direction neighbor and the next one CCW), and loop products
# pick up a stitch factor each time the loop crosses a cut. The exact,
# verified laws — uniform over all 12 pentagons — are:
#
#   * edge symmetry: tau(q->p) == tau(p->q)^{-1}          (60 pairs)
#   * corner triple products p -> q -> r -> p, where q, r are p's
#     neighbors k and k+1 (a loop around a triangle corner, based in
#     p's frame), classified by k:
#       k=0    -> STITCH = P6(1, 0)   (p's own cut ends at this corner;
#                                      this product DERIVES the stitch)
#       k=1, 3 -> IDENTITY            (no cut ends here; the flat case)
#       k=2    -> rotation_about(1, zeta^3)  (q's cut ends here;
#                                      zeta^3 = q's center in p's frame)
#       k=4    -> rotation_about(1, zeta^0)  (r = neighbors[p][0]'s cut)
#   * pentagon 6-factor CCW loop p -> q0 -> ... -> q4 -> p:
#       P6(2, omega)  — two cut crossings' worth of stitch, not one.
#
# Do not "simplify" these expected values back to the idealized
# equation; the k-classification is the real condition.
# ---------------------------------------------------------------------------


def _build_tau() -> tuple:
    out = []
    for p in range(12):
        row = []
        for k in range(5):
            _q, _ap, rot_idx, j = CROSS_PENTAGON[p][k]
            aq = NEIGHBOR_ANGLE_IDX[j]
            row.append(P6(-rot_idx, EIS_UNITS[aq]))
        out.append(tuple(row))
    return tuple(out)


TAU = _build_tau()


def tau_between(a: int, b: int) -> P6:
    """The transition ``a -> b`` for adjacent pentagons ``a``, ``b``."""
    k = list(dodec.neighbors[a]).index(b)
    return TAU[a][k]


def cut_rays(base: int, res: int) -> list[tuple[int, Eis, Eis]]:
    """The six deleted-wedge cut rays near ``base``, as exact
    ``(owner, apex, unit_direction)`` triples in ``base``'s scaled
    res-N frame: its own cut (apex at the origin, direction ``zeta``)
    and each neighbor pentagon's cut (apex at ``zeta^angle * GN``)
    pulled back through the inverse TAU rotation. ``owner`` is the
    pentagon whose chart carries the cut — its identity travels with
    the ray, so consumers never decode it from list position.

    This is the executable form of the branch-cut geometry documented
    in the comment block above — each pentagon's cut ends at its k=0
    corner and runs along its own chart's 60-degree direction.
    """
    gn = get_rot_eis(res)
    rays = [(base, EIS_ZERO, EIS_ZETA)]
    for j, a in enumerate(NEIGHBOR_ANGLE_IDX):
        apex = EIS_UNITS[a] * gn
        direction = EIS_UNITS[TAU[base][j].inverse().u] * EIS_ZETA
        rays.append((int(dodec.neighbors[base][j]), apex, direction))
    return rays


def _corner_product(p: int, k: int) -> P6:
    """Holonomy of the loop ``p -> q -> r -> p`` around the triangle
    corner between neighbors ``k`` and ``k+1``, based in ``p``'s frame."""
    q = dodec.neighbors[p][k]
    r = dodec.neighbors[p][(k + 1) % 5]
    return tau_between(r, p).compose(tau_between(q, r)).compose(
        tau_between(p, q))


# The stitch, DERIVED from three cross-pentagon transitions rather
# than hardcoded (_check_cocycle asserts it equals STITCH == P6(1, 0)).
DERIVED_STITCH = _corner_product(0, 0)

# The expected loop products — the single copy of the branch-cut-aware
# spec, consumed by _check_cocycle at import and re-asserted with
# readable parametrized failures in tests/test_cocycle.py.
CORNER_EXPECTED: dict[int, P6] = {
    0: STITCH,
    1: IDENTITY,
    2: rotation_about(1, EIS_UNITS[NEIGHBOR_ANGLE_IDX[2]]),
    3: IDENTITY,
    4: rotation_about(1, EIS_UNITS[NEIGHBOR_ANGLE_IDX[0]]),
}
PENTAGON_LOOP_EXPECTED = P6(2, EIS_UNITS[2])   # zeta^2 = omega


def _check_cocycle() -> None:
    """Build-time consistency spec for the cocycle (see comment block
    above). A future edit that breaks orientation anywhere in
    dodec.neighbors / CROSS_PENTAGON fails here, loudly, at import."""
    for p in range(12):
        for k in range(5):
            q = dodec.neighbors[p][k]
            assert tau_between(q, p) == tau_between(p, q).inverse(), \
                f'edge symmetry broken at p={p}, k={k}'

    for p in range(12):
        for k in range(5):
            prod = _corner_product(p, k)
            assert prod == CORNER_EXPECTED[k], \
                f'corner condition broken at p={p}, k={k}: {prod}'

    for p in range(12):
        cycle = [p, *dodec.neighbors[p], p]
        prod = IDENTITY
        for a, b in zip(cycle, cycle[1:]):
            prod = tau_between(a, b).compose(prod)
        assert prod == PENTAGON_LOOP_EXPECTED, \
            f'pentagon loop broken at p={p}: {prod}'

    assert DERIVED_STITCH == STITCH, DERIVED_STITCH


_check_cocycle()
