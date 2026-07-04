"""Ring-1 neighbor walk and point location via the holonomy cocycle —
exact integer arithmetic, no floats in the combinatorics.

Instead of resolving seam ambiguities from 3D geometry, the walk
*carries the groupoid arrow* — it transports the source cell's center
(or the query point, for point location) through every frame change
along with the step endpoint, and that transported pair decides every
ambiguity deterministically. (The
earlier geometric implementation — sphere-distance twin picks,
cross-product tiebreaks — was retired after matching this walk with
exact ordered-list equality on every cell at res 0-6, ~1.37M cells;
see git history.)

State is ``(p, Z, witness)``: current pentagon frame ``p``, developed
step endpoint ``Z`` (exact :class:`~mu3.eisenstein.Eis` scaled to
res-N lattice units, ``Z = z * get_rot(N)``), and a seam-side witness
transported through every frame change alongside ``Z``. The pair
``(Z, witness)`` is the p6 arrow in point form — a p6 element is
determined by the images of two points. One event loop
(:func:`_resolve`) serves both entry points; only the witness differs:
:class:`_ExactWitness` (ring walks — the transported source center)
and :class:`_FloatWitness` (point location — the unsnapped query, with
the CW tie nudge).

Why no sphere-distance disambiguation is needed
-----------------------------------------------

A phantom digit string (leading nonzero digit 1) means the developed
endpoint sits in the ghost subtree straddling the pentagon's deleted-
wedge seam. Its two digit-rotation twins are *different surface
points*; which one the walk reached is decided by whether the walked
segment crossed the cut ray (clockwise through the 60-degree ray, so
the canonical position is ``zeta**-1 * Z``) or entered from the glue
side (``zeta * Z``). That bit is exactly which side of the seam *line*
the witness lies on — ``cw_side``.

The endpoint alone is provably insufficient (fails for seam-crossing
walks like ``(p, 2) -> (p, 6)``), and the source in its *original*
frame is insufficient after a cross-pentagon hop (fails for
``(p, 6) -> (q0, 2)``) — hence the arrow must be carried. Both wrong
rules are pinned by named regressions in ``tests/test_neighbor.py``.

The earlier algebraic walk (``NEIGHBOR_TRANS``, removed in ``ec7048c``)
failed at these seams because digits + root carry lose the seam-side
bit. Its bottom-up carry is now revived as the interior fast tier —
:func:`_step_fast` owns the dispatch rule.
"""

from typing import Protocol, Sequence

from . import dodec
from .cell import cell_resolution, is_valid_cell
from .cross_pentagon import NEIGHBOR_ANGLE_IDX, TAU
from .eisenstein import (
    DIGIT_OFFSET,
    UNIT_EXP,
    UNITS,
    ZERO,
    ZETA,
    ZETA_INV,
    Eis,
    carry_digits,
    extract_digits,
    first_nonzero_digit,
    from_complex,
    get_rot_eis,
    scaled_center,
)
from .face_lattice import rotate_digit_ccw
from .p6 import P6

# Root residue zeta^m -> neighbor index k (deleted m=1 handled apart):
# the inverse of the angle-index table.
_M_TO_K = {a: k for k, a in enumerate(NEIGHBOR_ANGLE_IDX)}

# A ring-1 walk resolves in at most: extract, stitch, extract, hop,
# extract, phantom fix — i.e. <= 2 transformations between extracts.
# Algebraically: (1+omega) * (-omega) = 1, so un-stitching a deleted-
# direction residue produces at most a primary-direction residue,
# which then hops in one step.
_MAX_ITERS = 3


_ZETA_INV_C = ZETA_INV.to_complex()


class _Witness(Protocol):
    """The seam-side oracle :func:`_resolve` carries alongside the
    endpoint: it answers every stitch/phantom question (``cw_side``)
    and is transported through every frame change (``stitch``,
    ``hop``). The two implementations differ ONLY in arithmetic
    domain and tie policy."""

    def cw_side(self, gn: Eis) -> bool: ...

    def stitch(self, u: Eis) -> None: ...

    def hop(self, g: P6, gn: Eis) -> None: ...


class _ExactWitness:
    """Ring-walk witness: the transported source cell center, exact.

    Tie rule: a source EXACTLY on the seam line (``sp.b == 0``, e.g. a
    240-degree seam-line source) resolves CCW — the strict ``> 0``.
    (The float witness's tie is deliberately CW — see
    :class:`_FloatWitness`.)
    """

    __slots__ = ('s',)

    def __init__(self, s: Eis):
        self.s = s

    def cw_side(self, gn: Eis) -> bool:
        """True iff the witness lies on the CW side of the deleted-
        wedge seam line (the 60-degree ray through the root center) —
        i.e. the stitch resolves by ``zeta**-1``.

        Multiplying by ``conj(gn)`` first undoes the odd-resolution
        Class III rotation (+19.106 deg), putting the seam back on the
        exact lattice 60-degree line; forgetting it passes even
        resolutions and fails odd ones.
        """
        sp = ZETA_INV * (self.s * gn.conj())
        return sp.b > 0

    def stitch(self, u: Eis) -> None:
        self.s = u * self.s

    def hop(self, g: P6, gn: Eis) -> None:
        self.s = g.apply_scaled(self.s, gn)


class _FloatWitness:
    """Point-location witness: the unsnapped float query (consumed by
    ``resolve_position``; see its docstring for why the witness is
    needed at all).

    Tie rule — the ONE deliberate difference from the exact witness: a
    query ON the cut line resolves CW, matching the forward convention
    that the deleted wedge is ``[0, 60)`` degrees — a query on the
    primary ray forward-stitches exactly onto the cut, and its cell's
    canonical position is the un-stitched (CW) rep. Since the witness
    is float (projection round-trip noise ~1e-13 relative), "on the
    cut" is implemented by nudging the decision boundary a relative
    1e-9 into the CCW side: the closed cut belongs to CW, the anti-cut
    ray (180 degrees, ``sp.real < 0``) stays CCW.
    """

    __slots__ = ('w',)

    def __init__(self, w: complex):
        self.w = w

    def cw_side(self, gn: Eis) -> bool:
        sp = _ZETA_INV_C * (self.w * gn.to_complex().conjugate())
        return sp.imag > -1e-9 * sp.real

    def stitch(self, u: Eis) -> None:
        self.w = u.to_complex() * self.w

    def hop(self, g: P6, gn: Eis) -> None:
        # Float mirror of P6.apply_scaled (t * gn stays exact, then
        # floats) — the complex domain keeps it from routing through
        # the method itself.
        self.w = UNITS[g.u].to_complex() * self.w + (g.t * gn).to_complex()


def _resolve(
    p: int, z: Eis, witness: _Witness, res: int
) -> tuple[tuple[int, ...], int]:
    """Resolve a developed position to its canonical cell, carrying the
    arrow. ``z`` = scaled endpoint (exact); ``witness`` answers every
    seam-side question (stitch direction, phantom twin) and is
    transported through every frame change alongside ``z`` — an
    :class:`_ExactWitness` for ring walks, a :class:`_FloatWitness`
    for point location. The single event loop for both.

    Returns ``(cell, rot)`` where ``rot`` (an int mod 6) is the net
    rotation the walk applied to the developed picture — the arrow's
    rotation part. A physical direction vector expressed in the
    starting representation maps to the returned cell's canonical
    representation by ``zeta**rot``; equivalently, a direction that
    reads as digit ``e`` at the start reads as
    ``rotate_digit_ccw(e, rot)`` at the destination. Each stitch
    contributes its unit exponent, each cross-pentagon hop contributes
    ``TAU[p][k].u``, and the final phantom digit rotation contributes
    its step count (rotating digits CCW by ``n`` = rotating the
    position by ``zeta**n``).
    """
    gn = get_rot_eis(res)
    rot = 0
    for _ in range(_MAX_ITERS):
        digits, root = extract_digits(z, res)
        if root == ZERO:
            if first_nonzero_digit(digits) == 1:
                # Phantom: pick the twin on the witness's side of the
                # seam. Rotating about the root center commutes with
                # digit extraction, so rotate the digits directly
                # (0 -> 0 keeps deep zeros fixed).
                steps = 5 if witness.cw_side(gn) else 1
                digits = [rotate_digit_ccw(d, steps) for d in digits]
                rot += steps
            return (p, *digits), rot % 6
        m = UNIT_EXP.get(root)
        if m is None:
            raise RuntimeError(
                f'_resolve: non-unit root residue {root} at p={p}, z={z}'
            )
        if m == 1:
            # Deleted root direction: stitch toward the witness's side.
            if witness.cw_side(gn):
                u, du = ZETA_INV, 5
            else:
                u, du = ZETA, 1
            z = u * z
            witness.stitch(u)
            rot += du
        else:
            # Cross-pentagon hop: apply tau (scaled) to both points.
            k = _M_TO_K[m]
            g = TAU[p][k]
            z = g.apply_scaled(z, gn)
            witness.hop(g, gn)
            p = dodec.neighbors[p][k]
            rot += g.u
    raise AssertionError(
        f'_resolve: walks must resolve within {_MAX_ITERS} '
        f'extractions; stuck at p={p}, z={z}'
    )


def resolve_position(p: int, w: complex, res: int) -> tuple[int, ...]:
    """Canonical cell for a scaled query position — the point-location
    entry (consumed by ``index.vec3_to_cell_raw``).

    ``w`` is the query in pentagon ``p``'s frame scaled to res-N
    lattice units (``z_query * get_rot(res)`` with ``z_query`` from
    ``_sphere_to_flat``). It is snapped to the nearest lattice point
    exactly (``from_complex``), and then doubles as the *witness*: it
    plays the role the transported source center plays in the ring
    walk — every seam-side question is answered by which side of the
    cut the witness lies (:class:`_FloatWitness`, including the CW tie
    rule), through the same event loop (:func:`_resolve`).

    In the initial frame the witness looks redundant: inverse
    projection returns rendered (post-forward-stitch) coordinates,
    where a phantom-form snap always resolves CW (its CW twin is the
    same 3D point by injectivity of the forward map — which is also
    why the old sphere-distance twin pick always chose it). But a
    cross-pentagon hop lands the position near the *neighbor's* cut in
    coordinates that are not rendered ones, and there the twin choice
    genuinely varies — hence the witness.
    """
    cell, _ = _resolve(p, from_complex(w), _FloatWitness(w), res)
    return cell


def _position_step(
    cell_t: tuple, res: int, d: int
) -> tuple[tuple[int, ...], int]:
    """The pure position walk for one step: develop the endpoint,
    resolve with the arrow. The seam-event fallback of
    :func:`_step_fast`, and the oracle the carry tier is verified
    against (exhaustive equality test + scripts/verify_carry_walk.py).
    """
    z_c = scaled_center(cell_t[1:])
    return _resolve(
        cell_t[0], z_c + DIGIT_OFFSET[d], _ExactWitness(z_c), res
    )


def _step_fast(
    cell_t: tuple, res: int, d: int
) -> tuple[tuple[int, ...], int]:
    """One walk step: digit-carry fast path for the flat interior,
    exact arrow walk for every seam event.

    The dispatch is exact — no float, no margin: the carry escaping
    the root (a root-level residue: stitch/hop territory) or landing
    on a phantom-FORM string (leading nonzero digit 1: the witness
    twin fix, including the pentagon ``d=1`` collapse) IS the seam
    event. Otherwise the carry result is exactly what the position
    walk would extract, with a trivial arrow (``rot == 0``) — pinned
    by the exhaustive carry-vs-walk equality test.
    """
    digits = carry_digits(cell_t[1:], d)
    if digits is not None and first_nonzero_digit(digits) != 1:
        return (cell_t[0], *digits), 0
    return _position_step(cell_t, res, d)


def step(cell: Sequence[int], d: int) -> tuple[tuple[int, ...], int]:
    """One ring-1 walk from ``cell`` in digit direction ``d``:
    ``(dest, rot)``.

    ``rot`` (mod 6) is the arrow's rotation part between the two
    cells' frames: a physical direction that reads as digit ``e`` from
    ``cell`` reads as ``rotate_digit_ccw(e, rot)`` from ``dest``. In
    particular the reverse edge's direction is
    ``rotate_digit_ccw(opposite(d), rot)`` — see ``mu3.edge``.

    Permissive at the deleted direction: walking ``d=1`` out of a
    pentagon-center cell collapses onto the ``d=2`` neighbor (the
    stitch self-loop). The edge layer forbids that pair; this walk
    primitive reports where the step lands.
    """
    cell_t = tuple(int(x) for x in cell)
    d = int(d)
    if not 1 <= d <= 6:
        raise ValueError(f'step: direction must be in 1..6, got {d}')
    if not is_valid_cell(cell_t):
        raise ValueError(f'step: invalid cell {cell_t}')
    return _step_fast(cell_t, cell_resolution(cell_t), d)


def cell_ring1(cell: Sequence[int]) -> list[tuple[int, ...]]:
    """All ring-1 neighbors of ``cell``, CCW around the source on the
    sphere with the primary-direction neighbor last (walk order
    D = 1..6), computed exactly via the cocycle.

    Pentagon cells (any resolution, including res 0) return 5
    neighbors — the deleted-direction walk collapses onto the d=2
    neighbor and dedups; hex cells return 6.
    """
    cell_t = tuple(int(x) for x in cell)
    if not is_valid_cell(cell_t):
        raise ValueError(f'cell_ring1: invalid cell {cell_t}')
    res = cell_resolution(cell_t)

    seen: set[tuple[int, ...]] = {cell_t}
    out: list[tuple[int, ...]] = []
    for D in (1, 2, 3, 4, 5, 6):
        nb, _ = _step_fast(cell_t, res, D)
        if nb in seen:
            continue
        seen.add(nb)
        out.append(nb)
    return out
