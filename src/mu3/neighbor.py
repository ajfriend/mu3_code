"""Ring-1 neighbor walk via canonicalize + z_to_cell.

For each cell at res N, step in each of the 6 digit-offset directions.
For pentagon-adjacent hex cells, one or two of those steps land at a
*phantom corner* -- a position in the lattice that isn't a cell
center because one of the surrounding hexes is the deleted-d=1
phantom. We detect this when ``z_to_cell`` fails (non-zero root
residue) and recover by un-folding: rotate the position by -60 deg
in the canonical pentagon's frame, re-canonicalize, and try again.
The un-fold maps the phantom corner to its real-cell partner across
the icosa triangle.

Pentagon-adjacent hex cells return 5 neighbors (the deleted slot
collapses via the +60 deg intra-stitch self-loop); other hex cells
return 6.
"""

import cmath
import math
from functools import lru_cache
from typing import Sequence

from . import icosahedron
from .cell import _eisenstein_center, cell_resolution
from .cross_pentagon import (
    CROSS_PENTAGON,
    EISENSTEIN_UNITS,
    NEIGHBOR_ANGLE_IDX,
    canonicalize,
    z_to_cell,
)
from .face_lattice import digit_offset, get_rot, omega, rotate_digit_ccw


# -60 deg rotation = 1 / exp(i*60 deg) = 1 / (1 + omega) = -omega in Z[omega].
_UNFOLD_ROT = -omega


@lru_cache(maxsize=None)
def _res0_neighbors(base: int) -> tuple[tuple[int, ...], ...]:
    """The 5 neighbor pentagons at res 0, taken straight from dodec."""
    return tuple((int(b),) for b in icosahedron.vertex_neighbors()[base])


_MAX_RESIDUE_ITERS = 12


# Mapping digit ``D`` (2..6) to the k-index of the corresponding neighbor
# pentagon. ``digit_offset[D]`` equals ``EISENSTEIN_UNITS[idx]`` where
# ``idx`` is the angle index; ``k`` satisfies ``NEIGHBOR_ANGLE_IDX[k] == idx``.
_D_TO_K = {6: 0, 2: 1, 3: 2, 4: 3, 5: 4}


def _z_to_cell_with_residue(p: int, z: complex, res: int):
    """Like :func:`z_to_cell`, but returns ``(digits, residue)`` rather than
    raising on non-zero root residue.
    """
    from .face_lattice import digit_for_offset, divmod_ei, get_rot, s7a, s7b

    if abs(z) < 1e-12:
        return [0] * res, 0j

    digits = [0] * res
    cur = z * get_rot(res)
    for k in range(res, 0, -1):
        ratio = s7b if (k % 2) == 1 else s7a
        cur, r = divmod_ei(cur, ratio)
        d = digit_for_offset(r)
        if d < 0:
            raise RuntimeError(
                f"_z_to_cell_with_residue: bad remainder at level {k}: {r}"
            )
        digits[k - 1] = d
    return digits, cur


def _step_to_cell(z_n: complex, base: int, res: int) -> tuple[int, ...]:
    """Resolve a walked position ``z_n`` to a canonical cell tuple.

    Iterative, residue-driven absorption -- no canonicalize, no BFS:

    1. Run divmod from finest to coarsest level in the current pentagon
       ``p``'s frame. This always extracts a digit string; the residue
       at the root is ``digit_offset[D]`` for some ``D in {0..6}``.
    2. If ``D == 0``: residue is zero, cell is ``(p, *digits)``. Done.
    3. If ``D == 1``: the deleted-direction carry -- the position has
       a +60 deg stitched twin in the deleted wedge. Replace ``z`` with
       ``z * -omega`` (rotate -60 deg into the deleted wedge) and loop.
    4. If ``D in {2..6}``: the canonical cell lives in ``p``'s direction-D
       neighbor. Hop via :data:`CROSS_PENTAGON` and loop.

    Each iteration either rotates ``z`` or hops to a neighbor pentagon
    -- both isometric on the underlying 3D point. Converges in at most
    a couple of iterations for the resolutions we run; bounded by
    :data:`_MAX_RESIDUE_ITERS` as a safety net.
    """
    from .face_lattice import digit_for_offset

    p = base
    z = z_n
    for _ in range(_MAX_RESIDUE_ITERS):
        digits, residue = _z_to_cell_with_residue(p, z, res)
        if abs(residue) < 1e-9:
            return (p, *digits)
        D = digit_for_offset(residue)
        if D == 1:
            z = z * _UNFOLD_ROT     # rotate -60 deg into the deleted wedge
            continue
        if D in _D_TO_K:
            k = _D_TO_K[D]
            q, qci, ri, _ = CROSS_PENTAGON[p][k]
            z = (z - EISENSTEIN_UNITS[qci]) / EISENSTEIN_UNITS[ri]
            p = q
            continue
        raise RuntimeError(
            f"_step_to_cell: unexpected residue {residue} (D={D}) "
            f"at p={p}, z={z}"
        )
    raise RuntimeError(
        f"_step_to_cell: residue did not converge for z_n={z_n} from "
        f"base={base} at res={res}"
    )


def _has_leading_zero_d1(cell_t: tuple[int, ...]) -> bool:
    """True iff ``cell_t``'s digits have leading-zero d=1 (the deleted
    slot, excluded from canonical cell indexing)."""
    digits = cell_t[1:]
    first_nz = next((d for d in digits if d != 0), None)
    return first_nz == 1


def cell_ring1(cell: Sequence[int]) -> list[tuple[int, ...]]:
    """All ring-1 neighbors of ``cell`` at the same resolution.

    Walks 1 unit step in each of the 6 digit directions. When a walk
    lands at a phantom (deleted-form digit string), the canonical cell
    at that 3D point depends on the walk direction relative to the
    pentagon center: rotate the phantom's digits CCW or CW by 1
    according to the angular sense of the walk
    (``cross(z_source, step) >= 0`` -> CCW, else CW).

    Pentagon-center cells and pentagon-adjacent hex cells return 5
    neighbors; other hex cells return 6.

    At res 0 the neighbors are the 5 icosa-vertex pentagons adjacent
    to ``cell[0]``.
    """
    cell_t = tuple(int(x) for x in cell)
    if cell_resolution(cell_t) == 0:
        return list(_res0_neighbors(cell_t[0]))

    base = cell_t[0]
    digits = cell_t[1:]
    res = cell_resolution(cell_t)
    z_C = _eisenstein_center(digits)
    rot_N = get_rot(res)

    seen: set[tuple[int, ...]] = {cell_t}
    out: list[tuple[int, ...]] = []

    # Source 3D position for sphere-distance disambiguation.
    from .cell import cell_center
    src_3d = cell_center(cell_t)

    # Pass 1: walk all 6 directions in D=1..D=6 order, classifying each
    # as direct or phantom. Collect the direct cells into a set so
    # phantom-twin selection in Pass 2 can avoid duplicating them.
    raw: list[tuple[str, ...]] = []
    direct_set: set[tuple[int, ...]] = set()
    for D in (1, 2, 3, 4, 5, 6):
        step = digit_offset[D] / rot_N
        z_n = z_C + step
        nb = _step_to_cell(z_n, base, res)
        if _has_leading_zero_d1(nb):
            raw.append(("phantom", z_n, step, nb))
        else:
            raw.append(("direct", nb))
            direct_set.add(nb)

    # Pass 2: emit in walk order (CCW around the source on the sphere),
    # disambiguating phantoms with knowledge of all direct neighbors.
    for entry in raw:
        if entry[0] == "direct":
            nb = entry[1]
        else:
            _, z_n, step, raw_nb = entry
            # Pick between CCW and CW digit-rotation twins.
            # The +60 deg intra-pentagon stitch is a 3D identification,
            # but in the flat frame the "correct" rotation direction
            # depends on the source's position on the icosa surface.
            ccw = (raw_nb[0], *(rotate_digit_ccw(d, 1) for d in raw_nb[1:]))
            cw = (raw_nb[0], *(rotate_digit_ccw(d, 5) for d in raw_nb[1:]))
            # Prefer a twin that isn't self and isn't already a direct
            # neighbor -- those are real new ring-1 neighbors.
            ccw_avail = ccw != cell_t and ccw not in direct_set
            cw_avail = cw != cell_t and cw not in direct_set
            if ccw_avail and not cw_avail:
                nb = ccw
            elif cw_avail and not ccw_avail:
                nb = cw
            elif ccw_avail and cw_avail:
                ccw_3d = cell_center(ccw)
                cw_3d = cell_center(cw)
                d_ccw = float(((src_3d - ccw_3d) ** 2).sum())
                d_cw = float(((src_3d - cw_3d) ** 2).sum())
                if abs(d_ccw - d_cw) < 1e-12:
                    # On a tie, break with the angular sense of the walk.
                    cross = z_C.real * step.imag - z_C.imag * step.real
                    nb = ccw if cross >= 0 else cw
                else:
                    nb = ccw if d_ccw < d_cw else cw
            else:
                # Both are self or duplicates -- skip (deleted-direction
                # collapse).
                continue

        if nb in seen:
            continue
        seen.add(nb)
        out.append(nb)
    return out
