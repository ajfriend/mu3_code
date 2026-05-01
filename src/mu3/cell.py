"""Forward indexing: cell index -> sphere center and boundary.

A cell is a tuple ``(base, d_1, d_2, ..., d_N)`` where:
  - ``base`` is an icosa vertex index (0..11)
  - ``d_1..d_N`` are child digits in {0..6}
  - ``N`` = ``len(cell) - 1`` = the cell's resolution

At res 0 a cell is a 1-tuple ``(base,)``. The first nonzero child digit
cannot be 1 (pentagon-deleted direction).

Pipeline (pentagon-centric; same for cell centers and boundary corners):

1. Accumulate the Eisenstein position ``z`` from the digit sequence,
   centered on the base pentagon.
2. If ``arg(z) ∈ [0°, 60°)`` — the deleted wedge sitting immediately CCW
   of the primary direction — rotate by +60° onto d=2's wedge.
3. Classify the post-stitch angle into one of five 60° wedges, owned by
   face-digits ``d ∈ {2, 3, 4, 5, 6}``.
4. Read off barycentric weights ``(b_p, b_cw, b_ccw)`` for the flat
   wedge with corners ``(0, exp(i·(d-1)·60°), exp(i·d·60°))``.
5. Hand those weights to the active :class:`Projection` instantiated on
   the spherical triangle ``(V[p], V[n_cw], V[n_ccw])`` where
   ``n = vertex_neighbors[p]`` and ``n_cw = n[d-2]``,
   ``n_ccw = n[(d-1) % 5]``.

Pentagon-center cells fall out of the same pipeline: corner k=0 sits at
Eisenstein angle 30° (in the deleted wedge), gets rotated +60° to 90°,
and coincides with corner k=1 — so the 6-corner hex formula naturally
produces a 5-gon.
"""

import cmath
import itertools
import math
from functools import lru_cache
from typing import Iterator, Sequence

import numpy as np

from . import icosahedron
from .face_lattice import digit_offset, get_rot, s3, units
from .projection import AlphaSlerp, Projection

_PROJECTION_CLS: type[Projection] = AlphaSlerp

# +60° rotation in the pentagon-Eisenstein plane (stitching for the deleted wedge).
_ROT60 = cmath.exp(1j * math.pi / 3)

# Deleted wedge in Eisenstein (triangle convention): [0°, 60°). Sits
# immediately CCW of the primary direction at 0°. Points here get rotated
# +60° into d=2's (post-stitch) wedge [60°, 120°).
_DELETED_LO = 0.0
_DELETED_HI = 60.0

def _angle_deg(z: complex) -> float:
    return math.degrees(math.atan2(z.imag, z.real)) % 360.0


def _stitch(z: complex) -> complex:
    """If z sits in the deleted wedge, rotate it +60° onto d=2's wedge."""
    if z == 0j:
        return z
    if _DELETED_LO <= _angle_deg(z) < _DELETED_HI:
        return z * _ROT60
    return z


def _classify_stitched(z: complex) -> int:
    """Face-digit d whose Eisenstein wedge contains a POST-STITCH point z."""
    a = _angle_deg(z)
    if a < 120.0:
        return 2  # absorbs the (now-empty post-stitch) [0, 60) via the stitch
    if a < 180.0:
        return 3
    if a < 240.0:
        return 4
    if a < 300.0:
        return 5
    return 6


_INV_SQRT3 = 1.0 / math.sqrt(3.0)


@lru_cache(maxsize=None)
def _wedge_align_rot(d: int) -> complex:
    """``exp(-i·(d-1)·60°)`` — rotates digit ``d``'s wedge CW boundary onto +x."""
    return cmath.exp(-1j * math.radians((d - 1) * 60.0))


def _wedge_barycentric(z: complex, d: int) -> np.ndarray:
    """Barycentric weights ``(b_p, b_cw, b_ccw)`` for Eisenstein ``z`` in
    digit ``d``'s flat 60° wedge with corners
    ``(0, exp(i·(d-1)·60°), exp(i·d·60°))``.

    Barycentric is affine-invariant, so the same weights map directly onto
    the 3D triangle ``(V[p], V[n_cw], V[n_ccw])`` under the active
    :class:`Projection`.
    """
    z_aligned = z * _wedge_align_rot(d)
    b_ccw = 2.0 * z_aligned.imag * _INV_SQRT3
    b_cw = z_aligned.real - 0.5 * b_ccw
    b_p = 1.0 - b_cw - b_ccw
    return np.array([b_p, b_cw, b_ccw])


@lru_cache(maxsize=None)
def _projection(p: int, d: int) -> Projection:
    """Active :class:`Projection` on the spherical triangle
    ``(V[p], V[n_cw], V[n_ccw])``, where ``n = vertex_neighbors[p]``,
    ``n_cw = n[d-2]``, ``n_ccw = n[(d-1) % 5]``. 12 × 5 = 60 cached entries.
    """
    V = icosahedron.vertices()
    n = icosahedron.vertex_neighbors()[p]
    return _PROJECTION_CLS(V[p], V[n[d - 2]], V[n[(d - 1) % 5]])


def _project(z: complex, base: int) -> np.ndarray:
    """Eisenstein → stitch → wedge barycentric → projection → sphere."""
    if z == 0j:
        return icosahedron.vertices()[base].copy()
    z_s = _stitch(z)
    d = _classify_stitched(z_s)
    beta = _wedge_barycentric(z_s, d)
    return _projection(base, d).to_sphere(beta)


def _z_from_wedge_barycentric(beta: np.ndarray, d: int) -> complex:
    """Inverse of :func:`_wedge_barycentric`: barycentric `(b_p, b_cw, b_ccw)`
    on digit ``d``'s flat 60° wedge → Eisenstein ``z``."""
    b_cw = float(beta[1])
    b_ccw = float(beta[2])
    z_aligned = complex(b_cw + 0.5 * b_ccw, b_ccw * math.sqrt(3.0) / 2.0)
    return z_aligned / _wedge_align_rot(d)


def _sphere_to_flat(p3d: np.ndarray, base: int) -> complex:
    """Inverse of :func:`_project`: 3D unit vector inside ``base``'s spherical
    territory → Eisenstein ``z`` in ``base``'s flat frame.

    Identifies which of the 5 incident icosa triangles around ``base``
    contains ``p3d`` by trying each ``d ∈ {2..6}`` and accepting the one
    whose barycentric weights are all non-negative (within tolerance).
    Then runs the active :class:`Projection`'s ``to_bary`` on that
    triangle and converts the barycentric back to ``z`` in d's flat wedge.

    The returned ``z`` is in canonical post-stitch form (angle in
    ``[60°, 360°)`` at any non-degenerate point); the deleted wedge
    ``[0°, 60°)`` is unreachable here by construction.
    """
    eps = 1e-9
    best_d = -1
    best_beta: np.ndarray | None = None
    best_min = -float("inf")
    for d in range(2, 7):
        proj = _projection(base, d)
        try:
            beta = proj.to_bary(p3d)
        except RuntimeError:
            continue
        m = float(beta.min())
        if m >= -eps:
            return _z_from_wedge_barycentric(beta, d)
        if m > best_min:
            best_min = m
            best_d = d
            best_beta = beta
    # No triangle reported strictly-non-negative barycentric. This happens
    # for points exactly on a shared edge between two triangles, where
    # numerical noise pushes one barycentric slightly negative. Return the
    # best available — caller's polish step will catch any sign error.
    if best_d < 0 or best_beta is None:
        raise RuntimeError(
            f"_sphere_to_flat: no incident triangle of base {base} "
            f"contains {p3d}"
        )
    return _z_from_wedge_barycentric(best_beta, best_d)


def _eisenstein_center(digits: Sequence[int]) -> complex:
    """Pentagon-Eisenstein position ``z`` for a child-digit sequence."""
    z = 0j
    for k, d in enumerate(digits, start=1):
        if d == 0:
            continue
        z += digit_offset[d] / get_rot(k)
    return z


def cells_at_res(res: int) -> Iterator[tuple]:
    """Yield every valid mu3 cell at resolution ``res``, in canonical order
    (by base, then by digit-sequence lexicographic order, skipping any path
    whose first nonzero child digit is 1).

    Counts:
      - res 0:   12
      - res N:   12 * (7**N - (7**N - 1)//6)    [= 72, 492, 3432, ... for N=1,2,3]

    Streaming — no intermediate list is built.
    """
    if res < 0:
        raise ValueError(f"resolution must be >= 0; got {res}")
    for base in range(12):
        if res == 0:
            yield (base,)
            continue
        for digits in itertools.product(range(7), repeat=res):
            first_nz = next((d for d in digits if d != 0), None)
            if first_nz == 1:
                continue
            yield (base, *digits)


def is_valid_cell(cell) -> bool:
    """True iff ``cell`` is a valid mu3 cell index.

    A valid cell is a sequence of integers ``(base, d_1, ..., d_N)`` where:
      - ``base ∈ {0, ..., 11}`` (one of 12 icosa vertex pentagons)
      - each ``d_k ∈ {0, ..., 6}``
      - the first nonzero child digit (if any) is not 1
        (digit 1 is the pentagon-deleted direction, so no cell can step
        into it from the base pentagon or from a pentagon-center ancestor)

    Non-sequence inputs, non-integer elements, and empty inputs all return
    False rather than raising.
    """
    try:
        n = len(cell)
    except TypeError:
        return False
    if n < 1:
        return False

    def _is_int(x) -> bool:
        # accept plain ints and numpy integer scalars; reject bool (True/False
        # are ints by subclassing, but that's almost always a mistake here)
        return isinstance(x, (int, np.integer)) and not isinstance(x, bool)

    base = cell[0]
    if not _is_int(base) or not 0 <= int(base) < 12:
        return False

    seen_nonzero = False
    for d in cell[1:]:
        if not _is_int(d):
            return False
        dv = int(d)
        if not 0 <= dv <= 6:
            return False
        if not seen_nonzero and dv != 0:
            if dv == 1:
                return False
            seen_nonzero = True
    return True


def is_pentagon(cell) -> bool:
    """True iff ``cell`` is a pentagon-center cell.

    A pentagon-center cell has all-zero child digits -- its 3D position
    coincides with the base pentagon's icosa vertex. At res 0 every
    cell ``(b,)`` is a pentagon center; at res N the 12 pentagon
    centers are the paths ``(b, 0, 0, ..., 0)``.

    Does not validate the cell -- call :func:`is_valid_cell` separately
    if needed. Returns ``False`` for non-sequence inputs.
    """
    try:
        return all(d == 0 for d in cell[1:])
    except TypeError:
        return False


def cell_resolution(cell) -> int:
    """Resolution of ``cell``: 0 for ``(base,)``, ``N`` for
    ``(base, d_1, ..., d_N)``.

    Does not validate the cell -- call :func:`is_valid_cell` separately
    if needed.
    """
    return len(cell) - 1


def cell_center(cell: Sequence[int]) -> np.ndarray:
    """Unit 3-vector on the sphere for ``cell = (base, d_1, ..., d_N)``."""
    base, digits = cell[0], cell[1:]
    return _project(_eisenstein_center(digits), base)


def _spherical_polygon_area(V: np.ndarray) -> float:
    """Spherical polygon area on the unit sphere (steradians).

    Per-edge Van Oosterom–Strackee with chord identities, fanning
    around the polygon's own (unit-normalized) centroid:

        contribution_i = 2 · atan2(num, den)
        num = P · (A_i × (A_{i+1} − A_i))
        den = (1 + P · A_i) + (1 + P · A_{i+1}) − 0.5 · |A_{i+1} − A_i|²

    Two algebraic identities exploit |A| = |B| = 1 to dodge the
    cancellation traps in the textbook VOS formula:

      • A · B  =  1 − 0.5·|B − A|²       (denominator: replaces
        the 4-way cancellation `1 + P·A + A·B + B·P` with three
        independently-conditioned terms)
      • A × B  =  A × (B − A)            (numerator: cross-product
        through the small chord, avoiding cancellation on short edges)

    Choice of fan reference: the polygon's own unit centroid. With P
    inside the polygon, every (1 + P · A_i) ≈ 2 — bounded away from
    zero. This eliminates the antipodal precision loss that bites a
    fixed-pole reference (when a vertex approaches −P, computing
    `1 + P · V` as `1 + (near −1)` loses 12+ digits). One extra
    normalize per polygon; otherwise pure 3D arithmetic, no
    transcendentals per vertex.

    Stable across all cell sizes through mu3 res ≥ 20 (verified). At
    extreme radii (~1e-9 rad), this formulation actually outperforms
    the H3 lat/lng Cagnoli formula, which loses everything to f64
    underflow there.

    Sign convention: CCW-from-outside polygons containing the centroid
    (always true for convex spherical polygons, including mu3 cells)
    sum to a positive value directly. The +4π fallback catches any
    edge case where the signed sum comes out negative.

    Path to here (see ``todo/2026-04-30-spherical-polygon-area.md``):
      - Attempt 1: per-fan VOS, no chord identities — denominator
        cancellation broke at ε ≲ 8e-8 (res 14+).
      - Attempt 2: per-fan Tuynman mid-vector — same per-fan
        structural floor.
      - Attempt 3: per-edge Tuynman mid-vector with fixed S — the
        unit midpoint (S+X)/|S+X| blew up for X near −S.
      - Attempt 4: lat/lng Cagnoli (H3 port) — worked, but routed
        every vertex through asin/atan2.
      - Attempt 5 (this): per-edge VOS with chord identities and
        polygon-centroid fan reference — purely 3D, no
        transcendentals per vertex, beats Cagnoli at extreme radii.
    """
    # Polygon centroid (unit-normalized) as the fan reference.
    n = len(V)
    sx = sy = sz = 0.0
    for i in range(n):
        sx += float(V[i][0])
        sy += float(V[i][1])
        sz += float(V[i][2])
    pnorm = math.sqrt(sx * sx + sy * sy + sz * sz)
    PX = sx / pnorm
    PY = sy / pnorm
    PZ = sz / pnorm

    sum_val = 0.0
    c = 0.0  # Kahan compensation
    for i in range(n):
        j = (i + 1) % n
        ax, ay, az = float(V[i][0]), float(V[i][1]), float(V[i][2])
        bx, by, bz = float(V[j][0]), float(V[j][1]), float(V[j][2])

        # Chord (B − A): stable for short edges.
        dx = bx - ax
        dy = by - ay
        dz = bz - az

        # num = P · (A × (B − A))
        cx = ay * dz - az * dy
        cy = az * dx - ax * dz
        cz = ax * dy - ay * dx
        num = PX * cx + PY * cy + PZ * cz

        # den = (1 + P·A) + (1 + P·B) − 0.5 · |B − A|²
        pa = PX * ax + PY * ay + PZ * az
        pb = PX * bx + PY * by + PZ * bz
        d2 = dx * dx + dy * dy + dz * dz
        den = (1.0 + pa) + (1.0 + pb) - 0.5 * d2

        contribution = 2.0 * math.atan2(num, den)

        # Kahan compensated summation across edges.
        y = contribution - c
        t = sum_val + y
        c = (t - sum_val) - y
        sum_val = t

    # Centroid is interior to convex polygons → sum is positive
    # directly. Fallback in case of pathological non-convex inputs.
    if sum_val < 0.0:
        y = 4.0 * math.pi - c
        t = sum_val + y
        c = (t - sum_val) - y
        sum_val = t
    return sum_val


def cell_area(cell: Sequence[int]) -> float:
    """Spherical area of the cell, in steradians (unit-sphere).

    Always positive (the H3 cagnoli implementation in
    ``_spherical_polygon_area`` normalizes any negative signed-area sums
    that arise for southern-hemisphere or pole-wrapping cells). Sum
    over every cell at a given resolution equals 4π (the sphere).
    """
    return _spherical_polygon_area(cell_boundary(cell, closed=False))


def cell_boundary(cell: Sequence[int], closed: bool = True) -> np.ndarray:
    """Cell boundary as an (M, 3) array of unit 3-vectors.

    Hex cells return 6 vertices; pentagon-center cells (all-zero child
    digits, including the res-0 cell ``(base,)``) return 5 — stitching
    collapses corners k=3 and k=4 onto the same point. If ``closed``, the
    first vertex is repeated.
    """
    base, digits = cell[0], cell[1:]
    z = _eisenstein_center(digits)
    rot_N = get_rot(cell_resolution(cell))

    seen: set[tuple] = set()
    pts: list[np.ndarray] = []
    for k in range(6):
        corner_z = z + units[k] / (s3 * rot_N)
        p3 = _project(corner_z, base)
        key = tuple(np.round(p3, 12))
        if key in seen:
            continue
        seen.add(key)
        pts.append(p3)

    out = np.stack(pts, axis=0)
    if closed:
        out = np.vstack([out, out[0:1]])
    return out


def _polish_boundary(q: np.ndarray, V: np.ndarray) -> int | None:
    """Spherical point-in-polygon against a precomputed boundary.

    ``V`` is an ``(n, 3)`` CCW ring of unit 3-vectors (viewed from outside
    the sphere). Returns ``None`` if ``q`` is inside, else the index ``k``
    of the boundary edge ``q`` is farthest outside of (the edge from
    ``V[k]`` to ``V[(k+1) % n]``).

    ``V[k] × V[k+1]`` points into the polygon interior by the CCW
    convention, so a strictly negative dot with ``q`` means ``q`` is
    outside that edge.

    This is the primitive; callers in tight loops should hoist
    :func:`cell_boundary` out and call this directly.
    """
    n = len(V)
    worst_dot = 0.0
    worst_k: int | None = None
    for k in range(n):
        edge_normal = np.cross(V[k], V[(k + 1) % n])
        d = float(edge_normal @ q)
        if d < worst_dot:
            worst_dot = d
            worst_k = k
    return worst_k


def _polish_cell_sphere(q: np.ndarray, cell: Sequence[int]) -> int | None:
    """Spherical point-in-polygon for a mu3 cell. Wraps :func:`_polish_boundary`."""
    return _polish_boundary(q, cell_boundary(cell, closed=False))
