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
from .eisenstein import (
    DIGIT_OFFSET,
    UNIT_DIGITS,
    ZETA,
    Eis,
    first_nonzero_digit,
    in_deleted_wedge,
    scaled_corner,
)
from .face_lattice import digit_offset, get_rot, s3
from .projection import AlphaSlerp, Projection, Vec3

_PROJECTION_CLS: type[Projection] = AlphaSlerp


def active_projection_name() -> str:
    """Name of the active projection class. Keys projection-specific
    fitted constants (e.g. ``mu3.index._BOW_COEFFS``) so a projection
    swap fails loudly instead of silently using stale values."""
    return _PROJECTION_CLS.__name__

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


def _project(z: complex, base: int) -> Vec3:
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


def _pick_wedge(p3d: Vec3, base: int) -> int:
    """Digit ``d ∈ {2..6}`` whose 60° wedge contains ``p3d``, from the two
    nearest ``base``-neighbors — no transcendental.

    Every icosahedron base→neighbor edge is the same length, so
    ``V[n]·p3d = A + B·cos(φₙ)`` with ``A, B`` fixed across the 5 neighbors
    and ``φₙ`` the point's azimuth offset from spoke ``n``. That is monotone
    in ``|φₙ|``, so the two largest neighbor dot products are exactly the two
    spokes bracketing the point — i.e. the wedge, with no tangent frame and
    no ``atan2``. (These 5 neighbor dots are a subset of the
    ``argmax(V·p3d)`` pass that picks ``base``; a compiled caller can share
    that one pass — the reference just recomputes them, which is cheap.)

    Recovering the digit inverts :func:`_projection`'s pairing
    ``n_ccw = n[(d-1) % 5]``: wedge ``d`` owns the consecutive spoke pair
    ``(d-2, d-1) = (j, (j+1) % 5)``, so a top-two ``{i0, i1}`` with
    ``i1 = (i0+1) % 5`` gives ``d = i0 + 2``. A non-adjacent top-two only
    arises at the pentagon centre, where every wedge is valid and the
    downstream spherical polish is the authority.
    """
    nd = icosahedron.vertices()[icosahedron.vertex_neighbors()[base]] @ p3d
    order = np.argsort(nd)
    i0, i1 = int(order[-1]), int(order[-2])          # nearest, 2nd-nearest
    j = i0 if (i0 + 1) % 5 == i1 else i1             # CW spoke of the pair
    return j + 2


def _sphere_to_flat(p3d: Vec3, base: int) -> complex:
    """Inverse of :func:`_project`: 3D unit vector inside ``base``'s spherical
    territory → Eisenstein ``z`` in ``base``'s flat frame.

    ``base = argmax(V·p3d)`` fixes the pentagon; :func:`_pick_wedge` fixes
    the wedge from the same dot products; one projection inverse gives the
    barycentric, converted back to ``z`` in that wedge's flat frame. The
    wedge pick is exact — the point lies in ``base``'s Voronoi cell, hence
    inside the picked triangle — so no wedge search is needed.

    The returned ``z`` is in canonical post-stitch form (angle in
    ``[60°, 360°)`` at any non-degenerate point); the deleted wedge
    ``[0°, 60°)`` is unreachable here by construction.
    """
    d = _pick_wedge(p3d, base)
    beta = _projection(base, d).to_bary(p3d)
    return _z_from_wedge_barycentric(beta, d)


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
            if first_nonzero_digit(digits) == 1:
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


def cell_center(cell: Sequence[int]) -> Vec3:
    """Unit 3-vector on the sphere for ``cell = (base, d_1, ..., d_N)``."""
    base, digits = cell[0], cell[1:]
    return _project(_eisenstein_center(digits), base)


def _spherical_polygon_area(V: np.ndarray) -> float:
    """Spherical polygon area on the unit sphere (steradians):
    :func:`_signed_spherical_excess` (the numerics live there),
    normalized to positive.

    Sign convention: CCW-from-outside convex polygons (every mu3
    cell) sum to a positive value directly. The +4π fallback catches
    pathological non-convex inputs.
    """
    sum_val = _signed_spherical_excess(V)
    if sum_val < 0.0:
        sum_val += 4.0 * math.pi
    return sum_val


def _signed_spherical_excess(V: np.ndarray) -> float:
    """Signed spherical excess of the ring ``V``: positive for
    CCW-from-outside, negative for CW — the orientation bit
    ``mu3.polygon`` classifies holes with. No +4π normalization
    (that is :func:`_spherical_polygon_area`).

    Fan-triangulate around V[0] and sum the spherical excess of each
    fan triangle (V[0], V[i], V[i+1]) using a fully chord-based
    Van Oosterom–Strackee formula:

        E(x, y, z) = 2 · atan2(num, den)
        num = x · [(y − x) × (z − x)]
        den = 4 − ½ · (|x − y|² + |y − z|² + |z − x|²)

    Algebraically identical to the textbook VOS triangle formula
    `tan(E/2) = x·(y×z) / (1 + x·y + y·z + z·x)` (verify by expanding
    the chord identities), but every intermediate is now expressed in
    sums-of-squares of chord vectors:

      • 1 + x·V    =  ½·|x + V|²        (one direction of each dot)
      • 1 − A·B    =  ½·|B − A|²        (the other direction)
      • x × y      =  (y − x) × (z − x) + …   (cross through chords)

    The denominator follows directly from substituting
    `1 − A·B = ½·|B − A|²` for each of the three pairs, giving
    `4 − ½·(|x−y|² + |y−z|² + |z−x|²)`. Every term is a sum of squared
    real numbers — no `1 + (-1 + ε)` cancellation can arise regardless
    of the relative positions of x, y, z.

    The numerator `x · [(y − x) × (z − x)]` evaluates to the same
    `x · (y × z)` as the textbook form, but the cross product is
    constructed from two chord vectors of length O(arc) — its
    components are naturally O(arc²) without any cancellation in the
    dot product with x (which would otherwise reduce a fan-cross of
    O(arc) magnitude down to O(arc²)).

    Choice of fan apex: V[0]. For any small polygon (vertices all
    within ~90° of each other), V[0] is close to every other vertex,
    so each fan triangle has area at polygon_area / (n−2) scale — no
    outer-summation cancellation tower as resolution increases. The
    polygon centroid is an alternative (safer for hemisphere-spanning
    polygons), but for mu3's sub-90° cells the two are bit-identical
    and V[0] saves the centroid normalize.

    Stable across all cell sizes through mu3 res ≥ 20 (verified). At
    extreme radii (~1e-9 rad), this formulation outperforms the H3
    lat/lng Cagnoli formula, which loses everything to f64 underflow
    there.

    See ``~/work/sphere_area/notes/2026-04-30-spherical-polygon-area.qmd``
    for the full derivation and the journey through prior attempts.
    """
    n = len(V)
    x0, x1, x2 = float(V[0][0]), float(V[0][1]), float(V[0][2])

    sum_val = 0.0
    c = 0.0  # Kahan compensation
    for i in range(1, n - 1):
        y0, y1, y2 = float(V[i][0]),     float(V[i][1]),     float(V[i][2])
        z0, z1, z2 = float(V[i + 1][0]), float(V[i + 1][1]), float(V[i + 1][2])

        # Three chords forming the triangle (x = V[0], y, z).
        yx0, yx1, yx2 = y0 - x0, y1 - x1, y2 - x2  # y - x
        zx0, zx1, zx2 = z0 - x0, z1 - x1, z2 - x2  # z - x
        zy0, zy1, zy2 = z0 - y0, z1 - y1, z2 - y2  # z - y

        # num = x · [(y - x) × (z - x)] — cross of two chords gives
        # an O(arc²) vector directly; dotting with x extracts the
        # signed-volume scalar without further cancellation.
        cx0 = yx1 * zx2 - yx2 * zx1
        cx1 = yx2 * zx0 - yx0 * zx2
        cx2 = yx0 * zx1 - yx1 * zx0
        num = x0 * cx0 + x1 * cx1 + x2 * cx2

        # den = 4 - ½·(|x-y|² + |y-z|² + |z-x|²) — chord-based, every
        # intermediate is a sum of squared reals.
        d_xy2 = yx0 * yx0 + yx1 * yx1 + yx2 * yx2
        d_yz2 = zy0 * zy0 + zy1 * zy1 + zy2 * zy2
        d_zx2 = zx0 * zx0 + zx1 * zx1 + zx2 * zx2
        den = 4.0 - 0.5 * (d_xy2 + d_yz2 + d_zx2)

        contribution = 2.0 * math.atan2(num, den)

        # Kahan compensated summation across fan triangles.
        cy = contribution - c
        t = sum_val + cy
        c = (t - sum_val) - cy
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


def _project_corner(corner: Eis, base: int, res: int) -> Vec3:
    """Exact S3-scaled corner -> unit 3-vector: the single flat-to-3D
    corner path (``cell_boundary`` rows and ``mu3.vertex`` positions
    both go through here; floats start at this boundary)."""
    return _project(corner.to_complex() / (s3 * get_rot(res)), base)


class _CellFrame:
    """Snapshot of one cell's intermediate algebra: validate once,
    compute once the state every corner/boundary product shares.
    Attributes are set at construction and never after — treat frames
    as throwaway snapshots (``frame.cell`` is the identity; don't
    carry a frame across a projection swap).
    """

    __slots__ = ('cell', 'base', 'res', 'sc3')

    def __init__(self, cell):
        cell_t = tuple(int(x) for x in cell)
        if not is_valid_cell(cell_t):
            raise ValueError(f'_CellFrame: invalid cell {cell_t}')
        self.cell = cell_t
        self.base = cell_t[0]
        self.res = cell_resolution(cell_t)
        # scaled_corner (the corner formula's single home) with the
        # center term hoisted: corner(d) = sc3 + DIGIT_OFFSET[d].
        self.sc3 = scaled_corner(cell_t[1:], 0)

    def corner(self, d: int) -> Eis:
        """Exact corner at digit ``d``, S3-scaled res-N frame."""
        return self.sc3 + DIGIT_OFFSET[d]

    def corner_vec3(self, d: int) -> Vec3:
        """Corner ``d`` on the sphere, via :func:`_project_corner`
        (stitch-aware; the pentagon ``d=1`` alias lands on the same 3D
        point as ``d=6``)."""
        return _project_corner(self.corner(d), self.base, self.res)

    def boundary(self, closed: bool = True) -> np.ndarray:
        """See :func:`cell_boundary` (its implementation)."""
        seen: set = set()
        pts: list[np.ndarray] = []
        for d in UNIT_DIGITS:
            corner = self.corner(d)
            key = (ZETA * corner if in_deleted_wedge(corner, self.res)
                   else corner)
            if key in seen:
                continue
            seen.add(key)
            pts.append(self.corner_vec3(d))

        out = np.stack(pts, axis=0)
        if closed:
            out = np.vstack([out, out[0:1]])
        return out


def cell_boundary(cell: Sequence[int], closed: bool = True) -> np.ndarray:
    """Cell boundary as an (M, 3) array of unit 3-vectors.

    Hex cells return 6 vertices; pentagon-center cells (all-zero child
    digits, including the res-0 cell ``(base,)``) return 5 — stitching
    collapses corner k=0 (in the deleted wedge) onto k=1. Corner dedup
    is exact: the key is the S3-scaled corner lattice point, stitched
    when it develops into the deleted wedge (see ``mu3.vertex``). If
    ``closed``, the first vertex is repeated.
    """
    return _CellFrame(cell).boundary(closed=closed)


def _polish_boundary(q: Vec3, V: np.ndarray) -> int | None:
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


def _polish_cell_sphere(q: Vec3, cell: Sequence[int]) -> int | None:
    """Spherical point-in-polygon for a mu3 cell. Wraps :func:`_polish_boundary`."""
    return _polish_boundary(q, cell_boundary(cell, closed=False))
