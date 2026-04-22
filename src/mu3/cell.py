"""Forward indexing: cell index -> sphere center and boundary.

A cell is a tuple ``(base, d_1, d_2, ..., d_N)`` where:
  - ``base`` is an icosa vertex index (0..11)
  - ``d_1..d_N`` are child digits in {0..6}
  - ``N`` = ``len(cell) - 1`` = the cell's resolution

At res 0 a cell is a 1-tuple ``(base,)``. The first nonzero child digit
cannot be 1 (pentagon-deleted direction).

Pipeline (same for pentagon and hex cells, centers and boundary corners):

1. Accumulate the Eisenstein position ``z`` from the digit sequence.
2. If ``arg(z) ∈ [180°, 240°)`` — the "deleted" wedge — rotate by +60°.
   This folds the lattice's fictitious d=1 sector onto d=5's real wedge.
3. Classify the post-stitch angle into one of five 60° wedges, each owned
   by one icosa face-digit (d=2, 3, 5, 4, 6).
4. Apply that face's similarity map (a complex scale+rotation pinned by
   the two non-pentagon corners of the face) to get face-2D coordinates.
5. Gnomonic-forward from the face's center to the unit sphere.

Pentagon-center cells fall out of the same pipeline: corner k=3 sits at
Eisenstein angle 210°, gets rotated +60° to 270°, and coincides with corner
k=4 — so the 6-corner hex formula naturally produces a 5-gon.
"""

from __future__ import annotations

import cmath
import itertools
import math
from functools import lru_cache
from typing import Iterator, Sequence

import numpy as np

from . import icosahedron
from .face_lattice import get_rot, h3_digit_offset, r_face, s3, units
from .projection import AlphaSlerp

# +60° rotation in the pentagon-Eisenstein plane (stitching for the deleted wedge).
_ROT60 = cmath.exp(1j * math.pi / 3)

# Deleted wedge in Eisenstein (triangle convention): [180°, 240°). Points here
# get rotated +60° into d=5's (post-stitch) wedge [240°, 300°).
_DELETED_LO = 180.0
_DELETED_HI = 240.0

# CCW digit cycle, matching icosahedron.pentagon_face_table's assignment.
_CCW_CYCLE = (2, 3, 5, 4, 6)

# For each face-digit d, which digit-ray sits at the CCW (upper) boundary of
# d's Eisenstein wedge. (This is the digit whose neighbor vertex is shared
# between f_d and its CCW-next face in the fan around V[p].)
_UPPER_DIGIT = {2: 6, 3: 2, 5: 3, 4: 5, 6: 4}

# For each face-digit d, the two digit-rays bordering d's (post-stitch)
# Eisenstein wedge and the wedge's angular endpoints:
#   (cw_ray_digit, ccw_ray_digit, theta_cw_deg, theta_ccw_deg)
# d=4 is the stretched face; after stitching its wedge is [240°, 300°) and
# its CW ray at 240° stands in for the d=3 ray (that's where the stitched
# half of the wedge came from).
_WEDGE_ENDPOINTS = {
    2: (4, 6, 0.0,   60.0),
    3: (6, 2, 60.0,  120.0),
    5: (2, 3, 120.0, 180.0),
    4: (3, 5, 240.0, 300.0),
    6: (5, 4, 300.0, 360.0),
}


def _angle_deg(z: complex) -> float:
    return math.degrees(math.atan2(z.imag, z.real)) % 360.0


def _stitch(z: complex) -> complex:
    """If z sits in the deleted wedge, rotate it +60° onto d=5's wedge."""
    if z == 0j:
        return z
    if _DELETED_LO <= _angle_deg(z) < _DELETED_HI:
        return z * _ROT60
    return z


def _classify_stitched(z: complex) -> int:
    """Face-digit d whose Eisenstein wedge contains a POST-STITCH point z."""
    a = _angle_deg(z)
    if a < 60.0:
        return 2
    if a < 120.0:
        return 3
    if a < 180.0:
        return 5
    if a < 300.0:
        return 4  # only reachable post-stitch (the original [180, 240) is gone)
    return 6


def _neighbor_vertex_of_digit(p: int, d: int) -> int:
    """Icosa vertex index that pentagon p's d-ray points to.

    Derived from the shared-non-p vertex between f_{d_curr} and its CCW-next
    face, where d_curr is the digit whose face has d at its upper boundary.
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
        assert len(shared) == 1
        return shared.pop()
    raise ValueError(f"no face has upper-boundary digit {d}")


def _face_corner_2d(face: int, vertex: int) -> complex:
    """Position of icosa-vertex ``vertex`` in ``face``'s face-2D frame."""
    V = icosahedron.vertices()
    frames = icosahedron.face_frames()
    center, u, v = frames[face]
    p3 = V[vertex]
    q = p3 / np.dot(p3, center)
    return complex(np.dot(q, u), np.dot(q, v))


@lru_cache(maxsize=None)
def _similarity_maps(p: int) -> dict[int, tuple[complex, complex]]:
    """Per-pentagon similarity maps. Returns {face_digit d: (vb, A)} such that
    a POST-STITCH Eisenstein point z in face d's wedge maps to face-2D as
    ``vb + A * z``. A is a pure complex scale+rotation — the stretched face
    d=4 becomes a normal 60° wedge after stitching.
    """
    pft = icosahedron.pentagon_face_table()
    out: dict[int, tuple[complex, complex]] = {}
    for d, (d_cw, d_ccw, theta_a_deg, theta_b_deg) in _WEDGE_ENDPOINTS.items():
        face = int(pft[p, d - 2])
        vb = _face_corner_2d(face, p)
        n_cw = _neighbor_vertex_of_digit(p, d_cw)
        n_ccw = _neighbor_vertex_of_digit(p, d_ccw)
        v_a = _face_corner_2d(face, n_cw)
        v_b = _face_corner_2d(face, n_ccw)
        e_a = cmath.exp(1j * math.radians(theta_a_deg))
        e_b = cmath.exp(1j * math.radians(theta_b_deg))
        A = (v_a - vb) / e_a
        assert abs(A * e_b - (v_b - vb)) < 1e-9, (
            f"wedge d={d} at p={p} is not a pure similarity"
        )
        out[d] = (vb, A)
    return out


@lru_cache(maxsize=None)
def _alpha_slerp_for_face(face: int) -> AlphaSlerp:
    """Per-face AlphaSlerp projector, instantiated with the face's three
    icosa-vertex unit 3-vectors in CCW order (as given by the face table)."""
    V = icosahedron.vertices()
    F = icosahedron.faces()
    v0, v1, v2 = V[F[face, 0]], V[F[face, 1]], V[F[face, 2]]
    return AlphaSlerp(v0, v1, v2)


_SQRT3 = math.sqrt(3.0)


def _face_2d_to_barycentric(xy: complex, face: int) -> np.ndarray:
    """Barycentric coordinates on the face triangle for a face-2D point.

    In each face's face-2D frame the three corners sit at magnitude
    ``r_face`` at angles 0°, 120°, 240° (face_i_vertex at 0°, the other
    two at ±120°). That gives an explicit closed-form inverse, no linear
    solve needed. The output order matches face corner order
    ``(F[face, 0], F[face, 1], F[face, 2])``.
    """
    x, y = xy.real, xy.imag
    b0 = (2.0 * x / r_face + 1.0) / 3.0
    rest = (1.0 - b0) / 2.0
    off = y / (r_face * _SQRT3)
    b1 = rest + off
    b2 = rest - off
    return np.array([b0, b1, b2])


def _project(z: complex, base: int) -> np.ndarray:
    """Eisenstein → stitch → per-face similarity → barycentric → α-slerp → sphere."""
    if z == 0j:
        return icosahedron.vertices()[base].copy()
    z_s = _stitch(z)
    d = _classify_stitched(z_s)
    vb, A = _similarity_maps(base)[d]
    xy = vb + A * z_s
    face = int(icosahedron.pentagon_face_table()[base, d - 2])
    beta = _face_2d_to_barycentric(xy, face)
    return _alpha_slerp_for_face(face).forward_barycentric(beta)


def _eisenstein_center(digits: Sequence[int]) -> complex:
    """Pentagon-Eisenstein position ``z`` for a child-digit sequence."""
    z = 0j
    for k, d in enumerate(digits, start=1):
        if d == 0:
            continue
        z += h3_digit_offset[d] / get_rot(k)
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


def cell_center(cell: Sequence[int]) -> np.ndarray:
    """Unit 3-vector on the sphere for ``cell = (base, d_1, ..., d_N)``."""
    base, digits = cell[0], cell[1:]
    return _project(_eisenstein_center(digits), base)


def _spherical_polygon_area(V: np.ndarray) -> float:
    """Signed spherical polygon area on the unit sphere (steradians).

    Van Oosterom–Strackee formula summed over a triangle fan from V[0].
    CCW rings (viewed from outside the sphere) give positive area.
    """
    n = len(V)
    total = 0.0
    v0 = V[0]
    for i in range(1, n - 1):
        a, b = V[i], V[i + 1]
        num = float(np.dot(v0, np.cross(a, b)))
        den = 1.0 + float(np.dot(v0, a)) + float(np.dot(a, b)) + float(np.dot(b, v0))
        total += 2.0 * math.atan2(num, den)
    return float(total)


def cell_area(cell: Sequence[int]) -> float:
    """Spherical area of the cell, in steradians (unit-sphere).

    Signed: CCW cells — the library's convention — are positive. Sum over
    every cell at a given resolution equals 4π (the sphere). For absolute
    area, take ``abs(cell_area(cell))``.
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
    rot_N = get_rot(len(digits))

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
