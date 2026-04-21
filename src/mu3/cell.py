"""Forward indexing: cell index -> sphere center and boundary.

A cell index is ``(base, digits)`` where ``base`` is an icosa vertex index
(0..11) and ``digits`` is a tuple in {0..6} whose length is the resolution.
The first nonzero digit cannot be 1 (pentagon-deleted direction).

The center is computed in pentagon-Eisenstein (a 6-fold-symmetric abstract
2D lattice centered at the base pentagon, with digit 1's direction declared
forbidden). The angular sector of the final Eisenstein position determines
which icosa face the point lives on; embedding into that face's 2D frame
and gnomonic-forward-projecting gives the sphere position.

Boundary corners are computed the same way, per-corner: each of the 6 (hex)
corners is placed in pentagon-Eisenstein at a hex-vertex offset from the
center, then independently dispatched through its own sector/face.
Pentagon-center cells (all-zero digits) use a separate direct 3D construction
since their 5-fold-symmetric vertex structure doesn't match the hex-vertex
formula.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from . import icosahedron
from .face_lattice import (
    get_rot,
    h3_digit_offset,
    omega,
    pentagon_skipped_digit,
    r_face,
    s3,
    units,
)
from .projection import Gnomonic


def _eisenstein_center(digits: Sequence[int]) -> complex:
    """Pentagon-Eisenstein position ``z`` for the given digit sequence."""
    z = 0j
    for k, d in enumerate(digits, start=1):
        if d == 0:
            continue
        z += h3_digit_offset[d] / get_rot(k)
    return z


def _face_digit_for_z(z: complex) -> int:
    """Return the digit ``d`` ∈ {2..6} whose sector contains ``arg(z)``.

    The deleted wedge (digit 1, centered at 240°) is "absorbed" into its
    two neighbors — half (210°-240°) goes to digit 3, half (240°-270°) to
    digit 5. A point in the deleted wedge "continues around" to whichever
    adjacent non-deleted face it's closer to.
    """
    angle = math.degrees(math.atan2(z.imag, z.real)) % 360.0
    if angle < 30.0 or angle >= 330.0:
        return 4  # i-axis direction
    elif angle < 90.0:
        return 6
    elif angle < 150.0:
        return 2
    elif angle < 240.0:
        return 3  # digit 3's sector expanded through deleted-wedge midline
    else:  # 240 <= angle < 330
        return 5  # digit 5's sector expanded through deleted-wedge midline


def _project_through_face(z: complex, base: int) -> np.ndarray:
    """Dispatch a pentagon-Eisenstein point ``z`` through its angular sector's
    face and gnomonic-forward to the sphere.
    """
    d = _face_digit_for_z(z)
    pft = icosahedron.pentagon_face_table()
    face = int(pft[base, d - 2])
    vb = icosahedron.v_base_face2d(base, face)
    A = icosahedron.pentagon_embed_factors()[base, d - 2]
    xy = vb + A * z

    frame = icosahedron.face_frames()[face]
    gn = Gnomonic(center=frame[0], up=frame[1])
    return gn.forward(np.array([xy.real, xy.imag]))


def cell_center(base: int, digits: Sequence[int]) -> np.ndarray:
    """Unit 3-vector on the sphere for the cell ``(base, digits)``."""
    z = _eisenstein_center(digits)
    if z == 0:
        return icosahedron.vertices()[base].copy()
    return _project_through_face(z, base)


def cell_boundary(
    base: int, digits: Sequence[int], closed: bool = True
) -> np.ndarray:
    """Cell boundary as an (M, 3) array of unit 3-vectors.

    Returns 6 vertices for hex cells, 5 for pentagon cells (all-zero digits).
    Each vertex is projected through its own angular sector's face, so cells
    straddling face boundaries have vertices on different faces (intended;
    this is the core of the pentagon-Eisenstein definition).

    If ``closed``, the first vertex is repeated at the end.
    """
    z = _eisenstein_center(digits)
    N = len(digits)

    if z == 0:
        return _pentagon_center_boundary(base, N, closed)

    corners = []
    for k in range(6):
        corner_offset = units[k] / (get_rot(N) * s3)
        vertex_z = z + corner_offset
        corners.append(_project_through_face(vertex_z, base))

    out = np.stack(corners, axis=0)
    if closed:
        out = np.vstack([out, out[0:1]])
    return out


def _pentagon_center_boundary(base: int, N: int, closed: bool) -> np.ndarray:
    """Boundary of a pentagon-center cell (all-zero digit path) at res N.

    The 5 vertices are at the pentagon's 5 incident face centers, each
    shrunk toward ``V[base]`` by a factor ``1/sqrt(7)^N`` in the pentagon's
    tangent plane (gnomonic-from-V[base]).
    """
    V = icosahedron.vertices()
    v = V[base]
    centers = icosahedron.face_centers()
    pft = icosahedron.pentagon_face_table()[base]

    gn = Gnomonic(center=v)
    scale = 1.0 / (math.sqrt(7) ** N)

    ccw_digit_order = (2, 3, 5, 4, 6)
    result = []
    for d in ccw_digit_order:
        f = int(pft[d - 2])
        xy_cf = gn.inverse(centers[f])
        result.append(gn.forward(xy_cf * scale))

    out = np.stack(result, axis=0)
    if closed:
        out = np.vstack([out, out[0:1]])
    return out
