"""Dodecahedron geometry: 12 faces in H3's icosahedron orientation.

Public interface uses integer face labels in ``0..11``. Internally each face
is built from a symbolic ``(axis, s1, s2)`` tuple over the standard-position
icosahedron (vertices = cyclic permutations of ``(0, +/-1, +/-phi)``):

    axis 0 (x):  ( 0,            s1,           s2 * phi )
    axis 1 (y):  ( s2 * phi,     0,            s1       )
    axis 2 (z):  ( s1,           s2 * phi,    0         )

That standard position is then rotated by ``_R_H3`` (see below) so the 12
vertices land on H3's water-aligned positions, matching H3's icosahedron
orientation. All public quantities (normals, pentagon_corners,
primary_tangents) are in this final H3-oriented frame.

See ``reports/nearest-dodecahedron-face.md`` for the design rationale and a
comparison with alternative labelings.
"""

import math

import numpy as np

from .rotation import rotation_matrix

PHI = (1.0 + math.sqrt(5.0)) / 2.0
NORM = math.sqrt(1.0 + PHI * PHI)

# The 12 dodecahedron faces in canonical order, as (axis, s1, s2) tuples.
_faces = [
    (0, -1, -1),
    (0, -1, +1),
    (0, +1, -1),
    (0, +1, +1),
    (1, -1, -1),
    (1, -1, +1),
    (1, +1, -1),
    (1, +1, +1),
    (2, -1, -1),
    (2, -1, +1),
    (2, +1, -1),
    (2, +1, +1),
]


# Unit outward normals (= dual icosahedron vertex directions), aligned with
# `_faces`. Built first in standard position; rotated into H3's orientation
# below.
normals = [
    np.roll(np.array([0.0, s1, s2*PHI]), axis) / NORM
    for (axis, s1, s2) in _faces
]


# Rotation matching H3's icosahedron orientation (vertices over water) with
# face 0 sent to H3's highest-latitude vertex (Greenland Sea, lat 64.7 N,
# lng 10.5 E), so face 0's pentagon encloses the geographic north pole.
_R_H3 = rotation_matrix(
    source = normals[0],
    target = np.array([0.420152433357666, 0.078145250933923, 0.904082547088666]),
    # H3 alignment, plus one 5-fold tick clockwise around face 0
    # (CW around the outward normal = negative twist).
    twist_rad = -0.028365119330803733 - (2 * math.pi / 5),
)

# Apply R_H3: overwrite normals into H3's orientation.
normals = [_R_H3 @ n for n in normals]


# Edge-adjacent neighbors as indices into `_faces`. Each row holds the 5
# neighbors in CCW order around the face's normal; column 0 is the same-axis
# neighbor (axis, -s1, s2), the one the primary tangent points toward.
neighbors = (
    ( 2,  5, 10,  8,  4),
    ( 3,  6,  8, 10,  7),
    ( 0,  4,  9, 11,  5),
    ( 1,  7, 11,  9,  6),
    ( 6,  9,  2,  0,  8),
    ( 7, 10,  0,  2, 11),
    ( 4,  8,  1,  3,  9),
    ( 5, 11,  3,  1, 10),
    (10,  1,  6,  4,  0),
    (11,  2,  4,  6,  3),
    ( 8,  0,  5,  7,  1),
    ( 9,  3,  7,  5,  2),
)


# Primary tangent at each face: unit vector at `normals[i]` pointing toward
# the primary-direction neighbor `normals[neighbors[i][0]]`, projected to the
# tangent plane. This is Option A from the report's "Primary direction" section.
def _tangent_to(n, m):
    proj = m - (m @ n) * n
    return proj / np.linalg.norm(proj)


primary_tangents = [
    _tangent_to(normals[i], normals[neighbors[i][0]])
    for i in range(12)
]


# Pentagon corners on the unit sphere: a (12, 5, 3) array. Corner ``k`` of
# face ``i`` is the unit-normalized centroid of the three mutually-adjacent
# face normals ``(i, neighbors[i][k], neighbors[i][(k + 1) % 5])`` --
# equivalently, the dodecahedron vertex shared by those three faces. CCW
# ordering inherits from `neighbors`.
def _pentagon_corners(i):
    n = normals[i]
    nbrs = neighbors[i]
    out = np.empty((5, 3))
    for k in range(5):
        c = n + normals[nbrs[k]] + normals[nbrs[(k + 1) % 5]]
        out[k] = c / np.linalg.norm(c)
    return out


pentagon_corners = np.stack([_pentagon_corners(i) for i in range(12)])


# Icosahedron edges as pairs of indices into `normals`, with ``i < j`` to
# dedupe. 30 edges; each connects two adjacent icosahedron vertices.
icosa_edges = tuple(
    (i, j)
    for i in range(12)
    for j in neighbors[i]
    if i < j
)
