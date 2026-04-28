"""Icosahedron base geometry: 12 vertices, 30 edges, 20 faces.

Vertices are placed on the unit sphere with one vertex at the north pole,
one at the south, and two antipodal pentagonal rings at latitude
:math:`\\pm \\arctan(1/2)`.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

_LAT = np.arctan(0.5)  # latitude of the two pentagonal rings


# Results of these builders are cached: the icosahedron is fixed, and
# callers in the projection pipeline hit these on every corner. The returned
# numpy arrays are treated as read-only (do not mutate in place).
@lru_cache(maxsize=None)
def vertices() -> np.ndarray:
    """Return the 12 unit-sphere icosahedron vertices as a (12, 3) array.

    Row order: north pole, 5 upper-ring vertices (longitudes 0, 72, ...),
    5 lower-ring vertices (longitudes 36, 108, ...), south pole.
    """
    lons_up = np.deg2rad(np.arange(5) * 72.0)
    lons_dn = np.deg2rad(np.arange(5) * 72.0 + 36.0)

    c, s = np.cos(_LAT), np.sin(_LAT)
    up = np.stack([c * np.cos(lons_up), c * np.sin(lons_up), np.full(5, s)], axis=1)
    dn = np.stack([c * np.cos(lons_dn), c * np.sin(lons_dn), np.full(5, -s)], axis=1)

    north = np.array([[0.0, 0.0, 1.0]])
    south = np.array([[0.0, 0.0, -1.0]])
    return np.vstack([north, up, dn, south])


@lru_cache(maxsize=None)
def faces() -> np.ndarray:
    """Return the 20 triangular faces as a (20, 3) array of vertex indices.

    Vertex indexing matches :func:`vertices`: 0 = north, 1..5 = upper ring,
    6..10 = lower ring, 11 = south.
    """
    faces = []
    # upper cap: 5 triangles around the north pole
    for i in range(5):
        faces.append((0, 1 + i, 1 + (i + 1) % 5))
    # equatorial belt: 10 triangles alternating up/down
    for i in range(5):
        u0, u1 = 1 + i, 1 + (i + 1) % 5
        d0, d1 = 6 + i, 6 + (i + 1) % 5
        faces.append((u0, d0, u1))  # point-down triangle
        faces.append((u1, d0, d1))  # point-up triangle
    # lower cap
    for i in range(5):
        faces.append((11, 6 + (i + 1) % 5, 6 + i))
    return np.array(faces, dtype=np.int64)


@lru_cache(maxsize=None)
def face_centers() -> np.ndarray:
    """Return (20, 3) unit-sphere face centers (centroids, renormalized)."""
    V = vertices()
    F = faces()
    c = V[F].mean(axis=1)
    return c / np.linalg.norm(c, axis=1, keepdims=True)


@lru_cache(maxsize=None)
def face_i_vertex() -> np.ndarray:
    """Return (20,) array: for each face, which of its vertex indices defines the i-axis.

    Placeholder: uses the face's first CCW vertex. Will be replaced with
    H3's per-face choice so that orientations match H3 exactly.
    """
    return faces()[:, 0].copy()


@lru_cache(maxsize=None)
def face_frames() -> np.ndarray:
    """Return (20, 3, 3): per-face orthonormal 3D frame.

    For face ``f``, rows are ``(center, u, v)``:
    ``u`` is the unit tangent at ``center`` pointing toward the i-axis vertex,
    ``v = cross(center, u)``.
    """
    V = vertices()
    C = face_centers()
    I = face_i_vertex()
    target = V[I]  # (20, 3)
    u_raw = target - (target * C).sum(axis=1, keepdims=True) * C
    u = u_raw / np.linalg.norm(u_raw, axis=1, keepdims=True)
    v = np.cross(C, u)
    return np.stack([C, u, v], axis=1)


@lru_cache(maxsize=None)
def vertex_neighbors() -> np.ndarray:
    """Return (12, 5) array: the 5 icosa-vertex neighbors of each vertex.

    Two vertices are adjacent iff they co-appear in some face. Neighbor
    order is unspecified (sorted ascending by index).
    """
    F = faces()
    adj: list[set[int]] = [set() for _ in range(12)]
    for a, b, c in F:
        adj[a].update((b, c))
        adj[b].update((a, c))
        adj[c].update((a, b))
    out = np.array([sorted(s) for s in adj], dtype=np.int64)
    assert out.shape == (12, 5), f"expected 5 neighbors per vertex, got {out.shape}"
    return out


# CCW digit sequence starting from digit 2 (skipping the deleted digit 1).
# Digits 1..6 march strictly CCW around the parent hex; with d=1 deleted in
# pentagons, the surviving CCW order is the sequential (2, 3, 4, 5, 6).
_PENTAGON_DIGIT_CCW = (2, 3, 4, 5, 6)


@lru_cache(maxsize=None)
def pentagon_face_table() -> np.ndarray:
    """Return (12, 5) int64: ``pentagon_face_table[p, d - 2]`` is the icosa
    face that digit ``d`` ∈ {2, 3, 4, 5, 6} points into from base pentagon ``p``.

    Derivation (no external tables):

    1. Enumerate the 5 faces incident to pentagon ``p``.
    2. Sort them CCW around ``V[p]`` in the tangent plane at ``V[p]``,
       looking from outside the sphere.
    3. Assign the smallest-index incident face to digit 2.
    4. Following CCW around the pentagon, assign digits in the sequential
       cycle ``2, 3, 4, 5, 6`` — d=1 is the deleted direction.
    """
    V = vertices()
    F = faces()
    C = face_centers()

    out = np.empty((12, 5), dtype=np.int64)
    for p in range(12):
        incident = np.array(sorted(f for f in range(20) if p in F[f]))
        assert len(incident) == 5, f"pentagon {p} has {len(incident)} incident faces"

        v = V[p]
        ref = np.array([0.0, 0.0, 1.0]) if abs(v[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = ref - np.dot(ref, v) * v
        u = u / np.linalg.norm(u)
        w = np.cross(v, u)

        angles = np.empty(5)
        for i, f in enumerate(incident):
            c_tangent = C[f] - np.dot(C[f], v) * v
            angles[i] = np.arctan2(np.dot(c_tangent, w), np.dot(c_tangent, u))

        # Smallest-index face is already at incident[0] (sorted ascending).
        # Rotate by its angle so it ends up first in the CCW ordering.
        rel_angles = (angles - angles[0]) % (2 * np.pi)
        ccw_order = np.argsort(rel_angles)
        faces_ccw = incident[ccw_order]

        for pos, d in enumerate(_PENTAGON_DIGIT_CCW):
            out[p, d - 2] = faces_ccw[pos]

    return out


def v_base_face2d(base: int, face: int) -> complex:
    """2D position of icosa vertex ``base`` in face ``face``'s local frame.

    Returned as a complex number (x + iy) in face-2D coordinates (gnomonic
    from face center). ``base`` must be a corner of ``face``.
    """
    V = vertices()
    frames = face_frames()
    center, u, v = frames[face]
    p = V[base]
    # Gnomonic inverse: p -> 2D tangent-plane coords at center.
    denom = p @ center
    q = p / denom
    return complex(q @ u, q @ v)


