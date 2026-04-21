"""Face <-> sphere projections.

A :class:`Projection` maps between points on a single icosahedron face
(planar, centered at the origin) and points on the unit sphere. The
interface is kept minimal so alternative maps — α-slerp, D_3-corrected,
equal-area tweaks — can be dropped in without touching the indexing layer.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Projection(Protocol):
    """Swappable face/sphere projection.

    Implementations take the face-center unit vector ``center`` (and, if
    needed, an orientation — see :class:`Gnomonic`) and expose ``forward``
    and ``inverse`` methods. Inputs and outputs are plain ``numpy`` arrays
    with a trailing axis of length 2 (planar) or 3 (sphere).
    """

    def forward(self, xy: np.ndarray) -> np.ndarray:
        """Planar face coordinates -> unit-sphere points (..., 3)."""
        ...

    def inverse(self, p: np.ndarray) -> np.ndarray:
        """Unit-sphere points -> planar face coordinates (..., 2)."""
        ...


class Gnomonic:
    """Gnomonic projection tangent to the sphere at ``center``.

    Straight lines on the plane map to great-circle arcs on the sphere.
    Area is badly distorted toward face corners — this is a starting
    point, not the final map.
    """

    def __init__(self, center: np.ndarray, up: np.ndarray | None = None) -> None:
        c = np.asarray(center, dtype=float)
        c = c / np.linalg.norm(c)
        if up is None:
            # arbitrary axis not parallel to c
            ref = np.array([0.0, 0.0, 1.0]) if abs(c[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            up = ref - np.dot(ref, c) * c
        u = np.asarray(up, dtype=float)
        u = u - np.dot(u, c) * c
        u = u / np.linalg.norm(u)
        v = np.cross(c, u)
        self.center = c
        self.u = u  # planar x-axis, tangent to sphere at center
        self.v = v  # planar y-axis

    def forward(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        x = xy[..., 0:1]
        y = xy[..., 1:2]
        p = self.center + x * self.u + y * self.v
        return p / np.linalg.norm(p, axis=-1, keepdims=True)

    def inverse(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        # scale each ray so it lies on the tangent plane at center
        denom = p @ self.center
        q = p / denom[..., None]
        x = q @ self.u
        y = q @ self.v
        return np.stack([x, y], axis=-1)
