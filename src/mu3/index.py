"""Cell indexing.

At res 0 the cell id is just the icosa vertex index (0..11): the base
pentagon centered at that vertex owns the spherical Voronoi region around
it. The goal is eventually a cell index scheme that is kink-free across
icosahedron edges and around pentagon cells. The Eisenstein-integer face
arithmetic explored in ``2026-04-02_eisint`` is the starting point for the
per-face coordinate system at higher resolutions.
"""

from __future__ import annotations

import numpy as np

from . import icosahedron
from .projection import Gnomonic


def latlng_to_vec(lat_deg, lng_deg) -> np.ndarray:
    """Batched lat/lng (degrees) to unit 3-vector. Trailing axis is size 3."""
    lat = np.deg2rad(np.asarray(lat_deg, dtype=float))
    lng = np.deg2rad(np.asarray(lng_deg, dtype=float))
    c = np.cos(lat)
    return np.stack([c * np.cos(lng), c * np.sin(lng), np.sin(lat)], axis=-1)


def latlng_to_cell(lat_deg, lng_deg):
    """Res-0 base pentagon cell id (0..11) for each input point.

    Scalar inputs return a Python int; array inputs return an int64 ndarray
    of the broadcast shape.
    """
    p = latlng_to_vec(lat_deg, lng_deg)
    cell, _ = _latlng_to_cell_detailed(p)
    if cell.shape == ():
        return int(cell)
    return cell


def _latlng_to_cell_detailed(p: np.ndarray):
    """Run the full pipeline, returning (final_cell, pre_polish_candidate).

    Used by tests to verify that polish never swaps at res 0.
    """
    V = icosahedron.vertices()
    F = icosahedron.faces()
    centers = icosahedron.face_centers()
    frames = icosahedron.face_frames()
    neighbors = icosahedron.vertex_neighbors()

    shape = p.shape[:-1]
    flat = p.reshape(-1, 3)
    n = flat.shape[0]

    face_idx = np.argmax(flat @ centers.T, axis=1)  # (n,)

    candidate = np.empty(n, dtype=np.int64)
    for k in range(n):
        f = face_idx[k]
        center, u, _ = frames[f]
        gn = Gnomonic(center=center, up=u)
        xy = gn.inverse(flat[k])
        corners_xy = gn.inverse(V[F[f]])
        d2 = ((corners_xy - xy) ** 2).sum(axis=-1)
        candidate[k] = F[f, int(np.argmin(d2))]

    # Polish: among {candidate} ∪ neighbors[candidate], take argmax(V @ p).
    # At res 0 this is a no-op when the 2D step is correct.
    final = candidate.copy()
    for k in range(n):
        c = candidate[k]
        pool = np.concatenate([[c], neighbors[c]])
        dots = V[pool] @ flat[k]
        final[k] = pool[int(np.argmax(dots))]

    return final.reshape(shape), candidate.reshape(shape)
