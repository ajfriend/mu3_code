"""Cell indexing.

At res 0 the cell id is just the icosa vertex index (0..11): the base
pentagon centered at that vertex owns the spherical Voronoi region around
it. Higher-resolution indexing layers on the per-pentagon Eisenstein
lattice (see ``mu3.face_lattice`` and ``mu3.cell``).
"""

import numpy as np

from . import icosahedron


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

    The base pentagon for a point on the unit sphere is the icosa vertex
    closest to the point: the Voronoi cells of icosa vertices on the
    sphere are precisely the dodecahedron faces. ``argmax(V @ p)`` gives
    that vertex directly.

    Polish is retained for parity with the previous behavior (and for
    higher-resolution use); at res 0 it is a no-op when the argmax is
    correct.
    """
    V = icosahedron.vertices()
    neighbors = icosahedron.vertex_neighbors()

    shape = p.shape[:-1]
    flat = p.reshape(-1, 3)
    n = flat.shape[0]

    candidate = np.argmax(flat @ V.T, axis=1).astype(np.int64)

    final = candidate.copy()
    for k in range(n):
        c = candidate[k]
        pool = np.concatenate([[c], neighbors[c]])
        final[k] = pool[int(np.argmax(V[pool] @ flat[k]))]

    return final.reshape(shape), candidate.reshape(shape)
