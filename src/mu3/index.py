"""Cell indexing.

At res 0 the cell id is just the icosa vertex index (0..11): the base
pentagon centered at that vertex owns the spherical Voronoi region around
it. Higher-resolution indexing layers on the per-pentagon Eisenstein
lattice (see ``mu3.face_lattice`` and ``mu3.cell``).
"""

import numpy as np

from . import icosahedron
from .cell import (
    _polish_boundary,
    _sphere_to_flat,
    cell_boundary,
    cell_center,
)
from .face_lattice import get_rot, rotate_digit_ccw, round_ei
from .neighbor import _has_leading_zero_d1, _step_to_cell, cell_ring1


def latlng_to_vec(lat_deg, lng_deg) -> np.ndarray:
    """Batched lat/lng (degrees) to unit 3-vector. Trailing axis is size 3."""
    lat = np.deg2rad(np.asarray(lat_deg, dtype=float))
    lng = np.deg2rad(np.asarray(lng_deg, dtype=float))
    c = np.cos(lat)
    return np.stack([c * np.cos(lng), c * np.sin(lng), np.sin(lat)], axis=-1)


def latlng_to_cell(lat_deg, lng_deg, res: int = 0) -> tuple[int, ...]:
    """Cell at the given resolution containing this lat/lng, as a tuple
    ``(base, d_1, ..., d_res)``. Scalar inputs only.

    For batched res-0 lookups (the previous ndarray-returning shortcut),
    use :func:`_latlng_to_cell_detailed` directly.
    """
    p = latlng_to_vec(lat_deg, lng_deg)
    if p.ndim > 1:
        raise NotImplementedError(
            "batched latlng_to_cell not yet implemented; "
            "use _latlng_to_cell_detailed for batched res-0 lookups"
        )
    return vec_to_cell(p, res)


def _latlng_to_cell_detailed(p: np.ndarray):
    """Run the full pipeline, returning (final_cell, pre_polish_candidate).

    The base pentagon for a point on the unit sphere is the icosa vertex
    closest to the point: the Voronoi cells of icosa vertices on the
    sphere are precisely the dodecahedron faces. ``argmax(V · p)`` gives
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


def vec_to_cell(p3d: np.ndarray, res: int = 0) -> tuple:
    """Scalar 3D unit vector -> canonical cell tuple at the given resolution.

    Wrapper around :func:`vec_to_cell_polished`. This is the function you
    want by default. ``latlng_to_cell`` is a thin wrapper over this.
    """
    return vec_to_cell_polished(p3d, res)


def vec_to_cell_raw(p3d: np.ndarray, res: int) -> tuple:
    """Forward pipeline only -- argmax base, snap to lattice, twin-disambiguate.
    No spherical polish.

    The returned cell is *geometrically close* to ``p3d`` but may not
    contain it: it can be off by at most one ring-1 hop. Use
    :func:`vec_to_cell_polished` (or :func:`vec_to_cell`) for the
    contained-cell guarantee.

    Steps:

    1. argmax over icosa vertices = base pentagon.
    2. ``_sphere_to_flat`` inverse-projects p3d to a flat z in base's frame.
    3. Snap z to the nearest lattice point at resolution ``res`` and run
       ``_step_to_cell`` to extract the digit string.
    4. If the result has leading-d=1 (z_C lies in base's deleted wedge --
       valid at higher res via the Gosper wiggle, see Section 4 of
       reports/surface-mediated-atlas.md), pick the rotation twin
       (CCW or CW) geographically closer to ``p3d``.
    """
    V = icosahedron.vertices()
    base = int(np.argmax(V @ p3d))
    if res == 0:
        return (base,)
    z = _sphere_to_flat(p3d, base)
    rot_N = get_rot(res)
    z_snapped = round_ei(z * rot_N) / rot_N
    candidate = _step_to_cell(z_snapped, base, res)
    if _has_leading_zero_d1(candidate):
        ccw = (candidate[0],
               *(rotate_digit_ccw(d, 1) for d in candidate[1:]))
        cw = (candidate[0],
              *(rotate_digit_ccw(d, 5) for d in candidate[1:]))
        ccw_3d = cell_center(ccw)
        cw_3d = cell_center(cw)
        candidate = ccw if np.linalg.norm(ccw_3d - p3d) < np.linalg.norm(cw_3d - p3d) else cw
    return candidate


def vec_to_cell_polished(p3d: np.ndarray, res: int) -> tuple:
    """:func:`vec_to_cell_raw` + single-hop spherical polish.

    The 1-ring around the raw candidate is sufficient buffer: forward-
    projection error is bounded by less than half a cell-edge length, so
    the candidate is always either correct or shares an edge with the
    correct cell. At res 0 polish is a no-op (raw is already the
    spherical Voronoi region of an icosa vertex).
    """
    candidate = vec_to_cell_raw(p3d, res)
    if res == 0:
        return candidate
    return _polish(p3d, candidate)


def _polish(p3d: np.ndarray, cell: tuple) -> tuple:
    """If ``p3d`` is inside ``cell``'s spherical boundary, returns ``cell``
    unchanged. Otherwise returns the ring-1 neighbor across the violated
    edge.

    Edge index ``k`` and ring-1 index align directly: walk-direction
    ``D = k + 1`` (hex) or ``k + 2`` (pentagon), and ``cell_ring1``
    returns the ring indexed by ``D - 1`` (hex) or ``D - 2`` (pentagon),
    so both cases collapse to ``cell_ring1(cell)[k]``.
    """
    boundary = cell_boundary(cell, closed=False)
    k = _polish_boundary(p3d, boundary)
    if k is None:
        return cell
    return cell_ring1(cell)[k]
