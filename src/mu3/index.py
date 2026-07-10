"""Cell indexing.

At res 0 the cell id is just the icosa vertex index (0..11): the base
pentagon centered at that vertex owns the spherical Voronoi region around
it. Higher-resolution indexing layers on the per-pentagon Eisenstein
lattice (see ``mu3.face_lattice`` and ``mu3.cell``).
"""

import numpy as np

from . import icosahedron
from .cell import (
    _eisenstein_center,
    _polish_boundary,
    _project,
    _sphere_to_flat,
    active_projection_name,
    cell_boundary,
    is_pentagon,
)
from .cross_pentagon import EISENSTEIN_UNITS
from .eisenstein import S3, ZETA_INV, from_complex
from .face_lattice import get_rot
from .neighbor import resolve_position, step
from .projection import Vec3


def latlng_to_vec3(lat_deg, lng_deg) -> np.ndarray:
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
    p = latlng_to_vec3(lat_deg, lng_deg)
    if p.ndim > 1:
        raise NotImplementedError(
            "batched latlng_to_cell not yet implemented; "
            "use _latlng_to_cell_detailed for batched res-0 lookups"
        )
    return vec3_to_cell(p, res)


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


def vec3_to_cell(p3d: Vec3, res: int = 0) -> tuple:
    """Scalar 3D unit vector -> canonical cell tuple at the given resolution.

    Wrapper around :func:`vec3_to_cell_polished`. This is the function you
    want by default. ``latlng_to_cell`` is a thin wrapper over this.
    """
    return vec3_to_cell_polished(p3d, res)


def vec3_to_cell_raw(p3d: Vec3, res: int) -> tuple:
    """Forward pipeline only -- argmax base, exact snap, exact resolve.
    No spherical polish.

    The returned cell is *geometrically close* to ``p3d`` but may not
    contain it: by the single-edge invariant it is the containing cell
    or the neighbor across exactly one violated edge — never anything
    further. Use :func:`vec3_to_cell_polished` (or :func:`vec3_to_cell`)
    for the contained-cell guarantee.

    Steps:

    1. argmax over icosa vertices = base pentagon.
    2. ``_sphere_to_flat`` inverse-projects p3d to a flat z in base's
       frame; scale to res-N lattice units.
    3. ``neighbor.resolve_position`` snaps to the nearest lattice
       point exactly and resolves to the canonical cell via the
       holonomy cocycle, with the unsnapped query as the seam-side
       witness. Phantom twins (leading-d=1 snaps -- the deleted-wedge
       Gosper wiggle) are decided by the witness's
       side of the cut, not by geometry.
    """
    return _vec3_to_cell_detailed(p3d, res)[0]


def _vec3_to_cell_detailed(p3d: Vec3, res: int):
    """The raw pipeline with its intermediates exposed: returns
    ``(cell, base, w)`` — ``w`` is the pulled-back query in ``base``'s
    frame at ``res`` (``None`` at res 0). Single home of the forward
    pipeline; :func:`vec3_to_cell_raw` is its docstring/spec."""
    V = icosahedron.vertices()
    base = int(np.argmax(V @ p3d))
    if res == 0:
        return (base,), base, None
    w = _sphere_to_flat(p3d, base) * get_rot(res)
    return resolve_position(base, w, res), base, w


def vec3_to_cell_polished(p3d: Vec3, res: int) -> tuple:
    """:func:`vec3_to_cell_raw` + the single-edge spherical polish.

    THE SINGLE-EDGE INVARIANT: the raw cell is the containing cell or
    the ring-1 neighbor across exactly one violated edge — never
    anything further. Flat and spherical corners agree exactly (both
    project the same lattice points), so the two descriptions disagree
    only inside per-edge bow lenses pinned at shared corners, each
    straddled by the raw cell and that one edge-neighbor. This is an
    invariant every admissible projection must uphold, NOT a search
    radius: its margin is measured and asserted per projection by
    ``tests/test_one_hop_contract.py``, and when a projection erodes it
    the projection (or its fitted band) is what gets fixed — checking
    2-3 hops is never the answer, since a bow escaping the
    edge-neighbor would mean flat adjacency no longer models spherical
    adjacency, voiding the indexing contract wholesale. At res 0 polish
    is a no-op (raw is already the spherical Voronoi region of an icosa
    vertex).

    The polish itself is the banded fast path (:func:`_polish_banded`);
    :func:`_polish` is the full-boundary reference it falls back to.
    """
    cell, base, w = _vec3_to_cell_detailed(p3d, res)
    if res == 0:
        return cell
    return _polish_banded(p3d, base, cell, w, res)


def _polish(p3d: Vec3, cell: tuple) -> tuple:
    """If ``p3d`` is inside ``cell``'s spherical boundary, returns ``cell``
    unchanged. Otherwise returns the neighbor across the violated edge.

    Scanning the cell's own edges is how the ONE violated edge is
    located; by the single-edge invariant that edge determines the
    unique answer — a single ``step`` in its walk direction
    ``D = k + 1`` (hex) or ``k + 2`` (pentagon). No other neighbor is
    ever a candidate.
    """
    boundary = cell_boundary(cell, closed=False)
    k = _polish_boundary(p3d, boundary)
    if k is None:
        return cell
    return step(cell, k + (2 if is_pentagon(cell) else 1))[0]


# --- banded polish ---------------------------------------------------
#
# The polish can only change the answer where the pulled-back geodesic
# edge and the flat hex edge disagree, and that mismatch (the "bow")
# vanishes at the cell corners — corners are projections of exact
# lattice points, identical in both descriptions — and peaks mid-edge.
# So a parabolic envelope in the edge parameter t bounds it:
#
#     bow(t) <= 4 * c_res * t * (1 - t)      (lattice units)
#
# Only edges whose flat chord distance is under this allowance need the
# spherical side test (2 corner projections + a cross product); with no
# such edge the raw cell is final with no projection work at all.
#
# THE FUNCTIONAL FORM IS PROJECTION-INDEPENDENT — it uses only exact
# shared corners and smooth edges, which every Projection here
# provides — but the coefficients c_res are NOT. They are fitted per
# projection by ``scripts/measure_polish_band.py`` (measured envelope
# x2 margin) and pinned with headroom against a fresh measurement by
# ``tests/test_one_hop_contract.py``. Swapping projections without
# refitting fails loudly: a missing table entry raises here, a stale
# one trips the test by name. The key is the class NAME only — a
# parameter change within the same class (e.g. retuned AlphaSlerp
# alphas) also needs a refit, and only the headroom test catches that.

_BOW_COEFFS: dict[str, dict[int, float]] = {
    'AlphaSlerp': {
        1: 0.0792,
        2: 0.1156,
        3: 0.0984,
        4: 0.1258,
        5: 0.1160,
        6: 0.1271,
        7: 0.1193,
        8: 0.1273,
        9: 0.1198,
        10: 0.1273,
        11: 0.1199,
        12: 0.1273,
    },
}

# Flat hex around a snapped lattice point, lattice units, derived from
# the exact tables: outward edge normals are the six units; corners sit
# at unit/s3 (the ``scaled_corner`` convention), angle 60°·j + 30°,
# |corner| = 1/√3. Edge j (normal ``_EDGE_NORMALS[j]``) runs corner
# j−1 -> corner j (CCW), so |chord|² = 1/3.
_EDGE_NORMALS = EISENSTEIN_UNITS
_S3_C = S3.to_complex()
_HEX_CORNERS = tuple(u / _S3_C for u in EISENSTEIN_UNITS)

_ROT_M60 = ZETA_INV.to_complex()

# Hexes whose closed hull comes this near a deleted-wedge boundary ray
# take the full polish: hex circumradius 1/√3 plus slop for the bow and
# for phantom corners sitting exactly on the cut.
_STITCH_CLEARANCE = 0.9


def _bow_coeff(res: int) -> float:
    """Envelope coefficient c_res for the active projection.

    The coefficient does NOT decay with res: away from the stitch the
    worst edges are those whose chords cross chart seams (spokes
    between adjacent face triangles), where the projections meet only
    C⁰ and the kink makes the relative bow scale-free. It stabilizes
    instead, alternating slightly with the odd/even resolution
    rotation — hence max of the two top entries beyond the table.
    ``test_bow_table_tail_stable`` pins that the tail has stabilized.
    """
    name = active_projection_name()
    table = _BOW_COEFFS.get(name)
    if table is None:
        raise KeyError(
            f'no polish band coefficients for projection {name!r}; '
            f'run scripts/measure_polish_band.py and add its output to '
            f'mu3.index._BOW_COEFFS'
        )
    if res in table:
        return table[res]
    top = max(table)
    return max(table[top], table[top - 1])


def _near_stitch(zp: complex) -> bool:
    """True if the hex around pentagon-frame position ``zp`` (lattice
    units) comes within :data:`_STITCH_CLEARANCE` of a deleted-wedge
    boundary ray
    (angles 0° and 60°). Across the cut the flat hex-edge model of the
    cell boundary is wrong — the pulled-back boundary jumps by the +60°
    stitch, pentagon corner pairs collapse, and phantom corners on the
    cut have twin reps at different 3D points — so those cells take the
    full polish."""
    for z in (zp, zp * _ROT_M60):
        d = abs(z.imag) if z.real > 0.0 else abs(z)
        if d < _STITCH_CLEARANCE:
            return True
    return False


def _cell_near_stitch(cell) -> bool:
    """:func:`_near_stitch` for a cell given by its index: is the hex
    around this cell's center in the guard region? The runtime guard in
    :func:`_polish_banded` and the fit/pin measurements
    (``scripts/measure_polish_band.py``, the ``test_bow_*`` tests) must
    agree on this predicate — the coefficient table only covers cells
    outside it — so both call this one."""
    res = len(cell) - 1
    # center * |rot| = pentagon-frame center at lattice scale
    return _near_stitch(_eisenstein_center(cell[1:]) * abs(get_rot(res)))


def _polish_banded(p3d: Vec3, base: int, cell: tuple, w: complex,
                   res: int) -> tuple:
    """Banded polish: side-test only the edges the point could actually
    have crossed. ``w`` is the pulled-back query in ``base``'s frame at
    ``res`` (the same value ``resolve_position`` snapped).

    Per edge, the flat chord distance ``d`` and edge parameter ``t``
    are a few multiplies on the snap remainder; edges with ``d`` at or
    above the bow allowance ``4·c·t·(1−t)`` (see module comment) cannot
    be violated. Typically no edge qualifies and the raw cell is final.
    A qualifying edge gets the exact great-circle side test against its
    two projected corners; only an actual crossing (or a stitch-region
    cell, see :func:`_near_stitch`) pays for :func:`_polish`.
    """
    rot = get_rot(res)
    z0c = from_complex(w).to_complex()
    # Pentagon-frame direction at lattice scale: undo rot's rotation
    # only (its magnitude √7^res is the lattice scale itself).
    if _near_stitch(z0c * (rot / abs(rot)).conjugate()):
        return _polish(p3d, cell)
    r = w - z0c
    coeff = _bow_coeff(res)
    for j in range(6):
        d = 0.5 - (r * _EDGE_NORMALS[j].conjugate()).real
        a = _HEX_CORNERS[j - 1]
        b = _HEX_CORNERS[j]
        t = ((r - a) * (b - a).conjugate()).real * 3.0
        if not 0.0 < t < 1.0:
            continue
        if d >= 4.0 * coeff * t * (1.0 - t) + 1e-4:
            continue
        v1 = _project((z0c + a) / rot, base)
        v2 = _project((z0c + b) / rot, base)
        # (v1 × v2) · p3d, written out — the inward-pointing edge
        # normal's side test on plain floats.
        if ((v1[1] * v2[2] - v1[2] * v2[1]) * p3d[0]
                + (v1[2] * v2[0] - v1[0] * v2[2]) * p3d[1]
                + (v1[0] * v2[1] - v1[1] * v2[0]) * p3d[2]) < 0.0:
            return _polish(p3d, cell)
    return cell
