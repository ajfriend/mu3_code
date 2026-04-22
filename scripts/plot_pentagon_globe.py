"""Globe plot from the 2D stitched Eisenstein lattice.

Uses the same stitching rule as scripts/plot_pentagon_lattice_stitched.py —
any point in the deleted wedge [180°, 240°) gets rotated +60°, putting it
in [240°, 300°) — and then projects each (stitched) point through a single
per-face similarity map.

After stitching, every point sits in one of five 60° wedges, each of which
maps by a pure similarity (60°-apex in Eisenstein → 60°-apex equilateral
face in face-2D). No 2×2 shear needed.
"""

from __future__ import annotations

import cmath
import itertools
import json
import math
from pathlib import Path

import numpy as np

from mu3 import icosahedron
from mu3.face_lattice import get_rot, h3_digit_offset, omega, s3, units


ROT60 = cmath.exp(1j * math.pi / 3)
DELETED_LO = 180.0
DELETED_HI = 240.0


def first_nonzero_digit(digits):
    for d in digits:
        if d != 0:
            return d
    return None


def eisenstein_center(digits) -> complex:
    z = 0j
    for k, d in enumerate(digits, start=1):
        if d == 0:
            continue
        z += h3_digit_offset[d] / get_rot(k)
    return z


def eisenstein_corners(z_c: complex, res: int) -> list[complex]:
    rot_N = get_rot(res)
    return [z_c + units[k] / (s3 * rot_N) for k in range(6)]


def _angle_deg(z: complex) -> float:
    return math.degrees(math.atan2(z.imag, z.real)) % 360.0


def stitch(z: complex) -> complex:
    if z == 0j:
        return z
    if DELETED_LO <= _angle_deg(z) < DELETED_HI:
        return z * ROT60
    return z


# --- Per-face similarity maps ---
#
# After stitching, each of 5 digit-labelled cells fills a 60° Eisenstein
# wedge. The wedge-to-face assignment uses the CCW-upper-digit convention
# (same as scripts/plot_pentagon_lattice_stitched.py's wedge_digit):
#
#   [0°, 60°)    → face-owned-by-digit-6  (CCW upper = d=6 at 60°)
#   [60°, 120°)  → face-owned-by-digit-2
#   [120°, 180°) → face-owned-by-digit-3
#   [240°, 300°) → face-owned-by-digit-5  (after stitching absorbs [180°, 240°))
#   [300°, 360°) → face-owned-by-digit-4
#
# For wedge labelled d (d ∈ {2, 3, 4, 5, 6}), its CW boundary is at digit
# d_cw's ray (at unit magnitude in Eisenstein) and its CCW boundary is at
# d's ray. Those two endpoints map to the two non-pentagon vertices of the
# icosa face owned by that digit.

CCW_CYCLE = (2, 3, 5, 4, 6)

# For each face-digit d (the icosa face pft[p, d-2]), the Eisenstein wedge
# it projects from, and the two digit-rays bordering that wedge.
#   (cw_ray_digit, ccw_ray_digit, theta_cw_deg, theta_ccw_deg)
# Face d=4 is the stretched one; after stitching it sits on wedge [240°, 300°).
# The CW boundary at 240° represents V[n_d=3] (stitched from Eis 180°).
WEDGE_ENDPOINTS = {
    2: (4, 6, 0.0,   60.0),
    3: (6, 2, 60.0,  120.0),
    5: (2, 3, 120.0, 180.0),
    4: (3, 5, 240.0, 300.0),  # post-stitch, cw ray represents d=3 via +60° rotation
    6: (5, 4, 300.0, 360.0),
}


def _neighbor_vertex_of_digit(p: int, d: int) -> int:
    """Icosa vertex index V[p]'s d-ray points to.

    Derived by: for CCW pair (d_i, d_{i+1}) in CCW_CYCLE, the vertex shared
    between f_{d_i} and f_{d_{i+1}} is the neighbor on the ray at the CCW
    upper-boundary of f_{d_i}'s Eisenstein wedge.
    """
    F = icosahedron.faces()
    pft = icosahedron.pentagon_face_table()
    # upper digit of each face's Eis wedge:
    UPPER = {2: 6, 3: 2, 5: 3, 4: 5, 6: 4}
    for i, d_curr in enumerate(CCW_CYCLE):
        if UPPER[d_curr] != d:
            continue
        d_next = CCW_CYCLE[(i + 1) % 5]
        f_curr = int(pft[p, d_curr - 2])
        f_next = int(pft[p, d_next - 2])
        shared = (set(int(x) for x in F[f_curr]) & set(int(x) for x in F[f_next])) - {p}
        assert len(shared) == 1
        return shared.pop()
    raise ValueError(f"no face has upper-boundary digit {d}")


def _face_corner_2d(face: int, vertex: int) -> complex:
    """2D position of ``vertex`` (an icosa vertex index) in face's face-2D frame."""
    V = icosahedron.vertices()
    frames = icosahedron.face_frames()
    center, u, v = frames[face]
    p3 = V[vertex]
    q = p3 / np.dot(p3, center)
    return complex(np.dot(q, u), np.dot(q, v))


def build_similarity_maps(p: int) -> dict[int, tuple[complex, complex]]:
    """For pentagon p, return {digit d: (vb, A)} such that a point ``z`` in
    Eisenstein wedge d (post-stitching) maps to ``vb + A * z`` in face d's
    face-2D frame."""
    pft = icosahedron.pentagon_face_table()
    out = {}
    for d, (d_cw, d_ccw, theta_a_deg, theta_b_deg) in WEDGE_ENDPOINTS.items():
        face = int(pft[p, d - 2])
        vb = _face_corner_2d(face, p)
        n_cw = _neighbor_vertex_of_digit(p, d_cw)
        n_ccw = _neighbor_vertex_of_digit(p, d_ccw)
        v_a = _face_corner_2d(face, n_cw)
        v_b = _face_corner_2d(face, n_ccw)
        e_a = cmath.exp(1j * math.radians(theta_a_deg))
        e_b = cmath.exp(1j * math.radians(theta_b_deg))
        # Similarity A such that A * e_a = v_a - vb AND A * e_b = v_b - vb.
        A = (v_a - vb) / e_a
        # sanity: the two constraints must agree for a pure similarity
        assert abs(A * e_b - (v_b - vb)) < 1e-9, \
            f"wedge d={d} at p={p} is not a pure similarity"
        out[d] = (vb, A)
    return out


def classify_stitched(z: complex) -> int:
    """Face-digit d whose Eisenstein wedge contains a POST-STITCH point z."""
    ang = _angle_deg(z)
    if ang < 60.0:
        return 2   # wedge [0°, 60°) → face d=2
    if ang < 120.0:
        return 3   # [60°, 120°) → face d=3
    if ang < 180.0:
        return 5   # [120°, 180°) → face d=5
    if ang < 300.0:
        return 4   # [240°, 300°) (post-stitch) → face d=4 (stretched)
    return 6       # [300°, 360°) → face d=6


def project_eis_point(z: complex, p: int, maps) -> np.ndarray:
    """Eisenstein → stitch → per-face similarity → gnomonic → sphere."""
    if z == 0j:
        return icosahedron.vertices()[p].copy()
    z_s = stitch(z)
    d = classify_stitched(z_s)
    vb, A = maps[d]
    xy = vb + A * z_s
    face = int(icosahedron.pentagon_face_table()[p, d - 2])
    frame = icosahedron.face_frames()[face]
    center, u_ax, v_ax = frame[0], frame[1], frame[2]
    p3 = center + xy.real * u_ax + xy.imag * v_ax
    return p3 / np.linalg.norm(p3)


def cell_ring(p: int, digits: tuple, maps) -> list:
    """Boundary of cell (p, digits). Uses the Eisenstein stitching+similarity
    pipeline for every corner — pentagon cells too. For pentagon cells the
    corner at Eis k=3 (angle 210°) stitches to Eis 270°, coinciding with
    corner k=4, so the resulting polygon has 5 distinct vertices naturally.
    """
    z_c = eisenstein_center(digits)
    res = len(digits)
    corners = eisenstein_corners(z_c, res)
    ring = [unit_to_lnglat(project_eis_point(c, p, maps)) for c in corners]
    ring.append(ring[0])
    return ring


def unit_to_lnglat(v: np.ndarray) -> list[float]:
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    lng = float(np.degrees(np.arctan2(y, x)))
    lat = float(np.degrees(np.arcsin(max(-1.0, min(1.0, z)))))
    return [lng, lat]


def icosahedron_edges() -> list:
    V = icosahedron.vertices()
    F = icosahedron.faces()
    edges: set[tuple[int, int]] = set()
    for face in F:
        a, b, c = int(face[0]), int(face[1]), int(face[2])
        for u, vv in ((a, b), (b, c), (c, a)):
            edges.add(tuple(sorted((u, vv))))
    return [[unit_to_lnglat(V[u]), unit_to_lnglat(V[vv])] for u, vv in edges]


def main() -> None:
    cells_by_res: dict[str, list] = {}
    for r in (0, 1, 2, 3):
        polys = []
        for p in range(12):
            maps = build_similarity_maps(p)
            if r == 0:
                polys.append([cell_ring(p, (), maps)])
                continue
            for digits in itertools.product(range(7), repeat=r):
                if first_nonzero_digit(digits) == 1:
                    continue
                polys.append([cell_ring(p, digits, maps)])
        cells_by_res[str(r)] = polys
        print(f"res {r}: {len(polys)} cells")

    icosa_lines = icosahedron_edges()
    data = {"cells_by_res": cells_by_res, "icosahedron_edges": icosa_lines}
    data_json = json.dumps(data)

    out = Path(__file__).resolve().parent.parent / "figures" / "mu3_globe_v2.html"
    out.parent.mkdir(exist_ok=True)

    HTML = ("""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>mu3 cells v2</title>
<style>
  body { font-family: -apple-system, sans-serif; margin: 20px; color: #222; background: #fff; }
  h1 { font-weight: 400; margin-bottom: 4px; } .sub { color: #555; font-size: 13px; margin-bottom: 20px; }
  .grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; max-width: 1200px; }
  .panel { border: 1px solid #ddd; padding: 12px; border-radius: 4px; }
  .panel h2 { margin: 0 0 8px 0; font-size: 14px; font-weight: 600; color: #334; }
  .panel .count { color: #888; font-size: 12px; margin: 0 0 6px 0; }
  .globe { width: 100%; aspect-ratio: 1; cursor: grab; } .globe:active { cursor: grabbing; }
</style></head><body>
<h1>mu3 cells on the sphere (v2 — stitched Eisenstein + per-face similarity)</h1>
<div class="sub">Drag any globe to rotate all; double-click to reset.</div>
<div class="grid">
  <div class="panel"><h2>Resolution 0</h2><div class="count" id="c-0"></div><div id="g-0" class="globe"></div></div>
  <div class="panel"><h2>Resolution 1</h2><div class="count" id="c-1"></div><div id="g-1" class="globe"></div></div>
  <div class="panel"><h2>Resolution 2</h2><div class="count" id="c-2"></div><div id="g-2" class="globe"></div></div>
  <div class="panel"><h2>Resolution 3</h2><div class="count" id="c-3"></div><div id="g-3" class="globe"></div></div>
</div>
<script type="module">
  import * as Plot from 'https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6/+esm';
  import * as d3 from 'https://cdn.jsdelivr.net/npm/d3@7/+esm';
  const DATA = __DATA_PLACEHOLDER__;
  for (const key of Object.keys(DATA.cells_by_res)) {
    DATA.cells_by_res[key] = DATA.cells_by_res[key].map(
      polygon => polygon.map(ring => ring.slice().reverse())
    );
  }
  for (const res of Object.keys(DATA.cells_by_res)) {
    document.getElementById('c-' + res).textContent = DATA.cells_by_res[res].length + ' cells';
  }
  const icosaFeature = { type: 'Feature', geometry: { type: 'MultiLineString', coordinates: DATA.icosahedron_edges } };
  const sharedRotate = [20, -30, 0];
  const renderAll = [];
  function renderAllGlobes() { for (const r of renderAll) r(); }
  function setupGlobe(res, containerId) {
    const container = document.getElementById(containerId);
    const gridFeature = { type: 'Feature', geometry: { type: 'MultiPolygon', coordinates: DATA.cells_by_res[res] } };
    function getSize() { return container.offsetWidth || 400; }
    function render() {
      const size = getSize();
      container.innerHTML = '';
      const plot = Plot.plot({
        width: size, height: size,
        projection: { type: 'orthographic', rotate: sharedRotate.slice(), inset: 2 },
        marks: [
          Plot.graticule({ strokeOpacity: 0.1 }),
          Plot.geo(icosaFeature, { stroke: '#888', strokeWidth: 1.5, strokeOpacity: 0.5 }),
          Plot.geo(gridFeature, { stroke: '#c33', strokeWidth: 0.5, fill: '#e55', fillOpacity: 0.2 }),
          Plot.sphere({ strokeWidth: 2 })
        ]
      });
      container.appendChild(plot);
      const drag = d3.drag()
        .on('start', (e) => { drag.sx = e.x; drag.sy = e.y; drag.sr = sharedRotate.slice(); })
        .on('drag', (e) => {
          const dx = e.x - drag.sx, dy = e.y - drag.sy, k = 0.25;
          sharedRotate[0] = drag.sr[0] + dx * k;
          sharedRotate[1] = drag.sr[1] - dy * k;
          sharedRotate[2] = 0;
          renderAllGlobes();
        });
      d3.select(plot).call(drag);
      d3.select(plot).on('dblclick', () => {
        sharedRotate[0] = 20; sharedRotate[1] = -30; sharedRotate[2] = 0;
        renderAllGlobes();
      });
    }
    renderAll.push(render);
    render();
    window.addEventListener('resize', () => render());
  }
  setupGlobe('0', 'g-0'); setupGlobe('1', 'g-1'); setupGlobe('2', 'g-2'); setupGlobe('3', 'g-3');
</script></body></html>
""").replace("__DATA_PLACEHOLDER__", data_json)

    out.write_text(HTML)
    print(f"\nwrote {out}")
    print(f"size: {out.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
