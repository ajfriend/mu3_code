"""D3 globe view: ring-1 from (0, 2) at res 1.

Source hex, expected ring-1 cells, and the 3D points of all 6 walks D=1..6
(with sphere-to-cell handoff). Five of six walks land at correct ring-1
cells; D=6 stitches back to source (self-loop)."""

import cmath
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _globe_template import icosa_edge_coords, render_globe_page, unit_to_lnglat

from mu3 import cell_ring1, dodec
from mu3.cell import (
    _classify_stitched, _eisenstein_center, _stitch, _wedge_barycentric,
    cell_resolution,
)
from mu3.cross_pentagon import z_to_cell
from mu3 import icosahedron
from mu3.face_lattice import digit_offset, get_rot, s3, units


SOURCE_CELL = (0, 2)


def project_gnomonic(z, base):
    if abs(z) < 1e-12:
        return icosahedron.vertices()[base].copy()
    z_s = _stitch(z)
    d = _classify_stitched(z_s)
    beta = _wedge_barycentric(z_s, d)
    V = icosahedron.vertices()
    n = icosahedron.vertex_neighbors()[base]
    n_cw = int(n[d - 2]); n_ccw = int(n[(d - 1) % 5])
    p = beta[0] * V[base] + beta[1] * V[n_cw] + beta[2] * V[n_ccw]
    return p / np.linalg.norm(p)


def _wedge_z_from_barycentric(beta, d):
    b_p, b_cw, b_ccw = float(beta[0]), float(beta[1]), float(beta[2])
    z_aligned = complex(b_cw + 0.5 * b_ccw, b_ccw * math.sqrt(3) / 2)
    return z_aligned * cmath.exp(1j * math.radians((d - 1) * 60.0))


def sphere_to_cell(p3d, res):
    V = icosahedron.vertices()
    base = int(np.argmax(V @ p3d))
    n = icosahedron.vertex_neighbors()[base]
    best_d, best_beta, best_min = -1, None, -2.0
    for d in range(2, 7):
        n_cw = int(n[d - 2]); n_ccw = int(n[(d - 1) % 5])
        M = np.column_stack([V[base], V[n_cw], V[n_ccw]])
        beta = np.linalg.solve(M, p3d)
        s = beta.sum()
        if abs(s) < 1e-12:
            continue
        beta = beta / s
        m = float(beta.min())
        if m > best_min:
            best_min = m; best_d = d; best_beta = beta
    z = _wedge_z_from_barycentric(best_beta, best_d)
    return z_to_cell(base, z, res)


def cell_center_g(cell):
    base, digits = cell[0], cell[1:]
    return project_gnomonic(_eisenstein_center(digits), base)


def cell_boundary_g(cell):
    base, digits = cell[0], cell[1:]
    z_C = _eisenstein_center(digits)
    rot_N = get_rot(cell_resolution(cell))
    pts = []
    for k in range(6):
        corner_z = z_C + units[k] / (s3 * rot_N)
        pts.append(project_gnomonic(corner_z, base))
    pts.append(pts[0])
    return np.stack(pts)


def main():
    base = SOURCE_CELL[0]
    res = cell_resolution(SOURCE_CELL)
    z_C = _eisenstein_center(SOURCE_CELL[1:])
    rot_N = get_rot(res)

    src_ring = [unit_to_lnglat(v) for v in cell_boundary_g(SOURCE_CELL)]
    src_center_3d = cell_center_g(SOURCE_CELL)
    src_center_lnglat = unit_to_lnglat(src_center_3d)

    expected = list(cell_ring1(SOURCE_CELL))
    expected_polys = []
    expected_labels = []
    for nb in expected:
        ring = [unit_to_lnglat(v) for v in cell_boundary_g(nb)]
        expected_polys.append([ring])
        expected_labels.append({
            "pos": unit_to_lnglat(cell_center_g(nb)),
            "text": str(nb),
        })

    walk_segments = []
    walk_markers_correct = []
    walk_markers_self = []
    for D in range(1, 7):
        z_n = z_C + digit_offset[D] / rot_N
        p3d = project_gnomonic(z_n, base)
        try:
            nb = sphere_to_cell(p3d, res)
        except Exception:
            nb = None
        ll = unit_to_lnglat(p3d)
        walk_segments.append([src_center_lnglat, ll])
        tag = f"D={D} → {nb}" if nb is not None else f"D={D} ERROR"
        if nb == SOURCE_CELL:
            walk_markers_self.append({"pos": ll, "text": tag + " (self)"})
        else:
            walk_markers_correct.append({"pos": ll, "text": tag})

    icosa_lines = icosa_edge_coords()

    panels = [{
        "id": "g-ring1",
        "layers": [
            {"type": "lines", "coords": icosa_lines, "style": {
                "stroke": "#888", "strokeWidth": 1.0, "strokeOpacity": 0.4,
            }},
            {"type": "polygons", "coords": expected_polys, "style": {
                "stroke": "#0a8a3a", "strokeWidth": 1.5,
                "fill": "#0a8a3a", "fillOpacity": 0.12,
            }},
            {"type": "polygons", "coords": [[src_ring]], "style": {
                "stroke": "#c33", "strokeWidth": 2.2,
                "fill": "#c33", "fillOpacity": 0.40,
            }},
            {"type": "arrows", "segments": walk_segments, "style": {
                "stroke": "#36c", "strokeWidth": 1.6, "strokeOpacity": 0.85,
            }},
            {"type": "labels", "items": walk_markers_correct, "style": {
                "fill": "#0a4a8a", "stroke": "white", "strokeWidth": 3,
                "fontSize": 12, "fontWeight": 700,
            }},
            {"type": "labels", "items": walk_markers_self, "style": {
                "fill": "#888", "stroke": "white", "strokeWidth": 3,
                "fontSize": 12, "fontWeight": 700,
            }},
            {"type": "labels", "items": expected_labels, "style": {
                "fill": "#0a8a3a", "stroke": "white", "strokeWidth": 3,
                "fontSize": 10, "fontWeight": 500,
            }},
        ],
    }]

    info_html = f"""<aside class="info">
  <h1>Ring-1 from {SOURCE_CELL}</h1>
  <p>Walk → gnomonic project → sphere_to_cell handoff. 5 of 6 walks land
     at correct ring-1 neighbors (blue labels). D=6 walks into the deleted
     wedge of (0,) and the +60° stitch sends z_n back to source's center
     — a self-loop (gray). The missed neighbor (0, 6) is not reachable
     by this walk pipeline; it would require atlas-as-d=6 or twin-rotation.</p>
  <p>Drag, scroll, double-click. Earth landmasses for orientation.</p>
</aside>"""

    out = Path(__file__).resolve().parent.parent / "figures" / "ring1_failing_globe.html"
    render_globe_page(
        title=f"mu3 ring-1 from {SOURCE_CELL}",
        info_html=info_html,
        panels=panels,
        output_path=out,
        layout="single",
    )


if __name__ == "__main__":
    main()
