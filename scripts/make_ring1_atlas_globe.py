"""Globe view: source cell (0, 6, 0) at res 2 plus the 3D points the
gnomonic-via-icosa-triangle (atlas) projection lands at for each of the 6
walks D=1..6.

ONE projection everywhere: gnomonic-via-icosa-triangle (linear-combo +
normalize on the spherical triangle for each post-stitch wedge). Cell
boundaries, cell centers, and walk endpoints all use the same mapping, so
geometry is internally consistent.
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _globe_template import icosa_edge_coords, render_globe_page, unit_to_lnglat

from mu3 import cell_ring1, dodec
from mu3.cell import (
    _classify_stitched,
    _eisenstein_center,
    _stitch,
    _wedge_barycentric,
    cell_resolution,
)
from mu3 import icosahedron
from mu3.face_lattice import digit_offset, get_rot, s3, units


SOURCE_CELL = (0, 6, 0)


def project_gnomonic(z, base):
    """Single consistent gnomonic-via-icosa-triangle projection. Mirrors
    mu3.cell._project but uses linear combination + normalize instead of
    AlphaSlerp's cubic."""
    if abs(z) < 1e-12:
        return icosahedron.vertices()[base].copy()
    z_s = _stitch(z)
    d = _classify_stitched(z_s)
    beta = _wedge_barycentric(z_s, d)
    V = icosahedron.vertices()
    n = icosahedron.vertex_neighbors()[base]
    n_cw = int(n[d - 2])
    n_ccw = int(n[(d - 1) % 5])
    p = beta[0] * V[base] + beta[1] * V[n_cw] + beta[2] * V[n_ccw]
    return p / np.linalg.norm(p)


def cell_center_g(cell):
    base, digits = cell[0], cell[1:]
    return project_gnomonic(_eisenstein_center(digits), base)


def cell_boundary_g(cell):
    """Six hex corners of ``cell`` projected via gnomonic-via-triangle."""
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

    # Source hex boundary on the sphere (gnomonic).
    src_ring = [unit_to_lnglat(v) for v in cell_boundary_g(SOURCE_CELL)]
    src_center_lnglat = unit_to_lnglat(cell_center_g(SOURCE_CELL))

    # Expected ring-1 neighbors (from existing cell_ring1).
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

    # Project z_n for each of the 6 walks via the SAME gnomonic projection.
    walk_segments = []
    walk_markers = []
    for D in range(1, 7):
        z_n = z_C + digit_offset[D] / rot_N
        p3d = project_gnomonic(z_n, base)
        ll = unit_to_lnglat(p3d)
        walk_markers.append({"pos": ll, "text": f"D={D}"})
        walk_segments.append([src_center_lnglat, ll])

    # Icosa edge skeleton.
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
            {"type": "labels", "items": walk_markers, "style": {
                "fill": "#36c", "stroke": "white", "strokeWidth": 3,
                "fontSize": 12, "fontWeight": 700,
            }},
            {"type": "labels", "items": expected_labels, "style": {
                "fill": "#0a8a3a", "stroke": "white", "strokeWidth": 3,
                "fontSize": 9, "fontWeight": 500,
            }},
        ],
    }]

    info_html = f"""<aside class="info">
  <h1>Atlas ring-1: source {SOURCE_CELL}</h1>
  <p>Single gnomonic projection (mu3 stitch + linear-combo on icosa triangles)
     used for source hex, expected ring-1 cells, AND walk endpoints —
     so the geometry is internally consistent.</p>
  <p>Red: source. Green: expected ring-1 neighbors. Blue arrows: walk D=1..6
     destinations. Each blue arrowhead should land at (or very near) the
     center of one green expected cell.</p>
  <p>Drag to rotate, scroll to zoom, double-click to reset.</p>
</aside>"""

    out = Path(__file__).resolve().parent.parent / "figures" / "ring1_atlas_globe.html"
    render_globe_page(
        title=f"mu3 ring-1 atlas walk for {SOURCE_CELL}",
        info_html=info_html,
        panels=panels,
        output_path=out,
        layout="single",
    )


if __name__ == "__main__":
    main()
