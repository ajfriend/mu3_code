"""Self-contained HTML page with 4 synchronized rotatable globes showing
mu3's res 0–3 cells on the sphere, with the 30 icosahedron edges overlaid
and Earth landmasses for orientation.

Cells come straight from the library: ``mu3.cell_boundary(cell)`` where
``cell = (base, d_1, ..., d_N)``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _globe_template import render_globe_page, unit_to_lnglat

from mu3 import cell_boundary, cells_at_res, dodec


def main():
    icosa_lines = [
        [unit_to_lnglat(dodec.normals[i]), unit_to_lnglat(dodec.normals[j])]
        for (i, j) in dodec.icosa_edges
    ]

    panels = []
    for r in (0, 1, 2, 3):
        polys = []
        for cell in cells_at_res(r):
            ring = [unit_to_lnglat(v) for v in cell_boundary(cell, closed=True)]
            polys.append([ring])
        panels.append({
            "id": f"g-{r}",
            "title": f"Resolution {r}",
            "subtitle": f"{len(polys)} cells",
            "layers": [
                {"type": "lines", "coords": icosa_lines, "style": {
                    "stroke": "#888", "strokeWidth": 1.5, "strokeOpacity": 0.5,
                }},
                {"type": "polygons", "coords": polys, "style": {
                    "stroke": "#c33", "strokeWidth": 0.5, "fill": "#e55", "fillOpacity": 0.2,
                }},
            ],
        })
        print(f"res {r}: {len(polys)} cells")

    info_html = """<aside class="info">
  <h1>mu3 cells on the sphere</h1>
  <p>Icosahedral DGGS, aperture 7. Pentagons at icosa vertices; hexes fill out at higher resolutions.</p>
  <p>Drag any globe to rotate all four. Scroll to zoom. Double-click to reset.</p>
  <p>Keys: <kbd>W</kbd>/<kbd>S</kbd> tilt, <kbd>A</kbd>/<kbd>D</kbd> roll, <kbd>Q</kbd>/<kbd>E</kbd> spin, <kbd>+</kbd>/<kbd>-</kbd> zoom.</p>
</aside>"""

    out = Path(__file__).resolve().parent.parent / "figures" / "mu3_globe.html"
    render_globe_page(
        title="mu3 cells on the sphere",
        info_html=info_html,
        panels=panels,
        output_path=out,
        layout="grid",
    )


if __name__ == "__main__":
    main()
