"""Single draggable globe showing the mu3 grid at resolution 3.

Pentagons are filled red; hexagons blue. Earth landmasses for orientation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _globe_template import render_globe_page, unit_to_lnglat

from mu3 import cell_boundary, cells_at_res, is_pentagon


def main():
    res = 3
    pent_polys = []
    hex_polys = []
    for cell in cells_at_res(res):
        ring = [unit_to_lnglat(v) for v in cell_boundary(cell, closed=True)]
        if is_pentagon(cell):
            pent_polys.append([ring])
        else:
            hex_polys.append([ring])

    print(f"res {res}: {len(pent_polys)} pentagons, {len(hex_polys)} hexagons")

    panels = [{
        "id": "g-res3",
        "layers": [
            {"type": "polygons", "coords": hex_polys, "style": {
                "stroke": "#246", "strokeWidth": 0.4,
                "fill": "#48d", "fillOpacity": 0.35,
            }},
            {"type": "polygons", "coords": pent_polys, "style": {
                "stroke": "#822", "strokeWidth": 0.6,
                "fill": "#e44", "fillOpacity": 0.55,
            }},
        ],
    }]

    info_html = """<aside class="info">
  <h1>mu3 grid · resolution 3</h1>
  <p>Pentagons red, hexagons blue. Drag to rotate; scroll to zoom; double-click to reset.</p>
  <p>Keys: <kbd>W</kbd>/<kbd>S</kbd> tilt, <kbd>A</kbd>/<kbd>D</kbd> roll, <kbd>Q</kbd>/<kbd>E</kbd> spin, <kbd>+</kbd>/<kbd>-</kbd> zoom.</p>
</aside>"""

    out = Path(__file__).resolve().parent.parent / "figures" / "mu3_res3_globe.html"
    render_globe_page(
        title="mu3 grid · resolution 3",
        info_html=info_html,
        panels=panels,
        output_path=out,
        layout="single",
    )


if __name__ == "__main__":
    main()
