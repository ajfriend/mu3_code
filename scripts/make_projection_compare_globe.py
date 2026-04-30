"""4-panel synced globe comparing all four Projections at resolution 3.

Hexagons are colored by spherical area (viridis, shared scale across
all panels). Pentagons stay solid red. The shared scale lets the
projection-induced area variance read at a glance — Gnomonic should
show the most spread, the equal-area maps the least.
"""

import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _globe_template import render_globe_page, unit_to_lnglat

from mu3 import cell as _cell
from mu3 import cell_area, cell_boundary, cells_at_res, is_pentagon
from mu3.projection import AlphaOnlySlerp, AlphaSlerp, Gnomonic, IVEAProjection


@contextmanager
def active_projection(cls):
    saved = _cell._PROJECTION_CLS
    _cell._PROJECTION_CLS = cls
    _cell._projection.cache_clear()
    try:
        yield
    finally:
        _cell._PROJECTION_CLS = saved
        _cell._projection.cache_clear()


def collect(cls):
    pent_polys, hex_polys, hex_areas = [], [], []
    with active_projection(cls):
        for cell in cells_at_res(3):
            ring = [unit_to_lnglat(v) for v in cell_boundary(cell, closed=True)]
            if is_pentagon(cell):
                pent_polys.append([ring])
            else:
                hex_polys.append([ring])
                hex_areas.append(abs(cell_area(cell)))
    return pent_polys, hex_polys, hex_areas


def main():
    classes = [
        (Gnomonic,        "Gnomonic",                   "g-gnom"),
        (IVEAProjection,  "IVEA (Slice & Dice)",        "g-ivea"),
        (AlphaSlerp,      "AlphaSlerp (rich, default)", "g-rich"),
        (AlphaOnlySlerp,  "AlphaOnlySlerp",             "g-only"),
    ]

    collected = []
    all_areas = []
    for cls, title, panel_id in classes:
        pent, hexp, hexa = collect(cls)
        collected.append((title, panel_id, pent, hexp, hexa))
        all_areas.extend(hexa)
        print(f"{title}: hex area [{min(hexa):.4e}, {max(hexa):.4e}]  "
              f"max/min = {max(hexa) / min(hexa):.4f}")

    domain = [min(all_areas), max(all_areas)]
    print(f"\nshared color domain: [{domain[0]:.4e}, {domain[1]:.4e}]  "
          f"max/min = {domain[1] / domain[0]:.4f}")

    panels = []
    for title, panel_id, pent, hexp, hexa in collected:
        panels.append({
            "id": panel_id,
            "title": title,
            "subtitle": f"{len(pent)} pentagons + {len(hexp)} hexagons",
            "layers": [
                {"type": "polygons", "coords": hexp, "values": hexa,
                 "domain": domain, "scheme": "viridis",
                 "style": {"stroke": "#0a0a0a", "strokeWidth": 0.15,
                           "strokeOpacity": 0.35, "fillOpacity": 0.85}},
                {"type": "polygons", "coords": pent, "style": {
                    "stroke": "#822", "strokeWidth": 0.6,
                    "fill": "#e44", "fillOpacity": 0.85,
                }},
            ],
        })

    info_html = f"""<aside class="info">
  <h1>mu3 res 3 · area distortion across projections</h1>
  <p>Hexagons colored by spherical area (viridis, shared scale ·
     {domain[0]:.3e} → {domain[1]:.3e} sr). Pentagons solid red.</p>
  <p>Drag any globe to rotate all three; scroll to zoom; double-click to reset.</p>
</aside>"""

    out = Path(__file__).resolve().parent.parent / "figures" / "mu3_projection_compare.html"
    render_globe_page(
        title="mu3 res 3 · area distortion across projections",
        info_html=info_html,
        panels=panels,
        output_path=out,
        layout="row",
    )


if __name__ == "__main__":
    main()
