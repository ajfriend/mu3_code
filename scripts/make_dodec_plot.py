"""Self-contained HTML page showing the 12 dodecahedron-face pentagons,
30 icosahedron edges, face labels, and primary-direction arrows from
``mu3.dodec``, on a rotatable orthographic globe with Earth landmasses
overlaid for orientation.

Output: ``figures/dodec_globe.html``.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _globe_template import icosa_edge_coords, render_globe_page, unit_to_lnglat

from mu3 import dodec


def main():
    pentagons = []
    for i in range(12):
        ring = [unit_to_lnglat(c) for c in dodec.pentagon_corners[i]]
        ring.append(ring[0])
        pentagons.append([ring])

    icosa_lines = icosa_edge_coords()

    arrow_angle = np.radians(20)
    primary_arrows = [
        [
            unit_to_lnglat(dodec.normals[i]),
            unit_to_lnglat(
                np.cos(arrow_angle) * dodec.normals[i]
                + np.sin(arrow_angle) * dodec.primary_tangents[i]
            ),
        ]
        for i in range(12)
    ]

    labels = [
        {
            "pos": unit_to_lnglat(dodec.normals[i]),
            "text": f"{i}: {tuple(dodec._faces[i])}",
        }
        for i in range(12)
    ]

    panel = {
        "id": "g",
        "layers": [
            {"type": "lines", "coords": icosa_lines, "style": {
                "stroke": "#888", "strokeWidth": 1.5, "strokeOpacity": 0.6,
            }},
            {"type": "polygons", "coords": pentagons, "style": {
                "stroke": "#c33", "strokeWidth": 1, "fill": "#e55", "fillOpacity": 0.15,
            }},
            {"type": "arrows", "segments": primary_arrows, "style": {
                "stroke": "#0a8a3a", "strokeWidth": 1.8, "strokeOpacity": 0.95,
            }},
            {"type": "labels", "items": labels, "style": {}},
        ],
    }

    info_html = """<aside class="info">
  <h1>mu3.dodec on the sphere</h1>
  <p>12 dodecahedron-face pentagons (red), 30 icosahedron edges (gray), face-normal centers labeled with <code>(axis, s1, s2)</code> from <code>dodec._faces</code>, and primary-direction arrows (green) from <code>dodec.primary_tangents</code>.</p>
  <p>Drag to rotate freely. Scroll to zoom. Double-click to reset.</p>
  <p>Keys: <kbd>W</kbd>/<kbd>S</kbd> tilt, <kbd>A</kbd>/<kbd>D</kbd> roll, <kbd>Q</kbd>/<kbd>E</kbd> spin, <kbd>+</kbd>/<kbd>-</kbd> zoom.</p>
</aside>"""

    out = Path(__file__).resolve().parent.parent / "figures" / "dodec_globe.html"
    render_globe_page(
        title="mu3.dodec on the sphere",
        info_html=info_html,
        panels=[panel],
        output_path=out,
        layout="single",
    )


if __name__ == "__main__":
    main()
