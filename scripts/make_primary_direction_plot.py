# /// script
# dependencies = ["numpy"]
# ///
"""Interactive d3 globe showing the 12 dodecahedron pentagon faces, the
icosahedron edges, per-pentagon primary-direction arrows, and tuple labels.

Geometry uses the standard-position icosahedron: 12 vertices at cyclic
permutations of (0, +/-1, +/-phi). This makes the (family, s_1, s_2) tuple
visible on the sphere (each of the 3 axes becomes a "zero axis" for one
family of 4 pentagons).

The primary direction on each pentagon is the tangent at its center pointing
toward the "same-family neighbor" (flip s_1). See reports/nearest-dodecahedron-face.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PHI = (1 + np.sqrt(5)) / 2


def build_vertices():
    """Return 12 unit vertices + list of (family, s1, s2) tuples.

    family = 0: (0, s1, s2*phi)
    family = 1: cyclic rotate once -> (s2*phi, 0, s1)
    family = 2: cyclic rotate twice -> (s1, s2*phi, 0)
    """
    verts = []
    tuples = []
    for f in range(3):
        for s1 in (1, -1):
            for s2 in (1, -1):
                base = np.array([0.0, s1, s2 * PHI])
                # cyclic rotate axis assignments f times: (x,y,z) -> (z,x,y)
                v = np.roll(base, f)
                verts.append(v)
                tuples.append((f, s1, s2))
    verts = np.array(verts)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    return verts, tuples


def primary_direction(f, s1, s2):
    """Unit tangent at vertex (f, s1, s2) pointing to same-family neighbor.

    At family 0: (0, -s1*phi, s2)/sqrt(1+phi^2). Cyclic rotate for other families.
    """
    t = np.array([0.0, -s1 * PHI, s2])
    t = np.roll(t, f)
    t = t / np.linalg.norm(t)
    return t


def find_neighbors(verts, i):
    """Indices of the 5 icosahedron-edge neighbors of vertex i."""
    # Edge <=> cos(angle) = phi / (phi + 2).
    v = verts[i]
    dots = verts @ v
    target = PHI / (PHI + 2)
    # tolerance is loose; the three cos-values for an icosahedron are
    # {1, phi/(phi+2) ~ 0.4472, -phi/(phi+2), -1}.
    return [j for j in range(12) if j != i and abs(dots[j] - target) < 1e-6]


def sort_ccw(center, neighbors):
    """Return indices 0..len-1 sorting `neighbors` CCW around `center`."""
    # Build an orthonormal tangent basis at center.
    up = np.array([0.0, 0.0, 1.0])
    if abs(center @ up) > 0.95:
        up = np.array([1.0, 0.0, 0.0])
    e1 = up - (up @ center) * center
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(center, e1)
    angles = [np.arctan2(n @ e2, n @ e1) for n in neighbors]
    return np.argsort(angles)


def pentagon_corners(verts, i):
    """5 corners of the Voronoi pentagon on the sphere around vertex i.

    Corner = centroid of the triangle (v, n_k, n_{k+1}) for consecutive
    CCW-sorted neighbors, projected to the sphere.
    """
    v = verts[i]
    nbr = find_neighbors(verts, i)
    order = sort_ccw(v, verts[nbr])
    nbr = [nbr[k] for k in order]
    corners = []
    for k in range(5):
        j = nbr[k]
        jn = nbr[(k + 1) % 5]
        c = (v + verts[j] + verts[jn]) / 3.0
        c = c / np.linalg.norm(c)
        corners.append(c)
    return np.array(corners), nbr


def icosahedron_edges(verts):
    """Return list of (i, j) index pairs for all 30 edges."""
    edges = set()
    for i in range(12):
        for j in find_neighbors(verts, i):
            edges.add((min(i, j), max(i, j)))
    return sorted(edges)


def unit_to_lnglat(v):
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    lng = float(np.degrees(np.arctan2(y, x)))
    lat = float(np.degrees(np.arcsin(max(-1.0, min(1.0, z)))))
    return [lng, lat]


def great_circle(p, q, n=24):
    """n+1 points along the great-circle arc from unit p to unit q."""
    dot = float(np.clip(p @ q, -1.0, 1.0))
    omega = np.arccos(dot)
    if omega < 1e-8:
        return [p.tolist()]
    sinw = np.sin(omega)
    ts = np.linspace(0.0, 1.0, n + 1)
    return [
        ((np.sin((1 - t) * omega) * p + np.sin(t * omega) * q) / sinw).tolist()
        for t in ts
    ]


def arc_to_lnglat(arc):
    return [unit_to_lnglat(np.array(p)) for p in arc]


def arrow_feature(start_v, tangent, length_rad=0.22, head_ratio=0.3, head_width_rad=0.04):
    """Great-circle arrow: shaft + two short head segments forming an arrowhead.

    Returns a list of lng/lat linestrings.
    """
    # shaft end
    end = start_v * np.cos(length_rad) + tangent * np.sin(length_rad)
    end = end / np.linalg.norm(end)

    # arrowhead: two short segments from a point `head_ratio` before the end,
    # splayed to the left and right in the tangent plane at that point.
    hp_dist = length_rad * (1 - head_ratio)
    hp = start_v * np.cos(hp_dist) + tangent * np.sin(hp_dist)
    hp = hp / np.linalg.norm(hp)
    # tangent at hp still roughly along `tangent` projected
    t_hp = tangent - (tangent @ hp) * hp
    t_hp = t_hp / np.linalg.norm(t_hp)
    side = np.cross(hp, t_hp)

    left = hp * np.cos(head_width_rad) + side * np.sin(head_width_rad)
    right = hp * np.cos(head_width_rad) - side * np.sin(head_width_rad)
    left = left / np.linalg.norm(left)
    right = right / np.linalg.norm(right)

    shaft = great_circle(start_v, end)
    head_l = great_circle(left, end)
    head_r = great_circle(right, end)
    return [arc_to_lnglat(shaft), arc_to_lnglat(head_l), arc_to_lnglat(head_r)]


def main():
    verts, tuples = build_vertices()

    # Pentagons
    pentagons = []
    for i in range(12):
        corners, _ = pentagon_corners(verts, i)
        # Close ring with dense samples along each pentagon edge (great circles)
        ring = []
        for k in range(5):
            a, b = corners[k], corners[(k + 1) % 5]
            arc = great_circle(a, b, n=12)
            # drop the last point to avoid duplicates; we'll close the ring at the end
            ring.extend(arc[:-1])
        ring.append(corners[0].tolist())
        pentagons.append([arc_to_lnglat(ring)])

    # Icosahedron edges (dense great-circle arcs)
    ico_lines = []
    for i, j in icosahedron_edges(verts):
        ico_lines.append(arc_to_lnglat(great_circle(verts[i], verts[j], n=16)))

    # Primary-direction arrows + labels
    arrows = []
    labels = []
    id_by_tuple = {}
    for i, (f, s1, s2) in enumerate(tuples):
        t = primary_direction(f, s1, s2)
        arrow_segs = arrow_feature(verts[i], t)
        arrows.extend(arrow_segs)
        packed = 4 * f + 2 * (1 if s1 == 1 else 0) + (1 if s2 == 1 else 0)
        id_by_tuple[(f, s1, s2)] = packed
        s1_str = "+" if s1 == 1 else "-"
        s2_str = "+" if s2 == 1 else "-"
        labels.append({
            "lnglat": unit_to_lnglat(verts[i]),
            "id": packed,
            "tuple": f"({f},{s1_str},{s2_str})",
        })

    data = {
        "pentagons": pentagons,
        "icosahedron_edges": ico_lines,
        "arrows": arrows,
        "labels": labels,
    }
    data_json = json.dumps(data)

    out = Path(__file__).resolve().parent.parent / "figures" / "primary_direction.html"
    out.parent.mkdir(exist_ok=True)

    HTML = (
        """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>mu3 pentagon primary directions</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 20px; color: #222; background: #fff; }
  h1 { font-weight: 400; margin-bottom: 4px; }
  .sub { color: #555; font-size: 13px; margin-bottom: 16px; max-width: 720px; line-height: 1.45; }
  .panel { border: 1px solid #ddd; padding: 12px; border-radius: 4px; max-width: 720px; }
  .globe { width: 100%; aspect-ratio: 1; cursor: grab; }
  .globe:active { cursor: grabbing; }
  .legend { font-size: 12px; color: #555; margin-top: 10px; line-height: 1.5; }
  .sw { display: inline-block; width: 14px; height: 14px; vertical-align: middle; margin-right: 4px; border: 1px solid rgba(0,0,0,0.1); }
</style>
</head>
<body>

<h1>Pentagon primary directions</h1>
<div class="sub">
  12 dodecahedron pentagon faces on the unit sphere, with icosahedron edges for reference.
  Each arrow points from a pentagon center toward its "same-family neighbor" --- the
  icosahedron-edge direction obtained by flipping s<sub>1</sub> in the (family, s<sub>1</sub>, s<sub>2</sub>) tuple.
  Labels show (family, s<sub>1</sub>, s<sub>2</sub>) above and packed id below.
  Drag to rotate; double-click to reset.
</div>

<div class="panel">
  <div id="globe" class="globe"></div>
  <div class="legend">
    <span class="sw" style="background:#e55; opacity:0.25"></span>pentagon faces &nbsp;
    <span class="sw" style="background:#888"></span>icosahedron edges &nbsp;
    <span class="sw" style="background:#0a7"></span>primary direction
  </div>
</div>

<script type="module">
  import * as Plot from 'https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6/+esm';
  import * as d3 from 'https://cdn.jsdelivr.net/npm/d3@7/+esm';

  const DATA = __DATA_PLACEHOLDER__;

  // d3 / Observable Plot expects CW outer rings for small polygons on a sphere.
  // Our pentagons come out CCW viewed from outside; reverse them.
  const pentagons = DATA.pentagons.map(polygon => polygon.map(ring => ring.slice().reverse()));

  const pentaFeature = { type: 'Feature', geometry: { type: 'MultiPolygon', coordinates: pentagons } };
  const icoFeature   = { type: 'Feature', geometry: { type: 'MultiLineString', coordinates: DATA.icosahedron_edges } };
  const arrowFeature = { type: 'Feature', geometry: { type: 'MultiLineString', coordinates: DATA.arrows } };
  const labelPoints  = DATA.labels.map(l => ({ lng: l.lnglat[0], lat: l.lnglat[1], tuple: l.tuple, id: l.id }));

  const rot = [20, -30, 0];
  const container = document.getElementById('globe');
  function getSize() { return container.offsetWidth || 600; }

  function render() {
    const size = getSize();
    container.innerHTML = '';

    const plot = Plot.plot({
      width: size,
      height: size,
      projection: { type: 'orthographic', rotate: rot.slice(), inset: 4 },
      marks: [
        Plot.graticule({ strokeOpacity: 0.08 }),
        Plot.geo(icoFeature, { stroke: '#888', strokeWidth: 1.2, strokeOpacity: 0.55 }),
        Plot.geo(pentaFeature, { stroke: '#c33', strokeWidth: 0.6, fill: '#e55', fillOpacity: 0.18 }),
        Plot.geo(arrowFeature, { stroke: '#0a7', strokeWidth: 2.0, strokeOpacity: 0.9 }),
        Plot.text(labelPoints, { x: 'lng', y: 'lat', text: 'tuple', dy: -8, fontSize: 10, fontWeight: 600, fill: '#113' }),
        Plot.text(labelPoints, { x: 'lng', y: 'lat', text: d => `#${d.id}`, dy: 6, fontSize: 9, fill: '#557' }),
        Plot.sphere({ strokeWidth: 1.5 })
      ]
    });

    container.appendChild(plot);

    const drag = d3.drag()
      .on('start', (e) => { drag.sx = e.x; drag.sy = e.y; drag.sr = rot.slice(); })
      .on('drag', (e) => {
        const dx = e.x - drag.sx, dy = e.y - drag.sy, k = 0.25;
        rot[0] = drag.sr[0] + dx * k;
        rot[1] = drag.sr[1] - dy * k;
        rot[2] = 0;
        render();
      });
    d3.select(plot).call(drag);
    d3.select(plot).on('dblclick', () => { rot[0] = 20; rot[1] = -30; rot[2] = 0; render(); });
  }

  render();
  window.addEventListener('resize', render);
</script>
</body>
</html>
"""
    ).replace("__DATA_PLACEHOLDER__", data_json)

    out.write_text(HTML)
    print(f"wrote {out}")
    print(f"size: {out.stat().st_size / 1024:.1f} KB")
    print(f"open with: open {out}")


if __name__ == "__main__":
    main()
