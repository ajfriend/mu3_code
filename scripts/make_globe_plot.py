"""Build a self-contained HTML page with 3 rotatable d3 globes showing
mu3's resolution 0, 1, and 2 cells, with the icosahedron edges overlaid
for reference."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from mu3 import cell_boundary, icosahedron


def unit_to_lnglat(v: np.ndarray) -> list[float]:
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    lng = float(np.degrees(np.arctan2(y, x)))
    lat = float(np.degrees(np.arcsin(max(-1.0, min(1.0, z)))))
    return [lng, lat]


def enumerate_cells(r: int):
    for base in range(12):
        if r == 0:
            yield (base, ())
            continue
        for digits in itertools.product(range(7), repeat=r):
            first_nonzero = next((d for d in digits if d != 0), None)
            if first_nonzero == 1:
                continue
            yield (base, digits)


def cell_ring(base: int, digits: tuple) -> list[list[float]]:
    bnd = cell_boundary(base, digits, closed=True)
    return [unit_to_lnglat(v) for v in bnd]


def icosahedron_edges() -> list[list[list[float]]]:
    V = icosahedron.vertices()
    F = icosahedron.faces()
    edges: set[tuple[int, int]] = set()
    for face in F:
        a, b, c = int(face[0]), int(face[1]), int(face[2])
        for u, v in ((a, b), (b, c), (c, a)):
            edges.add(tuple(sorted((u, v))))
    return [[unit_to_lnglat(V[u]), unit_to_lnglat(V[v])] for u, v in edges]


def main() -> None:
    cells_by_res: dict[str, list] = {}
    for r in (0, 1, 2):
        polys = []
        for base, digits in enumerate_cells(r):
            polys.append([cell_ring(base, digits)])
        cells_by_res[str(r)] = polys
        print(f"res {r}: {len(polys)} cells")

    icosa_lines = icosahedron_edges()
    print(f"icosahedron: {len(icosa_lines)} edges")

    data = {"cells_by_res": cells_by_res, "icosahedron_edges": icosa_lines}
    data_json = json.dumps(data, separators=(",", ":"))

    out = Path(__file__).resolve().parent.parent / "figures" / "mu3_globe.html"
    out.parent.mkdir(exist_ok=True)

    HTML = (
        """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>mu3 cells on the sphere</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 20px; color: #222; background: #fff; }
  h1 { font-weight: 400; margin-bottom: 4px; }
  .sub { color: #555; font-size: 13px; margin-bottom: 20px; }
  .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; max-width: 1400px; }
  .panel { border: 1px solid #ddd; padding: 12px; border-radius: 4px; }
  .panel h2 { margin: 0 0 8px 0; font-size: 14px; font-weight: 600; color: #334; }
  .panel .count { color: #888; font-size: 12px; margin: 0 0 6px 0; }
  .globe { width: 100%; aspect-ratio: 1; cursor: grab; }
  .globe:active { cursor: grabbing; }
  .hint { color: #888; font-size: 11px; margin-top: 6px; }
</style>
</head>
<body>

<h1>mu3 cells on the sphere</h1>
<div class="sub">Icosahedral DGGS, aperture 7. Base pentagons at icosa vertices; hexes fill out at higher resolutions. Drag any globe to rotate all; double-click to reset.</div>

<div class="grid">
  <div class="panel"><h2>Resolution 0</h2><div class="count" id="c-0"></div><div id="g-0" class="globe"></div></div>
  <div class="panel"><h2>Resolution 1</h2><div class="count" id="c-1"></div><div id="g-1" class="globe"></div></div>
  <div class="panel"><h2>Resolution 2</h2><div class="count" id="c-2"></div><div id="g-2" class="globe"></div></div>
</div>

<script type="module">
  import * as Plot from 'https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6/+esm';
  import * as d3 from 'https://cdn.jsdelivr.net/npm/d3@7/+esm';

  const DATA = __DATA_PLACEHOLDER__;

  // d3 expects CW outer rings; our rings are CCW GeoJSON-style. Reverse them.
  for (const key of Object.keys(DATA.cells_by_res)) {
    DATA.cells_by_res[key] = DATA.cells_by_res[key].map(
      polygon => polygon.map(ring => ring.slice().reverse())
    );
  }

  for (const res of Object.keys(DATA.cells_by_res)) {
    document.getElementById('c-' + res).textContent = DATA.cells_by_res[res].length + ' cells';
  }

  const icosaFeature = {
    type: 'Feature',
    geometry: {
      type: 'MultiLineString',
      coordinates: DATA.icosahedron_edges
    }
  };

  const sharedRotate = [20, -30, 0];
  const renderAll = [];
  function renderAllGlobes() { for (const r of renderAll) r(); }

  function setupGlobe(res, containerId) {
    const container = document.getElementById(containerId);
    const gridFeature = {
      type: 'Feature',
      geometry: { type: 'MultiPolygon', coordinates: DATA.cells_by_res[res] }
    };

    function getSize() {
      const w = container.offsetWidth || 400;
      return Math.max(280, Math.min(520, w));
    }

    function render() {
      const size = getSize();
      container.innerHTML = '';

      const plot = Plot.plot({
        width: size,
        height: size,
        projection: { type: 'orthographic', rotate: sharedRotate.slice(), inset: 1 },
        marks: [
          Plot.graticule({ strokeOpacity: 0.08 }),
          Plot.geo(icosaFeature, { stroke: '#888', strokeWidth: 1.0, strokeOpacity: 0.6 }),
          Plot.geo(gridFeature, { stroke: '#c33', strokeWidth: 0.7, fill: '#e55', fillOpacity: 0.15 }),
          Plot.sphere({ strokeWidth: 1.5 })
        ]
      });

      container.appendChild(plot);

      const drag = d3.drag()
        .on('start', (e) => {
          drag.sx = e.x; drag.sy = e.y;
          drag.sr = sharedRotate.slice();
        })
        .on('drag', (e) => {
          const dx = e.x - drag.sx, dy = e.y - drag.sy, k = 0.25;
          sharedRotate[0] = drag.sr[0] + dx * k;
          sharedRotate[1] = drag.sr[1] - dy * k;
          sharedRotate[2] = 0;
          renderAllGlobes();
        });
      d3.select(plot).call(drag);
      d3.select(plot).on('dblclick', () => {
        sharedRotate[0] = 20;
        sharedRotate[1] = -30;
        sharedRotate[2] = 0;
        renderAllGlobes();
      });
    }

    renderAll.push(render);
    render();
    window.addEventListener('resize', () => render());
  }

  setupGlobe('0', 'g-0');
  setupGlobe('1', 'g-1');
  setupGlobe('2', 'g-2');
</script>
</body>
</html>
"""
    ).replace("__DATA_PLACEHOLDER__", data_json)

    out.write_text(HTML)
    print(f"\nwrote {out}")
    print(f"size: {out.stat().st_size / 1024:.1f} KB")
    print(f"open with: open {out}")


if __name__ == "__main__":
    main()
