"""Self-contained HTML page with a rotatable d3 orthographic globe showing
the 12 dodecahedron-face pentagons and the 30 icosahedron edges from
``mu3.dodec``, with Earth landmasses overlaid for orientation.

Drag uses versor math (free pole movement); WASDQE keys nudge the rotation.

``mu3.dodec`` already places its vertices in H3's icosahedron orientation
(vertices over water), so no extra rotation is applied here.

Output: ``figures/dodec_globe.html``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mu3 import dodec


def unit_to_lnglat(v):
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    lng = float(np.degrees(np.arctan2(y, x)))
    lat = float(np.degrees(np.arcsin(max(-1.0, min(1.0, z)))))
    return [lng, lat]


def main():
    pentagons = []
    for i in range(12):
        ring = [unit_to_lnglat(c) for c in dodec.pentagon_corners[i]]
        ring.append(ring[0])
        pentagons.append([ring])

    icosa_lines = [
        [unit_to_lnglat(dodec.normals[i]), unit_to_lnglat(dodec.normals[j])]
        for (i, j) in dodec.icosa_edges
    ]

    face_centers = [unit_to_lnglat(n) for n in dodec.normals]

    # Primary direction: short great-circle arrow from each face center
    # toward its primary-tangent direction (~20 deg, < pentagon radius).
    arrow_angle = np.radians(20)
    primary_arrows = []
    for i in range(12):
        n = dodec.normals[i]
        t = dodec.primary_tangents[i]
        end = np.cos(arrow_angle) * n + np.sin(arrow_angle) * t
        primary_arrows.append([unit_to_lnglat(n), unit_to_lnglat(end)])

    data = {
        'pentagons': pentagons,
        'icosa_edges': icosa_lines,
        'face_centers': face_centers,
        'face_labels': [f'{i}: {tuple(dodec._faces[i])}' for i in range(12)],
        'primary_arrows': primary_arrows,
    }
    data_json = json.dumps(data)

    out = Path(__file__).resolve().parent.parent / 'figures' / 'dodec_globe.html'
    out.parent.mkdir(exist_ok=True)

    HTML = (
        """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>mu3.dodec on the sphere</title>
<style>
  html, body { margin: 0; padding: 0; height: 100%; box-sizing: border-box; }
  *, *::before, *::after { box-sizing: inherit; }
  body { display: flex; align-items: center; justify-content: center; padding: 32px; background: #fff; font-family: -apple-system, BlinkMacSystemFont, sans-serif; color: #222; overflow: hidden; }
  #g { aspect-ratio: 1; height: 100%; max-width: 100%; cursor: grab; outline: none; }
  #g:active { cursor: grabbing; }
  .info { position: fixed; top: 24px; left: 24px; max-width: 280px; font-size: 13px; line-height: 1.5; color: #333; }
  .info h1 { font-size: 18px; font-weight: 500; margin: 0 0 14px 0; color: #222; }
  .info p { margin: 0 0 12px 0; }
  .info code { background: #f3f3f5; padding: 1px 4px; border-radius: 3px; font-size: 12px; }
  kbd { font-family: ui-monospace, Menlo, monospace; font-size: 11px; background: #f3f3f5; border: 1px solid #ddd; border-radius: 3px; padding: 1px 5px; }
</style>
</head>
<body>

<div id="g" tabindex="0"></div>
<aside class="info">
  <h1>mu3.dodec on the sphere</h1>
  <p>12 dodecahedron-face pentagons (red), 30 icosahedron edges (gray), face-normal centers labeled with <code>(axis, s1, s2)</code> from <code>dodec._faces</code>. Earth landmasses overlaid for orientation.</p>
  <p>Drag to rotate freely (poles unconstrained). Scroll to zoom. Double-click to reset.</p>
  <p>Keys: <kbd>W</kbd>/<kbd>S</kbd> tilt, <kbd>A</kbd>/<kbd>D</kbd> roll, <kbd>Q</kbd>/<kbd>E</kbd> spin, <kbd>+</kbd>/<kbd>-</kbd> zoom. <kbd>Shift</kbd> = bigger steps.</p>
</aside>

<script type="module">
  import * as Plot from 'https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6/+esm';
  import * as d3 from 'https://cdn.jsdelivr.net/npm/d3@7/+esm';
  import { feature } from 'https://cdn.jsdelivr.net/npm/topojson-client@3/+esm';

  const DATA = __DATA_PLACEHOLDER__;

  // d3 expects CW outer rings; reverse the CCW rings we produced.
  const pentagons = DATA.pentagons.map(poly => poly.map(ring => ring.slice().reverse()));

  const pentagonsFeature = {
    type: 'Feature',
    geometry: { type: 'MultiPolygon', coordinates: pentagons }
  };
  const icosaFeature = {
    type: 'Feature',
    geometry: { type: 'MultiLineString', coordinates: DATA.icosa_edges }
  };
  const primaryArrowsFeature = {
    type: 'Feature',
    geometry: { type: 'MultiLineString', coordinates: DATA.primary_arrows }
  };

  const container = document.getElementById('g');
  const DEFAULT_ROTATE = [20, -30, 0];
  const DEFAULT_ZOOM = 1.0;
  const ZOOM_MIN = 0.5, ZOOM_MAX = 12;
  let rotate = DEFAULT_ROTATE.slice();
  let zoom = DEFAULT_ZOOM;

  const landTopo = await (await fetch('https://cdn.jsdelivr.net/npm/world-atlas@2/land-110m.json')).json();
  const land = feature(landTopo, landTopo.objects.land);

  function getSize() { return container.offsetWidth || 600; }

  function render() {
    const size = getSize();
    container.innerHTML = '';

    // For each primary-direction great-circle arc, add a tiny straight line
    // segment near the tip with markerEnd: 'arrow', since Plot.geo doesn't
    // place arrowhead markers on LineString segments.
    const arrowHeadMarks = DATA.primary_arrows.map(([a, b]) => {
      const interp = d3.geoInterpolate(a, b);
      return Plot.line([interp(0.85), b], {
        x: d => d[0],
        y: d => d[1],
        stroke: '#0a8a3a',
        strokeWidth: 1.8,
        strokeOpacity: 0.95,
        markerEnd: 'arrow',
      });
    });

    const plot = Plot.plot({
      width: size,
      height: size,
      projection: ({width, height}) =>
        d3.geoOrthographic()
          .rotate(rotate)
          .translate([width / 2, height / 2])
          .scale(zoom * (Math.min(width, height) / 2 - 2))
          .clipAngle(90),
      marks: [
        Plot.graticule({ strokeOpacity: 0.08 }),
        Plot.geo(land, { stroke: '#88b', fill: 'none', strokeWidth: 0.8, strokeOpacity: 0.7 }),
        Plot.geo(icosaFeature, { stroke: '#888', strokeWidth: 1.5, strokeOpacity: 0.6 }),
        Plot.geo(pentagonsFeature, { stroke: '#c33', strokeWidth: 1, fill: '#e55', fillOpacity: 0.15 }),
        Plot.geo(primaryArrowsFeature, { stroke: '#0a8a3a', strokeWidth: 1.8, strokeOpacity: 0.95 }),
        ...arrowHeadMarks,
        Plot.text(DATA.face_centers, {
          x: d => d[0],
          y: d => d[1],
          text: (_, i) => DATA.face_labels[i],
          fill: '#222',
          stroke: 'white',
          strokeWidth: 3,
          fontSize: 11,
          fontWeight: 600,
        }),
        Plot.sphere({ strokeWidth: 2 }),
      ]
    });

    container.appendChild(plot);
  }

  // Versor math for 3D rotation (free pole movement).
  const versor = {
    cartesian(e) {
      const l = e[0] * Math.PI / 180, p = e[1] * Math.PI / 180, cp = Math.cos(p);
      return [cp * Math.cos(l), cp * Math.sin(l), Math.sin(p)];
    },
    rotation(q) {
      return [
        Math.atan2(2*(q[0]*q[1]+q[2]*q[3]), 1-2*(q[1]*q[1]+q[2]*q[2])) * 180/Math.PI,
        Math.asin(Math.max(-1, Math.min(1, 2*(q[0]*q[2]-q[3]*q[1])))) * 180/Math.PI,
        Math.atan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]*q[2]+q[3]*q[3])) * 180/Math.PI
      ];
    },
    delta(v0, v1) {
      const w = this.cross(v0, v1), l = Math.sqrt(this.dot(w, w));
      if (!l) return [1, 0, 0, 0];
      const t = Math.acos(Math.max(-1, Math.min(1, this.dot(v0, v1)))) / 2, s = Math.sin(t);
      return [Math.cos(t), w[2]/l*s, -w[1]/l*s, w[0]/l*s];
    },
    multiply(a, b) {
      return [
        a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3],
        a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
        a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1],
        a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]
      ];
    },
    from(r) {
      const l = r[0]/2*Math.PI/180, p = r[1]/2*Math.PI/180, g = r[2]/2*Math.PI/180;
      const sl = Math.sin(l), cl = Math.cos(l), sp = Math.sin(p), cp = Math.cos(p), sg = Math.sin(g), cg = Math.cos(g);
      return [cl*cp*cg+sl*sp*sg, sl*cp*cg-cl*sp*sg, cl*sp*cg+sl*cp*sg, cl*cp*sg-sl*sp*cg];
    },
    cross(a, b) { return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; },
    dot(a, b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
  };

  // Drag bound once to the persistent container. The inner SVG is recreated
  // on every frame, so binding to it would lose the gesture mid-drag.
  const projection = d3.geoOrthographic();
  function configureProjection(rot) {
    const size = getSize();
    projection.rotate(rot)
      .translate([size / 2, size / 2])
      .scale(zoom * (size / 2 - 2));
  }

  let v0, q0, r0;
  const drag = d3.drag()
    .on('start', function(event) {
      configureProjection(rotate);
      const p = d3.pointer(event, container);
      const inv = projection.invert(p);
      if (!inv) { v0 = null; return; }
      v0 = versor.cartesian(inv);
      r0 = rotate.slice();
      q0 = versor.from(r0);
    })
    .on('drag', function(event) {
      if (!v0) return;
      configureProjection(r0);
      const p = d3.pointer(event, container);
      const inv = projection.invert(p);
      if (!inv) return;
      const v1 = versor.cartesian(inv);
      const delta = versor.delta(v0, v1);
      const q1 = versor.multiply(q0, delta);
      rotate = versor.rotation(q1);
      render();
    });

  d3.select(container).call(drag);
  d3.select(container).on('dblclick', () => {
    rotate = DEFAULT_ROTATE.slice();
    zoom = DEFAULT_ZOOM;
    render();
  });

  // Wheel-to-zoom on the container.
  container.addEventListener('wheel', (event) => {
    event.preventDefault();
    const factor = Math.exp(-event.deltaY * 0.0015);
    zoom = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, zoom * factor));
    render();
  }, { passive: false });

  // WASDQE keyboard nudges (versor-composed so screen-relative axes stay sane
  // regardless of pole position).
  document.addEventListener('keydown', (event) => {
    if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') return;
    const step = event.shiftKey ? 30 : 10;
    const k = event.key;

    // Q/E: spin in screen plane (gamma).
    if (k === 'q' || k === 'Q') {
      rotate = [rotate[0], rotate[1], (rotate[2] || 0) + step];
      render(); event.preventDefault();
    } else if (k === 'e' || k === 'E') {
      rotate = [rotate[0], rotate[1], (rotate[2] || 0) - step];
      render(); event.preventDefault();
    }
    // W/S: tilt forward/back (screen-y axis).
    else if (k === 'w' || k === 'W' || k === 's' || k === 'S') {
      const sign = (k === 'w' || k === 'W') ? 1 : -1;
      const ha = sign * step / 2 * Math.PI / 180;
      const q = versor.from(rotate);
      const qRot = [Math.cos(ha), 0, Math.sin(ha), 0];
      rotate = versor.rotation(versor.multiply(qRot, q));
      render(); event.preventDefault();
    }
    // A/D: roll (screen-x axis).
    else if (k === 'a' || k === 'A' || k === 'd' || k === 'D') {
      const sign = (k === 'a' || k === 'A') ? -1 : 1;
      const ha = sign * step / 2 * Math.PI / 180;
      const q = versor.from(rotate);
      const qRot = [Math.cos(ha), Math.sin(ha), 0, 0];
      rotate = versor.rotation(versor.multiply(qRot, q));
      render(); event.preventDefault();
    }
    // +/-: zoom.
    else if (k === '+' || k === '=') {
      zoom = Math.min(ZOOM_MAX, zoom * (event.shiftKey ? 1.5 : 1.15));
      render(); event.preventDefault();
    } else if (k === '-' || k === '_') {
      zoom = Math.max(ZOOM_MIN, zoom / (event.shiftKey ? 1.5 : 1.15));
      render(); event.preventDefault();
    }
  });

  render();
  window.addEventListener('resize', render);
</script>
</body>
</html>
"""
    ).replace('__DATA_PLACEHOLDER__', data_json)

    out.write_text(HTML)
    print(f'wrote {out}')
    print(f'size: {out.stat().st_size / 1024:.1f} KB')
    print(f'open with: open {out}')


if __name__ == '__main__':
    main()
