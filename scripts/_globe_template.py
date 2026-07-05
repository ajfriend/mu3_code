"""Shared HTML template for orthographic globe pages.

Used by ``make_globe_plot.py`` and ``make_dodec_plot.py``. Produces a
self-contained HTML page with:

- d3 orthographic projection.
- Versor-quaternion drag (free pole movement; persistent container so the
  inner SVG can be re-rendered mid-drag without losing the gesture).
- Wheel zoom + WASDQE keyboard controls + ``+``/``-`` for zoom +
  double-click to reset.
- Earth landmass outlines fetched from ``world-atlas@2/land-110m.json``.
- One or more synchronized panels (rotation/zoom shared across panels).

Two layouts:

- ``layout="single"`` — one centered globe filling the viewport.
- ``layout="grid"`` — 2x2 grid of synchronized globes (each labeled).

Each panel supplies a list of layers. Supported layer types:

- ``{"type": "polygons", "coords": MultiPolygon-coords, "style": {...}}``
- ``{"type": "lines",    "coords": MultiLineString-coords, "style": {...}}``
- ``{"type": "arrows",   "segments": [[a, b], ...], "style": {...}}``
- ``{"type": "labels",   "items": [{"pos": [lng, lat], "text": str}, ...], "style": {...}}``

``style`` is forwarded to ``Plot.geo`` / ``Plot.text`` more or less
directly (Plot's channel API).
"""

import json
from pathlib import Path

import numpy as np


def unit_to_lnglat(v) -> list[float]:
    """3D unit vector → [lng, lat] in degrees."""
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    lng = float(np.degrees(np.arctan2(y, x)))
    lat = float(np.degrees(np.arcsin(max(-1.0, min(1.0, z)))))
    return [lng, lat]


_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  html, body { margin: 0; padding: 0; height: 100%; box-sizing: border-box; }
  *, *::before, *::after { box-sizing: inherit; }
  body { background: #fff; font-family: -apple-system, BlinkMacSystemFont, sans-serif; color: #222; overflow: hidden; }

  .info { position: fixed; top: 24px; left: 24px; max-width: 280px; font-size: 13px; line-height: 1.5; color: #333; z-index: 10; }
  .info h1 { font-size: 18px; font-weight: 500; margin: 0 0 14px 0; color: #222; }
  .info p { margin: 0 0 12px 0; }
  .info code { background: #f3f3f5; padding: 1px 4px; border-radius: 3px; font-size: 12px; }
  kbd { font-family: ui-monospace, Menlo, monospace; font-size: 11px; background: #f3f3f5; border: 1px solid #ddd; border-radius: 3px; padding: 1px 5px; }

  body.single { padding: 32px; display: flex; align-items: center; justify-content: center; }
  body.single .panel-globe { aspect-ratio: 1; height: 100%; max-width: 100%; cursor: grab; outline: none; }

  body.grid { padding: 24px; }
  body.grid .info { position: static; top: auto; left: auto; max-width: none; margin-bottom: 12px; }
  body.grid .grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-template-rows: 1fr 1fr;
    gap: 16px;
    width: min(calc(100vh - 220px), calc(100vw - 48px));
    height: min(calc(100vh - 220px), calc(100vw - 48px));
    margin: 0 auto;
  }
  body.grid .panel { display: flex; flex-direction: column; min-width: 0; min-height: 0; }
  body.grid .panel-title { font-size: 12px; font-weight: 600; color: #334; }
  body.grid .panel-sub { font-size: 11px; color: #888; }
  body.grid .panel-globe { flex: 1 1 0; min-height: 0; width: 100%; cursor: grab; outline: none; }

  body.row { padding: 24px; }
  body.row .info { position: static; top: auto; left: auto; max-width: none; margin-bottom: 12px; }
  body.row .row { display: flex; gap: 16px; width: 100%; align-items: stretch; }
  body.row .panel { flex: 1 1 0; min-width: 0; display: flex; flex-direction: column; }
  body.row .panel-title { font-size: 12px; font-weight: 600; color: #334; }
  body.row .panel-sub { font-size: 11px; color: #888; }
  body.row .panel-globe { flex: 1 1 0; aspect-ratio: 1; max-height: calc(100vh - 200px); width: 100%; cursor: grab; outline: none; }

  .panel-globe:active { cursor: grabbing; }
</style>
</head>
<body class="__LAYOUT__">

__INFO_HTML__

__PANELS_HTML__

<script type="module">
  import * as Plot from 'https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6/+esm';
  import * as d3 from 'https://cdn.jsdelivr.net/npm/d3@7/+esm';
  import { feature } from 'https://cdn.jsdelivr.net/npm/topojson-client@3/+esm';

  const PANELS = __PANELS_JSON__;
  const DEFAULT_ROTATE = [20, -30, 0];
  const DEFAULT_ZOOM = 1.0;
  const ZOOM_MIN = 0.5, ZOOM_MAX = 12;
  let rotate = DEFAULT_ROTATE.slice();
  let zoom = DEFAULT_ZOOM;

  const landTopo = await (await fetch('https://cdn.jsdelivr.net/npm/world-atlas@2/land-110m.json')).json();
  const land = feature(landTopo, landTopo.objects.land);

  // Versor (quaternion) math for free 3D drag.
  const versor = {
    cartesian(e) {
      const l = e[0]*Math.PI/180, p = e[1]*Math.PI/180, cp = Math.cos(p);
      return [cp*Math.cos(l), cp*Math.sin(l), Math.sin(p)];
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
    dot(a, b)   { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
  };

  // Build Plot marks list from a panel's layers.
  function buildMarks(layers) {
    const marks = [
      Plot.graticule({ strokeOpacity: 0.08 }),
      Plot.geo(land, { stroke: '#88b', fill: 'none', strokeWidth: 0.8, strokeOpacity: 0.7 }),
    ];
    for (const layer of layers) {
      if (layer.type === 'polygons') {
        // d3 expects CW outer rings; our coords are CCW.
        if (layer.values) {
          const scheme = layer.scheme || 'Viridis';
          const interp = d3['interpolate' + scheme[0].toUpperCase() + scheme.slice(1)];
          const cs = d3.scaleSequential(interp).domain(layer.domain || [0, 1]);
          const features = layer.coords.map((poly, i) => ({
            type: 'Feature',
            geometry: { type: 'Polygon', coordinates: poly.map(ring => ring.slice().reverse()) },
            properties: { value: layer.values[i] },
          }));
          const fc = { type: 'FeatureCollection', features };
          const style = Object.assign({}, layer.style || {});
          delete style.fill;
          marks.push(Plot.geo(fc, Object.assign({ fill: d => cs(d.properties.value) }, style)));
        } else {
          const reversed = layer.coords.map(poly => poly.map(ring => ring.slice().reverse()));
          const feat = { type: 'Feature', geometry: { type: 'MultiPolygon', coordinates: reversed } };
          marks.push(Plot.geo(feat, layer.style || {}));
        }
      } else if (layer.type === 'lines') {
        const feat = { type: 'Feature', geometry: { type: 'MultiLineString', coordinates: layer.coords } };
        marks.push(Plot.geo(feat, layer.style || {}));
      } else if (layer.type === 'arrows') {
        const style = layer.style || {};
        const feat = { type: 'Feature', geometry: { type: 'MultiLineString', coordinates: layer.segments } };
        marks.push(Plot.geo(feat, style));
        for (const [a, b] of layer.segments) {
          const interp = d3.geoInterpolate(a, b);
          marks.push(Plot.line([interp(0.85), b], {
            x: d => d[0], y: d => d[1],
            stroke: style.stroke,
            strokeWidth: style.strokeWidth,
            strokeOpacity: style.strokeOpacity,
            markerEnd: 'arrow',
          }));
        }
      } else if (layer.type === 'labels') {
        const style = layer.style || {};
        marks.push(Plot.text(layer.items.map(it => it.pos), {
          x: d => d[0],
          y: d => d[1],
          text: (_, i) => layer.items[i].text,
          fill: style.fill || '#222',
          stroke: style.stroke || 'white',
          strokeWidth: style.strokeWidth || 3,
          fontSize: style.fontSize || 11,
          fontWeight: style.fontWeight || 600,
        }));
      }
    }
    marks.push(Plot.sphere({ strokeWidth: 2 }));
    return marks;
  }

  // Per-panel: render + drag + zoom. Drag is bound to the persistent container,
  // not the inner SVG (which is re-created every frame).
  const renders = [];
  function renderAll() { for (const r of renders) r(); }

  for (const panel of PANELS) {
    const container = document.getElementById(panel.id);
    function getSize() { return container.offsetWidth || 600; }

    function render() {
      const size = getSize();
      container.innerHTML = '';
      const plot = Plot.plot({
        width: size, height: size,
        projection: ({width, height}) =>
          d3.geoOrthographic()
            .rotate(rotate)
            .translate([width/2, height/2])
            .scale(zoom * (Math.min(width, height)/2 - 2))
            .clipAngle(90),
        marks: buildMarks(panel.layers),
      });
      container.appendChild(plot);
    }
    renders.push(render);

    const projection = d3.geoOrthographic();
    function configureProjection(rot) {
      const size = getSize();
      projection.rotate(rot)
        .translate([size/2, size/2])
        .scale(zoom * (size/2 - 2));
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
        renderAll();
      });

    d3.select(container).call(drag);
    d3.select(container).on('dblclick', () => {
      rotate = DEFAULT_ROTATE.slice();
      zoom = DEFAULT_ZOOM;
      renderAll();
    });
    container.addEventListener('wheel', (e) => {
      e.preventDefault();
      const factor = Math.exp(-e.deltaY * 0.0015);
      zoom = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, zoom * factor));
      renderAll();
    }, { passive: false });

    render();
  }

  window.addEventListener('resize', renderAll);

  // WASDQE keyboard nudges (versor-composed, screen-relative).
  document.addEventListener('keydown', (event) => {
    if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') return;
    const step = event.shiftKey ? 30 : 10;
    const k = event.key;
    if (k === 'q' || k === 'Q') { rotate = [rotate[0], rotate[1], (rotate[2] || 0) + step]; renderAll(); event.preventDefault(); }
    else if (k === 'e' || k === 'E') { rotate = [rotate[0], rotate[1], (rotate[2] || 0) - step]; renderAll(); event.preventDefault(); }
    else if (k === 'w' || k === 'W' || k === 's' || k === 'S') {
      const sign = (k === 'w' || k === 'W') ? 1 : -1;
      const ha = sign * step / 2 * Math.PI / 180;
      const q = versor.from(rotate);
      const qRot = [Math.cos(ha), 0, Math.sin(ha), 0];
      rotate = versor.rotation(versor.multiply(qRot, q));
      renderAll(); event.preventDefault();
    } else if (k === 'a' || k === 'A' || k === 'd' || k === 'D') {
      const sign = (k === 'a' || k === 'A') ? -1 : 1;
      const ha = sign * step / 2 * Math.PI / 180;
      const q = versor.from(rotate);
      const qRot = [Math.cos(ha), Math.sin(ha), 0, 0];
      rotate = versor.rotation(versor.multiply(qRot, q));
      renderAll(); event.preventDefault();
    } else if (k === '+' || k === '=') {
      zoom = Math.min(ZOOM_MAX, zoom * (event.shiftKey ? 1.5 : 1.15));
      renderAll(); event.preventDefault();
    } else if (k === '-' || k === '_') {
      zoom = Math.max(ZOOM_MIN, zoom / (event.shiftKey ? 1.5 : 1.15));
      renderAll(); event.preventDefault();
    }
  });
</script>
</body>
</html>
"""


def render_globe_page(*, title, info_html, panels, output_path: Path, layout="single"):
    """Render an orthographic globe page to ``output_path``.

    ``layout`` is ``"single"`` for one full-page globe, or ``"grid"`` for a
    2x2 synchronized layout (panels rotate/zoom together).
    """
    if layout == "single":
        if len(panels) != 1:
            raise ValueError("single layout requires exactly 1 panel")
        panels_html = (
            f'<div class="panel-globe" id="{panels[0]["id"]}" tabindex="0"></div>'
        )
    elif layout in ("grid", "row"):
        cells = []
        for p in panels:
            sub = (
                f'<div class="panel-sub">{p["subtitle"]}</div>'
                if p.get("subtitle") else ""
            )
            cells.append(
                f'<div class="panel">'
                f'<div class="panel-title">{p["title"]}</div>'
                f'{sub}'
                f'<div class="panel-globe" id="{p["id"]}" tabindex="0"></div>'
                f'</div>'
            )
        panels_html = f'<div class="{layout}">' + "".join(cells) + "</div>"
    else:
        raise ValueError(f"unknown layout {layout!r}")

    html = (
        _HTML
        .replace("__TITLE__", title)
        .replace("__INFO_HTML__", info_html or "")
        .replace("__LAYOUT__", layout)
        .replace("__PANELS_HTML__", panels_html)
        .replace("__PANELS_JSON__", json.dumps(panels))
    )

    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(html)
    print(f"wrote {output_path}")
    print(f"size: {output_path.stat().st_size / 1024:.1f} KB")
    print(f"open with: open {output_path}")


def icosa_edge_coords():
    """The icosahedron edge skeleton as lnglat line segments — the
    shared base layer of the globe scripts (style stays per-script)."""
    from mu3 import dodec
    return [
        [unit_to_lnglat(dodec.normals[i]), unit_to_lnglat(dodec.normals[j])]
        for (i, j) in dodec.icosa_edges
    ]
