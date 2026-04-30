"""Discrete cell-shape distortion across the four mu3 Projections at res 3.

Per-cell metrics (over hex cells only — pentagons are anomalies):
  - area max/min                  (already known)
  - edge ratio (max/min) globally
  - per-cell shape  = max edge / min edge (in arc length)
  - per-cell ang_dev = max |interior_angle − 120°| (degrees)

Mirrors experiment_alpha_hex_compare.py from the sibling distortion repo
(``/Users/aj/work/2026-04-18_distort/scripts/``).
"""

import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mu3 import cell as _cell
from mu3 import cell_area, cell_boundary, cells_at_res, is_pentagon
from mu3.projection import (
    AlphaOnlySlerp,
    AlphaSlerp,
    Gnomonic,
    IVEAProjection,
    KarcherProjection,
    LambertBaryProjection,
)


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


def arc(a, b):
    return float(np.arccos(np.clip(a @ b, -1.0, 1.0)))


def interior_angle(A, B, C):
    """Spherical interior angle at B between great-circles BA and BC."""
    tA = A - (A @ B) * B
    tC = C - (C @ B) * B
    nA = np.linalg.norm(tA)
    nC = np.linalg.norm(tC)
    if nA < 1e-12 or nC < 1e-12:
        return float(np.pi)
    return float(np.arccos(np.clip((tA @ tC) / (nA * nC), -1.0, 1.0)))


def measure(cls, res=3):
    edge_lengths = []
    cell_shapes = []
    cell_ang_devs_deg = []
    cell_areas = []
    n_hex = 0
    n_pent = 0
    with active_projection(cls):
        for cell in cells_at_res(res):
            ring = cell_boundary(cell, closed=False)
            n = len(ring)
            edges = np.array([arc(ring[i], ring[(i + 1) % n]) for i in range(n)])
            edge_lengths.extend(edges.tolist())
            if is_pentagon(cell):
                n_pent += 1
                continue
            n_hex += 1
            cell_shapes.append(float(edges.max() / edges.min()))
            angs = np.array([
                interior_angle(ring[(i - 1) % n], ring[i], ring[(i + 1) % n])
                for i in range(n)
            ])
            ang_devs = np.abs(angs - 2.0 * np.pi / 3.0)
            cell_ang_devs_deg.append(float(np.degrees(ang_devs.max())))
            cell_areas.append(abs(cell_area(cell)))
    edges = np.array(edge_lengths)
    shapes = np.array(cell_shapes)
    angs = np.array(cell_ang_devs_deg)
    areas = np.array(cell_areas)
    return {
        "n_hex": n_hex, "n_pent": n_pent,
        "area_r": float(areas.max() / areas.min()),
        "edge_r": float(edges.max() / edges.min()),
        "shape_mx": float(shapes.max()),
        "shape_p90": float(np.percentile(shapes, 90)),
        "shape_p50": float(np.percentile(shapes, 50)),
        "ang_dev": float(angs.max()),
        "ang_p90": float(np.percentile(angs, 90)),
        "ang_p50": float(np.percentile(angs, 50)),
    }


def main():
    classes = [
        ("Gnomonic",                   Gnomonic),
        ("IVEA (Slice & Dice)",        IVEAProjection),
        ("AlphaSlerp (rich, default)", AlphaSlerp),
        ("AlphaOnlySlerp",             AlphaOnlySlerp),
        ("Karcher (Riemannian mean)",  KarcherProjection),
        ("LambertBary (DEAD END)",     LambertBaryProjection),
    ]
    print(f"{'projection':<32s} {'n':>4s} {'area_r':>7s} {'edge_r':>7s} "
          f"{'shape_mx':>9s} {'sh_p90':>7s} {'sh_p50':>7s} "
          f"{'ang_mx°':>8s} {'ang_p90°':>9s} {'ang_p50°':>9s}")
    for name, cls in classes:
        m = measure(cls)
        print(f"{name:<32s} {m['n_hex']:4d} "
              f"{m['area_r']:7.4f} {m['edge_r']:7.4f} "
              f"{m['shape_mx']:9.4f} {m['shape_p90']:7.4f} {m['shape_p50']:7.4f} "
              f"{m['ang_dev']:8.3f} {m['ang_p90']:9.3f} {m['ang_p50']:9.3f}",
              flush=True)


if __name__ == "__main__":
    main()
