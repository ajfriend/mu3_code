'''Fit the banded-polish bow envelope for the ACTIVE projection.

For each resolution, measures the worst envelope coefficient

    c = bow(t) / (4 * t * (1 - t))

over sampled cell edges (bow = deviation of the pulled-back geodesic
from the flat chord, lattice units, sampled at several t), and prints a
ready-to-paste ``_BOW_COEFFS`` entry with a x2 operating margin.

Cells that the runtime stitch guard routes to the full polish are
excluded — the table only needs to cover edges the banded fast path
actually side-tests. Coverage: every cell at res <= 2; at higher res,
the pentagon vicinity (k-rings just outside the guard), the face-center
vicinity (interior-distortion peak), and a random sample.

Paste the output into ``mu3.index._BOW_COEFFS``; the fit is pinned with
headroom by ``tests/test_one_hop_contract.py``.
'''

import random

import numpy as np

from mu3 import icosahedron
from mu3.cell import (
    _sphere_to_flat,
    active_projection_name,
    cell_boundary,
    cells_at_res,
)
from mu3.eisenstein import first_nonzero_digit
from mu3.face_lattice import get_rot
from mu3.index import _cell_near_stitch, vec3_to_cell_raw
from mu3.neighbor import cell_ring1
from mu3.traversal import disk_k

RES_MAX = 12
N_RANDOM = 80
PENTAGON_RING_MAX = 3
T_SAMPLES = (1 / 6, 1 / 3, 1 / 2, 2 / 3, 5 / 6)
MARGIN = 2.0

rng = random.Random(20260710)


def random_cell(res):
    while True:
        digits = [rng.randrange(7) for _ in range(res)]
        if first_nonzero_digit(digits) != 1:
            return (rng.randrange(12), *digits)


def seam_cells(res):
    """Cells straddling chart seams (icosa edges / wedge spokes) — the
    worst-bow population, since adjacent wedge projections meet only
    C0 there. Locate cells at points along every icosa edge and take
    their 1-rings."""
    V = icosahedron.vertices()
    nbrs = icosahedron.vertex_neighbors()
    cells = set()
    for i in range(12):
        for j in nbrs[i]:
            if j < i:
                continue
            for t in (0.25, 0.5, 0.75):
                q = (1 - t) * V[i] + t * V[j]
                c = vec3_to_cell_raw(q / np.linalg.norm(q), res)
                cells.add(c)
                cells.update(cell_ring1(c))
    return cells


def sample_cells(res):
    if res <= 2:
        return list(cells_at_res(res))
    cells = seam_cells(res)
    for b in range(12):
        cells |= set(disk_k((b,) + (0,) * res, PENTAGON_RING_MAX))
    for _ in range(N_RANDOM):
        cells.add(random_cell(res))
    return cells


def envelope_coeff(res, cells):
    'Worst bow(t)/(4 t (1-t)) over non-guarded cell edges, and count.'
    rot = get_rot(res)
    worst = 0.0
    n_edges = 0
    for cell in cells:
        if _cell_near_stitch(cell):
            continue
        base = cell[0]
        B = cell_boundary(cell, closed=False)
        m = len(B)
        flats = [_sphere_to_flat(v, base) * rot for v in B]
        for k in range(m):
            v1, v2 = B[k], B[(k + 1) % m]
            z1, z2 = flats[k], flats[(k + 1) % m]
            chord = z2 - z1
            L2 = (chord * chord.conjugate()).real
            if L2 < 1e-18:
                continue
            n_edges += 1
            for t in T_SAMPLES:
                gm = (1 - t) * v1 + t * v2
                gm = gm / np.linalg.norm(gm)
                zm = _sphere_to_flat(gm, base) * rot
                th = ((zm - z1) * chord.conjugate()).real / L2
                th = min(0.98, max(0.02, th))
                bow = abs(zm - (z1 + th * chord))
                worst = max(worst, bow / (4.0 * th * (1.0 - th)))
    return worst, n_edges


print(f'active projection: {active_projection_name()}')
print(f'measured envelope coefficient and x{MARGIN:g} operating value:')
table = {}
for res in range(1, RES_MAX + 1):
    c, n_edges = envelope_coeff(res, sample_cells(res))
    table[res] = MARGIN * c
    print(f'  res {res:2d}: measured {c:.4f} over {n_edges} edges '
          f'-> operating {table[res]:.4f}')

print()
print('paste into mu3.index._BOW_COEFFS:')
print(f"    {active_projection_name()!r}: {{")
for res, v in table.items():
    print(f'        {res}: {v:.4f},')
print('    },')
