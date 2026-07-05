'''D3 globe view of rings 1..K_MAX around a cell.

Rings come from the BFS traversal oracle (``mu3.traversal``); every
polygon is the cell's actual spherical boundary and every label sits
at the cell's actual center — real lat/lngs throughout, no schematic
layout. Edit CELL / K_MAX below and rerun.
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _globe_template import icosa_edge_coords, render_globe_page, unit_to_lnglat

from mu3 import cell_boundary, cell_center
from mu3.traversal import grid_distances

# Source cell. (0, 2, 6, 6) gives clean 6k rings; try a
# pentagon-adjacent cell like (0, 0, 0, 2) to see rings wrap the
# 5-fold defect, or a pentagon center (b, 0, 0, 0) for 5k rings.
CELL = (0, 2, 6, 6)
K_MAX = 52

_PALETTE = (
    '#0a8a3a', '#1565c0', '#7b1fa2', '#e08a00',
    '#00838f', '#ad1457', '#558b2f', '#5d4037',
)


def ring_color(k: int) -> str:
    """Source cell red; rings cycle the palette (any k)."""
    return '#c33' if k == 0 else _PALETTE[(k - 1) % len(_PALETTE)]


def _lnglat_ring(cell):
    return [unit_to_lnglat(v) for v in cell_boundary(cell)]


def main():
    dists = grid_distances(CELL, K_MAX)
    by_ring: dict[int, list] = {k: [] for k in range(K_MAX + 1)}
    for cell, k in dists.items():
        by_ring[k].append(cell)

    layers = [
        {'type': 'lines', 'coords': icosa_edge_coords(), 'style': {
            'stroke': '#888', 'strokeWidth': 1.0, 'strokeOpacity': 0.4,
        }},
    ]
    for k in range(K_MAX, -1, -1):
        color = ring_color(k)
        layers.append({
            'type': 'polygons',
            'coords': [[_lnglat_ring(c)] for c in by_ring[k]],
            'style': {
                'stroke': color, 'strokeWidth': 2.0 if k == 0 else 1.4,
                'fill': color, 'fillOpacity': 0.35 if k == 0 else 0.15,
            },
        })
        layers.append({
            'type': 'labels',
            'items': [
                {'pos': unit_to_lnglat(cell_center(c)), 'text': str(k)}
                for c in by_ring[k]
            ],
            'style': {
                'fill': color, 'stroke': 'white', 'strokeWidth': 3,
                'fontSize': 11, 'fontWeight': 700,
            },
        })

    counts = ', '.join(
        f'ring {k}: {len(by_ring[k])}' for k in range(1, K_MAX + 1)
    )
    info_html = (
        f'<b>rings 1..{K_MAX} around {CELL}</b><br>'
        f'{counts}<br>'
        f'each cell drawn with its actual spherical boundary and '
        f'labeled with its grid distance at its actual center'
    )

    out = Path(__file__).resolve().parent.parent / 'figures' / 'ring_k_globe.html'
    render_globe_page(
        title=f'mu3 rings 1..{K_MAX} from {CELL}',
        info_html=info_html,
        panels=[{'id': 'g-ringk', 'layers': layers}],
        output_path=out,
        layout='single',
    )


if __name__ == '__main__':
    main()
