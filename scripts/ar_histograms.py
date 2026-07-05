# /// script
# requires-python = '>=3.11'
# dependencies = [
#     'numpy',
#     'matplotlib',
#     'mu3 @ git+https://github.com/ajfriend/mu3_code.git',
#     'skar @ git+https://github.com/ajfriend/skar_py.git',
# ]
# ///
"""mu3 aspect-ratio distribution by resolution.

The mu3 analogue of skar_py's `by_res_<system>.png`: solve every cell's
spherical boundary with skar's tightest-enclosing-cone solver and stack the
per-resolution AR histograms (coarsest at top), shared bins, log y.

AR = skar's enclosing-cone cross-section axis ratio (>= 1); AR = 1 is a
perfectly circular cell. Cells come from the active mu3 projection
(AlphaSlerp), so this measures that projection's cell shapes.

    uv run scripts/ar_histograms.py
"""

import time
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

import skar
from mu3 import cell_boundary, cells_at_res

# ----- knobs -------------------------------------------------------------
MAX_RES = 7
GAP_TOL = 1e-6
N_BINS = 60
COLOR = 'C0'
OUT = Path(__file__).resolve().parent / 'out' / 'ar_histograms.png'
# -------------------------------------------------------------------------


def solve_res(res):
    """AR of every cell at `res`. Returns (ars array, dnc count)."""
    ars, dnc = [], 0
    for cell in cells_at_res(res):
        verts = cell_boundary(cell, closed=False)
        r = skar.solve(verts, geo='vec3', gap_tol=GAP_TOL)
        if isinstance(r, skar.Converged):
            ars.append(r.aspect_ratio)
        else:
            dnc += 1
    return np.asarray(ars), dnc


by_res = {}
for res in range(MAX_RES + 1):
    t0 = time.perf_counter()
    by_res[res] = solve_res(res)
    a, dnc = by_res[res]
    print(f'[r{res}] {a.size:,} cells  dnc={dnc}  '
          f'median={np.median(a):.4f}  max={a.max():.4f}  '
          f'({time.perf_counter() - t0:.1f}s)', flush=True)

# ----- stats -------------------------------------------------------------
print(f'{"res":>3} {"n":>7} {"dnc":>4} {"min":>9} {"median":>9} {"p99":>9} {"max":>9}')
for res, (a, dnc) in by_res.items():
    print(f'{res:>3} {a.size:>7} {dnc:>4} {a.min():>9.4f} {np.median(a):>9.4f} '
          f'{np.percentile(a, 99):>9.4f} {a.max():>9.4f}')

# ----- plot (skar_py plot_by_resolution style) ---------------------------
allars = np.concatenate([a for a, _ in by_res.values()])
amax = float(np.percentile(allars, 99.9))
bins = np.linspace(1.0, amax, N_BINS + 1)

# Put the n/median note on the emptier horizontal third (skar_py convention).
mass = np.histogram(allars, bins=bins)[0]
third = max(len(mass) // 3, 1)
note_x, note_ha = ((0.985, 'right') if mass[:third].sum() >= mass[-third:].sum()
                   else (0.015, 'left'))

n = MAX_RES + 1
fig_h = 1.4 * n + 1.4
fig, axes = plt.subplots(n, 1, figsize=(10, fig_h), sharex=True, squeeze=False)
for ax, res in zip(axes[:, 0], range(MAX_RES + 1)):
    a, dnc = by_res[res]
    red = bool(dnc)
    counts = ax.hist(a, bins=bins, color=COLOR, edgecolor='white', linewidth=0.3)[0]
    ax.set_yscale('log')
    maxc = max(int(counts.max()), 1)
    ax.set_yticks([10.0 ** k for k in range(int(np.floor(np.log10(maxc))) + 1)])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel(f'r{res}', rotation=0, ha='right', va='center', labelpad=12,
                  fontsize=13, fontweight='bold', color='red' if red else '0.2')
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.25)
    note = f'n = {a.size:,}   median {np.median(a):.4f}' + (f'   DNC {dnc:,}' if red else '')
    ax.text(note_x, 0.9, note, transform=ax.transAxes, ha=note_ha, va='top',
            fontsize=11, color='red' if red else '0.4')
axes[-1, 0].set_xlabel('aspect ratio (shared bins, gap_tol = 1e-6)', fontsize=12)
fig.suptitle(f'mu3 (AlphaSlerp) aspect-ratio distribution by resolution '
             f'(coarsest at top; shared bins 1.00–{amax:.2f}, log y)',
             fontsize=14, y=1 - 0.4 / fig_h, va='top')
fig.tight_layout(rect=(0, 0, 1, 1 - 1.0 / fig_h))
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=130)
print('wrote', OUT)
