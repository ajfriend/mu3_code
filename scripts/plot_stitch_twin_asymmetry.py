"""Diagnostic: cell (0, 2, 5) at res 2 has a stitch twin (0, 1, 4) in the
deleted wedge. Walks from its canonical position miss neighbors that are
reachable from the twin position -- this breaks ring-1 symmetry.

Setup:

- Pentagon P=0 only, res 2 lattice.
- Cell ``(0, 2, 5)`` is at canonical position ``z_C = 0 + 0.247i`` --
  on the y-axis, at the boundary between the d=1 (deleted) wedge and
  the d=2 wedge.
- Its stitch twin ``(0, 1, 4)`` is at ``(0.214, 0.124i)``, in the deleted
  wedge. ``(0.214, 0.124i) * (1+omega) = (0, 0.247i)`` -- same 3D cell.
- Walks from canonical position give 6 destinations, but 2 collapse via
  digit canonicalization onto cells already reached by other walks --
  4 unique. The cell ``(0, 0, 6)`` is NOT among them.
- Walks from the twin position give a DIFFERENT 6 destinations,
  including ``(0, 0, 6)``.
- For ring-1 symmetry, ``(0, 0, 6)`` must be in ``ring1((0, 2, 5))`` --
  but my code (using only the canonical position) misses it.

Output: ``figures/stitch_twin_asymmetry.png``.
"""

import cmath
import itertools
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from mu3.cell import _eisenstein_center
from mu3.face_lattice import digit_offset, get_rot, omega, s3, units


RES = 2


def first_nonzero_digit(digits):
    for d in digits:
        if d != 0:
            return d
    return None


def enumerate_cells(res: int):
    if res == 0:
        yield ((), 0 + 0j)
        return
    for digits in itertools.product(range(7), repeat=res):
        z = 0 + 0j
        for k, d in enumerate(digits, start=1):
            if d == 0:
                continue
            z += digit_offset[d] / get_rot(k)
        yield (digits, z)


def hex_corners_local(center: complex, rot: complex):
    return [center + units[k] / (s3 * rot) for k in range(6)]


def draw_lattice(ax):
    rot_N = get_rot(RES)
    # Wedge skeleton + deleted wedge.
    for j in range(6):
        a, b, c = 0 + 0j, units[j], units[(j + 1) % 6]
        verts = [(a.real, a.imag), (b.real, b.imag), (c.real, c.imag)]
        if j == 0:  # deleted wedge [0, 60)
            ax.add_patch(Polygon(verts, closed=True, facecolor="#f0a04a",
                                 edgecolor="#d96b0f", linestyle="--",
                                 linewidth=0.7, alpha=0.30, zorder=1))
        else:
            ax.plot([v[0] for v in verts] + [verts[0][0]],
                    [v[1] for v in verts] + [verts[0][1]],
                    color="#888", lw=0.4, alpha=0.4, zorder=1)

    for digits, z_local in enumerate_cells(RES):
        if abs(z_local) > 0.55:
            continue
        corners = hex_corners_local(z_local, rot_N)
        xs = [c.real for c in corners] + [corners[0].real]
        ys = [c.imag for c in corners] + [corners[0].imag]

        is_pentagon = all(d == 0 for d in digits)
        is_deleted_child = first_nonzero_digit(digits) == 1

        if is_pentagon:
            ax.plot(xs, ys, color="#c33", lw=1.6, zorder=4)
            ax.fill(xs, ys, color="#c33", alpha=0.12, zorder=3)
        elif is_deleted_child:
            ax.plot(xs, ys, color="#d96b0f", lw=0.5, alpha=0.7,
                    ls="--", zorder=2.5)
            ax.fill(xs, ys, color="#fde7c8", alpha=0.4, zorder=2.4)
        else:
            ax.plot(xs, ys, color="#888", lw=0.3, alpha=0.5, zorder=2)
            ax.plot(z_local.real, z_local.imag, ".", color="#888",
                    markersize=1, alpha=0.5, zorder=2)


def label_cell(ax, digits, name, color, fontsize=8, offset=None,
               weight="normal"):
    z = _eisenstein_center(digits)
    if offset is None:
        offset = 0.06 * cmath.exp(1j * (cmath.phase(z) if abs(z) > 0.01 else 0))
    ax.annotate(name, xy=(z.real, z.imag),
                xytext=(z.real + offset.real, z.imag + offset.imag),
                fontsize=fontsize, ha="left", va="bottom",
                color=color, fontweight=weight, zorder=10,
                bbox=dict(boxstyle="round,pad=0.18", fc="white",
                          ec=color, lw=0.6, alpha=0.92),
                arrowprops=dict(arrowstyle="-", color=color, lw=0.6))


def main():
    fig, ax = plt.subplots(figsize=(13, 12))
    ax.set_aspect("equal")
    ax.axis("off")

    rot_N = get_rot(RES)

    draw_lattice(ax)

    # --- The cell (0, 2, 5) at canonical position ---
    digits_canon = (2, 5)
    z_canon = _eisenstein_center(digits_canon)   # (0, 0.247i)
    corners = hex_corners_local(z_canon, rot_N)
    xs = [c.real for c in corners] + [corners[0].real]
    ys = [c.imag for c in corners] + [corners[0].imag]
    ax.fill(xs, ys, color="#ffd9d9", alpha=0.85, zorder=6)
    ax.plot(xs, ys, color="#c33", lw=2.2, zorder=6.5)
    ax.plot(z_canon.real, z_canon.imag, "o", color="#c33", markersize=8, zorder=7)
    ax.annotate("(0, 2, 5) canonical", xy=(z_canon.real, z_canon.imag),
                xytext=(z_canon.real - 0.20, z_canon.imag + 0.08),
                fontsize=10, ha="right", va="bottom", color="#c33",
                fontweight="bold", zorder=10,
                bbox=dict(boxstyle="round,pad=0.22", fc="white",
                          ec="#c33", lw=0.8, alpha=0.95),
                arrowprops=dict(arrowstyle="-", color="#c33", lw=0.8))

    # --- The twin (0, 1, 4) at deleted-wedge position ---
    digits_twin = (1, 4)
    z_twin = _eisenstein_center(digits_twin)   # (0.214, 0.124i)
    corners_t = hex_corners_local(z_twin, rot_N)
    xs = [c.real for c in corners_t] + [corners_t[0].real]
    ys = [c.imag for c in corners_t] + [corners_t[0].imag]
    ax.fill(xs, ys, color="#fde7c8", alpha=0.85, zorder=6)
    ax.plot(xs, ys, color="#d96b0f", lw=2.0, ls="--", zorder=6.5)
    ax.plot(z_twin.real, z_twin.imag, "s", color="#d96b0f",
            markersize=9, zorder=7)
    ax.annotate("(0, 1, 4) stitch twin\n(in deleted wedge --\nsame 3D cell)",
                xy=(z_twin.real, z_twin.imag),
                xytext=(z_twin.real + 0.13, z_twin.imag - 0.08),
                fontsize=9, ha="left", va="top", color="#a04500",
                fontweight="bold", zorder=10,
                bbox=dict(boxstyle="round,pad=0.22", fc="white",
                          ec="#d96b0f", lw=0.8, alpha=0.95),
                arrowprops=dict(arrowstyle="-", color="#d96b0f", lw=0.8))

    # +60 deg stitch arrow twin -> canonical.
    ax.annotate("", xytext=(z_twin.real, z_twin.imag),
                xy=(z_canon.real, z_canon.imag),
                arrowprops=dict(arrowstyle="-|>", color="#d96b0f", lw=1.4,
                                ls=":", alpha=0.85, mutation_scale=14),
                zorder=8)
    sx = (z_twin.real + z_canon.real) / 2
    sy = (z_twin.imag + z_canon.imag) / 2
    ax.text(sx + 0.06, sy + 0.012,
            "*(1+omega) = +60 deg stitch",
            fontsize=8, color="#d96b0f", style="italic", zorder=9)

    # --- Walks from canonical (0, 2, 5) --- 4 unique destinations ---
    # Results: D=1->(0,2,6), D=2->(0,2,0), D=3->(0,2,4),
    # D=4->(0,0,2), D=5->(0,0,2)dup, D=6->(0,2,4)dup.
    walks_canon = [
        (1, (0, 2, 6), "D=1"),
        (2, (0, 2, 0), "D=2"),
        (3, (0, 2, 4), "D=3"),
        (4, (0, 0, 2), "D=4"),
        (5, "duplicate (0,0,2)", "D=5"),
        (6, "duplicate (0,2,4)", "D=6"),
    ]
    for D, target, lbl in walks_canon:
        step = digit_offset[D] / rot_N
        z_n = z_canon + step
        ax.annotate("", xytext=(z_canon.real, z_canon.imag),
                    xy=(z_n.real, z_n.imag),
                    arrowprops=dict(arrowstyle="-|>", color="#0a8a3a", lw=1.6,
                                    alpha=0.85, mutation_scale=12),
                    zorder=8)

    # --- Walks from twin (0, 1, 4) ---
    # Results: D=1->dup, D=2->self, D=3->(0,0,2)dup, D=4->(0,0,6) NEW,
    # D=5->(0,6,2) NEW, D=6->(0,2,6)dup.
    walks_twin = [
        (4, (0, 0, 6), "D=4 from twin\n -> (0, 0, 6) NEW"),
        (5, (0, 6, 2), "D=5 from twin\n -> (0, 6, 2) NEW"),
    ]
    for D, target, lbl in walks_twin:
        step = digit_offset[D] / rot_N
        z_n = z_twin + step
        ax.annotate("", xytext=(z_twin.real, z_twin.imag),
                    xy=(z_n.real, z_n.imag),
                    arrowprops=dict(arrowstyle="-|>", color="#9c27b0", lw=2.0,
                                    alpha=0.95, mutation_scale=14),
                    zorder=8.5)

    # --- Highlight (0, 0, 6) --- the missed neighbor ---
    z_006 = _eisenstein_center((0, 6))
    corners_006 = hex_corners_local(z_006, rot_N)
    xs = [c.real for c in corners_006] + [corners_006[0].real]
    ys = [c.imag for c in corners_006] + [corners_006[0].imag]
    ax.fill(xs, ys, color="#cfe1f7", alpha=0.85, zorder=6)
    ax.plot(xs, ys, color="#36c", lw=2.0, zorder=6.5)
    ax.plot(z_006.real, z_006.imag, "o", color="#36c", markersize=8, zorder=7)
    ax.annotate("(0, 0, 6)\nreachable ONLY\nfrom twin position",
                xy=(z_006.real, z_006.imag),
                xytext=(z_006.real + 0.14, z_006.imag - 0.10),
                fontsize=9, ha="left", va="top", color="#36c",
                fontweight="bold", zorder=10,
                bbox=dict(boxstyle="round,pad=0.22", fc="white",
                          ec="#36c", lw=0.8, alpha=0.95),
                arrowprops=dict(arrowstyle="-", color="#36c", lw=0.8))

    # --- Pentagon center label ---
    ax.plot(0, 0, "o", color="black", markersize=10,
            markerfacecolor="#c33", zorder=6)
    ax.text(-0.05, -0.07, "P=0",
            fontsize=11, ha="right", va="top", color="#c33",
            fontweight="bold", zorder=6,
            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                      ec="none", alpha=0.85))

    # Legend.
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#0a8a3a", lw=1.8, marker=">",
               label="walk from canonical (0, 2, 5) at (0, 0.247i)"),
        Line2D([0], [0], color="#9c27b0", lw=2.0, marker=">",
               label="walk from twin (0, 1, 4) at (0.214, 0.124i)"),
        Line2D([0], [0], color="#d96b0f", lw=1.4, ls=":",
               label="+60 deg stitch (twin <-> canonical)"),
        Patch(facecolor="#fde7c8", edgecolor="#d96b0f", linestyle="--",
              alpha=0.5, label="deleted-wedge cells (canonicalized away)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9,
              frameon=True, fancybox=True)

    ax.set_xlim(-0.55, 0.55)
    ax.set_ylim(-0.45, 0.55)

    ax.set_title(
        "Stitch-twin asymmetry: cell (0, 2, 5) has TWO flat representations\n"
        "Walks from canonical pos miss (0, 0, 6); walks from twin pos find it.\n"
        "Symmetry test fails: (0, 0, 6) -> ring1 -> (0, 2, 5), but reverse not via canonical-only walks.",
        fontsize=11,
    )

    out = (Path(__file__).resolve().parent.parent / "figures"
           / "stitch_twin_asymmetry.png")
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
