"""Highlight valid mu3 cells (base 0, res 3) whose z_C lands inside the
deleted wedge of pentagon (0,)'s flat frame. These are cells whose
accumulated digit offsets push the center across the [0°, 60°) boundary,
even though the digit string itself is non-phantom.

Output: ``figures/cells_in_deleted.png``.
"""

import cmath
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from mu3 import cells_at_res
from mu3.cell import _eisenstein_center, cell_resolution
from mu3.face_lattice import digit_offset, get_rot, omega, s3, units


def hex_corners(center: complex, rot: complex):
    return [center + units[k] / (s3 * rot) for k in range(6)]


def as_xy(zs):
    return [z.real for z in zs], [z.imag for z in zs]


def draw_hex(ax, center, rot, *, edge_color, face_color=None, lw=1.0,
             alpha=1.0, ls="-", zorder=2):
    corners = hex_corners(center, rot)
    xs, ys = as_xy(corners + [corners[0]])
    if face_color is not None:
        ax.fill(xs, ys, color=face_color, alpha=alpha, zorder=zorder)
    ax.plot(xs, ys, color=edge_color, lw=lw, ls=ls, zorder=zorder + 0.1)


def first_nonzero(digits):
    for d in digits:
        if d != 0:
            return d
    return None


def main():
    fig, ax = plt.subplots(figsize=(13, 13))
    ax.set_aspect("equal")
    ax.axis("off")

    rot3 = get_rot(3)

    # 6 unit triangles, deleted wedge shaded.
    for j in range(6):
        a, b, c = 0 + 0j, units[j], units[(j + 1) % 6]
        verts = [(a.real, a.imag), (b.real, b.imag), (c.real, c.imag)]
        if j == 0:
            ax.add_patch(Polygon(
                verts, closed=True, facecolor="#888",
                edgecolor="#555", linestyle="--", lw=0.8,
                alpha=0.30, zorder=1,
            ))
            cx = (a.real + b.real + c.real) / 3
            cy = (a.imag + b.imag + c.imag) / 3
            ax.text(cx + 0.05, cy + 0.05, "deleted wedge\n[0°, 60°)",
                    ha="center", va="center", fontsize=11, color="#444",
                    style="italic", zorder=1.5)
        else:
            ax.plot([v[0] for v in verts] + [verts[0][0]],
                    [v[1] for v in verts] + [verts[0][1]],
                    color="black", lw=0.7, alpha=0.5, zorder=1)

    # All res-3 cells of pentagon (0,) -- draw light hexes for the bulk;
    # highlight the ones whose z_C is in the deleted wedge.
    in_deleted = []
    for c in cells_at_res(3):
        if c[0] != 0:
            continue
        z = _eisenstein_center(c[1:])
        if abs(z) < 1e-12:
            continue
        ang = math.degrees(cmath.phase(z)) % 360.0
        if 0.0 < ang < 60.0:
            in_deleted.append((c, z, ang))
            draw_hex(ax, z, rot3, edge_color="#c33", face_color="#ffd9d9",
                     lw=1.4, alpha=0.85, zorder=5)
            ax.plot(z.real, z.imag, ".", color="#c33", markersize=3, zorder=6)
        else:
            # Also draw ones in d=2 and d=6 wedges very faintly for context.
            if 60.0 <= ang < 120.0 or 300.0 <= ang < 360.0:
                draw_hex(ax, z, rot3, edge_color="#bbb", face_color=None,
                         lw=0.3, alpha=0.45, zorder=2)

    # Pentagon center marker.
    ax.plot(0, 0, "o", color="black", markerfacecolor="#666",
            markersize=10, zorder=6)
    ax.text(-0.04, -0.05, "(0,) pentagon center",
            ha="right", va="top", fontsize=10, color="#333", zorder=6)

    # Annotate a few examples.
    for c, z, ang in in_deleted[:6]:
        nudge = 0.07 * cmath.exp(1j * cmath.phase(z))
        ax.text(z.real + nudge.real, z.imag + nudge.imag,
                f"{c}\n{ang:.1f}°",
                ha="left", va="center", fontsize=7.5, color="#a01010",
                zorder=8,
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="#c33", lw=0.5, alpha=0.95))

    ax.set_xlim(-0.3, 1.05)
    ax.set_ylim(-0.3, 1.05)

    ax.set_title(
        f"Res-3 valid cells (base=0) whose z_C lands strictly inside "
        f"(0,)'s deleted wedge: {len(in_deleted)} cells.\n"
        "Each has a non-phantom digit string but the cumulative offset crosses "
        "the [0°, 60°) boundary.\n"
        "Faint gray hexes show the surrounding d=2 and d=6 lattice for context.",
        fontsize=10,
    )

    out = (Path(__file__).resolve().parent.parent / "figures"
           / "cells_in_deleted.png")
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}, {len(in_deleted)} cells in deleted wedge")


if __name__ == "__main__":
    main()
