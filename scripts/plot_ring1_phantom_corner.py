"""Diagnostic: the D=1 walk from (0, 6, 0) at res 2 lands on a 3-cell
corner. Three lattice positions claim the same 3D point: the expected
ring-1 neighbor (0, 6, 1), the wrong-side neighbor (0, 5, 6), and the
deleted-wedge phantom (0, 1, 2). The atlas (project + index back) alone
can't tell which is the right ring-1 neighbor without an extra tiebreaker.

Output: ``figures/ring1_phantom_corner.png``.
"""

import cmath
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from mu3.cell import _eisenstein_center
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


def main():
    fig, ax = plt.subplots(figsize=(13, 13))
    ax.set_aspect("equal")
    ax.axis("off")

    rot1 = get_rot(1)
    rot2 = get_rot(2)

    # ----- 6 unit triangles around pentagon P=0; deleted wedge shaded.
    for j in range(6):
        a, b, c = 0 + 0j, units[j], units[(j + 1) % 6]
        verts = [(a.real, a.imag), (b.real, b.imag), (c.real, c.imag)]
        if j == 0:
            ax.add_patch(Polygon(
                verts, closed=True, facecolor="#888",
                edgecolor="#555", linestyle="--", lw=0.7,
                alpha=0.22, zorder=1,
            ))
            cx = (a.real + b.real + c.real) / 3
            cy = (a.imag + b.imag + c.imag) / 3
            ax.text(cx, cy + 0.07, "deleted wedge\n(phantom flat region)",
                    ha="center", va="center", fontsize=8.5, color="#555",
                    style="italic", zorder=1.5)
        else:
            ax.plot([v[0] for v in verts] + [verts[0][0]],
                    [v[1] for v in verts] + [verts[0][1]],
                    color="black", lw=0.7, alpha=0.55, zorder=1)

    # ----- res-1 cells around pentagon (0,).
    res1_subjects = {2: ("d=2 wedge", "#aaa"),
                      3: ("d=3 wedge", "#aaa"),
                      4: ("d=4 wedge", "#aaa"),
                      5: ("d=5 wedge", "#36c"),
                      6: ("d=6 wedge", "#c33")}
    for d, (lbl, col) in res1_subjects.items():
        z = digit_offset[d] / rot1
        draw_hex(ax, z, rot1, edge_color=col, face_color="white",
                 alpha=0.0, lw=1.5 if d in (5, 6) else 0.8, zorder=2)

    # ----- res-2 sub-cells inside (0, 6).
    z_06 = digit_offset[6] / rot1
    for d2 in range(7):
        if d2 == 0:
            z = z_06
        else:
            z = z_06 + digit_offset[d2] / rot2
        # Note (0, 6, 1) is a phantom-cell representation that we'll
        # highlight differently below; here we draw it lightly.
        is_source = (d2 == 0)
        is_target = (d2 == 1)
        if is_source:
            draw_hex(ax, z, rot2, edge_color="#c33",
                     face_color="#ffd9d9", lw=2.0, alpha=0.85, zorder=5)
            ax.text(z.real, z.imag, "(0,6,0)\nsource",
                    ha="center", va="center", fontsize=9.5, color="#c33",
                    fontweight="bold", zorder=6)
        elif is_target:
            draw_hex(ax, z, rot2, edge_color="#0a8a3a",
                     face_color="#d6f4dd", lw=2.0, alpha=0.85, zorder=5)
            ax.text(z.real, z.imag, "(0,6,1)\nexpected\nring-1 nbr",
                    ha="center", va="center", fontsize=8.5, color="#0a8a3a",
                    fontweight="bold", zorder=6)
        else:
            draw_hex(ax, z, rot2, edge_color="#666",
                     face_color="#f4f4f4", lw=0.7, alpha=0.6, zorder=3)
            ax.text(z.real, z.imag, f"(0,6,{d2})",
                    ha="center", va="center", fontsize=7.5, color="#666",
                    zorder=4)

    # ----- res-2 sub-cells inside (0, 5) -- the "wrong side" cell.
    z_05 = digit_offset[5] / rot1
    z_056 = z_05 + digit_offset[6] / rot2
    draw_hex(ax, z_056, rot2, edge_color="#d96b0f",
             face_color="#fde7c8", lw=2.0, alpha=0.85, zorder=5)
    ax.text(z_056.real, z_056.imag, "(0,5,6)\nwrong-side\nneighbor",
            ha="center", va="center", fontsize=8.5, color="#d96b0f",
            fontweight="bold", zorder=6)

    # Outline (0, 5) lightly.
    draw_hex(ax, z_05, rot1, edge_color="#36c", lw=1.0,
             face_color=None, zorder=2)

    # ----- The phantom hex (0, 1, 2) -- in the deleted wedge.
    z_01 = digit_offset[1] / rot1
    z_012 = z_01 + digit_offset[2] / rot2
    draw_hex(ax, z_012, rot2, edge_color="#a26ec3",
             face_color="#e9d5f2", lw=2.0, ls="--", alpha=0.65, zorder=5)
    ax.text(z_012.real, z_012.imag, "(0,1,2)\nphantom\n(invalid cell)",
            ha="center", va="center", fontsize=8, color="#7a3ea0",
            fontweight="bold", style="italic", zorder=6)

    # ----- The walk D=1 from (0, 6, 0).
    z_C = z_06
    step = digit_offset[1] / rot2
    z_n = z_C + step
    ax.annotate("", xytext=(z_C.real, z_C.imag),
                xy=(z_n.real, z_n.imag),
                arrowprops=dict(arrowstyle="-|>", color="#0a8a3a",
                                lw=2.5, mutation_scale=20),
                zorder=8)
    midx = (z_C.real + z_n.real) / 2
    midy = (z_C.imag + z_n.imag) / 2
    ax.text(midx + 0.04, midy - 0.04, "D=1 walk",
            ha="left", va="top", fontsize=10, color="#0a8a3a",
            fontweight="bold", zorder=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                      ec="#0a8a3a", lw=0.6, alpha=0.95))

    # ----- z_n marker.
    ax.plot(z_n.real, z_n.imag, "X", color="black", markersize=16,
            markeredgewidth=2.2, zorder=10)
    angle_deg = math.degrees(cmath.phase(z_n)) % 360.0
    ax.annotate(
        f"z_n at {angle_deg:.1f}° (deleted-wedge boundary)\n"
        f"|z_n| ≈ {abs(z_n):.3f}\n"
        "3D point is on the V[0]–V[2] icosa edge:\n"
        "shared by (0,6,1), (0,5,6), and the\n"
        "(0,1,2) phantom -- all three claim it.",
        xy=(z_n.real, z_n.imag),
        xytext=(z_n.real + 0.32, z_n.imag - 0.30),
        fontsize=9, ha="left", va="top", color="black", zorder=11,
        bbox=dict(boxstyle="round,pad=0.3", fc="#fff3c4",
                  ec="black", lw=0.8, alpha=0.95),
        arrowprops=dict(arrowstyle="-", color="black", lw=0.8),
    )

    # ----- Pentagon center.
    ax.plot(0, 0, "o", color="black", markerfacecolor="#888",
            markersize=9, zorder=6)
    ax.text(-0.04, -0.05, "pentagon (0,)",
            ha="right", va="top", fontsize=9, color="#444",
            zorder=6)

    # ----- View bounds: zoom on the (0, 6) ↔ deleted wedge ↔ (0, 5) area.
    R = 0.95
    cx = 0.45
    cy = -0.05
    ax.set_xlim(cx - R, cx + R)
    ax.set_ylim(cy - R, cy + R)

    ax.set_title(
        "D=1 walk from (0, 6, 0) at res 2 lands on a 3-cell corner.\n"
        "The 3D point is shared by (0,6,1) [expected], (0,5,6) [wrong-side], "
        "and the (0,1,2) phantom.\n"
        "Atlas project-and-index alone cannot disambiguate without an "
        "extra tiebreaker (e.g. digit-rotation twin pick).",
        fontsize=10,
    )

    out = (Path(__file__).resolve().parent.parent / "figures"
           / "ring1_phantom_corner.png")
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
