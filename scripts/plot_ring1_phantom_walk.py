"""Diagnostic: why the D=6 walk from (0, 2) at res 1 doesn't reach (0, 6).

Both cells live in pentagon P=0 at res 1, on the ring of 5 pentagon-adjacent
cells around the pentagon center. They're CCW-adjacent on that ring, so
``(0, 6)`` is a real ring-1 neighbor of ``(0, 2)``. But:

- The lattice unit step is ``1 / get_rot(1) = 1 / sqrt(7) ~ 0.378``.
- The flat distance from ``(0, 2)`` at angle 120 deg to ``(0, 6)`` at angle 0
  deg is ``sqrt(3) / sqrt(7) ~ 0.655`` -- not a unit step.
- The D=6 walk lands at ``z_n = (1+omega) / (3+omega)`` at angle 40.9 deg,
  inside (0,)'s deleted wedge ``[0 deg, 60 deg)``.
- The +60 deg intra-pentagon stitch maps that ``z_n`` back to (0, 2)'s own
  center -- a self-loop, not a new neighbor.

So a single lattice walk can't reach (0, 6); the existing twin-rotation
disambiguation is what recovers it.

Output: ``figures/ring1_phantom_walk.png``.
"""

import cmath
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon

from mu3.cell import _eisenstein_center
from mu3.face_lattice import digit_offset, get_rot, omega, s3, units


P = 0
RES = 1


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
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.set_aspect("equal")
    ax.axis("off")

    rot_N = get_rot(RES)

    # ----- 6 unit triangles around pentagon P=0; deleted wedge shaded.
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
            ax.text(cx, cy, "deleted\nwedge", ha="center", va="center",
                    fontsize=9, color="#555", style="italic", zorder=1.5)
        else:
            ax.plot([v[0] for v in verts] + [verts[0][0]],
                    [v[1] for v in verts] + [verts[0][1]],
                    color="black", lw=0.8, alpha=0.6, zorder=1)

    # ----- Pentagon center + 5 neighbor pentagon centers.
    NEIGHBOR_ANGLE_DEG = (0, 120, 180, 240, 300)
    ax.plot(0, 0, "o", color="black", markerfacecolor="#c33",
            markersize=11, zorder=6)
    ax.text(-0.04, -0.04, "(0,)\npentagon center",
            ha="right", va="top", fontsize=10, color="#c33",
            fontweight="bold", zorder=6)
    for ang in NEIGHBOR_ANGLE_DEG:
        c = cmath.exp(1j * math.radians(ang))
        ax.plot(c.real, c.imag, "o", color="#666", markersize=6, zorder=4)

    # ----- 5 res-1 cells around the pentagon (the pentagon-adjacent ring).
    res1_cells = {(P, d): _eisenstein_center((d,)) for d in (2, 3, 4, 5, 6)}
    for (b, d), zc in res1_cells.items():
        is_subject = (d == 2)
        is_target = (d == 6)
        if is_subject:
            face = "#ffd9d9"
            edge = "#c33"
            label = f"({b}, {d})\nsource"
            label_color = "#c33"
        elif is_target:
            face = "#cfe9ff"
            edge = "#36c"
            label = f"({b}, {d})\nactual ring-1 nbr\n(missed by walk)"
            label_color = "#36c"
        else:
            face = "#eee"
            edge = "#777"
            label = f"({b}, {d})"
            label_color = "#777"
        draw_hex(ax, zc, rot_N, edge_color=edge, face_color=face,
                 lw=1.6 if is_subject or is_target else 1.0,
                 alpha=0.85 if is_subject or is_target else 0.55,
                 zorder=5 if is_subject or is_target else 3)
        ax.plot(zc.real, zc.imag, ".", color=edge, markersize=5, zorder=5.5)
        # nudge label outward radially
        nudge = 0.18 * cmath.exp(1j * cmath.phase(zc))
        ax.text(zc.real + nudge.real, zc.imag + nudge.imag, label,
                ha="center", va="center", fontsize=9.5, color=label_color,
                fontweight="bold" if (is_subject or is_target) else "normal",
                zorder=7,
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec=edge, lw=0.7, alpha=0.95))

    # ----- The D=6 walk from (0, 2).
    z_C = res1_cells[(P, 2)]
    step = digit_offset[6] / rot_N
    z_n = z_C + step

    ax.annotate("", xytext=(z_C.real, z_C.imag),
                xy=(z_n.real, z_n.imag),
                arrowprops=dict(arrowstyle="-|>", color="#0a8a3a", lw=2.5,
                                mutation_scale=20),
                zorder=8)
    midx = (z_C.real + z_n.real) / 2
    midy = (z_C.imag + z_n.imag) / 2
    ax.text(midx + 0.04, midy + 0.04,
            f"D=6 walk\n(unit step 1/√7 ≈ {abs(step):.3f})",
            ha="left", va="bottom", fontsize=9, color="#0a8a3a",
            fontweight="bold", zorder=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                      ec="#0a8a3a", lw=0.6, alpha=0.95))

    # ----- z_n marker (in deleted wedge).
    ax.plot(z_n.real, z_n.imag, "X", color="black", markersize=14,
            markeredgewidth=2.0, zorder=10)
    angle_deg = math.degrees(cmath.phase(z_n)) % 360.0
    ax.annotate(
        f"z_n at {angle_deg:.1f}° in deleted wedge\n"
        f"|z_n| = {abs(z_n):.3f}",
        xy=(z_n.real, z_n.imag),
        xytext=(z_n.real + 0.45, z_n.imag - 0.30),
        fontsize=9, ha="left", va="top", color="black", zorder=11,
        bbox=dict(boxstyle="round,pad=0.3", fc="#fff3c4",
                  ec="black", lw=0.8, alpha=0.95),
        arrowprops=dict(arrowstyle="-", color="black", lw=0.8),
    )

    # ----- Stitch: z_n * (1 + omega) lands back on (0, 2)'s center.
    z_stitched = z_n * (1 + omega)
    ax.annotate("", xytext=(z_n.real, z_n.imag),
                xy=(z_stitched.real, z_stitched.imag),
                arrowprops=dict(arrowstyle="-|>", color="#d96b0f", lw=2.0,
                                ls=":", alpha=0.9, mutation_scale=18),
                zorder=8)
    sx = (z_n.real + z_stitched.real) / 2
    sy = (z_n.imag + z_stitched.imag) / 2
    ax.text(sx + 0.05, sy + 0.0,
            "+60° stitch\n× (1+ω)",
            ha="left", va="center", fontsize=9, color="#d96b0f",
            style="italic", fontweight="bold", zorder=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                      ec="#d96b0f", lw=0.6, alpha=0.95))

    # ----- Self-loop indicator at (0, 2).
    ax.plot(z_stitched.real, z_stitched.imag, "o",
            markerfacecolor="none", markeredgecolor="#d96b0f",
            markersize=22, markeredgewidth=2.2, zorder=11)

    # ----- Direct line (0, 2) -> (0, 6) showing the actual ring edge in flat.
    z6 = res1_cells[(P, 6)]
    direct_dist = abs(z6 - z_C)
    ax.annotate("", xytext=(z_C.real, z_C.imag),
                xy=(z6.real, z6.imag),
                arrowprops=dict(arrowstyle="<|-|>", color="#36c", lw=1.6,
                                ls=(0, (4, 3)), alpha=0.7,
                                mutation_scale=14),
                zorder=7)
    mx = (z_C.real + z6.real) / 2
    my = (z_C.imag + z6.imag) / 2
    ax.text(mx, my - 0.08,
            f"flat dist {direct_dist:.3f}\n= √3/√7\n(not a unit step)",
            ha="center", va="top", fontsize=9, color="#36c",
            zorder=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                      ec="#36c", lw=0.5, alpha=0.95))

    # ----- View bounds.
    R = 1.35
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)

    ax.set_title(
        "Why the D=6 lattice walk from (0, 2) misses (0, 6) at res 1.\n"
        "Walk lands in (0,)'s deleted wedge; the +60° stitch sends it back to "
        "(0, 2) — a self-loop.\n"
        "(0, 6) is at flat distance √3/√7, not reachable by any single unit "
        "walk in the 6 lattice directions.",
        fontsize=10,
    )

    out = (Path(__file__).resolve().parent.parent / "figures"
           / "ring1_phantom_walk.png")
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
