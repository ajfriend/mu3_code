"""Show both CCW and CW twins of the phantoms from (0, 2, 6, 6) D=1, D=5, D=6.

Each of the 3 phantom positions has TWO Eisenstein-rotation twins
around p=0's center:

- CCW twin (multiply position by 1+omega): rotates +60 deg around p=0.
- CW twin (multiply position by -omega): rotates -60 deg around p=0.

Both twins are valid (non-deleted-form) digit strings AND both are
children of pentagon 0 (base = 0). But neither sits at unit step from
the source (0, 2, 6, 6).

Output: ``figures/phantom_both_twins.png``.
"""

from __future__ import annotations

import cmath
import itertools
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from mu3.cell import _eisenstein_center
from mu3.face_lattice import digit_offset, get_rot, omega, s3, units, rotate_digit_ccw


RES = 3
DELETED_HI_RAD = math.pi / 3


def first_nonzero_digit(digits):
    for d in digits:
        if d != 0:
            return d
    return None


def enumerate_cells(res):
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


def hex_corners_local(center, rot):
    return [center + units[k] / (s3 * rot) for k in range(6)]


def draw_lattice(ax):
    rot_N = get_rot(RES)
    for j in range(6):
        a, b, c = 0 + 0j, units[j], units[(j + 1) % 6]
        verts = [(a.real, a.imag), (b.real, b.imag), (c.real, c.imag)]
        if j == 0:
            ax.add_patch(Polygon(verts, closed=True, facecolor="#f0a04a",
                                 edgecolor="#d96b0f", linestyle="--",
                                 linewidth=0.7, alpha=0.30, zorder=1))
        else:
            ax.plot([v[0] for v in verts] + [verts[0][0]],
                    [v[1] for v in verts] + [verts[0][1]],
                    color="#bbb", lw=0.4, alpha=0.4, zorder=1)

    for digits, z_local in enumerate_cells(RES):
        if abs(z_local) > 0.5:
            continue
        corners = hex_corners_local(z_local, rot_N)
        xs = [c.real for c in corners] + [corners[0].real]
        ys = [c.imag for c in corners] + [corners[0].imag]

        is_pentagon = all(d == 0 for d in digits)
        is_deleted_child = first_nonzero_digit(digits) == 1

        if is_pentagon:
            ax.plot(xs, ys, color="#c33", lw=1.2, alpha=0.7, zorder=4)
            ax.fill(xs, ys, color="#c33", alpha=0.10, zorder=3)
        elif is_deleted_child:
            ax.plot(xs, ys, color="#d96b0f", lw=0.4, alpha=0.5,
                    ls="--", zorder=2.5)
            ax.fill(xs, ys, color="#fde7c8", alpha=0.25, zorder=2.4)
        else:
            ax.plot(xs, ys, color="#ccc", lw=0.3, alpha=0.4, zorder=2)


def draw_filled_cell(ax, z_C, rot_N, color, *, alpha=0.55, lw=2.0):
    corners = hex_corners_local(z_C, rot_N)
    xs = [c.real for c in corners] + [corners[0].real]
    ys = [c.imag for c in corners] + [corners[0].imag]
    ax.fill(xs, ys, color=color, alpha=alpha, zorder=6)
    ax.plot(xs, ys, color=color, lw=lw, zorder=6.5)
    ax.plot(z_C.real, z_C.imag, "o", color=color, markersize=5, zorder=7)


def main():
    fig, ax = plt.subplots(figsize=(14, 13))
    ax.set_aspect("equal")
    ax.axis("off")
    rot_N = get_rot(RES)

    SOURCE = (2, 6, 6)
    z_source = _eisenstein_center(SOURCE)

    draw_lattice(ax)

    # Pentagon center.
    ax.plot(0, 0, "o", color="black", markersize=10,
            markerfacecolor="#c33", zorder=12)
    ax.text(-0.02, -0.05, "P=0", fontsize=11, ha="right", va="top",
            color="#c33", fontweight="bold", zorder=12)

    # Source cell.
    draw_filled_cell(ax, z_source, rot_N, "#c33")
    ax.text(z_source.real, z_source.imag, "(0, 2, 6, 6)\nsource",
            fontsize=8, ha="center", va="center", color="#c33",
            fontweight="bold", zorder=13,
            bbox=dict(boxstyle="round,pad=0.16", fc="white",
                      ec="#c33", lw=0.6, alpha=0.95))

    # Walk arrows to phantoms + draw both twins.
    phantom_walks = [(1, "D=1"), (5, "D=5"), (6, "D=6")]
    for D, label in phantom_walks:
        step = digit_offset[D] / rot_N
        z_n = z_source + step

        # Walk arrow to phantom.
        ax.annotate("", xytext=(z_source.real, z_source.imag),
                    xy=(z_n.real, z_n.imag),
                    arrowprops=dict(arrowstyle="-|>", color="#0a8a3a",
                                    lw=1.6, alpha=0.85, mutation_scale=11),
                    zorder=8)
        ax.plot(z_n.real, z_n.imag, "X", color="#d96b0f",
                markersize=10, markeredgewidth=1.8, zorder=8.5)

        # Find phantom digits.
        # divmod-extract via the existing _z_to_cell_with_residue.
        from mu3.neighbor import _z_to_cell_with_residue
        phantom_digits, _ = _z_to_cell_with_residue(0, z_n, RES)
        phantom_digits = tuple(phantom_digits)

        # CCW twin.
        ccw_digits = tuple(rotate_digit_ccw(d, 1) for d in phantom_digits)
        z_ccw = _eisenstein_center(ccw_digits)
        draw_filled_cell(ax, z_ccw, rot_N, "#9c27b0", alpha=0.5)
        ax.text(z_ccw.real, z_ccw.imag,
                "(0, " + ", ".join(str(d) for d in ccw_digits) + ")\nCCW twin",
                fontsize=7, ha="center", va="center", color="#9c27b0",
                fontweight="bold", zorder=11,
                bbox=dict(boxstyle="round,pad=0.14", fc="white",
                          ec="#9c27b0", lw=0.5, alpha=0.95))
        # Dotted arrow phantom -> CCW twin.
        ax.annotate("", xytext=(z_n.real, z_n.imag),
                    xy=(z_ccw.real, z_ccw.imag),
                    arrowprops=dict(arrowstyle="-|>", color="#9c27b0",
                                    lw=1.0, ls=":", alpha=0.6,
                                    mutation_scale=10),
                    zorder=7.5)

        # CW twin.
        cw_digits = tuple(rotate_digit_ccw(d, 5) for d in phantom_digits)
        z_cw = _eisenstein_center(cw_digits)
        draw_filled_cell(ax, z_cw, rot_N, "#0a7d6c", alpha=0.5)
        ax.text(z_cw.real, z_cw.imag,
                "(0, " + ", ".join(str(d) for d in cw_digits) + ")\nCW twin",
                fontsize=7, ha="center", va="center", color="#0a7d6c",
                fontweight="bold", zorder=11,
                bbox=dict(boxstyle="round,pad=0.14", fc="white",
                          ec="#0a7d6c", lw=0.5, alpha=0.95))
        ax.annotate("", xytext=(z_n.real, z_n.imag),
                    xy=(z_cw.real, z_cw.imag),
                    arrowprops=dict(arrowstyle="-|>", color="#0a7d6c",
                                    lw=1.0, ls=":", alpha=0.6,
                                    mutation_scale=10),
                    zorder=7.5)

    # Legend.
    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], color="#0a8a3a", lw=1.6, marker=">",
               label="walk from source to phantom"),
        Line2D([0], [0], color="#d96b0f", lw=0, marker="X",
               markersize=10, label="phantom (leading-zero d=1, no canonical cell)"),
        Line2D([0], [0], color="#9c27b0", lw=1.0, ls=":",
               label="CCW digit rotation (+60 deg around P=0)"),
        Line2D([0], [0], color="#0a7d6c", lw=1.0, ls=":",
               label="CW digit rotation (-60 deg around P=0)"),
    ]
    ax.legend(handles=legend, loc="lower left", fontsize=9, frameon=True)

    ax.set_xlim(-0.50, 0.55)
    ax.set_ylim(-0.30, 0.55)

    ax.set_title(
        "Phantoms from (0, 2, 6, 6) walks D=1, D=5, D=6.\n"
        "Both CCW and CW digit rotations land at children of P=0 "
        "(all start with 0, *, *, *), but FAR from source.",
        fontsize=11,
    )

    out = (Path(__file__).resolve().parent.parent / "figures"
           / "phantom_both_twins.png")
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
