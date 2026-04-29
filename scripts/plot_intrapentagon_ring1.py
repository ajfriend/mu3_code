"""Intra-pentagon ring-1 study at res 2 -- two-panel layout.

Same 4 cells in P=0 on both panels. Their ring-1 neighbors all sit
within P=0's flat-frame territory, so this isolates the deleted-wedge
handling from any cross-pentagon re-rooting.

Cells (and what makes each interesting):

- ``(0, 2, 5)`` at ``(0, 0.247i)``: sits on the y-axis (the d=1/d=2
  wedge bisector). Two walks land on the deleted-wedge boundary.
- ``(0, 0, 6)`` at ``(0.143, 0)``: pentagon-adjacent (inner ring, d=6
  wedge). Walks brush deleted wedge from below.
- ``(0, 0, 2)`` at ``(-0.071, 0.124i)``: pentagon-adjacent (inner ring,
  d=2 wedge). Walks brush deleted wedge.
- ``(0, 0, 4)`` at ``(-0.071, -0.124i)``: pentagon-adjacent (d=4 wedge,
  opposite from deleted wedge). All 6 walks clean -- control.

Left panel: deleted wedge, walk arrows (color-coded by destination).
Right panel: cell labels at the centers of each colored cell and all
its ring-1 neighbors -- minimal clutter.

Output: ``figures/intrapentagon_ring1.png``.
"""

from __future__ import annotations

import cmath
import itertools
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from mu3.cell import _eisenstein_center
from mu3.face_lattice import digit_offset, get_rot, omega, s3, units


RES = 2
DELETED_HI_RAD = math.pi / 3   # 60 deg


# Test cells: (digits, color).
CELLS = [
    ((2, 5),  "#c33"),     # boundary case (red)
    ((0, 6),  "#36c"),     # pentagon-adjacent d=6 (blue)
    ((0, 2),  "#9c27b0"),  # pentagon-adjacent d=2 (purple)
    ((0, 4),  "#1a7d3c"),  # control, d=4 (green)
]


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


def neighbor_positions(z_C: complex, rot_N: complex):
    """Return list of 6 ``(D, z_n)`` for the walked positions from ``z_C``."""
    return [(D, z_C + digit_offset[D] / rot_N) for D in (1, 2, 3, 4, 5, 6)]


def cell_at_position(z: complex, rot_N: complex):
    """Find the digit string of the lattice cell whose center is at z
    (in flat unfolding). Returns None if not a lattice point.
    """
    for digits, z_local in enumerate_cells(RES):
        if abs(z - z_local) < 1e-6:
            return digits
    return None


def draw_lattice(ax, *, show_deleted_wedge: bool):
    rot_N = get_rot(RES)
    # Wedges + deleted shading.
    for j in range(6):
        a, b, c = 0 + 0j, units[j], units[(j + 1) % 6]
        verts = [(a.real, a.imag), (b.real, b.imag), (c.real, c.imag)]
        if j == 0 and show_deleted_wedge:
            ax.add_patch(Polygon(verts, closed=True, facecolor="#f0a04a",
                                 edgecolor="#d96b0f", linestyle="--",
                                 linewidth=0.7, alpha=0.30, zorder=1))
        else:
            ax.plot([v[0] for v in verts] + [verts[0][0]],
                    [v[1] for v in verts] + [verts[0][1]],
                    color="#bbb", lw=0.4, alpha=0.35, zorder=1)

    for digits, z_local in enumerate_cells(RES):
        if abs(z_local) > 0.55:
            continue
        corners = hex_corners_local(z_local, rot_N)
        xs = [c.real for c in corners] + [corners[0].real]
        ys = [c.imag for c in corners] + [corners[0].imag]

        is_pentagon = all(d == 0 for d in digits)
        is_deleted_child = first_nonzero_digit(digits) == 1

        if is_pentagon:
            ax.plot(xs, ys, color="#c33", lw=1.2, alpha=0.7, zorder=4)
            ax.fill(xs, ys, color="#c33", alpha=0.08, zorder=3)
        elif is_deleted_child and show_deleted_wedge:
            ax.plot(xs, ys, color="#d96b0f", lw=0.4, alpha=0.55,
                    ls="--", zorder=2.5)
            ax.fill(xs, ys, color="#fde7c8", alpha=0.30, zorder=2.4)
        else:
            ax.plot(xs, ys, color="#ccc", lw=0.3, alpha=0.5, zorder=2)


def draw_filled_cell(ax, z_C, rot_N, color, *, alpha=0.45, lw=1.6):
    corners = hex_corners_local(z_C, rot_N)
    xs = [c.real for c in corners] + [corners[0].real]
    ys = [c.imag for c in corners] + [corners[0].imag]
    ax.fill(xs, ys, color=color, alpha=alpha, zorder=6)
    ax.plot(xs, ys, color=color, lw=lw, zorder=6.5)
    ax.plot(z_C.real, z_C.imag, "o", color=color, markersize=5, zorder=7)


def draw_neighbor_outline(ax, z_C, rot_N, color):
    """Light outline for a neighbor cell (not filled)."""
    corners = hex_corners_local(z_C, rot_N)
    xs = [c.real for c in corners] + [corners[0].real]
    ys = [c.imag for c in corners] + [corners[0].imag]
    ax.plot(xs, ys, color=color, lw=1.0, alpha=0.65, zorder=5.5)


def draw_walk_arrows(ax, z_C, rot_N):
    for D, z_n in neighbor_positions(z_C, rot_N):
        a = math.atan2(z_n.imag, z_n.real) % (2 * math.pi) if z_n != 0 else None
        if a is not None and 0 < a < DELETED_HI_RAD - 1e-6:
            arrow_color = "#d96b0f"
            lw = 2.2
        elif a is not None and (
            abs(a - DELETED_HI_RAD) < 1e-6 or abs(a) < 1e-6
            or abs(a - 2 * math.pi) < 1e-6
        ):
            arrow_color = "#f0a04a"
            lw = 2.0
        else:
            arrow_color = "#0a8a3a"
            lw = 1.4
        ax.annotate("", xytext=(z_C.real, z_C.imag),
                    xy=(z_n.real, z_n.imag),
                    arrowprops=dict(arrowstyle="-|>", color=arrow_color,
                                    lw=lw, alpha=0.9, mutation_scale=11),
                    zorder=8)


def draw_cell_label(ax, digits, color, fontsize=8.0):
    z_C = _eisenstein_center(digits)
    text = "(0, " + ", ".join(str(d) for d in digits) + ")"
    ax.text(z_C.real, z_C.imag, text,
            fontsize=fontsize, ha="center", va="center",
            color=color, fontweight="bold", zorder=11,
            bbox=dict(boxstyle="round,pad=0.16", fc="white",
                      ec=color, lw=0.6, alpha=0.95))


def main():
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    rot_N = get_rot(RES)

    for ax, show_deleted in zip(axes, (True, False)):
        ax.set_aspect("equal")
        ax.axis("off")
        draw_lattice(ax, show_deleted_wedge=show_deleted)

        # Pentagon center marker (both panels).
        ax.plot(0, 0, "o", color="black", markersize=8,
                markerfacecolor="#c33", zorder=12)

        # Filled colored cells (both panels).
        for digits, color in CELLS:
            z_C = _eisenstein_center(digits)
            draw_filled_cell(ax, z_C, rot_N, color)

        if show_deleted:
            # LEFT: walk arrows.
            for digits, color in CELLS:
                z_C = _eisenstein_center(digits)
                draw_walk_arrows(ax, z_C, rot_N)
        else:
            # RIGHT: cell labels at centers (colored cells + all their
            # ring-1 neighbor positions that ARE valid lattice cells).
            seen_label_positions = set()

            def label_if_unseen(digits, color):
                key = digits
                if key in seen_label_positions:
                    return
                seen_label_positions.add(key)
                draw_cell_label(ax, digits, color)

            for digits, color in CELLS:
                # Label the colored cell itself.
                label_if_unseen(digits, color)
                # Outline + label each of its ring-1 neighbor cells (the
                # ones that exist in the lattice at the walked position --
                # i.e., not phantoms).
                z_C = _eisenstein_center(digits)
                for D, z_n in neighbor_positions(z_C, rot_N):
                    nb_digits = cell_at_position(z_n, rot_N)
                    if nb_digits is None:
                        continue
                    if nb_digits == tuple([0] * RES):
                        # Pentagon center, draw label as well.
                        pass
                    draw_neighbor_outline(
                        ax, _eisenstein_center(nb_digits), rot_N, color)
                    label_if_unseen(nb_digits, "#222")

        ax.set_xlim(-0.55, 0.55)
        ax.set_ylim(-0.50, 0.55)

    axes[0].set_title(
        "Walks from each cell (P=0, res 2)\n"
        "Green = clean; orange = lands on deleted-wedge boundary; "
        "dark orange = lands INSIDE deleted wedge",
        fontsize=10,
    )
    axes[1].set_title(
        "Cell labels at every relevant position\n"
        "Colored cells + all their ring-1 destinations",
        fontsize=10,
    )

    fig.suptitle(
        "Intra-pentagon ring-1 study (P=0, res 2). "
        "All 4 cells have ring-1 entirely inside P=0 -- isolates the "
        "deleted-wedge handling from cross-pentagon re-rooting.",
        fontsize=12, y=1.02,
    )

    out = (Path(__file__).resolve().parent.parent / "figures"
           / "intrapentagon_ring1.png")
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
