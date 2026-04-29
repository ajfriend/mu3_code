"""Diagram of the obstacle in the geometric ring-1 walk.

Walking from cell ``(0, 2)`` in direction D=1 at res 1 lands at a position
that is a 3-hex corner -- shared by ``(0, 2)``, the deleted ``(0, 1)``,
and ``(5, 4)`` -- not at any cell center. ``z_to_cell`` cannot recover
digits from a corner, which is what blew up the pure geometric
``cell_ring1``.

This figure shows the case at res 1 in the joint flat frame of pentagons
P=0 and Q=5 (k=1, j=2, joint rotation 120 deg). The walk vector is drawn
from (0, 2)'s center; the destination is annotated as the 3-cell corner.
"""

from __future__ import annotations

import cmath
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from mu3 import dodec
from mu3.cell import _eisenstein_center
from mu3.face_lattice import digit_offset, get_rot, omega, s3, units


P = 0
K_INDEX = 1
Q = dodec.neighbors[P][K_INDEX]   # = 5
J_INDEX = list(dodec.neighbors[Q]).index(P)
NEIGHBOR_ANGLE_DEG = (0.0, 120.0, 180.0, 240.0, 300.0)
ALPHA_P = math.radians(NEIGHBOR_ANGLE_DEG[K_INDEX])
ALPHA_Q = math.radians(NEIGHBOR_ANGLE_DEG[J_INDEX])
Q_CENTER = cmath.exp(1j * ALPHA_P)
Q_LOCAL_TO_JOINT_ROT = cmath.exp(1j * (ALPHA_P + math.pi - ALPHA_Q))


def to_joint(z_local: complex, which: str) -> complex:
    if which == "p":
        return z_local
    return Q_CENTER + Q_LOCAL_TO_JOINT_ROT * z_local


def hex_corners_local(center_local: complex, rot: complex) -> list[complex]:
    return [center_local + units[k] / (s3 * rot) for k in range(6)]


def draw_hex(ax, center_joint, rot_l2j, rot_N, *, fill, edge,
             edge_lw=0.7, fill_alpha=0.4, zorder=2, dashed=False):
    corners_local = [center_local_dummy / 1.0  # placeholder
                     for center_local_dummy in [0]]
    # Actually compute corners in local then map to joint.
    corners_local = [units[k] / (s3 * rot_N) for k in range(6)]
    corners_joint = [center_joint + rot_l2j * c for c in corners_local]
    xs = [c.real for c in corners_joint] + [corners_joint[0].real]
    ys = [c.imag for c in corners_joint] + [corners_joint[0].imag]
    ls = "--" if dashed else "-"
    ax.fill(xs, ys, color=fill, alpha=fill_alpha, zorder=zorder)
    ax.plot(xs, ys, color=edge, lw=edge_lw, ls=ls, zorder=zorder + 0.5)


def main():
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.set_aspect("equal")
    ax.axis("off")

    rot_N = get_rot(1)

    # ----- Pentagon outlines (level-0 wedges) -----
    DELETED_J = 0  # deleted wedge spans local angles [0°, 60°)
    for which, center_joint, rot_l2j, color, deleted_color in [
        ("p", 0 + 0j, 1 + 0j, "#c33", "#f0a04a"),
        ("q", Q_CENTER, Q_LOCAL_TO_JOINT_ROT, "#36c", "#9bc4f0"),
    ]:
        for j in range(6):
            a_l, b_l, c_l = 0 + 0j, units[j], units[(j + 1) % 6]
            a, b, c = (to_joint(a_l, which), to_joint(b_l, which),
                       to_joint(c_l, which))
            verts = [(a.real, a.imag), (b.real, b.imag), (c.real, c.imag)]
            if j == DELETED_J:
                ax.add_patch(Polygon(verts, closed=True,
                                     facecolor=deleted_color, alpha=0.30,
                                     edgecolor=deleted_color, linestyle="--",
                                     linewidth=0.8, zorder=1))
            else:
                ax.plot([v[0] for v in verts] + [verts[0][0]],
                        [v[1] for v in verts] + [verts[0][1]],
                        color=color, lw=0.6, alpha=0.5, zorder=1)

    # ----- Res-1 cells of P=0 -----
    p_cell_centers_local = {d: digit_offset[d] / rot_N for d in range(7)}
    p_cell_centers_local[0] = 0 + 0j

    # Real cells (d=0..6, with d=1 marked deleted/phantom).
    for d in range(7):
        z_l = p_cell_centers_local[d]
        z_j = to_joint(z_l, "p")
        is_deleted = (d == 1)
        is_target = (d == 2)
        fill = "#fde7c8" if is_deleted else ("#ffd9d9" if is_target else "#f5f0eb")
        edge = "#d96b0f" if is_deleted else ("#c33" if is_target else "#888")
        edge_lw = 1.6 if is_target else (1.0 if is_deleted else 0.5)
        draw_hex(ax, z_j, 1 + 0j, rot_N,
                 fill=fill, edge=edge, edge_lw=edge_lw,
                 fill_alpha=0.55, zorder=2 if is_target else 1.5,
                 dashed=is_deleted)
        # Center dot + label.
        ax.plot(z_j.real, z_j.imag, ".",
                color=("#d96b0f" if is_deleted else "k"),
                markersize=4, zorder=4)
        if is_deleted:
            ax.text(z_j.real, z_j.imag - 0.05, "(0, 1)\nDELETED",
                    fontsize=8, ha="center", va="top",
                    color="#d96b0f", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white",
                              ec="#d96b0f", lw=0.6, alpha=0.95))
        else:
            label = f"(0, {d})" if d > 0 else "(0,)"
            color = "#c33" if is_target else "#222"
            weight = "bold" if is_target else "normal"
            ax.text(z_j.real + 0.012, z_j.imag - 0.04, label,
                    fontsize=8, ha="left", va="top", color=color,
                    fontweight=weight, zorder=5)

    # ----- Res-1 cells of Q=5 (only those near the joint corner) -----
    for d in range(7):
        z_l = p_cell_centers_local[d]
        z_j = to_joint(z_l, "q")
        # Only draw cells inside the view.
        if abs(z_j - 0.0) > 1.4 and abs(z_j - Q_CENTER) > 0.7:
            continue
        is_target = (d == 4)
        is_deleted_q = (d == 1)
        fill = "#fde7c8" if is_deleted_q else ("#cfe1f7" if is_target else "#eaf2ff")
        edge = "#d96b0f" if is_deleted_q else ("#36c" if is_target else "#88a")
        edge_lw = 1.6 if is_target else (1.0 if is_deleted_q else 0.5)
        draw_hex(ax, z_j, 1 + 0j, rot_N,
                 fill=fill, edge=edge, edge_lw=edge_lw,
                 fill_alpha=0.55, zorder=2 if is_target else 1.5,
                 dashed=is_deleted_q)
        ax.plot(z_j.real, z_j.imag, ".",
                color=("#d96b0f" if is_deleted_q else "k"),
                markersize=4, zorder=4)
        if is_deleted_q:
            ax.text(z_j.real, z_j.imag - 0.05, "(5, 1)\nDELETED",
                    fontsize=7, ha="center", va="top",
                    color="#d96b0f", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white",
                              ec="#d96b0f", lw=0.6, alpha=0.9))
        else:
            label = f"(5, {d})" if d > 0 else "(5,)"
            color = "#36c" if is_target else "#446"
            weight = "bold" if is_target else "normal"
            ax.text(z_j.real - 0.012, z_j.imag + 0.04, label,
                    fontsize=8, ha="right", va="bottom", color=color,
                    fontweight=weight, zorder=5)

    # ----- Walk arrow from (0, 2) in direction D=1 -----
    z_C = digit_offset[2] / rot_N
    z_n = z_C + digit_offset[1] / rot_N

    ax.annotate("", xytext=(z_C.real, z_C.imag),
                xy=(z_n.real, z_n.imag),
                arrowprops=dict(arrowstyle="-|>", color="#0a8a3a", lw=2.2,
                                mutation_scale=18),
                zorder=8)
    midx = (z_C.real + z_n.real) / 2
    midy = (z_C.imag + z_n.imag) / 2
    ax.text(midx + 0.04, midy + 0.06,
            "step: D=1\n(deleted dir)", fontsize=9, ha="left", va="bottom",
            color="#0a8a3a", fontweight="bold", zorder=8,
            bbox=dict(boxstyle="round,pad=0.22", fc="white",
                      ec="#0a8a3a", lw=0.7, alpha=0.95))

    # ----- The 3-cell corner (the obstacle) -----
    ax.plot(z_n.real, z_n.imag, "X", color="black",
            markersize=14, markeredgewidth=2.0, zorder=9)
    ax.text(z_n.real + 0.06, z_n.imag,
            "z_n: 3-hex corner\n"
            "shared by (0,2), (0,1)-deleted, (5,4)\n"
            "→ NOT a cell center;\n"
            "z_to_cell can't recover digits",
            fontsize=9, ha="left", va="center",
            color="black", zorder=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="#fff3c4",
                      ec="black", lw=0.8, alpha=0.95))

    # Pentagon centers + labels.
    for which, ctr, color, txt in [
        ("p", 0 + 0j, "#c33", f"P=0 (center)"),
        ("q", Q_CENTER, "#36c", f"Q=5 (center)"),
    ]:
        ax.plot(ctr.real, ctr.imag, "o", color="black",
                markersize=8, markerfacecolor=color, zorder=6)
        ax.text(ctr.real, ctr.imag - 0.10, txt, fontsize=10,
                ha="center", va="top", color=color, fontweight="bold",
                zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="none", alpha=0.85))

    # Shared icosa edge.
    ax.plot([0, Q_CENTER.real], [0, Q_CENTER.imag],
            color="#222", lw=1.6, alpha=0.5, zorder=2,
            solid_capstyle="round")

    view_center = 0.5 * (0 + Q_CENTER)
    view_radius = 0.95
    ax.set_xlim(view_center.real - view_radius, view_center.real + view_radius + 0.1)
    ax.set_ylim(view_center.imag - view_radius, view_center.imag + view_radius)

    ax.set_title(
        "The deleted-direction obstacle: D=1 from (0, 2) at res 1 lands at a 3-hex corner\n"
        "(0, 1) is the deleted wedge of pentagon 0; the walk's destination is on its phantom border\n"
        "→ z_to_cell can't recover digits directly; phantom-twin disambiguation resolves it",
        fontsize=11,
    )

    out = (Path(__file__).resolve().parent.parent / "figures"
           / "deleted_direction_obstacle.png")
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
