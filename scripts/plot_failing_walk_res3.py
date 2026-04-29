"""Diagnostic: the failing geometric ring-1 walk from (0, 2, 1, 1) D=6 at res 3.

Renders pentagon P=0 and a neighbor (P=5 at 120 deg) in joint flat
frame, with the res-3 cell lattice drawn. Highlights:

- ``(0, 2, 1, 1)`` -- the starting cell (red).
- ``z_n`` -- the walk's destination (black X). Walked from ``(0,2,1,1)``
  in direction ``D=6`` (primary +x at this resolution).
- ``z_C(0, 4, 4, 3)`` -- the digit-extraction "ghost" centre (orange).
  ``z_to_cell(p=0, z_n, res=3)`` returns digits ``(4, 4, 3)`` with
  residue ``1+omega`` at root: ``z_n = z_C(0,4,4,3) + (1+omega)``.
  Note that ``(0, 4, 4, 3)`` itself sits in p=8's territory (240 deg
  side), so it's not a valid canonical cell index for p=0.
- ``z_n * (-omega)`` -- the would-be un-fold target (light gray X).
  Lies in p=0's deleted wedge ``[0, 60) deg``, so canonicalize stitches
  it by ``*(1+omega)``, returning to ``z_n`` -- the un-fold is a no-op.
- The ``+(1+omega)`` shift arrow from ghost to ``z_n``.

Output: ``figures/failing_walk_res3.png``.
"""

import cmath
import itertools
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from mu3 import dodec
from mu3.cell import _eisenstein_center
from mu3.face_lattice import digit_offset, get_rot, omega, s3, units


P = 0
K_INDEX = 1
Q = dodec.neighbors[P][K_INDEX]
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


def hex_corners_local(center_local: complex, rot: complex):
    return [center_local + units[k] / (s3 * rot) for k in range(6)]


def draw_lattice(ax, which, res, *, color_main, color_hex, color_deleted):
    rot_N = get_rot(res)
    # Wedge skeleton.
    for j in range(6):
        a_l, b_l, c_l = 0 + 0j, units[j], units[(j + 1) % 6]
        a, b, c = (to_joint(a_l, which), to_joint(b_l, which),
                   to_joint(c_l, which))
        verts = [(a.real, a.imag), (b.real, b.imag), (c.real, c.imag)]
        if j == 0:  # deleted wedge [0, 60)
            ax.add_patch(Polygon(verts, closed=True, facecolor=color_deleted,
                                 edgecolor=color_deleted, linestyle="--",
                                 linewidth=0.6, alpha=0.30, zorder=1))
        else:
            ax.plot([v[0] for v in verts] + [verts[0][0]],
                    [v[1] for v in verts] + [verts[0][1]],
                    color=color_main, lw=0.4, alpha=0.35, zorder=1)

    view_center = 0.5 * (0 + Q_CENTER)
    view_radius = abs(Q_CENTER) / 2 + 0.95
    for digits, z_local in enumerate_cells(res):
        z_joint = to_joint(z_local, which)
        if abs(z_joint - view_center) > view_radius + 0.15:
            continue
        corners_local = hex_corners_local(z_local, rot_N)
        corners_joint = [to_joint(c, which) for c in corners_local]
        xs = [c.real for c in corners_joint] + [corners_joint[0].real]
        ys = [c.imag for c in corners_joint] + [corners_joint[0].imag]

        is_pentagon = len(digits) == 0 or all(d == 0 for d in digits)
        is_deleted_child = first_nonzero_digit(digits) == 1

        if is_pentagon:
            ax.plot(xs, ys, color=color_main, lw=1.6, zorder=4)
            ax.fill(xs, ys, color=color_main, alpha=0.15, zorder=3)
        elif is_deleted_child:
            ax.plot(xs, ys, color=color_deleted, lw=0.4, alpha=0.55, zorder=2)
            ax.fill(xs, ys, color=color_deleted, alpha=0.22, zorder=2)
        else:
            ax.plot(xs, ys, color=color_hex, lw=0.3, alpha=0.45, zorder=2)
            ax.plot(z_joint.real, z_joint.imag, ".", color=color_hex,
                    markersize=0.8, alpha=0.45, zorder=2)


def main():
    fig, ax = plt.subplots(figsize=(14, 13))
    ax.set_aspect("equal")
    ax.axis("off")

    res = 3
    rot_N = get_rot(res)

    # Lattices for P and Q.
    draw_lattice(ax, "p", res,
                 color_main="#c33", color_hex="#a55", color_deleted="#f0a04a")
    draw_lattice(ax, "q", res,
                 color_main="#36c", color_hex="#558", color_deleted="#9bc4f0")

    # Shared icosa edge.
    ax.plot([0, Q_CENTER.real], [0, Q_CENTER.imag],
            color="#222", lw=2.0, alpha=0.6, zorder=5,
            solid_capstyle="round")

    # ----- Origin cell (0, 2, 1, 1) -----
    cell_digits = (2, 1, 1)
    z_C = _eisenstein_center(cell_digits)
    # Outline as filled red hex.
    corners = hex_corners_local(z_C, rot_N)
    xs = [c.real for c in corners] + [corners[0].real]
    ys = [c.imag for c in corners] + [corners[0].imag]
    ax.fill(xs, ys, color="#ffd9d9", alpha=0.85, zorder=6)
    ax.plot(xs, ys, color="#c33", lw=2.0, zorder=6.5)
    ax.plot(z_C.real, z_C.imag, "o", color="#c33", markersize=6, zorder=7)
    ax.text(z_C.real - 0.025, z_C.imag,
            "(0, 2, 1, 1)\nstart",
            fontsize=10, ha="right", va="center", color="#c33",
            fontweight="bold", zorder=7,
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec="#c33", lw=0.7, alpha=0.95))

    # ----- Walk arrow D=6 -----
    D = 6
    step = digit_offset[D] / rot_N
    z_n = z_C + step
    ax.annotate("", xytext=(z_C.real, z_C.imag),
                xy=(z_n.real, z_n.imag),
                arrowprops=dict(arrowstyle="-|>", color="#0a8a3a", lw=2.4,
                                mutation_scale=18),
                zorder=8)
    midx = (z_C.real + z_n.real) / 2
    midy = (z_C.imag + z_n.imag) / 2
    ax.text(midx + 0.012, midy - 0.018,
            "D=6", fontsize=10, ha="left", va="top",
            color="#0a8a3a", fontweight="bold", zorder=8,
            bbox=dict(boxstyle="round,pad=0.18", fc="white",
                      ec="#0a8a3a", lw=0.6, alpha=0.95))

    # ----- z_n: walked destination (the failing point) -----
    ax.plot(z_n.real, z_n.imag, "X", color="black",
            markersize=15, markeredgewidth=2.2, zorder=10)
    ax.annotate(
        f"z_n = {z_n.real:.4f} + {z_n.imag:.4f}i\n"
        f"|z_n*rot(3)| = |4+11omega| -- on the lattice grid\n"
        "but z_to_cell(p=0, z_n, 3) has residue 1+omega",
        xy=(z_n.real, z_n.imag),
        xytext=(z_n.real + 0.22, z_n.imag + 0.18),
        fontsize=9, ha="left", va="bottom", color="black", zorder=11,
        bbox=dict(boxstyle="round,pad=0.3", fc="#fff3c4",
                  ec="black", lw=0.8, alpha=0.95),
        arrowprops=dict(arrowstyle="-", color="black", lw=0.8),
    )

    # ----- Ghost cell (0, 4, 4, 3): z_n - (1+omega) -----
    z_ghost = z_n - (1 + omega)
    ghost_corners = hex_corners_local(z_ghost, rot_N)
    xs = [c.real for c in ghost_corners] + [ghost_corners[0].real]
    ys = [c.imag for c in ghost_corners] + [ghost_corners[0].imag]
    ax.plot(xs, ys, color="#d96b0f", lw=1.2, ls="--", zorder=6)
    ax.fill(xs, ys, color="#fde7c8", alpha=0.45, zorder=5.5)
    ax.plot(z_ghost.real, z_ghost.imag, "s", color="#d96b0f",
            markersize=8, zorder=7)
    ax.annotate(
        "z_C(0, 4, 4, 3) = z_n - (1+omega)\n"
        "but this position is in p=8's territory,\n"
        "so (0, 4, 4, 3) is not a valid index for p=0",
        xy=(z_ghost.real, z_ghost.imag),
        xytext=(z_ghost.real - 0.55, z_ghost.imag - 0.18),
        fontsize=9, ha="left", va="top", color="#a04500", zorder=11,
        bbox=dict(boxstyle="round,pad=0.3", fc="#fff7e8",
                  ec="#d96b0f", lw=0.8, alpha=0.95),
        arrowprops=dict(arrowstyle="-", color="#d96b0f", lw=0.8),
    )

    # The +(1+omega) shift arrow ghost -> z_n.
    ax.annotate("", xytext=(z_ghost.real, z_ghost.imag),
                xy=(z_n.real, z_n.imag),
                arrowprops=dict(arrowstyle="-|>", color="#d96b0f",
                                lw=1.6, ls=":", alpha=0.85,
                                mutation_scale=14),
                zorder=8)
    sx = (z_ghost.real + z_n.real) / 2
    sy = (z_ghost.imag + z_n.imag) / 2
    ax.text(sx + 0.04, sy + 0.02, "+(1+omega)",
            fontsize=9, ha="left", va="bottom",
            color="#d96b0f", style="italic", zorder=9)

    # ----- Un-fold target z_n * (-omega) -----
    z_unfold = z_n * (-omega)
    ax.plot(z_unfold.real, z_unfold.imag, "X", color="#888",
            markersize=12, markeredgewidth=1.6, zorder=9)
    ax.annotate(
        f"z_n * (-omega), arg = 19.84 deg\n"
        "lands in p=0's deleted wedge --\n"
        "canonicalize re-stitches by *(1+omega)\n"
        "back to z_n. Un-fold is a no-op.",
        xy=(z_unfold.real, z_unfold.imag),
        xytext=(z_unfold.real + 0.28, z_unfold.imag - 0.32),
        fontsize=9, ha="left", va="top", color="#555", zorder=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5",
                  ec="#888", lw=0.7, alpha=0.95),
        arrowprops=dict(arrowstyle="-", color="#888", lw=0.7),
    )

    # ----- Pentagon centers + labels -----
    for which, ctr, color, txt in [
        ("p", 0 + 0j, "#c33", "P=0"),
        ("q", Q_CENTER, "#36c", f"Q={Q}"),
    ]:
        ax.plot(ctr.real, ctr.imag, "o", color="black",
                markersize=10, markerfacecolor=color, zorder=6)
        ax.text(ctr.real, ctr.imag - 0.085, txt, fontsize=11,
                ha="center", va="top", color=color, fontweight="bold",
                zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="none", alpha=0.85))

    # Bisector to p=10 (left edge of p=0's territory at 180 deg) for reference.
    # And bisector to p=8 (240 deg).
    for ang_deg, label, ls in [(120, "bisector to p=5", ":"),
                               (240, "bisector to p=8", ":")]:
        ang = math.radians(ang_deg)
        # Perpendicular bisector at distance 0.5 from origin, perpendicular
        # to direction ang. Parameterize as 0.5*exp(i*ang) + t*i*exp(i*ang).
        c0 = 0.5 * cmath.exp(1j * ang)
        d = 1j * cmath.exp(1j * ang)
        pts = [c0 + t * d for t in (-1.4, 1.4)]
        ax.plot([p.real for p in pts], [p.imag for p in pts],
                color="#aaa", lw=0.7, ls=ls, alpha=0.7, zorder=1.5)

    view_center = 0.5 * (0 + Q_CENTER)
    view_radius = abs(Q_CENTER) / 2 + 0.95
    ax.set_xlim(view_center.real - view_radius, view_center.real + view_radius + 0.1)
    ax.set_ylim(view_center.imag - view_radius, view_center.imag + view_radius)

    ax.set_title(
        "Failing geometric ring-1 walk: cell (0, 2, 1, 1), D=6, res 3.\n"
        "z_n is on the lattice grid (z_n*rot(3) = 4+11*omega) and inside p=0's Voronoi cell,\n"
        "but z_to_cell(p=0, z_n, 3) gives residue 1+omega -- the deleted-direction unit at the pentagon level.\n"
        "Un-fold doesn't help (no-op). What is the canonical cell at z_n?",
        fontsize=11,
    )

    out = (Path(__file__).resolve().parent.parent / "figures"
           / "failing_walk_res3.png")
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
