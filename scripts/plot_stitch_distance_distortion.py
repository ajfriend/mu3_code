"""The stitch creates ring-1 pairs at non-unit flat distance.

At res 2, walking ``D=6`` from cell ``(0, 0, 2)`` lands at the 3-hex
corner shared by ``(0, 0, 2)``, ``(0, 0, 6)``, and the deleted phantom
``(0, 0, 1)``. The +60 deg intra-pentagon stitch identifies the corner
with a 3D point that's geometrically adjacent to both ``(0, 0, 2)``
and ``(0, 0, 6)``.

But in the FLAT unfolding, the canonical positions of ``(0, 0, 2)``
at ``(-0.071, 0.124i)`` and ``(0, 0, 6)`` at ``(0.143, 0)`` are at
flat distance ``sqrt(3)/7`` -- NOT the unit step ``1/7``. The flat
unfolding stretches the wide T_1 icosa triangle (where the deleted
wedge lives) so distances inside it don't match 3D distances.

Two panels:

- LEFT: deleted wedge, walk arrow ``(0, 0, 2) D=6``, the 3-hex corner.
- RIGHT: cell labels, with the 3D-distance-equivalent "twin" relationship
  between ``(0, 0, 2)`` and ``(0, 0, 6)`` indicated.

Output: ``figures/stitch_distance_distortion.png``.
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


def draw_lattice(ax, *, show_deleted_wedge: bool):
    rot_N = get_rot(RES)
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


def draw_filled_cell(ax, z_C, rot_N, color, *, alpha=0.55, lw=2.0):
    corners = hex_corners_local(z_C, rot_N)
    xs = [c.real for c in corners] + [corners[0].real]
    ys = [c.imag for c in corners] + [corners[0].imag]
    ax.fill(xs, ys, color=color, alpha=alpha, zorder=6)
    ax.plot(xs, ys, color=color, lw=lw, zorder=6.5)
    ax.plot(z_C.real, z_C.imag, "o", color=color, markersize=6, zorder=7)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    rot_N = get_rot(RES)

    # The three cells of interest.
    z_002 = _eisenstein_center((0, 2))    # (0, 0, 2) at (-0.071, 0.124)
    z_006 = _eisenstein_center((0, 6))    # (0, 0, 6) at (0.143, 0)
    z_001 = _eisenstein_center((0, 1))    # (0, 0, 1) phantom at (0.071, 0.124)

    for ax, show_deleted in zip(axes, (True, False)):
        ax.set_aspect("equal")
        ax.axis("off")
        draw_lattice(ax, show_deleted_wedge=show_deleted)

        # Pentagon center.
        ax.plot(0, 0, "o", color="black", markersize=10,
                markerfacecolor="#c33", zorder=12)

        # Filled cells: (0, 0, 2) blue and (0, 0, 6) purple.
        draw_filled_cell(ax, z_002, rot_N, "#36c")
        draw_filled_cell(ax, z_006, rot_N, "#9c27b0")

        if show_deleted:
            # LEFT panel: arrows + phantom corner.

            # Walk arrow (0, 0, 2) D=6.
            step = digit_offset[6] / rot_N
            ax.annotate("", xytext=(z_002.real, z_002.imag),
                        xy=(z_001.real, z_001.imag),
                        arrowprops=dict(arrowstyle="-|>", color="#0a8a3a",
                                        lw=2.5, mutation_scale=15),
                        zorder=8)
            ax.text((z_002.real + z_001.real) / 2 + 0.005,
                    (z_002.imag + z_001.imag) / 2 + 0.02,
                    "D=6 walk\n(unit step)",
                    fontsize=9, color="#0a8a3a", fontweight="bold",
                    ha="left", va="bottom", zorder=8.5,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white",
                              ec="#0a8a3a", lw=0.5, alpha=0.9))

            # Mark the 3-hex corner.
            ax.plot(z_001.real, z_001.imag, "X", color="black",
                    markersize=14, markeredgewidth=2.0, zorder=9)
            ax.annotate("3-hex corner\n(phantom (0, 0, 1)\nin deleted wedge)",
                        xy=(z_001.real, z_001.imag),
                        xytext=(z_001.real + 0.16, z_001.imag - 0.08),
                        fontsize=9, color="#a04500", fontweight="bold",
                        ha="left", va="top", zorder=10,
                        bbox=dict(boxstyle="round,pad=0.22", fc="#fff7e8",
                                  ec="#d96b0f", lw=0.7, alpha=0.95),
                        arrowprops=dict(arrowstyle="-", color="#d96b0f",
                                        lw=0.7))

            # Distance from (0, 0, 2) to (0, 0, 6) in flat (the failing test
            # measures this).
            ax.annotate("", xytext=(z_002.real, z_002.imag),
                        xy=(z_006.real, z_006.imag),
                        arrowprops=dict(arrowstyle="-", color="#222",
                                        lw=1.2, ls=":", alpha=0.7),
                        zorder=7.5)
            d_flat = abs(z_006 - z_002)
            unit = 1.0 / 7
            ax.text((z_002.real + z_006.real) / 2,
                    (z_002.imag + z_006.imag) / 2 - 0.04,
                    f"flat distance = sqrt(3)/7 ≈ {d_flat:.3f}\n"
                    f"= {d_flat / unit:.3f} × unit_step",
                    fontsize=9, color="#222", fontweight="bold",
                    ha="center", va="top", zorder=10,
                    bbox=dict(boxstyle="round,pad=0.22", fc="white",
                              ec="#222", lw=0.7, alpha=0.95))

        else:
            # RIGHT panel: cell labels.
            for (digits, z, color) in [
                ((0, 2), z_002, "#36c"),
                ((0, 6), z_006, "#9c27b0"),
                ((0, 1), z_001, "#d96b0f"),
            ]:
                text = "(0, " + ", ".join(str(d) for d in digits) + ")"
                if digits == (0, 1):
                    text += "\nDELETED\n(phantom)"
                ax.text(z.real, z.imag, text,
                        fontsize=9, ha="center", va="center",
                        color=color, fontweight="bold", zorder=11,
                        bbox=dict(boxstyle="round,pad=0.18", fc="white",
                                  ec=color, lw=0.7, alpha=0.96))

            # Annotation: the issue.
            ax.text(0.05, -0.42,
                    "My code says: (0, 0, 2)'s ring-1 includes (0, 0, 6)\n"
                    "via D=6 walk to the 3-hex corner, then CW digit\n"
                    "rotation maps phantom (0, 0, 1) -> (0, 0, 6).\n\n"
                    "But flat distance |z_C(0,0,2) - z_C(0,0,6)| = sqrt(3)/7,\n"
                    "NOT 1/7 (the unit step).\n\n"
                    "Test ring1_eisenstein_step_length expects unit step\n"
                    "for same-base ring-1 -- so this fails.\n\n"
                    "Question: is (0, 0, 6) really a ring-1 neighbor of\n"
                    "(0, 0, 2), or did the stitch take us too far?",
                    fontsize=9, ha="left", va="top",
                    color="#222", zorder=15,
                    bbox=dict(boxstyle="round,pad=0.3", fc="#fff7e8",
                              ec="#222", lw=0.8, alpha=0.95))

        ax.set_xlim(-0.55, 0.55)
        ax.set_ylim(-0.50, 0.50)

    axes[0].set_title(
        "Walk (0, 0, 2) D=6 lands at 3-hex corner in deleted wedge\n"
        "Distance to (0, 0, 6) in FLAT unfolding = sqrt(3) × unit_step",
        fontsize=10,
    )
    axes[1].set_title(
        "Cell labels for the three cells sharing the corner\n"
        "Same-base ring-1 with non-unit flat distance -- stitch artifact",
        fontsize=10,
    )

    fig.suptitle(
        "Stitch-induced flat-distance distortion at res 2 in P=0.",
        fontsize=12, y=1.02,
    )

    out = (Path(__file__).resolve().parent.parent / "figures"
           / "stitch_distance_distortion.png")
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
