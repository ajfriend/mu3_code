"""Failure case: ``(0, 2, 6, 6)`` D=1 walk at res 3.

Source cell ``(0, 2, 6, 6)`` is at flat ``(0.122, 0.353i)`` -- in the
d=2 wedge, NOT pentagon-adjacent. Walking direction D=1 lands at
``z_n = (0.163, 0.389i)`` at flat angle 67 deg (just outside the
deleted wedge).

The divmod-extracted digit string for ``z_n`` is ``(0, 1, 2, 3)`` --
leading-zero d=1, so excluded by ``cells_at_res``. There is NO valid
non-deleted-form digit string for this position.

My current code applies the standard +60 deg CCW stitch, mapping
``(0, 1, 2, 3) -> (0, 2, 3, 4)``. But ``(0, 2, 3, 4)``'s flat position
is ``(-0.255, 0.336i)`` -- a totally different 3D point.

Two panels, zoomed to show the res-3 lattice around (0, 2, 6, 6):

- LEFT: walk arrows from (0, 2, 6, 6). 3 walks land at real cells
  (green); 3 walks land at phantom positions (orange = leading-zero
  d=1 with no canonical representation).
- RIGHT: cell labels. Shows where the phantom CCW-rotation lands
  (the wrongly-claimed neighbor far from source).

Output: ``figures/res3_phantom_failure.png``.
"""

import cmath
import itertools
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from mu3.cell import _eisenstein_center
from mu3.face_lattice import digit_offset, get_rot, omega, s3, units


RES = 3
DELETED_HI_RAD = math.pi / 3


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


def cell_at_position(z: complex, tol: float = 1e-6):
    """Find a digit string whose ``_eisenstein_center`` is z, or None."""
    for digits, zc in enumerate_cells(RES):
        if abs(z - zc) < tol:
            return digits
    return None


# View centered on (0, 2, 6, 6).
SOURCE_DIGITS = (2, 6, 6)
VIEW_CENTER = _eisenstein_center(SOURCE_DIGITS)
VIEW_RADIUS = 0.30


def in_view(z: complex) -> bool:
    return abs(z - VIEW_CENTER) <= VIEW_RADIUS


def draw_lattice(ax, *, show_deleted_wedge: bool):
    rot_N = get_rot(RES)

    # Wedge skeleton.
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
                    color="#bbb", lw=0.4, alpha=0.4, zorder=1)

    # Cells in view.
    for digits, z_local in enumerate_cells(RES):
        if not in_view(z_local):
            continue
        corners = hex_corners_local(z_local, rot_N)
        xs = [c.real for c in corners] + [corners[0].real]
        ys = [c.imag for c in corners] + [corners[0].imag]

        is_pentagon = all(d == 0 for d in digits)
        is_deleted_child = first_nonzero_digit(digits) == 1

        if is_pentagon:
            ax.plot(xs, ys, color="#c33", lw=1.2, alpha=0.7, zorder=4)
            ax.fill(xs, ys, color="#c33", alpha=0.10, zorder=3)
        elif is_deleted_child and show_deleted_wedge:
            ax.plot(xs, ys, color="#d96b0f", lw=0.4, alpha=0.6,
                    ls="--", zorder=2.5)
            ax.fill(xs, ys, color="#fde7c8", alpha=0.30, zorder=2.4)
        else:
            ax.plot(xs, ys, color="#aaa", lw=0.3, alpha=0.45, zorder=2)


def draw_filled_cell(ax, z_C, rot_N, color, *, alpha=0.55, lw=2.0):
    corners = hex_corners_local(z_C, rot_N)
    xs = [c.real for c in corners] + [corners[0].real]
    ys = [c.imag for c in corners] + [corners[0].imag]
    ax.fill(xs, ys, color=color, alpha=alpha, zorder=6)
    ax.plot(xs, ys, color=color, lw=lw, zorder=6.5)
    ax.plot(z_C.real, z_C.imag, "o", color=color, markersize=5, zorder=7)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    rot_N = get_rot(RES)
    z_source = _eisenstein_center(SOURCE_DIGITS)

    # 6 walks from source.
    walks = []
    for D in (1, 2, 3, 4, 5, 6):
        step = digit_offset[D] / rot_N
        z_n = z_source + step
        nb_digits = cell_at_position(z_n)
        # Determine "kind": real cell, phantom-deleted-form
        if nb_digits is None:
            kind = "phantom_no_match"
        else:
            first_nz = first_nonzero_digit(nb_digits)
            if first_nz == 1:
                kind = "phantom_leading_d1"
            else:
                kind = "real"
        walks.append((D, z_n, nb_digits, kind))

    for ax, show_deleted in zip(axes, (True, False)):
        ax.set_aspect("equal")
        ax.axis("off")
        draw_lattice(ax, show_deleted_wedge=show_deleted)

        # Source cell (red).
        draw_filled_cell(ax, z_source, rot_N, "#c33")

        if show_deleted:
            # LEFT panel: walk arrows.
            for D, z_n, nb_digits, kind in walks:
                if kind == "real":
                    arrow_color = "#0a8a3a"
                    lw = 1.6
                elif kind == "phantom_leading_d1":
                    arrow_color = "#d96b0f"
                    lw = 2.2
                else:
                    arrow_color = "#888"
                    lw = 1.4
                ax.annotate("", xytext=(z_source.real, z_source.imag),
                            xy=(z_n.real, z_n.imag),
                            arrowprops=dict(arrowstyle="-|>", color=arrow_color,
                                            lw=lw, alpha=0.92, mutation_scale=12),
                            zorder=8)
                # D label at arrowhead.
                ax.text(z_n.real, z_n.imag, str(D),
                        fontsize=7, ha="center", va="center",
                        color="white", zorder=9,
                        bbox=dict(boxstyle="circle,pad=0.10",
                                  fc=arrow_color, ec="none", alpha=0.92))

            # Mark the (0, 2, 6, 6) position.
            ax.text(z_source.real, z_source.imag - 0.022,
                    "(0, 2, 6, 6)\nsource",
                    fontsize=8, ha="center", va="top",
                    color="#c33", fontweight="bold", zorder=11,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white",
                              ec="#c33", lw=0.6, alpha=0.95))

        else:
            # RIGHT panel: labels + show wrong CCW twin.
            # Label source.
            ax.text(z_source.real, z_source.imag,
                    "(0, 2, 6, 6)",
                    fontsize=8, ha="center", va="center",
                    color="#c33", fontweight="bold", zorder=11,
                    bbox=dict(boxstyle="round,pad=0.16", fc="white",
                              ec="#c33", lw=0.6, alpha=0.95))

            # For each walk, label real cells.
            for D, z_n, nb_digits, kind in walks:
                if kind == "real":
                    text = "(0, " + ", ".join(str(d) for d in nb_digits) + ")"
                    ax.text(z_n.real, z_n.imag, text,
                            fontsize=7, ha="center", va="center",
                            color="#0a8a3a", fontweight="bold", zorder=11,
                            bbox=dict(boxstyle="round,pad=0.14", fc="white",
                                      ec="#0a8a3a", lw=0.5, alpha=0.92))
                elif kind == "phantom_leading_d1":
                    # X mark at phantom.
                    ax.plot(z_n.real, z_n.imag, "X", color="#d96b0f",
                            markersize=12, markeredgewidth=2.0, zorder=10)
                    text = (
                        "phantom: (0, " +
                        ", ".join(str(d) for d in nb_digits) + ")"
                    )
                    ax.text(z_n.real, z_n.imag - 0.018, text,
                            fontsize=6.5, ha="center", va="top",
                            color="#a04500", zorder=11,
                            bbox=dict(boxstyle="round,pad=0.12", fc="white",
                                      ec="#d96b0f", lw=0.5, alpha=0.95))

            # For each phantom walk, draw the wrongly-claimed CCW twin.
            from mu3.face_lattice import rotate_digit_ccw
            for D, z_n, nb_digits, kind in walks:
                if kind != "phantom_leading_d1":
                    continue
                rotated_digits = tuple(rotate_digit_ccw(d, 1)
                                       for d in nb_digits)
                z_twin = _eisenstein_center(rotated_digits)
                # Draw an arrow from source to the wrongly-claimed twin.
                # Don't filter by view -- highlight that it's far.
                ax.annotate("", xytext=(z_source.real, z_source.imag),
                            xy=(z_twin.real, z_twin.imag),
                            arrowprops=dict(arrowstyle="-|>",
                                            color="#9c27b0", lw=1.4,
                                            ls=":", alpha=0.85,
                                            mutation_scale=12),
                            zorder=8)
                # Mark twin (might be out of view but plot anyway).
                ax.plot(z_twin.real, z_twin.imag, "*", color="#9c27b0",
                        markersize=16, zorder=12)
                text = "(0, " + ", ".join(str(d) for d in rotated_digits) + ")"
                ax.text(z_twin.real, z_twin.imag - 0.025,
                        f"CCW twin\n{text}\n(WRONG --\nfar from source)",
                        fontsize=7, ha="center", va="top",
                        color="#9c27b0", fontweight="bold", zorder=12,
                        bbox=dict(boxstyle="round,pad=0.16", fc="white",
                                  ec="#9c27b0", lw=0.7, alpha=0.95))

        # View bounds.
        # For the right panel, show wider view to include twins.
        if show_deleted:
            ax.set_xlim(VIEW_CENTER.real - VIEW_RADIUS,
                        VIEW_CENTER.real + VIEW_RADIUS)
            ax.set_ylim(VIEW_CENTER.imag - VIEW_RADIUS,
                        VIEW_CENTER.imag + VIEW_RADIUS)
        else:
            ax.set_xlim(-0.45, 0.45)
            ax.set_ylim(0.0, 0.55)

    axes[0].set_title(
        f"6 walks from (0, 2, 6, 6). 3 land at REAL cells (green);\n"
        f"3 land at PHANTOMS (orange) -- positions whose only digit\n"
        f"string is leading-zero d=1.",
        fontsize=10,
    )
    axes[1].set_title(
        "Wider view. Real-cell neighbors labeled green; phantoms in\n"
        "orange. Purple arrows + stars show where my code WRONGLY maps\n"
        "phantoms to (CCW digit rotation lands far from source).",
        fontsize=10,
    )

    fig.suptitle(
        "Res-3 phantom failure: 3 of (0, 2, 6, 6)'s walks have no real "
        "ring-1 neighbor in the canonical lattice.",
        fontsize=12, y=1.02,
    )

    out = (Path(__file__).resolve().parent.parent / "figures"
           / "res3_phantom_failure.png")
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
