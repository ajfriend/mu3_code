"""Plot the pentagon-Eisenstein 2D lattice for one pentagon at res 0, 1, 2.

Shows the abstract 2D hex grid centered at a pentagon (icosa vertex V[b]),
with the deleted digit-1 direction shaded. At each resolution, the hexes
tile the 2D plane — which should make obvious whether the internal lattice
math is consistent, independent of any sphere projection.
"""

from __future__ import annotations

import cmath
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Wedge

from mu3.face_lattice import get_rot, h3_digit_offset, omega, s3, units


def hex_corners(center: complex, rot: complex) -> list[complex]:
    return [center + units[k] / (s3 * rot) for k in range(6)]


def as_xy(zs: list[complex]):
    return [z.real for z in zs], [z.imag for z in zs]


def triangles_around_pentagon() -> list[tuple[int, tuple[complex, complex, complex]]]:
    """Return 6 equilateral unit triangles, tagged by j-index."""
    return [(j, (0 + 0j, units[j], units[(j + 1) % 6])) for j in range(6)]


DELETED_TRIANGLE_J = 3  # whole-triangle deletion: j=3 spans 180°→240°


def first_nonzero_digit(digits: tuple[int, ...]) -> int | None:
    for d in digits:
        if d != 0:
            return d
    return None


def enumerate_cells(res: int):
    """Yield (digits, z_center) for every child of the center pentagon at
    resolution res, including digit-1 descendants (no preemptive filtering)."""
    import itertools
    if res == 0:
        yield ((), 0 + 0j)
        return
    for digits in itertools.product(range(7), repeat=res):
        z = 0 + 0j
        for k, d in enumerate(digits, start=1):
            if d == 0:
                continue
            z += h3_digit_offset[d] / get_rot(k)
        yield (digits, z)


def plot_res(ax: plt.Axes, res: int, view_radius: float) -> None:
    rot_N = get_rot(res)

    # 6 equilateral unit triangles; shade the deleted one in gray.
    for j, (a, b, c) in triangles_around_pentagon():
        xs = [a.real, b.real, c.real, a.real]
        ys = [a.imag, b.imag, c.imag, a.imag]
        if j == DELETED_TRIANGLE_J:
            ax.fill(xs, ys, color="#888", alpha=0.28, zorder=0)
            ax.plot(xs, ys, color="#666", lw=0.8, ls="--", alpha=0.5, zorder=1)
        else:
            ax.plot(xs, ys, color="#2a6fb3", lw=1.3, alpha=0.8, zorder=1)

    # enumerate every descendant of the center pentagon at this resolution;
    # draw the center pentagon in red, "deleted direction" children (first
    # nonzero digit = 1) in orange, and the rest as gray hexes.
    for digits, z_c in enumerate_cells(res):
        corners = hex_corners(z_c, rot_N)
        xs, ys = as_xy(corners + [corners[0]])
        is_pentagon = (len(digits) == 0) or all(d == 0 for d in digits)
        is_deleted_child = first_nonzero_digit(digits) == 1
        if is_pentagon:
            ax.plot(xs, ys, color="#c33", lw=2.0, zorder=4)
            ax.fill(xs, ys, color="#c33", alpha=0.12, zorder=3)
            # mark the hex corner that sits inside the deleted triangle —
            # the one that collapses away when the hex folds into a pentagon.
            deleted_corner = corners[DELETED_TRIANGLE_J]
            ax.plot(deleted_corner.real, deleted_corner.imag,
                    marker="x", color="#c33", markersize=10,
                    markeredgewidth=2.2, zorder=6)
        elif is_deleted_child:
            ax.plot(xs, ys, color="#d96b0f", lw=1.0, zorder=3)
            ax.fill(xs, ys, color="#f0a04a", alpha=0.55, zorder=2)
            ax.plot(z_c.real, z_c.imag, ".", color="#8a3a00", markersize=2.0, zorder=3)
        else:
            ax.plot(xs, ys, color="#666", lw=0.8, zorder=2)
            ax.plot(z_c.real, z_c.imag, ".", color="#666", markersize=2.0, zorder=2)

    # wedge labels at each triangle centroid. Wedge j (between units[j] and
    # units[j+1]) is labeled by the digit at its CCW boundary (at angle 60°·(j+1)).
    # With j=3 deleted, that slot is digit 1 (boundary at 240°).
    wedge_digit = {0: 6, 1: 2, 2: 3, 3: 1, 4: 5, 5: 4}
    for j in range(6):
        centroid = (units[j] + units[(j + 1) % 6]) / 3
        pt = centroid * (1.08 / abs(centroid))  # push just outside unit circle
        d = wedge_digit[j]
        label = f"d={d}" + ("\n(deleted)" if j == DELETED_TRIANGLE_J else "")
        color = "#844" if j == DELETED_TRIANGLE_J else "#264"
        ax.text(pt.real, pt.imag, label,
                fontsize=9, ha="center", va="center",
                color=color, zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))

    # origin marker
    ax.plot(0, 0, "k.", markersize=6, zorder=7)
    ax.text(0.03, 0.03, "V[b]", fontsize=9, zorder=7)

    # omega-rotation fixed points: 1, omega, omega²=-1-omega (cube roots of unity).
    # These are the 3 points that 120° rotation (×omega) cycles, at unit distance
    # on the directions 0°, 120°, 240°.
    omega_points = [(1 + 0j, "ω⁰ = 1"),
                    (omega, "ω¹ = ω"),
                    (omega * omega, "ω² = -1-ω")]
    for z, label in omega_points:
        ax.plot(z.real, z.imag, marker="*", color="#b8860b",
                markersize=14, markeredgecolor="black",
                markeredgewidth=0.6, zorder=8)
        offset = z * 0.12 / abs(z)
        ax.text(z.real + offset.real, z.imag + offset.imag, label,
                fontsize=8, ha="center", va="center",
                color="#704000", zorder=8,
                bbox=dict(boxstyle="round,pad=0.15", fc="#fff8e0",
                          ec="#b8860b", lw=0.5, alpha=0.9))

    ax.set_xlim(-view_radius, view_radius)
    ax.set_ylim(-view_radius, view_radius)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"pentagon-Eisenstein, res {res}  (|rot|²={abs(rot_N)**2:.0f}, "
                 f"arg(rot)={math.degrees(cmath.phase(rot_N)):.1f}°)",
                 fontsize=11)


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    for ax, res in zip(axes.flat, (0, 1, 2, 3)):
        plot_res(ax, res, view_radius=1.3)

    fig.suptitle("mu3 pentagon-Eisenstein lattice, one base  "
                 "(red hex = pentagon cell; gray hexes = other cells; "
                 "blue triangles = 5 kept unit triangles; shaded triangle = whole triangle deleted)",
                 fontsize=11, y=1.02)
    plt.tight_layout()

    out = Path(__file__).resolve().parent.parent / "figures" / "pentagon_lattice.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
