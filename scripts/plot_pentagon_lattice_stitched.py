"""Pentagon-Eisenstein 2D lattice plot, after d=1 pruning + stitching.

Same four subplots as plot_pentagon_lattice.py (res 0, 1, 2, 3), but:

  1. Every cell whose first nonzero digit is 1 (the "orange" deleted-direction
     descendants) is removed entirely.
  2. Any lattice point that still lands inside the deleted triangle
     [180°, 240°) is rotated by +60° (×exp(iπ/3)), which lifts it onto the
     d=5 face triangle [240°, 300°). This is the "whole d=1 wedge absorbed
     into d=5" stitching — the 2D analog of the cross-face seam handling
     that will happen in 3D.

After the operation the shaded (d=1) triangle contains no cells, no cell
centers, and no hex corners. Cells that straddled the d=3 / d=1 boundary
appear visibly "folded": their corner on the d=1 side has been rotated into
d=5, so their outline jumps across the deleted triangle. That folding is
what the stitching does on the sphere (the jumping corners in 2D are the
corners that end up on an adjacent icosahedron face in 3D).
"""

from __future__ import annotations

import cmath
import itertools
import math
from pathlib import Path

import matplotlib.pyplot as plt

from mu3.face_lattice import get_rot, h3_digit_offset, omega, s3, units


ROT60 = cmath.exp(1j * math.pi / 3)

# The deleted triangle, in pentagon-plane (z-space) coordinates: the 60°
# wedge between units[3] (180°) and units[4] (240°).
DELETED_LO = 180.0
DELETED_HI = 240.0
DELETED_TRIANGLE_J = 3


def _angle_deg(z: complex) -> float:
    return math.degrees(math.atan2(z.imag, z.real)) % 360.0


def stitch(z: complex) -> complex:
    """If z sits in the deleted triangle, rotate it +60° into d=5's wedge."""
    if z == 0j:
        return z
    if DELETED_LO <= _angle_deg(z) < DELETED_HI:
        return z * ROT60
    return z


def hex_corners(center: complex, rot: complex) -> list[complex]:
    return [center + units[k] / (s3 * rot) for k in range(6)]


def as_xy(zs):
    return [z.real for z in zs], [z.imag for z in zs]


def first_nonzero_digit(digits):
    for d in digits:
        if d != 0:
            return d
    return None


def enumerate_kept_cells(res: int):
    """Yield (digits, z_center) for every descendant of the center pentagon
    at resolution ``res``, dropping any whose first nonzero digit is 1."""
    if res == 0:
        yield ((), 0 + 0j)
        return
    for digits in itertools.product(range(7), repeat=res):
        if first_nonzero_digit(digits) == 1:
            continue
        z = 0 + 0j
        for k, d in enumerate(digits, start=1):
            if d == 0:
                continue
            z += h3_digit_offset[d] / get_rot(k)
        yield (digits, z)


def plot_res(ax: plt.Axes, res: int, view_radius: float) -> None:
    rot_N = get_rot(res)

    # 6 equilateral unit triangles; shade the deleted one in gray.
    for j in range(6):
        a, b, c = 0 + 0j, units[j], units[(j + 1) % 6]
        xs = [a.real, b.real, c.real, a.real]
        ys = [a.imag, b.imag, c.imag, a.imag]
        if j == DELETED_TRIANGLE_J:
            ax.fill(xs, ys, color="#888", alpha=0.28, zorder=0)
            ax.plot(xs, ys, color="#666", lw=0.8, ls="--", alpha=0.5, zorder=1)
        else:
            ax.plot(xs, ys, color="#2a6fb3", lw=1.3, alpha=0.8, zorder=1)

    for digits, z_c in enumerate_kept_cells(res):
        is_pentagon = len(digits) == 0 or all(d == 0 for d in digits)
        raw_corners = hex_corners(z_c, rot_N)

        if is_pentagon:
            # Pentagon cell: drop the "deleted" corner entirely. The surviving
            # 5 corners form a 5-gon whose long edge (k=2 → k=4) spans the
            # 120° arc across the deleted triangle.
            kept = [c for k, c in enumerate(raw_corners) if k != DELETED_TRIANGLE_J]
            xs, ys = as_xy(kept + [kept[0]])
            ax.plot(xs, ys, color="#c33", lw=2.0, zorder=4)
            ax.fill(xs, ys, color="#c33", alpha=0.12, zorder=3)
        else:
            stitched = [stitch(c) for c in raw_corners]
            z_mark = stitch(z_c)
            xs, ys = as_xy(stitched + [stitched[0]])
            was_straddler = any(c != sc for c, sc in zip(raw_corners, stitched))
            color = "#2a8f4f" if was_straddler else "#666"
            lw = 1.1 if was_straddler else 0.8
            ax.plot(xs, ys, color=color, lw=lw, zorder=3 if was_straddler else 2)
            ax.plot(z_mark.real, z_mark.imag, ".", color=color,
                    markersize=2.0, zorder=3 if was_straddler else 2)

    wedge_digit = {0: 6, 1: 2, 2: 3, 3: 1, 4: 5, 5: 4}
    for j in range(6):
        centroid = (units[j] + units[(j + 1) % 6]) / 3
        pt = centroid * (1.08 / abs(centroid))
        d = wedge_digit[j]
        label = f"d={d}" + ("\n(deleted)" if j == DELETED_TRIANGLE_J else "")
        color = "#844" if j == DELETED_TRIANGLE_J else "#264"
        ax.text(pt.real, pt.imag, label,
                fontsize=9, ha="center", va="center",
                color=color, zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))

    ax.plot(0, 0, "k.", markersize=6, zorder=7)
    ax.text(0.03, 0.03, "V[b]", fontsize=9, zorder=7)

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
    ax.set_title(f"pentagon-Eisenstein (stitched), res {res}  "
                 f"(|rot|²={abs(rot_N)**2:.0f}, "
                 f"arg(rot)={math.degrees(cmath.phase(rot_N)):.1f}°)",
                 fontsize=11)


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    for ax, res in zip(axes.flat, (0, 1, 2, 3)):
        plot_res(ax, res, view_radius=1.3)

    fig.suptitle(
        "mu3 pentagon-Eisenstein lattice after d=1 pruning + stitching  "
        "(red = pentagon 5-gon; green = cells whose corners were rotated "
        "out of the deleted triangle; gray = untouched cells)",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()

    out = Path(__file__).resolve().parent.parent / "figures" / "pentagon_lattice_stitched.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
