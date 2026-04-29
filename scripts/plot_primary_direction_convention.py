"""Diagram fixing the digit-numbering convention via the primary direction.

Layout:

* Hexagon centered at origin, divided into 6 unit-equilateral-triangle wedges.
* Primary direction = horizontal +x (0 deg). It points along a wedge
  boundary, not through a wedge interior.
* Digits go strictly CCW around the hexagon, 1 -> 6, starting from the
  deleted wedge (which lives in the upper-left, ~150 deg from primary):
    d=1 (deleted), d=2, d=3, d=4, d=5, d=6.
* Pentagon surviving CCW cycle: (2, 3, 4, 5, 6) - sequential.

This drops H3's IJK bit-encoding (we never used it) in favor of digits
that are simply positions around the hex.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Polygon


# Wedge j spans angles [j * 60 deg, (j + 1) * 60 deg). With primary at 0 deg
# and digits going CCW starting with d=1 immediately CCW of primary:
#   j=0 [0..60]:    d=1 (deleted)
#   j=1 [60..120]:  d=2
#   j=2 [120..180]: d=3
#   j=3 [180..240]: d=4
#   j=4 [240..300]: d=5
#   j=5 [300..360]: d=6
WEDGE_DIGIT = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}
DELETED_WEDGE = 0


def hex_corner(j: int) -> complex:
    """Unit-distance corner at angle j * 60 deg."""
    a = math.radians(60 * j)
    return complex(math.cos(a), math.sin(a))


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    ax.set_aspect("equal")
    ax.axis("off")

    # 6 wedges (origin -> corner j -> corner j+1).
    for j in range(6):
        a, b = hex_corner(j), hex_corner((j + 1) % 6)
        verts = [(0, 0), (a.real, a.imag), (b.real, b.imag)]
        if j == DELETED_WEDGE:
            ax.add_patch(Polygon(verts, closed=True, facecolor="#bbb",
                                 edgecolor="#666", linestyle="--",
                                 linewidth=1.2, alpha=0.55, zorder=1))
        else:
            ax.add_patch(Polygon(verts, closed=True, facecolor="#eaf2ff",
                                 edgecolor="#2a6fb3", linewidth=1.2,
                                 alpha=0.85, zorder=1))

    # Outer hexagon outline.
    hex_pts = [(hex_corner(j).real, hex_corner(j).imag) for j in range(6)]
    ax.add_patch(Polygon(hex_pts, closed=True, facecolor="none",
                         edgecolor="#2a6fb3", linewidth=1.6, zorder=2))

    # Digit labels at wedge centroids.
    for j in range(6):
        a, b = hex_corner(j), hex_corner((j + 1) % 6)
        c = (a + b) / 3.0  # centroid of wedge triangle (origin, a, b)
        d = WEDGE_DIGIT[j]
        is_deleted = j == DELETED_WEDGE
        text = f"d={d}" + ("\n(deleted)" if is_deleted else "")
        color = "#844" if is_deleted else "#264"
        ax.text(c.real, c.imag, text, ha="center", va="center",
                fontsize=14, fontweight="bold", color=color, zorder=4,
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec="none", alpha=0.9))

    # Primary direction arrow along +x axis (boundary between d=6 and d=2).
    ax.add_patch(FancyArrow(
        0, 0, 1.18, 0,
        width=0.012, head_width=0.085, head_length=0.10,
        length_includes_head=True,
        facecolor="#0a8a3a", edgecolor="#0a8a3a",
        linewidth=1.4, zorder=5,
    ))
    ax.text(1.32, 0.0, "primary\ndirection",
            ha="left", va="center",
            fontsize=12, color="#0a8a3a", fontweight="bold")

    # Origin marker.
    ax.plot(0, 0, "ko", markersize=5, zorder=6)
    ax.text(-0.06, -0.06, "pentagon\ncenter",
            ha="right", va="top", fontsize=10, color="#222")

    # Annotate which boundary the primary direction lies on.
    ax.text(0.5, -0.045, "boundary (d=6 | d=1)",
            ha="center", va="top", fontsize=9, color="#0a8a3a",
            style="italic")

    ax.set_xlim(-1.35, 2.15)
    ax.set_ylim(-1.35, 1.35)
    ax.set_title(
        "Pentagon digit convention: 1-6 CCW around the hex, d=1 deleted\n"
        "d=1 sits immediately CCW of primary; surviving cycle (2, 3, 4, 5, 6)",
        fontsize=12,
    )

    out = Path(__file__).resolve().parent.parent / "figures" / "primary_direction_convention.png"
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
