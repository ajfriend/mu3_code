"""Plot one pentagon face plus its 5 neighbors with the res-N lattice.

For a chosen base pentagon ``(P,)`` -- defaults to ``(5,)`` -- this lays out
P at the origin and its 5 edge-neighbors around it in the same flat joint
frame used by ``plot_failing_walk_res3.py``: neighbor ``k`` sits at angle
``NEIGHBOR_ANGLE_DEG[k]`` (= 0, 120, 180, 240, 300 deg), with the missing
60 deg slot being P's deleted wedge.

For each of the 6 hexagonal face territories, the script draws:

- the 5 kept icosa triangles (CCW from primary), plus a shaded
  "deleted triangle" at the 0..60 deg wedge,
- every cell at the chosen resolution (default res 3),
- pentagon-center cells (the all-zero-digit path for each face)
  outlined in red,
- "deleted-direction" cells (first nonzero digit = 1) shaded orange --
  these are the ones excluded from ``cells_at_res``,
- the shared icosa edges between P and each neighbor (heavy black lines).

Output: ``figures/pentagon_neighborhood.png``.
"""

import cmath
import itertools
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from mu3 import dodec
from mu3.face_lattice import digit_offset, get_rot, s3, units


# --- Configuration ---------------------------------------------------------

P = 5                  # base pentagon to center
RES = 2                # lattice resolution to draw
SHOW_DELETED = True    # color the "deleted-direction" cells (first nonzero digit = 1)

# Edit me: angle of neighbor k around P in the joint frame. The 60 deg slot
# (P's deleted wedge) is intentionally absent.
NEIGHBOR_ANGLE_DEG = (0.0, 120.0, 180.0, 240.0, 300.0)


# --- Geometry --------------------------------------------------------------

def neighbor_transform(p: int, k: int):
    """Return (center, rot) sending neighbor k's local frame into the joint
    frame so its shared edge with p lines up with p's edge at angle alpha_p.
    """
    q = dodec.neighbors[p][k]
    j = list(dodec.neighbors[q]).index(p)
    alpha_p = math.radians(NEIGHBOR_ANGLE_DEG[k])
    alpha_q = math.radians(NEIGHBOR_ANGLE_DEG[j])
    center = cmath.exp(1j * alpha_p)
    rot = cmath.exp(1j * (alpha_p + math.pi - alpha_q))
    return q, center, rot


# Per-face: (face_id, center_in_joint, local_to_joint_rot, color_main, color_hex)
FACE_PALETTE = (
    "#c33",  # P (red)
    "#36c",  # k=0 (blue)
    "#0a8a3a",  # k=1 (green)
    "#a26ec3",  # k=2 (purple)
    "#d96b0f",  # k=3 (orange)
    "#0aa0a0",  # k=4 (teal)
)


def all_faces():
    """Yield (face_id, center, rot, color) for P and its 5 neighbors."""
    yield (P, 0 + 0j, 1 + 0j, FACE_PALETTE[0])
    for k in range(5):
        q, center, rot = neighbor_transform(P, k)
        yield (q, center, rot, FACE_PALETTE[k + 1])


def to_joint_factory(center: complex, rot: complex):
    def f(z_local: complex) -> complex:
        return center + rot * z_local
    return f


# --- Lattice enumeration ---------------------------------------------------

def first_nonzero_digit(digits):
    for d in digits:
        if d != 0:
            return d
    return None


def enumerate_cells(res: int):
    """Yield (digits, z_local) for every hex center at this resolution,
    INCLUDING leading-d=1 paths (so we can color them as 'deleted')."""
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


def hex_corners_local(center_local: complex, rot_N: complex):
    return [center_local + units[k] / (s3 * rot_N) for k in range(6)]


# --- Drawing ---------------------------------------------------------------

def draw_face(ax, center, rot, color_main, *, res, is_center=False):
    """Draw one face's triangle skeleton + deleted wedge + lattice."""
    rot_N = get_rot(res)
    to_joint = to_joint_factory(center, rot)

    # 6 unit triangles around the face center.
    for j in range(6):
        a_l, b_l, c_l = 0 + 0j, units[j], units[(j + 1) % 6]
        a, b, c = to_joint(a_l), to_joint(b_l), to_joint(c_l)
        verts = [(a.real, a.imag), (b.real, b.imag), (c.real, c.imag)]
        ax.plot([v[0] for v in verts] + [verts[0][0]],
                [v[1] for v in verts] + [verts[0][1]],
                color="black", lw=0.7, alpha=0.7, zorder=1)

    # Lattice cells.
    for digits, z_local in enumerate_cells(res):
        z_joint = to_joint(z_local)
        is_pentagon = len(digits) == 0 or all(d == 0 for d in digits)
        is_deleted_child = first_nonzero_digit(digits) == 1
        if is_deleted_child and not SHOW_DELETED:
            continue

        corners_local = hex_corners_local(z_local, rot_N)
        corners_joint = [to_joint(c) for c in corners_local]
        xs = [c.real for c in corners_joint] + [corners_joint[0].real]
        ys = [c.imag for c in corners_joint] + [corners_joint[0].imag]

        if is_pentagon:
            ax.fill(xs, ys, color=color_main, zorder=4)
            ax.plot(xs, ys, color=color_main, lw=1.6, zorder=4)
        elif is_deleted_child:
            ax.plot(xs, ys, color="#d96b0f", lw=0.5, alpha=0.7, zorder=2.5)
            ax.fill(xs, ys, color="#f0a04a", alpha=0.55, zorder=2.4)
        else:
            ax.plot(xs, ys, color=color_main, lw=0.4, alpha=0.7, zorder=2)
            if is_center:
                ax.plot(z_joint.real, z_joint.imag, ".",
                        color="#c33", markersize=3, zorder=4.5)


def draw_face_label(ax, face_id, center, color):
    ax.text(center.real, center.imag - 0.10,
            f"({face_id},)",
            fontsize=11, ha="center", va="top",
            color=color, fontweight="bold", zorder=6,
            bbox=dict(boxstyle="round,pad=0.22", fc="white",
                      ec="none", alpha=0.9))


# --- Main ------------------------------------------------------------------

def main():
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_aspect("equal")
    ax.axis("off")

    for face_id, center, rot, color in all_faces():
        draw_face(ax, center, rot, color, res=RES, is_center=(face_id == P))

    for face_id, center, _, color in all_faces():
        draw_face_label(ax, face_id, center, color)

    # View bounds: P + 5 neighbors at unit distance, hexes extend ~r_face=0.7639.
    R = 1.95
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)

    out = (Path(__file__).resolve().parent.parent / "figures"
           / "pentagon_neighborhood.png")
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
