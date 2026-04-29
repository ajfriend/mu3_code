"""Two adjacent pentagons unfolded into a joint Eisenstein flat frame
-- GENERAL case for any adjacency.

Given pentagons P and Q = ``dodec.neighbors[P][k]`` for k in {0..4}, we
unfold their tangent-plane Eisenstein lattices into a single 2D flat
frame. Two ingredients:

1. Q's position in P's flat frame: ``Q at z = exp(i·α_P)`` at unit
   distance, where α_P is the flat-Eisenstein angle of P's k-th
   neighbor:

       k         0    1    2    3    4
       α_P (°)   0   120  180  240  300

   (k=0 is the merged-corner direction along P's primary; the other 4
   are the d=2, d=3, d=4, d=5 ray directions in P's lattice.)

2. Q's frame orientation: Q's local +x (= Q's primary direction) maps
   to joint angle ``α_P + 180° − α_Q``, where α_Q is the analogous
   angle for P in Q's neighbor list (j = ``neighbors[Q].index(P)``).

   The intuition: from Q's center, P is at flat angle α_Q in Q's local
   frame. In the joint frame, the direction Q → P is α_P + 180°
   (opposite of P → Q). For these to match, Q-local must be rotated by
   ``α_P + 180° − α_Q`` to land in the joint frame.

Re-rooting formula: ``z_q_joint = Q_CENTER + Q_LOCAL_TO_JOINT_ROT * z_q_local``,
or equivalently ``z_q_local = (z_p_joint − Q_CENTER) / Q_LOCAL_TO_JOINT_ROT``.

The merged-corner case (k = j = 0) gives ``Q_CENTER = 1`` and
``rotation = exp(iπ) = −1`` — exactly the special-case picture in
``plot_pentagon_lattice_pair.py``. All 5 adjacency types are
parameterized by ``K_INDEX`` below.

Output: ``figures/pentagon_lattice_pair_general.png``.
"""

import cmath
import itertools
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from mu3 import dodec
from mu3.face_lattice import digit_offset, get_rot, s3, units


# Tweak K_INDEX (0..4) to step through adjacency types:
#   0: merged-corner neighbor (= primary-direction target)
#   1..4: d=2 / d=3 / d=4 / d=5 ray directions in P's lattice
P = 0
K_INDEX = 1
Q = dodec.neighbors[P][K_INDEX]
J_INDEX = list(dodec.neighbors[Q]).index(P)


def _neighbor_angle_deg(k: int) -> float:
    """Flat-Eisenstein angle (deg) of the k-th CCW neighbor from primary."""
    return [0.0, 120.0, 180.0, 240.0, 300.0][k]


ALPHA_P_DEG = _neighbor_angle_deg(K_INDEX)
ALPHA_Q_DEG = _neighbor_angle_deg(J_INDEX)
ROT_DEG = ALPHA_P_DEG + 180.0 - ALPHA_Q_DEG

ALPHA_P = math.radians(ALPHA_P_DEG)
Q_CENTER = cmath.exp(1j * ALPHA_P)
Q_LOCAL_TO_JOINT_ROT = cmath.exp(1j * math.radians(ROT_DEG))


# Each pentagon's deleted wedge sits at local flat [0°, 60°). j=0 in our
# convention.
DELETED_TRIANGLE_J = 0


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


def hex_corners_local(center_local: complex, rot: complex) -> list[complex]:
    return [center_local + units[k] / (s3 * rot) for k in range(6)]


def to_joint(z_local: complex, which: str) -> complex:
    if which == "p":
        return z_local
    return Q_CENTER + Q_LOCAL_TO_JOINT_ROT * z_local


def plot_pentagon(ax, which: str, res: int, *,
                  color_main: str, color_hex: str, color_deleted: str,
                  pentagon_idx: int) -> None:
    rot_N = get_rot(res)
    center_joint = 0 + 0j if which == "p" else Q_CENTER
    rot_l2j = 1 + 0j if which == "p" else Q_LOCAL_TO_JOINT_ROT

    # 6 wedge triangles around parent. Shade the deleted one.
    for j in range(6):
        a_l, b_l, c_l = 0 + 0j, units[j], units[(j + 1) % 6]
        a, b, c = (to_joint(a_l, which), to_joint(b_l, which),
                   to_joint(c_l, which))
        verts = [(a.real, a.imag), (b.real, b.imag), (c.real, c.imag)]
        if j == DELETED_TRIANGLE_J:
            ax.add_patch(Polygon(verts, closed=True, facecolor=color_deleted,
                                 edgecolor=color_deleted, linestyle="--",
                                 linewidth=0.8, alpha=0.45, zorder=1))
        else:
            ax.plot([v[0] for v in verts] + [verts[0][0]],
                    [v[1] for v in verts] + [verts[0][1]],
                    color=color_main, lw=0.6, alpha=0.4, zorder=1)

    # Cells at this resolution.
    view_center = 0.5 * (0 + Q_CENTER)
    view_radius = abs(Q_CENTER) / 2 + 1.55
    for digits, z_local in enumerate_cells(res):
        z_joint = to_joint(z_local, which)
        if abs(z_joint - view_center) > view_radius + 0.3:
            continue
        corners_local = hex_corners_local(z_local, rot_N)
        corners_joint = [to_joint(c, which) for c in corners_local]
        xs = [c.real for c in corners_joint] + [corners_joint[0].real]
        ys = [c.imag for c in corners_joint] + [corners_joint[0].imag]

        is_pentagon = len(digits) == 0 or all(d == 0 for d in digits)
        is_deleted_child = first_nonzero_digit(digits) == 1

        if is_pentagon:
            ax.plot(xs, ys, color=color_main, lw=2.0, zorder=4)
            ax.fill(xs, ys, color=color_main, alpha=0.18, zorder=3)
        elif is_deleted_child:
            ax.plot(xs, ys, color=color_deleted, lw=0.5, alpha=0.7, zorder=3)
            ax.fill(xs, ys, color=color_deleted, alpha=0.35, zorder=2)
        else:
            ax.plot(xs, ys, color=color_hex, lw=0.4, alpha=0.55, zorder=2)
            ax.plot(z_joint.real, z_joint.imag, ".", color=color_hex,
                    markersize=1.0, alpha=0.55, zorder=2)

    # Center marker + label + primary direction arrow.
    ax.plot(center_joint.real, center_joint.imag, "k.",
            markersize=6, zorder=7)
    label_offset = 0.08 * cmath.exp(
        1j * math.radians(_neighbor_angle_deg(0) if which == "p" else 0)
    )
    if which == "p":
        # Push label opposite Q's direction so it doesn't collide.
        label_offset = -0.1 * Q_CENTER / abs(Q_CENTER)
    else:
        # Push label opposite P's direction (joint = -Q_CENTER + Q_CENTER = away).
        away = (Q_CENTER - 0) / abs(Q_CENTER)
        label_offset = 0.1 * away
    ax.text(center_joint.real + label_offset.real,
            center_joint.imag + label_offset.imag,
            f"{which.upper()}={pentagon_idx}", fontsize=10,
            ha="center", va="center", zorder=7,
            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                      ec="none", alpha=0.85))

    primary_joint = rot_l2j * 1
    arrow_len = 0.42
    ax.annotate(
        "", xytext=(center_joint.real, center_joint.imag),
        xy=(center_joint.real + arrow_len * primary_joint.real,
            center_joint.imag + arrow_len * primary_joint.imag),
        arrowprops=dict(arrowstyle="-|>", color=color_main, lw=2.2,
                        mutation_scale=14),
        zorder=8,
    )


def plot_res(ax, res: int) -> None:
    rot_N = get_rot(res)

    plot_pentagon(ax, "p", res,
                  color_main="#c33", color_hex="#a55",
                  color_deleted="#f0a04a", pentagon_idx=P)
    plot_pentagon(ax, "q", res,
                  color_main="#36c", color_hex="#558",
                  color_deleted="#9bc4f0", pentagon_idx=Q)

    # Shared edge: from P (0,0) to Q (Q_CENTER).
    ax.plot([0, Q_CENTER.real], [0, Q_CENTER.imag],
            color="#222", lw=2.4, zorder=5, solid_capstyle="round")

    view_center = 0.5 * (0 + Q_CENTER)
    view_radius = abs(Q_CENTER) / 2 + 1.55
    ax.set_xlim(view_center.real - view_radius, view_center.real + view_radius)
    ax.set_ylim(view_center.imag - view_radius, view_center.imag + view_radius)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(
        f"res {res}  (|rot|²={abs(rot_N)**2:.0f}, "
        f"arg(rot)={math.degrees(cmath.phase(rot_N)):.1f}°)",
        fontsize=11,
    )


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, res in zip(axes.flat, (0, 1, 2, 3)):
        plot_res(ax, res)

    n_p = dodec.normals[P]
    n_q = dodec.normals[Q]
    fig.suptitle(
        f"mu3 cross-pentagon Eisenstein lattice — general unfold  ·  "
        f"P={P}  Q={Q}  (Q = neighbors[P][{K_INDEX}],  P = neighbors[Q][{J_INDEX}])\n"
        f"α_P = {ALPHA_P_DEG:.0f}°,  α_Q = {ALPHA_Q_DEG:.0f}°,  "
        f"Q_local→joint rotation = {ROT_DEG % 360:.0f}°,  "
        f"|Q_center| = {abs(Q_CENTER):.3f}\n"
        f"normal[{P}] = ({n_p[0]:+.3f}, {n_p[1]:+.3f}, {n_p[2]:+.3f})  ·  "
        f"normal[{Q}] = ({n_q[0]:+.3f}, {n_q[1]:+.3f}, {n_q[2]:+.3f})",
        fontsize=11, y=1.00,
    )
    plt.tight_layout()

    out = (Path(__file__).resolve().parent.parent / "figures"
           / "pentagon_lattice_pair_general.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")
    print(f"  P={P}, Q={Q}, k={K_INDEX}, j={J_INDEX}")
    print(f"  α_P={ALPHA_P_DEG}°, α_Q={ALPHA_Q_DEG}°, rotation={ROT_DEG % 360:.1f}°")


if __name__ == "__main__":
    main()
