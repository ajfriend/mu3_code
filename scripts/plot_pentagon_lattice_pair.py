"""Two adjacent pentagons unfolded into a joint Eisenstein flat frame.

Picks two real merged-corner-neighbor pentagons P and Q (where P's
primary direction points at Q and Q's primary points back at P) and
unfolds them into a single 2D flat coordinate system:

- P at the origin, primary direction along +x.
- Q at (1, 0), primary direction along -x (Q's local frame is rotated
  180° around the shared edge).
- The shared icosahedron edge runs from (0, 0) to (1, 0).
- P's deleted wedge sits above the shared edge (joint angles 0°-60°
  from P).
- Q's deleted wedge sits below it (joint angles 180°-240° from Q).

In this view, re-rooting from P's frame to Q's frame is a single
shift+rotate: ``z_q_local = (1 - z_p_joint) / (-1) = z_p_joint - 1``,
or equivalently ``z_q_joint = 1 - z_p_local``. The two pentagons'
lattices share the spoke from (0, 0) to (1, 0) exactly.

Output: ``figures/pentagon_lattice_pair.png``.
"""

import cmath
import itertools
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from mu3 import dodec
from mu3.face_lattice import digit_offset, get_rot, omega, s3, units


P = 0
Q = dodec.neighbors[P][0]
assert dodec.neighbors[Q][0] == P, "P and Q must be merged-corner neighbors"


# Joint frame: P at origin (local frame == joint frame); Q at (1, 0) with
# its local frame rotated 180° about its center.
P_CENTER = 0 + 0j
Q_CENTER = 1 + 0j
Q_LOCAL_TO_JOINT_ROT = -1 + 0j  # exp(i * pi)

# Deleted wedge: j=0 spans local angles [0°, 60°) (immediately CCW of primary).
DELETED_TRIANGLE_J = 0


def first_nonzero_digit(digits):
    for d in digits:
        if d != 0:
            return d
    return None


def enumerate_cells(res: int):
    """Yield (digits, z_local) for every res-N descendant of the parent
    pentagon (including digit-1 paths)."""
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


def plot_pentagon(ax, which: str, res: int, view_radius: float, *,
                  color_main: str, color_hex: str, color_deleted: str,
                  pentagon_idx: int) -> None:
    rot_N = get_rot(res)
    center_joint = P_CENTER if which == "p" else Q_CENTER
    rot_l2j = 1 + 0j if which == "p" else Q_LOCAL_TO_JOINT_ROT

    # Six wedge triangles around the parent. Shade the deleted one.
    for j in range(6):
        a_l, b_l, c_l = 0 + 0j, units[j], units[(j + 1) % 6]
        a, b, c = (to_joint(a_l, which), to_joint(b_l, which),
                   to_joint(c_l, which))
        verts = [(a.real, a.imag), (b.real, b.imag), (c.real, c.imag)]
        if j == DELETED_TRIANGLE_J:
            ax.add_patch(Polygon(verts, closed=True, facecolor=color_deleted,
                                 edgecolor=color_deleted,
                                 linestyle="--", linewidth=0.8,
                                 alpha=0.45, zorder=1))
        else:
            ax.plot([v[0] for v in verts] + [verts[0][0]],
                    [v[1] for v in verts] + [verts[0][1]],
                    color=color_main, lw=0.6, alpha=0.4, zorder=1)

    # Cells at this resolution.
    for digits, z_local in enumerate_cells(res):
        z_joint = to_joint(z_local, which)
        if abs(z_joint - 0.5) > view_radius:
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
    label_dx = -0.06 if which == "p" else 0.06
    ha = "right" if which == "p" else "left"
    ax.text(center_joint.real + label_dx, center_joint.imag - 0.06,
            f"{which.upper()}={pentagon_idx}", fontsize=10, ha=ha, va="top",
            zorder=7,
            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                      ec="none", alpha=0.85))

    primary_joint = rot_l2j * 1
    arrow_len = 0.4
    ax.annotate(
        "", xytext=(center_joint.real, center_joint.imag),
        xy=(center_joint.real + arrow_len * primary_joint.real,
            center_joint.imag + arrow_len * primary_joint.imag),
        arrowprops=dict(arrowstyle="-|>", color=color_main, lw=2.2,
                        mutation_scale=14),
        zorder=8,
    )


def plot_res(ax, res: int, view_radius: float) -> None:
    rot_N = get_rot(res)

    plot_pentagon(ax, "p", res, view_radius,
                  color_main="#c33", color_hex="#a55",
                  color_deleted="#f0a04a", pentagon_idx=P)
    plot_pentagon(ax, "q", res, view_radius,
                  color_main="#36c", color_hex="#558",
                  color_deleted="#9bc4f0", pentagon_idx=Q)

    # Shared edge: from (0, 0) to (1, 0), bold.
    ax.plot([0, 1], [0, 0], color="#222", lw=2.4, zorder=5,
            solid_capstyle="round")

    ax.set_xlim(0.5 - view_radius, 0.5 + view_radius)
    ax.set_ylim(-view_radius, view_radius)
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
        plot_res(ax, res, view_radius=1.55)

    n_p = dodec.normals[P]
    n_q = dodec.normals[Q]
    fig.suptitle(
        f"mu3 cross-pentagon Eisenstein lattice  ·  "
        f"pentagons {P} (red) and {Q} (blue) are merged-corner neighbors\n"
        f"primary directions point at each other along the shared edge; "
        f"deleted wedges sit on opposite sides\n"
        f"normal[{P}] = ({n_p[0]:+.3f}, {n_p[1]:+.3f}, {n_p[2]:+.3f})  ·  "
        f"normal[{Q}] = ({n_q[0]:+.3f}, {n_q[1]:+.3f}, {n_q[2]:+.3f})",
        fontsize=11, y=1.00,
    )
    plt.tight_layout()

    out = Path(__file__).resolve().parent.parent / "figures" / "pentagon_lattice_pair.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
