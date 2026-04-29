"""Diagnostic: ring-1 walks from (0, 2) at res 1 using the corrected
pipeline -- walk → forward gnomonic project → sphere-to-cell handoff.

Five of the six walks succeed (cross-pentagon neighbors all reachable
via the generic 3D handoff). The sixth (D=6 into the deleted wedge)
still self-loops to source because mu3's +60° stitch sends z_n back to
the source's own z_C.

Output: ``figures/ring1_failing_simple.png``.
"""

import cmath
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from mu3 import cell_ring1
from mu3.cell import (
    _classify_stitched, _eisenstein_center, _stitch, _wedge_barycentric,
)
from mu3.cross_pentagon import z_to_cell
from mu3 import icosahedron
from mu3.face_lattice import digit_offset, get_rot, omega, s3, units


SOURCE_CELL = (0, 2)
RES = 1
BASE = 0


def project_gnomonic(z, base):
    if abs(z) < 1e-12:
        return icosahedron.vertices()[base].copy()
    z_s = _stitch(z)
    d = _classify_stitched(z_s)
    beta = _wedge_barycentric(z_s, d)
    V = icosahedron.vertices()
    n = icosahedron.vertex_neighbors()[base]
    n_cw = int(n[d - 2]); n_ccw = int(n[(d - 1) % 5])
    p = beta[0] * V[base] + beta[1] * V[n_cw] + beta[2] * V[n_ccw]
    return p / np.linalg.norm(p)


def _wedge_z_from_barycentric(beta, d):
    b_p, b_cw, b_ccw = float(beta[0]), float(beta[1]), float(beta[2])
    z_aligned = complex(b_cw + 0.5 * b_ccw, b_ccw * math.sqrt(3) / 2)
    return z_aligned * cmath.exp(1j * math.radians((d - 1) * 60.0))


def sphere_to_cell(p3d, res):
    V = icosahedron.vertices()
    base = int(np.argmax(V @ p3d))
    n = icosahedron.vertex_neighbors()[base]
    best_d, best_beta, best_min = -1, None, -2.0
    for d in range(2, 7):
        n_cw = int(n[d - 2]); n_ccw = int(n[(d - 1) % 5])
        M = np.column_stack([V[base], V[n_cw], V[n_ccw]])
        beta = np.linalg.solve(M, p3d)
        s = beta.sum()
        if abs(s) < 1e-12:
            continue
        beta = beta / s
        m = float(beta.min())
        if m > best_min:
            best_min = m; best_d = d; best_beta = beta
    z = _wedge_z_from_barycentric(best_beta, best_d)
    return z_to_cell(base, z, res)


def hex_corners(center: complex, rot: complex):
    return [center + units[k] / (s3 * rot) for k in range(6)]


def as_xy(zs):
    return [z.real for z in zs], [z.imag for z in zs]


def draw_hex(ax, center, rot, *, edge_color, face_color=None, lw=1.0,
             alpha=1.0, ls="-", zorder=2):
    corners = hex_corners(center, rot)
    xs, ys = as_xy(corners + [corners[0]])
    if face_color is not None:
        ax.fill(xs, ys, color=face_color, alpha=alpha, zorder=zorder)
    ax.plot(xs, ys, color=edge_color, lw=lw, ls=ls, zorder=zorder + 0.1)


def main():
    fig, ax = plt.subplots(figsize=(13, 13))
    ax.set_aspect("equal")
    ax.axis("off")

    rot1 = get_rot(RES)
    z_C = _eisenstein_center(SOURCE_CELL[1:])

    # ---- 6 unit triangles + deleted wedge highlight.
    for j in range(6):
        a, b, c = 0 + 0j, units[j], units[(j + 1) % 6]
        verts = [(a.real, a.imag), (b.real, b.imag), (c.real, c.imag)]
        if j == 0:
            ax.add_patch(Polygon(
                verts, closed=True, facecolor="#888",
                edgecolor="#555", linestyle="--", lw=0.7,
                alpha=0.25, zorder=1,
            ))
            cx = (a.real + b.real + c.real) / 3
            cy = (a.imag + b.imag + c.imag) / 3
            ax.text(cx, cy + 0.07, "deleted wedge",
                    ha="center", va="center", fontsize=9, color="#555",
                    style="italic", zorder=1.5)
        else:
            ax.plot([v[0] for v in verts] + [verts[0][0]],
                    [v[1] for v in verts] + [verts[0][1]],
                    color="black", lw=0.7, alpha=0.5, zorder=1)

    # ---- 5 res-1 cells; source highlighted, others labeled.
    expected = list(cell_ring1(SOURCE_CELL))
    for d in (2, 3, 4, 5, 6):
        z = digit_offset[d] / rot1
        cell = (BASE, d)
        is_source = (cell == SOURCE_CELL)
        is_expected = (cell in expected)
        if is_source:
            face = "#ffd9d9"; edge = "#c33"; lw = 2.2
        elif is_expected:
            face = "#d6f4dd"; edge = "#0a8a3a"; lw = 1.5
        else:
            face = "#f4f4f4"; edge = "#777"; lw = 0.8
        draw_hex(ax, z, rot1, edge_color=edge, face_color=face,
                 lw=lw, alpha=0.85 if is_source else 0.55, zorder=4)
        ax.plot(z.real, z.imag, ".", color=edge, markersize=4, zorder=5)
        nudge = 0.15 * cmath.exp(1j * cmath.phase(z))
        label = f"({BASE},{d})"
        if is_source:
            label += "\nsource"
        elif is_expected:
            label += "\nexpected\n(d=1 wedge)"
        ax.text(z.real + nudge.real, z.imag + nudge.imag, label,
                ha="center", va="center",
                fontsize=9 if is_source or is_expected else 8,
                color=edge,
                fontweight="bold" if is_source else "normal",
                zorder=7,
                bbox=dict(boxstyle="round,pad=0.18", fc="white",
                          ec=edge, lw=0.6, alpha=0.95))

    # ---- The 6 walks: project to 3D, hand off to sphere_to_cell.
    walk_info = []
    for D in range(1, 7):
        step = digit_offset[D] / rot1
        z_n = z_C + step
        p3d = project_gnomonic(z_n, BASE)
        try:
            nb = sphere_to_cell(p3d, RES)
        except Exception:
            nb = None
        walk_info.append((D, z_n, nb))

    # Color each walk by whether it correctly identifies an expected ring-1.
    for D, z_n, nb in walk_info:
        in_expected = nb in expected
        is_source = nb == SOURCE_CELL
        is_phantom = nb is not None and len(nb) > 1 and nb[1] == 1 and nb not in expected
        if in_expected:
            color = "#0a8a3a"
            tag = f"D={D} → {nb} ✓"
        elif is_source:
            color = "#999"
            tag = f"D={D} → SELF"
        elif is_phantom:
            color = "#a26ec3"
            tag = f"D={D} → {nb} (phantom)"
        elif nb is None:
            color = "#d96b0f"
            tag = f"D={D} → ERROR"
        else:
            color = "#d96b0f"
            tag = f"D={D} → {nb} (wrong)"
        ax.annotate("", xytext=(z_C.real, z_C.imag),
                    xy=(z_n.real, z_n.imag),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    lw=2.0, mutation_scale=14, alpha=0.85),
                    zorder=8)
        ax.plot(z_n.real, z_n.imag, "X", color=color,
                markersize=10, markeredgewidth=1.6, zorder=10)
        # tag near arrowhead
        out_dir = (z_n - z_C) / abs(z_n - z_C) if abs(z_n - z_C) > 1e-9 else 0
        tag_pos = z_n + out_dir * 0.13
        ax.text(tag_pos.real, tag_pos.imag, tag,
                ha="center", va="center", fontsize=8.5, color=color,
                fontweight="bold", zorder=11,
                bbox=dict(boxstyle="round,pad=0.18", fc="white",
                          ec=color, lw=0.6, alpha=0.95))

    # Pentagon center marker.
    ax.plot(0, 0, "o", color="black", markerfacecolor="#888",
            markersize=10, zorder=6)
    ax.text(-0.05, -0.06, "(0,) pentagon",
            ha="right", va="top", fontsize=9, color="#444", zorder=6)

    # View bounds.
    R = 1.55
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)

    ax.set_title(
        f"Ring-1 from {SOURCE_CELL} at res {RES}: walk → gnomonic project → "
        "sphere_to_cell handoff.\n"
        "Green = correct ring-1 hit. Gray = self-loop "
        "(D=6 stitches z_n back to source's z_C).\n"
        "5 of 6 walks reach correct cells (incl. cross-pentagon). "
        "(0, 6) is the only missed neighbor; would need twin-rotation.",
        fontsize=10,
    )

    out = (Path(__file__).resolve().parent.parent / "figures"
           / "ring1_failing_simple.png")
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
