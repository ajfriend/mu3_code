# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy"]
# ///
"""Verify the geometric 1.2584 corner-aspect floor.

Sample aspect ratio at a sequence of points approaching a face corner
along the medial line. If the corner floor is geometric, aspect should
approach 1.2584 regardless of (α, η, κ).
"""

import sys
sys.path.insert(0, "src")

import math
import numpy as np

from probe_alpha_slerp_aspect import forward


def aspect_at(beta, alpha, eta, kappa, h=1e-5):
    inv_sqrt3 = 1.0 / math.sqrt(3.0)
    b1, b2 = beta[1], beta[2]
    def F_(x, y):
        return forward(alpha, eta, kappa, 0.0, 0.0,
                       np.array([1.0 - x - y, x, y]))
    dvx = (F_(b1 + h, b2) - F_(b1 - h, b2)) / (2 * h)
    dvy = (F_(b1 - h * inv_sqrt3, b2 + 2 * h * inv_sqrt3)
           - F_(b1 + h * inv_sqrt3, b2 - 2 * h * inv_sqrt3)) / (2 * h)
    J = np.stack([dvx, dvy], axis=1)
    sv = np.linalg.svd(J, compute_uv=False)
    return sv[0] / sv[1]


def main():
    print(f"{'eps':>10}  {'aspect (1.149,...)':>20}  {'aspect (0.737,...)':>20}")
    # Approach corner V0 (b = (1, 0, 0)) along the medial line:
    # b = (1 - 2*eps, eps, eps).
    for eps in [0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]:
        beta = np.array([1 - 2 * eps, eps, eps])
        a1 = aspect_at(beta, 1.149, 0.121, 0.170)
        a2 = aspect_at(beta, 0.737, -0.708, -3.265)
        print(f"{eps:>10.4f}  {a1:>20.6f}  {a2:>20.6f}")


if __name__ == "__main__":
    main()
