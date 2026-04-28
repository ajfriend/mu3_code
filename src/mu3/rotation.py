"""3x3 rotation matrix from a source vector, target vector, and twist.

The rotation sends ``source`` to ``target`` along the great-circle shortest
path, then twists by ``twist_rad`` about the target axis (CCW viewed from
outside).
"""

import math

import numpy as np


def rotation_matrix(source, target, twist_rad):
    """Build the 3x3 rotation described in the module docstring."""
    s = source / np.linalg.norm(source)
    t = target / np.linalg.norm(target)
    return _twist(t, twist_rad) @ _align(s, t)


def _align(s, t):
    """Shortest-path rotation matrix taking unit ``s`` to unit ``t`` (Rodrigues)."""
    c = float(s @ t)
    if c < -1 + 1e-9:
        # Antipodal: 180-degree rotation about any axis perpendicular to s.
        seed = [1.0, 0.0, 0.0] if abs(s[0]) < 0.9 else [0.0, 1.0, 0.0]
        axis = np.cross(s, seed)
        axis /= np.linalg.norm(axis)
        K = _skew(axis)
        return np.eye(3) + 2 * (K @ K)
    K = _skew(np.cross(s, t))
    return np.eye(3) + K + (K @ K) / (1 + c)


def _twist(axis, angle):
    """Rotation matrix for ``angle`` about unit ``axis`` (Rodrigues)."""
    K = _skew(axis)
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


def _skew(v):
    """Skew-symmetric (cross-product) matrix of a 3-vector."""
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
