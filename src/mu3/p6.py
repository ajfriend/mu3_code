"""The group ``p6`` of hex-lattice symmetries, exactly.

Elements are maps ``z -> zeta**u * z + t`` with ``u`` an int mod 6
(a sixth-root-of-unity rotation) and ``t`` an exact Eisenstein
integer translation. Composition and inverse are pure ring
arithmetic:

    (u1, t1) o (u2, t2) = (u1 + u2,  zeta**u1 * t2 + t1)
    (u, t)**-1          = (-u,      -(zeta**-u * t))

Formally ``p6 = Z[omega] semidirect C6`` — the holonomy group of the
mu3 cone surface.
Every ``CROSS_PENTAGON`` transition is one of these elements; the
+60 deg intra-pentagon stitch is :data:`STITCH`.
"""

from .eisenstein import UNITS, ZERO, Eis


class P6:
    """The map ``z -> zeta**u * z + t``, ``u`` mod 6, ``t`` an Eis."""

    __slots__ = ('u', 't')

    def __init__(self, u: int, t: Eis):
        self.u = u % 6
        self.t = t

    def compose(self, other: 'P6') -> 'P6':
        """``self o other`` (``other`` applied first)."""
        return P6(self.u + other.u, UNITS[self.u] * other.t + self.t)

    def inverse(self) -> 'P6':
        ui = (-self.u) % 6
        return P6(ui, -(UNITS[ui] * self.t))

    def apply(self, z: Eis) -> Eis:
        return UNITS[self.u] * z + self.t

    def apply_scaled(self, z: Eis, gn: Eis) -> Eis:
        """Apply in a res-N scaled frame (``Z = z * gn``): the rotation
        part is scale-free, the translation scales by ``gn``. The home
        of that convention: every EXACT chart transition on scaled
        state goes through here (the float point-location witness
        mirrors it in complex arithmetic — ``_FloatWitness.hop``)."""
        return UNITS[self.u] * z + self.t * gn

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, P6):
            return NotImplemented
        return self.u == other.u and self.t == other.t

    def __hash__(self) -> int:
        return hash((self.u, self.t))

    def __repr__(self) -> str:
        return f'P6({self.u}, {self.t!r})'


IDENTITY = P6(0, ZERO)

# The +60 deg rotation about the origin: g_{zeta, 0}. This is the
# intra-pentagon stitch, derived (not assumed) from the cocycle in
# cross_pentagon._check_cocycle.
STITCH = P6(1, ZERO)


def rotation_about(u: int, c: Eis) -> P6:
    """Rotation by ``zeta**u`` about the point ``c``:
    ``z -> zeta**u * (z - c) + c``."""
    return P6(u, c - UNITS[u % 6] * c)
