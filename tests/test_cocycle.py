"""The cocycle laws, re-asserted with parametrized, readable failures.

The expected values live in ONE place — ``cross_pentagon.CORNER_EXPECTED``
/ ``PENTAGON_LOOP_EXPECTED``, also consumed by the import-time
``_check_cocycle`` — so the spec cannot drift between module and test.
These are the *exact* branch-cut-aware laws; see the comment block in
cross_pentagon.py for why they differ from the idealized cocycle identity.
"""

import pytest

from mu3 import dodec
from mu3.cross_pentagon import (
    CORNER_EXPECTED,
    DERIVED_STITCH,
    NEIGHBOR_ANGLE_IDX,
    PENTAGON_LOOP_EXPECTED,
    TAU,
    _corner_product,
    tau_between,
)
from mu3.eisenstein import UNITS, ZERO
from mu3.p6 import IDENTITY, P6


@pytest.mark.parametrize('p', range(12))
def test_edge_symmetry(p):
    for k in range(5):
        q = dodec.neighbors[p][k]
        assert tau_between(q, p) == tau_between(p, q).inverse()


@pytest.mark.parametrize('p', range(12))
def test_tau_maps_centers(p):
    """tau(p->q) sends q's center (in p's frame) to q's origin and p's
    origin to p's center in q's frame."""
    for k in range(5):
        q = dodec.neighbors[p][k]
        j = list(dodec.neighbors[q]).index(p)
        g = TAU[p][k]
        assert g.apply(UNITS[NEIGHBOR_ANGLE_IDX[k]]) == ZERO
        assert g.apply(ZERO) == UNITS[NEIGHBOR_ANGLE_IDX[j]]


@pytest.mark.parametrize('k', range(5))
def test_corner_classification(k):
    for p in range(12):
        assert _corner_product(p, k) == CORNER_EXPECTED[k], f'p={p}'


@pytest.mark.parametrize('p', range(12))
def test_pentagon_loop(p):
    cycle = [p, *dodec.neighbors[p], p]
    prod = IDENTITY
    for a, b in zip(cycle, cycle[1:]):
        prod = tau_between(a, b).compose(prod)
    assert prod == PENTAGON_LOOP_EXPECTED


def test_derived_stitch():
    """The +60 deg stitch is a theorem of the cross-pentagon table:
    the k=0 corner product equals g_{zeta,0} (the constant the retired
    float ``_maybe_stitch`` hardcoded)."""
    assert DERIVED_STITCH == P6(1, ZERO)
