import numpy as np
import pytest

from mu3 import icosahedron
from mu3.projection import (
    AlphaIVEAProjection,
    AlphaOnlySlerp,
    AlphaSlerp,
    AlphaSlerpExtended,
    IVEAProjection,
    KarcherPolyProjection,
    KarcherProjection,
    LambertBaryProjection,
    TunedKarcherProjection,
)

PROJECTION_CLASSES = [
    AlphaSlerp,
    AlphaOnlySlerp,
    IVEAProjection,
    AlphaIVEAProjection,  # at default α=1.0 it's exactly IVEA
    LambertBaryProjection,
    KarcherProjection,
    TunedKarcherProjection,  # at default (η=0.121, κ=0.170)
    KarcherPolyProjection,   # at default kappas=() it's exactly Karcher
    AlphaSlerpExtended,      # at default (λ, μ)=(0,0) it's exactly AlphaSlerp
]


def _proj_for_face(cls, face: int):
    V = icosahedron.vertices()
    F = icosahedron.faces()
    return cls(V[F[face, 0]], V[F[face, 1]], V[F[face, 2]])


@pytest.mark.parametrize("cls", PROJECTION_CLASSES)
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_roundtrip_random_interior(cls, seed):
    proj = _proj_for_face(cls, 0)
    rng = np.random.default_rng(seed)
    betas = rng.dirichlet([2.0, 2.0, 2.0], size=200)
    for b in betas:
        p = proj.to_sphere(b)
        assert np.isclose(np.linalg.norm(p), 1.0, atol=1e-12)
        b_back = proj.to_bary(p)
        assert np.allclose(b_back, b, atol=1e-10)


@pytest.mark.parametrize("cls", PROJECTION_CLASSES)
@pytest.mark.parametrize("corner", [
    np.array([1.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0]),
    np.array([0.0, 0.0, 1.0]),
])
def test_corners_are_fixed_points(cls, corner):
    proj = _proj_for_face(cls, 0)
    p = proj.to_sphere(corner)
    # Forward at a corner must land on the corresponding face vertex
    assert np.allclose(p, proj.V[int(np.argmax(corner))], atol=1e-14)
    b_back = proj.to_bary(p)
    assert np.allclose(b_back, corner, atol=1e-12)


@pytest.mark.parametrize("cls", PROJECTION_CLASSES)
@pytest.mark.parametrize("beta", [
    np.array([0.5, 0.5, 0.0]),
    np.array([0.5, 0.0, 0.5]),
    np.array([0.0, 0.5, 0.5]),
])
def test_edge_midpoints_roundtrip(cls, beta):
    proj = _proj_for_face(cls, 0)
    p = proj.to_sphere(beta)
    b_back = proj.to_bary(p)
    assert np.allclose(b_back, beta, atol=1e-10)


@pytest.mark.parametrize("cls", PROJECTION_CLASSES)
def test_centroid_roundtrip(cls):
    proj = _proj_for_face(cls, 0)
    beta = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    p = proj.to_sphere(beta)
    b_back = proj.to_bary(p)
    assert np.allclose(b_back, beta, atol=1e-12)


@pytest.mark.parametrize("cls", PROJECTION_CLASSES)
@pytest.mark.parametrize("face", list(range(20)))
def test_per_face_roundtrip(cls, face):
    proj = _proj_for_face(cls, face)
    rng = np.random.default_rng(1000 + face)
    betas = rng.dirichlet([2.0, 2.0, 2.0], size=20)
    for b in betas:
        p = proj.to_sphere(b)
        b_back = proj.to_bary(p)
        assert np.allclose(b_back, b, atol=1e-10)


@pytest.mark.parametrize("cls", PROJECTION_CLASSES)
@pytest.mark.parametrize("eps", [1e-3, 1e-5, 1e-7, 1e-10])
def test_near_corner_roundtrip_no_singularity(cls, eps):
    """Near-corner inputs must invert without raising and must roundtrip.

    Regression for the singular-Jacobian failure mode the inherited
    2D FD-Newton in AlphaSlerp hits when η=κ=0 (i.e., for AlphaOnlySlerp
    before the 1D fast-path override). All 20 faces × 3 corners × 4 ε.
    """
    near_corners = [
        np.array([1.0 - 2 * eps, eps, eps]),
        np.array([eps, 1.0 - 2 * eps, eps]),
        np.array([eps, eps, 1.0 - 2 * eps]),
    ]
    for face in range(20):
        proj = _proj_for_face(cls, face)
        for b in near_corners:
            p = proj.to_sphere(b)
            b_back = proj.to_bary(p)  # must not raise
            assert np.allclose(b_back, b, atol=1e-9), (face, eps, b, b_back)


def test_alpha_ivea_at_alpha_1_is_ivea():
    """At α=1, the cubic collapses to identity; output must match IVEA."""
    V = icosahedron.vertices()
    F = icosahedron.faces()
    rng = np.random.default_rng(42)
    for face in range(20):
        ivea = IVEAProjection(V[F[face, 0]], V[F[face, 1]], V[F[face, 2]])
        aivea = AlphaIVEAProjection(V[F[face, 0]], V[F[face, 1]], V[F[face, 2]],
                                     alpha=1.0)
        for b in rng.dirichlet([2.0, 2.0, 2.0], size=20):
            p_ivea = ivea.to_sphere(b)
            p_aivea = aivea.to_sphere(b)
            assert np.allclose(p_ivea, p_aivea, atol=1e-13), (face, b)
            b_back_ivea = ivea.to_bary(p_ivea)
            b_back_aivea = aivea.to_bary(p_aivea)
            assert np.allclose(b_back_ivea, b_back_aivea, atol=1e-12), (face, b)


@pytest.mark.parametrize("alpha", [0.7, 0.9, 1.2, 1.5])
def test_alpha_ivea_roundtrip_off_default(alpha):
    """Forward-then-inverse must recover β at non-default α values."""
    V = icosahedron.vertices()
    F = icosahedron.faces()
    rng = np.random.default_rng(1000 + int(alpha * 100))
    for face in range(20):
        proj = AlphaIVEAProjection(V[F[face, 0]], V[F[face, 1]], V[F[face, 2]],
                                    alpha=alpha)
        for b in rng.dirichlet([2.0, 2.0, 2.0], size=20):
            p = proj.to_sphere(b)
            assert np.isclose(np.linalg.norm(p), 1.0, atol=1e-12)
            b_back = proj.to_bary(p)
            assert np.allclose(b_back, b, atol=1e-10), (face, alpha, b, b_back)


def test_tuned_karcher_at_zero_params_is_karcher():
    """At (η, κ) = (0, 0), the cubic correction collapses to 1; output
    must match plain Karcher exactly."""
    V = icosahedron.vertices()
    F = icosahedron.faces()
    rng = np.random.default_rng(42)
    for face in range(20):
        karcher = KarcherProjection(V[F[face, 0]], V[F[face, 1]], V[F[face, 2]])
        tuned = TunedKarcherProjection(V[F[face, 0]], V[F[face, 1]], V[F[face, 2]],
                                        eta=0.0, kappa=0.0)
        for b in rng.dirichlet([2.0, 2.0, 2.0], size=20):
            p_k = karcher.to_sphere(b)
            p_t = tuned.to_sphere(b)
            assert np.allclose(p_k, p_t, atol=1e-12), (face, b, p_k, p_t)
            b_back_k = karcher.to_bary(p_k)
            b_back_t = tuned.to_bary(p_t)
            assert np.allclose(b_back_k, b_back_t, atol=1e-11), (face, b)


@pytest.mark.parametrize("eta,kappa", [
    (0.121, 0.170),  # AlphaSlerp's optimum
    (0.05, -0.05),   # arbitrary off-default
    (0.2, 0.0),      # η-only
    (0.0, 0.2),      # κ-only
])
def test_tuned_karcher_roundtrip_off_default(eta, kappa):
    """Forward-then-inverse must recover β at non-trivial (η, κ)."""
    V = icosahedron.vertices()
    F = icosahedron.faces()
    rng = np.random.default_rng(int((eta + kappa) * 1000) + 7)
    for face in range(20):
        proj = TunedKarcherProjection(V[F[face, 0]], V[F[face, 1]], V[F[face, 2]],
                                       eta=eta, kappa=kappa)
        for b in rng.dirichlet([2.0, 2.0, 2.0], size=20):
            p = proj.to_sphere(b)
            assert np.isclose(np.linalg.norm(p), 1.0, atol=1e-12)
            b_back = proj.to_bary(p)
            assert np.allclose(b_back, b, atol=1e-9), (face, eta, kappa, b, b_back)


def test_karcher_poly_at_empty_kappas_is_karcher():
    """At kappas=(), the polynomial correction is identically zero;
    output must match plain Karcher exactly."""
    V = icosahedron.vertices()
    F = icosahedron.faces()
    rng = np.random.default_rng(7)
    for face in range(20):
        karcher = KarcherProjection(V[F[face, 0]], V[F[face, 1]], V[F[face, 2]])
        poly = KarcherPolyProjection(V[F[face, 0]], V[F[face, 1]], V[F[face, 2]],
                                      kappas=())
        for b in rng.dirichlet([2.0, 2.0, 2.0], size=20):
            p_k = karcher.to_sphere(b)
            p_p = poly.to_sphere(b)
            assert np.allclose(p_k, p_p, atol=1e-12), (face, b)
            assert np.allclose(karcher.to_bary(p_k), poly.to_bary(p_p), atol=1e-11)


@pytest.mark.parametrize("kappas", [
    (0.1,),                # κ₁ only — like TunedKarcher's κ
    (0.0, 0.1),            # κ₂ only
    (-0.2, 0.15),          # both signs
    (0.05, -0.05, 0.02),   # cubic too
])
def test_karcher_poly_roundtrip_off_default(kappas):
    """Forward-then-inverse must recover β at non-trivial polynomial corrections."""
    V = icosahedron.vertices()
    F = icosahedron.faces()
    rng = np.random.default_rng(int(sum(abs(k) * 100 for k in kappas)) + 1)
    for face in range(20):
        proj = KarcherPolyProjection(V[F[face, 0]], V[F[face, 1]], V[F[face, 2]],
                                      kappas=kappas)
        for b in rng.dirichlet([2.0, 2.0, 2.0], size=20):
            p = proj.to_sphere(b)
            assert np.isclose(np.linalg.norm(p), 1.0, atol=1e-12)
            b_back = proj.to_bary(p)
            assert np.allclose(b_back, b, atol=1e-9), (face, kappas, b, b_back)


def test_alpha_slerp_extended_at_zero_higher_order_is_alphaslerp():
    """At (λ, μ) = (0, 0) the higher-order terms vanish; output must
    match plain AlphaSlerp exactly."""
    V = icosahedron.vertices()
    F = icosahedron.faces()
    rng = np.random.default_rng(1717)
    for face in range(20):
        plain = AlphaSlerp(V[F[face, 0]], V[F[face, 1]], V[F[face, 2]])
        extended = AlphaSlerpExtended(V[F[face, 0]], V[F[face, 1]], V[F[face, 2]],
                                       lambd=0.0, mu=0.0)
        for b in rng.dirichlet([2.0, 2.0, 2.0], size=20):
            p_plain = plain.to_sphere(b)
            p_ext = extended.to_sphere(b)
            assert np.allclose(p_plain, p_ext, atol=1e-13), (face, b)
            b_back_plain = plain.to_bary(p_plain)
            b_back_ext = extended.to_bary(p_ext)
            assert np.allclose(b_back_plain, b_back_ext, atol=1e-12), (face, b)


@pytest.mark.parametrize("lambd,mu", [
    (0.1, 0.05),
    (-0.2, 0.0),
    (0.0, 0.15),
    (0.3, -0.1),
])
def test_alpha_slerp_extended_roundtrip_off_default(lambd, mu):
    """Forward-then-inverse must recover β at non-trivial (λ, μ)."""
    V = icosahedron.vertices()
    F = icosahedron.faces()
    rng = np.random.default_rng(int(abs(lambd * 1000) + abs(mu * 1000)) + 91)
    for face in range(20):
        proj = AlphaSlerpExtended(V[F[face, 0]], V[F[face, 1]], V[F[face, 2]],
                                   lambd=lambd, mu=mu)
        for b in rng.dirichlet([2.0, 2.0, 2.0], size=20):
            p = proj.to_sphere(b)
            assert np.isclose(np.linalg.norm(p), 1.0, atol=1e-12)
            b_back = proj.to_bary(p)
            assert np.allclose(b_back, b, atol=1e-9), (face, lambd, mu, b, b_back)
