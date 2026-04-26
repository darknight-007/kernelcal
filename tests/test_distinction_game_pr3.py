"""
PR-3 tests for ``kernelcal.distinction_game``.

Covers the three new pieces of public API:

1. **Bayesian-Dirichlet Q_s refit** (``q_s_fit.py``).
2. **EM joint fit of (λ, Q_s)** (``kernel_mix.fit_kernel_mix_em`` plus
   ``fit_kernel_mix`` SLSQP path).
3. **Spectral smoothing + consistency score** (``spectral.py``).

Plus an end-to-end ``fit_distinction_game`` smoke test on a serialised
SceneGraph dict.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pytest

from kernelcal.distinction_game import (
    DistinctionGameFit,
    KernelClaim,
    PHX_URBAN_V0,
    Taxonomy,
    bayesian_refit_q_s,
    bayesian_refit_q_s_table,
    build_scene_graph,
    count_evidence_triples,
    default_q_s,
    default_q_s_table,
    fit_distinction_game,
    fit_kernel_mix,
    fit_kernel_mix_em,
    graph_smooth_posteriors,
    posteriors_array_from_scene_graph,
    spectral_consistency_score,
    uniform_lambdas,
)
from kernelcal.distinction_game.q_s import ConfusionMatrix


# ---------------------------------------------------------------------------
# Tiny taxonomy + synthetic kernels for hand-computable checks
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_taxonomy() -> Taxonomy:
    return Taxonomy(name="t3", categories=("a", "b", "c"))


@pytest.fixture
def perfect_kernel(tiny_taxonomy):
    """A 'perfect' kernel: native label == ground truth, no noise."""
    m = np.eye(3)  # 3x3 identity, columns sum to 1
    return ConfusionMatrix(
        source_id="perfect",
        taxonomy=tiny_taxonomy,
        native_labels=("a", "b", "c"),
        matrix=m,
    )


@pytest.fixture
def noisy_kernel(tiny_taxonomy):
    """A 'mostly correct but smeary' kernel."""
    m = np.array([
        [0.7, 0.15, 0.15],
        [0.2, 0.7,  0.2 ],
        [0.1, 0.15, 0.65],
    ])
    return ConfusionMatrix(
        source_id="noisy",
        taxonomy=tiny_taxonomy,
        native_labels=("a", "b", "c"),
        matrix=m,
    )


@pytest.fixture
def flat_kernel(tiny_taxonomy):
    """A kernel that's noise-only — every native label fires equally
    in every category."""
    m = np.full((3, 3), 1.0 / 3.0)
    return ConfusionMatrix(
        source_id="flat",
        taxonomy=tiny_taxonomy,
        native_labels=("a", "b", "c"),
        matrix=m,
    )


# ---------------------------------------------------------------------------
# Bayesian-Dirichlet refit
# ---------------------------------------------------------------------------

class TestBayesianRefit:
    def test_alpha_infinity_returns_prior(self, perfect_kernel):
        """With ``alpha=1e9`` and modest counts, the refit must be
        (essentially) the prior — the prior dominates."""
        N = np.array([
            [10.0, 0.0,  0.0],
            [ 0.0, 10.0, 0.0],
            [ 0.0, 0.0,  10.0],
        ])
        post = bayesian_refit_q_s(perfect_kernel, N, alpha=1e9)
        np.testing.assert_allclose(post.matrix, perfect_kernel.matrix, atol=1e-6)

    def test_alpha_zero_returns_empirical(self, noisy_kernel):
        """With ``alpha=0`` the refit is the empirical MLE
        (column-normalised counts)."""
        N = np.array([
            [50.0, 5.0,   1.0],
            [10.0, 90.0,  10.0],
            [ 5.0, 10.0,  60.0],
        ])
        post = bayesian_refit_q_s(noisy_kernel, N, alpha=0.0)
        expected = N / N.sum(axis=0, keepdims=True)
        np.testing.assert_allclose(post.matrix, expected, atol=1e-9)
        # And the result is still column-stochastic (the post-init
        # validator would have raised otherwise).
        np.testing.assert_allclose(post.matrix.sum(axis=0), 1.0, atol=1e-9)

    def test_columns_remain_stochastic(self, noisy_kernel):
        N = np.random.default_rng(0).integers(0, 50, size=(3, 3)).astype(float)
        post = bayesian_refit_q_s(noisy_kernel, N, alpha=5.0)
        np.testing.assert_allclose(post.matrix.sum(axis=0), 1.0, atol=1e-9)

    def test_refit_pulls_toward_data(self, noisy_kernel):
        """Strong, consistent evidence about category 'a' must shift
        column 0 of the posterior closer to that data than the prior."""
        # Data says: when truth=a, native label is *almost always* 'a'.
        N = np.zeros((3, 3))
        N[0, 0] = 1000.0
        post = bayesian_refit_q_s(noisy_kernel, N, alpha=5.0)
        prior_col0 = noisy_kernel.matrix[:, 0]   # (0.7, 0.2, 0.1)
        post_col0 = post.matrix[:, 0]
        # Posterior P(native='a' | truth=a) > prior P(native='a' | truth=a)
        assert post_col0[0] > prior_col0[0]
        assert post_col0[0] > 0.95

    def test_refit_table_keeps_unobserved_at_prior(self, perfect_kernel, noisy_kernel):
        """Sources with zero counts retain their priors."""
        prior_table = {"perfect": perfect_kernel, "noisy": noisy_kernel}
        N_perfect = 100 * np.eye(3)
        result = bayesian_refit_q_s_table(
            prior_table,
            counts_by_source={"perfect": N_perfect},
            alpha=10.0,
        )
        # 'noisy' had no data — its posterior is the prior.
        np.testing.assert_allclose(
            result.table["noisy"].matrix, noisy_kernel.matrix, atol=1e-9,
        )
        assert result.n_evidence["perfect"] == int(N_perfect.sum())
        assert result.n_evidence["noisy"] == 0

    def test_negative_counts_rejected(self, perfect_kernel):
        N = -np.ones((3, 3))
        with pytest.raises(ValueError, match="non-negative"):
            bayesian_refit_q_s(perfect_kernel, N)


class TestCountEvidenceTriples:
    def test_aggregates_weights(self, perfect_kernel, tiny_taxonomy):
        triples = [
            ("perfect", "a", 0, 1.0),
            ("perfect", "a", 0, 0.5),     # weighted partial vote
            ("perfect", "b", 1, 2.0),
            ("perfect", "c", 0, 1.0),     # mis-classification
        ]
        counts = count_evidence_triples(
            triples,
            prior_table={"perfect": perfect_kernel},
            taxonomy=tiny_taxonomy,
        )
        N = counts["perfect"]
        assert N[0, 0] == pytest.approx(1.5)
        assert N[1, 1] == pytest.approx(2.0)
        assert N[2, 0] == pytest.approx(1.0)
        assert N.sum() == pytest.approx(4.5)

    def test_unknown_source_dropped(self, perfect_kernel, tiny_taxonomy):
        triples = [("ghost", "a", 0, 1.0), ("perfect", "a", 0, 1.0)]
        counts = count_evidence_triples(
            triples,
            prior_table={"perfect": perfect_kernel},
            taxonomy=tiny_taxonomy,
        )
        assert "ghost" not in counts
        assert counts["perfect"][0, 0] == pytest.approx(1.0)

    def test_unknown_native_label_dropped_by_default(
        self, perfect_kernel, tiny_taxonomy,
    ):
        triples = [("perfect", "z", 0, 1.0)]
        counts = count_evidence_triples(
            triples,
            prior_table={"perfect": perfect_kernel},
            taxonomy=tiny_taxonomy,
        )
        np.testing.assert_array_equal(counts["perfect"], np.zeros((3, 3)))

    def test_unknown_native_label_raises_when_strict(
        self, perfect_kernel, tiny_taxonomy,
    ):
        triples = [("perfect", "z", 0, 1.0)]
        with pytest.raises(KeyError):
            count_evidence_triples(
                triples,
                prior_table={"perfect": perfect_kernel},
                taxonomy=tiny_taxonomy,
                drop_unknown_native=False,
            )


# ---------------------------------------------------------------------------
# fit_kernel_mix SLSQP path
# ---------------------------------------------------------------------------

def _claim(source: str, native: str, score: float = 1.0) -> KernelClaim:
    poly = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]
    return KernelClaim.from_polygon(source, native, score, poly)


class TestSLSQPLambdaFit:
    def test_informative_kernel_outweighs_noise(
        self, noisy_kernel, flat_kernel, tiny_taxonomy,
    ):
        """When 'noisy' is informative and 'flat' is pure noise, the
        SLSQP fit must give clearly higher weight to 'noisy'.

        We use the *noisy* kernel (max diagonal ~0.7) rather than the
        identity-perfect kernel because identity confusion makes the
        log-likelihood saturate above some λ-threshold, leaving the
        SLSQP search on a flat plateau (any λ_perfect > ~0.5 is
        already optimal). The noisy kernel produces a strictly concave
        likelihood on the simplex, so the optimum is well-defined."""
        q_s_table = {"noisy": noisy_kernel, "flat": flat_kernel}

        rng = np.random.default_rng(0)
        regions = []
        true_cs = []
        for _ in range(200):
            c = int(rng.integers(0, 3))
            true_cs.append(c)
            # 'noisy' fires its native label according to its own Q_s.
            native_noisy = ("a", "b", "c")[
                int(rng.choice(3, p=noisy_kernel.matrix[:, c]))
            ]
            # 'flat' just emits a uniformly random native label.
            native_flat = ("a", "b", "c")[int(rng.integers(0, 3))]
            regions.append([
                _claim("noisy", native_noisy, 1.0),
                _claim("flat",  native_flat,  1.0),
            ])

        supervised_pairs = list(zip(regions, true_cs))
        fit = fit_kernel_mix(
            ["noisy", "flat"], q_s_table,
            supervised_pairs=supervised_pairs,
        )
        assert fit.method == "slsqp"
        # 'flat' contributes literally zero log-evidence (its rows are
        # constant), so any non-trivial fit must drive its weight far
        # below the informative kernel's.
        assert fit.lambda_for("noisy") > 0.7
        assert fit.lambda_for("flat") < 0.3
        np.testing.assert_allclose(fit.lambdas.sum(), 1.0, atol=1e-6)

    def test_no_pairs_falls_back_to_uniform(self, perfect_kernel, flat_kernel):
        q_s_table = {"perfect": perfect_kernel, "flat": flat_kernel}
        fit = fit_kernel_mix(["perfect", "flat"], q_s_table)
        assert fit.method == "uniform"
        np.testing.assert_allclose(fit.lambdas, [0.5, 0.5])


class TestKernelMixEM:
    def test_em_recovers_q_s_with_known_anchors(
        self, tiny_taxonomy, noisy_kernel, perfect_kernel,
    ):
        """If we feed enough OSM-anchored regions, the EM Q_s update
        on the 'noisy' kernel must move its column 0 closer to the
        empirical truth."""
        rng = np.random.default_rng(123)
        prior_table = {"noisy": noisy_kernel, "perfect": perfect_kernel}

        # Generate regions where truth is sampled uniformly; 'noisy'
        # fires its native label per its true Q (we use an *easier*
        # generative kernel here: 0.9 on the diagonal so the empirical
        # column 0 mass on label 'a' should be near 0.9, beating the
        # prior's 0.7).
        true_noisy = np.array([
            [0.9, 0.10, 0.10],
            [0.05, 0.85, 0.10],
            [0.05, 0.05, 0.80],
        ])

        regions = []
        anchors = []
        for _ in range(300):
            c = int(rng.integers(0, 3))
            anchors.append(c)
            # Sample the noisy kernel's native label according to true_noisy.
            r_idx = int(rng.choice(3, p=true_noisy[:, c]))
            native_noisy = ("a", "b", "c")[r_idx]
            # 'perfect' just emits the right native label.
            native_perfect = ("a", "b", "c")[c]
            regions.append([
                _claim("noisy",   native_noisy,   1.0),
                _claim("perfect", native_perfect, 1.0),
            ])

        result = fit_kernel_mix_em(
            regions, prior_table,
            taxonomy=tiny_taxonomy,
            sources=["noisy", "perfect"],
            anchors=anchors,
            fit_q_s=True,
            alpha_q_s=2.0,
            max_iter=10,
        )
        post_noisy = result.q_s_table["noisy"].matrix
        # The diagonal of the noisy kernel should pull toward the
        # generative truth (0.9), away from the prior (0.7).
        assert post_noisy[0, 0] > noisy_kernel.matrix[0, 0]
        assert post_noisy[0, 0] > 0.80
        # Mixture should still favour 'perfect' (it's literally perfect).
        assert result.mix.lambda_for("perfect") > result.mix.lambda_for("noisy")
        assert result.converged or result.n_iter == 10
        assert len(result.log_likelihood_history) >= 1

    def test_em_unsupervised_does_not_crash(
        self, tiny_taxonomy, noisy_kernel, perfect_kernel,
    ):
        prior_table = {"noisy": noisy_kernel, "perfect": perfect_kernel}
        regions = [
            [_claim("noisy", "a", 1.0), _claim("perfect", "a", 1.0)],
            [_claim("noisy", "b", 1.0), _claim("perfect", "b", 1.0)],
            [_claim("noisy", "c", 1.0), _claim("perfect", "c", 1.0)],
        ]
        result = fit_kernel_mix_em(
            regions, prior_table,
            taxonomy=tiny_taxonomy,
            sources=["noisy", "perfect"],
            anchors=None,
            fit_q_s=False,
            max_iter=5,
        )
        np.testing.assert_allclose(result.mix.lambdas.sum(), 1.0, atol=1e-6)
        assert result.n_anchored_regions == 0
        assert result.n_unsupervised_regions == 3

    def test_em_bp_e_step_matches_unary_default(
        self, tiny_taxonomy, noisy_kernel, perfect_kernel,
    ):
        """With no spatial factors supplied, the PR-4 BP E-step is a
        parity path for the original row-wise MaxCal posterior."""
        prior_table = {"noisy": noisy_kernel, "perfect": perfect_kernel}
        regions = [
            [_claim("noisy", "a", 0.7), _claim("perfect", "a", 1.0)],
            [_claim("noisy", "b", 0.7), _claim("perfect", "b", 1.0)],
            [_claim("noisy", "c", 0.7), _claim("perfect", "c", 1.0)],
        ]
        base = fit_kernel_mix_em(
            regions, prior_table,
            taxonomy=tiny_taxonomy,
            sources=["noisy", "perfect"],
            anchors=None,
            fit_q_s=False,
            max_iter=2,
        )
        bp = fit_kernel_mix_em(
            regions, prior_table,
            taxonomy=tiny_taxonomy,
            sources=["noisy", "perfect"],
            anchors=None,
            fit_q_s=False,
            max_iter=2,
            e_step="bp",
        )

        np.testing.assert_allclose(bp.mix.lambdas, base.mix.lambdas, atol=1e-6)
        assert bp.mix.metadata["e_step"] == "bp"


# ---------------------------------------------------------------------------
# Spectral smoothing
# ---------------------------------------------------------------------------

def _path_laplacian(n: int) -> np.ndarray:
    """Combinatorial Laplacian of an unweighted path graph."""
    A = np.zeros((n, n))
    for i in range(n - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0
    D = np.diag(A.sum(axis=1))
    return D - A


class TestSpectral:
    def test_constant_signal_is_unchanged(self):
        L = _path_laplacian(6)
        P = np.tile([0.5, 0.3, 0.2], (6, 1))
        out = graph_smooth_posteriors(P, L, tau=10.0)
        np.testing.assert_allclose(out, P, atol=1e-10)

    def test_smoothing_reduces_dirichlet_energy(self):
        """A noisy posterior on a path should have a lower Dirichlet
        energy after smoothing."""
        L = _path_laplacian(8)
        rng = np.random.default_rng(0)
        P = np.zeros((8, 3))
        # Clean signal: first half is mostly category 0, second half category 1.
        for i in range(8):
            P[i, 0 if i < 4 else 1] = 0.9
            P[i, 2] = 0.1
        # Add jagged noise.
        noise = 0.3 * rng.standard_normal(P.shape)
        P_noisy = np.clip(P + noise, 0.05, None)
        P_noisy = P_noisy / P_noisy.sum(axis=1, keepdims=True)

        e_before = spectral_consistency_score(P_noisy, L)
        smoothed = graph_smooth_posteriors(P_noisy, L, tau=2.0, kernel="tikhonov")
        e_after = spectral_consistency_score(smoothed, L)
        assert e_after < e_before
        # Smoothed rows still sum to 1.
        np.testing.assert_allclose(smoothed.sum(axis=1), 1.0, atol=1e-9)

    def test_heat_kernel_with_eigendecomposition(self):
        L = _path_laplacian(10)
        eigvals, eigvecs = np.linalg.eigh(L)
        rng = np.random.default_rng(1)
        P = rng.random((10, 4))
        P = P / P.sum(axis=1, keepdims=True)
        smoothed = graph_smooth_posteriors(
            P, laplacian=None,
            tau=0.5, kernel="heat",
            eigvals=eigvals, eigvecs=eigvecs,
        )
        np.testing.assert_allclose(smoothed.sum(axis=1), 1.0, atol=1e-9)
        assert (smoothed >= 0).all()

    def test_tau_zero_is_identity(self):
        L = _path_laplacian(4)
        rng = np.random.default_rng(2)
        P = rng.random((4, 3))
        P = P / P.sum(axis=1, keepdims=True)
        out = graph_smooth_posteriors(P, L, tau=0.0)
        np.testing.assert_allclose(out, P, atol=1e-12)

    def test_consistency_score_zero_for_constant(self):
        L = _path_laplacian(5)
        P = np.tile([0.4, 0.4, 0.2], (5, 1))
        e = spectral_consistency_score(P, L)
        assert abs(e) < 1e-10

    def test_consistency_score_high_for_alternating(self):
        L = _path_laplacian(6)
        # Alternating one-hot pattern => maximum Dirichlet energy.
        P = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [0, 1, 0],
        ], dtype=float)
        e = spectral_consistency_score(P, L)
        assert e > 0.0


# ---------------------------------------------------------------------------
# fit_distinction_game (end-to-end on a SceneGraph dict)
# ---------------------------------------------------------------------------

class TestFitDistinctionGame:
    def test_round_trip_through_scene_graph(self):
        """Build a small SceneGraph, serialise it, feed it back into
        fit_distinction_game; check we get a usable result."""
        poly = [[0.0, 0.0], [0.1, 0.0], [0.1, 0.1], [0.0, 0.1], [0.0, 0.0]]
        claims_a = [
            KernelClaim.from_polygon("osm",            "building", 0.95, poly,
                                     attributes={"tag": "building=yes"}),
            KernelClaim.from_polygon("grounding_dino", "building", 0.80, poly),
        ]
        # Shift to a separate region.
        poly_b = [[0.5, 0.5], [0.6, 0.5], [0.6, 0.6], [0.5, 0.6], [0.5, 0.5]]
        claims_b = [
            KernelClaim.from_polygon("osm",            "highway", 0.95, poly_b,
                                     attributes={"tag": "highway=residential"}),
            KernelClaim.from_polygon("grounding_dino", "road",    0.80, poly_b),
        ]
        sg = build_scene_graph(
            claims_a + claims_b,
            taxonomy=PHX_URBAN_V0,
            q_s_table=default_q_s_table(["osm", "grounding_dino"]),
        )
        sg_dict = sg.to_dict()
        # OSM source id in q_s_table is "osm"; we need to alias it as
        # an OSM anchor source for the pipeline to recognise it.
        result = fit_distinction_game(
            [sg_dict],
            prior_q_s_table=default_q_s_table(["osm", "grounding_dino"]),
            osm_anchor_sources=("osm",),
            fit_q_s=True,
            alpha_q_s=10.0,
            max_iter=5,
        )
        assert isinstance(result, DistinctionGameFit)
        assert result.n_regions == sg.n_nodes
        # Both nodes were OSM-anchored, so both contribute.
        assert result.n_anchored_regions == sg.n_nodes
        np.testing.assert_allclose(result.mix.lambdas.sum(), 1.0, atol=1e-6)
        # Posteriors export round-trips.
        P = posteriors_array_from_scene_graph(sg_dict)
        assert P.shape == (sg.n_nodes, PHX_URBAN_V0.n)
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-6)

    def test_no_anchors_raises(self):
        """A SceneGraph with no OSM claims and consensus_fallback=False
        leaves nothing to fit."""
        poly = [[0.0, 0.0], [0.1, 0.0], [0.1, 0.1], [0.0, 0.1], [0.0, 0.0]]
        claims = [KernelClaim.from_polygon("grounding_dino", "rock", 0.7, poly)]
        sg = build_scene_graph(
            claims,
            taxonomy=PHX_URBAN_V0,
            q_s_table=default_q_s_table(["grounding_dino"]),
        )
        # No OSM source -> no anchors -> all regions unsupervised. EM still
        # runs (just unsupervised). Make sure that path doesn't crash.
        result = fit_distinction_game(
            [sg.to_dict()],
            prior_q_s_table=default_q_s_table(["grounding_dino"]),
            osm_anchor_sources=("osm",),
            fit_q_s=False,
            max_iter=3,
        )
        assert result.n_anchored_regions == 0
        assert result.n_unsupervised_regions == sg.n_nodes
