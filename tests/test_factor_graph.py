from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from kernelcal.distinction_game import (
    KernelClaim,
    Taxonomy,
    graph_smooth_posteriors,
    spectral_consistency_score,
)
from kernelcal.distinction_game.factor_graph import (
    Factor,
    FactorGraph,
    PairwiseAssociationFactor,
    PairwiseSpatialFactor,
    UnaryPerceptualFactor,
    loopy_bp,
)
from kernelcal.distinction_game.q_s import ConfusionMatrix


@pytest.fixture
def tiny_taxonomy() -> Taxonomy:
    return Taxonomy(
        name="tiny",
        categories=("a", "b", "c"),
        super_classes={"a": "x", "b": "x"},
    )


@pytest.fixture
def tiny_q(tiny_taxonomy) -> ConfusionMatrix:
    return ConfusionMatrix(
        source_id="src",
        taxonomy=tiny_taxonomy,
        native_labels=("a", "b", "c"),
        matrix=np.array([
            [0.80, 0.10, 0.10],
            [0.10, 0.80, 0.10],
            [0.10, 0.10, 0.80],
        ]),
    )


def _claim(label: str, source: str = "src", score: float = 1.0) -> KernelClaim:
    return KernelClaim.from_polygon(
        source,
        label,
        score,
        [(0, 0), (1, 0), (1, 1), (0, 0)],
    )


def _exact_marginals(graph: FactorGraph):
    vids = list(graph.variables)
    ks = [graph.variables[v].n_states for v in vids]
    log_w = []
    states = []
    for assignment in product(*[range(k) for k in ks]):
        by_var = dict(zip(vids, assignment))
        lp = 0.0
        for v, state in by_var.items():
            lp += graph.variables[v].log_prior[state]
        for factor in graph.factors:
            idx = tuple(by_var[v] for v in factor.variables)
            lp += factor.log_table[idx]
        states.append(by_var)
        log_w.append(lp)
    log_w = np.asarray(log_w)
    log_w = log_w - log_w.max()
    w = np.exp(log_w)
    w = w / w.sum()
    out = {v: np.zeros(graph.variables[v].n_states) for v in vids}
    for weight, by_var in zip(w, states):
        for v, state in by_var.items():
            out[v][state] += weight
    return out


def test_two_variable_chain_matches_exact_marginals(tiny_taxonomy):
    graph = FactorGraph()
    graph.add_variable("x", 3, prior=[0.8, 0.1, 0.1])
    graph.add_variable("y", 3, prior=[0.1, 0.8, 0.1])
    graph.add_factor(PairwiseSpatialFactor("x", "y", taxonomy=tiny_taxonomy, beta=0.7))

    bp = loopy_bp(graph, max_iter=20, damping=0.0, tol=1e-10)
    exact = _exact_marginals(graph)

    assert bp.converged
    np.testing.assert_allclose(bp.posteriors["x"], exact["x"], atol=1e-9)
    np.testing.assert_allclose(bp.posteriors["y"], exact["y"], atol=1e-9)


def test_three_variable_loop_converges_near_exact(tiny_taxonomy):
    graph = FactorGraph()
    graph.add_variable("x", 3, prior=[0.8, 0.1, 0.1])
    graph.add_variable("y", 3, prior=[0.1, 0.8, 0.1])
    graph.add_variable("z", 3, prior=[0.1, 0.2, 0.7])
    graph.add_factor(PairwiseSpatialFactor("x", "y", taxonomy=tiny_taxonomy, beta=0.4))
    graph.add_factor(PairwiseSpatialFactor("y", "z", taxonomy=tiny_taxonomy, beta=0.4))
    graph.add_factor(PairwiseSpatialFactor("z", "x", taxonomy=tiny_taxonomy, beta=0.4))

    bp = loopy_bp(graph, max_iter=100, damping=0.5, tol=1e-7)
    exact = _exact_marginals(graph)

    assert bp.converged
    for vid in ("x", "y", "z"):
        np.testing.assert_allclose(bp.posteriors[vid], exact[vid], atol=3e-2)


def test_unary_perceptual_reproduces_maxcal_argmax(tiny_taxonomy, tiny_q):
    graph = FactorGraph()
    graph.add_variable("r0", 3)
    graph.add_factor(UnaryPerceptualFactor(
        "r0",
        [_claim("b")],
        q_s_table={"src": tiny_q},
        lambdas={"src": 1.0},
        taxonomy=tiny_taxonomy,
    ))

    bp = loopy_bp(graph, max_iter=5, damping=0.0)

    assert int(np.argmax(bp.posteriors["r0"])) == 1
    np.testing.assert_allclose(bp.posteriors["r0"], [0.1, 0.8, 0.1], atol=1e-9)


def test_spatial_beta_zero_is_unary_only(tiny_taxonomy):
    graph = FactorGraph()
    graph.add_variable("x", 3, prior=[0.8, 0.1, 0.1])
    graph.add_variable("y", 3, prior=[0.1, 0.8, 0.1])
    graph.add_factor(PairwiseSpatialFactor("x", "y", taxonomy=tiny_taxonomy, beta=0.0))

    bp = loopy_bp(graph, max_iter=20, damping=0.0)

    np.testing.assert_allclose(bp.posteriors["x"], [0.8, 0.1, 0.1], atol=1e-9)
    np.testing.assert_allclose(bp.posteriors["y"], [0.1, 0.8, 0.1], atol=1e-9)


def test_strong_spatial_factor_pulls_neighbors_together(tiny_taxonomy):
    graph = FactorGraph()
    graph.add_variable("x", 3, prior=[0.55, 0.35, 0.10])
    graph.add_variable("y", 3, prior=[0.35, 0.55, 0.10])
    graph.add_factor(PairwiseSpatialFactor("x", "y", taxonomy=tiny_taxonomy, beta=4.0))

    bp = loopy_bp(graph, max_iter=50, damping=0.0)

    assert bp.posteriors["x"][2] < 0.03
    assert bp.posteriors["y"][2] < 0.03
    assert bp.posteriors["x"][:2].sum() > 0.97
    assert bp.posteriors["y"][:2].sum() > 0.97


def test_spatial_bp_agrees_with_spectral_smoother_direction(tiny_taxonomy):
    """PR-4 spatial factors should have the same qualitative contract
    as PR-3 spectral smoothing: reduce Dirichlet energy on a known path."""
    priors = {
        "x": np.array([0.92, 0.04, 0.04]),
        "y": np.array([0.04, 0.92, 0.04]),
        "z": np.array([0.92, 0.04, 0.04]),
    }
    graph = FactorGraph()
    for vid, prior in priors.items():
        graph.add_variable(vid, 3, prior=prior)
    graph.add_factor(PairwiseSpatialFactor("x", "y", taxonomy=tiny_taxonomy, beta=1.2))
    graph.add_factor(PairwiseSpatialFactor("y", "z", taxonomy=tiny_taxonomy, beta=1.2))

    bp = loopy_bp(graph, max_iter=50, damping=0.2)
    unary = np.vstack([priors[v] for v in ("x", "y", "z")])
    bp_arr = np.vstack([bp.posteriors[v] for v in ("x", "y", "z")])
    lap = np.array([
        [1.0, -1.0, 0.0],
        [-1.0, 2.0, -1.0],
        [0.0, -1.0, 1.0],
    ])
    spectral = graph_smooth_posteriors(unary, laplacian=lap, tau=1.0)

    assert spectral_consistency_score(bp_arr, lap) < spectral_consistency_score(unary, lap)
    assert spectral_consistency_score(spectral, lap) < spectral_consistency_score(unary, lap)


def test_hard_association_forces_equal_map_state():
    graph = FactorGraph()
    graph.add_variable("a", 2, prior=[0.9, 0.1])
    graph.add_variable("b", 2, prior=[0.1, 0.9])
    graph.add_factor(PairwiseAssociationFactor("a", "b", n_states=2, eps=0.0))

    bp = loopy_bp(graph, max_iter=20, damping=0.0)

    assert int(np.argmax(bp.posteriors["a"])) == int(np.argmax(bp.posteriors["b"]))
    np.testing.assert_allclose(bp.posteriors["a"], [0.5, 0.5], atol=1e-9)
    np.testing.assert_allclose(bp.posteriors["b"], [0.5, 0.5], atol=1e-9)


def test_generic_factor_validates_shape():
    graph = FactorGraph()
    graph.add_variable("x", 2)
    with pytest.raises(ValueError, match="dimensions"):
        Factor(variables=("x",), log_table=np.zeros((2, 2)))
