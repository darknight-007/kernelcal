"""Tests for PR-A.1 of CR-2026-04-26: CityGraph -> FluidGraph adapter.

Covers
------

* :func:`kernelcal.urban.adapter.to_fluid_graph` -- construction,
  edge-length convention, error handling on malformed ``W``.

* :func:`kernelcal.urban.adapter.fluid_graph_connected_components` --
  agreement with the source CityGraph's *topological* β₀ (count of
  connected components of the ``W > 0`` adjacency, acceptance
  criterion **A1**).  We use topological β₀ rather than spectral
  ``betti_zero(eigvals)`` because in ``road_knn`` mode two clusters
  may be linked by a numerically tiny weight that is below the
  spectral tolerance yet still represents a valid graph edge -- both
  counts are "right" by their definitions, and the adapter's job is
  to preserve connectivity (every ``W_ij > 0`` ⇒ one edge), so the
  topological measure is the operationally meaningful invariant.

* End-to-end smoke that a synthetic ``CityGraph`` (grid layout from
  PR-C's :mod:`kernelcal.urban.synthetic`) round-trips to a
  ``FluidGraph`` that runs ``simulate_kernel_fluid_sparse`` for 50
  steps without crashing and conserves mass through the renorm
  ledger.
"""

from __future__ import annotations

import numpy as np
import pytest

from kernelcal.fluid import (
    FluidGraph,
    FluidSimulationConfig,
    PotentialLandscape,
    SparseFluidGraph,
    simulate_kernel_fluid_sparse,
)
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as scipy_cc

from kernelcal.urban import (
    CityGraph,
    betti_zero,
    fluid_graph_connected_components,
    make_fringe_layout,
    make_fringe_road_segments,
    make_grid_layout,
    make_grid_road_segments,
    synthetic_city_graph,
    to_fluid_graph,
)


# ---------------------------------------------------------------------------\
# Helpers
# ---------------------------------------------------------------------------\


def _grid_city(n_x: int = 5, n_y: int = 5, k: int = 4) -> CityGraph:
    """Convenience: small synthetic grid CityGraph."""
    positions = make_grid_layout(
        n_blocks_x=n_x, n_blocks_y=n_y, jitter_m=2.0, seed=0
    )
    return synthetic_city_graph(
        name="test_grid",
        place="synthetic://grid",
        positions=positions,
        k=k,
    )


def _fringe_city(n_buildings: int = 25, n_seeds: int = 4, k: int = 3) -> CityGraph:
    """Convenience: small synthetic fringe CityGraph."""
    positions = make_fringe_layout(
        n_buildings=n_buildings, n_seeds=n_seeds, seed=0
    )
    return synthetic_city_graph(
        name="test_fringe",
        place="synthetic://fringe",
        positions=positions,
        k=k,
    )


def _fringe_city_road_knn(
    n_buildings: int = 25, n_seeds: int = 4, k: int = 3
) -> CityGraph:
    """Synthetic CityGraph in road_knn mode -- exercises the
    second adjacency-construction path the adapter must handle."""
    positions = make_fringe_layout(
        n_buildings=n_buildings, n_seeds=n_seeds, seed=1
    )
    road_nodes, road_edges = make_fringe_road_segments(
        n_seeds=n_seeds, seed=1
    )
    return synthetic_city_graph(
        name="test_fringe_road",
        place="synthetic://fringe_road",
        positions=positions,
        road_nodes=road_nodes,
        road_edges=road_edges,
        k=k,
    )


# ---------------------------------------------------------------------------\
# Construction
# ---------------------------------------------------------------------------\


class TestToFluidGraphConstruction:
    def test_returns_fluid_graph_for_grid(self):
        city = _grid_city()
        fg = to_fluid_graph(city)
        assert isinstance(fg, FluidGraph)
        assert fg.num_nodes == city.W.shape[0]
        assert len(fg.edges) > 0

    def test_returns_fluid_graph_for_fringe(self):
        city = _fringe_city()
        fg = to_fluid_graph(city)
        assert isinstance(fg, FluidGraph)
        assert fg.num_nodes == city.W.shape[0]
        assert len(fg.edges) > 0

    def test_edge_count_matches_strict_upper_triangle_of_W(self):
        city = _grid_city()
        n = city.W.shape[0]
        triu_i, triu_j = np.triu_indices(n, k=1)
        expected_edges = int(np.sum(city.W[triu_i, triu_j] > 0))
        fg = to_fluid_graph(city)
        assert len(fg.edges) == expected_edges

    def test_edges_are_canonical(self):
        city = _grid_city()
        fg = to_fluid_graph(city)
        for i, j in fg.edges:
            assert i < j

    def test_edge_lengths_are_inverse_weights(self):
        city = _grid_city()
        fg = to_fluid_graph(city, weight_floor=1e-6)
        # Every edge length must equal 1/max(W[i,j], floor) within
        # round-trip noise.
        for e, (i, j) in enumerate(fg.edges):
            expected = 1.0 / max(float(city.W[i, j]), 1e-6)
            assert abs(float(fg.edge_lengths[e]) - expected) < 1e-12, (
                f"edge ({i},{j}): expected ell={expected:.6e}, "
                f"got {fg.edge_lengths[e]:.6e}"
            )

    def test_weight_floor_caps_lengths_at_high_weights(self):
        city = _grid_city()
        # A larger floor caps edge lengths from above (since
        # ell = 1 / max(W, floor)).  When floor > all W values, all
        # edge lengths equal 1/floor.
        max_w = float(np.max(city.W[city.W > 0]))
        fg = to_fluid_graph(city, weight_floor=max_w * 10.0)
        expected = 1.0 / (max_w * 10.0)
        np.testing.assert_allclose(
            np.asarray(fg.edge_lengths), expected, rtol=1e-12
        )

    def test_return_sparse_yields_paired_sparse_graph(self):
        city = _grid_city()
        fg, sg = to_fluid_graph(city, return_sparse=True)
        assert isinstance(fg, FluidGraph)
        assert isinstance(sg, SparseFluidGraph)
        assert sg.num_nodes == fg.num_nodes
        assert sg.num_edges == len(fg.edges)
        np.testing.assert_array_equal(sg.edge_lengths, fg.edge_lengths)


# ---------------------------------------------------------------------------\
# Acceptance criterion A1: connected-component count round-trip
# ---------------------------------------------------------------------------\


def _city_topological_components(city: CityGraph) -> int:
    """β₀ of a CityGraph counted *topologically* on its ``W > 0``
    adjacency.

    This is the operational invariant the adapter must preserve --
    in contrast to the spectral :func:`betti_zero` count which reads
    near-zero eigenvalues of ``L`` and can register weakly-connected
    clusters as separate components when their bridging weight falls
    below the spectral tolerance.  The adapter, by construction,
    creates one edge per positive ``W_ij``, so topological β₀ is
    preserved exactly.
    """
    adj = (np.asarray(city.W) > 0).astype(np.int8)
    n_components, _ = scipy_cc(csr_matrix(adj), directed=False)
    return int(n_components)


def _city_spectral_components(city: CityGraph, tol: float = 1e-6) -> int:
    """Spectral β₀ via :func:`betti_zero`.  Used in one test to
    show *how* the topological and spectral measures can diverge in
    ``road_knn`` mode -- this is informational, not an invariant the
    adapter is required to satisfy."""
    return betti_zero(city.eigvals, tol=tol)


class TestConnectedComponentsRoundTrip:
    def test_grid_city_components_match(self):
        """Acceptance criterion **A1**: a grid CityGraph round-trips
        through the adapter with the same connected-component count."""
        city = _grid_city()
        fg = to_fluid_graph(city)
        assert fluid_graph_connected_components(fg) == _city_topological_components(city)

    def test_fringe_city_components_match(self):
        city = _fringe_city()
        fg = to_fluid_graph(city)
        assert fluid_graph_connected_components(fg) == _city_topological_components(city)

    def test_road_knn_components_match(self):
        """Road-aware k-NN may produce more disconnected components than
        Euclidean k-NN; the adapter must preserve that signal."""
        city = _fringe_city_road_knn()
        fg = to_fluid_graph(city)
        assert fluid_graph_connected_components(fg) == _city_topological_components(city)

    def test_topological_and_spectral_can_diverge_in_road_knn(self):
        """Documentation test: in ``road_knn`` mode, the spectral β₀
        from ``betti_zero(eigvals)`` and the topological β₀ from
        ``W > 0`` adjacency may legitimately disagree (when two
        clusters are linked by a numerically tiny weight that's below
        the spectral tolerance).  This test pins that observed
        behaviour so the choice of "topological" in A1 is auditable."""
        city = _fringe_city_road_knn()
        topo = _city_topological_components(city)
        spec = _city_spectral_components(city, tol=1e-6)
        # Either they agree (clean case), or spectral counts more
        # components than topological (weak-bridge case).  Spectral
        # never undercounts topological β₀ for a non-negative graph
        # Laplacian -- that's the inequality used in PR-C's spectrum
        # helpers as well.
        assert spec >= topo

    def test_explicitly_disconnected_W_round_trips(self):
        """Build a CityGraph by hand whose ``W`` has two disjoint
        components (8x8 with no cross-component edges) and verify the
        adapter preserves the partition."""
        n = 8
        W = np.zeros((n, n), dtype=float)
        # Component A: nodes 0-3 in a chain
        for k in range(3):
            W[k, k + 1] = 1.0
            W[k + 1, k] = 1.0
        # Component B: nodes 4-7 in a chain
        for k in range(4, 7):
            W[k, k + 1] = 1.0
            W[k + 1, k] = 1.0
        D = np.diag(W.sum(axis=1))
        L = D - W
        eigvals, eigvecs = np.linalg.eigh(L)
        eigvals = np.maximum(eigvals, 0.0)
        city = CityGraph(
            name="manual_split",
            place="synthetic://manual",
            positions=np.zeros((n, 2)),
            traits=np.zeros((n, 4)),
            L=L,
            W=W,
            eigvals=eigvals,
            eigvecs=eigvecs,
            n_buildings=n,
            bounds_m=(0.0, 0.0, 1.0, 1.0),
        )
        # Source city has topological β₀ = 2.
        assert _city_topological_components(city) == 2
        fg = to_fluid_graph(city)
        assert fluid_graph_connected_components(fg) == 2


# ---------------------------------------------------------------------------\
# Error handling
# ---------------------------------------------------------------------------\


class TestErrorHandling:
    def test_rejects_non_square_W(self):
        n = 4
        W = np.zeros((n, n + 1), dtype=float)  # not square
        city = CityGraph(
            name="bad", place="synthetic://bad",
            positions=np.zeros((n, 2)), traits=np.zeros((n, 4)),
            L=np.eye(n), W=W,
            eigvals=np.zeros(n), eigvecs=np.eye(n),
            n_buildings=n, bounds_m=(0, 0, 1, 1),
        )
        with pytest.raises(ValueError, match="square"):
            to_fluid_graph(city)

    def test_rejects_asymmetric_W(self):
        n = 4
        W = np.zeros((n, n), dtype=float)
        W[0, 1] = 1.0
        W[1, 0] = 0.5  # asymmetric
        D = np.diag(W.sum(axis=1))
        L = D - W
        city = CityGraph(
            name="bad", place="synthetic://bad",
            positions=np.zeros((n, 2)), traits=np.zeros((n, 4)),
            L=L, W=W,
            eigvals=np.zeros(n), eigvecs=np.eye(n),
            n_buildings=n, bounds_m=(0, 0, 1, 1),
        )
        with pytest.raises(ValueError, match="symmetric"):
            to_fluid_graph(city)

    def test_rejects_negative_W(self):
        n = 4
        W = np.zeros((n, n), dtype=float)
        W[0, 1] = -1.0
        W[1, 0] = -1.0
        D = np.diag(W.sum(axis=1))
        L = D - W
        city = CityGraph(
            name="bad", place="synthetic://bad",
            positions=np.zeros((n, 2)), traits=np.zeros((n, 4)),
            L=L, W=W,
            eigvals=np.zeros(n), eigvecs=np.eye(n),
            n_buildings=n, bounds_m=(0, 0, 1, 1),
        )
        with pytest.raises(ValueError, match="non-negative"):
            to_fluid_graph(city)

    def test_rejects_edgeless_W(self):
        n = 4
        W = np.zeros((n, n), dtype=float)
        L = np.zeros((n, n), dtype=float)
        city = CityGraph(
            name="bad", place="synthetic://bad",
            positions=np.zeros((n, 2)), traits=np.zeros((n, 4)),
            L=L, W=W,
            eigvals=np.zeros(n), eigvecs=np.eye(n),
            n_buildings=n, bounds_m=(0, 0, 1, 1),
        )
        with pytest.raises(ValueError, match="nonzero"):
            to_fluid_graph(city)


# ---------------------------------------------------------------------------\
# End-to-end smoke: adapter + sparse solver
# ---------------------------------------------------------------------------\


class TestEndToEndWithSparseSolver:
    def test_grid_city_runs_through_sparse_solver(self):
        """The whole point of A.1: a CityGraph plugs straight into the
        PR-A.0 sparse solver and conserves mass via the logged renorm
        signal.  This is the smoke that A.2's multi-component lift
        will inherit."""
        city = _grid_city()
        fg, sg = to_fluid_graph(city, return_sparse=True)
        n = fg.num_nodes
        rng = np.random.default_rng(0)
        landscape = PotentialLandscape(
            loss=rng.uniform(0.0, 1.0, size=n),
            cost=rng.uniform(0.0, 1.0, size=n),
            info=rng.uniform(0.0, 1.0, size=n),
        )
        cfg = FluidSimulationConfig(
            steps=50, dt=0.005, phase_switch_step=10**9
        )
        result = simulate_kernel_fluid_sparse(
            fg, landscape, cfg,
            track_rho_history=False,
            sparse_graph=sg,
        )
        # Mass conservation post-renorm at machine eps.
        assert float(np.max(result.mass_error)) < 1e-12
        # Ledger fields populated and finite.
        assert np.all(np.isfinite(result.floor_mass_inserted))
        assert np.all(np.isfinite(result.renormalize_correction))

    def test_fringe_city_runs_through_sparse_solver(self):
        city = _fringe_city()
        fg, sg = to_fluid_graph(city, return_sparse=True)
        n = fg.num_nodes
        landscape = PotentialLandscape(
            loss=np.zeros(n), cost=np.zeros(n), info=np.zeros(n)
        )
        cfg = FluidSimulationConfig(
            steps=20, dt=0.01, phase_switch_step=10**9
        )
        result = simulate_kernel_fluid_sparse(
            fg, landscape, cfg,
            track_rho_history=False,
            sparse_graph=sg,
        )
        assert float(np.max(result.mass_error)) < 1e-12
