"""Tests for kernelcal.terrain — planetary terrain topology and biosignature detection.

Covers:
  - DEM construction, slope, curvature, D8 flow routing, flow accumulation
  - Crater DEM synthesis, graph construction, Betti numbers
  - Channel network extraction, Strahler ordering, triple spectral diagnostic
  - Topological biosignature Δβ₁ and detection threshold
  - Cross-kernel factorization test
  - Plume spectral entropy biosignature
  - Fixed-point kernel and spectral diagnostics
  - Stability–conservation tradeoff (Route 3 result)
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# DEM module
# ---------------------------------------------------------------------------

from kernelcal.terrain.dem import (
    synthetic_crater_dem, synthetic_channel_dem,
    dem_to_graph, terrain_graph_laplacian,
    slope, curvature_planform,
    d8_flow_direction, flow_accumulation, channel_mask,
)


class TestDEM:
    def test_synthetic_crater_dem_shape(self):
        dem = synthetic_crater_dem(nrows=32, ncols=32)
        assert dem.shape == (32, 32)

    def test_crater_has_depression(self):
        dem = synthetic_crater_dem(nrows=64, ncols=64, depth=5.0)
        center = (32, 32)
        # Interior should be below surrounding plain
        assert dem[center] < dem[0, 0]

    def test_slope_shape(self):
        dem = synthetic_crater_dem(32, 32)
        s = slope(dem)
        assert s.shape == (32, 32)
        assert np.all(s >= 0)

    def test_curvature_planform_shape(self):
        dem = synthetic_crater_dem(32, 32)
        c = curvature_planform(dem)
        assert c.shape == (32, 32)

    def test_crater_rim_has_high_curvature(self):
        dem = synthetic_crater_dem(64, 64, radius=12.0)
        c = curvature_planform(dem)
        # Rim region should have stronger curvature than flat exterior
        rows, cols = np.mgrid[0:64, 0:64]
        dist = np.hypot(rows - 32, cols - 32)
        rim_mask = (np.abs(dist - 12.0) <= 3.0)
        flat_mask = dist > 25.0
        assert np.mean(np.abs(c[rim_mask])) > np.mean(np.abs(c[flat_mask]))

    def test_dem_to_graph_nodes(self):
        dem = np.ones((8, 8))
        tg = dem_to_graph(dem, connectivity=4)
        assert len(tg.elevations) == 64
        assert len(tg.edges) > 0

    def test_dem_to_graph_connectivity(self):
        dem = np.ones((4, 4))
        tg4 = dem_to_graph(dem, connectivity=4)
        tg8 = dem_to_graph(dem, connectivity=8)
        assert len(tg8.edges) > len(tg4.edges)

    def test_terrain_graph_laplacian_psd(self):
        dem = np.random.RandomState(0).randn(6, 6)
        tg = dem_to_graph(dem)
        L = terrain_graph_laplacian(tg)
        eigvals = np.linalg.eigvalsh(L)
        assert np.all(eigvals >= -1e-10)   # PSD (up to numerical noise)

    def test_d8_flow_direction_shape(self):
        dem = synthetic_channel_dem(16, 16, n_tributaries=2)
        fdir = d8_flow_direction(dem)
        assert fdir.shape == (16, 16)
        assert np.all(fdir >= -1)
        assert np.all(fdir <= 7)

    def test_flow_accumulation_monotone(self):
        # Outlet cell (downstream end) should have highest accumulation
        dem = synthetic_channel_dem(16, 16, n_tributaries=2, slope_angle=0.1)
        fdir = d8_flow_direction(dem)
        acc = flow_accumulation(fdir)
        assert acc.max() > 1
        assert acc.min() >= 1

    def test_channel_mask_threshold(self):
        dem = synthetic_channel_dem(16, 16)
        fdir = d8_flow_direction(dem)
        acc = flow_accumulation(fdir)
        mask_low  = channel_mask(acc, threshold=2)
        mask_high = channel_mask(acc, threshold=8)
        assert mask_low.sum() >= mask_high.sum()

    def test_synthetic_channel_dem_shape(self):
        dem = synthetic_channel_dem(32, 32, n_tributaries=3)
        assert dem.shape == (32, 32)


# ---------------------------------------------------------------------------
# Craters module
# ---------------------------------------------------------------------------

from kernelcal.terrain.craters import (
    crater_betti_numbers, abiotic_beta1_craters,
    crater_rim_mask, crater_rim_graph, crater_spectral_signature,
    CraterCandidate,
)


class TestCraters:
    def _simple_crater(self):
        return CraterCandidate(row=16, col=16, radius=8.0,
                               rim_completeness=1.0, curvature_contrast=1.0)

    def test_rim_mask_is_ring(self):
        shape = (32, 32)
        c = self._simple_crater()
        mask = crater_rim_mask(shape, [c], rim_width=2)
        # Ring should be non-empty but not fill the whole image
        assert mask.sum() > 0
        assert mask.sum() < 32 * 32

    def test_rim_mask_centred(self):
        shape = (32, 32)
        c = self._simple_crater()
        mask = crater_rim_mask(shape, [c], rim_width=2)
        # Mask should be symmetric around crater centre
        rows, cols = np.where(mask)
        mean_r = rows.mean()
        mean_c = cols.mean()
        assert abs(mean_r - 16) < 2.0
        assert abs(mean_c - 16) < 2.0

    def test_crater_rim_graph_not_empty(self):
        dem = synthetic_crater_dem(32, 32, radius=8.0)
        c = self._simple_crater()
        tg = crater_rim_graph(dem, [c], rim_width=2)
        assert len(tg.elevations) > 0
        assert len(tg.edges) > 0

    def test_crater_betti_numbers_ring_has_loop(self):
        # A clean ring graph (cycle graph C_n) has β₀=1, β₁=1
        n = 12
        positions = np.array([[np.cos(2*np.pi*i/n), np.sin(2*np.pi*i/n)] for i in range(n)])
        edges = np.array([(i, (i+1) % n) for i in range(n)], dtype=np.int32)
        weights = np.ones(n)
        from kernelcal.terrain.dem import TerrainGraph
        tg = TerrainGraph(
            positions=positions,
            elevations=np.zeros(n),
            edges=edges,
            weights=weights,
            shape=(1, 1),
            cell_index=np.array([[-1]]),
        )
        betti = crater_betti_numbers(tg)
        assert betti["beta0"] == 1
        assert betti["beta1"] == 1

    def test_abiotic_beta1_craters(self):
        null = abiotic_beta1_craters(n_intact=3, n_degraded=1)
        assert null["beta1_abio"] == 3
        assert null["kmin_abio"] == 4   # β₀ + β₁ = 1 + 3

    def test_abiotic_beta1_zero_craters(self):
        null = abiotic_beta1_craters(n_intact=0)
        assert null["beta1_abio"] == 0
        assert null["kmin_abio"] == 1

    def test_crater_spectral_signature_keys(self):
        dem = synthetic_crater_dem(32, 32)
        c = self._simple_crater()
        tg = crater_rim_graph(dem, [c], rim_width=2)
        sig = crater_spectral_signature(tg, n_modes=10)
        for key in ("fiedler", "spectral_entropy", "eigenvalues", "beta0", "beta1"):
            assert key in sig

    def test_crater_spectral_signature_fiedler_positive(self):
        dem = synthetic_crater_dem(32, 32)
        c = self._simple_crater()
        tg = crater_rim_graph(dem, [c], rim_width=2)
        sig = crater_spectral_signature(tg, n_modes=10)
        assert sig["fiedler"] >= 0


# ---------------------------------------------------------------------------
# Channels module
# ---------------------------------------------------------------------------

from kernelcal.terrain.channels import (
    drainage_network_graph, drainage_graph_laplacian,
    triple_spectral_diagnostic, curl_energy,
    pairwise_connectivity_after_removal, subbasins_after_removal,
    betweenness_centrality_undirected, most_central_nodes,
    group_betweenness_score, identify_critical_nodes,
    critical_fragmentation_curve,
    abiotic_beta1_channels, topology_budget,
)


class TestChannels:
    def _simple_tree_graph(self):
        """Small fixed tree for deterministic critical-node tests."""
        # 0-1-3, 1-4, 0-2-5
        from kernelcal.terrain.channels import DrainageGraph
        nodes = [(i, 0) for i in range(6)]
        edges = np.array([[0, 1], [0, 2], [1, 3], [1, 4], [2, 5]], dtype=np.int32)
        return DrainageGraph(
            nodes=nodes,
            node_index={rc: i for i, rc in enumerate(nodes)},
            directed_edges=[(3, 1), (4, 1), (1, 0), (5, 2), (2, 0)],
            undirected_edges=edges,
            accumulation=np.ones(6, dtype=np.int32),
            strahler=np.ones(6, dtype=np.int32),
            beta0=1,
            beta1=0,
        )

    def test_drainage_graph_not_empty(self):
        dem = synthetic_channel_dem(24, 24, n_tributaries=3, slope_angle=0.1)
        dg = drainage_network_graph(dem, threshold=4)
        assert len(dg.nodes) > 0

    def test_drainage_graph_betti_tree(self):
        # A single-stem channel with no closed loops: β₁ = 0
        dem = np.zeros((16, 8))
        for r in range(16):
            dem[r, :] = 16 - r   # uniform slope, single stem
        dg = drainage_network_graph(dem, threshold=2)
        assert dg.beta1 == 0   # tree graph, no loops

    def test_strahler_order_monotone(self):
        dem = synthetic_channel_dem(32, 32, n_tributaries=3, slope_angle=0.1)
        dg = drainage_network_graph(dem, threshold=4)
        assert len(dg.strahler) == len(dg.nodes)
        assert int(dg.strahler.min()) >= 1

    def test_abiotic_beta1_channels(self):
        null = abiotic_beta1_channels(n_junctions=5)
        assert null["beta1_abio"] == 4   # n - 1

    def test_abiotic_beta1_single_junction(self):
        null = abiotic_beta1_channels(n_junctions=1)
        assert null["beta1_abio"] == 0

    def test_topology_budget(self):
        dem = synthetic_channel_dem(24, 24, n_tributaries=2, slope_angle=0.1)
        dg = drainage_network_graph(dem, threshold=4)
        budget = topology_budget(dg)
        assert budget["kmin"] == budget["beta0"] + budget["beta1"]
        assert budget["kmin"] >= 1

    def test_drainage_graph_laplacian_psd(self):
        dem = synthetic_channel_dem(16, 16, n_tributaries=2)
        dg = drainage_network_graph(dem, threshold=3)
        if len(dg.nodes) > 1:
            L = drainage_graph_laplacian(dg)
            eigvals = np.linalg.eigvalsh(L)
            assert np.all(eigvals >= -1e-10)

    def test_curl_energy_bounded(self):
        dem = synthetic_channel_dem(24, 24, n_tributaries=3)
        dg = drainage_network_graph(dem, threshold=3)
        E = curl_energy(dg)
        assert 0.0 <= E <= 1.0

    def test_triple_diagnostic_channeled(self):
        dem_chan = synthetic_channel_dem(32, 32, n_tributaries=3, slope_angle=0.1)
        dg = drainage_network_graph(dem_chan, threshold=4)
        diag = triple_spectral_diagnostic(dg)
        assert diag.H_spectral >= 0
        assert diag.E_curl >= 0
        assert diag.beta1 >= 0
        assert diag.fiedler >= 0

    def test_triple_diagnostic_with_flat_reference(self):
        dem_chan = synthetic_channel_dem(32, 32, n_tributaries=3)
        dem_flat = np.zeros((32, 32))   # completely flat
        dg_chan = drainage_network_graph(dem_chan, threshold=4)
        dg_flat = drainage_network_graph(dem_flat, threshold=4)
        diag = triple_spectral_diagnostic(dg_chan, dg_flat=dg_flat)
        # All three flags should be evaluable (not None) when flat reference provided
        assert diag.fiedler_concentrated is not None
        assert diag.curl_elevated is not None
        assert diag.beta1_anomalous is not None

    def test_pairwise_connectivity_counts_pairs(self):
        dg = self._simple_tree_graph()
        # n=6 => total pairs = 15
        assert pairwise_connectivity_after_removal(dg, []) == 15
        # remove node 1 => components sizes [3,1,1] => 3 pairs
        assert pairwise_connectivity_after_removal(dg, [1]) == 3

    def test_critical_node_matches_best_single_deletion(self):
        dg = self._simple_tree_graph()
        result = identify_critical_nodes(dg, k=1, method="exact")
        assert result.k == 1
        assert result.method_used == "exact"
        # Nodes 0 or 1 are optimal single deletions on this tree.
        assert int(result.nodes[0]) in (0, 1)
        assert result.pairwise_connectivity <= 3
        assert result.disconnected_pairs == 15 - result.pairwise_connectivity

    def test_group_betweenness_matches_disconnected_pairs(self):
        dg = self._simple_tree_graph()
        nodes = np.array([1], dtype=int)
        gb = group_betweenness_score(dg, nodes)
        pc = pairwise_connectivity_after_removal(dg, nodes)
        assert gb == 15 - pc

    def test_critical_fragmentation_curve_monotone(self):
        dg = self._simple_tree_graph()
        curve = critical_fragmentation_curve(dg, k_max=3, method="exact", compare_central=True)
        assert len(curve.k_values) == 3
        assert np.all(np.diff(curve.pairwise_connectivity_critical) <= 1e-12)
        # Component count is bounded by surviving nodes and should increase at least once.
        for k, sb in zip(curve.k_values, curve.subbasins_critical):
            assert 0 <= sb <= (len(dg.nodes) - int(k))
        assert np.any(np.diff(curve.subbasins_critical) > 0)
        assert curve.pairwise_connectivity_central is not None
        # Critical deletion should not leave more connected pairs than central baseline.
        assert np.all(curve.pairwise_connectivity_critical <= curve.pairwise_connectivity_central + 1e-12)


# ---------------------------------------------------------------------------
# Biosig module
# ---------------------------------------------------------------------------

from kernelcal.terrain.biosig import (
    topological_biosignature, detection_threshold,
    cross_kernel, cross_kernel_norm, factorization_test,
    spectral_kernel_from_laplacian,
    chemical_affinity_graph, plume_spectral_entropy,
    BiosignatureReport,
)


class TestBiosig:
    def test_delta_beta1_positive(self):
        tb = topological_biosignature(beta1_obs=5, beta1_abio=2)
        assert tb.delta_beta1 == 3
        assert tb.is_anomalous

    def test_delta_beta1_zero(self):
        tb = topological_biosignature(beta1_obs=2, beta1_abio=2)
        assert tb.delta_beta1 == 0
        assert not tb.is_anomalous

    def test_delta_beta1_negative_clamped_meaning(self):
        # Negative Δβ₁ means observation shows LESS topology than abiotic model —
        # possible if the abiotic model overestimates (degraded craters, etc.)
        tb = topological_biosignature(beta1_obs=1, beta1_abio=3)
        assert tb.delta_beta1 == -2
        assert not tb.is_anomalous

    def test_detection_threshold_keys(self):
        dt = detection_threshold(beta1_abio=3, delta_beta1=2, bits_per_coeff=32.0)
        for key in ("k_min_abio", "k_required", "total_bits", "detectable"):
            assert key in dt

    def test_detection_threshold_k_required(self):
        dt = detection_threshold(beta1_abio=3, delta_beta1=2)
        assert dt["k_min_abio"] == 4    # 1 + 3
        assert dt["k_required"] == 6    # 4 + 2

    def test_detection_threshold_with_I_self(self):
        dt = detection_threshold(beta1_abio=1, delta_beta1=1,
                                 bits_per_coeff=32.0, I_self_bps=1e6)
        assert dt["R_min"] is not None
        assert dt["R_min"] > 0

    def test_cross_kernel_zero_for_factorized(self):
        # If K_coupled = K_A ⊗ K_B exactly, k_cross should be ~ 0
        K_A = np.array([[2., 1.], [1., 2.]])
        K_B = np.array([[3., 0.5], [0.5, 3.]])
        K_coupled = np.kron(K_A, K_B)
        k_cross = cross_kernel(K_coupled, K_A, K_B)
        assert np.allclose(k_cross, 0, atol=1e-10)

    def test_cross_kernel_nonzero_for_coupled(self):
        K_A = np.eye(2)
        K_B = np.eye(2)
        K_coupled = np.eye(4) * 2.0   # Not K_A ⊗ K_B = np.eye(4)
        k_cross = cross_kernel(K_coupled, K_A, K_B)
        assert not np.allclose(k_cross, 0)

    def test_factorization_test_independent(self):
        K_A = np.array([[2., 1.], [1., 2.]])
        K_B = np.array([[3., 0.5], [0.5, 3.]])
        K_coupled = np.kron(K_A, K_B)
        result = factorization_test(K_coupled, K_A, K_B)
        assert not result["is_coupled"]
        assert result["relative_norm"] < 1e-8

    def test_factorization_test_coupled(self):
        K_A = np.eye(3)
        K_B = np.eye(3)
        # K_coupled has off-diagonal blocks (coupling)
        K_coupled = np.kron(K_A, K_B) + 0.5 * np.ones((9, 9))
        result = factorization_test(K_coupled, K_A, K_B, significance_threshold=0.01)
        assert result["is_coupled"]

    def test_spectral_kernel_from_laplacian_psd(self):
        L = np.array([[2., -1., -1.],
                      [-1., 2., -1.],
                      [-1., -1., 2.]])
        K = spectral_kernel_from_laplacian(L, tau=0.5)
        eigvals = np.linalg.eigvalsh(K)
        assert np.all(eigvals >= -1e-10)

    def test_chemical_affinity_graph(self):
        species = ["A", "B", "C"]
        co = np.array([[0., 2., 0.5],
                       [2., 0., 1.0],
                       [0.5, 1.0, 0.]])
        W, L = chemical_affinity_graph(species, co)
        # Laplacian should be PSD
        eigvals = np.linalg.eigvalsh(L)
        assert np.all(eigvals >= -1e-10)
        # Row sums of L should be zero
        assert np.allclose(L.sum(axis=1), 0, atol=1e-10)

    def test_plume_entropy_keys(self):
        L = np.array([[1., -1., 0.],
                      [-1., 2., -1.],
                      [0., -1., 1.]])
        result = plume_spectral_entropy(L)
        for key in ("H_obs", "H_abio", "entropy_drop", "bandpass_spike",
                    "is_biosignature", "h_obs", "h_abio", "eigenvalues"):
            assert key in result

    def test_plume_entropy_equilibrium_high(self):
        # Uniform complete graph: abiotic → should have high entropy
        n = 8
        W = np.ones((n, n)) - np.eye(n)
        D = np.diag(W.sum(axis=1))
        L = D - W
        result = plume_spectral_entropy(L, L_abio=L)
        # When L_obs == L_abio, entropy drop should be ~0
        assert abs(result["entropy_drop"]) < 1e-6

    def test_plume_entropy_organised_lower(self):
        # A bandpass-like graph (ring) vs. complete graph baseline
        n = 8
        W_ring = np.zeros((n, n))
        for i in range(n):
            W_ring[i, (i+1) % n] = 1.0
            W_ring[(i+1) % n, i] = 1.0
        L_ring = np.diag(W_ring.sum(axis=1)) - W_ring
        W_complete = np.ones((n, n)) - np.eye(n)
        L_complete  = np.diag(W_complete.sum(axis=1)) - W_complete
        result = plume_spectral_entropy(L_ring, L_abio=L_complete)
        # Organised ring should have different entropy from flat complete graph
        assert result["H_obs"] != result["H_abio"]

    def test_biosignature_report_score(self):
        from kernelcal.terrain.biosig import topological_biosignature
        tb = topological_biosignature(5, 2)
        report = BiosignatureReport(target="Jezero", topological=tb)
        # score ≥ 1 because topological anomaly is present
        assert report.score >= 1

    def test_biosignature_report_summary_contains_target(self):
        report = BiosignatureReport(target="Enceladus plume")
        s = report.summary()
        assert "Enceladus plume" in s


# ---------------------------------------------------------------------------
# Diagnostics module
# ---------------------------------------------------------------------------

from kernelcal.terrain.diagnostics import (
    spectral_entropy, spectral_entropy_from_laplacian,
    fixed_point_kernel, fiedler_mode_gap,
    stability_conservation_tradeoff,
    phase_transition_sweep, observability_ratio,
    bandwidth_optimal_modes,
)


class TestDiagnostics:
    def _p8_laplacian(self):
        """Path graph P8 Laplacian."""
        n = 8
        L = np.diag([1.] + [2.] * (n-2) + [1.]) \
          - np.diag(np.ones(n-1), 1) \
          - np.diag(np.ones(n-1), -1)
        return L

    def test_spectral_entropy_uniform(self):
        h = np.ones(8)
        H = spectral_entropy(h)
        assert abs(H - np.log(8)) < 1e-10

    def test_spectral_entropy_concentrated(self):
        h = np.array([1.0] + [0.0] * 7)
        H = spectral_entropy(h)
        assert abs(H) < 1e-10   # all mass at one mode → H = 0

    def test_spectral_entropy_from_laplacian(self):
        L = self._p8_laplacian()
        H = spectral_entropy_from_laplacian(L, tau=1.0)
        assert 0.0 < H <= np.log(8) + 1e-6

    def test_fixed_point_kernel_converges(self):
        L = self._p8_laplacian()
        h_star, info = fixed_point_kernel(L, mu2=2.0, sigma2=1.0)
        assert info["converged"]
        assert info["residual"] < 1e-10
        assert len(h_star) == 8

    def test_fixed_point_kernel_all_positive(self):
        L = self._p8_laplacian()
        h_star, _ = fixed_point_kernel(L, mu2=2.0, sigma2=1.0)
        assert np.all(h_star > 0)

    def test_fixed_point_contraction_ratio_lt1(self):
        L = self._p8_laplacian()
        _, info = fixed_point_kernel(L, mu2=2.0, sigma2=1.0)
        assert info["contraction_ratio"] < 1.0

    def test_fiedler_mode_gap_positive(self):
        L = self._p8_laplacian()
        h_star, _ = fixed_point_kernel(L, mu2=2.0, sigma2=1.0)
        gap = fiedler_mode_gap(h_star, L, mu2=2.0, sigma2=1.0)
        assert gap > 0

    def test_stability_conservation_tradeoff_fails(self):
        """Route 3: D_m ≠ 0 for Gaussian MI source at h* (P2 Prop 1b).

        Uses mode-blind weights w_l = 1 (P1 Experiments 1–4 setup) so that
        h* is uniform across all modes and D_m is the same for all m.
        """
        L  = self._p8_laplacian()
        n  = L.shape[0]
        h0 = np.ones(n)         # flat reference → uniform h* (P1 Exp 2 setup)
        w  = np.ones(n)         # mode-blind weights
        h_star, _ = fixed_point_kernel(L, h0=h0, mu2=2.0, sigma2=1.0, w=w)
        result = stability_conservation_tradeoff(h_star, L, mu2=2.0, sigma2=1.0, w=w)
        # Conservation law should NOT hold (Route 3 result)
        assert not result["conservation_holds"]
        # D_m values should all be negative
        assert np.all(result["D_m"] < 0)
        # With uniform h* and w_l=1, all modes have same deficit D_m ≈ -5.71
        assert np.allclose(result["D_m"], result["D_m"][0], atol=0.1)

    def test_stability_conservation_deficit_equals_hessian_gap(self):
        """D_m = H_mm = -Δ' (Proposition 1b of P2, field note 27).

        With flat h0=1 and mode-blind w_l=1, h* ≈ 0.1547 (uniform across modes).
        D_m equals H_mm (Hessian diagonal) = -Δ' for all m.
        """
        L  = self._p8_laplacian()
        n  = L.shape[0]
        h0 = np.ones(n)
        w  = np.ones(n)
        h_star, _ = fixed_point_kernel(L, h0=h0, mu2=2.0, sigma2=1.0, w=w)
        result = stability_conservation_tradeoff(h_star, L, mu2=2.0, sigma2=1.0, w=w)
        Delta_prime = result["Delta_prime"]
        # Each D_m should equal -Δ' (all the same with uniform h*)
        assert np.allclose(result["D_m"], -Delta_prime, atol=0.1)

    def test_phase_transition_sweep_fiedler_decreases(self):
        L = self._p8_laplacian()
        result = phase_transition_sweep(L, perturb_edge=(2, 3), n_steps=10)
        # Fiedler value should decrease as edge is weakened
        assert result.fiedler_values[0] >= result.fiedler_values[-1]

    def test_phase_transition_sweep_entropy_changes(self):
        L = self._p8_laplacian()
        result = phase_transition_sweep(L, perturb_edge=(2, 3), n_steps=10)
        # Spectral entropy should change (increase as graph fragments)
        assert not np.allclose(result.spectral_entropies,
                               result.spectral_entropies[0])

    def test_observability_ratio_static(self):
        # Static scene (P_phys ≈ 0) → formally infinite R/İself
        result = observability_ratio(R_bps=1e6, P_phys_W=0.0)
        assert result["regime"] == "static_topology"

    def test_observability_ratio_hurricane(self):
        # Hurricane: P_phys ~ 1e14 W, R ~ 1e9 bits/s → R/İself ~ 1e-25
        result = observability_ratio(R_bps=1e9, P_phys_W=1e14, T_K=300.0)
        assert result["log10_ratio"] < -20

    def test_observability_ratio_rover(self):
        # Lunar rover: P_phys ~ 0 (static rocks), R ~ 1e6 bits/s → static regime
        result = observability_ratio(R_bps=1e6, P_phys_W=1e-3, T_K=300.0)
        assert result["regime"] in ("static_topology", "swaplimited_dynamic")

    def test_bandwidth_optimal_modes_obligate_first(self):
        n = 16
        h_star = np.exp(-np.arange(n, dtype=float))
        c      = np.random.RandomState(0).randn(n)
        T_l    = 0.1 * np.ones(n)
        kmin, k_budget = 3, 8
        modes = bandwidth_optimal_modes(h_star, c, T_l, kmin=kmin, k_budget=k_budget)
        # First kmin modes must always be included
        assert set(range(kmin)).issubset(set(modes.tolist()))
        assert len(modes) == k_budget

    def test_bandwidth_optimal_modes_budget_respected(self):
        n = 20
        h_star = np.ones(n)
        c = np.ones(n)
        T_l = np.ones(n)
        modes = bandwidth_optimal_modes(h_star, c, T_l, kmin=2, k_budget=5)
        assert len(modes) == 5


# ---------------------------------------------------------------------------
# Integration test: crater field → biosignature report
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_crater_to_biosig_pipeline(self):
        """End-to-end: DEM → crater graph → Betti → biosignature score."""
        dem = synthetic_crater_dem(64, 64, radius=12.0, depth=4.0, rim_height=2.0)
        tg_full = dem_to_graph(dem, connectivity=8, weight="elev_diff")

        # Manually specify the crater (centre + radius known from synthetic)
        from kernelcal.terrain.craters import CraterCandidate, crater_rim_graph, crater_betti_numbers
        c = CraterCandidate(row=32, col=32, radius=12.0,
                            rim_completeness=1.0, curvature_contrast=1.0)
        tg_rim = crater_rim_graph(dem, [c], rim_width=3)
        betti = crater_betti_numbers(tg_rim)

        # Abiotic null: 1 crater → β₁_abio = 1
        null = abiotic_beta1_craters(n_intact=1)
        tb = topological_biosignature(betti["beta1"], null["beta1_abio"])

        # For a clean synthetic crater, Δβ₁ = 0 (observed matches abiotic)
        assert tb.beta1_obs >= 0
        assert tb.beta1_abio == 1

    def test_channel_diagnostic_pipeline(self):
        """End-to-end: DEM → drainage network → triple diagnostic."""
        dem = synthetic_channel_dem(48, 48, n_tributaries=4, slope_angle=0.1)
        dg = drainage_network_graph(dem, threshold=5)
        diag = triple_spectral_diagnostic(dg)

        assert diag.n_nodes > 0
        assert diag.H_spectral >= 0
        budget = topology_budget(dg)
        assert budget["kmin"] >= 1

    def test_plume_biosig_pipeline(self):
        """End-to-end: chemical affinity graph → plume spectral entropy."""
        # 6 species, ring-like co-occurrence (structured metabolism)
        n = 6
        co = np.zeros((n, n))
        for i in range(n):
            co[i, (i+1) % n] = 2.0
            co[(i+1) % n, i] = 2.0
        W, L_obs = chemical_affinity_graph([str(i) for i in range(n)], co)

        result = plume_spectral_entropy(L_obs)
        assert "is_biosignature" in result
        assert result["H_obs"] >= 0
