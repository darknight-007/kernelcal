"""Tests for PR-C of CR-2026-04-26: planned-vs-organic ΔH receipt.

Covers
------

* :mod:`kernelcal.urban.spectrum` -- ``betti_zero``, ``spectral_entropy``,
  ``normalised_top_k_spectrum``, ``sigma_matched_spectrum_diff``.

* :mod:`kernelcal.urban.synthetic` -- ``make_grid_layout``,
  ``make_fringe_layout``, ``synthetic_city_graph`` in both
  ``knn`` and ``road_knn`` modes.

* The end-to-end synthetic experiment in
  ``experiments/planned_vs_organic_dH.py``: structural-prediction
  smoke that runs the script on the committed YAML configs and
  checks the receipt structure + the β₀ disconnection-as-signal
  claim.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from kernelcal.urban import (
    CityGraph,
    betti_zero,
    make_fringe_layout,
    make_fringe_road_segments,
    make_grid_layout,
    make_grid_road_segments,
    normalised_top_k_spectrum,
    sigma_matched_spectrum_diff,
    spectral_diagnostics,
    spectral_entropy,
    synthetic_city_graph,
)


REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------\
# spectrum helpers
# ---------------------------------------------------------------------------\

class TestBettiZero:
    def test_two_disconnected_components(self):
        eigs = np.array([0.0, 0.0, 0.5, 1.0, 1.5])
        assert betti_zero(eigs) == 2

    def test_tolerance_handles_numerical_drift(self):
        eigs = np.array([1e-15, -1e-15, 0.5, 1.0])
        assert betti_zero(eigs) == 2

    def test_only_strict_zero_below_tol(self):
        eigs = np.array([1e-3, 1e-12, 1.0])
        assert betti_zero(eigs, tol=1e-9) == 1

    def test_rejects_non_1d(self):
        with pytest.raises(ValueError):
            betti_zero(np.zeros((3, 3)))


class TestSpectralEntropy:
    def test_uniform_spectrum_normalised_entropy_equals_one(self):
        eigs = np.ones(10)
        H = spectral_entropy(eigs)
        assert H == pytest.approx(1.0, abs=1e-12)

    def test_dirac_spectrum_zero_entropy(self):
        eigs = np.zeros(10)
        eigs[0] = 1.0
        H = spectral_entropy(eigs)
        assert H == pytest.approx(0.0, abs=1e-12)

    def test_zero_eigenvalues_dropped(self):
        # Two real masses (0.5 each) plus two near-zero modes.  Result
        # should be the same as if the zero modes were never there.
        eigs = np.array([0.0, 0.0, 0.5, 0.5])
        H = spectral_entropy(eigs)
        assert H == pytest.approx(1.0, abs=1e-12)

    def test_un_normalised_path(self):
        eigs = np.ones(4)
        H = spectral_entropy(eigs, normalise=False)
        assert H == pytest.approx(math.log(4), abs=1e-12)

    def test_only_zero_eigenvalues_returns_zero(self):
        H = spectral_entropy(np.array([0.0, 0.0, 1e-15]))
        assert H == 0.0

    def test_concentrated_lower_than_uniform(self):
        # A sharply peaked spectrum has lower normalised entropy than
        # a uniform one at the same n.  This is the structural
        # interpretation the receipt depends on.
        n = 10
        peaked = np.zeros(n)
        peaked[0] = 0.95
        peaked[1:] = 0.05 / (n - 1)
        uniform = np.ones(n) / n
        assert spectral_entropy(peaked) < spectral_entropy(uniform)


class TestTopKSpectrum:
    def test_sums_to_one(self):
        eigs = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        p = normalised_top_k_spectrum(eigs, k=3)
        assert p.sum() == pytest.approx(1.0, abs=1e-12)
        assert p.size == 3

    def test_drops_zero_modes(self):
        eigs = np.array([0.0, 1.0, 2.0])
        p = normalised_top_k_spectrum(eigs, k=10)
        assert p.size == 2

    def test_sigma_matched_distance_self_is_zero(self):
        eigs = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        d = sigma_matched_spectrum_diff(eigs, eigs, k=4)
        assert d == pytest.approx(0.0, abs=1e-12)

    def test_sigma_matched_distance_concentration_vs_uniform(self):
        # Concentrated vs uniform spectra over the same number of
        # non-zero modes: TV distance is positive and bounded by 1.
        peaked = np.array([10.0, 0.5, 0.5, 0.5])
        uniform = np.array([1.0, 1.0, 1.0, 1.0])
        d = sigma_matched_spectrum_diff(peaked, uniform, k=4)
        assert 0.0 < d <= 1.0


class TestSpectralDiagnosticsBundle:
    def test_dict_shape_and_types(self):
        eigs = np.linspace(0.0, 1.0, 8)
        d = spectral_diagnostics(eigs)
        assert set(d.keys()) >= {
            "n_eigvals", "beta_0",
            "spectral_entropy_nats", "spectral_entropy_normalised",
            "top_k",
        }
        assert isinstance(d["beta_0"], int)
        assert isinstance(d["top_k"], list)


# ---------------------------------------------------------------------------\
# synthetic CityGraph builders
# ---------------------------------------------------------------------------\

class TestSyntheticLayouts:
    def test_grid_layout_shape(self):
        positions = make_grid_layout(n_blocks_x=5, n_blocks_y=4, block_size_m=50.0)
        assert positions.shape == (20, 2)

    def test_fringe_layout_size_matches_request(self):
        positions = make_fringe_layout(n_buildings=50, n_seeds=4, seed=7)
        assert positions.shape[0] == 50
        assert positions.shape[1] == 2

    def test_grid_road_segments_consistent(self):
        nodes, edges = make_grid_road_segments(n_blocks_x=4, n_blocks_y=3, block_size_m=10.0)
        assert nodes.shape == (12, 2)
        # 4*3 grid: (3 horizontal × 3 rows) + (4 columns × 2 vertical) = 9 + 8 = 17
        assert edges.shape == (17, 2)

    def test_fringe_road_segments_have_branches(self):
        nodes, edges = make_fringe_road_segments(
            n_seeds=3, n_branches_per_seed=2, branch_length_m=50.0,
        )
        # backbone (n_seeds - 1) + branches (n_seeds * n_branches)
        assert edges.shape[0] == (3 - 1) + 3 * 2


class TestSyntheticCityGraphKNN:
    def test_grid_knn_one_component(self):
        positions = make_grid_layout(n_blocks_x=6, n_blocks_y=6, jitter_m=2.0)
        cg = synthetic_city_graph("g", "synthetic:grid", positions, k=8)
        assert isinstance(cg, CityGraph)
        assert cg.graph_mode == "knn"
        assert cg.eigvals.shape == (positions.shape[0],)
        assert cg.eigvals[0] < 1e-6
        assert betti_zero(cg.eigvals) == 1

    def test_road_knn_grid_one_component(self):
        positions = make_grid_layout(n_blocks_x=5, n_blocks_y=5)
        nodes, edges = make_grid_road_segments(n_blocks_x=5, n_blocks_y=5)
        cg = synthetic_city_graph(
            "g", "synthetic:grid", positions,
            road_nodes=nodes, road_edges=edges, k=4,
        )
        assert cg.graph_mode == "road_knn"
        assert cg.road_meta["synthetic"] is True
        assert betti_zero(cg.eigvals) == 1

    def test_road_knn_fringe_disconnected(self):
        # The fringe road network is intentionally sparse; clusters
        # should disconnect under road_knn even when they look close
        # by Euclidean distance.
        positions = make_fringe_layout(n_buildings=48, n_seeds=6, seed=42)
        nodes, edges = make_fringe_road_segments(n_seeds=6, seed=42)
        cg_road = synthetic_city_graph(
            "f", "synthetic:fringe", positions,
            road_nodes=nodes, road_edges=edges, k=6,
        )
        cg_knn = synthetic_city_graph(
            "f", "synthetic:fringe", positions, k=6,
        )
        b0_road = betti_zero(cg_road.eigvals)
        b0_knn = betti_zero(cg_knn.eigvals)
        # The structural prediction: road_knn surfaces more
        # disconnection on the fringe than Euclidean k-NN does.
        assert b0_road >= b0_knn
        assert b0_road >= 2

    def test_synthetic_is_psd(self):
        positions = make_grid_layout(n_blocks_x=4, n_blocks_y=4)
        cg = synthetic_city_graph("g", "synthetic:grid", positions, k=4)
        assert cg.eigvals.min() >= -1e-9
        assert np.allclose(cg.W, cg.W.T)


# ---------------------------------------------------------------------------\
# end-to-end synthetic receipt
# ---------------------------------------------------------------------------\

class TestPlannedVsOrganicReceipt:
    """Run the experiment script in synthetic mode and check the receipt."""

    @pytest.fixture(scope="class")
    def receipt(self, tmp_path_factory) -> dict:
        tmp = tmp_path_factory.mktemp("planned_vs_organic")
        out = tmp / "receipt.json"
        env = {"PYTHONPATH": str(REPO)}
        result = subprocess.run(
            [
                sys.executable,
                str(REPO / "experiments" / "planned_vs_organic_dH.py"),
                "--mode", "synthetic",
                "--grid-config", str(REPO / "experiments/configs/synthetic_grid.yaml"),
                "--fringe-config", str(REPO / "experiments/configs/synthetic_fringe.yaml"),
                "--output", str(out),
            ],
            capture_output=True, text=True, env={**env},
        )
        assert result.returncode == 0, result.stderr
        return json.loads(out.read_text())

    def test_receipt_structure(self, receipt):
        assert "viewports" in receipt
        assert "delta_H" in receipt
        assert "delta_delta_H" in receipt
        assert "beta_0" in receipt
        assert receipt["mode"] == "synthetic"
        for v in ("grid", "fringe"):
            assert v in receipt["viewports"]
            for m in ("road_knn", "knn"):
                assert m in receipt["viewports"][v]["modes"]

    def test_disconnection_as_signal_in_road_knn(self, receipt):
        # Fringe has more components than grid in road_knn mode.
        b0 = receipt["beta_0"]
        assert b0["fringe"]["road_knn"] > b0["grid"]["road_knn"]
        assert b0["grid"]["road_knn"] == 1

    def test_road_knn_mode_amplifies_disconnection_on_fringe(self, receipt):
        # The road_knn fringe sees more components than the Euclidean
        # knn fringe -- this is the FN 107 disconnection-as-signal
        # claim.
        b0 = receipt["beta_0"]
        assert b0["fringe"]["road_knn"] >= b0["fringe"]["knn"]

    def test_runtime_under_threshold(self, receipt):
        assert receipt["runtime_seconds"] < 30.0
