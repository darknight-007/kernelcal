"""Tests for ``bishop_rocks_graph_explorer`` — the k-NN graph explorer over
rock-centroid point data.

Covers only the analytical primitives (no matplotlib, no CSV I/O):

- ``LocalFrame``: lon/lat → local metres projection
- ``knn_edges``: k-nearest-neighbour edge set
- ``adjacency_from_edges`` + ``fiedler_value``: Laplacian smallest nonzero
- ``MOVE_DIRS`` + ``_normalize``: directional candidate utilities for greedy
  motion policy
- ``spiral_path``: rectangular spiral is bbox-clamped and has the requested
  number of points
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# The explorer script lives under ``examples/bishop/`` after PR-E; add that
# directory to sys.path so ``importorskip`` can find the module by basename.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "examples" / "bishop"))

bishop = pytest.importorskip("bishop_rocks_graph_explorer")


# ---------------------------------------------------------------------------
# LocalFrame (equirectangular lon/lat -> metres)
# ---------------------------------------------------------------------------


class TestLocalFrame:
    def test_origin_maps_to_zero(self):
        frame = bishop.LocalFrame(lon0=-118.44, lat0=37.45)
        x, y = frame.to_xy(np.array([-118.44]), np.array([37.45]))
        assert abs(float(x[0])) < 1e-6
        assert abs(float(y[0])) < 1e-6

    def test_one_degree_latitude_is_about_111km(self):
        frame = bishop.LocalFrame(lon0=-118.44, lat0=37.45)
        _, y = frame.to_xy(np.array([-118.44]), np.array([38.45]))
        assert 111_000 < float(y[0]) < 111_500

    def test_longitude_scales_with_cos_latitude(self):
        frame = bishop.LocalFrame(lon0=0.0, lat0=60.0)
        x, _ = frame.to_xy(np.array([1.0]), np.array([60.0]))
        # 111,320 * cos(60°) ≈ 55,660 m
        assert 55_000 < float(x[0]) < 56_500


# ---------------------------------------------------------------------------
# Graph construction (k-NN edges)
# ---------------------------------------------------------------------------


class TestGraphEdges:
    def _square_grid(self, n=3, spacing=1.0) -> np.ndarray:
        """3x3 integer grid of 9 points; spacing controls the lattice constant."""
        ys, xs = np.mgrid[0:n, 0:n]
        return np.c_[xs.ravel() * spacing, ys.ravel() * spacing].astype(float)

    def test_knn_edges_symmetric_and_deduped(self):
        xy = self._square_grid()
        edges = bishop.knn_edges(xy, k=2)
        for a, b in edges:
            assert a < b, "edge tuples must be sorted (a, b) with a < b"
        assert len(edges) == len(set(edges))

    def test_knn_k_eq_1_gives_at_least_n_over_2_edges(self):
        xy = self._square_grid()     # 9 points
        edges = bishop.knn_edges(xy, k=1)
        assert len(edges) >= len(xy) // 2

    def test_knn_edges_trait_only_excludes_nodes_without_traits(self):
        # 4 points in a line. Node 1 has no traits, so it must have no edges.
        xy = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        has_trait = np.array([True, False, True, True], dtype=bool)
        edges = bishop.knn_edges_trait_only(xy, k=2, has_trait=has_trait)
        for a, b in edges:
            assert has_trait[a] and has_trait[b]
        assert all(1 not in e for e in edges)

    def test_trait_mask_for_coords_matches_lonlat_subset(self):
        coords = bishop.pd.DataFrame(
            {"lon": [-1.0, 0.0, 1.0], "lat": [10.0, 11.0, 12.0]}
        )
        traits = bishop.pd.DataFrame(
            {
                "lon": [0.0, 1.0],
                "lat": [11.0, 12.0],
                "area_m2": [1.0, 2.0],
                "eccentricity": [0.5, 0.7],
            }
        )
        mask = bishop.trait_mask_for_coords(coords, traits)
        assert mask.tolist() == [False, True, True]

# ---------------------------------------------------------------------------
# Laplacian Fiedler value
# ---------------------------------------------------------------------------


class TestFiedler:
    def test_fiedler_zero_for_disconnected_graph(self):
        # Two disjoint edges ⇒ 2 components ⇒ λ₂ = 0 (two zero eigenvalues).
        edges = {(0, 1), (2, 3)}
        A = bishop.adjacency_from_edges(4, edges)
        assert bishop.fiedler_value(A) == pytest.approx(0.0, abs=1e-8)

    def test_fiedler_positive_for_path_graph(self):
        # P4: 0-1-2-3; λ₂ = 2 - 2cos(π/4) ≈ 0.586
        edges = {(0, 1), (1, 2), (2, 3)}
        A = bishop.adjacency_from_edges(4, edges)
        assert bishop.fiedler_value(A) == pytest.approx(
            2.0 - 2.0 * math.cos(math.pi / 4.0), rel=1e-3
        )

    def test_fiedler_zero_on_empty_graph(self):
        A = bishop.adjacency_from_edges(5, set())
        assert bishop.fiedler_value(A) == pytest.approx(0.0, abs=1e-8)


# ---------------------------------------------------------------------------
# Directional-candidate utilities (greedy motion policy)
# ---------------------------------------------------------------------------


class TestDirectionalUtilities:
    def test_legacy_radius_and_quadrant_apis_removed(self):
        assert not hasattr(bishop, "radius_edges")
        assert not hasattr(bishop, "quadrant_metrics")

    def test_move_dirs_contains_stay_and_cardinals(self):
        names = [name for name, _vec in bishop.MOVE_DIRS]
        assert "STAY" in names
        for k in ("N", "S", "E", "W", "NE", "NW", "SE", "SW"):
            assert k in names

    def test_normalize_unit_and_zero(self):
        ux, uy = bishop._normalize(3.0, 4.0)
        assert ux == pytest.approx(0.6)
        assert uy == pytest.approx(0.8)
        zx, zy = bishop._normalize(0.0, 0.0)
        assert zx == 0.0
        assert zy == 0.0


# ---------------------------------------------------------------------------
# Spiral path (fallback planner)
# ---------------------------------------------------------------------------


class TestSpiralPath:
    def test_shape_and_bbox_clamp(self):
        pts = bishop.spiral_path(0.0, 10.0, 0.0, 10.0, step=2.0, n_steps=25)
        assert pts.shape == (25, 2)
        assert pts[:, 0].min() >= 0.0 - 1e-9
        assert pts[:, 0].max() <= 10.0 + 1e-9
        assert pts[:, 1].min() >= 0.0 - 1e-9
        assert pts[:, 1].max() <= 10.0 + 1e-9

    def test_starts_at_bbox_centre(self):
        pts = bishop.spiral_path(-5.0, 5.0, -5.0, 5.0, step=1.0, n_steps=3)
        assert pts[0, 0] == pytest.approx(0.0)
        assert pts[0, 1] == pytest.approx(0.0)
