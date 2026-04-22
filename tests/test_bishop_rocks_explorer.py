"""Tests for ``bishop_rocks_graph_explorer`` — the k-NN graph explorer over
rock-centroid point data.

Covers only the analytical primitives (no matplotlib, no CSV I/O):

- ``LocalFrame``: lon/lat → local metres projection
- ``knn_edges``: k-nearest-neighbour edge set
- ``adjacency_from_edges`` + ``fiedler_value``: Laplacian smallest nonzero
- ``build_betti_quadrant_candidates``: quadrant splitter + unseen fraction
  that feeds the shared ``kernelcal.graph_explorer`` planner
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
# --knn-max-edge-m: Euclidean max-edge-length cap on k-NN edges
# ---------------------------------------------------------------------------


class TestKnnMaxEdge:
    def _line_points(self) -> np.ndarray:
        # 4 collinear rocks spaced at 1 m, 1 m, 3 m intervals.
        #   index : 0    1    2    3
        #   x (m) : 0.0  1.0  2.0  5.0
        # Nearest-neighbour distances:
        #   0 <-> 1 = 1.0   1 <-> 2 = 1.0   2 <-> 3 = 3.0
        # A cap at 2.0 m must drop the (2, 3) edge but keep (0, 1) and (1, 2).
        return np.array(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [5.0, 0.0]], dtype=float
        )

    def test_none_cap_matches_plain_knn(self):
        xy = self._line_points()
        baseline = bishop.knn_edges(xy, k=2)
        assert bishop.knn_edges(xy, k=2, max_edge_m=None) == baseline
        assert bishop.knn_edges(xy, k=2, max_edge_m=0.0) == baseline
        assert bishop.knn_edges(xy, k=2, max_edge_m=-3.0) == baseline
        assert bishop.knn_edges(xy, k=2, max_edge_m=float("nan")) == baseline
        assert bishop.knn_edges(xy, k=2, max_edge_m=float("inf")) == baseline

    def test_cap_drops_edges_longer_than_threshold(self):
        xy = self._line_points()
        edges = bishop.knn_edges(xy, k=2, max_edge_m=2.0)
        # Long edge (2, 3) with length 3.0 m must be gone; short edges stay.
        assert (2, 3) not in edges
        assert (0, 1) in edges
        assert (1, 2) in edges

    def test_cap_can_fully_isolate_a_node(self):
        xy = self._line_points()
        edges = bishop.knn_edges(xy, k=2, max_edge_m=0.5)
        # Every pairwise distance exceeds 0.5 m → empty edge set.
        assert edges == set()

    def test_cap_propagates_through_trait_only(self):
        xy = self._line_points()
        has_trait = np.array([True, True, True, True], dtype=bool)
        full = bishop.knn_edges_trait_only(xy, k=2, has_trait=has_trait)
        capped = bishop.knn_edges_trait_only(
            xy, k=2, has_trait=has_trait, max_edge_m=2.0
        )
        assert (2, 3) in full and (2, 3) not in capped
        # Short edges unchanged.
        assert capped & {(0, 1), (1, 2)} == {(0, 1), (1, 2)}


# ---------------------------------------------------------------------------
# Per-coord circular-equivalent diameter + size-based edge filter
# ---------------------------------------------------------------------------


class TestCoordDiameterAndSizeFilter:
    def _frames(self):
        # 4 coord rocks at x=0..3 m (y=0); two matched traits with distinct
        # areas, two without any trait row.
        coords = bishop.pd.DataFrame({
            "lon": [0.0, 0.0, 0.0, 0.0],
            "lat": [0.0, 0.0, 0.0, 0.0],
            "x_m": [0.0, 1.0, 2.0, 3.0],
            "y_m": [0.0, 0.0, 0.0, 0.0],
        })
        traits = bishop.pd.DataFrame({
            "lon": [0.0, 0.0],
            "lat": [0.0, 0.0],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            # d = 2*sqrt(area/pi); area=pi/4 -> d=1 m, area=pi/400 -> d=0.1 m
            "area_m2": [math.pi * 0.25, math.pi / 400.0],
            "eccentricity": [0.0, 0.0],
        })
        return coords, traits

    def test_returns_circular_equivalent_diameter(self):
        coords, traits = self._frames()
        d = bishop.coord_diameter_m_for_coords(coords, traits, tol_m=0.01)
        assert d[0] == pytest.approx(1.0, rel=1e-6)
        assert d[1] == pytest.approx(0.1, rel=1e-6)
        # Unmatched rocks are NaN.
        assert np.isnan(d[2])
        assert np.isnan(d[3])

    def test_empty_traits_gives_all_nan(self):
        coords, _ = self._frames()
        empty = bishop.pd.DataFrame(
            columns=["lon", "lat", "x_m", "y_m", "area_m2", "eccentricity"]
        )
        d = bishop.coord_diameter_m_for_coords(coords, empty, tol_m=1.0)
        assert d.shape == (4,)
        assert np.all(np.isnan(d))

    def test_tolerance_rejects_far_matches(self):
        coords, traits = self._frames()
        # Shift traits 10 m away; with tol_m = 0.5 no match should succeed.
        traits = traits.copy()
        traits["x_m"] = traits["x_m"] + 10.0
        d = bishop.coord_diameter_m_for_coords(coords, traits, tol_m=0.5)
        assert np.all(np.isnan(d))

    def test_size_filter_excludes_small_rocks_from_edges(self):
        # 4 colinear rocks: 0 (d=1m), 1 (d=0.1m), 2 (d=1m), 3 (d=1m).
        # With min_edge_diameter=0.5m, node 1 should drop out of the edge
        # set (same behaviour as no-trait), leaving edges only among 0,2,3.
        coords, traits = self._frames()
        coords = bishop.pd.concat([
            coords,
            bishop.pd.DataFrame({
                "lon": [0.0, 0.0], "lat": [0.0, 0.0],
                "x_m": [2.0, 3.0], "y_m": [0.0, 0.0],
            })
        ], ignore_index=True)
        # Add trait rows for rocks at x=2 and x=3 with d=1 m.
        extra = bishop.pd.DataFrame({
            "lon": [0.0, 0.0], "lat": [0.0, 0.0],
            "x_m": [2.0, 3.0], "y_m": [0.0, 0.0],
            "area_m2": [math.pi * 0.25, math.pi * 0.25],
            "eccentricity": [0.0, 0.0],
        })
        traits = bishop.pd.concat([traits, extra], ignore_index=True)

        has_trait = bishop.trait_mask_for_coords(coords, traits, tol_m=0.01)
        diameters = bishop.coord_diameter_m_for_coords(coords, traits, tol_m=0.01)
        size_ok = np.where(np.isfinite(diameters), diameters >= 0.5, False)
        edge_eligible = has_trait & size_ok
        # Node at original index 1 (x=1, d=0.1m) is trait-linked but fails size.
        assert has_trait[1] and not edge_eligible[1]

        xy = np.c_[
            coords["x_m"].to_numpy(float), coords["y_m"].to_numpy(float)
        ]
        edges = bishop.knn_edges_trait_only(xy, k=2, has_trait=edge_eligible)
        # Node 1 (the 10 cm pebble) must not appear in any edge.
        assert all(1 not in e for e in edges)
        # The large rocks still form edges among themselves.
        large_nodes = {a for a, _ in edges} | {b for _, b in edges}
        assert 1 not in large_nodes
        assert len(edges) >= 1

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
# Legacy-API removal — make sure the trimmed bishop module is in lock-step
# with DEM and no alternate motion policies / utilities remain behind.
# ---------------------------------------------------------------------------


class TestLegacyAPIsRemoved:
    @pytest.mark.parametrize("attr", [
        "radius_edges",
        "quadrant_metrics",
        "MOVE_DIRS",
        "_normalize",
        "spiral_path",
        "fiedler_value_and_ipr",
    ])
    def test_attribute_not_exposed(self, attr):
        assert not hasattr(bishop, attr), (
            f"bishop_rocks_graph_explorer should no longer expose {attr!r} "
            "after the quadrant-Betti-only refactor."
        )


# ---------------------------------------------------------------------------
# Betti-quadrant candidate builder (shared planner with the DEM explorer)
# ---------------------------------------------------------------------------


class TestBettiQuadrantCandidates:
    def _setup(self):
        """Four clearly-separated clusters, one in each diagonal quadrant."""
        rng = np.random.default_rng(0)

        def cluster(cx, cy, n):
            return rng.normal(loc=(cx, cy), scale=0.8, size=(n, 2))

        # Variable counts per quadrant so the "best β₁/n" result is meaningful.
        xy = np.vstack([
            cluster(-10.0, +10.0, 30),   # NW — dense
            cluster(+10.0, +10.0, 10),   # NE — sparse
            cluster(-10.0, -10.0, 20),   # SW
            cluster(+10.0, -10.0, 25),   # SE
        ])
        has_trait = np.ones(len(xy), dtype=bool)
        return xy, has_trait

    def test_returns_one_candidate_per_quadrant(self):
        xy, has_trait = self._setup()
        cands = bishop.build_betti_quadrant_candidates(
            0.0, 0.0, scan_side_m=40.0, step_m=10.0,
            coord_xy=xy, coord_has_trait=has_trait, knn=4,
        )
        names = sorted(c.name for c in cands)
        assert names == ["NE", "NW", "SE", "SW"]
        for c in cands:
            assert c.n_nodes > 0
            assert c.beta0 >= 1
            assert c.beta1 >= 0

    def test_target_positions_match_metric_convention(self):
        xy, has_trait = self._setup()
        cands = bishop.build_betti_quadrant_candidates(
            0.0, 0.0, scan_side_m=40.0, step_m=10.0,
            coord_xy=xy, coord_has_trait=has_trait, knn=4,
        )
        by_name = {c.name: c for c in cands}
        # NW is (-x, +y) in metric coords.
        assert by_name["NW"].extra["target_xy"] == pytest.approx((-10.0, 10.0))
        assert by_name["NE"].extra["target_xy"] == pytest.approx((10.0, 10.0))
        assert by_name["SW"].extra["target_xy"] == pytest.approx((-10.0, -10.0))
        assert by_name["SE"].extra["target_xy"] == pytest.approx((10.0, -10.0))

    def test_empty_window_returns_no_candidates(self):
        xy, has_trait = self._setup()
        # Centre far outside the clusters → the square window is empty.
        cands = bishop.build_betti_quadrant_candidates(
            1000.0, 1000.0, scan_side_m=10.0, step_m=5.0,
            coord_xy=xy, coord_has_trait=has_trait, knn=4,
        )
        assert cands == []

    def test_raster_coverage_reduces_unseen_frac(self):
        """Matches the drone-DEM explorer: ``unseen = 1 - mean(visited[target])``.

        With an unpainted raster every target is fully unseen; painting the
        raster everywhere drops unseen to 0 at every candidate's target.
        """
        from kernelcal.graph_explorer import CoverageRaster
        xy, has_trait = self._setup()

        fresh = CoverageRaster(bbox=(-25.0, 25.0, -25.0, 25.0), resolution_m=1.0)
        no_cov = bishop.build_betti_quadrant_candidates(
            0.0, 0.0, scan_side_m=50.0, step_m=10.0,
            coord_xy=xy, coord_has_trait=has_trait, knn=4,
            visited_raster=fresh,
        )

        painted = CoverageRaster(bbox=(-25.0, 25.0, -25.0, 25.0), resolution_m=1.0)
        # Paint the entire raster visited by marking a huge square covering
        # the whole bbox.
        painted.mark_square(0.0, 0.0, 1000.0)
        all_cov = bishop.build_betti_quadrant_candidates(
            0.0, 0.0, scan_side_m=50.0, step_m=10.0,
            coord_xy=xy, coord_has_trait=has_trait, knn=4,
            visited_raster=painted,
        )
        assert all(c.unseen_frac == 1.0 for c in no_cov)
        assert all(c.unseen_frac == 0.0 for c in all_cov)

    def test_unseen_matches_dem_area_semantics(self):
        """Rockless target area should not be treated as attractive.

        Unlike the previous per-rock semantics, the area-based unseen is
        well-defined even when the target window has no rocks: a target
        sitting entirely inside the painted region returns unseen=0
        regardless of how sparse the rock cloud is there.  This is the
        DEM-style behaviour the user requested.
        """
        from kernelcal.graph_explorer import CoverageRaster
        # Rocks only in NE.
        rng = np.random.default_rng(1)
        xy = rng.normal(loc=(5.0, 5.0), scale=0.5, size=(20, 2))
        has_trait = np.ones(len(xy), dtype=bool)

        # Pre-paint the SW region as visited.
        raster = CoverageRaster(bbox=(-30.0, 30.0, -30.0, 30.0), resolution_m=0.5)
        raster.mark_square(-15.0, -15.0, 20.0)

        cands = bishop.build_betti_quadrant_candidates(
            5.0, 5.0, scan_side_m=4.0, step_m=20.0,
            coord_xy=xy, coord_has_trait=has_trait, knn=3,
            visited_raster=raster,
        )
        by_name = {c.name: c for c in cands}
        # SW target at (-15, -15) lies fully inside the painted area → 0.0.
        if "SW" in by_name:
            assert by_name["SW"].unseen_frac == 0.0

    def test_bbox_clamps_target_positions(self):
        xy, has_trait = self._setup()
        # Tight bbox inside the clusters; targets should be clamped to it.
        bbox = (-3.0, 3.0, -3.0, 3.0)
        cands = bishop.build_betti_quadrant_candidates(
            0.0, 0.0, scan_side_m=50.0, step_m=10.0,
            coord_xy=xy, coord_has_trait=has_trait, knn=4,
            bbox=bbox,
        )
        for c in cands:
            tx, ty = c.extra["target_xy"]
            assert -3.0 <= tx <= 3.0 + 1e-9
            assert -3.0 <= ty <= 3.0 + 1e-9
