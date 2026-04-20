"""Tests for ``bishop_rocks_graph_explorer`` — the quadrant-adaptive graph
explorer over rock-centroid point data.

Covers only the analytical primitives (no matplotlib, no CSV I/O):

- ``LocalFrame``: lon/lat → local metres projection
- ``knn_edges``: k-nearest-neighbour edge set
- ``radius_edges``: radius-ball edge set
- ``adjacency_from_edges`` + ``fiedler_value``: Laplacian smallest nonzero
- ``quadrant_metrics``: per-quadrant β₀/β₁/λ₂ scoring, momentum factor,
  unseen-rock bonus, and direction-of-best consistency with the data
- ``spiral_path``: rectangular spiral is bbox-clamped and has the requested
  number of points
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

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
# Graph construction (k-NN and radius edges)
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

    def test_radius_edges_empty_when_radius_tiny(self):
        xy = self._square_grid(spacing=1.0)
        assert bishop.radius_edges(xy, r=0.1) == set()

    def test_radius_edges_one_catches_cardinal_neighbours(self):
        xy = self._square_grid(n=3, spacing=1.0)   # centre at index 4
        edges = bishop.radius_edges(xy, r=1.01)
        # centre should connect to its 4 cardinal neighbours (indices 1, 3, 5, 7)
        cardinal = {(1, 4), (3, 4), (4, 5), (4, 7)}
        assert cardinal.issubset(edges)

    def test_radius_edges_grow_with_radius(self):
        xy = self._square_grid(n=4, spacing=1.0)   # 16 points
        small = bishop.radius_edges(xy, r=1.01)
        large = bishop.radius_edges(xy, r=2.5)
        assert small.issubset(large)
        assert len(large) > len(small)


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
# Quadrant metrics
# ---------------------------------------------------------------------------


class TestQuadrantMetrics:
    def _build_quadrants(self, seed=0):
        """Place dense cluster in NE, sparse cluster in SW (centred at origin)."""
        rng = np.random.default_rng(seed)
        ne = rng.normal(loc=(4.0, 4.0), scale=0.8, size=(40, 2))
        sw = rng.normal(loc=(-4.0, -4.0), scale=0.8, size=(6, 2))
        xy = np.vstack([ne, sw])
        A = bishop.adjacency_from_edges(len(xy), bishop.knn_edges(xy, k=4))
        return xy, A

    def test_quadrant_count_and_names(self):
        xy, A = self._build_quadrants()
        qs = bishop.quadrant_metrics(
            xy, A, center=(0.0, 0.0),
            w_beta1=1.0, w_fiedler=1.0,
        )
        names = [q.name for q in qs]
        assert names == ["NE", "NW", "SW", "SE"]

    def test_dense_ne_cluster_wins_on_beta1(self):
        """With only β₁ weighted, the big NE cluster wins (more cycles)."""
        xy, A = self._build_quadrants()
        qs = bishop.quadrant_metrics(
            xy, A, center=(0.0, 0.0),
            w_beta1=1.0, w_fiedler=0.0,
        )
        ne = next(q for q in qs if q.name == "NE")
        sw = next(q for q in qs if q.name == "SW")
        assert ne.n_nodes > sw.n_nodes
        assert ne.beta1 > sw.beta1
        assert ne.score > sw.score

    def test_empty_quadrant_scores_zero(self):
        xy = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 1.5]])   # all NE
        A = bishop.adjacency_from_edges(3, bishop.knn_edges(xy, k=1))
        qs = bishop.quadrant_metrics(
            xy, A, center=(0.0, 0.0),
            w_beta1=1.0, w_fiedler=1.0,
        )
        for q in qs:
            if q.name != "NE":
                assert q.n_nodes == 0
                assert q.score == 0.0

    def test_momentum_bonus_and_penalty(self):
        xy, A = self._build_quadrants()
        # No momentum.
        q0 = {q.name: q.score for q in bishop.quadrant_metrics(
            xy, A, center=(0.0, 0.0),
            w_beta1=1.0, w_fiedler=1.0, w_momentum=0.0,
        )}
        # Prev move was toward NE → NE score should grow, SW should shrink.
        q_ne = {q.name: q.score for q in bishop.quadrant_metrics(
            xy, A, center=(0.0, 0.0),
            w_beta1=1.0, w_fiedler=1.0,
            prev_dir=(1.0 / math.sqrt(2), 1.0 / math.sqrt(2)),
            w_momentum=0.45,
        )}
        assert q_ne["NE"] > q0["NE"]
        assert q_ne["SW"] < q0["SW"] or q0["SW"] == 0.0

    def test_unseen_bonus(self):
        xy, A = self._build_quadrants()
        # Mark *all* NE rocks as already-seen ⇒ unseen_mask zero in NE.
        n = len(xy)
        unseen = np.ones(n, dtype=bool)
        ne_idx = np.where((xy[:, 0] > 0) & (xy[:, 1] > 0))[0]
        unseen[ne_idx] = False
        q_with = {q.name: q.score for q in bishop.quadrant_metrics(
            xy, A, center=(0.0, 0.0),
            w_beta1=0.0, w_fiedler=0.0,
            w_unseen=5.0, unseen_mask=unseen,
        )}
        # NE has zero unseen rocks → score 0; SW has unseen rocks → positive score.
        assert q_with["NE"] == 0.0
        assert q_with["SW"] > 0.0


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
