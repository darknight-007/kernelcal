"""Tests for ``kernelcal.graph_explorer`` — the shared quadrant-Betti planner
used by both the drone-DEM and bishop-rocks explorers.

Covers the public contract without depending on either example script:

- :func:`score_betti_candidate` is the canonical
  ``w_beta1·clip(β₁/n) − w_beta0·clip(β₀/n) + w_unseen·unseen`` formula.
- :func:`choose_best_candidate` applies the revisit penalty, breaks ties
  cyclically, and handles the empty-candidate edge case.
- Quadrant convention helpers round-trip between image (row grows south) and
  metric (y grows north) sign conventions as expected.
"""

from __future__ import annotations

import math

import pytest

import numpy as np

from kernelcal.graph_explorer import (
    BettiWeights,
    CameraModel,
    Candidate,
    CoverageRaster,
    QUADRANT_NAMES,
    QUADRANT_OFFSETS_IMAGE,
    QUADRANT_OFFSETS_METRIC,
    ScoredCandidate,
    choose_best_candidate,
    score_betti_candidate,
)


# ---------------------------------------------------------------------------
# score_betti_candidate
# ---------------------------------------------------------------------------


class TestBettiScore:
    def test_formula_matches_paper(self):
        # score = 2.5 * clip(β₁/n=0.1, 0, 1)   = 0.25
        #       - 0.5 * clip(β₀/n=1/30, 0, 1) ≈ -0.01667
        #       + 5.0 * unseen=0.5            = 2.5
        #       => ≈ 2.7333
        w = BettiWeights()
        cand = Candidate(
            name="NW", position=(0, 0),
            beta0=1, beta1=3, n_nodes=30, unseen_frac=0.5,
        )
        s = score_betti_candidate(cand, w)
        assert s == pytest.approx(
            2.5 * 0.1 - 0.5 * (1.0 / 30.0) + 5.0 * 0.5, rel=1e-9
        )

    def test_beta_terms_are_clipped_to_unit_interval(self):
        # β₁/n = 10/5 = 2.0 but should be clipped to 1.0.
        w = BettiWeights(w_beta1=1.0, w_beta0=0.0, w_unseen=0.0, revisit_penalty=0.0)
        cand = Candidate("SW", (0, 0), beta0=0, beta1=10, n_nodes=5, unseen_frac=0.0)
        assert score_betti_candidate(cand, w) == pytest.approx(1.0)

    def test_zero_nodes_do_not_raise(self):
        # n_nodes=0 is clamped to 1 internally; β₀/β₁/unseen stay zero.
        w = BettiWeights()
        cand = Candidate("NE", (0, 0), beta0=0, beta1=0, n_nodes=0, unseen_frac=0.0)
        assert math.isfinite(score_betti_candidate(cand, w))

    def test_higher_unseen_always_wins_when_topology_ties(self):
        w = BettiWeights()
        base = dict(beta0=1, beta1=1, n_nodes=10)
        low = Candidate("NW", (0, 0), **base, unseen_frac=0.1)
        high = Candidate("SE", (1, 1), **base, unseen_frac=0.9)
        assert score_betti_candidate(high, w) > score_betti_candidate(low, w)


# ---------------------------------------------------------------------------
# choose_best_candidate
# ---------------------------------------------------------------------------


class TestChooseBestCandidate:
    def _three_cands(self) -> list[Candidate]:
        return [
            Candidate("NW", (0, 0), beta0=1, beta1=3, n_nodes=30, unseen_frac=0.5),
            Candidate("NE", (0, 1), beta0=5, beta1=0, n_nodes=30, unseen_frac=0.1),
            Candidate("SW", (1, 0), beta0=1, beta1=3, n_nodes=30, unseen_frac=0.5),
        ]

    def test_empty_candidates_returns_none(self):
        w = BettiWeights()
        best, s, scored = choose_best_candidate([], w)
        assert best is None
        assert s == 0.0
        assert scored == []

    def test_returns_scored_list_in_input_order(self):
        w = BettiWeights()
        cands = self._three_cands()
        _, _, scored = choose_best_candidate(cands, w)
        assert [sc.candidate.name for sc in scored] == ["NW", "NE", "SW"]
        assert all(isinstance(sc, ScoredCandidate) for sc in scored)

    def test_revisit_penalty_is_subtracted_for_matching_positions(self):
        w = BettiWeights(revisit_penalty=0.5)
        cands = self._three_cands()
        _, _, scored = choose_best_candidate(
            cands, w, recent_positions=[(0, 0)]
        )
        assert scored[0].score == pytest.approx(scored[2].score - 0.5)

    def test_tie_break_rotates_cyclically(self):
        # NW and SW tie. tie_break_index 0 => NW, index 1 => SW, index 2 => NW again.
        w = BettiWeights()
        cands = self._three_cands()
        assert choose_best_candidate(cands, w, tie_break_index=0)[0].name == "NW"
        assert choose_best_candidate(cands, w, tie_break_index=1)[0].name == "SW"
        assert choose_best_candidate(cands, w, tie_break_index=2)[0].name == "NW"

    def test_revisit_penalty_breaks_tie_against_recent_candidate(self):
        # Without penalty NW and SW tie; with penalty on NW, SW wins outright.
        w = BettiWeights(revisit_penalty=0.5)
        cands = self._three_cands()
        best, _, _ = choose_best_candidate(
            cands, w, recent_positions=[(0, 0)], tie_break_index=0
        )
        assert best.name == "SW"


# ---------------------------------------------------------------------------
# Quadrant conventions
# ---------------------------------------------------------------------------


class TestQuadrantConventions:
    def test_names_and_keys_line_up(self):
        assert set(QUADRANT_NAMES) == set(QUADRANT_OFFSETS_IMAGE.keys())
        assert set(QUADRANT_NAMES) == set(QUADRANT_OFFSETS_METRIC.keys())
        assert len(QUADRANT_NAMES) == 4

    def test_image_vs_metric_agree_on_east_west_axis(self):
        # East column sign matches between conventions (x grows east in both;
        # the sign convention only flips on the north/south axis).
        for name in QUADRANT_NAMES:
            img_dc = QUADRANT_OFFSETS_IMAGE[name][1]
            metric_dx = QUADRANT_OFFSETS_METRIC[name][0]
            assert img_dc == metric_dx, f"east/west mismatch at {name}"

    def test_image_vs_metric_flip_on_north_south_axis(self):
        # Image row sign is opposite of metric y sign (row grows south,
        # y grows north).
        for name in QUADRANT_NAMES:
            img_dr = QUADRANT_OFFSETS_IMAGE[name][0]
            metric_dy = QUADRANT_OFFSETS_METRIC[name][1]
            assert img_dr == -metric_dy, f"north/south not flipped at {name}"


# ---------------------------------------------------------------------------
# CameraModel
# ---------------------------------------------------------------------------


class TestCameraModel:
    def test_footprint_side_m_matches_formula(self):
        cam = CameraModel(altitude_m=100.0, fov_deg=60.0, resolution_m=1.0)
        expected = 2.0 * 100.0 * math.tan(math.radians(30.0))
        assert cam.footprint_side_m == pytest.approx(expected, rel=1e-12)

    def test_footprint_side_px_rounds_and_clamps(self):
        # side_m ≈ 2·100·tan(30°) ≈ 115.47 → 115 px at 1 m/px (round-half-to-even).
        cam = CameraModel(altitude_m=100.0, fov_deg=60.0, resolution_m=1.0)
        assert cam.footprint_side_px == 115

    def test_footprint_side_px_has_minimum_of_five(self):
        # Tiny FOV / altitude would round to 0 px; clamped to 5.
        cam = CameraModel(altitude_m=0.01, fov_deg=1.0, resolution_m=1.0)
        assert cam.footprint_side_px == 5

    def test_dem_defaults_give_expected_footprint(self):
        # Same defaults the drone-DEM example uses in its CLI help.
        cam = CameraModel(altitude_m=2000.0, fov_deg=100.0, resolution_m=90.0)
        expected_side_m = 2.0 * 2000.0 * math.tan(math.radians(50.0))
        assert cam.footprint_side_m == pytest.approx(expected_side_m, rel=1e-12)
        assert cam.footprint_side_px == int(round(expected_side_m / 90.0))


# ---------------------------------------------------------------------------
# CoverageRaster
# ---------------------------------------------------------------------------


class TestCoverageRaster:
    def test_shape_and_bbox_are_quantised_to_resolution(self):
        r = CoverageRaster(bbox=(0.0, 10.0, 0.0, 5.0), resolution_m=2.0)
        assert r.shape == (3, 5)  # ceil(5/2)=3 rows, ceil(10/2)=5 cols
        # Covered bbox may exceed the input by < resolution (here exactly one cell).
        assert r.bbox == (0.0, 10.0, 0.0, 6.0)

    def test_invalid_bbox_or_resolution_raises(self):
        with pytest.raises(ValueError):
            CoverageRaster(bbox=(5.0, 5.0, 0.0, 1.0), resolution_m=1.0)
        with pytest.raises(ValueError):
            CoverageRaster(bbox=(0.0, 1.0, 0.0, 1.0), resolution_m=0.0)

    def test_fresh_mask_is_fully_unseen(self):
        r = CoverageRaster(bbox=(-5.0, 5.0, -5.0, 5.0), resolution_m=0.5)
        assert r.visited_fraction == 0.0
        assert r.unseen_fraction_at(0.0, 0.0, 2.0) == 1.0

    def test_mark_square_then_query_matches_dem_semantics(self):
        # Paint a 4m square at (0,0); mean over that same region is 1 → unseen 0.
        r = CoverageRaster(bbox=(-5.0, 5.0, -5.0, 5.0), resolution_m=0.5)
        r.mark_square(0.0, 0.0, 4.0)
        assert r.unseen_fraction_at(0.0, 0.0, 4.0) == 0.0
        # A non-overlapping square stays fully unseen.
        assert r.unseen_fraction_at(4.0, 4.0, 1.0) == 1.0

    def test_partial_overlap_gives_partial_unseen(self):
        # Paint [0,4]×[0,4]; query [2,6]×[2,6] overlaps on [2,4]×[2,4].
        # In mask pixels (0.5 m each): painted cols/rows 0..7, query cols/rows 4..11.
        # Query region is 8×8=64 pixels; overlap is 4×4=16 → visited frac 0.25.
        r = CoverageRaster(bbox=(0.0, 8.0, 0.0, 8.0), resolution_m=0.5)
        r.mark_square(2.0, 2.0, 4.0)
        assert r.unseen_fraction_at(4.0, 4.0, 4.0) == pytest.approx(0.75)

    def test_out_of_bounds_target_is_fully_unseen(self):
        r = CoverageRaster(bbox=(0.0, 10.0, 0.0, 10.0), resolution_m=1.0)
        r.mark_square(5.0, 5.0, 10.0)  # paint everything
        # A target entirely outside the raster is treated as unseen by the
        # DEM-style ``mean`` on an empty slice.
        assert r.unseen_fraction_at(100.0, 100.0, 1.0) == 1.0

    def test_mark_square_outside_bbox_is_noop(self):
        r = CoverageRaster(bbox=(0.0, 10.0, 0.0, 10.0), resolution_m=1.0)
        r.mark_square(-100.0, -100.0, 2.0)
        assert r.visited_fraction == 0.0

    def test_extent_matches_bbox(self):
        r = CoverageRaster(bbox=(1.0, 5.0, 2.0, 6.0), resolution_m=1.0)
        assert r.extent() == r.bbox

    def test_mask_origin_is_south_west(self):
        # Painting at the SW corner should set mask[0, 0] = True.
        r = CoverageRaster(bbox=(0.0, 10.0, 0.0, 10.0), resolution_m=1.0)
        r.mark_square(0.5, 0.5, 1.0)
        assert r.mask[0, 0] == True
        # NE corner paint lands in the last row/col.
        r.mark_square(9.5, 9.5, 1.0)
        assert r.mask[-1, -1] == True
        # Nothing else should be painted between those two corners.
        middle = r.mask.copy()
        middle[0, 0] = False
        middle[-1, -1] = False
        assert not middle.any()
        _ = np  # silence lint; numpy only needed implicitly above
