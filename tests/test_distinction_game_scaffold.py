"""
PR-1 scaffold tests for ``kernelcal.distinction_game``.

Exercises the public surface end-to-end on a synthetic 4-region tile
that touches every interesting behaviour of the pipeline:

1. **Region A (building)**: OSM, Grounding-DINO, SAM2 all agree on
   ``building`` while MR-rocks fires a high-confidence ``rock`` claim.
   The fused posterior must come out as ``c = building`` *despite*
   the MR-rocks vote, which is the central §3.4 / §0.7 disagreement
   the design doc was written to handle.

2. **Region B (real rock)**: GD ``rock`` + MR-rocks ``rock``, no OSM
   coverage. The fused posterior must come out as ``c = debris``.

3. **Region C (tree)**: OSM ``natural=tree`` + GD ``tree``. Fused
   posterior must be ``c = tree``.

4. **Region D (road)**: OSM ``highway`` + GD ``road``. Fused posterior
   must be ``c = road``.

In addition to the per-region category checks we verify:

- Claim-merging by IoU produces exactly 4 nodes (not one per claim).
- Each node carries provenance: the originating ``KernelClaim``
  instances are reachable from ``node.claims`` with their native
  labels intact.
- The :class:`SceneGraph` round-trips through ``to_dict`` cleanly.
- ``KernelMixFit`` lambdas sum to 1 and the ``method`` is
  ``"uniform"`` (the PR-1 baseline).
- The taxonomy ships exactly the 10 PHX_URBAN_V0 categories with
  the expected ordering.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pytest

from kernelcal.distinction_game import (
    KernelClaim,
    PHX_URBAN_V0,
    SceneGraph,
    Taxonomy,
    available_sources,
    build_scene_graph,
    default_q_s,
    default_q_s_table,
    fit_kernel_mix,
    uniform_lambdas,
)


# ---------------------------------------------------------------------------
# Tile fixtures
# ---------------------------------------------------------------------------

def _square(cx: float, cy: float, half: float) -> List[List[float]]:
    """Axis-aligned square in normalised image coords centred on ``(cx, cy)``."""
    return [
        [cx - half, cy - half],
        [cx + half, cy - half],
        [cx + half, cy + half],
        [cx - half, cy + half],
        [cx - half, cy - half],
    ]


def _building_a_claims() -> List[KernelClaim]:
    """Region A: building near (0.2, 0.2). Four overlapping claims."""
    poly = _square(0.2, 0.2, 0.10)
    return [
        KernelClaim.from_polygon("osm",            "building",      0.95, poly,
                                 attributes={"tag": "building=yes"}),
        KernelClaim.from_polygon("grounding_dino", "building",      0.80, poly,
                                 attributes={"phrase": "building"}),
        KernelClaim.from_polygon("sam2",           "<sam_segment>", 0.90, poly),
        KernelClaim.from_polygon("mr_rocks",       "rock",          0.65, poly,
                                 attributes={"model_id": "bishop_ntl_rgb_e0049"}),
    ]


def _rock_b_claims() -> List[KernelClaim]:
    """Region B: real rock near (0.55, 0.55). No OSM coverage."""
    poly = _square(0.55, 0.55, 0.05)
    return [
        KernelClaim.from_polygon("grounding_dino", "rock", 0.70, poly,
                                 attributes={"phrase": "rock"}),
        KernelClaim.from_polygon("mr_rocks",       "rock", 0.85, poly),
    ]


def _tree_c_claims() -> List[KernelClaim]:
    """Region C: tree near (0.78, 0.18)."""
    poly = _square(0.78, 0.18, 0.07)
    return [
        KernelClaim.from_polygon("osm",            "natural=tree", 0.95, poly,
                                 attributes={"tag": "natural=tree"}),
        KernelClaim.from_polygon("grounding_dino", "tree",         0.85, poly,
                                 attributes={"phrase": "tree"}),
    ]


def _road_d_claims() -> List[KernelClaim]:
    """Region D: road segment near (0.25, 0.72)."""
    poly = _square(0.25, 0.72, 0.13)
    return [
        KernelClaim.from_polygon("osm",            "highway", 0.95, poly,
                                 attributes={"tag": "highway=residential"}),
        KernelClaim.from_polygon("grounding_dino", "road",    0.80, poly,
                                 attributes={"phrase": "road"}),
    ]


@pytest.fixture
def four_region_tile() -> List[KernelClaim]:
    return [
        *_building_a_claims(),
        *_rock_b_claims(),
        *_tree_c_claims(),
        *_road_d_claims(),
    ]


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

class TestTaxonomy:
    def test_phx_urban_v0_has_10_categories(self):
        assert PHX_URBAN_V0.n == 10
        assert PHX_URBAN_V0.categories[0] == "unknown"
        assert "building" in PHX_URBAN_V0.categories
        assert "debris"   in PHX_URBAN_V0.categories

    def test_index_round_trip(self):
        for i, name in enumerate(PHX_URBAN_V0.categories):
            assert PHX_URBAN_V0.index_of(name) == i
            assert PHX_URBAN_V0.name_of(i) == name

    def test_super_class_lookup(self):
        assert PHX_URBAN_V0.super_class_of("building") == "structure"
        assert PHX_URBAN_V0.super_class_of("debris") == "natural"
        assert PHX_URBAN_V0.super_class_of("unknown") is None

    def test_indices_in_super_class(self):
        veg_idx = PHX_URBAN_V0.indices_in_super_class("vegetation")
        assert sorted(veg_idx) == sorted([
            PHX_URBAN_V0.index_of("tree"),
            PHX_URBAN_V0.index_of("vegetation_other"),
        ])

    def test_duplicate_categories_rejected(self):
        with pytest.raises(ValueError, match="duplicate categories"):
            Taxonomy(name="bad", categories=("a", "a"))


# ---------------------------------------------------------------------------
# Q_s priors
# ---------------------------------------------------------------------------

class TestQs:
    def test_every_source_has_a_default(self):
        sources = available_sources()
        assert set(sources) >= {
            "osm", "grounding_dino", "sam2", "grounded_sam2",
            "mr_rocks", "mr_house",
        }
        for sid in sources:
            qs = default_q_s(sid)
            assert qs.taxonomy is PHX_URBAN_V0
            np.testing.assert_allclose(qs.matrix.sum(axis=0), 1.0, atol=1e-6)

    def test_mr_rocks_splay_present(self):
        """Verify the §3.4 splay is encoded in the prior: P(rock | building)
        is meaningful (≥ 0.3) and P(rock | debris) is the maximum."""
        qs = default_q_s("mr_rocks")
        row = qs.likelihood_row("rock")
        debris_idx = PHX_URBAN_V0.index_of("debris")
        building_idx = PHX_URBAN_V0.index_of("building")
        assert row[debris_idx]   == row.max()
        assert row[building_idx] >= 0.3
        assert row[building_idx] <  row[debris_idx]

    def test_sam2_is_flat_in_log_likelihood(self):
        """SAM2 must contribute zero log-evidence: every category has
        the same fire rate so log Q_SAM(<sam_segment> | c) is constant."""
        qs = default_q_s("sam2")
        row = qs.likelihood_row("<sam_segment>")
        assert np.allclose(row, row[0])

    def test_mr_house_peaks_on_building(self):
        """MR-house's fire signal must peak on c = building. The
        eureka damage classes are collapsed to a single 'house'
        fire signal in PR-1; per-claim damage lives in
        ``KernelClaim.attributes`` instead of in ``Q_s``."""
        qs = default_q_s("mr_house")
        bldg = PHX_URBAN_V0.index_of("building")
        row = qs.likelihood_row("house")
        assert row.argmax() == bldg
        assert row[bldg] >= 0.5

    def test_validator_rejects_non_stochastic(self):
        from kernelcal.distinction_game.q_s import ConfusionMatrix
        bad = np.array([[0.5, 0.5], [0.6, 0.6]])
        # Make a 2-category synthetic taxonomy
        tx = Taxonomy(name="t", categories=("a", "b"))
        with pytest.raises(ValueError, match="columns do not sum to 1"):
            ConfusionMatrix(
                source_id="bad",
                taxonomy=tx,
                native_labels=("y0", "y1"),
                matrix=bad,
            )


# ---------------------------------------------------------------------------
# Kernel mix
# ---------------------------------------------------------------------------

class TestKernelMix:
    def test_uniform_lambdas_normalise(self):
        fit = uniform_lambdas(["osm", "grounding_dino", "mr_rocks"], PHX_URBAN_V0)
        assert fit.method == "uniform"
        np.testing.assert_allclose(fit.lambdas.sum(), 1.0)
        assert fit.lambda_for("osm") == pytest.approx(1 / 3)

    def test_fit_kernel_mix_returns_uniform_in_pr1(self):
        q_s_table = default_q_s_table(["osm", "mr_rocks"])
        fit = fit_kernel_mix(["osm", "mr_rocks"], q_s_table=q_s_table)
        assert fit.method == "uniform"
        assert fit.converged is True
        np.testing.assert_allclose(fit.lambdas, [0.5, 0.5])

    def test_fit_rejects_unknown_source(self):
        q_s_table = default_q_s_table(["osm"])
        with pytest.raises(KeyError, match="missing matrices"):
            fit_kernel_mix(["osm", "future_kernel"], q_s_table=q_s_table)


# ---------------------------------------------------------------------------
# build_scene_graph (the integration test)
# ---------------------------------------------------------------------------

class TestSceneGraphBuild:
    def test_four_regions_yield_four_nodes(self, four_region_tile):
        sg = build_scene_graph(
            four_region_tile,
            taxonomy=PHX_URBAN_V0,
            q_s_table=default_q_s_table(),
        )
        assert sg.n_nodes == 4

    def test_each_node_carries_provenance(self, four_region_tile):
        sg = build_scene_graph(
            four_region_tile,
            taxonomy=PHX_URBAN_V0,
            q_s_table=default_q_s_table(),
        )
        for node in sg.nodes:
            assert len(node.claims) >= 1
            for claim in node.claims:
                assert claim.source_id
                assert claim.native_label

    def test_building_wins_over_rocks_kernel(self, four_region_tile):
        """Region A — the central §3.4 disagreement test. OSM + GD + SAM
        say building, MR-rocks says 'rock' with high confidence; the
        fused posterior must be ``building`` regardless."""
        sg = build_scene_graph(
            four_region_tile,
            taxonomy=PHX_URBAN_V0,
            q_s_table=default_q_s_table(),
        )
        node_a = next(n for n in sg.nodes if "osm" in n.sources and "mr_rocks" in n.sources)
        assert node_a.category == "building"
        # Posterior should be peaked above the uniform floor (1/10) by a
        # comfortable margin, even with MR-rocks pulling some mass into
        # debris / bare_ground.
        assert node_a.score > 0.25
        # And building should beat debris by at least ~2x (the
        # 'distinction-kernel reframe' working — see §3.0 of the doc).
        debris_idx = PHX_URBAN_V0.index_of("debris")
        assert node_a.category_posterior[debris_idx] < 0.5 * node_a.score

    def test_real_rock_resolves_to_debris(self, four_region_tile):
        """Region B — only MR-rocks + GD-rock fired, no OSM coverage."""
        sg = build_scene_graph(
            four_region_tile,
            taxonomy=PHX_URBAN_V0,
            q_s_table=default_q_s_table(),
        )
        node_b = next(
            n for n in sg.nodes
            if set(n.sources) == {"grounding_dino", "mr_rocks"}
        )
        assert node_b.category == "debris"

    def test_tree_resolves_to_tree(self, four_region_tile):
        sg = build_scene_graph(
            four_region_tile,
            taxonomy=PHX_URBAN_V0,
            q_s_table=default_q_s_table(),
        )
        tree_nodes = [n for n in sg.nodes if n.category == "tree"]
        assert len(tree_nodes) == 1
        node_c = tree_nodes[0]
        assert "osm" in node_c.sources
        assert any(c.native_label == "natural=tree" for c in node_c.claims)

    def test_road_resolves_to_road(self, four_region_tile):
        sg = build_scene_graph(
            four_region_tile,
            taxonomy=PHX_URBAN_V0,
            q_s_table=default_q_s_table(),
        )
        road_nodes = [n for n in sg.nodes if n.category == "road"]
        assert len(road_nodes) == 1
        node_d = road_nodes[0]
        assert any(c.native_label == "highway" for c in node_d.claims)

    def test_to_dict_round_trips_cleanly(self, four_region_tile):
        sg = build_scene_graph(
            four_region_tile,
            taxonomy=PHX_URBAN_V0,
            q_s_table=default_q_s_table(),
        )
        d = sg.to_dict()
        # Schema sanity
        assert d["schema_version"]
        assert d["taxonomy"]["name"] == "phx_urban_v0"
        assert len(d["nodes"]) == sg.n_nodes
        assert "category_histogram" in d
        # Provenance round-trip
        for node_dict in d["nodes"]:
            assert "claims" in node_dict
            assert all("source_id" in c for c in node_dict["claims"])

    def test_kernels_queried_includes_silent_sources(self, four_region_tile):
        """Allow the orchestrator to declare that it queried kernels
        which produced no claims (e.g. mr_house on a tile with no
        houses) — those should show up in ``kernels_queried`` even
        though no node has them as a source."""
        sg = build_scene_graph(
            four_region_tile,
            taxonomy=PHX_URBAN_V0,
            q_s_table=default_q_s_table(),
            kernels_queried=["osm", "grounding_dino", "sam2",
                             "mr_rocks", "mr_house"],
        )
        assert "mr_house" in sg.kernels_queried
        assert all("mr_house" not in n.sources for n in sg.nodes)

    def test_min_score_drops_low_confidence_claims(self):
        poly = _square(0.5, 0.5, 0.1)
        claims = [
            KernelClaim.from_polygon("grounding_dino", "rock", 0.10, poly),
            KernelClaim.from_polygon("mr_rocks",       "rock", 0.85, poly),
        ]
        sg = build_scene_graph(
            claims,
            taxonomy=PHX_URBAN_V0,
            q_s_table=default_q_s_table(["mr_rocks", "grounding_dino"]),
            min_score=0.5,
        )
        assert sg.n_nodes == 1
        # The low-confidence GD claim should have been dropped.
        node = sg.nodes[0]
        assert all(c.source_id == "mr_rocks" for c in node.claims)

    def test_empty_input_yields_empty_graph(self):
        sg = build_scene_graph(
            [],
            taxonomy=PHX_URBAN_V0,
            q_s_table=default_q_s_table(),
        )
        assert sg.n_nodes == 0
        assert sg.n_edges == 0

    def test_adjacency_edges_link_nearby_nodes(self):
        """Two nodes with overlapping bbox centroids should be linked
        by an ``"adjacent"`` edge; far-apart ones should not."""
        nearby_a = _square(0.20, 0.20, 0.05)
        nearby_b = _square(0.23, 0.23, 0.05)  # within proximity
        far_c    = _square(0.80, 0.80, 0.05)

        # IoU between nearby_a and nearby_b is just barely below
        # the default 0.5 IoU merge threshold, so they stay distinct.
        claims = [
            KernelClaim.from_polygon("osm",            "building",       0.9, nearby_a),
            KernelClaim.from_polygon("grounding_dino", "building",       0.9, nearby_b),
            KernelClaim.from_polygon("osm",            "natural=tree",   0.9, far_c),
        ]
        sg = build_scene_graph(
            claims,
            taxonomy=PHX_URBAN_V0,
            q_s_table=default_q_s_table(["osm", "grounding_dino"]),
            iou_threshold=0.99,  # force them to NOT merge
        )
        assert sg.n_nodes == 3
        # The two nearby nodes should be linked; the far one should
        # contribute no edges.
        assert sg.n_edges == 1


# ---------------------------------------------------------------------------
# Smoke test for the §3.0 "distinction-kernel" reframe
# ---------------------------------------------------------------------------

def test_mr_rocks_alone_does_not_call_a_roof_a_rock():
    """Sanity check on the §3.0 reframe: when MR-rocks is the *only*
    source, its splayed prior places non-trivial mass on c=building
    (not just c=debris). The single-source posterior should reflect
    the splay rather than collapsing to the argmax of P(rock | c)
    only.

    In particular, P(c = building | MR-rocks fired with score 1.0)
    should be ≥ 10% — that's the §3.4 + Figs. 0.7 evidence the
    framework needs to be aware of."""
    poly = _square(0.5, 0.5, 0.1)
    claim = KernelClaim.from_polygon("mr_rocks", "rock", 1.0, poly)
    sg = build_scene_graph(
        [claim],
        taxonomy=PHX_URBAN_V0,
        q_s_table=default_q_s_table(["mr_rocks"]),
    )
    assert sg.n_nodes == 1
    posterior = sg.nodes[0].category_posterior
    bldg = PHX_URBAN_V0.index_of("building")
    debris = PHX_URBAN_V0.index_of("debris")
    assert posterior[debris]   == posterior.max()
    assert posterior[bldg]     >= 0.10
