from __future__ import annotations

import numpy as np
import pytest

from kernelcal.distinction_game import (
    Taxonomy,
    collapse_scene_graphs,
    data_associate,
    temporal_links,
)
from kernelcal.distinction_game.q_s import ConfusionMatrix


@pytest.fixture
def tiny_taxonomy() -> Taxonomy:
    return Taxonomy(name="tiny", categories=("unknown", "building", "road"))


@pytest.fixture
def q_table(tiny_taxonomy):
    return {
        "src": ConfusionMatrix(
            source_id="src",
            taxonomy=tiny_taxonomy,
            native_labels=("building", "road"),
            matrix=np.array([
                [0.2, 0.9, 0.1],
                [0.8, 0.1, 0.9],
            ]),
        )
    }


def _node(node_id, label, bbox, *, score=1.0, osm_id=None, source_id="src"):
    attrs = {"osm_id": osm_id} if osm_id else {}
    geo_poly = [
        [bbox[0], bbox[1]],
        [bbox[2], bbox[1]],
        [bbox[2], bbox[3]],
        [bbox[0], bbox[3]],
        [bbox[0], bbox[1]],
    ]
    return {
        "id": node_id,
        "category_index": 1 if label == "building" else 2,
        "category": label,
        "score": score,
        "region": {
            "bbox": list(bbox),
            "geo_polygon": geo_poly,
            "area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
        },
        "claims": [{
            "id": f"c-{node_id}",
            "source_id": source_id,
            "native_label": label,
            "score": score,
            "polygon": list(geo_poly),
            "bbox": list(bbox),
            "geo_polygon": geo_poly,
            "area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
            "attributes": attrs,
        }],
    }


def test_data_associate_clusters_by_iou():
    sg1 = {"session_id": "a", "nodes": [_node("n1", "building", (0, 0, 0.001, 0.001))]}
    sg2 = {"session_id": "b", "nodes": [_node("n2", "building", (0.00005, 0.00005, 0.00105, 0.00105))]}

    clusters = data_associate([sg1, sg2], iou_thresh=0.5)

    assert len(clusters) == 1
    assert clusters[0] == [(0, "n1"), (1, "n2")]


def test_data_associate_clusters_by_osm_id_even_without_overlap():
    sg1 = {"session_id": "a", "nodes": [_node("n1", "building", (0, 0, 0.001, 0.001), osm_id="42")]}
    sg2 = {"session_id": "b", "nodes": [_node("n2", "building", (10, 10, 10.001, 10.001), osm_id="42")]}

    clusters = data_associate([sg1, sg2], iou_thresh=0.9, osm_id_match=True)

    assert len(clusters) == 1


def test_temporal_links_connect_consecutive_observations():
    sg1 = {"session_id": "a", "nodes": [_node("n1", "building", (0, 0, 0.001, 0.001))]}
    sg2 = {"session_id": "b", "nodes": [_node("n2", "building", (0.00005, 0.00005, 0.00105, 0.00105))]}
    sg3 = {"session_id": "c", "nodes": [_node("n3", "building", (0.00006, 0.00006, 0.00106, 0.00106))]}

    links = temporal_links([sg1, sg2, sg3])

    assert links == [((0, "n1"), (1, "n2")), ((1, "n2"), (2, "n3"))]


def test_collapse_scene_graphs_merges_shared_entity(q_table, tiny_taxonomy):
    sg1 = {
        "session_id": "a",
        "nodes": [
            _node("b1", "building", (0, 0, 0.001, 0.001), score=0.9),
            _node("r1", "road", (0.01, 0, 0.011, 0.001), score=0.9),
        ],
        "edges": [{"source": "b1", "target": "r1", "relation": "adjacent", "weight": 1.0}],
    }
    sg2 = {
        "session_id": "b",
        "nodes": [_node("b2", "building", (0.00002, 0.00002, 0.00102, 0.00102), score=0.8)],
        "edges": [],
    }

    fused = collapse_scene_graphs(
        [sg1, sg2],
        q_s_table=q_table,
        lambdas={"src": 1.0},
        taxonomy=tiny_taxonomy,
        beta_spatial=0.0,
        iou_thresh=0.5,
        bp_damping=0.0,
    )
    payload = fused.to_dict()

    assert payload["provenance"]["n_input_nodes"] == 3
    assert payload["provenance"]["n_clusters"] == 2
    assert len(payload["nodes"]) == 2
    building = max(payload["nodes"], key=lambda n: len(n["merged_from"]))
    assert building["category"] == "building"
    assert len(building["merged_from"]) == 2
    assert payload["bp_diagnostics"]["n_variables"] == 2
    assert payload["provenance"]["n_temporal_links"] == 1
    assert payload["fusion_metadata"]["persistence_alpha"] == 0.95


def test_collapse_three_scene_graphs_sharpens_shared_entity(q_table, tiny_taxonomy):
    sgs = [
        {
            "session_id": f"sg-{i}",
            "nodes": [_node(
                f"b{i}",
                "building",
                (0.00001 * i, 0.00001 * i, 0.001 + 0.00001 * i, 0.001 + 0.00001 * i),
                score=1.0,
            )],
            "edges": [],
        }
        for i in range(3)
    ]

    single = collapse_scene_graphs(
        [sgs[0]],
        q_s_table=q_table,
        lambdas={"src": 1.0},
        taxonomy=tiny_taxonomy,
        beta_spatial=0.0,
        bp_damping=0.0,
    ).to_dict()["nodes"][0]
    fused = collapse_scene_graphs(
        sgs,
        q_s_table=q_table,
        lambdas={"src": 1.0},
        taxonomy=tiny_taxonomy,
        beta_spatial=0.0,
        bp_damping=0.0,
    ).to_dict()

    assert fused["provenance"]["n_input_nodes"] == 3
    assert fused["provenance"]["n_clusters"] == 1
    node = fused["nodes"][0]
    assert len(node["merged_from"]) == 3
    assert node["category"] == "building"
    assert node["category_posterior"][1] > single["category_posterior"][1]


def test_collapse_scene_graphs_preserves_provenance(q_table, tiny_taxonomy):
    sg = {"session_id": "sg-a", "nodes": [_node("n1", "road", (0, 0, 1, 1))], "edges": []}

    fused = collapse_scene_graphs(
        [sg],
        q_s_table=q_table,
        lambdas={"src": 1.0},
        taxonomy=tiny_taxonomy,
        bp_damping=0.0,
    ).to_dict()

    node = fused["nodes"][0]
    assert node["merged_from"] == [{
        "scene_graph_index": 0,
        "scene_graph_id": "sg-a",
        "node_id": "n1",
    }]
    assert node["category"] == "road"
    assert fused["fusion_metadata"]["method"] == "factor_graph_bp"


def test_data_associate_zero_area_points_across_grid_cell_boundary():
    """Two zero-area bboxes 1.4e-5 deg apart land in different cells of
    the geographic grid (cell_size = 1e-3) but must still associate
    when they are well within ``centroid_eps``. This guards against a
    regression where ``_grid_candidate_pairs`` only indexed each bbox
    in its single cell and never compared boundary-straddling points.
    """
    sg1 = {"session_id": "a", "nodes": [_node("n1", "building", (0.0009, 0.0009, 0.0009, 0.0009))]}
    sg2 = {"session_id": "b", "nodes": [_node("n2", "building", (0.001, 0.001, 0.001, 0.001))]}

    clusters = data_associate(
        [sg1, sg2], iou_thresh=0.99, osm_id_match=False, centroid_eps=0.0005,
    )

    assert len(clusters) == 1
    assert clusters[0] == [(0, "n1"), (1, "n2")]


def test_data_associate_cell_pad_does_not_fuse_distant_points():
    """Sanity check the new pad: two zero-area points well outside
    ``centroid_eps`` must still resolve to two distinct clusters even
    though the pad lets them share candidate cells.
    """
    sg1 = {"session_id": "a", "nodes": [_node("n1", "building", (0.0, 0.0, 0.0, 0.0))]}
    sg2 = {"session_id": "b", "nodes": [_node("n2", "building", (0.01, 0.01, 0.01, 0.01))]}

    clusters = data_associate(
        [sg1, sg2], iou_thresh=0.99, osm_id_match=False, centroid_eps=0.0001,
    )

    assert len(clusters) == 2


def test_collapse_anchor_only_claim_contributes_to_posterior(tiny_taxonomy):
    """Anchor sources whose Q_s is provided but whose λ was not fitted
    must still influence the fused posterior. The deepgis-xr glue
    backfills λ=1.0 for them; here we simulate the same condition by
    passing a Q_s for ``osm`` and a λ map that contains ``osm``: the
    fused category must follow the OSM evidence.
    """
    q_table_with_osm = {
        "osm": ConfusionMatrix(
            source_id="osm",
            taxonomy=tiny_taxonomy,
            native_labels=("building", "road"),
            matrix=np.array([
                [0.05, 0.95, 0.05],
                [0.95, 0.05, 0.95],
            ]),
        ),
    }
    sg = {
        "session_id": "anchor-only",
        "nodes": [_node("n1", "building", (0.0, 0.0, 0.001, 0.001), source_id="osm")],
    }
    fused = collapse_scene_graphs(
        [sg],
        q_s_table=q_table_with_osm,
        lambdas={"osm": 1.0},
        taxonomy=tiny_taxonomy,
        bp_damping=0.0,
    ).to_dict()

    node = fused["nodes"][0]
    assert node["category"] == "building"
    assert node["category_posterior"][1] > node["category_posterior"][2]
    assert fused["bp_diagnostics"]["n_unknown_source_claims"] == 0


def test_collapse_unknown_source_claim_is_recorded_in_diagnostics(q_table, tiny_taxonomy):
    """Claims from a source the fit never knew about should be tagged
    in ``bp_diagnostics`` so the operator can re-fit before trusting
    the fused posterior.
    """
    sg = {
        "session_id": "ghost-source",
        "nodes": [
            _node("n1", "building", (0.0, 0.0, 0.001, 0.001), source_id="src"),
            _node("n2", "road", (0.01, 0.0, 0.011, 0.001), source_id="ghost-kernel"),
        ],
    }
    fused = collapse_scene_graphs(
        [sg],
        q_s_table=q_table,
        lambdas={"src": 1.0},
        taxonomy=tiny_taxonomy,
        bp_damping=0.0,
    ).to_dict()

    diag = fused["bp_diagnostics"]
    assert diag["n_unknown_source_claims"] == 1
    assert diag["unknown_sources"] == ["ghost-kernel"]


def test_collapse_preserves_explicit_zero_edge_weight(q_table, tiny_taxonomy):
    """An edge with explicit ``weight=0.0`` must not be promoted to a
    spatial factor with effective weight 1.0. Regression for
    ``edge.get('weight', 1.0) or 1.0``.
    """
    sg = {
        "session_id": "zero-weight",
        "nodes": [
            _node("a", "building", (0.0, 0.0, 0.001, 0.001), score=0.9),
            _node("b", "road", (0.01, 0.0, 0.011, 0.001), score=0.9),
        ],
        "edges": [{"source": "a", "target": "b", "relation": "adjacent", "weight": 0.0}],
    }
    fused = collapse_scene_graphs(
        [sg],
        q_s_table=q_table,
        lambdas={"src": 1.0},
        taxonomy=tiny_taxonomy,
        beta_spatial=10.0,
        bp_damping=0.0,
    ).to_dict()

    assert len(fused["edges"]) == 1
    edge = fused["edges"][0]
    assert edge["weight"] == 0.0
    merged = edge["attributes"]["merged_from"]
    assert merged[0]["weight"] == 0.0


def test_collapse_accumulates_merged_from_for_duplicate_edges(q_table, tiny_taxonomy):
    """When two scene graphs both contain the same cluster pair, the
    fused edge's ``merged_from`` provenance must record both source
    rows, not only the first one encountered.
    """
    sg1 = {
        "session_id": "sg-1",
        "nodes": [
            _node("a1", "building", (0.0, 0.0, 0.001, 0.001), score=0.9),
            _node("b1", "road", (0.0008, 0.0, 0.0018, 0.001), score=0.9),
        ],
        "edges": [{"source": "a1", "target": "b1", "relation": "adjacent", "weight": 0.5}],
    }
    sg2 = {
        "session_id": "sg-2",
        "nodes": [
            _node("a2", "building", (0.00002, 0.00002, 0.00102, 0.00102), score=0.9),
            _node("b2", "road", (0.00082, 0.00002, 0.00182, 0.00102), score=0.9),
        ],
        "edges": [{"source": "a2", "target": "b2", "relation": "adjacent", "weight": 0.8}],
    }
    fused = collapse_scene_graphs(
        [sg1, sg2],
        q_s_table=q_table,
        lambdas={"src": 1.0},
        taxonomy=tiny_taxonomy,
        beta_spatial=0.0,
        iou_thresh=0.5,
        bp_damping=0.0,
    ).to_dict()

    assert len(fused["edges"]) == 1
    edge = fused["edges"][0]
    merged = edge["attributes"]["merged_from"]
    assert len(merged) == 2
    sg_idxs = sorted(m["scene_graph_index"] for m in merged)
    assert sg_idxs == [0, 1]
    weights = sorted(m["weight"] for m in merged)
    assert weights == [0.5, 0.8]
    assert edge["weight"] == 0.8
