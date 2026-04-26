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


def _node(node_id, label, bbox, *, score=1.0, osm_id=None):
    attrs = {"osm_id": osm_id} if osm_id else {}
    geo_poly = [
        [bbox[0], bbox[1]],
        [bbox[2], bbox[1]],
        [bbox[2], bbox[3]],
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
            "source_id": "src",
            "native_label": label,
            "score": score,
            "polygon": [
                [bbox[0], bbox[1]],
                [bbox[2], bbox[1]],
                [bbox[2], bbox[3]],
                [bbox[0], bbox[1]],
            ],
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
