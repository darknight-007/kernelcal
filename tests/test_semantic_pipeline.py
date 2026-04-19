"""Tests for kernelcal.semantic — pure-Python HITL semantic SLAM pipeline.

No GPU / torch / transformers dependencies: the whole pipeline is exercised
with :class:`StubSegmenter` backends so the tests run on CI.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from kernelcal.semantic import (
    ActiveQuerySampler,
    ClassRegistry,
    ClassSpec,
    EnsembleFrameResult,
    InstanceMask,
    MOTION_DYNAMIC,
    MOTION_STATIC,
    NoveltyReport,
    NoveltyWeights,
    QueryBudget,
    SegmenterEnsemble,
    STATUS_KNOWN,
    STATUS_UNCERTAIN,
    STATUS_UNKNOWN,
    StubSegmenter,
    filter_candidates,
    score_frame,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _fake_image(h: int = 128, w: int = 128) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


@pytest.fixture
def registry() -> ClassRegistry:
    return ClassRegistry.urban_default()


@pytest.fixture
def image() -> np.ndarray:
    return _fake_image()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_urban_default_contains_expected_classes(registry: ClassRegistry):
    assert {"tree", "house", "road", "car"}.issubset(set(registry.names))
    assert "other" in registry  # catch-all
    assert registry["car"].motion == MOTION_DYNAMIC
    assert registry["tree"].motion == MOTION_STATIC


def test_registry_prior_vector_is_normalised(registry: ClassRegistry):
    names, q = registry.prior_vector()
    assert len(names) == len(q)
    assert np.isclose(q.sum(), 1.0)
    assert np.all(q > 0)


def test_registry_grounding_prompt_skips_other(registry: ClassRegistry):
    prompt = registry.grounding_prompt()
    assert "tree" in prompt and "house" in prompt
    assert "other" not in prompt
    assert prompt.endswith(" .")


def test_registry_prototype_update_and_classification(registry: ClassRegistry):
    emb_a = np.array([1.0, 0.0, 0.0])
    emb_b = np.array([0.9, 0.1, 0.0])
    emb_c = np.array([0.0, 0.0, 1.0])
    registry.apply_response("tree", embedding=emb_a)
    registry.apply_response("tree", embedding=emb_b)
    registry.apply_response("car", embedding=emb_c)

    name, sim = registry.classify_by_prototype(np.array([1.0, 0.0, 0.0]))
    assert name == "tree"
    assert sim > 0.9


def test_registry_extension_via_label_response(registry: ClassRegistry):
    before = len(registry)
    spec = registry.apply_response(
        "streetlight",
        is_new_class=True,
        motion=MOTION_STATIC,
        aliases=["lamp post", "lampost"],
    )
    assert len(registry) == before + 1
    assert spec.is_user_seed is False
    assert "lamp post" in registry["streetlight"].aliases
    # New class shows up in grounding prompt too
    assert "streetlight" in registry.grounding_prompt()


def test_registry_rejects_unknown_without_is_new_class(registry: ClassRegistry):
    with pytest.raises(KeyError):
        registry.apply_response("gargoyle", is_new_class=False)


def test_registry_roundtrip(tmp_path: Path, registry: ClassRegistry):
    registry.apply_response("tree", embedding=np.array([1.0, 0.0]))
    p = tmp_path / "reg.json"
    registry.save(p)
    loaded = ClassRegistry.load(p)
    assert set(loaded.names) == set(registry.names)
    assert loaded["tree"].prototype is not None
    assert np.allclose(loaded["tree"].prototype, np.array([1.0, 0.0]))


# ---------------------------------------------------------------------------
# StubSegmenter / Ensemble
# ---------------------------------------------------------------------------


def test_stub_segmenter_returns_four_quadrants(image: np.ndarray):
    seg = StubSegmenter(name="maskrcnn")
    out = seg.segment(image)
    assert len(out) == 4
    assert {o.proposed_label for o in out} == {"tree", "house", "road", "car"}
    assert all(isinstance(o, InstanceMask) for o in out)
    assert all(o.area_px > 0 for o in out)


def test_ensemble_process_frame_classifies_known(
    registry: ClassRegistry, image: np.ndarray
):
    ensemble = SegmenterEnsemble(
        registry=registry,
        closed_set=StubSegmenter(name="maskrcnn"),
        panoptic=None,
        open_vocab=None,
        confident_score=0.7,
    )
    result = ensemble.process_frame(image)
    assert isinstance(result, EnsembleFrameResult)
    assert len(result.resolved) == 4
    labels = {r.final_label for r in result.resolved if r.status == STATUS_KNOWN}
    assert "tree" in labels and "road" in labels
    # Score 0.60 on car is below 0.7 → should be uncertain
    car = next(r for r in result.resolved if r.instance.proposed_label == "car")
    assert car.status in (STATUS_UNCERTAIN, STATUS_UNKNOWN)


def test_ensemble_hs_novelty_zero_on_repeat(
    registry: ClassRegistry, image: np.ndarray
):
    ensemble = SegmenterEnsemble(
        registry=registry,
        closed_set=StubSegmenter(name="maskrcnn"),
    )
    _ = ensemble.process_frame(image)
    second = ensemble.process_frame(image)
    assert second.hs_novelty_vs_prev == pytest.approx(0.0, abs=1e-6)


def test_ensemble_panoptic_merges_unclaimed(registry: ClassRegistry, image: np.ndarray):
    def _panoptic_instances(img, prompt):
        # One big panoptic mask that does NOT overlap any stub quadrant
        h, w = img.shape[:2]
        m = np.zeros((h, w), dtype=bool)
        m[h // 4 : h // 4 + 4, w // 4 : w // 4 + 4] = True  # tiny, unclaimed
        return [InstanceMask(
            mask=m,
            bbox=(w // 4, h // 4, w // 4 + 4, h // 4 + 4),
            score=0.9,
            proposed_label=None,  # class-agnostic
            embedding=np.zeros(16),
            source="sam",
        )]

    panoptic = StubSegmenter(
        name="sam",
        kind="panoptic",
        instances_fn=_panoptic_instances,
    )
    ensemble = SegmenterEnsemble(
        registry=registry,
        closed_set=StubSegmenter(name="maskrcnn"),
        panoptic=panoptic,
        panoptic_every=1,
    )
    result = ensemble.process_frame(image)
    sources = {r.instance.source for r in result.resolved}
    assert "maskrcnn" in sources
    assert "sam" in sources  # unclaimed mask made it through


# ---------------------------------------------------------------------------
# Novelty
# ---------------------------------------------------------------------------


def test_novelty_score_frame_produces_report_per_instance(
    registry: ClassRegistry, image: np.ndarray
):
    ensemble = SegmenterEnsemble(
        registry=registry,
        closed_set=StubSegmenter(name="maskrcnn"),
        confident_score=0.99,   # force everything into non-known states
    )
    result = ensemble.process_frame(image)
    reports = score_frame(result)
    assert len(reports) == len(result.resolved)
    assert all(0.0 <= r.score <= 1.0 for r in reports)
    # With everything non-known and some with low score, most should be >0
    assert any(r.score > 0.0 for r in reports)


def test_novelty_spectral_signal_lifts_score(
    registry: ClassRegistry, image: np.ndarray
):
    ensemble = SegmenterEnsemble(
        registry=registry,
        closed_set=StubSegmenter(name="maskrcnn"),
    )
    result = ensemble.process_frame(image)
    base = score_frame(result)
    hot = score_frame(
        result,
        spectral_context={
            "d_entropy": 2.0,
            "d_lambda1": 1.5,
            "d_beta1": 2.0,
            "D_t": 10.0,
            "delta_prime": 5.71,
        },
    )
    # Adding structural-change evidence should not *decrease* novelty scores
    assert all(h.score + 1e-9 >= b.score for h, b in zip(hot, base))
    # and should strictly lift at least one
    assert any(h.score > b.score + 1e-6 for h, b in zip(hot, base))


def test_filter_candidates_respects_thresholds(
    registry: ClassRegistry, image: np.ndarray
):
    ensemble = SegmenterEnsemble(
        registry=registry,
        closed_set=StubSegmenter(name="maskrcnn"),
    )
    result = ensemble.process_frame(image)
    reports = score_frame(result)
    # Absurdly strict thresholds filter everything out
    empty = filter_candidates(reports, min_score=2.0, min_area_px=1)
    assert empty == []
    lenient = filter_candidates(reports, min_score=-1.0, min_area_px=0)
    assert lenient == reports


# ---------------------------------------------------------------------------
# Active query sampler
# ---------------------------------------------------------------------------


def test_active_query_respects_budget(
    registry: ClassRegistry, image: np.ndarray
):
    ensemble = SegmenterEnsemble(
        registry=registry,
        closed_set=StubSegmenter(name="maskrcnn"),
        confident_score=0.99,  # everything goes into the unknown/uncertain bin
    )
    result = ensemble.process_frame(image)
    reports = score_frame(result)
    sampler = ActiveQuerySampler(
        budget=QueryBudget(
            max_queries=2,
            cost_budget_seconds=100.0,
            max_per_bucket=5,
        )
    )
    candidates = sampler.build_candidates(reports)
    plan = sampler.plan(candidates)
    assert len(plan.selected) <= 2
    assert plan.probabilities.shape == (len(candidates),)
    assert plan.cycle_index == 1
    # Diagnostics look sane
    assert 0.0 <= plan.diagnostics["cost_mean"]
    assert np.isfinite(plan.diagnostics["entropy_nats"])


def test_active_query_empty_pool_returns_empty_plan():
    sampler = ActiveQuerySampler()
    plan = sampler.plan([])
    assert plan.selected == []
    assert plan.probabilities.shape == (0,)


def test_active_query_spatial_cap(
    registry: ClassRegistry, image: np.ndarray
):
    ensemble = SegmenterEnsemble(
        registry=registry,
        closed_set=StubSegmenter(name="maskrcnn"),
        confident_score=0.99,
    )
    result = ensemble.process_frame(image)
    reports = score_frame(result)
    sampler = ActiveQuerySampler(
        budget=QueryBudget(
            max_queries=10,
            cost_budget_seconds=1000.0,
            max_per_bucket=1,
        )
    )
    # All candidates share the same bucket → cap of 1 must bind
    candidates = sampler.build_candidates(
        reports, spatial_bucket_fn=lambda r: "tile_0_0"
    )
    plan = sampler.plan(candidates)
    assert len(plan.selected) <= 1
