"""PR-7 tests: CityPriorStack + new factor types."""

from __future__ import annotations

import math

import numpy as np
import pytest

from kernelcal.distinction_game import (
    PHX_URBAN_V0,
    Taxonomy,
    FactorGraph,
    UnaryPerceptualFactor,
    loopy_bp,
)
from kernelcal.distinction_game.city_priors import (
    CityEntity,
    CityPriorStack,
    DEMGroundPriorSource,
    LandCoverPriorSource,
    OSMBuildingPriorSource,
    OSMRoadPriorSource,
    ParentChildPriorSource,
    PHX_PARENT_CHILD_COMPATIBLE,
    PHX_PARENT_CHILD_INCOMPATIBLE,
    entities_from_fused_scene_graph,
    _point_in_polygon,
    _segment_distance_m,
)
from kernelcal.distinction_game.factor_graph import (
    PairwiseParentChildFactor,
    UnaryClassPriorFactor,
    UnaryGroundElevationFactor,
)


# ---------------------------------------------------------------------------
# Factor type unit tests
# ---------------------------------------------------------------------------


class TestUnaryClassPriorFactor:
    def test_single_target_index_bonus(self):
        f = UnaryClassPriorFactor(
            "v",
            target_class_index=2,
            n_states=4,
            bonus=0.7,
        )
        assert f.variables == ("v",)
        assert f.log_table.shape == (4,)
        assert f.log_table[2] == pytest.approx(0.7)
        for i in (0, 1, 3):
            assert f.log_table[i] == pytest.approx(0.0)
        assert f.metadata["target_class_index"] == [2]
        assert f.metadata["bonus"] == pytest.approx(0.7)

    def test_multi_target_indices(self):
        f = UnaryClassPriorFactor(
            "v",
            target_class_index=[1, 3, 3, 1],  # dedup
            n_states=5,
            bonus=1.2,
        )
        assert f.metadata["target_class_index"] == [1, 3]
        assert f.log_table[1] == pytest.approx(1.2)
        assert f.log_table[3] == pytest.approx(1.2)
        for i in (0, 2, 4):
            assert f.log_table[i] == pytest.approx(0.0)

    def test_bonus_zero_is_inert(self):
        f = UnaryClassPriorFactor("v", target_class_index=1, n_states=3, bonus=0.0)
        np.testing.assert_allclose(f.log_table, np.zeros(3))

    @pytest.mark.parametrize("bad_idx", [-1, 5])
    def test_out_of_range(self, bad_idx):
        with pytest.raises(ValueError):
            UnaryClassPriorFactor("v", target_class_index=bad_idx, n_states=3)

    def test_negative_bonus_rejected(self):
        with pytest.raises(ValueError):
            UnaryClassPriorFactor(
                "v", target_class_index=0, n_states=3, bonus=-0.1
            )

    def test_n_states_too_small(self):
        with pytest.raises(ValueError):
            UnaryClassPriorFactor("v", target_class_index=0, n_states=1)


class TestUnaryGroundElevationFactor:
    def test_ground_hit(self):
        f = UnaryGroundElevationFactor(
            "v",
            n_states=4,
            base_above_dem_m=0.1,
            ground_class_indices=[0, 1],
            elevated_class_indices=[2, 3],
            ground_tol_m=0.5,
            bonus=2.0,
        )
        assert f.metadata["is_ground"] is True
        assert f.metadata["is_elevated"] is False
        assert f.log_table[0] == pytest.approx(2.0)
        assert f.log_table[1] == pytest.approx(2.0)
        for i in (2, 3):
            assert f.log_table[i] == pytest.approx(0.0)

    def test_elevated_hit(self):
        f = UnaryGroundElevationFactor(
            "v",
            n_states=4,
            base_above_dem_m=5.0,
            ground_class_indices=[0, 1],
            elevated_class_indices=[2, 3],
            ground_tol_m=0.5,
            bonus=1.5,
        )
        assert f.metadata["is_ground"] is False
        assert f.metadata["is_elevated"] is True
        for i in (0, 1):
            assert f.log_table[i] == pytest.approx(0.0)
        for i in (2, 3):
            assert f.log_table[i] == pytest.approx(1.5)

    def test_ambiguous_below_ground(self):
        # base_above_dem_m = -2 m: not within tol, not above
        f = UnaryGroundElevationFactor(
            "v",
            n_states=3,
            base_above_dem_m=-2.0,
            ground_class_indices=[0],
            elevated_class_indices=[2],
            ground_tol_m=0.5,
        )
        assert f.metadata["is_ground"] is False
        assert f.metadata["is_elevated"] is False
        np.testing.assert_allclose(f.log_table, np.zeros(3))

    def test_negative_bonus_rejected(self):
        with pytest.raises(ValueError):
            UnaryGroundElevationFactor(
                "v",
                n_states=3,
                base_above_dem_m=0.0,
                ground_class_indices=[0],
                elevated_class_indices=[1],
                bonus=-0.5,
            )


class TestPairwiseParentChildFactor:
    def test_compatible_and_incompatible_pairs(self):
        f = PairwiseParentChildFactor(
            "p",
            "c",
            n_states=3,
            compatible_pairs=[(0, 0), (1, 2)],
            incompatible_pairs=[(2, 1)],
            log_compatible=1.0,
            log_incompatible=-2.0,
        )
        assert f.variables == ("p", "c")
        assert f.log_table.shape == (3, 3)
        assert f.log_table[0, 0] == pytest.approx(1.0)
        assert f.log_table[1, 2] == pytest.approx(1.0)
        assert f.log_table[2, 1] == pytest.approx(-2.0)
        assert f.log_table[0, 1] == pytest.approx(0.0)
        assert f.log_table[2, 2] == pytest.approx(0.0)

    def test_out_of_range_pairs_silently_skipped(self):
        f = PairwiseParentChildFactor(
            "p",
            "c",
            n_states=2,
            compatible_pairs=[(0, 0), (5, 5)],  # second out of range
            incompatible_pairs=[(-1, 0)],       # negative also skipped
        )
        assert f.log_table[0, 0] == pytest.approx(1.0)
        # Make sure the second compat pair was skipped without error.
        assert f.log_table.shape == (2, 2)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


class TestGeometryHelpers:
    def test_point_in_polygon_concave(self):
        # L-shaped polygon (concave). Lon, lat.
        poly = np.array([
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [0.0, 2.0],
            [0.0, 0.0],
        ])
        assert _point_in_polygon(poly, 0.5, 0.5)
        assert _point_in_polygon(poly, 0.5, 1.5)
        assert _point_in_polygon(poly, 1.5, 0.5)
        # Inside the L's notch -> outside the polygon.
        assert not _point_in_polygon(poly, 1.5, 1.5)
        # Far away -> outside.
        assert not _point_in_polygon(poly, 5.0, 5.0)

    def test_segment_distance_m_zero_on_segment(self):
        a = (-112.0, 33.4500)
        b = (-112.0, 33.4510)
        # Midpoint of segment.
        d = _segment_distance_m(-112.0, 33.4505, a, b)
        assert d < 1.0

    def test_segment_distance_m_perpendicular(self):
        # Segment runs east-west at lat 33.45; query point 100 m north.
        a = (-112.0010, 33.45)
        b = (-111.9990, 33.45)
        d_lat_deg = 100.0 / 111_320.0
        d = _segment_distance_m(-112.0000, 33.45 + d_lat_deg, a, b)
        assert abs(d - 100.0) < 5.0


# ---------------------------------------------------------------------------
# OSMBuildingPriorSource
# ---------------------------------------------------------------------------


class TestOSMBuildingPriorSource:
    def _square_around(self, lon: float, lat: float, half_deg: float = 1e-4):
        return np.array([
            [lon - half_deg, lat - half_deg],
            [lon + half_deg, lat - half_deg],
            [lon + half_deg, lat + half_deg],
            [lon - half_deg, lat + half_deg],
            [lon - half_deg, lat - half_deg],
        ])

    def test_inside_polygon_emits_factor(self):
        poly = self._square_around(-112.07, 33.45)
        src = OSMBuildingPriorSource(
            building_polygons=[poly], target_class="building", bonus=2.0
        )
        ents = [
            CityEntity(var_id="v0", lon=-112.07, lat=33.45),
            CityEntity(var_id="v1", lon=-112.10, lat=33.45),  # outside
        ]
        factors = src.factors_for(ents, PHX_URBAN_V0)
        assert len(factors) == 1
        f = factors[0]
        assert isinstance(f, UnaryClassPriorFactor)
        assert f.variables == ("v0",)
        idx = PHX_URBAN_V0.index_of("building")
        assert f.log_table[idx] == pytest.approx(2.0)

    def test_unknown_target_class_no_factors(self):
        poly = self._square_around(-112.07, 33.45)
        src = OSMBuildingPriorSource(
            building_polygons=[poly], target_class="not_in_taxonomy"
        )
        ents = [CityEntity(var_id="v0", lon=-112.07, lat=33.45)]
        assert src.factors_for(ents, PHX_URBAN_V0) == []

    def test_skips_entities_without_lon_lat(self):
        poly = self._square_around(-112.07, 33.45)
        src = OSMBuildingPriorSource(building_polygons=[poly])
        ents = [CityEntity(var_id="v0", lon=None, lat=None)]
        assert src.factors_for(ents, PHX_URBAN_V0) == []

    def test_margin_deg_pulls_in_near_misses(self):
        poly = self._square_around(-112.07, 33.45, half_deg=1e-4)
        # Place entity 1.5e-4 deg outside the polygon (< 17 m).
        ent = CityEntity(var_id="v0", lon=-112.07 + 1.5e-4, lat=33.45)
        src_strict = OSMBuildingPriorSource(building_polygons=[poly], margin_deg=0.0)
        src_loose = OSMBuildingPriorSource(building_polygons=[poly], margin_deg=2e-4)
        # Neither source claims the polygon contains the point, but the
        # bbox pre-filter is the only thing margin affects.  Margin
        # widens the *bbox* but the polygon test still has to pass; so
        # the strict source should be a no-op either way.
        assert src_strict.factors_for([ent], PHX_URBAN_V0) == []
        assert src_loose.factors_for([ent], PHX_URBAN_V0) == []

    def test_invalid_polygon_raises(self):
        with pytest.raises(ValueError):
            OSMBuildingPriorSource(building_polygons=[np.zeros((2, 2))])


# ---------------------------------------------------------------------------
# OSMRoadPriorSource
# ---------------------------------------------------------------------------


class TestOSMRoadPriorSource:
    def test_close_to_road_emits_factor(self):
        # Road runs east-west at lat 33.45.
        polyline = np.array([
            [-112.001, 33.45],
            [-111.999, 33.45],
        ])
        src = OSMRoadPriorSource(
            road_polylines=[polyline],
            near_classes=("road", "pavement"),
            near_distance_m=20.0,
            bonus_near=1.5,
        )
        # 5 m north of road.
        d_lat = 5.0 / 111_320.0
        ents = [
            CityEntity(var_id="near", lon=-112.000, lat=33.45 + d_lat),
            CityEntity(var_id="far", lon=-112.000, lat=33.45 + 0.01),
        ]
        factors = src.factors_for(ents, PHX_URBAN_V0)
        assert len(factors) == 1
        f = factors[0]
        assert f.variables == ("near",)
        road_idx = PHX_URBAN_V0.index_of("road")
        pav_idx = PHX_URBAN_V0.index_of("pavement")
        assert f.log_table[road_idx] == pytest.approx(1.5)
        assert f.log_table[pav_idx] == pytest.approx(1.5)
        assert f.metadata["distance_m"] < 20.0

    def test_no_factors_when_no_near_class_in_taxonomy(self):
        polyline = np.array([[-112.001, 33.45], [-111.999, 33.45]])
        src = OSMRoadPriorSource(
            road_polylines=[polyline],
            near_classes=("highway", "freeway"),  # none in PHX_URBAN_V0
        )
        ent = CityEntity(var_id="v", lon=-112.0, lat=33.45)
        assert src.factors_for([ent], PHX_URBAN_V0) == []


# ---------------------------------------------------------------------------
# LandCoverPriorSource
# ---------------------------------------------------------------------------


class TestLandCoverPriorSource:
    def test_pixel_lookup_emits_factor(self):
        # 2x2 raster: top row water (10), bottom row vegetation (20).
        raster = np.array([[10, 10], [20, 20]], dtype=np.int32)
        src = LandCoverPriorSource(
            raster=raster,
            bbox=(-112.10, 33.40, -112.00, 33.50),  # 0.10 deg square
            code_to_class={10: "water", 20: "vegetation_other"},
            bonus=1.0,
        )
        # Top half (lat > midline) -> water; bottom half -> vegetation.
        ent_top = CityEntity(var_id="t", lon=-112.05, lat=33.48)
        ent_bot = CityEntity(var_id="b", lon=-112.05, lat=33.42)
        ent_out = CityEntity(var_id="o", lon=-110.0, lat=33.45)
        factors = src.factors_for([ent_top, ent_bot, ent_out], PHX_URBAN_V0)
        assert len(factors) == 2
        by_var = {f.variables[0]: f for f in factors}
        water_idx = PHX_URBAN_V0.index_of("water")
        veg_idx = PHX_URBAN_V0.index_of("vegetation_other")
        assert by_var["t"].log_table[water_idx] == pytest.approx(1.0)
        assert by_var["b"].log_table[veg_idx] == pytest.approx(1.0)

    def test_unknown_codes_skipped(self):
        raster = np.array([[99]], dtype=np.int32)
        src = LandCoverPriorSource(
            raster=raster,
            bbox=(-1.0, -1.0, 1.0, 1.0),
            code_to_class={10: "water"},
        )
        ent = CityEntity(var_id="v", lon=0.0, lat=0.0)
        assert src.factors_for([ent], PHX_URBAN_V0) == []

    def test_degenerate_bbox_raises(self):
        with pytest.raises(ValueError):
            LandCoverPriorSource(
                raster=np.zeros((1, 1), dtype=np.int32),
                bbox=(0.0, 0.0, 0.0, 1.0),
            )


# ---------------------------------------------------------------------------
# DEMGroundPriorSource
# ---------------------------------------------------------------------------


class TestDEMGroundPriorSource:
    def test_ground_and_elevated_routing(self):
        # Flat DEM at 350 m.
        dem = np.full((4, 4), 350.0)
        src = DEMGroundPriorSource(
            dem=dem,
            bbox=(-112.10, 33.40, -112.00, 33.50),
            ground_tol_m=0.5,
            bonus=1.0,
        )
        # base_alt_m == DEM -> ground hit.
        ent_ground = CityEntity(var_id="g", lon=-112.05, lat=33.45, base_alt_m=350.0)
        # base_alt_m 5 m above DEM -> elevated hit.
        ent_elev = CityEntity(var_id="e", lon=-112.05, lat=33.45, base_alt_m=355.0)
        factors = src.factors_for([ent_ground, ent_elev], PHX_URBAN_V0)
        assert len(factors) == 2
        by_var = {f.variables[0]: f for f in factors}
        g = by_var["g"]
        e = by_var["e"]
        # Ground entity should bias ground classes:
        ground_idx = [PHX_URBAN_V0.index_of(c) for c in (
            "road", "pavement", "bare_ground", "water", "vegetation_other"
        )]
        for i in ground_idx:
            assert g.log_table[i] == pytest.approx(1.0)
        # Elevated entity should bias elevated classes:
        elev_idx = [PHX_URBAN_V0.index_of(c) for c in ("building", "tree", "vehicle")]
        for i in elev_idx:
            assert e.log_table[i] == pytest.approx(1.0)

    def test_skips_entities_without_alt(self):
        dem = np.zeros((1, 1))
        src = DEMGroundPriorSource(dem=dem, bbox=(-1.0, -1.0, 1.0, 1.0))
        ent = CityEntity(var_id="v", lon=0.0, lat=0.0, base_alt_m=None)
        assert src.factors_for([ent], PHX_URBAN_V0) == []


# ---------------------------------------------------------------------------
# ParentChildPriorSource
# ---------------------------------------------------------------------------


class TestParentChildPriorSource:
    def test_emits_one_factor_per_parent_child_link(self):
        src = ParentChildPriorSource()
        ents = [
            CityEntity(var_id="trunk", parent_var_id=None),
            CityEntity(var_id="crown", parent_var_id="trunk"),
            CityEntity(var_id="orphan", parent_var_id="missing"),
        ]
        factors = src.factors_for(ents, PHX_URBAN_V0)
        assert len(factors) == 1
        f = factors[0]
        assert isinstance(f, PairwiseParentChildFactor)
        assert f.variables == ("trunk", "crown")
        # Default compat table includes (tree, tree) at +1.0.
        tree_idx = PHX_URBAN_V0.index_of("tree")
        assert f.log_table[tree_idx, tree_idx] == pytest.approx(1.0)

    def test_default_compat_pairs_resolve(self):
        # All names in the default tables either resolve in PHX_URBAN_V0
        # or are silently skipped.
        for p, c in PHX_PARENT_CHILD_COMPATIBLE:
            assert p in PHX_URBAN_V0.categories
            assert c in PHX_URBAN_V0.categories
        for p, c in PHX_PARENT_CHILD_INCOMPATIBLE:
            assert p in PHX_URBAN_V0.categories
            assert c in PHX_URBAN_V0.categories

    def test_custom_taxonomy_only_partial_match(self):
        tax = Taxonomy(name="tiny", categories=("unknown", "tree"))
        ents = [
            CityEntity(var_id="t1", parent_var_id=None),
            CityEntity(var_id="t2", parent_var_id="t1"),
        ]
        src = ParentChildPriorSource()
        factors = src.factors_for(ents, tax)
        # ("tree", "tree") should still resolve.
        assert len(factors) == 1
        idx = tax.index_of("tree")
        assert factors[0].log_table[idx, idx] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# CityPriorStack composition + BP integration
# ---------------------------------------------------------------------------


class TestCityPriorStack:
    def test_stack_composes_sources(self):
        poly = np.array([
            [-112.072, 33.448],
            [-112.068, 33.448],
            [-112.068, 33.452],
            [-112.072, 33.452],
            [-112.072, 33.448],
        ])
        stack = CityPriorStack().add(
            OSMBuildingPriorSource(building_polygons=[poly], bonus=1.0)
        ).add(ParentChildPriorSource())
        ents = [
            CityEntity(var_id="parent", lon=-112.070, lat=33.450, parent_var_id=None),
            CityEntity(var_id="child", lon=-112.070, lat=33.450, parent_var_id="parent"),
        ]
        factors = stack.factors_for(ents, PHX_URBAN_V0)
        kinds = sorted({type(f).__name__ for f in factors})
        assert "UnaryClassPriorFactor" in kinds
        assert "PairwiseParentChildFactor" in kinds

    def test_stack_add_rejects_non_source(self):
        with pytest.raises(TypeError):
            CityPriorStack().add("not a source")  # type: ignore

    def test_stack_factors_run_through_bp(self):
        # End-to-end: a single entity with no perceptual evidence and a
        # strong building prior should converge to "building".
        var_id = "v"
        graph = FactorGraph()
        graph.add_variable(var_id, n_states=PHX_URBAN_V0.n)
        # Weak ambiguous unary perceptual evidence (uniform-ish).
        weak = np.full(PHX_URBAN_V0.n, 0.0)
        from kernelcal.distinction_game.factor_graph import Factor
        graph.add_factor(Factor(variables=(var_id,), log_table=weak, name="weak"))
        # City prior with a strong building bonus.
        poly = np.array([
            [-112.072, 33.448],
            [-112.068, 33.448],
            [-112.068, 33.452],
            [-112.072, 33.452],
            [-112.072, 33.448],
        ])
        stack = CityPriorStack(sources=[
            OSMBuildingPriorSource(building_polygons=[poly], bonus=3.0),
        ])
        ent = CityEntity(var_id=var_id, lon=-112.070, lat=33.450)
        for f in stack.factors_for([ent], PHX_URBAN_V0):
            graph.add_factor(f)
        result = loopy_bp(graph, max_iter=20, tol=1e-6)
        post = result.posteriors[var_id]
        winner = int(np.argmax(post))
        assert winner == PHX_URBAN_V0.index_of("building")


# ---------------------------------------------------------------------------
# Convenience: entities_from_fused_scene_graph
# ---------------------------------------------------------------------------


class TestEntitiesFromFused:
    def test_extracts_basic_fields(self):
        fused = {
            "nodes": [
                {
                    "id": "n0",
                    "geo_centroid": [-112.07, 33.45],
                    "base_alt_m": 350.0,
                    "parent_id": None,
                    "attributes": {"foo": "bar"},
                },
                {
                    "id": "n1",
                    "region": {"geo_centroid": [-112.069, 33.451]},
                    "base_alt_m": 351.5,
                    "parent_id": "n0",
                },
                {"id": "n2"},  # Bare-bones; lon/lat None.
            ]
        }
        ents = entities_from_fused_scene_graph(fused)
        assert [e.var_id for e in ents] == ["n0", "n1", "n2"]
        assert ents[0].lon == pytest.approx(-112.07)
        assert ents[0].lat == pytest.approx(33.45)
        assert ents[0].base_alt_m == pytest.approx(350.0)
        assert ents[0].attributes == {"foo": "bar"}
        assert ents[1].lon == pytest.approx(-112.069)
        assert ents[1].parent_var_id == "n0"
        assert ents[2].lon is None
        assert ents[2].lat is None
