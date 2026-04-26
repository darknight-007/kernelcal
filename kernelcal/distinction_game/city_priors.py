"""
PR-7 -- city-scale priors for the SceneGraph fusion factor graph.

Motivation
----------
4D semantic SLAM at city scale relies on more than per-frame perceptual
claims: there's a vast amount of *external* evidence per tile -- OSM
building footprints, road centerlines, land-cover rasters, DEM tiles,
3D building extrusions from Microsoft / Google -- that should be folded
in as priors over each fused entity's class distribution.

This module turns that external evidence into the categorical
:class:`~kernelcal.distinction_game.factor_graph.Factor` types defined
in :mod:`kernelcal.distinction_game.factor_graph` (see PR-7's new
``UnaryClassPriorFactor`` / ``UnaryGroundElevationFactor`` /
``PairwiseParentChildFactor``).

The :class:`CityPriorStack` is a plug-in registry: each
:class:`CityPriorSource` implements ``factors_for(entities, taxonomy)``
and the stack composes them.  Concrete sources shipped here are
deliberately dependency-light:

* :class:`OSMBuildingPriorSource`, :class:`OSMRoadPriorSource` --
  point-in-polygon / distance-to-polyline tests against pre-fetched
  OSM features.  ``kernelcal.urban.city_graph`` is used opportunistically
  to populate the source from a bbox when ``osmnx`` is installed; tests
  always go through the in-memory constructor.
* :class:`LandCoverPriorSource` -- raster lookup against a 2D NumPy
  array indexed by lat/lon (callers convert their tile to this form).
* :class:`DEMGroundPriorSource` -- raster lookup giving DEM elevation;
  emits :class:`UnaryGroundElevationFactor`.
* :class:`ParentChildPriorSource` -- uses each entity's ``parent_id``
  to emit :class:`PairwiseParentChildFactor`.

Sources operate on a small :class:`CityEntity` view that the caller
extracts from a :class:`~kernelcal.distinction_game.collapse.FusedSceneGraph`
*or* a list of :class:`~kernelcal.distinction_game.geometry.Superquadric`
objects (the rover side).  Both flows reduce to the same dataclass.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from .factor_graph import (
    Factor,
    PairwiseParentChildFactor,
    UnaryClassPriorFactor,
    UnaryGroundElevationFactor,
)
from .taxonomy import Taxonomy


# ---------------------------------------------------------------------------
# Entity view
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CityEntity:
    """Lightweight per-entity view consumed by :class:`CityPriorStack`.

    Attributes
    ----------
    var_id
        The categorical variable id used in the factor graph.  Sources
        emit factors keyed by this id.
    lon, lat
        WGS84 geographic centroid in decimal degrees.  ``None`` for
        entities that have no city-frame location yet (sources skip
        them).
    base_alt_m
        Ellipsoidal altitude (m) of the entity's *base* in WGS84.  Used
        by :class:`DEMGroundPriorSource` to compute base-above-DEM.
        ``None`` skips the elevation factor for this entity.
    parent_var_id
        Variable id of the parent entity, if any (e.g. tree-trunk
        carrying a crown).  Used by :class:`ParentChildPriorSource`.
    attributes
        Free-form per-entity metadata propagated from the fused
        SceneGraph (kept in factor metadata for traceability; sources
        may also use it for class hints, e.g. ``attributes['osm_tags']``).
    """

    var_id: Hashable
    lon: Optional[float] = None
    lat: Optional[float] = None
    base_alt_m: Optional[float] = None
    parent_var_id: Optional[Hashable] = None
    attributes: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Source ABC + Stack
# ---------------------------------------------------------------------------


class CityPriorSource(ABC):
    """Abstract base for plug-in city-prior sources.

    Subclasses implement :meth:`factors_for` which returns a list of
    factors derived from external evidence for the given entities.
    Implementations should be **idempotent** and **side-effect-free**
    so the same stack can be reused across many tiles.
    """

    name: str = "city_prior_source"

    @abstractmethod
    def factors_for(
        self,
        entities: Sequence[CityEntity],
        taxonomy: Taxonomy,
    ) -> List[Factor]:
        """Return a list of factors keyed by entity ``var_id``."""
        raise NotImplementedError


@dataclass
class CityPriorStack:
    """Compose multiple :class:`CityPriorSource` instances.

    Examples
    --------
    >>> from kernelcal.distinction_game import (
    ...     CityPriorStack, OSMBuildingPriorSource, ParentChildPriorSource,
    ...     PHX_URBAN_V0,
    ... )
    >>> stack = CityPriorStack(sources=[
    ...     OSMBuildingPriorSource(building_polygons=[...]),
    ...     ParentChildPriorSource(),
    ... ])
    >>> factors = stack.factors_for(entities, PHX_URBAN_V0)
    """

    sources: List[CityPriorSource] = field(default_factory=list)

    def add(self, source: CityPriorSource) -> "CityPriorStack":
        if not isinstance(source, CityPriorSource):
            raise TypeError(
                f"expected CityPriorSource subclass; got {type(source).__name__}"
            )
        self.sources.append(source)
        return self

    def factors_for(
        self,
        entities: Sequence[CityEntity],
        taxonomy: Taxonomy,
    ) -> List[Factor]:
        out: List[Factor] = []
        for src in self.sources:
            out.extend(src.factors_for(entities, taxonomy))
        return out


# ---------------------------------------------------------------------------
# Helpers: lat/lon point-in-polygon and distance-to-polyline
# ---------------------------------------------------------------------------


def _bbox_contains(
    bbox: Tuple[float, float, float, float],
    lon: float,
    lat: float,
) -> bool:
    """Inclusive (lon_min, lat_min, lon_max, lat_max) bbox test."""
    lo_n, la_n, lo_x, la_x = bbox
    return (lo_n <= lon <= lo_x) and (la_n <= lat <= la_x)


def _polygon_bbox(poly: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    arr = np.asarray(poly, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 3:
        raise ValueError(
            f"polygon must be (N>=3, 2) lon/lat; got shape {arr.shape}"
        )
    return (
        float(arr[:, 0].min()),
        float(arr[:, 1].min()),
        float(arr[:, 0].max()),
        float(arr[:, 1].max()),
    )


def _point_in_polygon(poly: np.ndarray, lon: float, lat: float) -> bool:
    """Ray-casting point-in-polygon (lon/lat).

    Polygon must be a closed ring as ``(N, 2)`` array of ``[lon, lat]``;
    works for non-convex polygons.  Geographic coordinates are treated
    as planar; this is fine at city scale (<= a few km) where a bbox
    pre-filter rejects everything else.
    """
    n = poly.shape[0]
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = float(poly[i, 0]), float(poly[i, 1])
        xj, yj = float(poly[j, 0]), float(poly[j, 1])
        intersect = ((yi > lat) != (yj > lat)) and (
            lon < (xj - xi) * (lat - yi) / (yj - yi + 1e-30) + xi
        )
        if intersect:
            inside = not inside
        j = i
    return inside


# Approximate metres per degree on the WGS84 ellipsoid (latitude-dependent).
_METRES_PER_DEG_LAT = 111_320.0


def _metres_per_deg_lon(lat_deg: float) -> float:
    return float(_METRES_PER_DEG_LAT * math.cos(math.radians(lat_deg)))


def _segment_distance_m(
    lon: float,
    lat: float,
    a: Tuple[float, float],
    b: Tuple[float, float],
) -> float:
    """Approximate metric distance from (lon, lat) to segment ``a-b``.

    Uses a local equirectangular projection at the query point.  Plenty
    accurate for the few-hundred-metre scales we care about for road
    proximity priors.
    """
    cos_lat = math.cos(math.radians(lat))
    ax = (a[0] - lon) * _METRES_PER_DEG_LAT * cos_lat
    ay = (a[1] - lat) * _METRES_PER_DEG_LAT
    bx = (b[0] - lon) * _METRES_PER_DEG_LAT * cos_lat
    by = (b[1] - lat) * _METRES_PER_DEG_LAT
    dx = bx - ax
    dy = by - ay
    L2 = dx * dx + dy * dy
    if L2 < 1e-12:
        return float(math.hypot(ax, ay))
    t = -(ax * dx + ay * dy) / L2
    t = max(0.0, min(1.0, t))
    px = ax + t * dx
    py = ay + t * dy
    return float(math.hypot(px, py))


# ---------------------------------------------------------------------------
# OSM-based sources
# ---------------------------------------------------------------------------


@dataclass
class OSMBuildingPriorSource(CityPriorSource):
    """Push entities whose centroid lies inside an OSM building footprint
    toward the building-class distribution.

    Parameters
    ----------
    building_polygons
        List of polygon rings, each ``(N, 2)`` array-likes of
        ``[lon, lat]`` in degrees.  Tests provide synthetic squares;
        production callers can populate from
        ``kernelcal.urban.city_graph.fetch_buildings_bbox`` (see
        :func:`from_geodataframe`).
    target_class
        Taxonomy category to bias toward.  Default ``"building"``.
    bonus
        Log-likelihood bonus added on hit.
    margin_deg
        Optional padding (degrees) around each polygon's bbox to absorb
        small SLAM lateral drift before the in-polygon test.
    """

    name: str = "osm_buildings"
    building_polygons: Sequence[np.ndarray] = field(default_factory=list)
    target_class: str = "building"
    bonus: float = 1.5
    margin_deg: float = 0.0
    _bboxes: Optional[List[Tuple[float, float, float, float]]] = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        polys: List[np.ndarray] = []
        bboxes: List[Tuple[float, float, float, float]] = []
        for poly in self.building_polygons:
            arr = np.asarray(poly, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 3:
                raise ValueError(
                    f"building polygons must be (N>=3, 2) lon/lat; got shape {arr.shape}"
                )
            polys.append(arr)
            bboxes.append(_polygon_bbox(arr))
        # Cache for fast pre-filter.
        object.__setattr__(self, "building_polygons", polys)
        object.__setattr__(self, "_bboxes", bboxes)

    @classmethod
    def from_geodataframe(
        cls,
        gdf: Any,
        *,
        target_class: str = "building",
        bonus: float = 1.5,
        margin_deg: float = 0.0,
    ) -> "OSMBuildingPriorSource":
        """Construct from a GeoPandas GeoDataFrame in EPSG:4326.

        Each row's geometry must be a ``Polygon`` or ``MultiPolygon``
        in lon/lat.  Multi-polygons are expanded to one entry per ring.
        """
        polys: List[np.ndarray] = []
        try:
            from shapely.geometry import MultiPolygon, Polygon  # type: ignore
        except Exception as exc:  # pragma: no cover -- exercised only with shapely
            raise ImportError("shapely required for from_geodataframe()") from exc
        for geom in gdf.geometry:
            if geom is None or geom.is_empty:
                continue
            if isinstance(geom, Polygon):
                xy = np.asarray(geom.exterior.coords, dtype=np.float64)
                polys.append(xy)
            elif isinstance(geom, MultiPolygon):
                for sub in geom.geoms:
                    xy = np.asarray(sub.exterior.coords, dtype=np.float64)
                    polys.append(xy)
        return cls(
            building_polygons=polys,
            target_class=target_class,
            bonus=bonus,
            margin_deg=float(margin_deg),
        )

    def factors_for(
        self,
        entities: Sequence[CityEntity],
        taxonomy: Taxonomy,
    ) -> List[Factor]:
        try:
            target_idx = taxonomy.index_of(self.target_class)
        except KeyError:
            return []
        if not self.building_polygons:
            return []
        out: List[Factor] = []
        m = float(self.margin_deg)
        bboxes = self._bboxes or []
        for ent in entities:
            if ent.lon is None or ent.lat is None:
                continue
            lon = float(ent.lon)
            lat = float(ent.lat)
            for poly, bbox in zip(self.building_polygons, bboxes):
                padded = (bbox[0] - m, bbox[1] - m, bbox[2] + m, bbox[3] + m)
                if not _bbox_contains(padded, lon, lat):
                    continue
                if _point_in_polygon(poly, lon, lat):
                    out.append(
                        UnaryClassPriorFactor(
                            ent.var_id,
                            target_class_index=target_idx,
                            n_states=taxonomy.n,
                            bonus=float(self.bonus),
                            name=f"osm_building:{ent.var_id}",
                            metadata={"source": self.name, "target": self.target_class},
                        )
                    )
                    break
        return out


@dataclass
class OSMRoadPriorSource(CityPriorSource):
    """Bias entities near OSM road centerlines toward road / pavement
    classes; bias entities far from roads against those classes.

    Parameters
    ----------
    road_polylines
        List of polylines, each ``(N>=2, 2)`` array-like of
        ``[lon, lat]``.
    near_classes
        Taxonomy categories to push toward when the entity is within
        ``near_distance_m`` of any road.  Default ``("road", "pavement",
        "vehicle")``.
    near_distance_m
        Distance (m) within which a road hit fires.
    bonus_near
        Bonus added to ``near_classes`` on a near hit.
    """

    name: str = "osm_roads"
    road_polylines: Sequence[np.ndarray] = field(default_factory=list)
    near_classes: Tuple[str, ...] = ("road", "pavement", "vehicle")
    near_distance_m: float = 5.0
    bonus_near: float = 1.0
    _road_bboxes: Optional[List[Tuple[float, float, float, float]]] = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        polys: List[np.ndarray] = []
        bboxes: List[Tuple[float, float, float, float]] = []
        for poly in self.road_polylines:
            arr = np.asarray(poly, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 2:
                raise ValueError(
                    f"road polylines must be (N>=2, 2) lon/lat; got shape {arr.shape}"
                )
            polys.append(arr)
            bboxes.append(
                (
                    float(arr[:, 0].min()),
                    float(arr[:, 1].min()),
                    float(arr[:, 0].max()),
                    float(arr[:, 1].max()),
                )
            )
        object.__setattr__(self, "road_polylines", polys)
        object.__setattr__(self, "_road_bboxes", bboxes)

    def _min_distance_m(self, lon: float, lat: float) -> float:
        best = float("inf")
        # Coarse degree-budget for the bbox pre-filter at this latitude.
        deg_budget = self.near_distance_m / max(_metres_per_deg_lon(lat), 1.0)
        for poly, bbox in zip(self.road_polylines, self._road_bboxes or []):
            padded = (
                bbox[0] - deg_budget,
                bbox[1] - deg_budget,
                bbox[2] + deg_budget,
                bbox[3] + deg_budget,
            )
            if not _bbox_contains(padded, lon, lat):
                continue
            for k in range(poly.shape[0] - 1):
                a = (float(poly[k, 0]), float(poly[k, 1]))
                b = (float(poly[k + 1, 0]), float(poly[k + 1, 1]))
                d = _segment_distance_m(lon, lat, a, b)
                if d < best:
                    best = d
                    if best == 0.0:
                        return best
        return best

    def factors_for(
        self,
        entities: Sequence[CityEntity],
        taxonomy: Taxonomy,
    ) -> List[Factor]:
        if not self.road_polylines:
            return []
        near_idx: List[int] = []
        for c in self.near_classes:
            try:
                near_idx.append(taxonomy.index_of(c))
            except KeyError:
                continue
        if not near_idx:
            return []
        out: List[Factor] = []
        for ent in entities:
            if ent.lon is None or ent.lat is None:
                continue
            d = self._min_distance_m(float(ent.lon), float(ent.lat))
            if d <= float(self.near_distance_m):
                out.append(
                    UnaryClassPriorFactor(
                        ent.var_id,
                        target_class_index=near_idx,
                        n_states=taxonomy.n,
                        bonus=float(self.bonus_near),
                        name=f"osm_road_near:{ent.var_id}",
                        metadata={
                            "source": self.name,
                            "distance_m": float(d),
                            "targets": list(self.near_classes),
                        },
                    )
                )
        return out


# ---------------------------------------------------------------------------
# Land-cover raster source
# ---------------------------------------------------------------------------


@dataclass
class LandCoverPriorSource(CityPriorSource):
    """Bias entities toward the class implied by a land-cover raster.

    Parameters
    ----------
    raster
        2D ``np.ndarray`` of integer class codes.  Index ``[row, col]``.
    bbox
        ``(lon_min, lat_min, lon_max, lat_max)`` in degrees.  Row 0 is
        the *northernmost* row (i.e. ``raster[0, :]`` corresponds to
        ``lat_max``); typical raster reader convention.
    code_to_class
        Mapping from raster code -> taxonomy category name.  Codes not
        in the mapping are skipped (no factor emitted).
    bonus
        Log-likelihood bonus added on a class hit.
    """

    name: str = "land_cover"
    raster: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.int32))
    bbox: Tuple[float, float, float, float] = (-180.0, -90.0, 180.0, 90.0)
    code_to_class: Mapping[int, str] = field(default_factory=dict)
    bonus: float = 1.0

    def __post_init__(self) -> None:
        if self.raster.ndim != 2:
            raise ValueError(f"raster must be 2D; got {self.raster.shape}")
        if len(self.bbox) != 4:
            raise ValueError("bbox must be (lon_min, lat_min, lon_max, lat_max)")
        lo_n, la_n, lo_x, la_x = self.bbox
        if not (lo_n < lo_x and la_n < la_x):
            raise ValueError(f"degenerate bbox: {self.bbox}")

    def _sample(self, lon: float, lat: float) -> Optional[int]:
        lo_n, la_n, lo_x, la_x = self.bbox
        if not (lo_n <= lon <= lo_x and la_n <= lat <= la_x):
            return None
        h, w = self.raster.shape
        # Clamp to inside (avoid OOB at the upper edges).
        col = int(np.clip(np.floor((lon - lo_n) / (lo_x - lo_n) * w), 0, w - 1))
        # Row 0 = north.
        row = int(np.clip(np.floor((la_x - lat) / (la_x - la_n) * h), 0, h - 1))
        return int(self.raster[row, col])

    def factors_for(
        self,
        entities: Sequence[CityEntity],
        taxonomy: Taxonomy,
    ) -> List[Factor]:
        if not self.code_to_class:
            return []
        # Pre-resolve code -> taxonomy index, skipping unknown classes once.
        code_to_idx: Dict[int, int] = {}
        for code, cls_name in self.code_to_class.items():
            try:
                code_to_idx[int(code)] = taxonomy.index_of(cls_name)
            except KeyError:
                continue
        if not code_to_idx:
            return []
        out: List[Factor] = []
        for ent in entities:
            if ent.lon is None or ent.lat is None:
                continue
            code = self._sample(float(ent.lon), float(ent.lat))
            if code is None:
                continue
            target_idx = code_to_idx.get(int(code))
            if target_idx is None:
                continue
            out.append(
                UnaryClassPriorFactor(
                    ent.var_id,
                    target_class_index=target_idx,
                    n_states=taxonomy.n,
                    bonus=float(self.bonus),
                    name=f"land_cover:{ent.var_id}",
                    metadata={
                        "source": self.name,
                        "code": int(code),
                        "target": self.code_to_class[int(code)],
                    },
                )
            )
        return out


# ---------------------------------------------------------------------------
# DEM ground source
# ---------------------------------------------------------------------------


@dataclass
class DEMGroundPriorSource(CityPriorSource):
    """Bias entities toward ground vs. elevated semantics using a DEM.

    Parameters
    ----------
    dem
        2D ``np.ndarray[float]`` of ellipsoidal elevations (m).
    bbox
        ``(lon_min, lat_min, lon_max, lat_max)`` in degrees, same
        convention as :class:`LandCoverPriorSource`.
    ground_classes
        Taxonomy categories that should have base-z near the DEM
        surface (default: ``("road", "pavement", "bare_ground", "water",
        "vegetation_other")``).
    elevated_classes
        Taxonomy categories that should sit above the DEM surface
        (default: ``("building", "tree", "vehicle")``).
    ground_tol_m
        Tolerance band (m) around the DEM elevation for which the
        entity counts as "ground".
    bonus
        Log-likelihood bonus added on the matching half.
    """

    name: str = "dem_ground"
    dem: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float64))
    bbox: Tuple[float, float, float, float] = (-180.0, -90.0, 180.0, 90.0)
    ground_classes: Tuple[str, ...] = (
        "road",
        "pavement",
        "bare_ground",
        "water",
        "vegetation_other",
    )
    elevated_classes: Tuple[str, ...] = ("building", "tree", "vehicle")
    ground_tol_m: float = 0.5
    bonus: float = 1.0

    def __post_init__(self) -> None:
        if self.dem.ndim != 2:
            raise ValueError(f"dem must be 2D; got {self.dem.shape}")
        if len(self.bbox) != 4:
            raise ValueError("bbox must be (lon_min, lat_min, lon_max, lat_max)")
        lo_n, la_n, lo_x, la_x = self.bbox
        if not (lo_n < lo_x and la_n < la_x):
            raise ValueError(f"degenerate bbox: {self.bbox}")
        if self.ground_tol_m < 0.0:
            raise ValueError("ground_tol_m must be >= 0")

    def _sample_dem(self, lon: float, lat: float) -> Optional[float]:
        lo_n, la_n, lo_x, la_x = self.bbox
        if not (lo_n <= lon <= lo_x and la_n <= lat <= la_x):
            return None
        h, w = self.dem.shape
        col = int(np.clip(np.floor((lon - lo_n) / (lo_x - lo_n) * w), 0, w - 1))
        row = int(np.clip(np.floor((la_x - lat) / (la_x - la_n) * h), 0, h - 1))
        return float(self.dem[row, col])

    def factors_for(
        self,
        entities: Sequence[CityEntity],
        taxonomy: Taxonomy,
    ) -> List[Factor]:
        ground_idx: List[int] = []
        for c in self.ground_classes:
            try:
                ground_idx.append(taxonomy.index_of(c))
            except KeyError:
                continue
        elevated_idx: List[int] = []
        for c in self.elevated_classes:
            try:
                elevated_idx.append(taxonomy.index_of(c))
            except KeyError:
                continue
        if not ground_idx and not elevated_idx:
            return []
        out: List[Factor] = []
        for ent in entities:
            if (
                ent.lon is None
                or ent.lat is None
                or ent.base_alt_m is None
            ):
                continue
            dem = self._sample_dem(float(ent.lon), float(ent.lat))
            if dem is None:
                continue
            base_above = float(ent.base_alt_m) - dem
            out.append(
                UnaryGroundElevationFactor(
                    ent.var_id,
                    n_states=taxonomy.n,
                    base_above_dem_m=base_above,
                    ground_class_indices=ground_idx,
                    elevated_class_indices=elevated_idx,
                    ground_tol_m=float(self.ground_tol_m),
                    bonus=float(self.bonus),
                    name=f"dem_ground:{ent.var_id}",
                )
            )
        return out


# ---------------------------------------------------------------------------
# Parent-child source
# ---------------------------------------------------------------------------


# Default class-pair compatibility table for PHX_URBAN_V0.
# Each entry is ``(parent_class_name, child_class_name)``.
PHX_PARENT_CHILD_COMPATIBLE: Tuple[Tuple[str, str], ...] = (
    ("tree", "tree"),                   # crown sharing the trunk's class
    ("tree", "vegetation_other"),       # canopy with shrub understorey
    ("building", "vehicle"),            # garage / carport scenarios
    ("building", "building"),           # building parts (annex / wing)
    ("road", "vehicle"),                # vehicle-on-road parent linkage
    ("road", "pavement"),               # sidewalk-along-road
    ("pavement", "pavement"),
    ("vegetation_other", "vegetation_other"),
)

PHX_PARENT_CHILD_INCOMPATIBLE: Tuple[Tuple[str, str], ...] = (
    ("water", "building"),
    ("water", "vehicle"),
    ("water", "tree"),
    ("building", "tree"),
    ("road", "tree"),
    ("road", "building"),
)


@dataclass
class ParentChildPriorSource(CityPriorSource):
    """Emit a :class:`PairwiseParentChildFactor` for each parent-child
    linkage in the entity list.

    The compatibility table is taxonomy-keyed by *category name* so
    different taxonomies can re-use the source without re-indexing.
    Pairs whose names are not present in the active taxonomy are
    silently skipped.
    """

    name: str = "parent_child"
    compatible_pairs: Sequence[Tuple[str, str]] = PHX_PARENT_CHILD_COMPATIBLE
    incompatible_pairs: Sequence[Tuple[str, str]] = PHX_PARENT_CHILD_INCOMPATIBLE
    log_compatible: float = 1.0
    log_incompatible: float = -1.5

    def factors_for(
        self,
        entities: Sequence[CityEntity],
        taxonomy: Taxonomy,
    ) -> List[Factor]:
        # Pre-resolve name pairs -> index pairs once.
        def _resolve(pairs: Sequence[Tuple[str, str]]) -> List[Tuple[int, int]]:
            out: List[Tuple[int, int]] = []
            for p_name, c_name in pairs:
                try:
                    pi = taxonomy.index_of(p_name)
                    ci = taxonomy.index_of(c_name)
                except KeyError:
                    continue
                out.append((pi, ci))
            return out

        comp = _resolve(self.compatible_pairs)
        incomp = _resolve(self.incompatible_pairs)
        if not comp and not incomp:
            return []
        ent_by_id: Dict[Hashable, CityEntity] = {ent.var_id: ent for ent in entities}
        out: List[Factor] = []
        for ent in entities:
            if ent.parent_var_id is None:
                continue
            if ent.parent_var_id not in ent_by_id:
                continue
            out.append(
                PairwiseParentChildFactor(
                    ent.parent_var_id,
                    ent.var_id,
                    n_states=taxonomy.n,
                    compatible_pairs=comp,
                    incompatible_pairs=incomp,
                    log_compatible=float(self.log_compatible),
                    log_incompatible=float(self.log_incompatible),
                    name=f"parent_child:{ent.parent_var_id}->{ent.var_id}",
                )
            )
        return out


# ---------------------------------------------------------------------------
# Convenience: build entities from a fused SceneGraph
# ---------------------------------------------------------------------------


def entities_from_fused_scene_graph(
    fused_dict: Mapping[str, Any],
) -> List[CityEntity]:
    """Build :class:`CityEntity` views from a serialized
    :class:`~kernelcal.distinction_game.collapse.FusedSceneGraph`.

    Looks at each fused node for the standard fields written by
    :func:`kernelcal.distinction_game.collapse.collapse_scene_graphs`:
    ``id``, ``geo_centroid`` (``[lon, lat]``), ``base_alt_m``, and
    ``parent_id``.  Missing fields collapse to ``None`` so sources
    silently skip them.
    """
    nodes = fused_dict.get("nodes") or []
    out: List[CityEntity] = []
    for node in nodes:
        var_id = str(node.get("id"))
        lon, lat = None, None
        gc = node.get("geo_centroid") or node.get("region", {}).get("geo_centroid")
        if gc and len(gc) >= 2:
            lon = float(gc[0])
            lat = float(gc[1])
        base_alt = node.get("base_alt_m")
        if base_alt is not None:
            base_alt = float(base_alt)
        parent_id = node.get("parent_id")
        if parent_id is not None:
            parent_id = str(parent_id)
        out.append(
            CityEntity(
                var_id=var_id,
                lon=lon,
                lat=lat,
                base_alt_m=base_alt,
                parent_var_id=parent_id,
                attributes=dict(node.get("attributes") or {}),
            )
        )
    return out


__all__ = [
    "CityEntity",
    "CityPriorSource",
    "CityPriorStack",
    "DEMGroundPriorSource",
    "LandCoverPriorSource",
    "OSMBuildingPriorSource",
    "OSMRoadPriorSource",
    "ParentChildPriorSource",
    "PHX_PARENT_CHILD_COMPATIBLE",
    "PHX_PARENT_CHILD_INCOMPATIBLE",
    "entities_from_fused_scene_graph",
]
