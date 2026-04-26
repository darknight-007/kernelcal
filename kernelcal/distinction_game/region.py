"""
Per-tile ground-set objects: :class:`Region` (an entry in ``R_t``)
and :class:`KernelClaim` (one source's reading of one region).

The design doc (§3, §11.1) treats ``R_t`` as the shared ground set on
which every kernel ``k_s`` lives, but in practice each external source
produces its own polygons (OSM features, SAM masks, MR boxes, GD
boxes…) and the *first* job of :func:`build_scene_graph` is to merge
those into a coherent :class:`Region` set. PR-1 keeps the geometry
representation deliberately light — vertex array + cached bbox — so
``kernelcal`` does not gain a hard dependency on ``shapely`` or
``geopandas``. The deepgis-xr orchestrator can layer a richer GIS
representation on top before passing claims in.

A :class:`KernelClaim` is the single unit of input the multi-source
fusion machinery accepts: *source ``s`` saw a region with this
geometry, attached this native label with this confidence*. Geometry
on a claim is in normalised image coordinates ``[0, 1]`` (the same
basis the Mask R-CNN service emits in ``masks_polygons_norm``); the
``geo`` field optionally carries the same polygon already projected
to lon/lat for the Cesium frontend.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Geometry helpers (numpy only — no shapely dependency)
# ---------------------------------------------------------------------------

def _as_xy_array(points: Sequence[Sequence[float]]) -> np.ndarray:
    """Coerce ``[(x, y), ...]`` into a contiguous ``(n, 2)`` float64 array.

    Closes the ring if the caller didn't (last vertex == first vertex);
    rejects rings with fewer than 3 distinct vertices.
    """
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(
            f"polygon vertices must have shape (n, 2); got {arr.shape}"
        )
    if arr.shape[0] < 3:
        raise ValueError(
            f"polygon needs at least 3 vertices; got {arr.shape[0]}"
        )
    if not np.allclose(arr[0], arr[-1]):
        arr = np.vstack([arr, arr[:1]])
    return arr


def _bbox_from_xy(xy: np.ndarray) -> Tuple[float, float, float, float]:
    """Return ``(xmin, ymin, xmax, ymax)`` for a closed ring."""
    return (
        float(xy[:, 0].min()),
        float(xy[:, 1].min()),
        float(xy[:, 0].max()),
        float(xy[:, 1].max()),
    )


def _polygon_area(xy: np.ndarray) -> float:
    """Shoelace area of a closed ring; always non-negative."""
    x = xy[:, 0]
    y = xy[:, 1]
    return float(0.5 * np.abs(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1])))


def bbox_iou(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    """Axis-aligned-bounding-box IoU. Used by the spatial-association
    step of :func:`build_scene_graph` to decide whether two
    :class:`KernelClaim` instances should be merged into the same node.

    PR-1 deliberately uses bbox-IoU (cheap, shapely-free) rather than
    full polygon IoU. The §4 fusion math is agnostic to which IoU
    signal feeds in; PR-3 will optionally swap in shapely-based
    polygon IoU when the dependency is available.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


# ---------------------------------------------------------------------------
# Region: the ground-set entry
# ---------------------------------------------------------------------------

@dataclass
class Region:
    """One entry in the shared ground set ``R_t`` for a tile.

    A :class:`Region` is created by the spatial-association step of
    :func:`~kernelcal.distinction_game.scene_graph.build_scene_graph`
    when one or more :class:`KernelClaim` instances cluster spatially.
    The geometry stored here is the *consensus* geometry — currently
    the highest-area claim, but PR-3 may switch to a per-pixel union.

    Attributes
    ----------
    id
        Stable identifier for the lifetime of the scene graph. Auto-
        generated as ``"r-<uuid>"`` if omitted.
    polygon
        Closed ring of vertices in normalised image coordinates
        ``[0, 1]`` (ground-truth basis used by the Mask R-CNN service
        for ``masks_polygons_norm``).
    bbox
        Cached ``(xmin, ymin, xmax, ymax)`` in the same basis as
        :attr:`polygon`.
    area
        Polygon area in the normalised basis (so the whole tile = 1.0).
    geo_polygon
        Optional lon/lat ring of the same outline, populated when the
        caller has already projected the polygon to geographic
        coordinates. Shape ``(n, 2)`` with columns ``[lon, lat]``.
    mask_rle
        Optional COCO-style run-length-encoded raster mask
        ``{"size": [H, W], "counts": "..."}`` for kernels (SAM,
        Mask R-CNN) that emit per-pixel masks.
    image_size
        Optional ``(width, height)`` of the source tile in pixels;
        lets consumers de-normalise ``polygon`` if they need pixel
        coordinates.
    attributes
        Free-form metadata bag for downstream consumers (e.g.
        ``year_built`` from a parcel record, ``height`` from a DSM).
    """

    polygon: np.ndarray
    bbox: Tuple[float, float, float, float]
    area: float
    id: str = field(default_factory=lambda: f"r-{uuid.uuid4().hex[:12]}")
    geo_polygon: Optional[np.ndarray] = None
    mask_rle: Optional[Dict[str, Any]] = None
    image_size: Optional[Tuple[int, int]] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    # ---- Constructors --------------------------------------------------

    @classmethod
    def from_polygon(
        cls,
        points: Sequence[Sequence[float]],
        *,
        id: Optional[str] = None,
        geo_polygon: Optional[Sequence[Sequence[float]]] = None,
        mask_rle: Optional[Dict[str, Any]] = None,
        image_size: Optional[Tuple[int, int]] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> "Region":
        """Build a :class:`Region` from a vertex sequence in normalised
        image coordinates. Closes the ring if needed and caches bbox + area.
        """
        xy = _as_xy_array(points)
        geo_xy: Optional[np.ndarray] = None
        if geo_polygon is not None:
            geo_xy = _as_xy_array(geo_polygon)
        return cls(
            polygon=xy,
            bbox=_bbox_from_xy(xy),
            area=_polygon_area(xy),
            id=id or f"r-{uuid.uuid4().hex[:12]}",
            geo_polygon=geo_xy,
            mask_rle=mask_rle,
            image_size=image_size,
            attributes=dict(attributes or {}),
        )

    # ---- Serialization -------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly view. ``polygon`` is rendered as a list of
        ``[x, y]`` pairs; ``geo_polygon`` (if present) becomes
        ``[[lon, lat], ...]``.
        """
        out: Dict[str, Any] = {
            "id": self.id,
            "polygon": self.polygon.tolist(),
            "bbox": list(self.bbox),
            "area": self.area,
            "image_size": list(self.image_size) if self.image_size else None,
            "attributes": dict(self.attributes),
        }
        if self.geo_polygon is not None:
            out["geo_polygon"] = self.geo_polygon.tolist()
        if self.mask_rle is not None:
            out["mask_rle"] = self.mask_rle
        return out


# ---------------------------------------------------------------------------
# KernelClaim: one source's reading of one region
# ---------------------------------------------------------------------------

@dataclass
class KernelClaim:
    """One source's reading of one region.

    Multiple claims (from different kernels, or from the same kernel
    on overlapping detections) get spatially associated into a shared
    :class:`Region` by :func:`~kernelcal.distinction_game.scene_graph.build_scene_graph`.
    The fused category posterior on the resulting :class:`SceneNode`
    is computed from the per-source ``Q_s`` evaluated on each claim's
    :attr:`native_label`.

    Attributes
    ----------
    source_id
        Which kernel produced this claim. Must match a key in the
        ``q_s_table`` passed to ``build_scene_graph`` (see
        :mod:`kernelcal.distinction_game.q_s` for the canonical names).
    native_label
        The kernel's own label string (e.g. ``"rock"`` for MR-rocks,
        ``"building"`` for OSM, ``"house_damage_2"`` for MR-house, the
        text query for Grounding-DINO; SAM's class-agnostic outputs
        get the literal string ``"<sam_segment>"``).
    score
        Native confidence in ``[0, 1]``. Used by
        :func:`build_scene_graph` to weight the claim's contribution
        and to set :attr:`SceneNode.score`.
    polygon
        Closed ring of vertices in normalised image coordinates
        ``[0, 1]``.
    bbox / area
        Cached helpers in the same basis as :attr:`polygon`.
    geo_polygon
        Optional lon/lat ring (see :class:`Region`).
    mask_rle / image_size
        Optional raster mask + tile size (see :class:`Region`).
    attributes
        Free-form per-claim metadata (e.g. ``model_id`` for the
        specific MR checkpoint, ``phrase`` for the GD query, ``tag``
        for OSM).
    """

    source_id: str
    native_label: str
    score: float
    polygon: np.ndarray
    bbox: Tuple[float, float, float, float]
    area: float
    geo_polygon: Optional[np.ndarray] = None
    mask_rle: Optional[Dict[str, Any]] = None
    image_size: Optional[Tuple[int, int]] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: f"c-{uuid.uuid4().hex[:12]}")

    # ---- Constructors --------------------------------------------------

    @classmethod
    def from_polygon(
        cls,
        source_id: str,
        native_label: str,
        score: float,
        points: Sequence[Sequence[float]],
        *,
        geo_polygon: Optional[Sequence[Sequence[float]]] = None,
        mask_rle: Optional[Dict[str, Any]] = None,
        image_size: Optional[Tuple[int, int]] = None,
        attributes: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None,
    ) -> "KernelClaim":
        """Convenience constructor parallel to :meth:`Region.from_polygon`."""
        if not 0.0 <= score <= 1.0:
            raise ValueError(
                f"claim score must be in [0, 1]; got {score!r} for "
                f"source {source_id!r}"
            )
        xy = _as_xy_array(points)
        geo_xy: Optional[np.ndarray] = None
        if geo_polygon is not None:
            geo_xy = _as_xy_array(geo_polygon)
        return cls(
            source_id=source_id,
            native_label=native_label,
            score=float(score),
            polygon=xy,
            bbox=_bbox_from_xy(xy),
            area=_polygon_area(xy),
            geo_polygon=geo_xy,
            mask_rle=mask_rle,
            image_size=image_size,
            attributes=dict(attributes or {}),
            id=id or f"c-{uuid.uuid4().hex[:12]}",
        )

    # ---- Serialization -------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "id": self.id,
            "source_id": self.source_id,
            "native_label": self.native_label,
            "score": self.score,
            "polygon": self.polygon.tolist(),
            "bbox": list(self.bbox),
            "area": self.area,
            "attributes": dict(self.attributes),
        }
        if self.geo_polygon is not None:
            out["geo_polygon"] = self.geo_polygon.tolist()
        if self.mask_rle is not None:
            out["mask_rle"] = self.mask_rle
        if self.image_size is not None:
            out["image_size"] = list(self.image_size)
        return out

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "KernelClaim":
        """Reconstruct a :class:`KernelClaim` from its :meth:`to_dict`
        round-trip form.

        Tolerates absent geometry fields (``polygon``/``bbox``/``area``)
        by defaulting to zero — useful when reading SceneGraph payloads
        whose claims may have been thinned for storage. PR-3 fits use
        the score and native label, not the geometry, so the empty
        defaults are harmless on that path.
        """
        polygon = payload.get("polygon")
        bbox = payload.get("bbox")
        area = float(payload.get("area", 0.0) or 0.0)
        if polygon is None:
            polygon_arr = np.zeros((0, 2), dtype=np.float64)
        else:
            polygon_arr = np.asarray(polygon, dtype=np.float64)
        if bbox is None:
            bbox_t: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
        else:
            bbox_t = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        geo_polygon = payload.get("geo_polygon")
        geo_polygon_arr = (
            np.asarray(geo_polygon, dtype=np.float64) if geo_polygon is not None else None
        )
        image_size = payload.get("image_size")
        image_size_t = tuple(image_size) if image_size is not None else None
        return cls(
            source_id=str(payload["source_id"]),
            native_label=str(payload["native_label"]),
            score=float(payload.get("score", 0.0) or 0.0),
            polygon=polygon_arr,
            bbox=bbox_t,
            area=area,
            geo_polygon=geo_polygon_arr,
            mask_rle=payload.get("mask_rle"),
            image_size=image_size_t,
            attributes=dict(payload.get("attributes") or {}),
            id=str(payload.get("id") or f"c-{uuid.uuid4().hex[:12]}"),
        )


# ---------------------------------------------------------------------------
# Convenience: bulk filtering / conversion
# ---------------------------------------------------------------------------

def filter_claims(
    claims: Iterable[KernelClaim],
    *,
    min_score: float = 0.0,
    sources: Optional[Sequence[str]] = None,
) -> List[KernelClaim]:
    """Drop low-confidence claims and optionally restrict to a subset of sources."""
    keep = []
    src_filter = set(sources) if sources is not None else None
    for c in claims:
        if c.score < min_score:
            continue
        if src_filter is not None and c.source_id not in src_filter:
            continue
        keep.append(c)
    return keep
