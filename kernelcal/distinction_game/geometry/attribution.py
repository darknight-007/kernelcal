"""Sensor-data attribution onto posed superquadrics.

Three attributors map raw earth-rover sensor streams into per-SQ
:class:`SuperquadricPropertyStore` updates:

* :class:`LidarIntensityAttributor` -- Velodyne VLP-16 returns ``->``
  per-SQ intensity, density, and return-ratio statistics.

* :class:`MicaSenseAttributor` -- 5-band Altum multispectral + LWIR
  thermal frames ``->`` per-SQ NDVI, NDRE, GNDVI, surface temperature,
  and (optionally) chlorophyll/LAI proxies.

* :class:`OceanOpticsAttributor` -- single-fiber UV-VIS-NIR
  spectrometer through a collimating lens at the camera bore-sight
  ``->`` per-SQ :class:`SpectrumAccumulator` updates plus derived
  vegetation-index properties.

All three share a small spatial index (:class:`SQSpatialIndex`) that
provides bbox-broad-phase + narrow-phase point-membership and
ray-first-hit queries.

Geometry conventions
--------------------

* World frame is the rover's session-local ENU (or UTM-offset) frame
  (see :mod:`.frames` in PR-5.7).  All SQs are posed in this frame.
* Sensor poses (camera, LiDAR, spectrometer) are SE(3) transforms
  taking *body* points into *world* points.  Bore-sight rays start
  at the sensor origin and point along ``+x_sensor`` by convention
  (configurable via ``bore_sight_axis``).
* Camera intrinsics are the standard pinhole model
  ``K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`` with no distortion
  (callers should rectify before attribution).

Public API
----------

::

    from kernelcal.distinction_game.geometry.attribution import (
        SQSpatialIndex,
        LidarIntensityAttributor,
        MicaSenseAttributor,
        OceanOpticsAttributor,
        ndvi, ndre, gndvi,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .accumulators import SuperquadricPropertyStore
from .properties import PropertyId
from .spectrum import SpectrumAccumulator
from .superquadric import Superquadric


# ---------------------------------------------------------------------------
# Spatial index
# ---------------------------------------------------------------------------


@dataclass
class SQSpatialIndex:
    """O(N) bbox broad-phase + narrow-phase spatial index over SQs.

    Optimized for the small-to-medium SQ counts (10s-1000s) the
    earth-rover sees in a rolling perception window.  Builds once
    and queries:

    * :meth:`query_point` -- which SQ contains a 3D point (returns
      first hit by SQ list order; for inside-outside ties the
      earlier SQ wins).
    * :meth:`query_points` -- vectorized point membership: returns
      the index of the first containing SQ per query (or ``-1``).
    * :meth:`first_hit_ray` -- ray cast: returns ``(sq_idx, t)`` of
      the closest entry along a ray (negative-distance gate).
    * :meth:`project_into_image` -- list of ``(sq_idx, mask, depth)``
      for every SQ whose silhouette intersects an image plane.

    Implementation
    --------------

    AABBs are precomputed once via :meth:`Superquadric.bbox_3d`.  The
    broad phase iterates over all bboxes (numpy vectorized);
    narrow-phase calls the SQ's ``contains`` / ``inside_outside``.
    A real KD-tree could replace this for >>1k SQs but the constant
    factors here are small.
    """

    sqs: List[Superquadric] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._mins = np.zeros((0, 3))
        self._maxs = np.zeros((0, 3))
        self._rebuild_aabbs()

    def add(self, sq: Superquadric) -> int:
        """Add an SQ to the index; returns its index."""
        self.sqs.append(sq)
        self._rebuild_aabbs()
        return len(self.sqs) - 1

    def extend(self, sqs: Iterable[Superquadric]) -> None:
        for sq in sqs:
            self.sqs.append(sq)
        self._rebuild_aabbs()

    def _rebuild_aabbs(self) -> None:
        if not self.sqs:
            self._mins = np.zeros((0, 3))
            self._maxs = np.zeros((0, 3))
            return
        mins = np.zeros((len(self.sqs), 3))
        maxs = np.zeros((len(self.sqs), 3))
        for i, sq in enumerate(self.sqs):
            lo, hi = sq.bbox_3d()
            mins[i] = lo
            maxs[i] = hi
        self._mins = mins
        self._maxs = maxs

    # ---- Point queries ----------------------------------------------

    def candidate_indices_for_point(self, point: np.ndarray) -> np.ndarray:
        """Indices of SQs whose AABB contains the point."""
        if len(self.sqs) == 0:
            return np.zeros(0, dtype=np.int64)
        p = np.asarray(point, dtype=float).reshape(3)
        in_box = np.all((p >= self._mins) & (p <= self._maxs), axis=1)
        return np.where(in_box)[0]

    def query_point(self, point: np.ndarray) -> Optional[int]:
        """Return the index of the first SQ containing ``point``, or None."""
        for idx in self.candidate_indices_for_point(point):
            sq = self.sqs[int(idx)]
            if bool(sq.contains(np.asarray(point, dtype=float).reshape(1, 3))[0]):
                return int(idx)
        return None

    def query_points(self, points: np.ndarray) -> np.ndarray:
        """Vectorized membership: ``(N,)`` indices, ``-1`` for no hit."""
        if len(self.sqs) == 0:
            return -np.ones(len(points), dtype=np.int64)
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        out = -np.ones(len(pts), dtype=np.int64)
        # Broad-phase: per-SQ AABB filter, then SQ-level contains() check.
        for i, sq in enumerate(self.sqs):
            lo = self._mins[i]
            hi = self._maxs[i]
            in_box = np.all((pts >= lo) & (pts <= hi), axis=1)
            if not np.any(in_box):
                continue
            still_open = (out < 0) & in_box
            if not np.any(still_open):
                continue
            mask_pts = pts[still_open]
            inside = sq.contains(mask_pts)
            # Map back to original indices.
            open_idx = np.where(still_open)[0]
            out[open_idx[inside]] = i
        return out

    # ---- Ray queries -------------------------------------------------

    def first_hit_ray(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        *,
        t_min: float = 0.0,
        t_max: float = 100.0,
        n_samples: int = 64,
    ) -> Optional[Tuple[int, float]]:
        """Closest SQ entry along a ray.

        Implementation: AABB-prune candidates with the standard slab
        test, then sample ``n_samples`` points along each candidate's
        ``[t_enter, t_exit]`` interval and find the first inside.

        Returns ``(sq_idx, t)`` or None.
        """
        if len(self.sqs) == 0:
            return None
        o = np.asarray(origin, dtype=float).reshape(3)
        d = np.asarray(direction, dtype=float).reshape(3)
        d_norm = float(np.linalg.norm(d))
        if d_norm < 1e-12:
            return None
        d = d / d_norm

        best: Optional[Tuple[int, float]] = None
        for i in range(len(self.sqs)):
            t_enter, t_exit = _ray_aabb(o, d, self._mins[i], self._maxs[i])
            if t_exit < max(t_enter, t_min) or t_enter > t_max:
                continue
            t_lo = max(t_enter, t_min)
            t_hi = min(t_exit, t_max)
            if t_hi <= t_lo:
                continue
            ts = np.linspace(t_lo, t_hi, int(n_samples))
            pts = o + ts[:, None] * d[None, :]
            inside = self.sqs[i].contains(pts)
            if not np.any(inside):
                continue
            j = int(np.argmax(inside))
            t_hit = float(ts[j])
            if best is None or t_hit < best[1]:
                best = (i, t_hit)
        return best


def _ray_aabb(o: np.ndarray, d: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> Tuple[float, float]:
    """Standard ray-AABB slab test.  Returns ``(t_enter, t_exit)``.

    Handles d_axis=0 (ray parallel to a slab) without producing NaNs:
    parallel rays are inside the slab iff ``lo <= origin <= hi`` along
    that axis; otherwise the ray misses the AABB entirely.
    """
    o = np.asarray(o, dtype=float)
    d = np.asarray(d, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    t_enter = -math.inf
    t_exit = math.inf
    for ax in range(3):
        if abs(d[ax]) < 1e-12:
            # Parallel: check origin is inside the slab.
            if o[ax] < lo[ax] or o[ax] > hi[ax]:
                return math.inf, -math.inf  # no intersection
            continue
        t1 = (lo[ax] - o[ax]) / d[ax]
        t2 = (hi[ax] - o[ax]) / d[ax]
        t_min = min(t1, t2)
        t_max = max(t1, t2)
        if t_min > t_enter:
            t_enter = t_min
        if t_max < t_exit:
            t_exit = t_max
    return t_enter, t_exit


# ---------------------------------------------------------------------------
# Helpers: SE(3) transforms and pinhole projection
# ---------------------------------------------------------------------------


def _apply_se3(R: np.ndarray, t: np.ndarray, points: np.ndarray) -> np.ndarray:
    """``world = R @ body + t`` for a (N, 3) batch."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    return pts @ np.asarray(R, dtype=float).T + np.asarray(t, dtype=float).reshape(3)


def _world_to_pixel(
    world_pts: np.ndarray,
    R_cam_world: np.ndarray,
    t_cam_world: np.ndarray,
    K: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project (N, 3) world points to pixel coords via pinhole.

    ``R_cam_world, t_cam_world`` give the camera *world->camera* pose:
    a world point ``p_w`` maps to camera frame as ``p_c = R @ p_w + t``.

    Returns ``(uv (N, 2), depth (N,))``.  Behind-camera points have
    ``depth <= 0``.
    """
    p_w = np.asarray(world_pts, dtype=float).reshape(-1, 3)
    R = np.asarray(R_cam_world, dtype=float).reshape(3, 3)
    t = np.asarray(t_cam_world, dtype=float).reshape(3)
    p_c = p_w @ R.T + t
    z = p_c[:, 2]
    eps = 1e-6
    z_safe = np.where(np.abs(z) < eps, eps, z)
    u = K[0, 0] * (p_c[:, 0] / z_safe) + K[0, 2]
    v = K[1, 1] * (p_c[:, 1] / z_safe) + K[1, 2]
    return np.stack([u, v], axis=1), z


# ---------------------------------------------------------------------------
# Vegetation index helpers
# ---------------------------------------------------------------------------


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Normalized Difference Vegetation Index, robust to division-by-zero."""
    nir = np.asarray(nir, dtype=float)
    red = np.asarray(red, dtype=float)
    denom = nir + red
    out = np.zeros_like(nir)
    nz = denom > 1e-6
    out[nz] = (nir[nz] - red[nz]) / denom[nz]
    return out


def ndre(nir: np.ndarray, red_edge: np.ndarray) -> np.ndarray:
    """Red-edge NDVI; chlorophyll-content proxy."""
    nir = np.asarray(nir, dtype=float)
    re = np.asarray(red_edge, dtype=float)
    denom = nir + re
    out = np.zeros_like(nir)
    nz = denom > 1e-6
    out[nz] = (nir[nz] - re[nz]) / denom[nz]
    return out


def gndvi(nir: np.ndarray, green: np.ndarray) -> np.ndarray:
    """Green NDVI; leaf-nitrogen proxy."""
    nir = np.asarray(nir, dtype=float)
    g = np.asarray(green, dtype=float)
    denom = nir + g
    out = np.zeros_like(nir)
    nz = denom > 1e-6
    out[nz] = (nir[nz] - g[nz]) / denom[nz]
    return out


def evi(nir: np.ndarray, red: np.ndarray, blue: np.ndarray, *, L: float = 1.0) -> np.ndarray:
    """Enhanced Vegetation Index with Liu/Huete coefficients."""
    nir = np.asarray(nir, dtype=float)
    red = np.asarray(red, dtype=float)
    blue = np.asarray(blue, dtype=float)
    denom = nir + 6.0 * red - 7.5 * blue + L
    out = np.zeros_like(nir)
    nz = np.abs(denom) > 1e-6
    out[nz] = 2.5 * (nir[nz] - red[nz]) / denom[nz]
    return out


# ---------------------------------------------------------------------------
# 1. Lidar intensity attributor
# ---------------------------------------------------------------------------


@dataclass
class LidarIntensityAttributor:
    """Velodyne VLP-16 returns -> per-SQ intensity / density properties.

    Usage::

        attr = LidarIntensityAttributor(index)
        attr.attribute(points_xyzi=cloud, sensor_pose=(R_lw, t_lw))
        # ... ship: store = attr.stores[sq_id]
    """

    index: SQSpatialIndex
    stores: Dict[str, SuperquadricPropertyStore] = field(default_factory=dict)

    def _store_for(self, sq_id: str) -> SuperquadricPropertyStore:
        s = self.stores.get(sq_id)
        if s is None:
            s = SuperquadricPropertyStore(sq_id=sq_id)
            self.stores[sq_id] = s
        return s

    def attribute(
        self,
        points_xyzi: np.ndarray,
        *,
        sensor_pose: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        max_returns_per_sq: Optional[int] = None,
    ) -> Dict[str, SuperquadricPropertyStore]:
        """Map LiDAR returns into per-SQ intensity statistics.

        Parameters
        ----------
        points_xyzi
            ``(N, 4)`` array: ``[x, y, z, intensity]``.  XYZ are in
            **sensor** frame if ``sensor_pose`` is provided, otherwise
            already in world.  Intensity is normalized to ``[0, 255]``.
        sensor_pose
            Optional ``(R_world_sensor, t_world_sensor)`` SE(3) taking
            sensor points into world.
        max_returns_per_sq
            Optional cap: subsample to this many returns per SQ before
            updating accumulators.  Useful when one SQ dominates a
            scan and would otherwise drown the running stats.

        Returns
        -------
        ``{sq_id: SuperquadricPropertyStore}`` for every SQ that
        received any returns.
        """
        pts = np.asarray(points_xyzi, dtype=float).reshape(-1, 4)
        if pts.size == 0 or len(self.index.sqs) == 0:
            return self.stores

        xyz = pts[:, :3]
        intens = pts[:, 3]
        if sensor_pose is not None:
            R, t = sensor_pose
            xyz = _apply_se3(R, t, xyz)

        membership = self.index.query_points(xyz)
        touched: Dict[int, List[int]] = {}
        for j, sq_idx in enumerate(membership):
            if sq_idx < 0:
                continue
            touched.setdefault(int(sq_idx), []).append(j)

        for sq_idx, idx_list in touched.items():
            sq = self.index.sqs[sq_idx]
            store = self._store_for(sq.id)
            sample_idx = np.asarray(idx_list, dtype=np.int64)
            if max_returns_per_sq is not None and sample_idx.size > max_returns_per_sq:
                rng = np.random.default_rng(seed=int(sq_idx))
                sample_idx = rng.choice(sample_idx, max_returns_per_sq, replace=False)

            sample_intens = intens[sample_idx]
            store.update_batch(PropertyId.LIDAR_INTENSITY_MEAN, sample_intens)
            if sample_intens.size >= 2:
                std = float(np.std(sample_intens))
                store.update(PropertyId.LIDAR_INTENSITY_STD, std)

            # Density: returns / volume (saturate to property domain).
            vol = max(sq.volume(), 1e-6)
            density = float(sample_idx.size) / vol
            store.update(PropertyId.POINT_DENSITY, density)

        return self.stores


# ---------------------------------------------------------------------------
# 2. MicaSense Altum attributor (5 multispec + thermal)
# ---------------------------------------------------------------------------


@dataclass
class MicaSenseAttributor:
    """5-band + LWIR thermal -> per-SQ NDVI / NDRE / temperature props.

    The Altum delivers 6 co-registered bands per capture:

    * blue (~475 nm), green (~560 nm), red (~668 nm),
      red-edge (~717 nm), NIR (~842 nm), thermal (8-14 um in degC)

    For each captured frame this attributor:

    1. Identifies SQs in the camera FoV (broad-phase via AABB
       projection).
    2. Rasterizes each SQ's silhouette at the image resolution
       (using a coarse tessellation of the SQ surface).
    3. Aggregates per-band means inside the silhouette mask (excluding
       behind-camera pixels and saturated pixels).
    4. Derives NDVI / NDRE / GNDVI / EVI / surface-temp and updates
       the per-SQ store via Welford.

    Usage::

        attr = MicaSenseAttributor(index, K=K)
        attr.attribute(
            bands={"blue": ..., "green": ..., "red": ..., "red_edge": ...,
                   "nir": ..., "thermal_C": ...},
            camera_pose=(R_cw, t_cw),
        )
    """

    index: SQSpatialIndex
    K: np.ndarray
    image_shape: Optional[Tuple[int, int]] = None  # (H, W); inferred from bands if None
    n_lat: int = 16
    n_lon: int = 24
    saturation_value: float = 1.0  # bands assumed normalized to [0, 1]
    stores: Dict[str, SuperquadricPropertyStore] = field(default_factory=dict)

    def _store_for(self, sq_id: str) -> SuperquadricPropertyStore:
        s = self.stores.get(sq_id)
        if s is None:
            s = SuperquadricPropertyStore(sq_id=sq_id)
            self.stores[sq_id] = s
        return s

    def attribute(
        self,
        bands: Mapping[str, np.ndarray],
        *,
        camera_pose: Tuple[np.ndarray, np.ndarray],
        irradiance_normalize: bool = False,
        irradiance: Optional[Mapping[str, float]] = None,
    ) -> Dict[str, SuperquadricPropertyStore]:
        """Project SQ silhouettes and aggregate per-band band statistics.

        Parameters
        ----------
        bands
            Mapping of band name to ``(H, W)`` array.  Required keys
            depend on which derived properties you want; missing
            bands skip those properties gracefully.

            * ``"blue"``, ``"green"``, ``"red"``, ``"red_edge"``,
              ``"nir"``: reflectance in ``[0, 1]``.
            * ``"thermal_C"``: surface temperature in degC.
        camera_pose
            ``(R_cam_world, t_cam_world)`` taking world points into
            the camera frame.
        irradiance_normalize
            If True and ``irradiance`` provided, divide each band by
            its irradiance scalar before computing indices.
        irradiance
            Per-band scalar irradiance from the Altum DLS-2 sun
            sensor (W/m^2).  Only used when ``irradiance_normalize``.
        """
        if len(self.index.sqs) == 0 or not bands:
            return self.stores

        # Infer image shape from any provided band.
        first_band = next(iter(bands.values()))
        H, W = self.image_shape or first_band.shape[:2]

        # Optional irradiance normalization.
        norm_bands: Dict[str, np.ndarray] = {}
        for name, arr in bands.items():
            arr = np.asarray(arr, dtype=float)
            if irradiance_normalize and irradiance and name in irradiance:
                scale = max(float(irradiance[name]), 1e-6)
                arr = arr / scale
            norm_bands[name] = arr

        R_cw, t_cw = camera_pose

        for sq in self.index.sqs:
            mask, depth = self._project_silhouette(sq, R_cw, t_cw, H, W)
            n_pix = int(np.count_nonzero(mask))
            if n_pix < 8:
                continue
            store = self._store_for(sq.id)

            # Per-band means inside the silhouette.
            band_means: Dict[str, float] = {}
            for name, arr in norm_bands.items():
                if arr.shape[:2] != (H, W):
                    continue
                vals = arr[mask]
                # Drop non-finite pixels.  Apply saturation gating only
                # to reflectance bands (thermal is in degC, not [0, 1]).
                if name == "thermal_C":
                    valid = np.isfinite(vals)
                else:
                    valid = np.isfinite(vals) & (vals < self.saturation_value * 0.999)
                if not np.any(valid):
                    continue
                band_means[name] = float(np.mean(vals[valid]))

            # RGB updates (if R/G/B available).
            if {"red", "green", "blue"} <= band_means.keys():
                store.update(PropertyId.RGB_R_MEAN, np.clip(band_means["red"] * 255.0, 0, 255))
                store.update(PropertyId.RGB_G_MEAN, np.clip(band_means["green"] * 255.0, 0, 255))
                store.update(PropertyId.RGB_B_MEAN, np.clip(band_means["blue"] * 255.0, 0, 255))

            # Vegetation indices.
            if {"nir", "red"} <= band_means.keys():
                v = float(ndvi(np.array([band_means["nir"]]), np.array([band_means["red"]]))[0])
                store.update(PropertyId.NDVI, v)
            if {"nir", "red_edge"} <= band_means.keys():
                v = float(ndre(np.array([band_means["nir"]]), np.array([band_means["red_edge"]]))[0])
                store.update(PropertyId.NDRE, v)
            if {"nir", "green"} <= band_means.keys():
                v = float(gndvi(np.array([band_means["nir"]]), np.array([band_means["green"]]))[0])
                store.update(PropertyId.GNDVI, v)
            if {"nir", "red", "blue"} <= band_means.keys():
                v = float(
                    evi(
                        np.array([band_means["nir"]]),
                        np.array([band_means["red"]]),
                        np.array([band_means["blue"]]),
                    )[0]
                )
                store.update(PropertyId.EVI, v)

            # Thermal.
            if "thermal_C" in band_means:
                store.update(PropertyId.SURFACE_TEMP_C, band_means["thermal_C"])

        return self.stores

    def _project_silhouette(
        self,
        sq: Superquadric,
        R_cw: np.ndarray,
        t_cw: np.ndarray,
        H: int,
        W: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(silhouette_mask (H,W), depth (H,W))`` for an SQ."""
        # Sample SQ surface; project to image.
        surf = sq.surface_points(n_lat=self.n_lat, n_lon=self.n_lon)
        uv, depth = _world_to_pixel(surf, R_cw, t_cw, self.K)
        mask = np.zeros((H, W), dtype=bool)
        depth_img = np.full((H, W), np.inf)

        in_front = depth > 1e-3
        in_image = (
            (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        )
        good = in_front & in_image
        if not np.any(good):
            return mask, depth_img

        u_int = np.clip(uv[good, 0].astype(int), 0, W - 1)
        v_int = np.clip(uv[good, 1].astype(int), 0, H - 1)
        d_g = depth[good]

        # Splat: mark pixels covered by surface samples.  For the SQ
        # interior we use a convex-hull fill via the bounding rect of
        # the splatted points.  Cheaper than full polygon raster.
        mask[v_int, u_int] = True
        depth_img[v_int, u_int] = np.minimum(depth_img[v_int, u_int], d_g)

        if np.count_nonzero(mask) >= 4:
            v_idx, u_idx = np.where(mask)
            v_lo, v_hi = int(v_idx.min()), int(v_idx.max())
            u_lo, u_hi = int(u_idx.min()), int(u_idx.max())
            mask[v_lo : v_hi + 1, u_lo : u_hi + 1] = True
        return mask, depth_img


# ---------------------------------------------------------------------------
# 3. OceanOptics UV-VIS-NIR attributor (bore-sight ray cast)
# ---------------------------------------------------------------------------


@dataclass
class OceanOpticsAttributor:
    """Single-fiber spectrometer -> per-SQ spectrum + derived indices.

    Setup: collimating lens at the camera bore-sight, ~5-10 cm
    circular FoV, integrated 200-1100 nm in 1024-2048 channels.  Each
    spectrometer integration produces one ray cast (sensor origin +
    bore-sight direction, both in world coords) and the first-hit SQ
    receives the spectrum into its accumulator.

    The attributor also derives a small set of vegetation/water
    indices from the wavelength-resolved spectrum and pushes them
    into the same SQ's property accumulators -- so both the raw
    compressed spectrum *and* the indices are available for fusion.
    """

    index: SQSpatialIndex
    n_channels: int = 2048
    lambda_lo_nm: float = 200.0
    lambda_hi_nm: float = 1100.0
    bore_sight_t_max: float = 50.0  # meters; ignore far hits
    stores: Dict[str, SuperquadricPropertyStore] = field(default_factory=dict)

    def _store_for(self, sq: Superquadric) -> SuperquadricPropertyStore:
        s = self.stores.get(sq.id)
        if s is None:
            s = SuperquadricPropertyStore(sq_id=sq.id)
            s.spectrum = SpectrumAccumulator(
                n_channels=self.n_channels,
                lambda_lo_nm=self.lambda_lo_nm,
                lambda_hi_nm=self.lambda_hi_nm,
            )
            self.stores[sq.id] = s
        elif s.spectrum is None:
            s.spectrum = SpectrumAccumulator(
                n_channels=self.n_channels,
                lambda_lo_nm=self.lambda_lo_nm,
                lambda_hi_nm=self.lambda_hi_nm,
            )
        return s

    def attribute(
        self,
        spectrum: np.ndarray,
        wavelengths_nm: np.ndarray,
        *,
        bore_sight_origin: np.ndarray,
        bore_sight_direction: np.ndarray,
        weight: float = 1.0,
        derive_indices: bool = True,
    ) -> Optional[str]:
        """Cast bore-sight ray; on first hit, update the SQ's spectrum.

        Returns the hit SQ id, or None if no SQ was intersected.
        """
        s = np.asarray(spectrum, dtype=float).ravel()
        wl = np.asarray(wavelengths_nm, dtype=float).ravel()
        if s.size == 0 or s.size != wl.size:
            return None
        if not np.all(np.isfinite(s)):
            return None

        hit = self.index.first_hit_ray(
            bore_sight_origin, bore_sight_direction, t_max=self.bore_sight_t_max
        )
        if hit is None:
            return None
        sq_idx, _t = hit
        sq = self.index.sqs[int(sq_idx)]
        store = self._store_for(sq)
        store.spectrum.update(s, weight=weight)

        if derive_indices:
            self._update_derived_indices(store, s, wl)

        return sq.id

    def _update_derived_indices(
        self,
        store: SuperquadricPropertyStore,
        spectrum: np.ndarray,
        wavelengths_nm: np.ndarray,
    ) -> None:
        """Sample standard wavelength bands and push derived indices.

        Bands sampled (5-nm averages around band center):

        * Blue 475, Green 560, Red 668, RedEdge 717, NIR 842 nm
        """
        def _band_mean(center: float, half_width: float = 5.0) -> Optional[float]:
            mask = (wavelengths_nm >= center - half_width) & (
                wavelengths_nm <= center + half_width
            )
            if not np.any(mask):
                return None
            v = float(np.mean(spectrum[mask]))
            return v if math.isfinite(v) else None

        b = _band_mean(475.0)
        g = _band_mean(560.0)
        r = _band_mean(668.0)
        re = _band_mean(717.0)
        nir = _band_mean(842.0)

        if nir is not None and r is not None:
            store.update(
                PropertyId.NDVI,
                float(ndvi(np.array([nir]), np.array([r]))[0]),
            )
        if nir is not None and re is not None:
            store.update(
                PropertyId.NDRE,
                float(ndre(np.array([nir]), np.array([re]))[0]),
            )
        if nir is not None and g is not None:
            store.update(
                PropertyId.GNDVI,
                float(gndvi(np.array([nir]), np.array([g]))[0]),
            )
        if nir is not None and r is not None and b is not None:
            store.update(
                PropertyId.EVI,
                float(evi(np.array([nir]), np.array([r]), np.array([b]))[0]),
            )


__all__ = [
    "SQSpatialIndex",
    "LidarIntensityAttributor",
    "MicaSenseAttributor",
    "OceanOpticsAttributor",
    "ndvi",
    "ndre",
    "gndvi",
    "evi",
]
