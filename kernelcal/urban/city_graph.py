"""kernelcal.urban.city_graph — OSM building graph construction.

Pipeline
--------
1. fetch_buildings(place)       — download OSM building footprints via osmnx
                                   (with disk cache so repeated runs are free)
2. buildings_to_graph(gdf, ...) — build k-NN proximity graph on centroids
                                   with Gaussian edge weights
3. CityGraph dataclass          — bundles Laplacian, positions, and traits
                                   for downstream kernelcal diagnostics

Node traits extracted per building
-----------------------------------
  area_m2  : footprint area in m²  (always available)
  height_m : building height in m  (from OSM 'height' or 'building:levels')
  type_enc : integer-encoded building use  (residential=0, commercial=1, ...)
  compacity: perimeter² / (4π·area)  — shape factor (1 = circle, >1 = elongated)

Physical interpretation
-----------------------
In the P4 framework each building is a "semantic object" like a rock or
a channel node.  The building's traits are the "matter field" cₗ.
The proximity graph encodes the spatial substrate the controller acts on.
A planned city (Phoenix grid, Barcelona Eixample) should produce a
fixed-point kernel h* more concentrated than vacuum — ΔH < 0 — because
a low-mode zoning controller imposed large-scale order.  An informal
settlement should have ΔH ≈ 0.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.linalg import eigh as scipy_eigh

# ── optional heavy imports (guard for environments without geopandas) ──────
try:
    import osmnx as ox
    import geopandas as gpd
    from shapely.geometry import box as shapely_box
    HAS_OSM = True
except Exception:
    HAS_OSM = False


# ── building type encoding ─────────────────────────────────────────────────
_TYPE_MAP = {
    'residential': 0, 'house': 0, 'apartments': 0, 'detached': 0,
    'semidetached_house': 0, 'terrace': 0, 'bungalow': 0,
    'commercial': 1, 'retail': 1, 'office': 1, 'supermarket': 1,
    'shop': 1, 'mall': 1,
    'industrial': 2, 'warehouse': 2, 'factory': 2,
    'civic': 3, 'public': 3, 'school': 3, 'hospital': 3,
    'church': 3, 'mosque': 3, 'temple': 3, 'cathedral': 3,
    'yes': 0,    # unknown → treat as residential
}


@dataclass
class CityGraph:
    """Graph derived from OSM building footprints.

    Attributes
    ----------
    name        : city label
    place       : OSM query string used
    positions   : (N, 2) float — projected x, y in metres (local UTM)
    traits      : (N, 4) float — [area_m2, height_m, type_enc, compacity]
                  all columns normalised to [0, 1] range
    L           : (N, N) graph Laplacian
    W           : (N, N) adjacency (symmetric, non-negative)
    eigvals     : (N,) eigenvalues of L
    eigvecs     : (N, N) eigenvectors
    n_buildings : total buildings downloaded (before subsampling)
    bounds_m    : (xmin, ymin, xmax, ymax) in metres
    """
    name:        str
    place:       str
    positions:   np.ndarray
    traits:      np.ndarray
    L:           np.ndarray
    W:           np.ndarray
    eigvals:     np.ndarray
    eigvecs:     np.ndarray
    n_buildings: int
    bounds_m:    tuple[float, float, float, float]
    raw_gdf:     object = field(default=None, repr=False)  # GeoDataFrame


# ── disk cache helpers ─────────────────────────────────────────────────────

def _cache_path(place: str, cache_dir: Path) -> Path:
    safe = place.replace(',', '').replace(' ', '_').replace('/', '_')
    return cache_dir / f'buildings_{safe}.geojson'


def fetch_buildings(
    place: str,
    cache_dir: Path | str = Path('/tmp/kernelcal_urban_cache'),
    force_refresh: bool = False,
    timeout: int = 120,
    dist: int = 600,
) -> 'gpd.GeoDataFrame':
    """Download OSM building footprints within *dist* metres of *place*.

    Parameters
    ----------
    place         : address string passed to Nominatim geocoder
    cache_dir     : directory for .geojson cache files
    force_refresh : ignore cache and re-download
    timeout       : osmnx HTTP timeout in seconds
    dist          : radius in metres around the geocoded point (default 600 m)

    Returns
    -------
    GeoDataFrame with columns: geometry (Polygon), area_m2, height_m,
    building_type, type_enc, compacity
    """
    if not HAS_OSM:
        raise ImportError('osmnx and geopandas are required: pip install osmnx')

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cp = _cache_path(place, cache_dir)

    if cp.exists() and not force_refresh:
        print(f'    [cache] Loading {cp.name}')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            gdf = gpd.read_file(cp)
        return gdf

    print(f'    [OSM] Fetching buildings within {dist}m of: {place} …')
    ox.settings.timeout = timeout
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            gdf = ox.features_from_address(
                place, tags={'building': True}, dist=dist
            )
        except Exception as exc:
            raise RuntimeError(f'osmnx fetch failed for "{place}": {exc}') from exc

    # Keep only polygonal footprints
    gdf = gdf[gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])].copy()
    if len(gdf) == 0:
        raise ValueError(f'No building polygons found for: {place}')

    # Project to UTM for metric calculations
    gdf = gdf.to_crs(gdf.estimate_utm_crs())

    # Compute traits
    gdf['area_m2']   = gdf.geometry.area
    gdf['perimeter'] = gdf.geometry.length
    gdf['compacity'] = gdf['perimeter']**2 / (4 * math.pi * gdf['area_m2'].clip(1e-3))

    # Height: prefer 'height' field, then estimate from levels
    def _height(row) -> float:
        h = row.get('height', None)
        if h is not None:
            try:
                return float(str(h).split()[0])
            except Exception:
                pass
        lv = row.get('building:levels', None)
        if lv is not None:
            try:
                return float(lv) * 3.2
            except Exception:
                pass
        return 3.2   # default single-storey

    gdf['height_m'] = gdf.apply(_height, axis=1)

    # Building type encoding
    def _type_enc(row) -> int:
        bt = row.get('building', 'yes')
        if not isinstance(bt, str):
            bt = 'yes'
        return _TYPE_MAP.get(bt.lower(), 0)

    gdf['type_enc'] = gdf.apply(_type_enc, axis=1)

    # Centroid
    cents = gdf.geometry.centroid
    gdf['cx'] = cents.x
    gdf['cy'] = cents.y

    # Save cache (only geometry + computed columns)
    keep_cols = ['geometry', 'area_m2', 'height_m', 'type_enc',
                 'compacity', 'cx', 'cy']
    save_cols = [c for c in keep_cols if c in gdf.columns]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        gdf[save_cols].to_file(cp, driver='GeoJSON')
    print(f'    [cache] Saved {len(gdf)} buildings → {cp.name}')
    return gdf


# ── graph construction ─────────────────────────────────────────────────────

def buildings_to_graph(
    gdf:       'gpd.GeoDataFrame',
    name:      str,
    place:     str,
    k:         int   = 8,
    n_max:     int   = 2000,
    sigma_frac: float = 0.05,
    seed:      int   = 42,
) -> CityGraph:
    """Build k-NN proximity graph from building centroids.

    Parameters
    ----------
    gdf        : GeoDataFrame from fetch_buildings()
    name, place: labels
    k          : number of nearest neighbours per node
    n_max      : maximum nodes (subsample by area — keep largest buildings)
    sigma_frac : Gaussian width = sigma_frac × domain diagonal
    seed       : RNG seed for reproducible subsample

    Returns
    -------
    CityGraph with L, W, eigvals, eigvecs populated
    """
    # Extract centroid positions
    if 'cx' not in gdf.columns:
        cents = gdf.geometry.centroid
        x = cents.x.values
        y = cents.y.values
    else:
        x = gdf['cx'].values.astype(float)
        y = gdf['cy'].values.astype(float)

    areas     = gdf['area_m2'].values.astype(float) if 'area_m2' in gdf.columns \
                else np.ones(len(gdf))
    heights   = gdf['height_m'].values.astype(float) if 'height_m' in gdf.columns \
                else np.full(len(gdf), 3.2)
    type_enc  = gdf['type_enc'].values.astype(float) if 'type_enc' in gdf.columns \
                else np.zeros(len(gdf))
    compacity = gdf['compacity'].values.astype(float) if 'compacity' in gdf.columns \
                else np.ones(len(gdf))

    n_total = len(gdf)

    # Remove NaN centroids
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(areas)
    x, y   = x[valid], y[valid]
    areas  = areas[valid]
    heights = heights[valid]
    type_enc = type_enc[valid]
    compacity = compacity[valid]

    # Subsample to n_max by keeping largest buildings (most spatially significant)
    if len(x) > n_max:
        top_idx = np.argsort(areas)[::-1][:n_max]
        x, y   = x[top_idx], y[top_idx]
        areas  = areas[top_idx]
        heights = heights[top_idx]
        type_enc = type_enc[top_idx]
        compacity = compacity[top_idx]

    n = len(x)
    positions = np.column_stack([x, y])

    # Bounds
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    diag = math.hypot(xmax - xmin, ymax - ymin)

    # Adaptive sigma: fraction of domain diagonal, floor at median NN dist
    tree = cKDTree(positions)
    nn_dists, _ = tree.query(positions, k=min(2, n))
    median_nn = float(np.median(nn_dists[:, -1])) if n > 1 else 1.0
    sigma = max(sigma_frac * max(diag, 1.0), 2 * max(median_nn, 1e-3))

    # Build k-NN adjacency
    dists, inds = tree.query(positions, k=k + 1)   # col 0 = self
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j_idx in range(1, k + 1):
            j   = inds[i, j_idx]
            d   = dists[i, j_idx]
            w   = math.exp(-d**2 / sigma**2)
            if w > W[i, j]:
                W[i, j] = w
                W[j, i] = w

    D = np.diag(W.sum(axis=1))
    L = D - W

    # Eigenpairs
    eigvals, eigvecs = scipy_eigh(L)
    eigvals = np.maximum(eigvals, 0.0)

    # Normalise traits to [0, 1]
    def _norm(v: np.ndarray) -> np.ndarray:
        lo, hi = v.min(), v.max()
        return (v - lo) / (hi - lo + 1e-12)

    traits = np.column_stack([
        _norm(areas),
        _norm(heights),
        _norm(type_enc),
        _norm(np.clip(compacity, 0, 20)),
    ])

    return CityGraph(
        name=name,
        place=place,
        positions=positions,
        traits=traits,
        L=L,
        W=W,
        eigvals=eigvals,
        eigvecs=eigvecs,
        n_buildings=n_total,
        bounds_m=(xmin, ymin, xmax, ymax),
        raw_gdf=gdf,
    )
