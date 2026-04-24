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
    # How the adjacency in ``W`` was built. ``'knn'`` (the historical default)
    # uses Euclidean k-NN on building centroids; ``'road_knn'`` uses shortest-
    # path network distance on an OSM road graph snapped to those centroids.
    # Downstream spectral code only reads L/W, so this field is purely
    # descriptive; callers that care (e.g. HTTP analyzers) branch on it.
    graph_mode:  str = 'knn'
    # Optional road-aware metadata, populated only when graph_mode='road_knn'.
    road_meta:   dict = field(default_factory=dict, repr=False)


# ── disk cache helpers ─────────────────────────────────────────────────────

def _cache_path(place: str, cache_dir: Path) -> Path:
    safe = place.replace(',', '').replace(' ', '_').replace('/', '_')
    return cache_dir / f'buildings_{safe}.geojson'


def _compute_traits_inplace(gdf: 'gpd.GeoDataFrame') -> 'gpd.GeoDataFrame':
    """Project to local UTM, compute area/height/type/compacity traits and
    centroid columns on *gdf*. Shared by address- and bbox-based fetchers.

    Mutates the input GeoDataFrame and returns it for chaining convenience.
    """
    gdf = gdf[gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])].copy()
    if len(gdf) == 0:
        return gdf

    gdf = gdf.to_crs(gdf.estimate_utm_crs())

    gdf['area_m2']   = gdf.geometry.area
    gdf['perimeter'] = gdf.geometry.length
    gdf['compacity'] = gdf['perimeter']**2 / (4 * math.pi * gdf['area_m2'].clip(1e-3))

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
        return 3.2
    gdf['height_m'] = gdf.apply(_height, axis=1)

    def _type_enc(row) -> int:
        bt = row.get('building', 'yes')
        if not isinstance(bt, str):
            bt = 'yes'
        return _TYPE_MAP.get(bt.lower(), 0)
    gdf['type_enc'] = gdf.apply(_type_enc, axis=1)

    cents = gdf.geometry.centroid
    gdf['cx'] = cents.x
    gdf['cy'] = cents.y
    return gdf


def _bbox_cache_path(bbox: tuple[float, float, float, float],
                     cache_dir: Path) -> Path:
    """Cache key for a bbox: quantize to 1e-4° (~11 m) so neighbouring
    viewports hit the same cache entry."""
    south, west, north, east = bbox
    q = lambda v: f'{round(float(v), 4):+.4f}'
    safe = f'{q(south)}_{q(west)}_{q(north)}_{q(east)}'
    return cache_dir / f'buildings_bbox_{safe}.geojson'


def fetch_buildings_bbox(
    south: float,
    west: float,
    north: float,
    east: float,
    cache_dir: Path | str = Path('/tmp/kernelcal_urban_cache'),
    force_refresh: bool = False,
    timeout: int = 120,
) -> 'gpd.GeoDataFrame':
    """Download OSM building footprints inside a WGS84 bbox.

    The bbox mirrors CesiumJS's ``camera.computeViewRectangle`` convention:
    four floats in decimal degrees. All other behaviour (trait computation,
    UTM reprojection, disk cache) matches :func:`fetch_buildings`.

    Parameters
    ----------
    south, west, north, east : float
        Bounding box in degrees (EPSG:4326). Must satisfy
        ``south < north`` and ``west < east`` (no antimeridian wrap).
    cache_dir : Path | str
        Directory for .geojson cache files. Key is quantized to 1e-4° so
        nearby viewports share hits.
    force_refresh : bool
        Ignore cache and re-download.
    timeout : int
        osmnx HTTP timeout in seconds.

    Returns
    -------
    GeoDataFrame with columns: geometry (Polygon), area_m2, height_m,
    building_type, type_enc, compacity, cx, cy. Empty GeoDataFrame if
    no building polygons exist in the bbox (does *not* raise).

    Raises
    ------
    ImportError  : osmnx/geopandas missing.
    ValueError   : degenerate bbox (south >= north or west >= east).
    RuntimeError : osmnx/Overpass fetch failed after *timeout* seconds.
    """
    if not HAS_OSM:
        raise ImportError('osmnx and geopandas are required: pip install osmnx')
    if not (south < north and west < east):
        raise ValueError(
            f'Degenerate bbox: south={south}, west={west}, '
            f'north={north}, east={east}'
        )

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cp = _bbox_cache_path((south, west, north, east), cache_dir)

    if cp.exists() and not force_refresh:
        print(f'    [cache] Loading {cp.name}')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            gdf = gpd.read_file(cp)
        return gdf

    print(f'    [OSM] Fetching buildings in bbox '
          f'({south:.4f},{west:.4f})–({north:.4f},{east:.4f}) …')
    ox.settings.timeout = timeout
    # osmnx renamed the bbox-tuple convention between 1.x and 2.x:
    #   * 1.x  : bbox=(north, south, east, west)   -- same order as legacy positional
    #   * 2.x  : bbox=(left,  bottom, right, top)  -- i.e. (west, south, east, north)
    # Pick the right tuple based on the installed major version. This avoids
    # silently passing swapped lat/lon (which produces a degenerate polygon and
    # later raises "cannot convert float NaN to integer" inside osmnx).
    try:
        _osmnx_major = int(str(getattr(ox, '__version__', '1.0')).split('.')[0])
    except (ValueError, AttributeError):
        _osmnx_major = 1
    if _osmnx_major >= 2:
        bbox_arg = (west, south, east, north)   # left, bottom, right, top
    else:
        bbox_arg = (north, south, east, west)   # legacy
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            try:
                gdf = ox.features_from_bbox(
                    bbox=bbox_arg, tags={'building': True},
                )
            except TypeError:
                # Very old osmnx (<1.7) lacked the `bbox` kwarg.
                gdf = ox.features_from_bbox(
                    north, south, east, west,
                    tags={'building': True},
                )
        except Exception as exc:
            raise RuntimeError(
                f'osmnx bbox fetch failed for ({south},{west},{north},{east}): {exc}'
            ) from exc

    gdf = _compute_traits_inplace(gdf)
    if len(gdf) == 0:
        # Write an empty sentinel so repeated empty viewports don't re-hit OSM
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            gpd.GeoDataFrame(geometry=[]).to_file(cp, driver='GeoJSON')
        print(f'    [cache] No buildings; wrote empty sentinel → {cp.name}')
        return gdf

    keep_cols = ['geometry', 'area_m2', 'height_m', 'type_enc',
                 'compacity', 'cx', 'cy']
    save_cols = [c for c in keep_cols if c in gdf.columns]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        gdf[save_cols].to_file(cp, driver='GeoJSON')
    print(f'    [cache] Saved {len(gdf)} buildings → {cp.name}')
    return gdf


# ── road-graph fetch (option 1: road-aware proximity) ─────────────────────

def _road_cache_path(bbox: tuple[float, float, float, float],
                     network_type: str,
                     simplify: bool,
                     cache_dir: Path) -> Path:
    """Cache key for an OSM road graph bbox, quantized to 1e-4° (~11 m).

    ``network_type`` and ``simplify`` are part of the key so drive / walk /
    simplified graphs don't collide.
    """
    south, west, north, east = bbox
    q = lambda v: f'{round(float(v), 4):+.4f}'
    safe = (f'{q(south)}_{q(west)}_{q(north)}_{q(east)}_'
            f'{network_type}_s{int(bool(simplify))}')
    return cache_dir / f'roads_bbox_{safe}.graphml'


def fetch_road_graph_bbox(
    south: float,
    west: float,
    north: float,
    east: float,
    network_type: str = 'drive',
    simplify: bool = True,
    cache_dir: Path | str = Path('/tmp/kernelcal_urban_cache'),
    force_refresh: bool = False,
    timeout: int = 120,
):
    """Download an OSM road network in a WGS84 bbox, UTM-projected & undirected.

    Companion to :func:`fetch_buildings_bbox` for the ``graph_mode='road_knn'``
    branch of :func:`buildings_to_graph_via_roads`. All osmnx behaviour
    (bbox-tuple ordering across 1.x/2.x, disk cache) mirrors that function.

    Parameters
    ----------
    south, west, north, east : float
        Bounding box in decimal degrees (EPSG:4326). ``south < north`` and
        ``west < east`` required.
    network_type : str
        Forwarded to ``ox.graph_from_bbox``. One of
        ``'drive' | 'drive_service' | 'walk' | 'bike' | 'all' | 'all_private'``.
        Default ``'drive'``; use ``'all'`` in medinas / pedestrian-dominant
        fabrics (Venice, Marrakech) where footpaths matter.
    simplify : bool
        osmnx topology simplification — collapses degree-2 interstitial nodes.
        Keep True for faster Dijkstra; set False if you need fine-grained
        snap distances.
    cache_dir : Path | str
        Directory for .graphml cache files.
    force_refresh : bool
        Ignore cache and re-download.
    timeout : int
        osmnx HTTP timeout (seconds).

    Returns
    -------
    networkx.MultiGraph or None
        Undirected multigraph with edge attribute ``'length'`` (metres), node
        attributes ``'x'`` / ``'y'`` in local UTM metres. Returns ``None`` if
        the bbox has no road segments. Node attributes ``lon`` / ``lat`` are
        copied from the unprojected graph so callers can redisplay in WGS84.

    Raises
    ------
    ImportError  : osmnx / geopandas / networkx missing.
    ValueError   : degenerate bbox.
    RuntimeError : osmnx fetch failed.
    """
    if not HAS_OSM:
        raise ImportError('osmnx and geopandas are required: pip install osmnx')
    if not (south < north and west < east):
        raise ValueError(
            f'Degenerate bbox: south={south}, west={west}, '
            f'north={north}, east={east}'
        )
    # networkx is a hard transitive dep of osmnx but import here for clarity.
    import networkx as nx

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cp = _road_cache_path(
        (south, west, north, east), network_type, simplify, cache_dir,
    )

    if cp.exists() and not force_refresh:
        print(f'    [cache] Loading {cp.name}')
        try:
            G_cached = ox.load_graphml(cp)
        except Exception as exc:
            print(f'    [cache] Failed to read {cp.name} ({exc}); re-fetching')
        else:
            # Empty-graph sentinel: file was written for a bbox with no roads.
            if G_cached.number_of_nodes() == 0:
                return None
            return G_cached

    print(f'    [OSM] Fetching road network ({network_type}) in bbox '
          f'({south:.4f},{west:.4f})–({north:.4f},{east:.4f}) …')
    ox.settings.timeout = timeout
    # Same 1.x / 2.x bbox-tuple-order quirk as fetch_buildings_bbox.
    try:
        _osmnx_major = int(str(getattr(ox, '__version__', '1.0')).split('.')[0])
    except (ValueError, AttributeError):
        _osmnx_major = 1
    if _osmnx_major >= 2:
        bbox_arg = (west, south, east, north)   # left, bottom, right, top
    else:
        bbox_arg = (north, south, east, west)   # legacy

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            try:
                G = ox.graph_from_bbox(
                    bbox=bbox_arg,
                    network_type=network_type,
                    simplify=simplify,
                    retain_all=False,
                )
            except TypeError:
                # Pre-1.7 osmnx: positional (north, south, east, west).
                G = ox.graph_from_bbox(
                    north, south, east, west,
                    network_type=network_type,
                    simplify=simplify,
                    retain_all=False,
                )
        except (ValueError, nx.NetworkXPointlessConcept) as exc:
            # osmnx raises ValueError / NetworkXPointlessConcept when the
            # query returns no streets — legitimate empty result, cache
            # sentinel so we don't re-hit Overpass.
            print(f'    [OSM] No road network in bbox: {exc}')
            empty = nx.MultiGraph()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                ox.save_graphml(empty, cp)
            return None
        except Exception as exc:
            raise RuntimeError(
                f'osmnx road-graph fetch failed for '
                f'({south},{west},{north},{east}): {exc}'
            ) from exc

    # Collapse direction + parallel edges into a MultiGraph keyed by minimum
    # 'length'. Some osmnx versions put it on ox.convert, older on
    # ox.utils_graph; fall back to a best-effort nx.Graph cast.
    try:
        G = ox.convert.to_undirected(G)
    except AttributeError:
        try:
            G = ox.utils_graph.get_undirected(G)
        except AttributeError:
            G = G.to_undirected()

    # Stash WGS84 coords on nodes before projecting, so HTTP callers can
    # still render the graph on a 2D basemap without round-tripping through
    # geopandas.
    for n, data in G.nodes(data=True):
        if 'x' in data and 'y' in data and 'lon' not in data:
            data['lon'] = data['x']
            data['lat'] = data['y']

    # Project to a local UTM so edge 'length' attributes and Dijkstra distances
    # agree with the UTM coordinates used by _compute_traits_inplace.
    try:
        G = ox.project_graph(G)
    except Exception as exc:
        raise RuntimeError(f'ox.project_graph failed: {exc}') from exc

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ox.save_graphml(G, cp)
    print(f'    [cache] Saved road graph '
          f'(N={G.number_of_nodes()}, E={G.number_of_edges()}) → {cp.name}')
    return G


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

    gdf = _compute_traits_inplace(gdf)
    if len(gdf) == 0:
        raise ValueError(f'No building polygons found for: {place}')

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


def buildings_to_graph_from_bbox(
    south: float,
    west:  float,
    north: float,
    east:  float,
    name:  str | None = None,
    k:     int   = 8,
    n_max: int   = 1500,
    sigma_frac: float = 0.05,
    cache_dir: Path | str = Path('/tmp/kernelcal_urban_cache'),
    force_refresh: bool = False,
    timeout: int = 120,
    seed:  int   = 42,
) -> CityGraph | None:
    """Convenience one-shot: bbox → OSM fetch → proximity graph → CityGraph.

    Thin wrapper that chains :func:`fetch_buildings_bbox` into
    :func:`buildings_to_graph`. Intended for viewport-driven callers (e.g.,
    a live Cesium client POSTing ``camera.computeViewRectangle`` extents).

    Returns
    -------
    CityGraph on success, or ``None`` if the bbox contains no buildings.
    """
    gdf = fetch_buildings_bbox(
        south, west, north, east,
        cache_dir=cache_dir,
        force_refresh=force_refresh,
        timeout=timeout,
    )
    if len(gdf) == 0:
        return None

    label = name or f'bbox_{south:.4f}_{west:.4f}_{north:.4f}_{east:.4f}'
    place = f'bbox(S={south:.4f}, W={west:.4f}, N={north:.4f}, E={east:.4f})'
    return buildings_to_graph(
        gdf, name=label, place=place,
        k=k, n_max=n_max, sigma_frac=sigma_frac, seed=seed,
    )


# ── road-aware graph construction (option 1) ──────────────────────────────

def _snap_buildings_to_road_nodes(
    x: np.ndarray,
    y: np.ndarray,
    G_roads,
) -> tuple[np.ndarray, np.ndarray]:
    """Snap (x, y) in UTM metres to the nearest road node id in *G_roads*.

    Returns ``(snap_node_ids, snap_offsets_m)`` where ``snap_offsets_m`` is
    the straight-line metric offset from each centroid to its snap node.
    The road graph is assumed to already be in the same UTM CRS.
    """
    # osmnx.distance.nearest_nodes accepts array-like X/Y in the graph's CRS.
    # return_dist=True gives us the snap offset we add to network distance.
    try:
        nodes, offsets = ox.distance.nearest_nodes(
            G_roads, X=x, Y=y, return_dist=True,
        )
    except TypeError:
        # Very old osmnx signature: nearest_nodes(G, X, Y) returned only nodes.
        nodes = ox.distance.nearest_nodes(G_roads, X=x, Y=y)
        # Fall back to KD-tree snap offsets from node coordinates.
        node_xy = np.array([
            [G_roads.nodes[n].get('x', 0.0), G_roads.nodes[n].get('y', 0.0)]
            for n in nodes
        ])
        offsets = np.hypot(x - node_xy[:, 0], y - node_xy[:, 1]).tolist()
    return np.asarray(nodes), np.asarray(offsets, dtype=float)


def _network_distance_matrix(
    snap_nodes: np.ndarray,
    snap_offsets: np.ndarray,
    G_roads,
    cutoff_m: float | None,
) -> np.ndarray:
    """Compute pairwise building-to-building network distance in metres.

    For each unique snap node, runs a single-source Dijkstra out to
    ``cutoff_m`` once and caches the result — so the cost scales with the
    number of *distinct* snap nodes, not with N². Distances follow
    ``d(i, j) = snap_offsets[i] + path_len(snap[i], snap[j]) + snap_offsets[j]``
    and are symmetrized since road traffic direction is ignored here.

    Unreachable pairs (or pairs beyond the cutoff) receive ``np.inf`` so the
    k-NN ranker can skip them cleanly.
    """
    import networkx as nx

    n = len(snap_nodes)
    d_net = np.full((n, n), np.inf, dtype=float)
    np.fill_diagonal(d_net, 0.0)

    cache: dict = {}
    for i in range(n):
        s_i = snap_nodes[i]
        if s_i not in cache:
            try:
                cache[s_i] = nx.single_source_dijkstra_path_length(
                    G_roads, s_i, cutoff=cutoff_m, weight='length',
                )
            except nx.NodeNotFound:
                # Snap node fell outside retain_all=False connected component.
                cache[s_i] = {s_i: 0.0}
        lengths = cache[s_i]
        off_i = snap_offsets[i]
        for j in range(n):
            if i == j:
                continue
            s_j = snap_nodes[j]
            if s_j in lengths:
                d_net[i, j] = off_i + lengths[s_j] + snap_offsets[j]

    # Symmetrize (undirected Dijkstra + cutoff can still leave A→B finite
    # while B→A is beyond cutoff via a longer route).
    d_sym = np.minimum(d_net, d_net.T)
    return d_sym


def buildings_to_graph_via_roads(
    gdf:    'gpd.GeoDataFrame',
    G_roads,
    name:   str,
    place:  str,
    k:      int = 8,
    n_max:  int = 1500,
    sigma_frac: float = 0.05,
    max_network_dist: float | None = None,
    seed:   int = 42,
) -> CityGraph:
    """Build k-NN proximity graph using **road-network distance** between
    building centroids.

    Mirrors :func:`buildings_to_graph` but replaces the Euclidean k-NN with
    shortest-path network distance on *G_roads* (expected UTM-projected,
    undirected, with ``'length'`` edge weights — exactly what
    :func:`fetch_road_graph_bbox` returns).

    Algorithm
    ---------
    1. Extract / clean centroids as in :func:`buildings_to_graph`.
    2. Snap each centroid to the nearest road node; record the metric
       snap offset.
    3. For each unique snap node, run a single-source Dijkstra out to
       ``max_network_dist`` (defaults to ``5·sigma``) and cache the
       reachable-node distance map.
    4. Build the pairwise distance matrix
       ``d_ij = offset_i + path(snap_i, snap_j) + offset_j`` (symmetrized).
    5. For each row, take the k smallest finite entries and assign Gaussian
       weights ``exp(-d² / σ²)``. σ is computed the same way as in
       :func:`buildings_to_graph` so spectra are directly comparable across
       ``graph_mode`` values.
    6. Eigendecompose the resulting Laplacian.

    Parameters
    ----------
    max_network_dist : float | None
        Dijkstra cutoff in metres. ``None`` → ``5·sigma`` (a good match to
        the Gaussian tail). Larger values increase connectivity but raise
        Dijkstra cost.

    Notes
    -----
    * Nodes whose snap node sits in an unreachable road component simply
      end up with fewer than k neighbours (or, in pathological cases,
      isolated) — the Laplacian then has extra zero modes and ``β₀``
      increases. This is the desired behaviour: informal settlements
      genuinely have disconnected sub-blocks and the spectrum should
      reflect that.
    * Returned :class:`CityGraph` has ``graph_mode='road_knn'`` and a
      populated ``road_meta`` dict so HTTP analyzers can surface snap /
      connectivity stats without re-running Dijkstra.
    """
    # ── trait extraction (mirrors buildings_to_graph; DRY via helper?) ────
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

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(areas)
    x, y   = x[valid], y[valid]
    areas  = areas[valid]
    heights = heights[valid]
    type_enc = type_enc[valid]
    compacity = compacity[valid]

    if len(x) > n_max:
        top_idx = np.argsort(areas)[::-1][:n_max]
        x, y   = x[top_idx], y[top_idx]
        areas  = areas[top_idx]
        heights = heights[top_idx]
        type_enc = type_enc[top_idx]
        compacity = compacity[top_idx]

    n = len(x)
    positions = np.column_stack([x, y])

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    diag = math.hypot(xmax - xmin, ymax - ymin)

    # σ chosen identically to the Euclidean path so the spectra are
    # comparable across graph_mode values.
    tree = cKDTree(positions)
    nn_dists, _ = tree.query(positions, k=min(2, n))
    median_nn = float(np.median(nn_dists[:, -1])) if n > 1 else 1.0
    sigma = max(sigma_frac * max(diag, 1.0), 2 * max(median_nn, 1e-3))

    # ── road snap + pairwise network distance ─────────────────────────────
    snap_nodes, snap_offsets = _snap_buildings_to_road_nodes(x, y, G_roads)
    cutoff = float(max_network_dist) if max_network_dist is not None else 5.0 * sigma
    d_net = _network_distance_matrix(snap_nodes, snap_offsets, G_roads, cutoff_m=cutoff)

    # ── k-NN from pairwise network distances ─────────────────────────────
    W = np.zeros((n, n), dtype=float)
    n_edges_added = 0
    n_isolated    = 0
    for i in range(n):
        row = d_net[i].copy()
        row[i] = np.inf
        # Top-k by smallest finite distance.
        order = np.argsort(row)
        picked = 0
        for j in order:
            d = row[j]
            if not np.isfinite(d):
                break
            if picked >= k:
                break
            w = math.exp(-d * d / (sigma * sigma))
            if w > W[i, j]:
                W[i, j] = w
                W[j, i] = w
                n_edges_added += 1
            picked += 1
        if picked == 0:
            n_isolated += 1

    D = np.diag(W.sum(axis=1))
    L = D - W

    eigvals, eigvecs = scipy_eigh(L)
    eigvals = np.maximum(eigvals, 0.0)

    def _norm(v: np.ndarray) -> np.ndarray:
        lo, hi = v.min(), v.max()
        return (v - lo) / (hi - lo + 1e-12)

    traits = np.column_stack([
        _norm(areas),
        _norm(heights),
        _norm(type_enc),
        _norm(np.clip(compacity, 0, 20)),
    ])

    # Count reachable pairs for diagnostics (exclude self).
    reachable = np.isfinite(d_net)
    np.fill_diagonal(reachable, False)
    n_reachable_pairs = int(reachable.sum())  # directed count (both orders)

    road_meta = {
        'n_road_nodes':     int(G_roads.number_of_nodes()),
        'n_road_edges':     int(G_roads.number_of_edges()),
        'snap_offset_m':    {
            'mean':   float(np.mean(snap_offsets)) if n else 0.0,
            'median': float(np.median(snap_offsets)) if n else 0.0,
            'max':    float(np.max(snap_offsets)) if n else 0.0,
        },
        'cutoff_m':             float(cutoff),
        'n_isolated_buildings': int(n_isolated),
        'n_reachable_pairs':    int(n_reachable_pairs // 2),
        'unique_snap_nodes':    int(len(set(snap_nodes.tolist()))),
    }

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
        graph_mode='road_knn',
        road_meta=road_meta,
    )


def buildings_to_graph_via_roads_from_bbox(
    south: float,
    west:  float,
    north: float,
    east:  float,
    name:  str | None = None,
    k:     int   = 8,
    n_max: int   = 1500,
    sigma_frac: float = 0.05,
    network_type: str = 'drive',
    simplify: bool = True,
    max_network_dist: float | None = None,
    cache_dir: Path | str = Path('/tmp/kernelcal_urban_cache'),
    force_refresh: bool = False,
    timeout: int = 120,
    seed:  int   = 42,
) -> CityGraph | None:
    """Convenience one-shot: bbox → buildings + road graph → road-aware CityGraph.

    Fetches both building footprints and the OSM road network for the same
    WGS84 bbox, then hands them to :func:`buildings_to_graph_via_roads`.
    Intended as the ``graph_mode='road_knn'`` backing for DeepGIS-XR's
    viewport analyzer.

    Returns
    -------
    CityGraph on success; ``None`` if the bbox has no buildings. If the
    bbox has buildings but no road network, falls back to the Euclidean
    :func:`buildings_to_graph` so the caller still gets a valid spectrum
    (the returned ``CityGraph.graph_mode`` is then ``'knn'`` and
    ``road_meta['fallback_reason']`` explains why).
    """
    gdf = fetch_buildings_bbox(
        south, west, north, east,
        cache_dir=cache_dir,
        force_refresh=force_refresh,
        timeout=timeout,
    )
    if len(gdf) == 0:
        return None

    G_roads = fetch_road_graph_bbox(
        south, west, north, east,
        network_type=network_type,
        simplify=simplify,
        cache_dir=cache_dir,
        force_refresh=force_refresh,
        timeout=timeout,
    )

    label = name or f'bbox_{south:.4f}_{west:.4f}_{north:.4f}_{east:.4f}'
    place = f'bbox(S={south:.4f}, W={west:.4f}, N={north:.4f}, E={east:.4f})'

    if G_roads is None or G_roads.number_of_nodes() == 0:
        # Degrade gracefully rather than 500. Caller can inspect
        # graph_mode / road_meta to surface a "no roads in viewport" banner.
        cg = buildings_to_graph(
            gdf, name=label, place=place,
            k=k, n_max=n_max, sigma_frac=sigma_frac, seed=seed,
        )
        cg.road_meta = {
            'fallback_reason': 'no_road_network',
            'requested_network_type': network_type,
        }
        return cg

    return buildings_to_graph_via_roads(
        gdf, G_roads,
        name=label, place=place,
        k=k, n_max=n_max, sigma_frac=sigma_frac,
        max_network_dist=max_network_dist, seed=seed,
    )
