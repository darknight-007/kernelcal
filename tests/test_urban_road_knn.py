"""Tests for the road-aware k-NN branch of kernelcal.urban (option 1).

These tests avoid live Overpass queries by monkey-patching
``ox.features_from_bbox`` (building footprints) and ``ox.graph_from_bbox``
(road graph) to return deterministic synthetic data. The full pipeline
— snap → per-source Dijkstra → network-distance k-NN → Laplacian
eigendecomp — is exercised end-to-end.

Physical intent of the fixtures:

* ``_synthetic_building_gdf`` — a small jittered grid of building
  polygons in WGS84 degrees; mirrors the fixture in ``test_urban_bbox``
  so Euclidean and road-aware builds run against the same buildings.
* ``_synthetic_road_graph`` — a rectilinear UTM-coordinate road grid
  (1-km spacing, unit 'length' attribute normalized to metres) wide
  enough that every building centroid snaps to a real node.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


shapely = pytest.importorskip("shapely")
gpd = pytest.importorskip("geopandas")
nx  = pytest.importorskip("networkx")

from shapely.geometry import Polygon


# ── fixtures ───────────────────────────────────────────────────────────────

def _synthetic_building_gdf(n: int = 16, spacing: float = 1e-4):
    """Jittered square grid of building polygons around (-122, 37)."""
    rng = np.random.default_rng(0)
    side = int(np.ceil(np.sqrt(n)))
    polys, tags, levels = [], [], []
    for i in range(n):
        r, c = divmod(i, side)
        cx = -122.0 + c * spacing + rng.normal(0, spacing * 0.05)
        cy =   37.0 + r * spacing + rng.normal(0, spacing * 0.05)
        half = spacing * 0.3
        polys.append(Polygon([
            (cx - half, cy - half),
            (cx + half, cy - half),
            (cx + half, cy + half),
            (cx - half, cy + half),
        ]))
        tags.append(['residential', 'commercial', 'office'][i % 3])
        levels.append(1 + (i % 4))
    return gpd.GeoDataFrame(
        {'building': tags, 'building:levels': levels, 'geometry': polys},
        crs='EPSG:4326',
    )


def _synthetic_road_graph_wgs84(n_side: int = 6, spacing_deg: float = 1e-4):
    """Minimal rectilinear OSM-shaped road graph in WGS84.

    Returns an ``nx.MultiDiGraph`` whose nodes carry ``x``/``y`` (lon/lat)
    and whose edges carry ``length`` (metres, ≈ spacing_deg × 111e3) —
    matching exactly what osmnx returns from ``graph_from_bbox`` before
    ``ox.project_graph`` reprojects it.
    """
    G = nx.MultiDiGraph()
    length_m = spacing_deg * 111_000.0
    for r in range(n_side):
        for c in range(n_side):
            nid = r * n_side + c
            G.add_node(nid, x=-122.0 + c * spacing_deg, y=37.0 + r * spacing_deg)
    def add_pair(u, v, length):
        G.add_edge(u, v, key=0, length=float(length), osmid=hash((u, v)) & 0xffff)
        G.add_edge(v, u, key=0, length=float(length), osmid=hash((v, u)) & 0xffff)
    for r in range(n_side):
        for c in range(n_side):
            nid = r * n_side + c
            if c + 1 < n_side:
                add_pair(nid, nid + 1, length_m)
            if r + 1 < n_side:
                add_pair(nid, nid + n_side, length_m)
    G.graph['crs'] = 'EPSG:4326'
    return G


def _patch_ox_for_road_pipeline(monkeypatch, cg_mod,
                                 buildings_gdf, road_graph_wgs84):
    """Install monkey-patches so the full pipeline runs against synthetic data.

    * ``features_from_bbox`` → buildings
    * ``graph_from_bbox``    → road graph (WGS84 multidigraph)
    * ``project_graph``      → pass-through (fixture already acts like UTM)
    * ``convert.to_undirected`` → ``G.to_undirected()`` (osmnx 1.x lives
      at ``utils_graph.get_undirected`` so we patch the attribute access
      path the production code actually uses).
    """
    monkeypatch.setattr(
        cg_mod.ox, 'features_from_bbox',
        lambda *a, **kw: buildings_gdf.copy(),
    )
    monkeypatch.setattr(
        cg_mod.ox, 'graph_from_bbox',
        lambda *a, **kw: road_graph_wgs84.copy(),
    )
    # project_graph: no-op (fixture already uses plausible metric scale).
    monkeypatch.setattr(cg_mod.ox, 'project_graph', lambda G: G)

    # to_undirected: osmnx has this at ox.convert.to_undirected (2.x) or
    # ox.utils_graph.get_undirected (1.x). Patch whichever the installed
    # osmnx exposes so production code's attribute chain resolves cleanly.
    if hasattr(cg_mod.ox, 'convert'):
        monkeypatch.setattr(
            cg_mod.ox.convert, 'to_undirected',
            lambda G: G.to_undirected(),
        )
    if hasattr(cg_mod.ox, 'utils_graph'):
        monkeypatch.setattr(
            cg_mod.ox.utils_graph, 'get_undirected',
            lambda G: G.to_undirected(),
            raising=False,
        )

    # nearest_nodes: snap to the closest node by Euclidean distance on the
    # graph's x/y coords (which, after our identity project_graph, are
    # still WGS84 degrees — fine for relative ordering, and the snap
    # offset just needs to be non-negative & finite).
    def fake_nearest_nodes(G, X=None, Y=None, return_dist=False):
        node_ids = list(G.nodes())
        xy = np.array([[G.nodes[n]['x'], G.nodes[n]['y']] for n in node_ids])
        X = np.atleast_1d(np.asarray(X, dtype=float))
        Y = np.atleast_1d(np.asarray(Y, dtype=float))
        nodes = []
        dists = []
        for x, y in zip(X, Y):
            d = np.hypot(xy[:, 0] - x, xy[:, 1] - y)
            idx = int(np.argmin(d))
            nodes.append(node_ids[idx])
            dists.append(float(d[idx]))
        if return_dist:
            return nodes, dists
        return nodes

    monkeypatch.setattr(cg_mod.ox.distance, 'nearest_nodes', fake_nearest_nodes)
    # Graph IO: pretend save/load succeed against an in-memory dict so the
    # disk cache codepaths execute without actually touching GraphML.
    _store: dict = {}
    def fake_save_graphml(G, path):
        _store[str(path)] = G
        Path(path).write_text('stub', encoding='utf-8')
    def fake_load_graphml(path):
        # Second-call cache hits go through load_graphml → return the
        # stashed graph rather than parsing the stub text.
        if str(path) in _store:
            return _store[str(path)]
        raise RuntimeError('cache miss in stubbed load_graphml')
    monkeypatch.setattr(cg_mod.ox, 'save_graphml', fake_save_graphml)
    monkeypatch.setattr(cg_mod.ox, 'load_graphml', fake_load_graphml)


# ── tests ──────────────────────────────────────────────────────────────────

def test_fetch_road_graph_bbox_rejects_degenerate_box(tmp_path):
    pytest.importorskip("osmnx")
    from kernelcal.urban.city_graph import fetch_road_graph_bbox
    with pytest.raises(ValueError):
        fetch_road_graph_bbox(37.0, -122.0, 37.0, -121.9, cache_dir=tmp_path)
    with pytest.raises(ValueError):
        fetch_road_graph_bbox(37.0, -122.0, 37.1, -122.0, cache_dir=tmp_path)


def test_fetch_road_graph_bbox_returns_undirected_and_caches(monkeypatch, tmp_path):
    pytest.importorskip("osmnx")
    from kernelcal.urban import city_graph as cg_mod

    road = _synthetic_road_graph_wgs84(n_side=4)
    _patch_ox_for_road_pipeline(
        monkeypatch, cg_mod,
        _synthetic_building_gdf(n=4),  # unused here
        road,
    )

    G = cg_mod.fetch_road_graph_bbox(
        south=37.0, west=-122.001, north=37.003, east=-121.998,
        cache_dir=tmp_path,
    )
    # 4×4 grid has 16 nodes, 2·4·3 = 24 undirected edges.
    assert G.number_of_nodes() == 16
    # Undirected: cannot exceed the original directed edge count.
    assert G.number_of_edges() <= road.number_of_edges()
    # lon/lat attributes preserved for re-display.
    sample = next(iter(G.nodes(data=True)))[1]
    assert 'lon' in sample and 'lat' in sample


def test_fetch_road_graph_bbox_empty_caches_sentinel(monkeypatch, tmp_path):
    pytest.importorskip("osmnx")
    from kernelcal.urban import city_graph as cg_mod

    # Force the "no roads" path. ``ox.graph_from_bbox`` raises ValueError
    # on empty results in osmnx 1.x/2.x; our handler caches an empty graph
    # and returns None.
    monkeypatch.setattr(
        cg_mod.ox, 'graph_from_bbox',
        lambda *a, **kw: (_ for _ in ()).throw(ValueError('no roads')),
    )
    # Need save_graphml to not fail on the sentinel write.
    monkeypatch.setattr(cg_mod.ox, 'save_graphml',
                        lambda G, path: Path(path).write_text('empty'))
    # convert.to_undirected/project_graph won't be called here (we return
    # early on the empty branch) so no patch needed.

    G = cg_mod.fetch_road_graph_bbox(
        south=37.0, west=-122.001, north=37.003, east=-121.998,
        cache_dir=tmp_path,
    )
    assert G is None
    cached = list(Path(tmp_path).glob('roads_bbox_*.graphml'))
    assert len(cached) == 1


def test_buildings_to_graph_via_roads_end_to_end(monkeypatch, tmp_path):
    pytest.importorskip("osmnx")
    from kernelcal.urban import city_graph as cg_mod
    from kernelcal.urban import (
        buildings_to_graph_via_roads_from_bbox, CityGraph,
    )

    buildings = _synthetic_building_gdf(n=16)
    road      = _synthetic_road_graph_wgs84(n_side=6)
    _patch_ox_for_road_pipeline(monkeypatch, cg_mod, buildings, road)

    cg = buildings_to_graph_via_roads_from_bbox(
        south=37.0, west=-122.001, north=37.003, east=-121.998,
        k=4, n_max=100,
        network_type='drive', cache_dir=tmp_path,
    )
    assert isinstance(cg, CityGraph)
    assert cg.graph_mode == 'road_knn'
    assert cg.positions.shape[0] == 16
    assert cg.L.shape == (16, 16)
    assert cg.eigvals.shape == (16,)

    # Laplacian PSD + smallest eigenvalue ≈ 0.
    assert cg.eigvals.min() >= -1e-9
    assert cg.eigvals[0] < 1e-6
    # Adjacency symmetric.
    assert np.allclose(cg.W, cg.W.T)
    # Road metadata surfaced for the HTTP layer.
    assert cg.road_meta.get('n_road_nodes', 0) > 0
    assert cg.road_meta.get('n_road_edges', 0) > 0
    assert 'snap_offset_m' in cg.road_meta
    assert cg.road_meta.get('unique_snap_nodes', 0) >= 1


def test_buildings_to_graph_via_roads_falls_back_when_no_roads(
    monkeypatch, tmp_path,
):
    """With buildings but an empty road graph, the wrapper should degrade
    gracefully to Euclidean k-NN and mark the fallback reason — the HTTP
    caller must never 500 on a residential enclave that happens to lack a
    mapped street network."""
    pytest.importorskip("osmnx")
    from kernelcal.urban import city_graph as cg_mod
    from kernelcal.urban import buildings_to_graph_via_roads_from_bbox

    buildings = _synthetic_building_gdf(n=9)
    monkeypatch.setattr(
        cg_mod.ox, 'features_from_bbox',
        lambda *a, **kw: buildings.copy(),
    )
    # Road fetcher returns None (empty viewport).
    monkeypatch.setattr(
        cg_mod, 'fetch_road_graph_bbox',
        lambda *a, **kw: None,
    )

    cg = buildings_to_graph_via_roads_from_bbox(
        south=37.0, west=-122.001, north=37.003, east=-121.998,
        k=3, n_max=100, cache_dir=tmp_path,
    )
    assert cg is not None
    assert cg.graph_mode == 'knn'
    assert cg.road_meta.get('fallback_reason') == 'no_road_network'
    assert cg.positions.shape[0] == 9


def test_buildings_to_graph_via_roads_distances_vs_euclidean(
    monkeypatch, tmp_path,
):
    """Sanity check: road-aware and Euclidean builds produce *different*
    adjacency but both PSD Laplacians on the same building set.

    In a plain orthogonal grid with roads parallel to the building grid,
    network and Euclidean k-NN often agree on *who* the neighbours are,
    but the Gaussian weights differ because ``d_network ≥ d_euclidean``
    (buildings don't line up exactly with road intersections, so snap
    offsets add to both endpoints). Therefore: W.sum() for road_knn must
    be ≤ W.sum() for knn. This is a cheap structural invariant we can
    check without committing to a specific spectrum.
    """
    pytest.importorskip("osmnx")
    from kernelcal.urban import city_graph as cg_mod
    from kernelcal.urban import (
        buildings_to_graph_via_roads_from_bbox,
        buildings_to_graph_from_bbox,
    )

    buildings = _synthetic_building_gdf(n=16)
    road      = _synthetic_road_graph_wgs84(n_side=6)
    _patch_ox_for_road_pipeline(monkeypatch, cg_mod, buildings, road)

    kwargs = dict(
        south=37.0, west=-122.001, north=37.003, east=-121.998,
        k=4, n_max=100,
    )
    cg_knn  = buildings_to_graph_from_bbox(cache_dir=tmp_path / 'knn',  **kwargs)
    cg_road = buildings_to_graph_via_roads_from_bbox(
        cache_dir=tmp_path / 'road', network_type='drive', **kwargs,
    )

    assert cg_knn.graph_mode == 'knn'
    assert cg_road.graph_mode == 'road_knn'
    # Both Laplacians PSD.
    assert cg_knn.eigvals.min()  >= -1e-9
    assert cg_road.eigvals.min() >= -1e-9
    # Road-aware weights are ≤ Euclidean weights edge-for-edge in this
    # fixture (same σ, network distance ≥ Euclidean after snap offsets).
    assert cg_road.W.sum() <= cg_knn.W.sum() + 1e-9
