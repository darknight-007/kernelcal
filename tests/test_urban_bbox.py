"""Tests for kernelcal.urban bbox-based OSM fetch and graph construction.

These tests avoid live Overpass queries: the osmnx `features_from_bbox`
call is monkey-patched to return a synthetic GeoDataFrame so the full
pipeline (trait extraction → UTM projection → k-NN graph → eigendecomp)
is exercised deterministically.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


shapely = pytest.importorskip("shapely")
gpd = pytest.importorskip("geopandas")

from shapely.geometry import Polygon


def _synthetic_building_gdf(n: int = 20, spacing: float = 1e-4):
    """N buildings arranged as a jittered 2D grid in WGS84 degrees.

    spacing of 1e-4 deg ≈ 11 m near the equator, giving a plausible city
    block-scale separation for k-NN graphs after UTM reprojection.
    """
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


def test_bbox_cache_path_is_quantized(tmp_path):
    from kernelcal.urban.city_graph import _bbox_cache_path
    a = _bbox_cache_path((37.0001, -122.0002, 37.0102, -121.9901), tmp_path)
    b = _bbox_cache_path((37.00014, -122.00019, 37.01021, -121.99012), tmp_path)
    assert a == b, 'bbox cache key should quantize to 1e-4 deg'


def test_fetch_buildings_bbox_rejects_degenerate_box(tmp_path):
    pytest.importorskip("osmnx")
    from kernelcal.urban.city_graph import fetch_buildings_bbox
    with pytest.raises(ValueError):
        fetch_buildings_bbox(37.0, -122.0, 37.0, -121.9, cache_dir=tmp_path)
    with pytest.raises(ValueError):
        fetch_buildings_bbox(37.0, -122.0, 37.1, -122.0, cache_dir=tmp_path)


def test_fetch_buildings_bbox_returns_traits(monkeypatch, tmp_path):
    pytest.importorskip("osmnx")
    from kernelcal.urban import city_graph as cg_mod

    synthetic = _synthetic_building_gdf(n=16)
    monkeypatch.setattr(
        cg_mod.ox, 'features_from_bbox',
        lambda *a, **kw: synthetic.copy(),
    )

    gdf = cg_mod.fetch_buildings_bbox(
        south=37.0, west=-122.001, north=37.002, east=-121.999,
        cache_dir=tmp_path,
    )
    assert len(gdf) == 16
    for col in ('area_m2', 'height_m', 'type_enc', 'compacity', 'cx', 'cy'):
        assert col in gdf.columns, f'missing trait column {col}'
    assert (gdf['area_m2'] > 0).all()
    assert (gdf['height_m'] >= 3.2).all()
    assert set(gdf['type_enc'].unique()).issubset({0, 1, 2, 3})


def test_fetch_buildings_bbox_uses_cache(monkeypatch, tmp_path):
    pytest.importorskip("osmnx")
    from kernelcal.urban import city_graph as cg_mod

    synthetic = _synthetic_building_gdf(n=8)
    calls = {'n': 0}
    def fake_fetch(*a, **kw):
        calls['n'] += 1
        return synthetic.copy()
    monkeypatch.setattr(cg_mod.ox, 'features_from_bbox', fake_fetch)

    bbox = dict(south=37.0, west=-122.001, north=37.002, east=-121.999)
    cg_mod.fetch_buildings_bbox(**bbox, cache_dir=tmp_path)
    cg_mod.fetch_buildings_bbox(**bbox, cache_dir=tmp_path)
    assert calls['n'] == 1, 'second call should be served from disk cache'


def test_fetch_buildings_bbox_empty_is_cached(monkeypatch, tmp_path):
    pytest.importorskip("osmnx")
    from kernelcal.urban import city_graph as cg_mod

    empty = gpd.GeoDataFrame(geometry=[], crs='EPSG:4326')
    monkeypatch.setattr(
        cg_mod.ox, 'features_from_bbox',
        lambda *a, **kw: empty.copy(),
    )
    gdf = cg_mod.fetch_buildings_bbox(
        south=0.0, west=0.0, north=0.01, east=0.01,
        cache_dir=tmp_path,
    )
    assert len(gdf) == 0
    # sentinel file exists → second call hits cache
    cached = list(Path(tmp_path).glob('buildings_bbox_*.geojson'))
    assert len(cached) == 1


def test_buildings_to_graph_from_bbox_end_to_end(monkeypatch, tmp_path):
    pytest.importorskip("osmnx")
    from kernelcal.urban import city_graph as cg_mod
    from kernelcal.urban import buildings_to_graph_from_bbox, CityGraph

    synthetic = _synthetic_building_gdf(n=25)
    monkeypatch.setattr(
        cg_mod.ox, 'features_from_bbox',
        lambda *a, **kw: synthetic.copy(),
    )

    cg = buildings_to_graph_from_bbox(
        south=37.0, west=-122.001, north=37.003, east=-121.998,
        k=4, n_max=100, cache_dir=tmp_path,
    )
    assert isinstance(cg, CityGraph)
    assert cg.positions.shape[0] == 25
    assert cg.L.shape == (25, 25)
    assert cg.eigvals.shape == (25,)
    # Laplacian is PSD: eigenvalues non-negative, smallest ≈ 0
    assert cg.eigvals.min() >= -1e-9
    assert cg.eigvals[0] < 1e-6
    # Adjacency is symmetric
    assert np.allclose(cg.W, cg.W.T)


def test_fetch_buildings_bbox_uses_correct_osmnx_tuple_order(monkeypatch, tmp_path):
    """osmnx 1.x expects bbox=(north, south, east, west); 2.x expects (west, south, east, north).

    Regression guard: previous PR1 code unconditionally passed the 2.x order
    even on osmnx 1.x, producing swapped lat/lon, a degenerate polygon, and
    finally `cannot convert float NaN to integer` deep inside osmnx.
    """
    pytest.importorskip("osmnx")
    from kernelcal.urban import city_graph as cg_mod

    captured = {}
    def capture(*args, **kwargs):
        captured['bbox'] = kwargs.get('bbox', args[0] if args else None)
        return _synthetic_building_gdf(n=4)

    monkeypatch.setattr(cg_mod.ox, 'features_from_bbox', capture)

    south, west, north, east = 37.0, -122.001, 37.002, -121.999

    monkeypatch.setattr(cg_mod.ox, '__version__', '1.9.4', raising=False)
    cg_mod.fetch_buildings_bbox(south, west, north, east, cache_dir=tmp_path / 'v1', force_refresh=True)
    assert captured['bbox'] == (north, south, east, west), \
        f'osmnx 1.x must receive (north, south, east, west); got {captured["bbox"]}'

    monkeypatch.setattr(cg_mod.ox, '__version__', '2.0.0', raising=False)
    cg_mod.fetch_buildings_bbox(south, west, north, east, cache_dir=tmp_path / 'v2', force_refresh=True)
    assert captured['bbox'] == (west, south, east, north), \
        f'osmnx 2.x must receive (west, south, east, north); got {captured["bbox"]}'


def test_buildings_to_graph_from_bbox_empty_returns_none(monkeypatch, tmp_path):
    pytest.importorskip("osmnx")
    from kernelcal.urban import city_graph as cg_mod
    from kernelcal.urban import buildings_to_graph_from_bbox

    empty = gpd.GeoDataFrame(geometry=[], crs='EPSG:4326')
    monkeypatch.setattr(
        cg_mod.ox, 'features_from_bbox',
        lambda *a, **kw: empty.copy(),
    )
    cg = buildings_to_graph_from_bbox(
        south=0.0, west=0.0, north=0.01, east=0.01,
        cache_dir=tmp_path,
    )
    assert cg is None
