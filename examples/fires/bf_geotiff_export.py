#!/usr/bin/env python3
"""
bf_geotiff_export.py
====================
Render the BF channel polygon masks and rock polygons as GeoTIFFs
(EPSG:4326, WGS-84 lon/lat) for visual verification in QGIS / GDAL.

Outputs (in geotiffs/)
----------------------
  bf_aug_2020_mask.tif          — Aug 2020 channel mask (uint8, 0/255)
  bf_oct_2020_mask.tif          — Oct 2020 channel mask
  bf_dec_2020_mask.tif          — Dec 2020 channel mask
  bf_feb_2021_mask.tif          — Feb 2021 channel mask
  bf_all_timestamps_rgb.tif     — 4-band RGB+A composite (each timestamp one colour)
  bf_temporal_count.tif         — float32, value = number of timestamps present
  rock_polygons_mask.tif        — Sierra Nevada rock mask (uint8)
  rock_eccentricity.tif         — float32, eccentricity per pixel
  rock_major_axis.tif           — float32, major axis length per pixel

All GeoTIFFs are cloud-optimised (tiled, LZW compressed) and readable by
QGIS, GDAL, GRASS, ArcGIS, or any rasterio/rio client.
"""

from __future__ import annotations

import sys
import math
import gzip
import sqlite3
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS

KCAL_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR  = KCAL_ROOT / 'datasets' / 'bf_mbtiles'
OUT_DIR   = KCAL_ROOT / 'geotiffs'
OUT_DIR.mkdir(exist_ok=True)

# Resolution in degrees (~1 m at 33.8 N latitude)
RES_DEG = 0.000009   # ~1 m/pixel

TIMESTAMPS = [
    ('Aug 2020', DATA_DIR / 'bf_aug_2020.mbtiles',        (33, 150, 243)),   # blue
    ('Oct 2020', DATA_DIR / 'bf_oct_2020.mbtiles',        (255, 152,   0)),   # orange
    ('Dec 2020', DATA_DIR / 'bf_dec_2020_vector.mbtiles', (76,  175,  80)),   # green
    ('Feb 2021', DATA_DIR / 'bf_feb_2021_3d.mbtiles',     (233,  30, 99)),    # pink
]

WGS84 = CRS.from_epsg(4326)


# ══════════════════════════════════════════════════════════════════════════════
# MVT POLYGON EXTRACTOR  (same minimal decoder as before)
# ══════════════════════════════════════════════════════════════════════════════

def _varint(data, pos):
    r = s = 0
    while True:
        b = data[pos]; pos += 1
        r |= (b & 0x7F) << s
        if not (b & 0x80): return r, pos
        s += 7

def _zigzag(n): return (n >> 1) ^ -(n & 1)

def _decode_rings(cmds):
    rings, ring, cx, cy = [], [], 0, 0
    i = 0
    while i < len(cmds):
        ci = cmds[i]; i += 1
        cid, cnt = ci & 7, ci >> 3
        if cid in (1, 2):
            for _ in range(cnt):
                cx += _zigzag(cmds[i]); i += 1
                cy += _zigzag(cmds[i]); i += 1
                if cid == 1 and ring: rings.append(ring); ring = []
                ring.append((cx, cy))
        elif cid == 7:
            if ring: rings.append(ring); ring = []
    if ring: rings.append(ring)
    return rings

def _parse_feature(data):
    geom, pos, N = [], 0, len(data)
    while pos < N:
        tag, pos = _varint(data, pos)
        w, f = tag & 7, tag >> 3
        if w == 0:
            v, pos = _varint(data, pos)
            if f == 4: geom.append(v)
        elif w == 2:
            l, pos = _varint(data, pos); blob = data[pos:pos+l]; pos += l
            if f == 4:
                ip = 0
                while ip < len(blob): v, ip = _varint(blob, ip); geom.append(v)
        elif w == 5: pos += 4
        elif w == 1: pos += 8
        else: break
    return geom

def _parse_value(data):
    pos, N = 0, len(data)
    while pos < N:
        tag, pos = _varint(data, pos)
        w, f = tag & 7, tag >> 3
        if w == 0: v, pos = _varint(data, pos); return v
        elif w == 2:
            l, pos = _varint(data, pos); s = data[pos:pos+l].decode('utf-8','replace'); pos += l; return s
        elif w == 5: v = np.frombuffer(data[pos:pos+4],'<f4')[0]; pos += 4; return float(v)
        elif w == 1: v = np.frombuffer(data[pos:pos+8],'<f8')[0]; pos += 8; return float(v)
        else: break
    return None

def _parse_layer(data):
    extent, feat_blobs, keys, values = 4096, [], [], []
    pos, N = 0, len(data)
    while pos < N:
        tag, pos = _varint(data, pos)
        w, f = tag & 7, tag >> 3
        if w == 0:
            v, pos = _varint(data, pos)
            if f == 5: extent = v
        elif w == 2:
            l, pos = _varint(data, pos); blob = data[pos:pos+l]; pos += l
            if   f == 2: feat_blobs.append(blob)
            elif f == 3: keys.append(blob.decode('utf-8','replace'))
            elif f == 4: values.append(_parse_value(blob))
        elif w == 5: pos += 4
        elif w == 1: pos += 8
        else: break

    polys = []
    for fb in feat_blobs:
        geom, tag_ids = [], []
        p = 0
        while p < len(fb):
            tag, p = _varint(fb, p)
            w, f = tag & 7, tag >> 3
            if w == 0:
                v, p = _varint(fb, p)
                if f == 4: geom.append(v)
                elif f == 2: tag_ids.append(v)
            elif w == 2:
                l, p = _varint(fb, p); blob = fb[p:p+l]; p += l
                if f == 4:
                    ip = 0
                    while ip < len(blob): v, ip = _varint(blob, ip); geom.append(v)
            elif w == 5: p += 4
            elif w == 1: p += 8
            else: break

        if not geom: continue
        props = {}
        for k in range(0, len(tag_ids)-1, 2):
            ki, vi = tag_ids[k], tag_ids[k+1]
            if ki < len(keys) and vi < len(values):
                props[keys[ki]] = values[vi]

        rings = _decode_rings(geom)
        frac_rings = [[(x/extent, y/extent) for x, y in ring]
                      for ring in rings if len(ring) >= 3]
        if frac_rings:
            polys.append({'rings': frac_rings, 'props': props})
    return polys

def _parse_tile(raw):
    try: raw = gzip.decompress(raw)
    except Exception: pass
    polys, pos, N = [], 0, len(raw)
    while pos < N:
        tag, pos = _varint(raw, pos)
        w, f = tag & 7, tag >> 3
        if w == 0: _, pos = _varint(raw, pos)
        elif w == 2:
            l, pos = _varint(raw, pos); blob = raw[pos:pos+l]; pos += l
            if f == 3: polys.extend(_parse_layer(blob))
        elif w == 5: pos += 4
        elif w == 1: pos += 8
        else: break
    return polys

def tile_bbox(z, x, y_tms):
    y = (1 << z) - 1 - y_tms; n = 1 << z
    lw = x/n*360-180; le = (x+1)/n*360-180
    ln = math.degrees(math.atan(math.sinh(math.pi*(1-2*y/n))))
    ls = math.degrees(math.atan(math.sinh(math.pi*(1-2*(y+1)/n))))
    return lw, ls, le, ln

def extract_polygons(path, target_zoom=20):
    con = sqlite3.connect(str(path))
    cur = con.cursor()
    cur.execute('SELECT DISTINCT zoom_level FROM tiles ORDER BY zoom_level DESC')
    zooms = [r[0] for r in cur.fetchall()]
    zoom = target_zoom if target_zoom in zooms else zooms[0]
    cur.execute('SELECT tile_column, tile_row, tile_data FROM tiles WHERE zoom_level=?', (zoom,))
    rows = cur.fetchall(); con.close()

    all_polys = []; seen = set()
    for tx, ty, blob in rows:
        lw, ls, le, ln = tile_bbox(zoom, tx, ty)
        sl, st = le-lw, ln-ls
        for poly in _parse_tile(bytes(blob)):
            ll_rings = [[(lw+fx*sl, ln-fy*st) for fx, fy in ring]
                        for ring in poly['rings']]
            cx = round(np.mean([p[0] for p in ll_rings[0]]), 6)
            cy = round(np.mean([p[1] for p in ll_rings[0]]), 6)
            if (cx, cy) not in seen:
                seen.add((cx, cy))
                all_polys.append({'rings': ll_rings, 'props': poly['props']})
    return all_polys


# ══════════════════════════════════════════════════════════════════════════════
# RASTERISATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def make_canvas(lon_min, lat_min, lon_max, lat_max, res):
    """Return (rows, cols, transform) for a raster covering the bbox at res degrees/pixel."""
    cols = max(1, int(math.ceil((lon_max - lon_min) / res)))
    rows = max(1, int(math.ceil((lat_max - lat_min) / res)))
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, cols, rows)
    return rows, cols, transform

def lonlat_to_px(lon, lat, transform):
    """Convert (lon, lat) to (col, row) pixel indices using an affine transform."""
    col = (lon - transform.c) / transform.a
    row = (lat - transform.f) / transform.e
    return int(col), int(row)

def rasterise_polygon(ring_lonlat, canvas, transform):
    """
    Fill a polygon ring onto a 2-D numpy canvas using scanline rasterisation.
    ring_lonlat: list of (lon, lat) tuples (exterior ring).
    canvas: 2-D float array (modified in-place).
    Returns canvas.
    """
    rows, cols = canvas.shape
    # Convert ring to pixel coords
    px = [(lonlat_to_px(lon, lat, transform)) for lon, lat in ring_lonlat]
    if len(px) < 3:
        return canvas

    xs = np.array([p[0] for p in px], dtype=float)
    ys = np.array([p[1] for p in px], dtype=float)

    y_min = max(0, int(np.floor(ys.min())))
    y_max = min(rows - 1, int(np.ceil(ys.max())))

    for y in range(y_min, y_max + 1):
        # Scanline intersection with polygon edges
        intersections = []
        n = len(xs)
        for i in range(n):
            j = (i + 1) % n
            y0, y1 = ys[i], ys[j]
            x0, x1 = xs[i], xs[j]
            if (y0 <= y < y1) or (y1 <= y < y0):
                if y1 != y0:
                    x_int = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
                    intersections.append(x_int)
        intersections.sort()
        for k in range(0, len(intersections) - 1, 2):
            x_left  = max(0, int(math.floor(intersections[k])))
            x_right = min(cols - 1, int(math.ceil(intersections[k + 1])))
            canvas[y, x_left:x_right + 1] = 1.0
    return canvas

def write_geotiff(path, data, transform, crs=WGS84, dtype=None,
                  nodata=None, descriptions=None):
    """Write a (bands, rows, cols) or (rows, cols) array as a GeoTIFF."""
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    bands, rows, cols = data.shape
    dtype = dtype or data.dtype

    profile = dict(
        driver    = 'GTiff',
        dtype     = str(dtype),
        width     = cols,
        height    = rows,
        count     = bands,
        crs       = crs,
        transform = transform,
        compress  = 'lzw',
        tiled     = True,
        blockxsize= 256,
        blockysize= 256,
    )
    if nodata is not None:
        profile['nodata'] = nodata

    with rasterio.open(path, 'w', **profile) as dst:
        for b in range(bands):
            dst.write(data[b].astype(dtype), b + 1)
            if descriptions and b < len(descriptions):
                dst.update_tags(b + 1, description=descriptions[b])

    print(f'  Written: {path.name}  [{cols}x{rows} px, {bands} band(s)]')


# ══════════════════════════════════════════════════════════════════════════════
# LOAD ALL POLYGON DATA
# ══════════════════════════════════════════════════════════════════════════════

print('Loading polygon data...')
ts_polys = []
for label, path, color in TIMESTAMPS:
    print(f'  {label}...', end=' ', flush=True)
    polys = extract_polygons(path, target_zoom=20)
    print(f'{len(polys)} polygons')
    ts_polys.append(polys)

print('  Rocks...', end=' ', flush=True)
rock_polys = extract_polygons(DATA_DIR / 'rockPoly_all_v0.mbtiles', target_zoom=18)
print(f'{len(rock_polys)} polygons')


# ══════════════════════════════════════════════════════════════════════════════
# COMPUTE SHARED BBOX FOR CHANNEL TIMESTAMPS
# ══════════════════════════════════════════════════════════════════════════════

all_lons, all_lats = [], []
for polys in ts_polys:
    for p in polys:
        for lon, lat in p['rings'][0]:
            all_lons.append(lon); all_lats.append(lat)

pad = RES_DEG * 10
LON_MIN = min(all_lons) - pad;  LON_MAX = max(all_lons) + pad
LAT_MIN = min(all_lats) - pad;  LAT_MAX = max(all_lats) + pad

ROWS, COLS, TRANSFORM = make_canvas(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, RES_DEG)
print(f'\nChannel canvas: {COLS} x {ROWS} pixels  (~{COLS*RES_DEG*111000:.0f} m x {ROWS*RES_DEG*111000:.0f} m)')
print(f'  Extent: lon [{LON_MIN:.6f}, {LON_MAX:.6f}]  lat [{LAT_MIN:.6f}, {LAT_MAX:.6f}]')
print(f'  Resolution: {RES_DEG*111000:.2f} m/pixel\n')


# ══════════════════════════════════════════════════════════════════════════════
# RASTERISE EACH TIMESTAMP -> INDIVIDUAL MASK TIFFS
# ══════════════════════════════════════════════════════════════════════════════

print('Rasterising channel masks...')
masks = []
fnames = ['bf_aug_2020_mask.tif', 'bf_oct_2020_mask.tif',
          'bf_dec_2020_mask.tif', 'bf_feb_2021_mask.tif']

for (label, path, color), polys, fname in zip(TIMESTAMPS, ts_polys, fnames):
    print(f'  {label}  ({len(polys)} polygons)...', end=' ', flush=True)
    canvas = np.zeros((ROWS, COLS), dtype=np.float32)
    for poly in polys:
        rasterise_polygon(poly['rings'][0], canvas, TRANSFORM)
    mask_u8 = (canvas * 255).astype(np.uint8)
    masks.append(canvas)
    write_geotiff(OUT_DIR / fname, mask_u8, TRANSFORM,
                  dtype='uint8', nodata=0,
                  descriptions=[f'{label} channel mask'])


# ══════════════════════════════════════════════════════════════════════════════
# TEMPORAL COUNT RASTER  (how many timestamps cover each pixel)
# ══════════════════════════════════════════════════════════════════════════════

print('\nTemporal count raster...')
count = np.stack(masks, axis=0).sum(axis=0).astype(np.float32)
write_geotiff(OUT_DIR / 'bf_temporal_count.tif', count, TRANSFORM,
              dtype='float32', nodata=-1,
              descriptions=['Num timestamps (0-4) present at each pixel'])


# ══════════════════════════════════════════════════════════════════════════════
# RGBA COMPOSITE  (each timestamp = one colour channel)
# ══════════════════════════════════════════════════════════════════════════════

print('\nRGBA composite (4-band)...')
# Band 1 = Aug (blue base in grayscale; render as separate bands for QGIS pseudocolor)
# We'll encode as: R=Feb, G=Dec, B=Aug, A=Oct  so colours emerge naturally
r_band = (masks[3] * 200).astype(np.uint8)   # Feb 2021  → red
g_band = (masks[2] * 200).astype(np.uint8)   # Dec 2020  → green
b_band = (masks[0] * 200).astype(np.uint8)   # Aug 2020  → blue
a_band = np.clip(count * 63, 0, 255).astype(np.uint8)  # alpha = coverage count

rgba = np.stack([r_band, g_band, b_band, a_band], axis=0)
write_geotiff(OUT_DIR / 'bf_all_timestamps_rgba.tif', rgba, TRANSFORM,
              dtype='uint8',
              descriptions=['Feb-2021 (R)', 'Dec-2020 (G)', 'Aug-2020 (B)', 'Coverage alpha'])

# Also a 4-band version with one band per timestamp for per-channel QGIS styling
four_band = np.stack([(m * 255).astype(np.uint8) for m in masks], axis=0)
write_geotiff(OUT_DIR / 'bf_4band_timestamps.tif', four_band, TRANSFORM,
              dtype='uint8',
              descriptions=['Aug-2020', 'Oct-2020', 'Dec-2020', 'Feb-2021'])


# ══════════════════════════════════════════════════════════════════════════════
# ROCK POLYGONS GEOTIFF  (Sierra Nevada — separate extent)
# ══════════════════════════════════════════════════════════════════════════════

print('\nRock polygon GeoTIFFs...')
r_lons, r_lats = [], []
for p in rock_polys:
    for lon, lat in p['rings'][0]:
        r_lons.append(lon); r_lats.append(lat)

r_pad = RES_DEG * 10
R_LON_MIN, R_LON_MAX = min(r_lons)-r_pad, max(r_lons)+r_pad
R_LAT_MIN, R_LAT_MAX = min(r_lats)-r_pad, max(r_lats)+r_pad
R_ROWS, R_COLS, R_TRANSFORM = make_canvas(R_LON_MIN, R_LAT_MIN, R_LON_MAX, R_LAT_MAX, RES_DEG)
print(f'  Rock canvas: {R_COLS} x {R_ROWS} pixels')

rock_mask   = np.zeros((R_ROWS, R_COLS), dtype=np.float32)
rock_ecc    = np.full((R_ROWS, R_COLS), -1, dtype=np.float32)
rock_major  = np.full((R_ROWS, R_COLS), -1, dtype=np.float32)

for poly in rock_polys:
    ring = poly['rings'][0]
    props = poly['props']
    ecc   = float(props.get('eccentrici', -1))
    major = float(props.get('major_leng', -1))

    # Rasterise mask
    temp = np.zeros((R_ROWS, R_COLS), dtype=np.float32)
    rasterise_polygon(ring, temp, R_TRANSFORM)

    # Where this polygon lands, fill attribute values
    px_mask = temp > 0
    rock_mask[px_mask] = 1.0
    if ecc >= 0:
        rock_ecc[px_mask] = ecc
    if major > 0:
        rock_major[px_mask] = major

write_geotiff(OUT_DIR / 'rock_mask.tif',
              (rock_mask * 255).astype(np.uint8), R_TRANSFORM,
              dtype='uint8', nodata=0,
              descriptions=['Rock polygon mask (255=rock)'])

write_geotiff(OUT_DIR / 'rock_eccentricity.tif',
              rock_ecc, R_TRANSFORM,
              dtype='float32', nodata=-1,
              descriptions=['Rock eccentricity (0=round, 1=elongated, -1=nodata)'])

write_geotiff(OUT_DIR / 'rock_major_axis.tif',
              rock_major, R_TRANSFORM,
              dtype='float32', nodata=-1,
              descriptions=['Rock major axis length [pixels], -1=nodata'])


# ══════════════════════════════════════════════════════════════════════════════
# PRINT QGIS LOAD INSTRUCTIONS
# ══════════════════════════════════════════════════════════════════════════════

print(f"""
All GeoTIFFs written to:  {OUT_DIR}/

  Channel masks (EPSG:4326, ~1m/px):
    bf_aug_2020_mask.tif          uint8 mask, Aug 2020
    bf_oct_2020_mask.tif          uint8 mask, Oct 2020
    bf_dec_2020_mask.tif          uint8 mask, Dec 2020
    bf_feb_2021_mask.tif          uint8 mask, Feb 2021
    bf_temporal_count.tif         float32, 0-4 = timestamps covering pixel
    bf_all_timestamps_rgba.tif    RGBA composite (R=Feb, G=Dec, B=Aug, A=count)
    bf_4band_timestamps.tif       4-band (one band per timestamp, uint8)

  Rock polygons (EPSG:4326, ~1m/px, Sierra Nevada):
    rock_mask.tif                 uint8 mask (255=rock)
    rock_eccentricity.tif         float32 attribute raster
    rock_major_axis.tif           float32 attribute raster

  To load in QGIS:
    Layer > Add Layer > Add Raster Layer > select any .tif
    For bf_all_timestamps_rgba.tif: use Multiband Color renderer (R/G/B/A).
    For bf_temporal_count.tif:      use Singleband Pseudocolor, colormap Magma.
    For rock_eccentricity.tif:      use Singleband Pseudocolor, colormap RdYlGn.
    CRS is EPSG:4326 — overlay directly on any WGS-84 basemap.
""")
