#!/usr/bin/env python3
"""
bf_vegetation_segment.py — Batch vegetation segmentation pipeline for Bobcat Fire.

Reads the four raster MBTiles timestamps, sends each tile to the Grounded-SAM-2
API running on the GPU server (192.168.0.232:5001), and converts detections to
TiledGISLabel-compatible CSV files for import into DeepGIS-XR.

Vegetation classes detected (dot-separated Grounding DINO prompt):
  shrub . unburned vegetation . burned shrub . burned vegetation . bare soil

Output CSVs import via:
  docker exec deepgis-xr_web_1 python manage.py import_rocks_labels \\
    --csv-file /app/data/vegetation_labels/bf_vegetation_aug2020.csv \\
    --category-name "shrub"

Architecture:
  DeepGIS-XR server (192.168.0.186:2222)
    └─ raster MBTiles (zoom 12-16)
    └─ DeepGIS-XR Django app (port 8060)
       └─ TiledGISLabel storage (PostgreSQL/SQLite)
  GPU inference server (192.168.0.232)
    └─ Grounded-SAM-2 (port 5001): combined detection + instance segmentation
    └─ Grounding DINO (port 5000): open-vocabulary detection only
"""

from __future__ import annotations
import csv, io, json, math, sqlite3, sys, time
from pathlib import Path
from datetime import datetime

import requests
from PIL import Image

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
DATA_DIR = Path('/home/jdas/dreams-lab-website-server/deepgis-xr/data')
OUT_DIR  = DATA_DIR / 'vegetation_labels'

GROUNDED_SAM_URL = 'http://192.168.0.232:5001'
GROUNDING_DINO_URL = 'http://192.168.0.232:5000'

# Prefer highest-res tiles; fall back to max available
TARGET_ZOOM = 16

TIMESTAMPS = [
    ('aug2020', DATA_DIR / 'bf_aug_2020_raster.mbtiles'),
    ('oct2020', DATA_DIR / 'bf_oct_2020_raster.mbtiles'),
    ('nov2020', DATA_DIR / 'bf_nov_2020.mbtiles'),
    ('dec2020', DATA_DIR / 'BF_12-20-2020.mbtiles'),
]

# Vegetation detection prompt — ordered by specificity
VEG_PROMPT = (
    'shrub . unburned shrub . burned shrub . '
    'burned vegetation . unburned vegetation . bare soil . rock'
)
BOX_THRESHOLD  = 0.25   # lower = more detections
TEXT_THRESHOLD = 0.20
SIMPLIFY_TOL   = 0.005  # polygon simplification (normalised coords)

# ──────────────────────────────────────────────
# TILE MATHS  (TMS ↔ WGS-84)
# ──────────────────────────────────────────────

def tile_bounds(z: int, x: int, y_tms: int):
    """Return (sw_lat, sw_lng, ne_lat, ne_lng) for a TMS tile (y=0 at south)."""
    y_map = (2**z - 1) - y_tms        # TMS → slippy-map convention (y=0 at north)
    n = 2**z
    lon_w = x / n * 360.0 - 180.0
    lon_e = (x + 1) / n * 360.0 - 180.0
    lat_n = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y_map / n))))
    lat_s = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y_map + 1) / n))))
    return lat_s, lon_w, lat_n, lon_e


def px_to_latlng(
    px: float, py: float,
    img_w: int, img_h: int,
    sw_lat: float, sw_lng: float,
    ne_lat: float, ne_lng: float,
) -> tuple[float, float]:
    """Pixel (0,0 = top-left) → (lat, lng)."""
    lng = sw_lng + (px / img_w) * (ne_lng - sw_lng)
    lat = ne_lat - (py / img_h) * (ne_lat - sw_lat)
    return lat, lng


# ──────────────────────────────────────────────
# INFERENCE CALLS
# ──────────────────────────────────────────────

def call_grounded_sam(img_bytes: bytes) -> list[dict]:
    """Call Grounded-SAM-2 /detect; return list of detection dicts."""
    try:
        r = requests.post(
            f'{GROUNDED_SAM_URL}/detect',
            files={'image': ('tile.jpg', img_bytes, 'image/jpeg')},
            data={
                'text_prompt':       VEG_PROMPT,
                'box_threshold':     BOX_THRESHOLD,
                'text_threshold':    TEXT_THRESHOLD,
                'mask_format':       'geojson',
                'simplify_tolerance': SIMPLIFY_TOL,
            },
            timeout=90,
        )
        if r.status_code != 200:
            return []
        data = r.json()
        if not data.get('success'):
            return []
        return data.get('results', {}).get('detections', [])
    except Exception as exc:
        print(f'      [SAM API] {exc}')
        return []


def call_grounding_dino_only(img_bytes: bytes) -> list[dict]:
    """
    Fallback: call Grounding DINO (boxes only, no masks).
    Returns detections in the same schema as call_grounded_sam().
    """
    try:
        r = requests.post(
            f'{GROUNDING_DINO_URL}/api/predict',
            files={'file': ('tile.jpg', img_bytes, 'image/jpeg')},
            data={
                'text_prompt':    VEG_PROMPT,
                'box_threshold':  BOX_THRESHOLD,
                'text_threshold': TEXT_THRESHOLD,
            },
            timeout=60,
        )
        if r.status_code != 200:
            return []
        data = r.json()
        # Normalise to same schema: {label, confidence, box, mask=None}
        raw = data.get('detections', data.get('results', {}).get('detections', []))
        return [{**d, 'mask': None} for d in raw]
    except Exception as exc:
        print(f'      [DINO API] {exc}')
        return []


# ──────────────────────────────────────────────
# MASK → GEOGRAPHIC POLYGON CONVERSION
# ──────────────────────────────────────────────

def _rings_to_geo(rings: list, img_w: int, img_h: int,
                  sw_lat: float, sw_lng: float,
                  ne_lat: float, ne_lng: float) -> list:
    geo_rings = []
    for ring in rings:
        geo_ring = []
        for pt in ring:
            # Handle normalised [0,1] OR pixel coordinates automatically
            px = pt[0] * img_w if pt[0] <= 1.0 else pt[0]
            py = pt[1] * img_h if pt[1] <= 1.0 else pt[1]
            lat, lng = px_to_latlng(px, py, img_w, img_h,
                                    sw_lat, sw_lng, ne_lat, ne_lng)
            geo_ring.append([lng, lat])   # GeoJSON: [lng, lat]
        geo_rings.append(geo_ring)
    return geo_rings


def mask_to_geo_polygon(mask_geojson: dict | None,
                        box: list[float],
                        img_w: int, img_h: int,
                        sw_lat: float, sw_lng: float,
                        ne_lat: float, ne_lng: float) -> dict | None:
    """Convert SAM mask GeoJSON (pixel or normalised coords) to geo Polygon."""
    if mask_geojson:
        gtype = mask_geojson.get('type', '')
        if gtype == 'Polygon':
            rings = mask_geojson['coordinates']
            geo_rings = _rings_to_geo(rings, img_w, img_h,
                                      sw_lat, sw_lng, ne_lat, ne_lng)
            return {'type': 'Polygon', 'coordinates': geo_rings}
        elif gtype == 'MultiPolygon':
            # Pick largest ring
            best = max(mask_geojson['coordinates'], key=lambda r: len(r[0]))
            geo_rings = _rings_to_geo(best, img_w, img_h,
                                      sw_lat, sw_lng, ne_lat, ne_lng)
            return {'type': 'Polygon', 'coordinates': geo_rings}

    # Fallback: bounding box
    x1, y1, x2, y2 = box
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
    ring = []
    for px, py in corners:
        lat, lng = px_to_latlng(px, py, img_w, img_h,
                                sw_lat, sw_lng, ne_lat, ne_lng)
        ring.append([lng, lat])
    return {'type': 'Polygon', 'coordinates': [ring]}


# ──────────────────────────────────────────────
# CSV ROW BUILDER
# ──────────────────────────────────────────────

CSV_FIELDS = [
    'northeast_Lat', 'northeast_Lng',
    'southwest_Lat', 'southwest_Lng',
    'zoom_level', 'label_type',
    'label_json', 'geometry',
    'class_name', 'confidence',
]


def make_row(det: dict,
             img_w: int, img_h: int,
             sw_lat: float, sw_lng: float,
             ne_lat: float, ne_lng: float,
             zoom: int) -> dict | None:
    label      = det.get('label', 'vegetation')
    confidence = float(det.get('confidence', 0.0))
    box        = det.get('box', [0, 0, img_w, img_h])
    mask       = det.get('mask', None)

    geo_poly = mask_to_geo_polygon(mask, box, img_w, img_h,
                                   sw_lat, sw_lng, ne_lat, ne_lng)
    if geo_poly is None:
        return None

    feature = {
        'type': 'Feature',
        'geometry': geo_poly,
        'properties': {'class': label, 'confidence': round(confidence, 4)},
    }
    label_json_str = json.dumps([feature])

    # WKT for the geometry column
    coords = geo_poly['coordinates'][0]
    wkt = 'POLYGON ((' + ', '.join(f'{pt[0]} {pt[1]}' for pt in coords) + '))'

    return {
        'northeast_Lat': round(ne_lat, 10),
        'northeast_Lng': round(ne_lng, 10),
        'southwest_Lat': round(sw_lat, 10),
        'southwest_Lng': round(sw_lng, 10),
        'zoom_level':    zoom,
        'label_type':    'P',
        'label_json':    label_json_str,
        'geometry':      wkt,
        'class_name':    label,
        'confidence':    round(confidence, 4),
    }


# ──────────────────────────────────────────────
# TILE PROCESSING
# ──────────────────────────────────────────────

def best_zoom(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        'SELECT zoom_level FROM tiles GROUP BY zoom_level ORDER BY zoom_level DESC LIMIT 1'
    ).fetchone()
    return row[0] if row else TARGET_ZOOM


def process_mbtiles(ts_name: str, mbtiles_path: Path) -> tuple[Path | None, int, int]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(mbtiles_path))

    # Choose zoom level
    zoom = TARGET_ZOOM
    count = conn.execute(
        'SELECT COUNT(*) FROM tiles WHERE zoom_level=?', (zoom,)
    ).fetchone()[0]
    if count == 0:
        zoom = best_zoom(conn)
        print(f'  zoom {TARGET_ZOOM} empty — using zoom {zoom}')
    tiles = conn.execute(
        'SELECT zoom_level, tile_column, tile_row, tile_data FROM tiles WHERE zoom_level=?',
        (zoom,)
    ).fetchall()

    n_total = len(tiles)
    print(f'  {n_total} tiles at zoom {zoom}')

    rows: list[dict] = []
    errors = 0

    for i, (z, x, y_tms, tile_data) in enumerate(tiles):
        sw_lat, sw_lng, ne_lat, ne_lng = tile_bounds(z, x, y_tms)

        # Decode tile image → JPEG bytes
        try:
            img = Image.open(io.BytesIO(tile_data))
            if img.mode not in ('RGB',):
                img = img.convert('RGB')
            img_w, img_h = img.size
            buf = io.BytesIO()
            img.save(buf, 'JPEG', quality=90)
            img_bytes = buf.getvalue()
        except Exception as exc:
            print(f'    tile {i+1}/{n_total} decode error: {exc}')
            errors += 1
            continue

        # Inference
        detections = call_grounded_sam(img_bytes)
        if not detections:
            detections = call_grounding_dino_only(img_bytes)

        for det in detections:
            row = make_row(det, img_w, img_h, sw_lat, sw_lng, ne_lat, ne_lng, z)
            if row:
                rows.append(row)

        if (i + 1) % 10 == 0 or (i + 1) == n_total:
            print(f'    {i+1}/{n_total} tiles — {len(rows)} labels so far')

        time.sleep(0.02)  # gentle on GPU server

    conn.close()

    if not rows:
        print('  => no detections')
        return None, 0, errors

    csv_path = OUT_DIR / f'bf_vegetation_{ts_name}.csv'
    with open(csv_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f'  => {len(rows)} labels → {csv_path}')
    return csv_path, len(rows), errors


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def health_check() -> bool:
    """Return True if Grounded-SAM-2 API is reachable."""
    try:
        r = requests.get(f'{GROUNDED_SAM_URL}/health', timeout=6)
        h = r.json()
        print(f'  Grounded-SAM-2: {h}')
        return h.get('status') == 'healthy'
    except Exception as exc:
        print(f'  Grounded-SAM-2 unreachable: {exc}')

    # Try Grounding DINO fallback
    try:
        r = requests.get(f'{GROUNDING_DINO_URL}/health', timeout=6)
        h = r.json()
        print(f'  Grounding DINO fallback: {h}')
        return True
    except Exception as exc:
        print(f'  Grounding DINO also unreachable: {exc}')
        return False


def main():
    print('=' * 62)
    print('Bobcat Fire — Vegetation Segmentation Pipeline')
    print(f'  SAM-2:  {GROUNDED_SAM_URL}')
    print(f'  DINO:   {GROUNDING_DINO_URL}')
    print(f'  Output: {OUT_DIR}')
    print('=' * 62)

    if not health_check():
        sys.exit('ERROR: No inference API reachable. Aborting.')
    print()

    results: list[tuple] = []
    for ts_name, mbtiles_path in TIMESTAMPS:
        if not mbtiles_path.exists():
            print(f'[SKIP] {ts_name}: {mbtiles_path.name} not found')
            results.append((ts_name, None, 0, 0))
            continue
        print(f'[{ts_name}]  {mbtiles_path.name}')
        t0 = time.time()
        csv_path, n_labels, n_errors = process_mbtiles(ts_name, mbtiles_path)
        elapsed = time.time() - t0
        results.append((ts_name, csv_path, n_labels, n_errors))
        print(f'  done in {elapsed:.1f}s\n')

    # ── Summary ──
    print('=' * 62)
    print('SUMMARY')
    print()
    for ts_name, csv_path, n_labels, n_errors in results:
        status = f'{n_labels} labels, {n_errors} errors'
        print(f'  {ts_name:<10} {status}')
        if csv_path:
            container_path = str(csv_path).replace(
                str(DATA_DIR), '/app/data'
            )
            print(f'    Import:')
            cat_guess = 'vegetation'
            print(
                f'    docker exec deepgis-xr_web_1 python manage.py import_rocks_labels \\\n'
                f'      --csv-file {container_path} \\\n'
                f'      --category-name "{cat_guess}"'
            )
    print()
    print('Note: run import_rocks_labels once per category type.')
    print('Categories map: shrub → "shrub", burned shrub → "burned_shrub", etc.')
    print('=' * 62)


if __name__ == '__main__':
    main()
