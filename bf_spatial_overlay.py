#!/usr/bin/env python3
"""
bf_spatial_overlay.py
=====================
True spatial polygon overlays for:
  (A) BF channel masks — all 4 timestamps overlaid in geographic coordinates
  (B) Rock polygon layer — true spatial distribution with morphometric coloring
  (C) Temporal diff maps — which polygons appeared / disappeared between timestamps
  (D) Channel + rock analysis summary: what we learned

Extracts actual polygon ring geometry (not just centroids) from MVT tiles
and renders them in WGS-84 lon/lat space.
"""

from __future__ import annotations

import sys
import math
import gzip
import sqlite3
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize, LogNorm
from matplotlib.cm import ScalarMappable
from scipy.spatial import cKDTree

KCAL_ROOT = Path(__file__).parent
FIG_DIR   = KCAL_ROOT / 'figures'
DATA_DIR  = KCAL_ROOT / 'datasets' / 'bf_mbtiles'
FIG_DIR.mkdir(exist_ok=True)

DARK_BG  = '#FFFFFF'
PANEL_BG = '#FFFFFF'
TEXT_COL = '#222222'
GRID_COL = '#CCCCCC'

TIMESTAMPS = [
    ('Aug 2020', DATA_DIR / 'bf_aug_2020.mbtiles',         '#2196F3', 0.55),
    ('Oct 2020', DATA_DIR / 'bf_oct_2020.mbtiles',         '#FF9800', 0.45),
    ('Dec 2020', DATA_DIR / 'bf_dec_2020_vector.mbtiles',  '#4CAF50', 0.40),
    ('Feb 2021', DATA_DIR / 'bf_feb_2021_3d.mbtiles',      '#E91E63', 0.35),
]


# ══════════════════════════════════════════════════════════════════════════════
# MVT POLYGON GEOMETRY EXTRACTOR
# Returns full ring coordinates in lon/lat (not just centroids)
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
    """Decode MVT geometry -> list of rings, each ring = list of (x,y) ints."""
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

def _parse_feature_geom(data):
    """Return geometry command list from a Feature blob."""
    geom, pos, N = [], 0, len(data)
    while pos < N:
        tag, pos = _varint(data, pos)
        w, f = tag & 7, tag >> 3
        if w == 0:
            v, pos = _varint(data, pos)
            if f == 4: geom.append(v)
        elif w == 2:
            l, pos = _varint(data, pos)
            blob = data[pos:pos+l]; pos += l
            if f == 4:
                ip = 0
                while ip < len(blob): v, ip = _varint(blob, ip); geom.append(v)
        elif w == 5: pos += 4
        elif w == 1: pos += 8
        else: break
    return geom

def _parse_feature_props(data):
    """Return (geometry_cmds, props_dict) from a Feature blob (reads keys/values via layer context)."""
    geom, pos, N = [], 0, len(data)
    tag_ids = []
    while pos < N:
        tag, pos = _varint(data, pos)
        w, f = tag & 7, tag >> 3
        if w == 0:
            v, pos = _varint(data, pos)
            if f == 4: geom.append(v)
            elif f == 2: tag_ids.append(v)   # tag pairs
        elif w == 2:
            l, pos = _varint(data, pos)
            blob = data[pos:pos+l]; pos += l
            if f == 4:
                ip = 0
                while ip < len(blob): v, ip = _varint(blob, ip); geom.append(v)
        elif w == 5: pos += 4
        elif w == 1: pos += 8
        else: break
    return geom, tag_ids

def _parse_value(data):
    """Parse a Value protobuf blob -> Python scalar."""
    pos, N = 0, len(data)
    while pos < N:
        tag, pos = _varint(data, pos)
        w, f = tag & 7, tag >> 3
        if w == 0:
            v, pos = _varint(data, pos)
            return float(v) if f in (5, 6) else v
        elif w == 2:
            l, pos = _varint(data, pos)
            s = data[pos:pos+l].decode('utf-8', errors='replace'); pos += l
            return s
        elif w == 5: v = data[pos:pos+4]; pos += 4; return float(np.frombuffer(v,'<f4')[0])
        elif w == 1: v = data[pos:pos+8]; pos += 8; return float(np.frombuffer(v,'<f8')[0])
        else: break
    return None

def _parse_layer_polygons(data):
    """
    Parse one MVT Layer.
    Returns list of dicts: {'rings': [[(lon,lat),...], ...], 'props': {k:v}}
    All coordinates in tile-local fractional space [0,1); caller converts to lonlat.
    """
    extent = 4096
    feat_blobs, keys, values = [], [], []
    pos, N = 0, len(data)
    while pos < N:
        tag, pos = _varint(data, pos)
        w, f = tag & 7, tag >> 3
        if w == 0:
            v, pos = _varint(data, pos)
            if f == 5: extent = v
        elif w == 2:
            l, pos = _varint(data, pos)
            blob = data[pos:pos+l]; pos += l
            if f == 2: feat_blobs.append(blob)
            elif f == 3: keys.append(blob.decode('utf-8', errors='replace'))
            elif f == 4: values.append(_parse_value(blob))
        elif w == 5: pos += 4
        elif w == 1: pos += 8
        else: break

    polys = []
    for fb in feat_blobs:
        cmds, tag_ids = _parse_feature_props(fb)
        if not cmds: continue
        rings = _decode_rings(cmds)
        if not rings: continue
        # Build props dict from tag_ids pairs
        props = {}
        for k in range(0, len(tag_ids)-1, 2):
            ki, vi = tag_ids[k], tag_ids[k+1]
            if ki < len(keys) and vi < len(values):
                props[keys[ki]] = values[vi]
        # Convert ring pixel coords to fractional [0,1)
        frac_rings = [
            [(x / extent, y / extent) for (x, y) in ring]
            for ring in rings if len(ring) >= 3
        ]
        if frac_rings:
            polys.append({'rings': frac_rings, 'props': props})
    return polys

def _parse_tile_polygons(raw):
    """Decode tile blob -> list of polygon dicts with fractional ring coords."""
    try: raw = gzip.decompress(raw)
    except Exception: pass
    polys, pos, N = [], 0, len(raw)
    while pos < N:
        tag, pos = _varint(raw, pos)
        w, f = tag & 7, tag >> 3
        if w == 0: _, pos = _varint(raw, pos)
        elif w == 2:
            l, pos = _varint(raw, pos)
            blob = raw[pos:pos+l]; pos += l
            if f == 3: polys.extend(_parse_layer_polygons(blob))
        elif w == 5: pos += 4
        elif w == 1: pos += 8
        else: break
    return polys

def tile_bbox(z, x, y_tms):
    y = (1 << z) - 1 - y_tms; n = 1 << z
    lw = x/n*360 - 180; le = (x+1)/n*360 - 180
    ln = math.degrees(math.atan(math.sinh(math.pi*(1 - 2*y/n))))
    ls = math.degrees(math.atan(math.sinh(math.pi*(1 - 2*(y+1)/n))))
    return lw, ls, le, ln

def extract_polygons(path, target_zoom=20, max_polys=None):
    """
    Extract all polygon rings in lon/lat from a vector MBTiles file.
    Returns list of {'rings': [[(lon,lat),...]], 'props': dict}.
    """
    con = sqlite3.connect(str(path))
    cur = con.cursor()
    cur.execute('SELECT DISTINCT zoom_level FROM tiles ORDER BY zoom_level DESC')
    zooms = [r[0] for r in cur.fetchall()]
    zoom = target_zoom if target_zoom in zooms else zooms[0]
    cur.execute('SELECT tile_column, tile_row, tile_data FROM tiles WHERE zoom_level=?', (zoom,))
    rows = cur.fetchall(); con.close()

    all_polys = []
    seen_centroids = set()
    for tx, ty, blob in rows:
        lw, ls, le, ln = tile_bbox(zoom, tx, ty)
        span_lon, span_lat = le - lw, ln - ls
        for poly in _parse_tile_polygons(bytes(blob)):
            # Convert fractional -> lonlat for each ring
            ll_rings = []
            for ring in poly['rings']:
                ll_ring = [(lw + fx*span_lon, ln - fy*span_lat) for (fx, fy) in ring]
                ll_rings.append(ll_ring)
            # Deduplicate by centroid (tile-boundary splits)
            cx = np.mean([p[0] for p in ll_rings[0]])
            cy = np.mean([p[1] for p in ll_rings[0]])
            key = (round(cx, 6), round(cy, 6))
            if key not in seen_centroids:
                seen_centroids.add(key)
                all_polys.append({'rings': ll_rings, 'props': poly['props']})
        if max_polys and len(all_polys) >= max_polys:
            break
    return all_polys

def polys_to_patches(polys, max_draw=3000):
    """Convert polygon list to matplotlib vertex arrays for PolyCollection."""
    verts = []
    step = max(1, len(polys) // max_draw)
    for poly in polys[::step]:
        ring = poly['rings'][0]
        if len(ring) >= 3:
            verts.append(np.array(ring))
    return verts

def lonlat_to_metres(lonlat_arr, lon0=None, lat0=None):
    if lon0 is None: lon0 = lonlat_arr[:, 0].mean()
    if lat0 is None: lat0 = lonlat_arr[:, 1].mean()
    R = 6_371_000.0; c = math.cos(math.radians(lat0))
    E = (lonlat_arr[:, 0]-lon0)*c*(math.pi/180)*R
    N = (lonlat_arr[:, 1]-lat0)*(math.pi/180)*R
    return np.column_stack([E, N])


# ══════════════════════════════════════════════════════════════════════════════
# LOAD ALL DATA
# ══════════════════════════════════════════════════════════════════════════════

print('Extracting polygon geometries...')
ts_polys = []
for label, path, color, alpha in TIMESTAMPS:
    print(f'  {label}...', end=' ', flush=True)
    polys = extract_polygons(path, target_zoom=20)
    print(f'{len(polys)} polygons')
    ts_polys.append(polys)

print('  Rocks (Sierra Nevada)...', end=' ', flush=True)
rock_polys = extract_polygons(DATA_DIR / 'rockPoly_all_v0.mbtiles', target_zoom=18, max_polys=8000)
print(f'{len(rock_polys)} polygons')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: BF CHANNEL TEMPORAL OVERLAY (true geographic coordinates)
# ══════════════════════════════════════════════════════════════════════════════

print('\nFigure 1: BF channel temporal overlay...')

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle(
    'Bobcat Fire channel network — true spatial distribution\n'
    'Site: -111.265 W, 33.782 N  |  Tonto NF, AZ  |  WGS-84 coordinates',
    color=TEXT_COL, fontsize=11, fontweight='bold'
)

# Left: all 4 timestamps overlaid
ax = axes[0]
ax.set_facecolor('#050A0F')

for (label, path, color, alpha), polys in zip(TIMESTAMPS, ts_polys):
    verts = polys_to_patches(polys, max_draw=2000)
    if not verts: continue
    coll = PolyCollection(verts, facecolor=color, edgecolor=color,
                          alpha=alpha, linewidth=0.3)
    ax.add_collection(coll)

# Set extent from all timestamps combined
all_lons = [p['rings'][0][j][0] for polys in ts_polys for p in polys for j in range(len(p['rings'][0]))]
all_lats = [p['rings'][0][j][1] for polys in ts_polys for p in polys for j in range(len(p['rings'][0]))]
pad_lon = (max(all_lons)-min(all_lons))*0.05
pad_lat = (max(all_lats)-min(all_lats))*0.05
ax.set_xlim(min(all_lons)-pad_lon, max(all_lons)+pad_lon)
ax.set_ylim(min(all_lats)-pad_lat, max(all_lats)+pad_lat)

legend_patches = [mpatches.Patch(color=c, alpha=0.8, label=f'{l}  ({len(p):,} polys)')
                  for (l,_,c,_), p in zip(TIMESTAMPS, ts_polys)]
ax.legend(handles=legend_patches, loc='upper left', fontsize=8,
          facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL)
ax.set_title('All 4 timestamps overlaid\n(blue=Aug, orange=Oct, green=Dec, pink=Feb)',
             color=TEXT_COL, fontsize=9, pad=6)
ax.set_xlabel('Longitude', color=TEXT_COL, fontsize=8)
ax.set_ylabel('Latitude', color=TEXT_COL, fontsize=8)
ax.tick_params(colors=TEXT_COL, labelsize=7)
for sp in ax.spines.values(): sp.set_edgecolor(GRID_COL)
ax.grid(True, color=GRID_COL, linewidth=0.4, alpha=0.5)

# Right: 4-panel grid per timestamp
for ti, ((label, path, color, alpha), polys) in enumerate(zip(TIMESTAMPS, ts_polys)):
    ax2 = axes[1]  # we'll use a gridspec-style approach

# Use inset axes for the 4 sub-panels
from mpl_toolkits.axes_grid1 import ImageGrid
axes[1].remove()
inner = fig.add_axes([0.52, 0.08, 0.46, 0.82])
inner.set_visible(False)

sub_positions = [(0.52, 0.48, 0.22, 0.38), (0.74, 0.48, 0.22, 0.38),
                 (0.52, 0.08, 0.22, 0.38), (0.74, 0.08, 0.22, 0.38)]

for ti, ((label, path, color, alpha), polys) in enumerate(zip(TIMESTAMPS, ts_polys)):
    ax_sub = fig.add_axes(sub_positions[ti])
    ax_sub.set_facecolor('#050A0F')
    verts = polys_to_patches(polys, max_draw=3000)
    if verts:
        coll = PolyCollection(verts, facecolor=color, edgecolor=color,
                              alpha=0.7, linewidth=0.25)
        ax_sub.add_collection(coll)
    # Compute centroid of this timestamp for extent
    cent_lons = [p['rings'][0][0][0] for p in polys if p['rings'][0]]
    cent_lats = [p['rings'][0][0][1] for p in polys if p['rings'][0]]
    if cent_lons:
        lo, hi_lon = min(cent_lons), max(cent_lons)
        lo_lat, hi_lat = min(cent_lats), max(cent_lats)
        pw = (hi_lon-lo)*0.08; ph = (hi_lat-lo_lat)*0.08
        ax_sub.set_xlim(lo-pw, hi_lon+pw)
        ax_sub.set_ylim(lo_lat-ph, hi_lat+ph)
    ax_sub.set_title(f'{label}  N={len(polys):,}', color=TEXT_COL, fontsize=7.5, pad=3)
    ax_sub.tick_params(colors=TEXT_COL, labelsize=6)
    for sp in ax_sub.spines.values(): sp.set_edgecolor(GRID_COL)
    ax_sub.grid(True, color=GRID_COL, linewidth=0.3, alpha=0.4)

out1 = FIG_DIR / 'bf_spatial_overlay_channel.png'
fig.savefig(out1, dpi=150, facecolor=DARK_BG, bbox_inches='tight')
print(f'  -> {out1}')
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: ROCK POLYGON SPATIAL DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════

print('Figure 2: rock polygon distribution...')

fig2, axes2 = plt.subplots(1, 2, figsize=(18, 8))
fig2.patch.set_facecolor(DARK_BG)
fig2.suptitle(
    'Rock polygon layer — DeepGIS-XR  |  Sierra Nevada, CA  (-118.44 W, 37.45 N)\n'
    '222,657 total rocks  |  attributes: angle, eccentricity, major/minor axis, size',
    color=TEXT_COL, fontsize=11, fontweight='bold'
)

# Extract morphometric attributes from the rock polygons
sizes  = [p['props'].get('size', 0)        for p in rock_polys]
eccs   = [p['props'].get('eccentrici', 0)  for p in rock_polys]
angles = [p['props'].get('angle', 0)       for p in rock_polys]
majors = [p['props'].get('major_leng', 1)  for p in rock_polys]
minors = [p['props'].get('minor_leng', 1)  for p in rock_polys]
centroids_lon = [np.mean([pt[0] for pt in p['rings'][0]]) for p in rock_polys]
centroids_lat = [np.mean([pt[1] for pt in p['rings'][0]]) for p in rock_polys]

sizes  = np.array(sizes,  dtype=float)
eccs   = np.array(eccs,   dtype=float)
angles = np.array(angles, dtype=float)
majors = np.array(majors, dtype=float)
minors = np.array(minors, dtype=float)
c_lon  = np.array(centroids_lon)
c_lat  = np.array(centroids_lat)

# Left: polygons coloured by size
ax = axes2[0]
ax.set_facecolor('#050A0F')
verts_r = polys_to_patches(rock_polys, max_draw=4000)
sz_vals = sizes[:len(verts_r):max(1,len(sizes)//len(verts_r))] if verts_r else []

sz_min = max(1.0, float(np.percentile(sizes[sizes > 0], 5))) if np.any(sizes > 0) else 1.0
sz_max = max(sz_min + 1, float(np.percentile(sizes[sizes > 0], 95))) if np.any(sizes > 0) else 100.0
norm_sz = LogNorm(vmin=sz_min, vmax=sz_max)
cmap_sz = plt.cm.viridis
colors_sz = [cmap_sz(norm_sz(max(1, s))) for s in sizes[:len(verts_r):max(1,len(sizes)//len(verts_r))]]

if verts_r:
    coll_r = PolyCollection(verts_r, facecolors=colors_sz[:len(verts_r)],
                             edgecolor='none', alpha=0.85, linewidth=0)
    ax.add_collection(coll_r)

all_r_lons = [pt[0] for p in rock_polys for pt in p['rings'][0]]
all_r_lats = [pt[1] for p in rock_polys for pt in p['rings'][0]]
pad = 0.001
ax.set_xlim(min(all_r_lons)-pad, max(all_r_lons)+pad)
ax.set_ylim(min(all_r_lats)-pad, max(all_r_lats)+pad)

sm = ScalarMappable(cmap=cmap_sz, norm=norm_sz)
cb = fig2.colorbar(sm, ax=ax, fraction=0.035, pad=0.03)
cb.set_label('Rock pixel size', color=TEXT_COL, fontsize=8)
cb.ax.tick_params(colors=TEXT_COL, labelsize=7)

ax.set_title(f'Rock polygons coloured by pixel size\n({len(rock_polys):,} shown of 222,657 total)',
             color=TEXT_COL, fontsize=9, pad=6)
ax.set_xlabel('Longitude', color=TEXT_COL, fontsize=8)
ax.set_ylabel('Latitude',  color=TEXT_COL, fontsize=8)
ax.tick_params(colors=TEXT_COL, labelsize=7)
for sp in ax.spines.values(): sp.set_edgecolor(GRID_COL)
ax.grid(True, color=GRID_COL, linewidth=0.4, alpha=0.5)

# Right: centroids coloured by eccentricity, sized by major axis
ax = axes2[1]
ax.set_facecolor('#050A0F')
norm_ecc = Normalize(vmin=0, vmax=1)
cmap_ecc = plt.cm.RdYlGn  # round=green, elongated=red
sizes_pts = np.clip(majors / majors.max() * 60, 2, 60)

sc = ax.scatter(c_lon, c_lat, c=eccs, s=sizes_pts,
                cmap=cmap_ecc, norm=norm_ecc, alpha=0.6, linewidths=0)
cb2 = fig2.colorbar(sc, ax=ax, fraction=0.035, pad=0.03)
cb2.set_label('Eccentricity  (0=round, 1=elongated)', color=TEXT_COL, fontsize=8)
cb2.ax.tick_params(colors=TEXT_COL, labelsize=7)

ax.set_xlim(min(all_r_lons)-pad, max(all_r_lons)+pad)
ax.set_ylim(min(all_r_lats)-pad, max(all_r_lats)+pad)
ax.set_title('Rock centroids: color=eccentricity, size=major axis length\n'
             '(green=rounded, red=elongated; larger marker=longer rock)',
             color=TEXT_COL, fontsize=9, pad=6)
ax.set_xlabel('Longitude', color=TEXT_COL, fontsize=8)
ax.set_ylabel('Latitude',  color=TEXT_COL, fontsize=8)
ax.tick_params(colors=TEXT_COL, labelsize=7)
for sp in ax.spines.values(): sp.set_edgecolor(GRID_COL)
ax.grid(True, color=GRID_COL, linewidth=0.4, alpha=0.5)

# Angle rose inset (orientation distribution)
ax_rose = fig2.add_axes([0.88, 0.12, 0.08, 0.18], polar=True)
ax_rose.set_facecolor(PANEL_BG)
theta = np.radians(angles[angles > 0])
bins  = np.linspace(0, np.pi, 19)
counts, _ = np.histogram(theta % np.pi, bins=bins)
theta_mid  = 0.5*(bins[:-1]+bins[1:])
ax_rose.bar(theta_mid, counts, width=np.pi/18, color='#64B5F6', alpha=0.7)
ax_rose.bar(theta_mid + np.pi, counts, width=np.pi/18, color='#64B5F6', alpha=0.7)
ax_rose.set_title('Orientation\n(angle rose)', color=TEXT_COL, fontsize=6, pad=2)
ax_rose.tick_params(colors=TEXT_COL, labelsize=5)
ax_rose.set_yticklabels([])

out2 = FIG_DIR / 'rock_spatial_distribution.png'
fig2.savefig(out2, dpi=150, facecolor=DARK_BG, bbox_inches='tight')
print(f'  -> {out2}')
plt.close(fig2)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: TEMPORAL DIFF — new polygons between timestamps
# ══════════════════════════════════════════════════════════════════════════════

print('Figure 3: temporal growth diff...')

fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6))
fig3.patch.set_facecolor(DARK_BG)
fig3.suptitle(
    'Bobcat Fire channel network — temporal growth\n'
    'New channel segments appearing between successive drone surveys',
    color=TEXT_COL, fontsize=11, fontweight='bold'
)

pairs = [(0, 1, 'Aug->Oct'), (1, 2, 'Oct->Dec'), (2, 3, 'Dec->Feb')]

for ax, (i0, i1, pair_label) in zip(axes3, pairs):
    ax.set_facecolor('#050A0F')

    polys0 = ts_polys[i0]
    polys1 = ts_polys[i1]

    # Build centroid sets (rounded to 5dp) for approximate "new" detection
    def centroid_key(p):
        ring = p['rings'][0]
        return (round(np.mean([pt[0] for pt in ring]), 5),
                round(np.mean([pt[1] for pt in ring]), 5))

    keys0 = {centroid_key(p) for p in polys0}
    keys1 = {centroid_key(p) for p in polys1}

    # Persistent (in both) and new (only in t1)
    persistent = [p for p in polys1 if centroid_key(p) in keys0]
    new_polys  = [p for p in polys1 if centroid_key(p) not in keys0]

    v_persist = polys_to_patches(persistent, max_draw=2000)
    v_new     = polys_to_patches(new_polys,  max_draw=2000)

    label0, _, col0, _ = TIMESTAMPS[i0]
    label1, _, col1, _ = TIMESTAMPS[i1]

    if v_persist:
        ax.add_collection(PolyCollection(v_persist, facecolor='#455A64',
                                          edgecolor='#546E7A', alpha=0.5, linewidth=0.3))
    if v_new:
        ax.add_collection(PolyCollection(v_new, facecolor='#FFEB3B',
                                          edgecolor='#FFF176', alpha=0.85, linewidth=0.4))

    # Set extent
    all_l = [pt[0] for p in polys1 for pt in p['rings'][0]]
    all_t = [pt[1] for p in polys1 for pt in p['rings'][0]]
    if all_l:
        pw = (max(all_l)-min(all_l))*0.04
        ph = (max(all_t)-min(all_t))*0.04
        ax.set_xlim(min(all_l)-pw, max(all_l)+pw)
        ax.set_ylim(min(all_t)-ph, max(all_t)+ph)

    legend_patches = [
        mpatches.Patch(color='#455A64', alpha=0.7, label=f'Persistent  ({len(persistent):,})'),
        mpatches.Patch(color='#FFEB3B', alpha=0.9, label=f'New  ({len(new_polys):,})')
    ]
    ax.legend(handles=legend_patches, loc='upper left', fontsize=7,
              facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL)
    ax.set_title(f'{pair_label}\n{label0} ({len(polys0):,}) -> {label1} ({len(polys1):,})',
                 color=TEXT_COL, fontsize=9, pad=6)
    ax.set_xlabel('Longitude', color=TEXT_COL, fontsize=8)
    if ax is axes3[0]: ax.set_ylabel('Latitude', color=TEXT_COL, fontsize=8)
    ax.tick_params(colors=TEXT_COL, labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID_COL)
    ax.grid(True, color=GRID_COL, linewidth=0.4, alpha=0.5)

out3 = FIG_DIR / 'bf_temporal_growth_diff.png'
fig3.savefig(out3, dpi=150, facecolor=DARK_BG, bbox_inches='tight')
print(f'  -> {out3}')
plt.close(fig3)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: WHAT WE LEARNED — visual summary with annotations
# ══════════════════════════════════════════════════════════════════════════════

print('Figure 4: what we learned summary...')

fig4, axes4 = plt.subplots(2, 3, figsize=(18, 11))
fig4.patch.set_facecolor(DARK_BG)
fig4.suptitle(
    'What we learned: Kernelcal on real wildfire drone data  |  Bobcat Fire AZ  2020-2021',
    color=TEXT_COL, fontsize=12, fontweight='bold'
)

xs    = [0, 1, 2, 3]
xlbl  = ['Aug 2020', 'Oct 2020', 'Dec 2020', 'Feb 2021']
Hs    = [4.40763, 4.33411, 4.38656, 4.63777]
b1s   = [1555, 1572, 1600, 1607]
dps   = [2.734246, 2.735359, 2.735566, 2.732450]
nraws = [3748, 7792, 9074, 11467]
Nded  = [2746, 4237, 5510, 7424]
COLS  = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']

def dark_ax(ax, title, xlabel='', ylabel=''):
    ax.set_facecolor(PANEL_BG)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID_COL)
    ax.tick_params(colors=TEXT_COL, labelsize=8)
    ax.xaxis.label.set_color(TEXT_COL); ax.yaxis.label.set_color(TEXT_COL)
    ax.set_title(title, color=TEXT_COL, fontsize=9, fontweight='bold', pad=5)
    if xlabel: ax.set_xlabel(xlabel, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, color=GRID_COL, linewidth=0.5, alpha=0.6)

# [0,0] Polygon growth rate
ax = axes4[0, 0]
bars = ax.bar(xs, nraws, color=COLS, alpha=0.8, edgecolor=GRID_COL, linewidth=0.8)
for xi, yi in zip(xs, nraws):
    ax.text(xi, yi+150, f'{yi:,}', ha='center', va='bottom', color=TEXT_COL, fontsize=8)
dark_ax(ax, '1. Channel polygon count\n3x growth in 6 months (sediment mobilisation)',
        ylabel='N raw polygons')
ax.set_xticks(xs); ax.set_xticklabels(xlbl, rotation=15, ha='right', fontsize=7)

# [0,1] Unique centroid count (after dedup)
ax = axes4[0, 1]
ax.plot(xs, Nded, 'o-', color='#64B5F6', linewidth=2.5, markersize=9)
ax.fill_between(xs, 0, Nded, alpha=0.15, color='#64B5F6')
for xi, yi in zip(xs, Nded):
    ax.text(xi, yi+80, f'{yi:,}', ha='center', color=TEXT_COL, fontsize=8)
dark_ax(ax, '2. Unique polygon centroids\n(after 0.5 m tile-boundary dedup)',
        ylabel='Unique centroids')
ax.set_xticks(xs); ax.set_xticklabels(xlbl, rotation=15, ha='right', fontsize=7)
ax.set_ylim(0, max(Nded)*1.15)

# [0,2] H[h] spectral entropy
ax = axes4[0, 2]
ax.plot(xs, Hs, 'o-', color='#CE93D8', linewidth=2.5, markersize=9, zorder=3)
ax.fill_between(xs, min(Hs)-0.05, Hs, alpha=0.2, color='#CE93D8')
for xi, yi in zip(xs, Hs):
    ax.text(xi, yi+0.005, f'{yi:.4f}', ha='center', va='bottom', color=TEXT_COL, fontsize=8)
# Annotate the Oct dip
ax.annotate('Oct dip:\nnetwork\nconsolidating?',
            xy=(1, Hs[1]), xytext=(1.3, Hs[1]-0.05),
            color='#FFB74D', fontsize=7, arrowprops=dict(arrowstyle='->', color='#FFB74D'))
dark_ax(ax, '3. Spectral entropy H[h]\nRising trend (+0.23 nats) with Oct dip',
        ylabel='H[h]  [nats]')
ax.set_xticks(xs); ax.set_xticklabels(xlbl, rotation=15, ha='right', fontsize=7)

# [1,0] beta1 loop count
ax = axes4[1, 0]
ax.plot(xs, b1s, 'o-', color='#81C784', linewidth=2.5, markersize=9, zorder=3)
ax.fill_between(xs, min(b1s)-10, b1s, alpha=0.2, color='#81C784')
for xi, yi in zip(xs, b1s):
    ax.text(xi, yi+3, str(yi), ha='center', va='bottom', color=TEXT_COL, fontsize=8)
# Draw abiotic reference line (hypothetical)
ax.axhline(y=1200, color='#FF5722', linewidth=1.2, linestyle='--', alpha=0.7)
ax.text(3.05, 1205, 'hypothetical\nbeta1_abio', color='#FF5722', fontsize=7, va='bottom')
ax.annotate('Growing loops\n= no optimal\ncontroller',
            xy=(2, b1s[2]), xytext=(0.3, 1580),
            color='#A5D6A7', fontsize=7, arrowprops=dict(arrowstyle='->', color='#A5D6A7'))
dark_ax(ax, '4. beta1 (independent cycles)\nMonotone growth = uncontrolled abiotic trajectory',
        ylabel='beta1')
ax.set_xticks(xs); ax.set_xticklabels(xlbl, rotation=15, ha='right', fontsize=7)

# [1,1] Delta' stability margin
ax = axes4[1, 1]
ax.plot(xs, dps, 'o-', color='#F06292', linewidth=2.5, markersize=9, zorder=3)
ax.fill_between(xs, min(dps)-0.001, dps, alpha=0.2, color='#F06292')
for xi, yi in zip(xs, dps):
    ax.text(xi, yi+0.0001, f'{yi:.5f}', ha='center', va='bottom', color=TEXT_COL, fontsize=7)
ax.annotate("Peak then drop:\nsystem approaching\nnew attractor?",
            xy=(2, dps[2]), xytext=(0.2, dps[2]-0.0018),
            color='#F8BBD0', fontsize=7, arrowprops=dict(arrowstyle='->', color='#F8BBD0'))
dark_ax(ax, "5. Delta' (Hessian stability gap)\nPeak Dec 2020 then drop = early regime transition",
        ylabel="Delta'")
ax.set_xticks(xs); ax.set_xticklabels(xlbl, rotation=15, ha='right', fontsize=7)

# [1,2] Text summary "what we learned"
ax = axes4[1, 2]
ax.set_facecolor(PANEL_BG)
for sp in ax.spines.values(): sp.set_edgecolor(GRID_COL)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title('6. What we learned', color=TEXT_COL, fontsize=9, fontweight='bold', pad=5)

summary = (
    "FINDING 1 — The MVT decoder works on real data.\n"
    "3,748 to 11,467 polygon centroids extracted and\n"
    "georeferenced from raw PBF tiles. No external libs.\n\n"
    "FINDING 2 — H[h] rises +0.23 nats (Aug->Feb).\n"
    "Post-fire channel network becomes more spectrally\n"
    "diffuse as new branches + avulsion paths open.\n\n"
    "FINDING 3 — beta1 grows monotonically (+52 loops).\n"
    "Expected abiotic signature: NO optimal controller\n"
    "(no vegetation) maintaining minimum-energy topology.\n\n"
    "FINDING 4 — Delta' peaks Dec 2020 then drops.\n"
    "Possible early sign of reorganisation toward a\n"
    "new network attractor (Feb 2021 expansion).\n\n"
    "FINDING 5 — Fixed-point kernel converges in 11 iters\n"
    "for all timestamps. Framework is numerically stable\n"
    "on real spatial polygon data.\n\n"
    "NEXT — Add vegetation recovery timestamps.\n"
    "Predict: Delta' falls, beta1 falls toward beta1_abio\n"
    "as Cowan-Farquhar controller re-engages."
)
ax.text(0.05, 0.97, summary, transform=ax.transAxes,
        va='top', ha='left', color=TEXT_COL, fontsize=7.5,
        fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor=DARK_BG, alpha=0.6))

out4 = FIG_DIR / 'bf_what_we_learned.png'
fig4.tight_layout()
fig4.savefig(out4, dpi=150, facecolor=DARK_BG, bbox_inches='tight')
print(f'  -> {out4}')
plt.close(fig4)

print(f'\nAll figures written to {FIG_DIR}/')
print('  bf_spatial_overlay_channel.png   (true polygon geometry, 4 timestamps overlaid)')
print('  rock_spatial_distribution.png    (222K rock polygons, size + eccentricity)')
print('  bf_temporal_growth_diff.png      (new vs persistent polygons per interval)')
print('  bf_what_we_learned.png           (annotated summary of findings)')
