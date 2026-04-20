#!/usr/bin/env python3
"""
bf_kernelcal_demo.py
====================
Kernelcal spectral diagnostics on the Bobcat Fire (BF) stream-channel
vector time series — AZ site (~-111.265 W, 33.782 N), Tonto National Forest.

Four timestamps (vector MBTiles):
  Aug 02 2020  bf_aug_2020.mbtiles          3,660 polygons
  Oct 03 2020  bf_oct_2020.mbtiles          9,572 polygons
  Dec 20 2020  bf_dec_2020_vector.mbtiles   9,333 polygons
  Feb 15 2021  bf_feb_2021_3d.mbtiles      12,006 polygons

Pipeline per timestamp
----------------------
  1. Decode Mapbox Vector Tile (PBF) polygons from MBTiles (SQLite).
  2. Compute polygon centroids in WGS-84 lon/lat.
  3. Project to a local metric frame (equirectangular, cm-level accuracy).
  4. Deduplicate centroids closer than 0.5 m (tile-boundary artefacts).
  5. Uniform spatial subsample to N_MAX points.
  6. Build symmetric k-NN Gaussian-weighted graph Laplacian (dense N x N).
  7. Run kernelcal terrain diagnostics:
       spectral_entropy_from_laplacian  ->  H[h]        (P1 Remark 7)
       fixed_point_kernel               ->  h*(lambda)  (P1 Corollary 1)
       fiedler_mode_gap                 ->  Delta'      (P1 Corollary 3)
       stability_conservation_tradeoff  ->  D_m = -D'   (P2 Proposition 1b)
  8. Estimate beta0, beta1 from graph topology.
  9. Print a summary table and physical interpretation.

Usage (local, MBTiles under this repo)
--------------------------------------
  cd manuscripts/software-kernelcal-deepgis-integration
  python3 bf_kernelcal_demo.py

Usage (remote DeepGIS-XR deployment)
------------------------------------
  Point DATA_DIR / TIMESTAMPS at your MBTiles tree and ensure ``kernelcal`` is on
  ``PYTHONPATH`` (e.g. editable install from this repo). Example layout::

    python3 /path/to/deepgis-xr/bf_kernelcal_demo.py
"""

from __future__ import annotations

import sys
import math
import gzip
import sqlite3
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

# ── kernelcal ──────────────────────────────────────────────────────────────
KCAL_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(KCAL_ROOT))
from kernelcal.terrain.diagnostics import (
    spectral_entropy_from_laplacian,
    fixed_point_kernel,
    fiedler_mode_gap,
    stability_conservation_tradeoff,
)

# ── CONFIG ──────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent.parent / 'datasets' / 'bf_mbtiles'
N_MAX    = 600    # max centroids after subsampling (dense N x N eigendecomp)
K_NN     = 6     # neighbours for graph construction
SIGMA_M  = 8.0   # RBF bandwidth [metres] for edge weights
DEDUP_M  = 0.5   # deduplication radius [metres]
MU2      = 2.0   # kernelcal fixed-point parameter mu2
SIGMA2   = 1.0   # kernelcal fixed-point parameter sigma2
TARGET_Z = 20    # preferred tile zoom for centroid extraction

TIMESTAMPS = [
    ('Aug-2020', DATA_DIR / 'bf_aug_2020.mbtiles'),
    ('Oct-2020', DATA_DIR / 'bf_oct_2020.mbtiles'),
    ('Dec-2020', DATA_DIR / 'bf_dec_2020_vector.mbtiles'),
    ('Feb-2021', DATA_DIR / 'bf_feb_2021_3d.mbtiles'),
]


# ══════════════════════════════════════════════════════════════════════════════
# MINIMAL MAPBOX VECTOR TILE (MVT / PBF) DECODER
# Handles both packed (wire-2) and unpacked (wire-0) repeated uint32 geometry.
# ══════════════════════════════════════════════════════════════════════════════

def _varint(data: bytes, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        b = data[pos]; pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7


def _zigzag(n: int) -> int:
    return (n >> 1) ^ -(n & 1)


def _decode_geometry(cmds: list[int]) -> list[list[tuple[int, int]]]:
    """MVT geometry command sequence -> list of rings (lists of int (x,y) pairs)."""
    rings: list[list[tuple[int, int]]] = []
    ring:  list[tuple[int, int]] = []
    cx = cy = 0
    i = 0
    while i < len(cmds):
        cmd_int   = cmds[i]; i += 1
        cmd_id    = cmd_int & 0x7
        cmd_count = cmd_int >> 3
        if cmd_id in (1, 2):           # MoveTo / LineTo
            for _ in range(cmd_count):
                cx += _zigzag(cmds[i]); i += 1
                cy += _zigzag(cmds[i]); i += 1
                if cmd_id == 1 and ring:
                    rings.append(ring); ring = []
                ring.append((cx, cy))
        elif cmd_id == 7:              # ClosePath
            if ring:
                rings.append(ring); ring = []
    if ring:
        rings.append(ring)
    return rings


def _parse_feature(data: bytes) -> list[int]:
    """Parse one MVT Feature protobuf blob; return geometry command list."""
    geom: list[int] = []
    pos, N = 0, len(data)
    while pos < N:
        tag, pos = _varint(data, pos)
        wire  = tag & 0x7
        field = tag >> 3
        if wire == 0:
            v, pos = _varint(data, pos)
            if field == 4:             # geometry, unpacked
                geom.append(v)
        elif wire == 2:
            length, pos = _varint(data, pos)
            blob = data[pos:pos + length]; pos += length
            if field == 4:             # geometry, packed uint32
                ip = 0
                while ip < len(blob):
                    v, ip = _varint(blob, ip)
                    geom.append(v)
        elif wire == 5:
            pos += 4
        elif wire == 1:
            pos += 8
        else:
            break
    return geom


def _parse_layer(data: bytes) -> list[tuple[float, float]]:
    """Parse one MVT Layer; return (fx, fy) fractional centroids in [0,1)."""
    extent = 4096
    feat_blobs: list[bytes] = []
    pos, N = 0, len(data)
    while pos < N:
        tag, pos = _varint(data, pos)
        wire  = tag & 0x7
        field = tag >> 3
        if wire == 0:
            v, pos = _varint(data, pos)
            if field == 5:
                extent = v
        elif wire == 2:
            length, pos = _varint(data, pos)
            blob = data[pos:pos + length]; pos += length
            if field == 2:             # Feature
                feat_blobs.append(blob)
        elif wire == 5:
            pos += 4
        elif wire == 1:
            pos += 8
        else:
            break

    centroids: list[tuple[float, float]] = []
    for fb in feat_blobs:
        cmds = _parse_feature(fb)
        if not cmds:
            continue
        for ring in _decode_geometry(cmds):
            if ring:
                fx = float(np.mean([pt[0] for pt in ring])) / extent
                fy = float(np.mean([pt[1] for pt in ring])) / extent
                centroids.append((fx, fy))
    return centroids


def _parse_tile(raw: bytes) -> list[tuple[float, float]]:
    """Decode one raw MBTiles tile blob; return (fx, fy) fractional centroids."""
    try:
        raw = gzip.decompress(raw)
    except Exception:
        pass

    centroids: list[tuple[float, float]] = []
    pos, N = 0, len(raw)
    while pos < N:
        tag, pos = _varint(raw, pos)
        wire  = tag & 0x7
        field = tag >> 3
        if wire == 0:
            _, pos = _varint(raw, pos)
        elif wire == 2:
            length, pos = _varint(raw, pos)
            blob = raw[pos:pos + length]; pos += length
            if field == 3:             # Tile.layers (field 3)
                centroids.extend(_parse_layer(blob))
        elif wire == 5:
            pos += 4
        elif wire == 1:
            pos += 8
        else:
            break
    return centroids


# ══════════════════════════════════════════════════════════════════════════════
# TILE COORDINATE MATHS
# MBTiles: y in TMS convention (y=0 south).  MVT: y=0 north within tile.
# ══════════════════════════════════════════════════════════════════════════════

def tile_bbox(z: int, x: int, y_tms: int) -> tuple[float, float, float, float]:
    """Return (lon_W, lat_S, lon_E, lat_N) degrees for a TMS tile."""
    y_xyz = (1 << z) - 1 - y_tms
    n     = 1 << z
    lon_w = x / n * 360.0 - 180.0
    lon_e = (x + 1) / n * 360.0 - 180.0
    lat_n = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y_xyz / n))))
    lat_s = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * (y_xyz + 1) / n))))
    return lon_w, lat_s, lon_e, lat_n


# ══════════════════════════════════════════════════════════════════════════════
# CENTROID EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_centroids(path: Path, target_zoom: int = TARGET_Z) -> np.ndarray:
    """Extract polygon centroids in (lon, lat) degrees from a vector MBTiles file."""
    con = sqlite3.connect(str(path))
    cur = con.cursor()
    cur.execute('SELECT DISTINCT zoom_level FROM tiles ORDER BY zoom_level DESC')
    zooms = [r[0] for r in cur.fetchall()]
    zoom  = target_zoom if target_zoom in zooms else zooms[0]

    cur.execute(
        'SELECT tile_column, tile_row, tile_data FROM tiles WHERE zoom_level = ?',
        (zoom,)
    )
    rows = cur.fetchall()
    con.close()

    lonlat: list[tuple[float, float]] = []
    for tx, ty_tms, blob in rows:
        lon_w, lat_s, lon_e, lat_n = tile_bbox(zoom, tx, ty_tms)
        span_lon = lon_e - lon_w
        span_lat = lat_n - lat_s
        for fx, fy in _parse_tile(bytes(blob)):
            lonlat.append((lon_w + fx * span_lon,
                           lat_n - fy * span_lat))  # MVT y=0 is north

    return np.array(lonlat) if lonlat else np.empty((0, 2))


# ══════════════════════════════════════════════════════════════════════════════
# COORDINATE UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def lonlat_to_metres(lonlat: np.ndarray) -> np.ndarray:
    """Equirectangular projection centred on the dataset. Returns (East, North) m."""
    lon0 = lonlat[:, 0].mean()
    lat0 = lonlat[:, 1].mean()
    R    = 6_371_000.0
    cos0 = math.cos(math.radians(lat0))
    E    = (lonlat[:, 0] - lon0) * cos0 * (math.pi / 180.0) * R
    N_m  = (lonlat[:, 1] - lat0) * (math.pi / 180.0) * R
    return np.column_stack([E, N_m])


def deduplicate(xy: np.ndarray, radius: float) -> np.ndarray:
    """Remove centroids within `radius` metres of an already-selected point."""
    if len(xy) == 0:
        return xy
    tree = cKDTree(xy)
    kept = np.ones(len(xy), dtype=bool)
    for i in range(len(xy)):
        if not kept[i]:
            continue
        for j in tree.query_ball_point(xy[i], radius):
            if j != i:
                kept[j] = False
    return xy[kept]


def subsample(xy: np.ndarray, n_max: int, seed: int = 42) -> np.ndarray:
    """Uniform random subsample to at most n_max points."""
    if len(xy) <= n_max:
        return xy
    idx = np.random.default_rng(seed).choice(len(xy), size=n_max, replace=False)
    return xy[idx]


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def build_laplacian(xy: np.ndarray, k: int, sigma: float) -> np.ndarray:
    """Symmetric k-NN Gaussian-weighted graph Laplacian L = D - A (dense)."""
    N    = len(xy)
    tree = cKDTree(xy)
    dists, idxs = tree.query(xy, k=k + 1)   # first hit is self

    A = np.zeros((N, N))
    for i in range(N):
        for r in range(1, k + 1):
            j = idxs[i, r]
            w = math.exp(-dists[i, r] ** 2 / (2.0 * sigma ** 2))
            A[i, j] += w
            A[j, i] += w
    A = np.minimum(A, 1.0)                   # cap symmetrised double-adds
    return np.diag(A.sum(axis=1)) - A


def betti_from_laplacian(L: np.ndarray) -> tuple[int, int]:
    """Estimate beta0 (components) and beta1 (loops) from the Laplacian spectrum."""
    eigvals = np.linalg.eigvalsh(L)
    beta0   = int(np.sum(np.abs(eigvals) < 1e-6))
    V       = L.shape[0]
    A       = np.diag(np.diag(L)) - L       # recover adjacency
    E       = int(np.sum(A > 1e-10)) // 2
    beta1   = max(0, E - V + beta0)
    return beta0, beta1


# ══════════════════════════════════════════════════════════════════════════════
# PER-TIMESTAMP ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyse(label: str, path: Path) -> dict | None:
    sep = '-' * 62
    print(f'\n{sep}')
    print(f'  {label}  |  {path.name}')
    print(sep)

    lonlat = extract_centroids(path)
    n_raw  = len(lonlat)
    print(f'  Centroids extracted  : {n_raw}')
    if n_raw < 20:
        print('  [SKIP] Too few centroids.')
        return None

    xy = lonlat_to_metres(lonlat)
    w  = xy[:, 0].max() - xy[:, 0].min()
    h  = xy[:, 1].max() - xy[:, 1].min()
    print(f'  Site extent          : {w:.1f} m (E-W) x {h:.1f} m (N-S)')

    xy = deduplicate(xy, DEDUP_M)
    print(f'  After dedup (r={DEDUP_M}m)  : {len(xy)} centroids')

    xy = subsample(xy, N_MAX)
    N  = len(xy)
    print(f'  Subsampled to N      : {N}')

    L = build_laplacian(xy, K_NN, SIGMA_M)
    print(f'  Laplacian built      : {N}x{N}  k={K_NN}  sigma={SIGMA_M} m')

    beta0, beta1 = betti_from_laplacian(L)
    print(f'  beta0 (components)   : {beta0}')
    print(f'  beta1 (loops)        : {beta1}')

    H = spectral_entropy_from_laplacian(L, tau=1.0)
    print(f'  Spectral entropy H   : {H:.5f} nats')

    h_star, info = fixed_point_kernel(L, mu2=MU2, sigma2=SIGMA2)
    conv = 'yes' if info['converged'] else 'NO (r={:.2e})'.format(info['residual'])
    print(f'  Fixed-point h*       : converged={conv}  ({info["n_iter"]} iters)')

    delta_prime = fiedler_mode_gap(h_star, L, mu2=MU2, sigma2=SIGMA2)
    print(f'  Hessian gap Delta\'   : {delta_prime:.6f}')

    sct = stability_conservation_tradeoff(h_star, L, mu2=MU2, sigma2=SIGMA2)
    deficit = sct['conservation_deficit']
    print(f'  Conserv. deficit     : {deficit:.6f}  (should track Delta\')')

    return dict(label=label, n_raw=n_raw, N=N, H=H,
                delta_prime=delta_prime, deficit=deficit,
                beta0=beta0, beta1=beta1, converged=info['converged'])


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print()
    print('KERNELCAL | Bobcat Fire channel network dynamics')
    print('Site: -111.265 W, 33.782 N  |  Tonto NF, AZ  |  4 timestamps')

    results = []
    for label, path in TIMESTAMPS:
        r = analyse(label, path)
        if r:
            results.append(r)

    if not results:
        print('\n[ERROR] No timestamps produced valid results.')
        return

    # ── Summary table ──────────────────────────────────────────────────────
    col_b0 = 'b0'
    col_b1 = 'b1'
    print()
    print('=' * 78)
    print('SUMMARY TABLE')
    print('=' * 78)
    hdr = (f"{'Timestamp':<11} {'N_polys':>8} {'N_graph':>8} "
           f"{'H[h]':>9} {'Delta_prime':>12} {'deficit':>10} "
           f"{col_b0:>4} {col_b1:>6}")
    print(hdr)
    print('-' * 78)
    for r in results:
        print(
            f"{r['label']:<11} {r['n_raw']:>8} {r['N']:>8} "
            f"{r['H']:>9.5f} {r['delta_prime']:>12.6f} {r['deficit']:>10.6f} "
            f"{r['beta0']:>4} {r['beta1']:>6}"
        )

    # ── Temporal dynamics ─────────────────────────────────────────────────
    print()
    print('=' * 78)
    print('TEMPORAL DYNAMICS  (Aug-2020 -> Feb-2021)')
    print('=' * 78)
    if len(results) >= 2:
        first, last = results[0], results[-1]
        dH   = last['H']           - first['H']
        db1  = last['beta1']       - first['beta1']
        dDp  = last['delta_prime'] - first['delta_prime']
        dN   = last['n_raw']       - first['n_raw']

        up   = lambda x: 'UP  ' if x > 0 else ('DOWN' if x < 0 else 'FLAT')
        print(f"  Polygon count    {up(dN)}   {first['n_raw']} -> {last['n_raw']}  (delta={dN:+d})")
        print(f"  H[h]             {up(dH)}   {first['H']:.5f} -> {last['H']:.5f}  (delta={dH:+.5f})")
        print(f"  Delta' (stab)    {up(dDp)}   {first['delta_prime']:.6f} -> {last['delta_prime']:.6f}  (delta={dDp:+.6f})")
        print(f"  beta1 (loops)    {up(db1)}   {first['beta1']} -> {last['beta1']}  (delta={db1:+d})")

    # ── Physical interpretation ────────────────────────────────────────────
    print()
    print('=' * 78)
    print('PHYSICAL INTERPRETATION')
    print('=' * 78)

    Hs   = [r['H']           for r in results]
    Dps  = [r['delta_prime'] for r in results]
    b1s  = [r['beta1']       for r in results]
    Nraw = [r['n_raw']       for r in results]

    dH_total = Hs[-1]  - Hs[0]  if len(Hs)  > 1 else 0.0
    db1_tot  = b1s[-1] - b1s[0] if len(b1s) > 1 else 0
    dDp_tot  = Dps[-1] - Dps[0] if len(Dps) > 1 else 0.0
    dN_tot   = Nraw[-1]- Nraw[0]if len(Nraw)> 1 else 0

    print()
    if dH_total > 0.05:
        print(f'  H[h] RISING (+{dH_total:.4f} nats).')
        print('  Channel graph becoming more spectrally diffuse: new branches,')
        print('  avulsion paths, and debris-jam bypasses added post-fire.')
    elif dH_total < -0.05:
        print(f'  H[h] FALLING ({dH_total:.4f} nats).')
        print('  Network concentrating: incision or pruning simplifying the graph.')
    else:
        print(f'  H[h] roughly stable (delta = {dH_total:.4f} nats).')

    print()
    if dN_tot > 0:
        print(f'  Polygon count GROWING ({Nraw[0]} -> {Nraw[-1]}, +{dN_tot}).')
        print('  More channel-feature segments: post-fire sediment mobilisation')
        print('  expanding the active network.')
    else:
        print(f'  Polygon count SHRINKING ({Nraw[0]} -> {Nraw[-1]}, {dN_tot}).')

    print()
    if db1_tot > 0:
        print(f'  beta1 GROWING (delta = +{db1_tot}).')
        print('  More independent loops: avulsion, fan-head switching, debris-jam')
        print('  bypasses creating new cycle structure — expected signature of a')
        print('  network WITHOUT an active optimal controller (no vegetation, no OCN).')
    elif db1_tot < 0:
        print(f'  beta1 SHRINKING (delta = {db1_tot}).')
        print('  Fewer loops: network simplifying toward a tree (incision dominant).')
    else:
        print('  beta1 unchanged across series.')

    print()
    mean_Dp = float(np.mean(Dps))
    print(f'  Stability gap Delta\' mean = {mean_Dp:.5f}.')
    print(f'  Conservation deficit tracks Delta\' (D_m = -Delta\' identity, Route 3).')
    if dDp_tot > 0:
        print('  Growing Delta\': fixed point MORE stable but leaks MORE information.')
        print('  System drifting further from OCN optimal-controller condition.')
    else:
        print('  Shrinking Delta\': system moving toward optimal-controller condition.')

    print()
    print('  KEY PREDICTION (P4 Hypothesis 2):')
    print('  If vegetation recovers, Delta\' should decrease and beta1 should')
    print('  return toward the abiotic baseline as the Cowan-Farquhar controller')
    print('  re-engages.  Monitoring these two numbers across timestamps is a')
    print('  ground-truth calibration of the topological biosignature on Earth.')
    print()
    print('Done.')


if __name__ == '__main__':
    main()
