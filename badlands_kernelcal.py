#!/usr/bin/env python3
"""
badlands_kernelcal.py — Abiotic null for the three-point controller narrative.

Replicates the GEE snippet:
    dem = ee.ImageCollection("USGS/3DEP/1m")
        .filterBounds(ee.Geometry.Rectangle([-111.82, 34.85, -111.70, 35.02]))
        .mosaic()
but via direct USGS TNM API download — no GEE account needed.

This is the Arizona "badlands" patch (Coconino County, AZ — Verde/Kaibab
arid plateau, sparse vegetation, no human controller).  Expected result:
    DeltaH ~ 0    (no controller concentrating the kernel)
    DeltaBeta1 ~ 0  (no excess topology above drainage null)

Compare against:
    Jezero delta:   DeltaH = -0.38,  DeltaBeta1 = +960   (fossil controller)
    Cities:         DeltaH = -0.62 to -0.69,  DeltaBeta1 = +470 to +1513
"""

from __future__ import annotations

import math
import os
import sys
import urllib.request
import warnings
from pathlib import Path

import numpy as np
from scipy.ndimage import label as scipy_label
from scipy.spatial import cKDTree
from scipy.linalg import eigh as scipy_eigh

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

KCAL_ROOT = Path(__file__).parent
sys.path.insert(0, str(KCAL_ROOT))
from kernelcal.terrain.diagnostics import (
    spectral_entropy,
    fixed_point_kernel,
    fiedler_mode_gap,
    stability_conservation_tradeoff,
)

try:
    import rasterio as _rasterio
    import rasterio
    import rasterio.transform
    import rasterio.crs
    from rasterio.merge import merge as rio_merge
    from rasterio.warp import reproject, Resampling, calculate_default_transform
    from rasterio.transform import from_origin
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

# ── CONFIG ──────────────────────────────────────────────────────────────────
BBOX        = (-111.82, 34.85, -111.70, 35.02)   # lon_min, lat_min, lon_max, lat_max
SITE_NAME   = 'AZ_Coconino_Badlands'
DATA_DIR    = KCAL_ROOT / 'datasets' / 'badlands'
FIG_DIR     = KCAL_ROOT / 'badlands_figures'
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

DOWNSAMPLE_M   = 10       # working resolution in metres (1m → 10m)
ACC_THR_FACTOR = 0.005    # flow-accumulation threshold fraction
N_MAX          = 1500     # max graph nodes
K_NN           = 6        # k-NN for graph
MU2            = 2.0
SIGMA2         = 1.0
BG             = '#0d1117'

# Known tiles (from TNM API query — prefer Coconino_2019 as primary survey)
TILE_URLS = [
    'https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1m/Projects/AZ_Coconino_2019_B19/TIFF/USGS_1M_12_x42y386_AZ_Coconino_2019_B19.tif',
    'https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1m/Projects/AZ_Coconino_2019_B19/TIFF/USGS_1M_12_x42y387_AZ_Coconino_2019_B19.tif',
    'https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1m/Projects/AZ_Coconino_2019_B19/TIFF/USGS_1M_12_x42y388_AZ_Coconino_2019_B19.tif',
    'https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1m/Projects/AZ_Coconino_2019_B19/TIFF/USGS_1M_12_x43y386_AZ_Coconino_2019_B19.tif',
    'https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1m/Projects/AZ_Coconino_2019_B19/TIFF/USGS_1M_12_x43y387_AZ_Coconino_2019_B19.tif',
    'https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1m/Projects/AZ_Coconino_2019_B19/TIFF/USGS_1M_12_x43y388_AZ_Coconino_2019_B19.tif',
]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: DOWNLOAD TILES
# ══════════════════════════════════════════════════════════════════════════════

def download_tiles() -> list[Path]:
    paths = []
    for url in TILE_URLS:
        fname = DATA_DIR / Path(url).name
        if fname.exists():
            print(f'  [cache] {fname.name}  ({fname.stat().st_size // 1024 // 1024} MB)')
        else:
            print(f'  [download] {fname.name} …', end='', flush=True)
            req = urllib.request.Request(url, headers={'User-Agent': 'kernelcal/1.0'})
            try:
                with urllib.request.urlopen(req, timeout=300) as r, open(fname, 'wb') as f:
                    total = 0
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        total += len(chunk)
                print(f'  {total // 1024 // 1024} MB')
            except Exception as exc:
                print(f'  FAILED: {exc}')
                if fname.exists():
                    fname.unlink()
                continue
        paths.append(fname)
    return paths


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: MOSAIC AND DOWNSAMPLE
# ══════════════════════════════════════════════════════════════════════════════

def mosaic_and_downsample(tile_paths: list[Path]) -> tuple[np.ndarray, object]:
    """Merge tiles, reproject to UTM, downsample to DOWNSAMPLE_M resolution.

    Returns (dem_array, rasterio_transform).
    """
    if not HAS_RASTERIO:
        raise ImportError('rasterio required: pip install rasterio')

    print(f'  Mosaicking {len(tile_paths)} tiles …')
    srcs = [rasterio.open(p) for p in tile_paths]
    mosaic_arr, mosaic_transform = rio_merge(srcs, nodata=-9999)
    meta = srcs[0].meta.copy()
    crs_src = srcs[0].crs
    for s in srcs:
        s.close()

    elev = mosaic_arr[0].astype(float)
    elev[elev == -9999] = np.nan
    print(f'  Mosaic shape: {elev.shape}  elev range: '
          f'{np.nanmin(elev):.0f} – {np.nanmax(elev):.0f} m')

    # Reproject to UTM Zone 12N (EPSG:32612)
    crs_dst = rasterio.crs.CRS.from_epsg(32612)
    dst_transform, dst_w, dst_h = calculate_default_transform(
        crs_src, crs_dst, elev.shape[1], elev.shape[0],
        *rasterio.transform.array_bounds(elev.shape[0], elev.shape[1],
                                         mosaic_transform)
    )

    elev_utm = np.full((dst_h, dst_w), np.nan, dtype=float)
    reproject(
        source=elev,
        destination=elev_utm,
        src_transform=mosaic_transform,
        src_crs=crs_src,
        dst_transform=dst_transform,
        dst_crs=crs_dst,
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    print(f'  UTM shape: {elev_utm.shape}  pixel size: ~1 m')

    # Downsample by block averaging
    factor = DOWNSAMPLE_M
    rows, cols = elev_utm.shape
    rows_c = (rows // factor) * factor
    cols_c = (cols // factor) * factor
    crop   = elev_utm[:rows_c, :cols_c]
    ds     = np.nanmean(
                 crop.reshape(rows_c // factor, factor,
                              cols_c // factor, factor),
                 axis=(1, 3))
    # Adjust transform for downsampled resolution
    x0 = dst_transform.c
    y0 = dst_transform.f
    ds_transform = from_origin(x0, y0,
                               dst_transform.a * factor,
                               -dst_transform.e * factor)
    print(f'  Downsampled shape: {ds.shape}  pixel size: {DOWNSAMPLE_M} m')
    return ds, ds_transform


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: FLOW ACCUMULATION AND NETWORK EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def d8_flow_direction(dem: np.ndarray) -> np.ndarray:
    """Vectorised D8 flow-direction (returns 0–7 indices into 8-neighbour)."""
    rows, cols = dem.shape
    padded = np.pad(dem, 1, mode='edge')
    nbr_offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    drops = np.full((rows, cols, 8), -np.inf)
    for k, (dr, dc) in enumerate(nbr_offsets):
        nbr = padded[1+dr:1+dr+rows, 1+dc:1+dc+cols]
        dist = math.sqrt(2) if abs(dr) + abs(dc) == 2 else 1.0
        drops[:, :, k] = (dem - nbr) / dist
    fdir = np.argmax(drops, axis=2)
    return fdir.astype(np.int8)


def flow_accumulation(fdir: np.ndarray) -> np.ndarray:
    rows, cols = fdir.shape
    acc = np.ones((rows, cols), dtype=np.int32)
    nbr_offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    receiver_delta = np.array(nbr_offsets, dtype=int)

    # Build in-degree map
    in_deg = np.zeros((rows, cols), dtype=np.int32)
    for r in range(rows):
        for c in range(cols):
            k  = fdir[r, c]
            nr = r + receiver_delta[k, 0]
            nc = c + receiver_delta[k, 1]
            if 0 <= nr < rows and 0 <= nc < cols and (nr != r or nc != c):
                in_deg[nr, nc] += 1

    # Topological sort (Kahn's algorithm)
    from collections import deque
    queue = deque()
    for r in range(rows):
        for c in range(cols):
            if in_deg[r, c] == 0:
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        k  = fdir[r, c]
        nr = r + receiver_delta[k, 0]
        nc = c + receiver_delta[k, 1]
        if 0 <= nr < rows and 0 <= nc < cols and (nr != r or nc != c):
            acc[nr, nc] += acc[r, c]
            in_deg[nr, nc] -= 1
            if in_deg[nr, nc] == 0:
                queue.append((nr, nc))
    return acc


def extract_network_nodes(dem: np.ndarray,
                          acc: np.ndarray,
                          transform) -> tuple[np.ndarray, np.ndarray]:
    """Return (xy_metres, elevations) for high-accumulation channel nodes."""
    channel_threshold = int(ACC_THR_FACTOR * acc.max())
    channel_threshold = max(channel_threshold, 5)
    mask = (acc >= channel_threshold) & np.isfinite(dem)
    rows_idx, cols_idx = np.where(mask)
    n_raw = len(rows_idx)
    print(f'  Channel threshold: {channel_threshold}  raw nodes: {n_raw:,}')

    if n_raw == 0:
        raise ValueError('No channel nodes above threshold — lower ACC_THR_FACTOR')

    # Pixel centres in UTM
    px_size = transform.a
    x0      = transform.c + px_size / 2
    y0      = transform.f - px_size / 2   # y decreases downward
    xs = x0 + cols_idx * px_size
    ys = y0 - rows_idx * px_size
    elevs = dem[rows_idx, cols_idx]

    xy = np.column_stack([xs, ys])

    # Keep highest-accumulation nodes up to N_MAX
    if n_raw > N_MAX:
        acc_vals = acc[rows_idx, cols_idx]
        top_idx  = np.argsort(acc_vals)[::-1][:N_MAX]
        xy    = xy[top_idx]
        elevs = elevs[top_idx]
        print(f'  Subsampled to top-{N_MAX} by accumulation')

    return xy, elevs


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: GRAPH + LAPLACIAN + KERNELCAL
# ══════════════════════════════════════════════════════════════════════════════

def build_graph_and_run(xy: np.ndarray) -> dict:
    n = len(xy)
    tree = cKDTree(xy)
    dists, inds = tree.query(xy, k=K_NN + 1)

    median_nn = float(np.median(dists[:, 1]))
    xr = xy[:, 0].max() - xy[:, 0].min()
    yr = xy[:, 1].max() - xy[:, 1].min()
    diag  = math.hypot(xr, yr)
    sigma = max(0.05 * max(diag, 1.0), 2 * max(median_nn, 1e-3))

    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j_idx in range(1, K_NN + 1):
            j = inds[i, j_idx]
            d = dists[i, j_idx]
            w = math.exp(-d**2 / sigma**2)
            if w > W[i, j]:
                W[i, j] = w
                W[j, i] = w

    D  = np.diag(W.sum(axis=1))
    L  = D - W
    eigvals, eigvecs = scipy_eigh(L)
    eigvals = np.maximum(eigvals, 0.0)

    n_zero  = int(np.sum(eigvals < 1e-6))
    w_modes = eigvals.copy()
    w_modes[:n_zero] = eigvals[n_zero] if n_zero < n else 1e-3

    h0     = np.maximum(np.exp(-eigvals), 1e-10)
    h_star, info = fixed_point_kernel(L, h0=h0, mu2=MU2, sigma2=SIGMA2, w=w_modes)
    h_star = np.maximum(h_star, 1e-8)

    H_obs   = spectral_entropy(h_star)
    H_vac   = spectral_entropy(h0)
    delta_H = H_obs - H_vac
    delta_p = fiedler_mode_gap(h_star, L, mu2=MU2, sigma2=SIGMA2, w=w_modes)

    n_edges     = int((W > 0).sum()) // 2
    beta0       = n_zero
    beta1       = max(0, n_edges - (n - beta0))
    e_null      = K_NN * n // 2
    delta_beta1 = beta1 - max(0, e_null - (n - 1))
    lam_fiedler = float(eigvals[n_zero]) if n_zero < n else 0.0

    return {
        'L': L, 'W': W, 'eigvals': eigvals, 'eigvecs': eigvecs,
        'h0': h0, 'h_star': h_star,
        'H_obs': H_obs, 'H_vac': H_vac, 'delta_H': delta_H,
        'delta_prime': delta_p,
        'beta0': beta0, 'beta1': beta1, 'delta_beta1': delta_beta1,
        'lam_fiedler': lam_fiedler,
        'n_edges': n_edges, 'n': n,
        'converged': info['converged'], 'n_iter': info['n_iter'],
    }


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def make_figures(dem: np.ndarray, acc: np.ndarray, xy: np.ndarray,
                 elevs: np.ndarray, d: dict):

    fig = plt.figure(figsize=(20, 13), facecolor=BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ev = d['eigvals']
    sort = np.argsort(ev)
    lam_sorted = ev[sort]
    hs = d['h_star'][sort]
    h0s = d['h0'][sort]

    # ── Panel A: DEM hillshade ────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_facecolor('#111827')
    valid = np.isfinite(dem)
    dem_show = dem.copy()
    dem_show[~valid] = np.nanmin(dem)
    ax_a.imshow(dem_show, cmap='terrain', origin='upper', aspect='equal')
    ax_a.set_title('DEM  (Arizona arid plateau\n10 m res, USGS 3DEP 1m)',
                   color='white', fontsize=9, fontweight='bold')
    ax_a.axis('off')
    ax_a.text(0.02, 0.03, f'elev: {np.nanmin(dem):.0f}–{np.nanmax(dem):.0f} m',
              transform=ax_a.transAxes, color='white', fontsize=7.5, va='bottom')

    # ── Panel B: Flow accumulation (log) ─────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_facecolor('#111827')
    log_acc = np.log1p(acc.astype(float))
    ax_b.imshow(log_acc, cmap='Blues', origin='upper', aspect='equal')
    ax_b.set_title('Flow accumulation  (log₁₀)\nD8 routing',
                   color='white', fontsize=9, fontweight='bold')
    ax_b.axis('off')

    # ── Panel C: Channel network nodes ───────────────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.set_facecolor('#111827')
    ax_c.scatter(xy[:, 0], xy[:, 1], c=elevs, cmap='terrain',
                 s=2, alpha=0.7, linewidths=0)
    ax_c.set_aspect('equal')
    ax_c.set_title(f'Channel network nodes  (N={d["n"]})\nk-NN proximity graph',
                   color='white', fontsize=9, fontweight='bold')
    ax_c.tick_params(colors='white', labelsize=7)
    for sp in ax_c.spines.values(): sp.set_edgecolor('gray')
    ax_c.text(0.02, 0.03,
              f'β₀={d["beta0"]}  β₁={d["beta1"]}  Δβ₁={d["delta_beta1"]:+d}',
              transform=ax_c.transAxes, color='white', fontsize=7.5, va='bottom')

    # ── Panel D: Eigenspectrum ────────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 0])
    ax_d.set_facecolor('#1a1f2e')
    n_show = min(80, len(ev))
    ax_d.bar(range(n_show), ev[:n_show], color='#4db8ff', width=1.0,
             edgecolor='none', alpha=0.85)
    ax_d.axvline(d['beta0'], color='white', lw=1.5, ls='--', alpha=0.7)
    ax_d.set_title('Laplacian eigenspectrum', color='white', fontsize=9, fontweight='bold')
    ax_d.set_xlabel('Mode index  l', color='white', fontsize=8)
    ax_d.set_ylabel('λₗ', color='white', fontsize=8)
    ax_d.tick_params(colors='white', labelsize=7)
    for sp in ax_d.spines.values(): sp.set_edgecolor('gray')
    ax_d.text(0.97, 0.97,
              f'λ_f={d["lam_fiedler"]:.4f}\nβ₀={d["beta0"]}',
              transform=ax_d.transAxes, ha='right', va='top',
              color='lightgray', fontsize=7.5,
              bbox=dict(boxstyle='round,pad=0.3', fc='black', alpha=0.5))

    # ── Panel E: h* vs h₀ ────────────────────────────────────────────────
    ax_e = fig.add_subplot(gs[1, 1])
    ax_e.set_facecolor('#1a1f2e')
    ax_e.fill_between(lam_sorted, h0s, hs, where=(hs >= h0s),
                      color='#22cc88', alpha=0.30, label='amplified')
    ax_e.fill_between(lam_sorted, h0s, hs, where=(hs < h0s),
                      color='#ff6644', alpha=0.30, label='suppressed')
    ax_e.plot(lam_sorted, h0s, '--', color='#888888', lw=1.5, label='h₀ vacuum')
    ax_e.plot(lam_sorted, hs, '-', color='#4db8ff', lw=2.2, label='h* fixed-pt')
    ax_e.set_title('MaxCal fixed-point kernel\nh*(λ) vs vacuum h₀(λ)',
                   color='white', fontsize=9, fontweight='bold')
    ax_e.set_xlabel('λₗ', color='white', fontsize=8)
    ax_e.set_ylabel('h(λ)', color='white', fontsize=8)
    ax_e.tick_params(colors='white', labelsize=7)
    for sp in ax_e.spines.values(): sp.set_edgecolor('gray')
    ax_e.legend(facecolor='black', edgecolor='gray', labelcolor='white',
                fontsize=7, loc='upper right')
    ax_e.text(0.03, 0.97,
              f'ΔH = {d["delta_H"]:+.3f} nats',
              transform=ax_e.transAxes, va='top',
              color='#ffdd66', fontsize=10, fontweight='bold',
              bbox=dict(boxstyle='round,pad=0.35', fc='black', alpha=0.7))

    # ── Panel F: Three-point narrative ────────────────────────────────────
    ax_f = fig.add_subplot(gs[1, 2])
    ax_f.set_facecolor('#1a1f2e')

    systems    = ['Badlands\n(this run)', 'Jezero\ndelta', 'Cities\n(mean)']
    dH_vals    = [d['delta_H'], -0.38, -0.645]
    db1_vals   = [d['delta_beta1'], 960, 960]   # Jezero Δβ₁ from field note 39
    colors_sys = ['#4db8ff', '#f7c59f', '#ff6b35']
    labels_sys = ['Abiotic', 'Fossil\ncontroller', 'Active\ncontroller']

    sc = ax_f.scatter(dH_vals, db1_vals, s=220, c=colors_sys,
                      edgecolors='white', linewidths=1.0, zorder=5)
    for i, (sx, sy, lab, clab) in enumerate(
            zip(dH_vals, db1_vals, systems, labels_sys)):
        ax_f.annotate(lab, (sx, sy),
                      textcoords='offset points', xytext=(8, 6),
                      color=colors_sys[i], fontsize=8.5, fontweight='bold')

    ax_f.axvline(0, color='white', lw=0.8, ls='--', alpha=0.4)
    ax_f.axhline(0, color='white', lw=0.8, ls='--', alpha=0.4)
    ax_f.set_xlabel('ΔH  (nats)\n← more controlled      abiotic →',
                    color='white', fontsize=9)
    ax_f.set_ylabel('Δβ₁  (topological excess)', color='white', fontsize=9)
    ax_f.set_title('Three-Point Controller Narrative\nΔH × Δβ₁ phase space',
                   color='white', fontsize=9, fontweight='bold')
    ax_f.tick_params(colors='white', labelsize=8)
    for sp in ax_f.spines.values(): sp.set_edgecolor('gray')

    fig.suptitle(
        f'Badlands Abiotic Null  ·  {SITE_NAME}  ·  kernelcal MaxCal\n'
        f'ΔH = {d["delta_H"]:+.3f}  Δβ₁ = {d["delta_beta1"]:+d}  '
        f'λ_fiedler = {d["lam_fiedler"]:.4f}  '
        f'(convergence in {d["n_iter"]} iters)',
        color='white', fontsize=12, fontweight='bold', y=1.02
    )

    out = FIG_DIR / f'{SITE_NAME}_kernelcal.png'
    fig.savefig(out, dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'\n  Saved {out}')
    return out


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print('=' * 65)
    print('Badlands Abiotic Null  —  kernelcal MaxCal Pipeline')
    print(f'  BBOX:  {BBOX}  (Arizona arid plateau, Coconino Co.)')
    print(f'  Data:  {DATA_DIR}')
    print(f'  Figs:  {FIG_DIR}')
    print('=' * 65)

    if not HAS_RASTERIO:
        print('\nERROR: rasterio not installed.  Run:')
        print('  pip install --break-system-packages rasterio')
        return

    # 1. Download
    print('\n[1] Downloading USGS 3DEP 1m tiles …')
    tile_paths = download_tiles()
    if not tile_paths:
        print('ERROR: No tiles downloaded.')
        return

    # 2. Mosaic + downsample
    print('\n[2] Mosaicking and downsampling …')
    dem, transform = mosaic_and_downsample(tile_paths)

    # 3. Flow routing
    print('\n[3] D8 flow routing …')
    # Fill isolated NaNs with local mean for flow routing
    dem_filled = dem.copy()
    nan_mask   = ~np.isfinite(dem_filled)
    if nan_mask.any():
        from scipy.ndimage import generic_filter as _gf
        dem_filled[nan_mask] = _gf(
            dem, np.nanmean, size=5, mode='reflect')[nan_mask]
    fdir = d8_flow_direction(dem_filled)
    acc  = flow_accumulation(fdir)
    print(f'  Max accumulation: {acc.max():,}')

    # 4. Extract nodes
    print('\n[4] Extracting channel network nodes …')
    xy, elevs = extract_network_nodes(dem_filled, acc, transform)
    print(f'  Final graph nodes: {len(xy)}')

    # 5. Graph + kernelcal
    print('\n[5] Building k-NN graph and running kernelcal …')
    d = build_graph_and_run(xy)
    print(f'  Converged: {d["converged"]}  ({d["n_iter"]} iters)')

    # 6. Print results
    print('\n' + '=' * 65)
    print('BADLANDS ABIOTIC NULL  —  RESULTS')
    print('=' * 65)
    print(f'  N nodes:      {d["n"]}')
    print(f'  N edges:      {d["n_edges"]}')
    print(f'  β₀:           {d["beta0"]}')
    print(f'  β₁:           {d["beta1"]}')
    print(f'  Δβ₁:          {d["delta_beta1"]:+d}')
    print(f'  λ_fiedler:    {d["lam_fiedler"]:.4f}')
    print(f'  H[h₀]:        {d["H_vac"]:.3f}')
    print(f'  H[h*]:        {d["H_obs"]:.3f}')
    print(f'  ΔH:           {d["delta_H"]:+.3f}  nats')
    print(f"  Δ':           {d['delta_prime']:.3f}")
    print()

    # 7. Three-point narrative comparison
    print('THREE-POINT NARRATIVE  (ΔH × Δβ₁)')
    print(f'  {"System":<22} {"ΔH":>7} {"Δβ₁":>8}  Controller')
    print(f'  {"-"*55}')
    print(f'  {"Badlands (this run)":<22} {d["delta_H"]:>+7.3f} {d["delta_beta1"]:>+8d}  '
          f'{"ABIOTIC NULL" if abs(d["delta_H"]) < 0.15 else "WEAK CONTROLLER"}')
    print(f'  {"Jezero delta":<22}   -0.380     +960  Fossil controller')
    print(f'  {"Cities (median)":<22}   -0.645    +1050  Active controller')
    print('=' * 65)

    if abs(d['delta_H']) < 0.15 and abs(d['delta_beta1']) < 200:
        print('\n✓  Abiotic null confirmed:  |ΔH| << 0.38  and  Δβ₁ ~ 0')
        print('   Three-point narrative is complete.')
    elif abs(d['delta_H']) < 0.38:
        print('\n~  Partial abiotic null:  ΔH between Badlands prediction and Jezero')
    else:
        print('\n⚠  Unexpected result — check threshold / graph parameters')

    # 8. Figures
    print('\n[6] Generating figures …')
    make_figures(dem, acc, xy, elevs, d)


if __name__ == '__main__':
    main()
