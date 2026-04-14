#!/usr/bin/env python3
"""
terrain_channel_graph.py
========================
Implement and visually verify the PHYSICALLY CORRECT terrain graph:

  Nodes : high-accumulation channel pixels from D8 flow accumulation
  Edges : rook adjacency (4-connectivity) or queen adjacency (8-connectivity)
          of channel pixels that share a grid boundary — NOT k-NN proximity

A rook edge between pixels i and j means they share a physical boundary
through which water can pass.  At confluences (≥3 channels meeting) this
creates closed loops — the only physically real loops a channel network
can have.

Produces 5 figures:
  fig_channel_graph_overview.png  — DEM + flow acc + channel mask (3-panel)
  fig_rook_vs_knn_local.png       — zoomed comparison of edge structure
  fig_junction_loops.png          — close-up of a confluence showing loops
  fig_eigenspectra_compare.png    — eigenspectrum: rook vs queen vs k-NN
  fig_phase_space_terrain.png     — phase space: all three edge methods

Run on the cached AZ Plateau (Coconino) 1-m DEM tiles.
"""

from __future__ import annotations
import math, sys, warnings
from pathlib import Path
from collections import deque

import numpy as np
from scipy.linalg import eigh as scipy_eigh
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm, Normalize

KCAL_ROOT = Path(__file__).parent
sys.path.insert(0, str(KCAL_ROOT))
from kernelcal.terrain.diagnostics import (
    spectral_entropy, fixed_point_kernel, fiedler_mode_gap,
)

try:
    import rasterio, rasterio.crs, rasterio.transform
    from rasterio.merge import merge as rio_merge
    from rasterio.warp import reproject, Resampling, calculate_default_transform
    from rasterio.transform import from_origin
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    raise RuntimeError('rasterio is required')

# ── paths & config ────────────────────────────────────────────────────────────
DATA_DIR  = KCAL_ROOT / 'datasets' / 'badlands'
FIG_DIR   = KCAL_ROOT / 'terrain_figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

DOWNSAMPLE_M   = 10       # working resolution
ACC_THR_FACTOR = 0.003    # fraction of max accumulation to define channels
N_MAX          = 3000     # max channel nodes kept
MU2            = 2.0
SIGMA2         = 1.0

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9.5,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 7.5, 'legend.framealpha': 0.95,
    'figure.dpi': 150, 'savefig.dpi': 250, 'savefig.bbox': 'tight',
})

C_ROOK  = '#009E73'   # green
C_QUEEN = '#56B4E9'   # blue
C_KNN   = '#D55E00'   # orange


# ══════════════════════════════════════════════════════════════════════════════
# DEM LOADING (reuses cached tiles from badlands_kernelcal.py)
# ══════════════════════════════════════════════════════════════════════════════

def load_dem() -> tuple[np.ndarray, object]:
    tile_paths = sorted(DATA_DIR.glob('*.tif'))
    if not tile_paths:
        raise FileNotFoundError(f'No DEM tiles in {DATA_DIR}')
    print(f'  Mosaicking {len(tile_paths)} tiles …')
    srcs = [rasterio.open(p) for p in tile_paths]
    mosaic_arr, mosaic_transform = rio_merge(srcs)
    crs_src = srcs[0].crs
    for s in srcs: s.close()

    elev = mosaic_arr[0].astype(float)
    elev[elev == -9999] = np.nan

    # Reproject to UTM 12N
    crs_dst = rasterio.crs.CRS.from_epsg(32612)
    dst_transform, dst_w, dst_h = calculate_default_transform(
        crs_src, crs_dst, elev.shape[1], elev.shape[0],
        *rasterio.transform.array_bounds(elev.shape[0], elev.shape[1],
                                         mosaic_transform))
    elev_utm = np.full((dst_h, dst_w), np.nan, dtype=float)
    reproject(elev, elev_utm,
              src_transform=mosaic_transform, src_crs=crs_src,
              dst_transform=dst_transform, dst_crs=crs_dst,
              resampling=Resampling.bilinear,
              src_nodata=np.nan, dst_nodata=np.nan)

    # Downsample
    factor = DOWNSAMPLE_M
    rows, cols = elev_utm.shape
    rc = (rows // factor) * factor
    cc = (cols // factor) * factor
    ds = np.nanmean(
        elev_utm[:rc, :cols // factor * factor].reshape(
            rc // factor, factor, cc // factor, factor),
        axis=(1, 3))
    ds_transform = from_origin(dst_transform.c, dst_transform.f,
                               dst_transform.a * factor,
                               -dst_transform.e * factor)
    print(f'  DEM shape: {ds.shape}  pixel: {DOWNSAMPLE_M} m  '
          f'elev: {np.nanmin(ds):.0f}–{np.nanmax(ds):.0f} m')
    return ds, ds_transform


# ══════════════════════════════════════════════════════════════════════════════
# D8 FLOW ACCUMULATION
# ══════════════════════════════════════════════════════════════════════════════

NBR8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
DIST8 = [math.sqrt(2),1,math.sqrt(2),1,1,math.sqrt(2),1,math.sqrt(2)]

def d8_flow_dir(dem: np.ndarray) -> np.ndarray:
    rows, cols = dem.shape
    padded = np.pad(dem, 1, mode='edge')
    drops = np.full((rows, cols, 8), -np.inf)
    for k, (dr, dc) in enumerate(NBR8):
        nbr = padded[1+dr:1+dr+rows, 1+dc:1+dc+cols]
        drops[:, :, k] = (dem - nbr) / DIST8[k]
    return np.argmax(drops, axis=2).astype(np.int8)

def flow_accum(fdir: np.ndarray) -> np.ndarray:
    rows, cols = fdir.shape
    acc = np.ones((rows, cols), dtype=np.int32)
    in_deg = np.zeros_like(acc)
    nbr = np.array(NBR8, dtype=int)
    for r in range(rows):
        for c in range(cols):
            k = fdir[r, c]; nr = r+nbr[k,0]; nc = c+nbr[k,1]
            if 0 <= nr < rows and 0 <= nc < cols and (nr != r or nc != c):
                in_deg[nr, nc] += 1
    q = deque((r, c) for r in range(rows) for c in range(cols)
              if in_deg[r, c] == 0)
    while q:
        r, c = q.popleft()
        k = fdir[r, c]; nr = r+nbr[k,0]; nc = c+nbr[k,1]
        if 0 <= nr < rows and 0 <= nc < cols and (nr != r or nc != c):
            acc[nr, nc] += acc[r, c]
            in_deg[nr, nc] -= 1
            if in_deg[nr, nc] == 0:
                q.append((nr, nc))
    return acc


# ══════════════════════════════════════════════════════════════════════════════
# THREE GRAPH CONSTRUCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def channel_mask(acc: np.ndarray, dem: np.ndarray) -> np.ndarray:
    thr = max(int(ACC_THR_FACTOR * acc.max()), 5)
    return (acc >= thr) & np.isfinite(dem)

def pixel_coords(mask: np.ndarray, transform) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (row_idx, col_idx, xy_metres) for channel pixels."""
    rows_idx, cols_idx = np.where(mask)
    px = abs(transform.a)
    x0 = transform.c + px / 2
    y0 = transform.f - px / 2
    xs = x0 + cols_idx * px
    ys = y0 - rows_idx * px
    return rows_idx, cols_idx, np.column_stack([xs, ys])

def subsample_by_accumulation(rows_idx, cols_idx, xy, acc, n_max):
    acc_vals = acc[rows_idx, cols_idx]
    if len(xy) > n_max:
        top = np.argsort(acc_vals)[::-1][:n_max]
        return rows_idx[top], cols_idx[top], xy[top]
    return rows_idx, cols_idx, xy


# ── ROOK adjacency (4-connectivity) ──────────────────────────────────────────
def build_rook_graph(rows_idx: np.ndarray, cols_idx: np.ndarray,
                     xy: np.ndarray, dem: np.ndarray,
                     px_size: float) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Connect channel pixels that share a grid edge (up/down/left/right).
    Edge weight: exp(-|Δelev| / σ_elev) — steeper = weaker connection.
    """
    n = len(xy)
    # Build fast lookup: (row, col) → node index
    rc_to_idx = {(r, c): i for i, (r, c) in enumerate(zip(rows_idx, cols_idx))}

    elev_vals = dem[rows_idx, cols_idx]
    elev_diffs = [abs(dem[rows_idx[i], cols_idx[i]] - dem[rows_idx[j], cols_idx[j]])
                  for i in range(n) for j in range(n) if i < j]
    sigma_elev = max(float(np.median(np.abs(np.diff(elev_vals)))), 0.1)

    W = np.zeros((n, n), dtype=float)
    edges_list = []  # (i, j) pairs for plotting

    rook_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for i, (r, c) in enumerate(zip(rows_idx, cols_idx)):
        for dr, dc in rook_offsets:
            j = rc_to_idx.get((r + dr, c + dc))
            if j is not None and j > i:
                d_elev = abs(float(dem[r, c]) - float(dem[r+dr, c+dc]))
                w = math.exp(-d_elev / (sigma_elev + 1e-6))
                W[i, j] = max(W[i, j], w)
                W[j, i] = W[i, j]
                edges_list.append((i, j))

    L = np.diag(W.sum(1)) - W
    print(f'  Rook  graph: {n} nodes  {len(edges_list)} edges  '
          f'(mean deg={2*len(edges_list)/n:.2f})')
    return L, W, edges_list


# ── QUEEN adjacency (8-connectivity) ─────────────────────────────────────────
def build_queen_graph(rows_idx: np.ndarray, cols_idx: np.ndarray,
                      xy: np.ndarray, dem: np.ndarray,
                      px_size: float) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Connect channel pixels that share a grid edge OR corner (full 8-neighbourhood).
    Diagonal edges are longer (√2 × pixel size); weight includes length penalty.
    """
    n = len(xy)
    rc_to_idx = {(r, c): i for i, (r, c) in enumerate(zip(rows_idx, cols_idx))}
    elev_vals = dem[rows_idx, cols_idx]
    sigma_elev = max(float(np.median(np.abs(np.diff(elev_vals)))), 0.1)

    W = np.zeros((n, n), dtype=float)
    edges_list = []
    queen_offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    queen_lengths = [px_size*math.sqrt(2), px_size, px_size*math.sqrt(2),
                     px_size, px_size, px_size*math.sqrt(2), px_size, px_size*math.sqrt(2)]

    for i, (r, c) in enumerate(zip(rows_idx, cols_idx)):
        for (dr, dc), length in zip(queen_offsets, queen_lengths):
            j = rc_to_idx.get((r + dr, c + dc))
            if j is not None and j > i:
                d_elev = abs(float(dem[r, c]) - float(dem[r+dr, c+dc]))
                # Combined weight: elevation similarity × distance penalty
                w = math.exp(-d_elev / (sigma_elev + 1e-6)) * math.exp(-length / px_size)
                W[i, j] = max(W[i, j], w)
                W[j, i] = W[i, j]
                edges_list.append((i, j))

    L = np.diag(W.sum(1)) - W
    print(f'  Queen graph: {n} nodes  {len(edges_list)} edges  '
          f'(mean deg={2*len(edges_list)/n:.2f})')
    return L, W, edges_list


# ── k-NN proximity (old method, for comparison) ───────────────────────────────
def build_knn_graph(xy: np.ndarray, k: int = 6) -> tuple[np.ndarray, np.ndarray, list]:
    n = len(xy)
    tree = cKDTree(xy)
    dists, inds = tree.query(xy, k=k+1)
    med_nn = float(np.median(dists[:, 1]))
    xr = xy[:, 0].max() - xy[:, 0].min()
    yr = xy[:, 1].max() - xy[:, 1].min()
    sigma = max(0.05 * math.hypot(xr, yr), 2 * max(med_nn, 1e-3))

    W = np.zeros((n, n), dtype=float)
    edges_list = []
    for i in range(n):
        for ki in range(1, k+1):
            j = inds[i, ki]; d = dists[i, ki]
            w = math.exp(-d**2 / sigma**2)
            if w > W[i, j]:
                W[i, j] = w; W[j, i] = w
                if j > i: edges_list.append((i, j))

    L = np.diag(W.sum(1)) - W
    print(f'  k-NN  graph: {n} nodes  {len(edges_list)} edges  '
          f'(mean deg={2*len(edges_list)/n:.2f})')
    return L, W, edges_list


# ── kernelcal runner ──────────────────────────────────────────────────────────
def run_kernelcal(L: np.ndarray, W: np.ndarray, label: str) -> dict:
    n = L.shape[0]
    ev, _ = scipy_eigh(L)
    ev = np.maximum(ev, 0.0)
    n_zero  = int(np.sum(ev < 1e-6))
    w_modes = ev.copy()
    w_modes[:n_zero] = ev[n_zero] if n_zero < n else 1e-3
    h0 = np.maximum(np.exp(-ev), 1e-10)
    h_star, info = fixed_point_kernel(L, h0=h0, mu2=MU2, sigma2=SIGMA2, w=w_modes)
    h_star = np.maximum(h_star, 1e-8)

    H_obs = spectral_entropy(h_star); H_vac = spectral_entropy(h0)
    dH = H_obs - H_vac
    n_edges = int((W > 0).sum()) // 2
    beta0   = n_zero
    beta1   = max(0, n_edges - (n - beta0))
    lam_f   = float(ev[n_zero]) if n_zero < n else 0.0

    print(f'  [{label:6s}]  ΔH={dH:+.4f}  β₀={beta0}  β₁={beta1}  '
          f'Δβ₁/N={beta1/n:+.4f}  λ_f={lam_f:.5f}')
    return dict(ev=ev, h0=h0, h_star=h_star, dH=dH, H_obs=H_obs, H_vac=H_vac,
                beta0=beta0, beta1=beta1, db1_N=beta1/n, lam_f=lam_f,
                n=n, n_edges=n_edges, label=label)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — DEM overview: hillshade + flow accumulation + channel mask
# ══════════════════════════════════════════════════════════════════════════════

def fig1_overview(dem, acc, mask, transform):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    fig.subplots_adjust(wspace=0.06)

    px = abs(transform.a)
    rows, cols = dem.shape
    x0 = transform.c; y0 = transform.f - rows * px
    extent = [x0, x0 + cols*px, y0, y0 + rows*px]

    # Hillshade
    ax = axes[0]
    dem_f = np.where(np.isfinite(dem), dem, np.nanmin(dem))
    dy, dx_dem = np.gradient(dem_f, px, px)
    slope = np.arctan(np.sqrt(dx_dem**2 + dy**2))
    aspect = np.arctan2(-dy, dx_dem)
    sun_az, sun_el = math.radians(315), math.radians(45)
    hs = (np.cos(sun_el)*np.cos(slope) +
          np.sin(sun_el)*np.sin(slope)*np.cos(sun_az - aspect))
    hs = np.clip(hs, 0, 1)
    ax.imshow(hs, cmap='gray', extent=extent, origin='upper', aspect='equal')
    ax.set_title('(a)  Hillshade (10 m DEM)', fontweight='bold', pad=4)
    ax.set_xlabel('Easting (m UTM)'); ax.set_ylabel('Northing (m UTM)')

    # Flow accumulation (log scale)
    ax2 = axes[1]
    ax2.imshow(hs, cmap='gray', extent=extent, origin='upper',
               aspect='equal', alpha=0.5)
    acc_show = np.where(acc > 1, acc.astype(float), np.nan)
    im = ax2.imshow(acc_show, cmap='Blues', norm=LogNorm(vmin=2, vmax=acc.max()),
                    extent=extent, origin='upper', aspect='equal', alpha=0.8)
    plt.colorbar(im, ax=ax2, fraction=0.04, pad=0.02).set_label('Flow accumulation (cells)')
    ax2.set_title('(b)  D8 flow accumulation', fontweight='bold', pad=4)
    ax2.set_xlabel('Easting (m UTM)')
    ax2.set_yticklabels([])

    # Channel mask
    ax3 = axes[2]
    ax3.imshow(hs, cmap='gray', extent=extent, origin='upper',
               aspect='equal', alpha=0.5)
    chan_rgba = np.zeros((*mask.shape, 4))
    chan_rgba[mask] = [0.0, 0.62, 0.45, 0.85]   # C_ROOK green
    ax3.imshow(chan_rgba, extent=extent, origin='upper', aspect='equal')
    n_chan = int(mask.sum())
    ax3.set_title(f'(c)  Channel mask  ({n_chan:,} pixels,\n'
                  f'threshold = {ACC_THR_FACTOR:.1%} × max acc)',
                  fontweight='bold', pad=4)
    ax3.set_xlabel('Easting (m UTM)')
    ax3.set_yticklabels([])

    fig.suptitle('AZ Plateau (Coconino, USGS 3DEP 1 m → 10 m)  ·  '
                 'D8 Flow Accumulation Pipeline',
                 fontsize=10, fontweight='bold', y=1.02)
    out = FIG_DIR / 'fig_channel_graph_overview.png'
    fig.savefig(out); plt.close(fig)
    print(f'  Saved {out.name}')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Rook vs k-NN edge structure comparison (full extent + zoom)
# ══════════════════════════════════════════════════════════════════════════════

def fig2_rook_vs_knn(dem, mask, transform,
                     xy, rows_idx, cols_idx,
                     rook_edges, knn_edges,
                     rook_W, knn_W):
    px = abs(transform.a)
    rows, cols_dem = dem.shape
    x0 = transform.c; y0 = transform.f - rows * px
    extent = [x0, x0 + cols_dem*px, y0, y0 + rows*px]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.subplots_adjust(hspace=0.35, wspace=0.10)

    dem_f = np.where(np.isfinite(dem), dem, np.nanmin(dem))
    dy, dx_dem = np.gradient(dem_f, px, px)
    slope = np.arctan(np.sqrt(dx_dem**2 + dy**2))
    aspect = np.arctan2(-dy, dx_dem)
    sun_az, sun_el = math.radians(315), math.radians(45)
    hs = np.clip(np.cos(math.radians(45))*np.cos(slope) +
                 np.sin(math.radians(45))*np.sin(slope)*
                 np.cos(math.radians(315) - aspect), 0, 1)

    def draw_graph(ax, edges, W_mat, color, title, zoom=False):
        ax.imshow(hs, cmap='gray', extent=extent, origin='upper', aspect='equal', alpha=0.6)
        # Channel pixels
        ax.scatter(xy[:, 0], xy[:, 1], s=2, color=color, alpha=0.5,
                   linewidths=0, zorder=3)
        # Edges
        segs = [[xy[i], xy[j]] for i, j in edges]
        weights = [W_mat[i, j] for i, j in edges]
        if segs:
            lc = LineCollection(segs, linewidths=0.4, color=color,
                                alpha=0.55, zorder=4)
            ax.add_collection(lc)
        ax.set_title(title, fontweight='bold', pad=4, color=color)
        ax.set_xlabel('Easting (m)')
        if zoom:
            # Centre of mass of channel pixels
            cx, cy = xy[:, 0].mean(), xy[:, 1].mean()
            half = 1500
            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cy - half, cy + half)
        else:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
        return ax

    draw_graph(axes[0, 0], rook_edges, rook_W, C_ROOK,
               f'(a)  Rook adjacency — full extent\n'
               f'{len(rook_edges)} edges (shared grid boundaries)')
    axes[0, 0].set_ylabel('Northing (m)')

    draw_graph(axes[0, 1], knn_edges, knn_W, C_KNN,
               f'(b)  k-NN proximity (k=6) — full extent\n'
               f'{len(knn_edges)} edges (analyst-imposed)')

    draw_graph(axes[1, 0], rook_edges, rook_W, C_ROOK,
               '(c)  Rook adjacency — 3 km zoom\n'
               '(edges only exist where pixels share a boundary)',
               zoom=True)
    axes[1, 0].set_ylabel('Northing (m)')

    draw_graph(axes[1, 1], knn_edges, knn_W, C_KNN,
               '(d)  k-NN proximity — same 3 km zoom\n'
               '(proximity edges cross channel boundaries freely)',
               zoom=True)

    fig.suptitle('Rook Adjacency vs. k-NN Proximity — AZ Plateau Channel Network\n'
                 'Rook edges are physically real (shared pixel boundary = shared water boundary);\n'
                 'k-NN edges are analyst-imposed proximity with no physical referent.',
                 fontsize=10, fontweight='bold', y=1.02)
    out = FIG_DIR / 'fig_rook_vs_knn_local.png'
    fig.savefig(out); plt.close(fig)
    print(f'  Saved {out.name}')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Close-up of confluence loops in rook graph
# ══════════════════════════════════════════════════════════════════════════════

def fig3_junction_loops(dem, mask, transform, xy, rows_idx, cols_idx,
                        rook_edges, rook_W):
    px = abs(transform.a)
    rows, cols_dem = dem.shape
    x0 = transform.c; y0 = transform.f - rows * px
    extent = [x0, x0 + cols_dem*px, y0, y0 + rows*px]

    dem_f = np.where(np.isfinite(dem), dem, np.nanmin(dem))
    dy, dx_dem = np.gradient(dem_f, px, px)
    slope = np.arctan(np.sqrt(dx_dem**2 + dy**2))
    aspect = np.arctan2(-dy, dx_dem)
    hs = np.clip(np.cos(math.radians(45))*np.cos(slope) +
                 np.sin(math.radians(45))*np.sin(slope)*
                 np.cos(math.radians(315) - aspect), 0, 1)

    # Find nodes with degree >= 3 (junction candidates)
    deg = (rook_W > 0).sum(1)
    junc_mask = deg >= 3
    junc_xy = xy[junc_mask]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.subplots_adjust(wspace=0.08)

    # Find 3 distinct confluences spread across the domain
    n_panels = 3
    if len(junc_xy) >= n_panels:
        # Cluster to find spatially separated junctions
        from scipy.spatial.distance import cdist
        selected = [0]
        for _ in range(n_panels - 1):
            dists = cdist(junc_xy[selected], junc_xy).min(0)
            selected.append(int(np.argmax(dists)))
        centres = [junc_xy[s] for s in selected]
    else:
        centres = [xy.mean(0)] * n_panels

    half = 600  # metres

    for ai, (ax, centre) in enumerate(zip(axes, centres)):
        ax.imshow(hs, cmap='gray', extent=extent, origin='upper',
                  aspect='equal', alpha=0.7)

        # Restrict to nodes in the zoom box
        in_box = ((xy[:, 0] >= centre[0]-half) & (xy[:, 0] <= centre[0]+half) &
                  (xy[:, 1] >= centre[1]-half) & (xy[:, 1] <= centre[1]+half))
        box_idx = np.where(in_box)[0]

        # Edges within box
        box_set = set(box_idx)
        segs = [[xy[i], xy[j]] for i, j in rook_edges
                if i in box_set and j in box_set]
        weights = [rook_W[i, j] for i, j in rook_edges
                   if i in box_set and j in box_set]

        if segs:
            lc = LineCollection(segs, linewidths=1.2, color=C_ROOK, alpha=0.8, zorder=4)
            ax.add_collection(lc)

        # Colour nodes by degree
        sc = ax.scatter(xy[in_box, 0], xy[in_box, 1],
                        c=deg[in_box], cmap='YlOrRd',
                        s=18, vmin=1, vmax=4, edgecolors='black',
                        linewidths=0.3, zorder=5)

        # Mark high-degree (junction) nodes
        junc_in = in_box & junc_mask
        if junc_in.any():
            ax.scatter(xy[junc_in, 0], xy[junc_in, 1],
                       s=60, color='yellow', marker='*',
                       edgecolors='black', linewidths=0.5, zorder=6)

        ax.set_xlim(centre[0]-half, centre[0]+half)
        ax.set_ylim(centre[1]-half, centre[1]+half)
        ax.set_title(f'({chr(97+ai)})  600 m × 600 m\n'
                     f'★ = junction (deg ≥ 3)  ·  loops at confluences',
                     fontweight='bold', pad=4, fontsize=9)
        ax.set_xlabel('Easting (m)')
        if ai == 0:
            ax.set_ylabel('Northing (m)')
        else:
            ax.set_yticklabels([])

    plt.colorbar(sc, ax=axes[-1], fraction=0.04, pad=0.02).set_label('Node degree')

    n_junc = int(junc_mask.sum())
    fig.suptitle(f'Confluence Loops in Rook-Adjacency Channel Graph\n'
                 f'{n_junc} junction nodes (degree ≥ 3) create physically real closed loops',
                 fontsize=10, fontweight='bold', y=1.02)
    out = FIG_DIR / 'fig_junction_loops.png'
    fig.savefig(out); plt.close(fig)
    print(f'  Saved {out.name}')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Eigenspectrum + h*(λ) comparison: rook vs queen vs k-NN
# ══════════════════════════════════════════════════════════════════════════════

def fig4_spectra(gd_rook, gd_queen, gd_knn):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    fig.subplots_adjust(hspace=0.45, wspace=0.35)

    for col, (gd, col_c, tag) in enumerate([
            (gd_rook,  C_ROOK,  'Rook (4-conn.)'),
            (gd_queen, C_QUEEN, 'Queen (8-conn.)'),
            (gd_knn,   C_KNN,   'k-NN (k=6, old)')]):

        ev   = np.sort(gd['ev'])
        n_sh = min(80, len(ev))
        ax_top = axes[0, col]
        ax_top.bar(range(n_sh), ev[:n_sh], color=col_c,
                   width=1.0, edgecolor='none', alpha=0.80)
        ax_top.set_title(tag, fontweight='bold', color=col_c, pad=4)
        ax_top.set_xlabel('Mode $l$')
        if col == 0: ax_top.set_ylabel('$\\lambda_l$')
        ax_top.text(0.97, 0.97,
                    f'$\\beta_0={gd["beta0"]}$\n$\\beta_1={gd["beta1"]}$\n'
                    f'$\\Delta\\beta_1/N={gd["db1_N"]:+.3f}$',
                    transform=ax_top.transAxes, ha='right', va='top', fontsize=7.5,
                    bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#cccccc'))

        sort = np.argsort(gd['ev'])
        lam  = gd['ev'][sort][:n_sh]
        h0s  = gd['h0'][sort][:n_sh]
        hs   = gd['h_star'][sort][:n_sh]
        ax_bot = axes[1, col]
        ax_bot.fill_between(lam, h0s, hs, where=(hs >= h0s),
                            color='#009E73', alpha=0.25, label='amplified')
        ax_bot.fill_between(lam, h0s, hs, where=(hs < h0s),
                            color='#D55E00', alpha=0.25, label='suppressed')
        ax_bot.plot(lam, h0s, '--', color='#888888', lw=1.2, label='$h_0$')
        ax_bot.plot(lam, hs,  '-',  color=col_c,    lw=1.8, label='$h^*$')
        ax_bot.set_xlabel('$\\lambda_l$')
        if col == 0: ax_bot.set_ylabel('$h(\\lambda)$')
        ax_bot.text(0.05, 0.06,
                    f'$\\Delta H = {gd["dH"]:+.3f}$ nats',
                    transform=ax_bot.transAxes, va='bottom',
                    fontsize=9, color=col_c, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#cccccc'))
        if col == 2: ax_bot.legend(fontsize=7, loc='upper right')

    fig.suptitle(
        'AZ Plateau — Eigenspectra and Fixed-Point Kernels\n'
        'Three graph constructions: rook adjacency, queen adjacency, k-NN proximity',
        fontsize=10, fontweight='bold')
    out = FIG_DIR / 'fig_eigenspectra_compare.png'
    fig.savefig(out); plt.close(fig)
    print(f'  Saved {out.name}')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Phase space: terrain methods vs street cities
# ══════════════════════════════════════════════════════════════════════════════

def fig5_phase_space(gd_rook, gd_queen, gd_knn):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Street city bootstrap medians (from osm_street_kernelcal.py run)
    CITIES = [
        dict(name='Barcelona', dH=-0.339, db1_N=0.608, color='#0072B2'),
        dict(name='Phoenix',   dH=-0.285, db1_N=0.482, color='#E69F00'),
        dict(name='Venice',    dH=-0.237, db1_N=0.190, color='#009E73'),
        dict(name='Marrakech', dH=-0.238, db1_N=0.247, color='#CC79A7'),
        dict(name='Houston',   dH=-0.301, db1_N=0.570, color='#D55E00'),
    ]

    # Background bands
    ax.axhspan(-0.05, 0.30, color='#EFF7FF', alpha=0.5, zorder=0)
    ax.axhspan(0.30, 0.70,  color='#FFF8E8', alpha=0.5, zorder=0)
    ax.axhspan(0.70, 2.50,  color='#EFFFEF', alpha=0.5, zorder=0)
    ax.text(-0.55, 0.02,  'Abiotic tier', fontsize=8, color='#0072B2', style='italic')
    ax.text(-0.55, 0.32,  'Intermediate', fontsize=8, color='#888888', style='italic')
    ax.text(-0.55, 0.72,  'Active ctrl.', fontsize=8, color='#009E73', style='italic')

    # k-NN terrain (old, shown as reference)
    ax.scatter(gd_knn['dH'], gd_knn['db1_N'], s=80,
               color=C_KNN, marker='o', edgecolors='black',
               linewidths=0.8, zorder=4, alpha=0.7, linestyle='--')
    ax.annotate('AZ Plateau\n(k-NN — old)',
                (gd_knn['dH'], gd_knn['db1_N']),
                xytext=(10, -20), textcoords='offset points',
                fontsize=7.5, color=C_KNN,
                arrowprops=dict(arrowstyle='->', color=C_KNN, lw=0.8))

    # Rook terrain
    ax.scatter(gd_rook['dH'], gd_rook['db1_N'], s=120,
               color=C_ROOK, marker='o', edgecolors='black',
               linewidths=1.0, zorder=6)
    ax.annotate('AZ Plateau\n(rook — physical)',
                (gd_rook['dH'], gd_rook['db1_N']),
                xytext=(-12, 15), textcoords='offset points',
                fontsize=8, color=C_ROOK, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C_ROOK, lw=1.0))

    # Queen terrain
    ax.scatter(gd_queen['dH'], gd_queen['db1_N'], s=80,
               color=C_QUEEN, marker='o', edgecolors='black',
               linewidths=0.8, zorder=5)
    ax.annotate('AZ Plateau\n(queen — physical)',
                (gd_queen['dH'], gd_queen['db1_N']),
                xytext=(12, 5), textcoords='offset points',
                fontsize=8, color=C_QUEEN,
                arrowprops=dict(arrowstyle='->', color=C_QUEEN, lw=0.8))

    # Street cities
    for c in CITIES:
        ax.scatter(c['dH'], c['db1_N'], s=65, color=c['color'],
                   marker='D', edgecolors='black', linewidths=0.7, zorder=5)
        ax.annotate(c['name'], (c['dH'], c['db1_N']),
                    xytext=(4, 3), textcoords='offset points',
                    fontsize=7.5, color=c['color'])

    ax.axhline(0, color='#cccccc', lw=0.6, ls=':')
    ax.axvline(0, color='#cccccc', lw=0.6, ls=':')
    ax.set_xlim(-0.60, 0.05)
    ax.set_ylim(-0.05, 1.5)
    ax.set_xlabel(r'$\Delta H = H[h^*] - H[h_0]$  (nats)', fontsize=10)
    ax.set_ylabel(r'$\Delta\beta_1 / N$  (normalised topological excess)', fontsize=10)
    ax.set_title('Phase Space: Physically-Motivated Graphs\n'
                 'Rook/queen terrain edges vs. OSM road network (cities)',
                 fontweight='bold', pad=6, fontsize=11)

    from matplotlib.lines import Line2D
    leg = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor=C_ROOK,
               markeredgecolor='black', markersize=10, label='Rook adjacency (terrain)'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=C_QUEEN,
               markeredgecolor='black', markersize=8, label='Queen adjacency (terrain)'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=C_KNN,
               markeredgecolor='black', markersize=8, label='k-NN proximity (old — invalid)'),
        Line2D([0],[0], marker='D', color='w', markerfacecolor='#888888',
               markeredgecolor='black', markersize=8, label='Cities (OSM road network)'),
    ]
    ax.legend(handles=leg, fontsize=8, loc='upper right', framealpha=0.97)

    out = FIG_DIR / 'fig_phase_space_terrain.png'
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f'  Saved {out.name}')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print('=' * 65)
    print('Physically-correct terrain channel graph — AZ Plateau')
    print(f'  Output: {FIG_DIR}')
    print('=' * 65)

    print('\n[1] Loading DEM …')
    dem, transform = load_dem()
    px_size = abs(transform.a)

    print('\n[2] D8 flow accumulation …')
    fdir = d8_flow_dir(dem)
    acc  = flow_accum(fdir)
    mask = channel_mask(acc, dem)
    print(f'  Channel pixels: {mask.sum():,}  '
          f'(threshold: acc ≥ {int(ACC_THR_FACTOR * acc.max())})')

    print('\n[3] Extracting channel nodes …')
    rows_idx, cols_idx, xy_all = pixel_coords(mask, transform)
    rows_idx, cols_idx, xy = subsample_by_accumulation(
        rows_idx, cols_idx, xy_all, acc, N_MAX)
    print(f'  Using {len(xy):,} channel nodes')

    print('\n[4] Building three graphs …')
    L_rook,  W_rook,  edges_rook  = build_rook_graph(
        rows_idx, cols_idx, xy, dem, px_size)
    L_queen, W_queen, edges_queen = build_queen_graph(
        rows_idx, cols_idx, xy, dem, px_size)
    L_knn,   W_knn,   edges_knn   = build_knn_graph(xy, k=6)

    print('\n[5] Running kernelcal …')
    gd_rook  = run_kernelcal(L_rook,  W_rook,  'rook')
    gd_queen = run_kernelcal(L_queen, W_queen, 'queen')
    gd_knn   = run_kernelcal(L_knn,   W_knn,   'k-NN')

    print('\n  ── Summary ──────────────────────────────────────────────')
    print(f'  {"Method":8s}  {"ΔH":>8}  {"β₀":>5}  {"β₁":>7}  {"Δβ₁/N":>8}  {"Edges":>7}')
    print('  ' + '─' * 55)
    for gd in [gd_rook, gd_queen, gd_knn]:
        print(f'  {gd["label"]:8s}  {gd["dH"]:+8.4f}  {gd["beta0"]:5d}  '
              f'{gd["beta1"]:7d}  {gd["db1_N"]:+8.4f}  {gd["n_edges"]:7d}')

    print('\n[6] Generating figures …')
    print('  [6a] DEM overview …')
    fig1_overview(dem, acc, mask, transform)
    print('  [6b] Rook vs k-NN comparison …')
    fig2_rook_vs_knn(dem, mask, transform, xy, rows_idx, cols_idx,
                     edges_rook, edges_knn, W_rook, W_knn)
    print('  [6c] Junction loops …')
    fig3_junction_loops(dem, mask, transform, xy, rows_idx, cols_idx,
                        edges_rook, W_rook)
    print('  [6d] Eigenspectra comparison …')
    fig4_spectra(gd_rook, gd_queen, gd_knn)
    print('  [6e] Phase space …')
    fig5_phase_space(gd_rook, gd_queen, gd_knn)

    print(f'\nAll figures → {FIG_DIR}')


if __name__ == '__main__':
    main()
