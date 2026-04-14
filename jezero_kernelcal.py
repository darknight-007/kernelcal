#!/usr/bin/env python3
"""
jezero_kernelcal.py — Spectral Kernel Dynamics on the Jezero Crater Delta DEM

Builds a flow-accumulation drainage graph from the Jezero 1 m DEM,
computes the graph Laplacian and its eigenpairs, then applies the
kernelcal Maximum Caliber (MaxCal) fixed-point framework to characterise
the spectral controller structure of the delta.

Figures produced (saved to ./jezero_figures/):
  Fig 1 — Topographic overview  (hillshade + elevation + Jezero context)
  Fig 2 — Flow accumulation + drainage network extraction
  Fig 3 — Channel network graph on DEM  (nodes = junctions/sources/sinks)
  Fig 4 — Graph Laplacian eigenspectrum  (Fiedler gap highlighted)
  Fig 5 — Fixed-point kernel h*(λ) vs vacuum h₀(λ)  (spectral reweighting)
  Fig 6 — Controller biosignature summary  (H[h*], Δ', k_min, Δβ₁)

Physical context
----------------
Jezero Crater (18.4°N, 77.4°E) is the Perseverance landing site.
The western delta (Neretva Vallis input) is a Noachian–Hesperian
fluvio-deltaic deposit — the primary astrobiology target of Mars 2020.
If the drainage network topology exceeds the abiotic null model
(Δβ₁ > 0), it is consistent with an optimal hydraulic controller
building structure above the spontaneous D8-equilibrium.
"""

from __future__ import annotations

import math, sys
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.linalg import eigh as scipy_eigh

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.colors import LightSource, Normalize, LogNorm
from matplotlib.cm import ScalarMappable
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ── kernelcal ──────────────────────────────────────────────────────────────
KCAL_ROOT = Path(__file__).parent
sys.path.insert(0, str(KCAL_ROOT))
from kernelcal.terrain.diagnostics import (
    spectral_entropy,
    fixed_point_kernel,
    fiedler_mode_gap,
    stability_conservation_tradeoff,
)

# ── PATHS & CONFIG ──────────────────────────────────────────────────────────
DEM_PATH  = Path.home() / 'Downloads' / 'swjez-DEM_1m.tif'
FIG_DIR   = Path(__file__).parent / 'jezero_figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Downsampling: 1 m DEM → CELL_SIZE m working grid
CELL_SIZE = 20          # metres per working-grid cell
N_MAX     = 2000        # max channel-network nodes for dense eigenpairs
K_NN      = 6           # k-NN graph connectivity
ACC_THR_FACTOR = 0.0004  # fraction of grid cells → channel threshold

# kernelcal source parameters (Gaussian MI, P4 Eq. 14)
MU2    = 2.0
SIGMA2 = 1.0

# Colormap choices with planetary science convention
CMAP_TOPO  = 'gist_earth'    # Mars-like terrain
CMAP_ACC   = 'Blues'
CMAP_EIGEN = 'magma'


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DEM LOADING & DOWNSAMPLING
# ══════════════════════════════════════════════════════════════════════════════

def load_and_downsample(path: Path, factor: int) -> tuple[np.ndarray, float, dict]:
    """Load raster, replace nodata with NaN, block-average by `factor`."""
    print(f'  Loading {path.name} …')
    with rasterio.open(path) as src:
        raw   = src.read(1).astype(np.float32)
        nodata = src.nodata
        meta   = {
            'crs':       src.crs.to_string(),
            'transform': src.transform,
            'bounds':    src.bounds,
            'orig_res':  src.res[0],
        }

    if nodata is not None:
        raw[raw == nodata] = np.nan
    if np.isnan(raw).mean() < 0.01:
        nodata = None   # essentially no nodata

    print(f'  Original shape: {raw.shape}  ({raw.shape[1]*src.res[0]/1e3:.1f} × '
          f'{raw.shape[0]*src.res[1]/1e3:.1f} km)')

    # Block-average downsample using strided reshape
    nr, nc = raw.shape
    nr_new = nr // factor
    nc_new = nc // factor
    crop = raw[:nr_new * factor, :nc_new * factor]
    dem  = np.nanmean(
        crop.reshape(nr_new, factor, nc_new, factor), axis=(1, 3)
    )
    cell_m = meta['orig_res'] * factor
    print(f'  Downsampled → {dem.shape}  cell = {cell_m:.0f} m')
    return dem, cell_m, meta


# ══════════════════════════════════════════════════════════════════════════════
# 2.  FLOW ROUTING  (vectorized D8 + topological accumulation)
# ══════════════════════════════════════════════════════════════════════════════

# 8-neighbor offsets and their metric distances (normalised to 1-cell = dx)
_OFF = np.array([(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)], dtype=int)
_DIST_MULT = np.array([math.sqrt(2),1,math.sqrt(2),1,1,math.sqrt(2),1,math.sqrt(2)])


def vectorized_d8(dem: np.ndarray, dx: float) -> np.ndarray:
    """Vectorized D8 single-flow direction.  Returns int8 array (-1 = sink)."""
    nrows, ncols = dem.shape
    dists = _DIST_MULT * dx
    fdir  = np.full((nrows, ncols), -1, dtype=np.int8)
    best  = np.full((nrows, ncols), -1e9)

    for d, (dr, dc) in enumerate(_OFF):
        # Shifted neighbour elevation
        nb = np.roll(np.roll(dem, -dr, axis=0), -dc, axis=1)
        # Zero out wrapped edges
        if dr > 0:  nb[-dr:,  :] = np.nan
        if dr < 0:  nb[:-dr,  :] = np.nan
        if dc > 0:  nb[:,  -dc:] = np.nan
        if dc < 0:  nb[:, :-dc:] = np.nan

        slope = (dem - nb) / dists[d]
        better = (~np.isnan(dem)) & (~np.isnan(nb)) & (slope > best)
        fdir[better] = d
        best[better] = slope[better]

    return fdir


def topological_flow_acc(fdir: np.ndarray) -> np.ndarray:
    """Accumulate upstream cell counts via elevation-ordered traversal."""
    nrows, ncols = fdir.shape
    acc = np.ones((nrows, ncols), dtype=np.int32)

    # Build downstream pointer arrays (flat indices)
    r_idx, c_idx = np.where(fdir >= 0)
    dr = _OFF[fdir[r_idx, c_idx], 0]
    dc = _OFF[fdir[r_idx, c_idx], 1]
    dst_r = np.clip(r_idx + dr, 0, nrows - 1)
    dst_c = np.clip(c_idx + dc, 0, ncols - 1)

    # Topological order: process cells high→low elevation so upstream always done first
    valid_mask = fdir >= 0
    all_r, all_c = np.where(~np.isnan(fdir.astype(float)))   # all valid
    # Sort valid cells by elevation descending (source first)
    elev_flat = np.full(nrows * ncols, np.nan)
    # Use fdir shape as proxy — we need the DEM here, but we passed only fdir
    # We'll return partial acc; caller re-runs with dem for proper ordering
    return acc, r_idx, c_idx, dst_r, dst_c


def fast_flow_acc(dem: np.ndarray, fdir: np.ndarray) -> np.ndarray:
    """Fast flow accumulation: sort cells high→low, propagate downstream."""
    nrows, ncols = dem.shape
    acc   = np.ones((nrows, ncols), dtype=np.int32)
    valid = ~np.isnan(dem)

    flat_idx = np.where(valid.ravel())[0]
    elev_order = flat_idx[np.argsort(dem.ravel()[flat_idx])[::-1]]   # high → low

    fd_flat = fdir.ravel()
    acc_flat = acc.ravel()

    for fi in elev_order:
        d = int(fd_flat[fi])
        if d < 0:
            continue
        r, c = divmod(fi, ncols)
        dr, dc = int(_OFF[d, 0]), int(_OFF[d, 1])
        nr, nc = r + dr, c + dc
        if 0 <= nr < nrows and 0 <= nc < ncols:
            acc_flat[nr * ncols + nc] += acc_flat[fi]

    return acc_flat.reshape(nrows, ncols)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  CHANNEL NETWORK → GRAPH → LAPLACIAN
# ══════════════════════════════════════════════════════════════════════════════

def extract_network_nodes(
    dem: np.ndarray, acc: np.ndarray, dx: float,
    n_max: int = N_MAX, threshold: int | None = None,
) -> np.ndarray:
    """Return (N, 3) array of (x_m, y_m, elev_m) for channel network nodes."""
    nrows, ncols = dem.shape
    if threshold is None:
        # Never exceed 10% of max accumulation so we get real channel cells
        max_acc = int(acc.max())
        threshold = min(
            max(10, int(nrows * ncols * ACC_THR_FACTOR)),
            max(10, max_acc // 10)
        )

    chan = (acc >= threshold) & (~np.isnan(dem))
    rows, cols = np.where(chan)
    elevs = dem[rows, cols]

    pts = np.column_stack([cols * dx, (nrows - 1 - rows) * dx, elevs])

    if len(pts) > n_max:
        # Spatially stratified subsample: keep highest-accumulation nodes
        row_pts = np.floor(pts[:, 1] / dx).astype(int)  # recover row from y
        col_pts = np.floor(pts[:, 0] / dx).astype(int)
        row_pts = np.clip(nrows - 1 - row_pts, 0, nrows - 1)
        col_pts = np.clip(col_pts, 0, ncols - 1)
        acc_vals = acc[row_pts, col_pts]
        top_idx = np.argsort(acc_vals)[::-1][:n_max]
        pts = pts[top_idx]

    print(f'  Channel threshold: {threshold} cells = '
          f'{threshold * dx**2 / 1e6:.2f} km²  |  nodes: {len(pts)}')
    return pts


def build_knn_laplacian(
    pts: np.ndarray, k: int = K_NN, sigma_frac: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Build symmetric k-NN graph and return dense Laplacian + adjacency."""
    n  = len(pts)
    xy = pts[:, :2]   # x, y in metres
    dz = pts[:, 2]    # elevation

    tree = cKDTree(xy)
    dists, inds = tree.query(xy, k=k + 1)   # includes self at index 0

    # Gaussian edge weights: w = exp(-d²/σ²)
    sigma = sigma_frac * (xy.max() - xy.min()).mean()
    W     = np.zeros((n, n))
    for i in range(n):
        for j_idx, (dist, j) in enumerate(zip(dists[i, 1:], inds[i, 1:])):
            w = math.exp(-dist**2 / (sigma**2 + 1e-6))
            W[i, j] = max(W[i, j], w)
            W[j, i] = max(W[j, i], w)

    D = np.diag(W.sum(axis=1))
    L = D - W
    return L, W


# ══════════════════════════════════════════════════════════════════════════════
# 4.  BETTI NUMBER ESTIMATION FROM LAPLACIAN
# ══════════════════════════════════════════════════════════════════════════════

def betti_from_laplacian(L: np.ndarray, W: np.ndarray) -> tuple[int, int]:
    """Estimate β₀ (components) and β₁ (independent cycles) from graph topology."""
    eigvals = np.linalg.eigvalsh(L)
    beta0   = int(np.sum(eigvals < 1e-6))           # nullity = #components
    n       = L.shape[0]
    edges   = int(np.sum(W > 0)) // 2              # undirected edges
    beta1   = edges - (n - beta0)                  # Euler: β₁ = E - (N - β₀)
    return beta0, max(0, beta1)


def abiotic_beta1_null(n_nodes: int, k: int = K_NN) -> int:
    """Abiotic null model for a k-NN graph built on D8 drainage nodes.

    A perfect D8 drainage *tree* has β₁_tree = 0.
    When we build a k-NN graph on N nodes, the expected number of extra
    cycles vs a spanning tree is  E - (N - β₀)  where E ≈ k*N/2.
    This is the structural baseline from graph construction alone.
    Δβ₁ = β₁_obs - β₁_null  then isolates *excess* topology beyond
    what a random k-NN graph of the same size would produce.
    """
    # Expected β₁ for a k-NN graph on n_nodes where each node has degree k:
    # E = k*n/2, spanning tree uses n-1 edges → β₁_null = k*n/2 - (n-1)
    e_null = k * n_nodes // 2
    return max(0, e_null - (n_nodes - 1))


# ══════════════════════════════════════════════════════════════════════════════
# 5.  HILLSHADE HELPER
# ══════════════════════════════════════════════════════════════════════════════

def make_hillshade(dem: np.ndarray, dx: float) -> np.ndarray:
    ls = LightSource(azdeg=315, altdeg=45)
    hs = ls.hillshade(dem, vert_exag=3, dx=dx, dy=dx)
    return hs


# ══════════════════════════════════════════════════════════════════════════════
# 6.  FIGURE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

# Shared style
TITLE_KW  = dict(fontsize=13, fontweight='bold', pad=8)
LABEL_KW  = dict(fontsize=10)
TICK_KW   = dict(labelsize=9)
ANNOT_KW  = dict(fontsize=8.5, color='white',
                 path_effects=[pe.withStroke(linewidth=2, foreground='black')])

def km_ticks(ax, values_m, axis='x'):
    km = [v / 1e3 for v in values_m]
    if axis == 'x':
        ax.set_xticks(values_m); ax.set_xticklabels([f'{v:.1f}' for v in km])
        ax.set_xlabel('Easting (km)', **LABEL_KW)
    else:
        ax.set_yticks(values_m); ax.set_yticklabels([f'{v:.1f}' for v in km])
        ax.set_ylabel('Northing (km)', **LABEL_KW)


# ── Fig 1: Topographic Overview ────────────────────────────────────────────
def fig_topo(dem, acc, dx, meta, chan_thr):
    """Publication-quality topographic map with hillshade + delta annotation."""
    nrows, ncols = dem.shape
    ext_km = [0, ncols * dx / 1e3, 0, nrows * dx / 1e3]   # [left,right,bottom,top]

    hs  = make_hillshade(dem, dx)
    elev_min, elev_max = np.nanpercentile(dem, [2, 98])

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    fig.patch.set_facecolor('#0d1117')
    for ax in axes:
        ax.set_facecolor('#0d1117')

    # Left: hillshaded elevation
    ax = axes[0]
    im_hs = ax.imshow(hs, origin='upper', cmap='gray',
                      extent=ext_km, alpha=0.5, vmin=0, vmax=1)
    im_el = ax.imshow(dem, origin='upper', cmap=CMAP_TOPO,
                      extent=ext_km, alpha=0.7,
                      vmin=elev_min, vmax=elev_max)
    # Channel network overlay
    chan_mask = (acc >= chan_thr) & (~np.isnan(dem))
    chan_rgba  = np.zeros((*chan_mask.shape, 4))
    chan_rgba[chan_mask] = [0.2, 0.7, 1.0, 0.85]
    ax.imshow(chan_rgba, origin='upper', extent=ext_km)

    ax.set_title('Jezero Crater — Topography & Drainage Network', color='white', **TITLE_KW)
    ax.set_xlabel('Easting (km)', color='white', **LABEL_KW)
    ax.set_ylabel('Northing (km)', color='white', **LABEL_KW)
    ax.tick_params(colors='white', **TICK_KW)
    for spine in ax.spines.values(): spine.set_edgecolor('white')

    cbar = fig.colorbar(im_el, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label('Elevation (m, MOLA)', color='white', fontsize=9)
    cbar.ax.yaxis.set_tick_params(color='white', labelsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    # Annotate delta lobe (northwest quadrant of the DEM = Neretva delta)
    # Delta centroid is roughly at x~3 km, y~20 km in the 12.8×25.2 km extent
    delta_x, delta_y = 3.2, 20.0
    ax.annotate('Western Delta\n(Neretva Vallis)', xy=(delta_x, delta_y),
                xytext=(delta_x + 2.5, delta_y - 2),
                arrowprops=dict(arrowstyle='->', color='yellow', lw=1.5),
                color='yellow', fontsize=8.5, fontweight='bold')
    # Annotate crater rim direction
    ax.text(0.98, 0.02, '↓ Crater Floor', transform=ax.transAxes,
            ha='right', va='bottom', color='lightyellow', fontsize=8,
            style='italic')
    ax.text(0.02, 0.98, '↑ Crater Rim', transform=ax.transAxes,
            ha='left', va='top', color='lightyellow', fontsize=8,
            style='italic')
    # Scale bar
    sb_len = 2.0   # km
    sb_x0  = ext_km[1] * 0.05
    sb_y0  = ext_km[3] * 0.04
    ax.plot([sb_x0, sb_x0 + sb_len], [sb_y0, sb_y0], 'w-', lw=3)
    ax.text(sb_x0 + sb_len / 2, sb_y0 + 0.3, f'{sb_len:.0f} km',
            ha='center', va='bottom', color='white', fontsize=8)

    # Right: flow accumulation log-scale
    ax2 = axes[1]
    acc_plot = np.where(np.isnan(dem), np.nan, acc.astype(float))
    valid_acc = acc_plot[~np.isnan(acc_plot)]
    vmin_acc = max(1, np.percentile(valid_acc, 60))
    vmax_acc = valid_acc.max()
    ax2.imshow(hs, origin='upper', cmap='gray',
               extent=ext_km, alpha=0.6, vmin=0, vmax=1)
    im_acc = ax2.imshow(
        acc_plot, origin='upper', cmap=CMAP_ACC, extent=ext_km,
        alpha=0.75,
        norm=LogNorm(vmin=max(1.0, vmin_acc), vmax=max(vmax_acc, vmin_acc + 1))
    )
    ax2.set_title('Flow Accumulation  (log scale)', color='white', **TITLE_KW)
    ax2.set_xlabel('Easting (km)', color='white', **LABEL_KW)
    ax2.tick_params(colors='white', **TICK_KW)
    for spine in ax2.spines.values(): spine.set_edgecolor('white')

    cbar2 = fig.colorbar(im_acc, ax=ax2, fraction=0.035, pad=0.02)
    cbar2.set_label('Upstream cells', color='white', fontsize=9)
    cbar2.ax.yaxis.set_tick_params(color='white', labelsize=8)
    plt.setp(cbar2.ax.yaxis.get_ticklabels(), color='white')

    # Annotate basin area
    total_basin_km2 = np.sum(~np.isnan(dem)) * dx**2 / 1e6
    ax2.text(0.02, 0.02, f'Basin area: {total_basin_km2:.1f} km²',
             transform=ax2.transAxes,
             ha='left', va='bottom', color='white', fontsize=8,
             bbox=dict(boxstyle='round,pad=0.3', fc='black', alpha=0.5))
    ax2.text(0.98, 0.02,
             f'Grid: {nrows}×{ncols} @ {dx:.0f} m/cell',
             transform=ax2.transAxes,
             ha='right', va='bottom', color='lightgray', fontsize=7.5)

    fig.suptitle(
        'Jezero Crater Paleolake Delta  ·  Perseverance Landing Site  (Mars 2020)',
        color='white', fontsize=14, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    out = FIG_DIR / 'fig1_jezero_topo.png'
    fig.savefig(out, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  Saved {out.name}')
    return out


# ── Fig 2: Channel Network Graph ──────────────────────────────────────────
def fig_network_graph(dem, pts, W, acc, dx, chan_thr):
    """Channel-network graph overlaid on DEM — nodes coloured by elevation."""
    nrows, ncols = dem.shape
    ext_km = [0, ncols * dx / 1e3, 0, nrows * dx / 1e3]
    hs     = make_hillshade(dem, dx)

    fig, ax = plt.subplots(figsize=(10, 14))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    ax.imshow(hs, origin='upper', cmap='gray',
              extent=ext_km, alpha=0.45, vmin=0, vmax=1)
    elev_min, elev_max = np.nanpercentile(dem, [2, 98])
    ax.imshow(dem, origin='upper', cmap=CMAP_TOPO,
              extent=ext_km, alpha=0.55, vmin=elev_min, vmax=elev_max)

    # Draw edges
    n = len(pts)
    x_km = pts[:, 0] / 1e3
    y_km = pts[:, 1] / 1e3
    for i in range(n):
        for j in range(i + 1, n):
            if W[i, j] > 0:
                ax.plot([x_km[i], x_km[j]], [y_km[i], y_km[j]],
                        '-', color='#00aaff', alpha=0.25, lw=0.5)

    # Draw nodes coloured by elevation
    el_norm = Normalize(vmin=pts[:, 2].min(), vmax=pts[:, 2].max())
    sc = ax.scatter(x_km, y_km, c=pts[:, 2], cmap='plasma',
                    norm=el_norm, s=6, zorder=5, linewidths=0)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label('Node elevation (m)', color='white', fontsize=9)
    cbar.ax.yaxis.set_tick_params(color='white', labelsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    # Degree distribution as inset
    deg = (W > 0).sum(axis=1)
    ax_in = ax.inset_axes([0.68, 0.02, 0.30, 0.18])
    ax_in.hist(deg, bins=range(1, deg.max() + 2), color='#00aaff',
               edgecolor='white', linewidth=0.4)
    ax_in.set_xlabel('Degree', color='white', fontsize=7)
    ax_in.set_ylabel('Count', color='white', fontsize=7)
    ax_in.set_title('Degree dist.', color='white', fontsize=7.5)
    ax_in.tick_params(colors='white', labelsize=6)
    ax_in.set_facecolor('#1a1f2e')
    for spine in ax_in.spines.values(): spine.set_edgecolor('gray')

    ax.set_title(f'Channel Network Graph  (N={n} nodes, k={K_NN}-NN)',
                 color='white', **TITLE_KW)
    ax.set_xlabel('Easting (km)', color='white', **LABEL_KW)
    ax.set_ylabel('Northing (km)', color='white', **LABEL_KW)
    ax.tick_params(colors='white', **TICK_KW)
    for spine in ax.spines.values(): spine.set_edgecolor('white')

    # Legend patches
    patch_node = mpatches.Patch(color='#cc44ff', label=f'{n} channel nodes')
    patch_edge = mpatches.Patch(color='#00aaff', label=f'{int((W>0).sum()//2)} graph edges')
    ax.legend(handles=[patch_node, patch_edge], loc='upper right',
              facecolor='black', edgecolor='gray',
              labelcolor='white', fontsize=8)

    plt.tight_layout()
    out = FIG_DIR / 'fig2_channel_graph.png'
    fig.savefig(out, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  Saved {out.name}')
    return out


# ── Fig 3: Eigenspectrum ───────────────────────────────────────────────────
def fig_eigenspectrum(eigvals, eigvecs, pts, dx):
    """Eigenvalue spectrum with Fiedler gap, and spatial map of Fiedler mode."""
    n   = len(eigvals)
    fig = plt.figure(figsize=(15, 6), facecolor='#0d1117')
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # Panel A: full eigenvalue spectrum
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('#0d1117')
    colors_e = plt.cm.magma(np.linspace(0.1, 0.9, n))
    ax1.bar(range(n), eigvals, color=colors_e, width=1.0, edgecolor='none')
    n_zero_ev = int(np.sum(eigvals < 1e-6))
    lam_f = float(eigvals[n_zero_ev]) if n_zero_ev < len(eigvals) else 0.0
    ax1.axvline(n_zero_ev, color='#ffcc00', lw=2, ls='--',
                label=f'λ_fiedler=λ_{n_zero_ev}={lam_f:.3f}  ({n_zero_ev} components)')
    ax1.set_xlabel('Eigenvalue index  l', color='white', **LABEL_KW)
    ax1.set_ylabel('λₗ', color='white', **LABEL_KW)
    ax1.set_title('Graph Laplacian Eigenspectrum', color='white', **TITLE_KW)
    ax1.tick_params(colors='white', **TICK_KW)
    for sp in ax1.spines.values(): sp.set_edgecolor('gray')
    ax1.legend(facecolor='black', edgecolor='gray', labelcolor='white', fontsize=8)
    ax1.text(0.6, 0.95,
             f'N = {n}\nλ_max = {eigvals[-1]:.2f}\nGap ratio = {eigvals[1]/max(eigvals[2],1e-6):.3f}',
             transform=ax1.transAxes, va='top', ha='left',
             color='lightgray', fontsize=8,
             bbox=dict(boxstyle='round,pad=0.4', fc='#1a1f2e', alpha=0.8))

    # Panel B: low-index eigenvalues (zoom on Fiedler region)
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor('#0d1117')
    k_show = min(40, n)
    ax2.stem(range(k_show), eigvals[:k_show],
             linefmt='#5599ff', markerfmt='o', basefmt='gray')
    n_zero_ev2 = int(np.sum(eigvals < 1e-6))
    lam_f2 = float(eigvals[n_zero_ev2]) if n_zero_ev2 < len(eigvals) else 0.0
    ax2.axvline(n_zero_ev2, color='#ffcc00', lw=1.5, ls='--')
    ax2.fill_between([n_zero_ev2 - 0.5, n_zero_ev2 + 1.5], 0, lam_f2,
                     color='#ffcc00', alpha=0.10, label=f'Spectral gap (β₀={n_zero_ev2})')
    ax2.set_xlabel('Eigenvalue index  l', color='white', **LABEL_KW)
    ax2.set_ylabel('λₗ', color='white', **LABEL_KW)
    ax2.set_title(f'First {k_show} Eigenvalues', color='white', **TITLE_KW)
    ax2.tick_params(colors='white', **TICK_KW)
    for sp in ax2.spines.values(): sp.set_edgecolor('gray')
    ax2.legend(facecolor='black', edgecolor='gray', labelcolor='white', fontsize=8)

    # Panel C: spatial map of first non-trivial eigenvector (mode l=β₀)
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor('#0d1117')
    n_zero_sp = int(np.sum(eigvals < 1e-6))
    fiedler_idx = min(n_zero_sp, len(eigvals) - 1)
    fiedler_vals = eigvecs[:, fiedler_idx]
    fv_norm = Normalize(vmin=fiedler_vals.min(), vmax=fiedler_vals.max())
    sc = ax3.scatter(
        pts[:, 0] / 1e3, pts[:, 1] / 1e3,
        c=fiedler_vals, cmap='RdBu_r', norm=fv_norm, s=8, zorder=3
    )
    cbar = fig.colorbar(sc, ax=ax3, fraction=0.04, pad=0.02)
    cbar.set_label('Fiedler amplitude', color='white', fontsize=8)
    cbar.ax.yaxis.set_tick_params(color='white', labelsize=7)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
    ax3.set_title(f'Fiedler Mode  (l={fiedler_idx})  — Inter-Basin Bisection',
                  color='white', **TITLE_KW)
    ax3.set_xlabel('Easting (km)', color='white', **LABEL_KW)
    ax3.set_ylabel('Northing (km)', color='white', **LABEL_KW)
    ax3.tick_params(colors='white', **TICK_KW)
    for sp in ax3.spines.values(): sp.set_edgecolor('gray')
    ax3.text(0.02, 0.98,
             'Red → Blue = natural\ngraph bisection',
             transform=ax3.transAxes, va='top', ha='left',
             color='lightgray', fontsize=7.5, style='italic')

    fig.suptitle('Spectral Decomposition of the Jezero Channel Network Graph',
                 color='white', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = FIG_DIR / 'fig3_eigenspectrum.png'
    fig.savefig(out, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  Saved {out.name}')
    return out


# ── Fig 4: Fixed-Point Kernel ──────────────────────────────────────────────
def fig_fixed_point_kernel(eigvals, h_star, h0, tradeoff):
    """h*(λ) vs h₀(λ): spectral reweighting by the MaxCal fixed point."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), facecolor='#0d1117')

    # Panel A: h*(λ) vs h₀(λ) as functions of λ
    ax = axes[0]
    ax.set_facecolor('#1a1f2e')
    sort_idx = np.argsort(eigvals)
    lam = eigvals[sort_idx]
    hs  = h_star[sort_idx]
    h0s = h0[sort_idx]

    ax.fill_between(lam, h0s, hs, where=(hs >= h0s),
                    color='#22cc88', alpha=0.25, label='h* > h₀  (mode amplified)')
    ax.fill_between(lam, h0s, hs, where=(hs < h0s),
                    color='#ff6644', alpha=0.25, label='h* < h₀  (mode suppressed)')
    ax.plot(lam, h0s, '--', color='#aaaaaa', lw=1.5, label='h₀(λ) = e^{-λ}  vacuum')
    ax.plot(lam, hs,  '-',  color='#ffcc00', lw=2.0, label='h*(λ)  fixed-point kernel')

    ax.set_xlabel('Eigenvalue  λₗ', color='white', **LABEL_KW)
    ax.set_ylabel('Spectral weight  h(λ)', color='white', **LABEL_KW)
    ax.set_title('Fixed-Point Kernel  h*(λ)  vs  Vacuum Reference  h₀(λ)',
                 color='white', **TITLE_KW)
    ax.tick_params(colors='white', **TICK_KW)
    for sp in ax.spines.values(): sp.set_edgecolor('gray')
    ax.legend(facecolor='black', edgecolor='gray', labelcolor='white', fontsize=8.5,
              loc='upper right')

    # Panel B: per-mode conservation deficit D_m
    ax2 = axes[1]
    ax2.set_facecolor('#1a1f2e')
    D_m = tradeoff['D_m'][sort_idx]
    ax2.bar(range(len(D_m)), D_m, color=np.where(D_m >= 0, '#22cc88', '#ff6644'),
            width=1.0, edgecolor='none', alpha=0.8)
    ax2.axhline(0, color='white', lw=0.8, ls='-')
    ax2.set_xlabel('Mode index  (sorted by λ)', color='white', **LABEL_KW)
    ax2.set_ylabel('Conservation deficit  Dₘ', color='white', **LABEL_KW)
    ax2.set_title('Per-Mode Conservation Deficit  Dₘ = ∂(Rₗ−Tₗ)/∂hₘ',
                  color='white', **TITLE_KW)
    ax2.tick_params(colors='white', **TICK_KW)
    for sp in ax2.spines.values(): sp.set_edgecolor('gray')

    deficit = tradeoff['conservation_deficit']
    delta_p = tradeoff['Delta_prime']
    ax2.text(0.98, 0.96,
             f'Mean |Dₘ| = {deficit:.4f}\nΔ\' = {delta_p:.4f}',
             transform=ax2.transAxes, va='top', ha='right',
             color='white', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.4', fc='black', alpha=0.6))
    green_p = mpatches.Patch(color='#22cc88', alpha=0.8, label='Dₘ ≥ 0  (source dominates)')
    red_p   = mpatches.Patch(color='#ff6644', alpha=0.8, label='Dₘ < 0  (sink dominates)')
    ax2.legend(handles=[green_p, red_p], facecolor='black', edgecolor='gray',
               labelcolor='white', fontsize=8)

    plt.tight_layout()
    out = FIG_DIR / 'fig4_fixed_point_kernel.png'
    fig.savefig(out, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  Saved {out.name}')
    return out


# ── Fig 5: Biosignature Summary ────────────────────────────────────────────
def fig_biosig_summary(eigvals, h_star, h0, tradeoff,
                       beta0, beta1, beta1_abio, n_nodes,
                       delta_x_km, delta_y_km, lam_fiedler=0.0):
    """Publication-quality one-page biosignature summary for planetary scientists."""

    H_obs   = spectral_entropy(h_star)
    H_vac   = spectral_entropy(h0)
    delta_H = H_obs - H_vac
    lam1    = lam_fiedler   # first non-trivial eigenvalue
    delta_p = tradeoff['Delta_prime']
    deficit = tradeoff['conservation_deficit']
    delta_b1 = beta1 - beta1_abio
    kmin     = beta0 + beta1
    kmin_abio = beta0 + beta1_abio

    # Interpretation
    if delta_b1 > 0:
        interp_color = '#44ee88'
        interp_text  = 'Δβ₁ > 0: topological excess\nconsistent with structured flow'
    else:
        interp_color = '#aaaaaa'
        interp_text  = 'Δβ₁ = 0: topology matches\nabiotic D8 null model'

    fig = plt.figure(figsize=(14, 10), facecolor='#0d1117')
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.40)

    # ── A: h*(λ) curve ──
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_facecolor('#1a1f2e')
    sort_idx = np.argsort(eigvals)
    lam = eigvals[sort_idx]; hs = h_star[sort_idx]; h0s = h0[sort_idx]
    ax_a.fill_between(lam, h0s, hs, where=(hs >= h0s), color='#22cc88', alpha=0.3)
    ax_a.fill_between(lam, h0s, hs, where=(hs < h0s),  color='#ff6644', alpha=0.3)
    ax_a.plot(lam, h0s, '--', color='#888888', lw=1.5, label='h₀  vacuum')
    ax_a.plot(lam, hs,  '-',  color='#ffcc00', lw=2.2, label='h*  fixed-point')
    ax_a.set_xlabel('λₗ', color='white', fontsize=9)
    ax_a.set_ylabel('h(λ)', color='white', fontsize=9)
    ax_a.set_title('Fixed-Point Kernel', color='white', fontsize=10, fontweight='bold')
    ax_a.tick_params(colors='white', labelsize=8)
    ax_a.legend(facecolor='#0d1117', edgecolor='gray', labelcolor='white', fontsize=7.5)
    for sp in ax_a.spines.values(): sp.set_edgecolor('gray')

    # ── B: eigenspectrum bar ──
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_facecolor('#1a1f2e')
    k_show = min(50, len(eigvals))
    ax_b.bar(range(k_show), eigvals[:k_show],
             color=plt.cm.magma(np.linspace(0.2, 0.9, k_show)), width=1.0)
    ax_b.axvline(1, color='#ffcc00', lw=1.5, ls='--', label=f'λ₁={lam1:.3f}')
    ax_b.set_xlabel('l', color='white', fontsize=9)
    ax_b.set_ylabel('λₗ', color='white', fontsize=9)
    ax_b.set_title('Eigenspectrum', color='white', fontsize=10, fontweight='bold')
    ax_b.tick_params(colors='white', labelsize=8)
    ax_b.legend(facecolor='#0d1117', edgecolor='gray', labelcolor='white', fontsize=7.5)
    for sp in ax_b.spines.values(): sp.set_edgecolor('gray')

    # ── C: Betti number comparison ──
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.set_facecolor('#1a1f2e')
    categories = ['β₀\n(components)', 'β₁\n(cycles)']
    obs_vals   = [beta0, beta1]
    abio_vals  = [1,     beta1_abio]
    x = np.arange(len(categories))
    w = 0.35
    bars_obs  = ax_c.bar(x - w/2, obs_vals,  w, color='#22cc88', label='Observed', alpha=0.85)
    bars_abio = ax_c.bar(x + w/2, abio_vals, w, color='#5577ff', label='Abiotic null', alpha=0.85)
    for bar in bars_obs:
        ax_c.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                  str(int(bar.get_height())), ha='center', color='white', fontsize=9)
    for bar in bars_abio:
        ax_c.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                  str(int(bar.get_height())), ha='center', color='white', fontsize=9)
    ax_c.set_xticks(x); ax_c.set_xticklabels(categories, color='white', fontsize=9)
    ax_c.set_ylabel('Count', color='white', fontsize=9)
    ax_c.set_title('Betti Numbers', color='white', fontsize=10, fontweight='bold')
    ax_c.tick_params(colors='white', labelsize=8)
    ax_c.legend(facecolor='#0d1117', edgecolor='gray', labelcolor='white', fontsize=7.5)
    for sp in ax_c.spines.values(): sp.set_edgecolor('gray')
    ax_c.text(0.5, 0.97, f'Δβ₁ = {delta_b1:+d}',
              transform=ax_c.transAxes, ha='center', va='top',
              color=interp_color, fontsize=11, fontweight='bold')

    # ── D: scalar diagnostics panel ──
    ax_d = fig.add_subplot(gs[1, :2])
    ax_d.set_facecolor('#0d1117')
    ax_d.axis('off')

    diag_rows = [
        ('Spectral entropy  H[h*]',    f'{H_obs:.4f} nats',  'entropy of fixed-point kernel'),
        ('Vacuum entropy  H[h₀]',      f'{H_vac:.4f} nats',  'maximum-diffuse reference'),
        ('Entropy excess  ΔH',          f'{delta_H:+.4f} nats',
         '> 0 → kernel more concentrated than vacuum'),
        ('Fiedler value  λ_fiedler',   f'{lam1:.4f}',
         f'first non-trivial gap  (β₀={beta0} sub-basins)'),
        ('Hessian gap  Δ\'',            f'{delta_p:.4f}',     'stability margin of fixed point'),
        ('Conservation deficit  ⟨|Dₘ|⟩', f'{deficit:.4f}',  'mean deviation from identity Rₗ = Tₗ'),
        ('k_min (obligate modes)',      f'{kmin}  (β₀+β₁)',   'minimum resolution for Jezero'),
        ('k_min abiotic null',         f'{kmin_abio}',        'from D8 drainage tree'),
        ('Network nodes  N',           str(n_nodes),          'channel cells subsampled'),
    ]

    col_x = [0.01, 0.38, 0.68]
    header_color = '#8888cc'
    ax_d.text(col_x[0], 0.97, 'Diagnostic', color=header_color,
              fontsize=9, fontweight='bold', va='top', transform=ax_d.transAxes)
    ax_d.text(col_x[1], 0.97, 'Value', color=header_color,
              fontsize=9, fontweight='bold', va='top', transform=ax_d.transAxes)
    ax_d.text(col_x[2], 0.97, 'Physical meaning', color=header_color,
              fontsize=9, fontweight='bold', va='top', transform=ax_d.transAxes)

    row_h = 0.88 / len(diag_rows)
    for i, (name, val, meaning) in enumerate(diag_rows):
        y = 0.90 - i * row_h
        row_bg = '#111827' if i % 2 == 0 else '#1a1f2e'
        ax_d.add_patch(mpatches.FancyBboxPatch(
            (0.0, y - row_h * 0.3), 1.0, row_h * 0.85,
            boxstyle='round,pad=0.005', fc=row_bg, ec='none',
            transform=ax_d.transAxes, zorder=0
        ))
        ax_d.text(col_x[0], y, name,    color='white',      fontsize=8.5,
                  va='center', transform=ax_d.transAxes)
        ax_d.text(col_x[1], y, val,     color='#ffdd66',    fontsize=8.5,
                  va='center', transform=ax_d.transAxes, fontweight='bold')
        ax_d.text(col_x[2], y, meaning, color='lightgray',  fontsize=8,
                  va='center', transform=ax_d.transAxes, style='italic')

    ax_d.set_title('Kernelcal Diagnostics — Jezero Delta Network',
                   color='white', fontsize=10, fontweight='bold', pad=6)

    # ── E: Interpretation box ──
    ax_e = fig.add_subplot(gs[1, 2])
    ax_e.set_facecolor('#1a1f2e')
    ax_e.axis('off')

    interp_lines = [
        ('Framework:', 'Spectral Kernel Dynamics\n(Das 2026 · P4)', '#8888cc'),
        ('Site:', 'Jezero Crater Delta\n(Perseverance landing)', 'white'),
        ('Topology:', interp_text, interp_color),
        ('Interpretation:',
         f'Fixed-point kernel shows\n{"concentrated" if delta_H < 0 else "diffuse"} spectral mass\n'
         f'{"→ low-mode controller active" if delta_H < 0 else "→ near-vacuum (abiotic)"}',
         '#ffcc00'),
        ('Next step:',
         'Cross-kernel ‖k_cross‖_HS\nwith SHERLOC/PIXL chemistry\nto test coupling',
         '#44aaff'),
    ]

    y = 0.96
    for label, body, color in interp_lines:
        ax_e.text(0.05, y, label, color='#8888cc', fontsize=8, fontweight='bold',
                  va='top', transform=ax_e.transAxes)
        y -= 0.06
        ax_e.text(0.05, y, body, color=color, fontsize=8,
                  va='top', transform=ax_e.transAxes)
        y -= 0.11

    ax_e.set_title('Scientific Interpretation', color='white',
                   fontsize=10, fontweight='bold', pad=6)
    for sp in ax_e.spines.values(): sp.set_edgecolor('gray')

    fig.suptitle(
        'Jezero Crater  ·  Spectral Kernel Biosignature Analysis  ·  kernelcal v0.1',
        color='white', fontsize=13, fontweight='bold', y=1.01
    )

    out = FIG_DIR / 'fig5_biosig_summary.png'
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  Saved {out.name}')
    return out


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print('=' * 65)
    print('Jezero Crater — Spectral Kernel Dynamics Pipeline')
    print(f'  DEM:     {DEM_PATH}')
    print(f'  Output:  {FIG_DIR}')
    print('=' * 65)

    # ── 1. Load & Downsample ───────────────────────────────────────────────
    print('\n[1] Loading DEM …')
    factor = CELL_SIZE   # 1m → 20m
    dem, dx, meta = load_and_downsample(DEM_PATH, factor)

    print(f'  Elevation range: {np.nanmin(dem):.1f} – {np.nanmax(dem):.1f} m')
    print(f'  Valid cells: {np.sum(~np.isnan(dem)):,}')

    # ── 2. Flow Routing ────────────────────────────────────────────────────
    print('\n[2] Computing D8 flow direction + accumulation …')
    fdir = vectorized_d8(dem, dx)
    acc  = fast_flow_acc(dem, fdir)
    print(f'  Max accumulation: {acc.max():,} cells = '
          f'{acc.max() * dx**2 / 1e6:.2f} km²')

    # ── 3. Channel Network Extraction ─────────────────────────────────────
    print('\n[3] Extracting channel network …')
    nrows, ncols = dem.shape
    chan_thr = max(50, int(nrows * ncols * ACC_THR_FACTOR))
    pts = extract_network_nodes(dem, acc, dx, n_max=N_MAX, threshold=chan_thr)
    n   = len(pts)

    # ── 4. Graph & Laplacian ───────────────────────────────────────────────
    print(f'\n[4] Building {K_NN}-NN graph (N={n}) and Laplacian …')
    L, W = build_knn_laplacian(pts, k=K_NN)
    print(f'  Edges: {int((W > 0).sum() // 2)}')

    # ── 5. Eigenpairs ──────────────────────────────────────────────────────
    print('\n[5] Computing eigenpairs (scipy.linalg.eigh) …')
    eigvals, eigvecs = scipy_eigh(L)
    eigvals = np.maximum(eigvals, 0.0)   # numerical floor
    # First non-trivial eigenvalue: skip β₀ zero modes
    n_zero = int(np.sum(eigvals < 1e-6))
    lam_fiedler = float(eigvals[n_zero]) if n_zero < len(eigvals) else 0.0
    print(f'  λ₀ = {eigvals[0]:.2e}  ({n_zero} zero eigenvalues = {n_zero} components)')
    print(f'  λ_fiedler = λ_{n_zero} = {lam_fiedler:.4f}  (first non-trivial spectral gap)')
    print(f'  λ_max = {eigvals[-1]:.4f}')

    # ── 6. kernelcal Diagnostics ───────────────────────────────────────────
    print('\n[6] Running kernelcal MaxCal fixed-point iteration …')
    w_modes   = eigvals.copy(); w_modes[0] = w_modes[1] if n > 1 else 1.0
    h0        = np.exp(-eigvals); h0 = np.maximum(h0, 1e-10)
    h_star, info = fixed_point_kernel(L, h0=h0, mu2=MU2, sigma2=SIGMA2, w=w_modes)
    print(f'  Converged: {info["converged"]}  ({info["n_iter"]} iters, '
          f'residual={info["residual"]:.2e})')

    # Clip h_star for numerically stable diagnostics (avoid 1/~0 in deficit)
    h_star_safe = np.maximum(h_star, 1e-8)

    H_obs = spectral_entropy(h_star_safe)
    H_vac = spectral_entropy(h0)
    delta_p = fiedler_mode_gap(h_star_safe, L, mu2=MU2, sigma2=SIGMA2, w=w_modes)
    tradeoff = stability_conservation_tradeoff(h_star_safe, L, mu2=MU2, sigma2=SIGMA2, w=w_modes)
    # Clip deficit values for display (high-λ modes hit numerical floor)
    tradeoff['D_m'] = np.clip(tradeoff['D_m'], -1e4, 1e4)
    tradeoff['conservation_deficit'] = float(np.mean(np.abs(tradeoff['D_m'])))
    h_star = h_star_safe

    print(f'  H[h*] = {H_obs:.4f} nats   H[h₀] = {H_vac:.4f} nats   ΔH = {H_obs-H_vac:+.4f}')
    print(f'  Δ\'    = {delta_p:.4f}')
    print(f'  ⟨|Dₘ|⟩ = {tradeoff["conservation_deficit"]:.4f}')

    # ── 7. Betti Numbers ───────────────────────────────────────────────────
    print('\n[7] Estimating Betti numbers …')
    beta0, beta1 = betti_from_laplacian(L, W)
    beta1_abio   = abiotic_beta1_null(n)
    delta_b1     = beta1 - beta1_abio
    print(f'  β₀ = {beta0}  (connected components)')
    print(f'  β₁ = {beta1}  (independent cycles)   abiotic null: {beta1_abio}')
    print(f'  Δβ₁ = {delta_b1:+d}')

    # ── 8. Figures ─────────────────────────────────────────────────────────
    print('\n[8] Generating figures …')
    # Approx delta position in km (northwest of crater — Neretva Vallis entry)
    delta_x_km = 3.0
    delta_y_km = ncols * dx / 1e3 * 0.8

    fig_topo(dem, acc, dx, meta, chan_thr)
    fig_network_graph(dem, pts, W, acc, dx, chan_thr)
    fig_eigenspectrum(eigvals, eigvecs, pts, dx)
    fig_fixed_point_kernel(eigvals, h_star, h0, tradeoff)
    n_zero = int(np.sum(eigvals < 1e-6))
    lam_fiedler = float(eigvals[n_zero]) if n_zero < len(eigvals) else 0.0
    fig_biosig_summary(eigvals, h_star, h0, tradeoff,
                       beta0, beta1, beta1_abio, n,
                       delta_x_km, delta_y_km,
                       lam_fiedler=lam_fiedler)

    # ── 9. Summary ─────────────────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('JEZERO DELTA — KERNELCAL RESULTS SUMMARY')
    print('=' * 65)
    print(f'  DEM resolution used      : {dx:.0f} m/cell  ({factor}× downsample from 1 m)')
    print(f'  Basin area               : {np.sum(~np.isnan(dem)) * dx**2 / 1e6:.1f} km²')
    print(f'  Channel nodes (N)        : {n}  (acc ≥ {chan_thr} cells = '
          f'{chan_thr*dx**2/1e6:.2f} km²)')
    print(f'  Fiedler value  λ_fiedler  : {lam_fiedler:.4f}  (λ_{n_zero}, first non-trivial)')
    print(f'  Spectral entropy  H[h*]  : {H_obs:.4f} nats  (vacuum H[h₀] = {H_vac:.4f})')
    print(f'  Hessian gap  Δ\'          : {delta_p:.4f}')
    print(f'  Betti numbers            : β₀={beta0}  β₁={beta1}  (null β₁={beta1_abio})')
    print(f'  Topological excess  Δβ₁  : {delta_b1:+d}')
    print(f'  Figures saved to         : {FIG_DIR}')
    print('=' * 65)
    print()
    print('  Physical interpretation:')
    if H_obs < H_vac:
        print('  → H[h*] < H[h₀]: kernel is more concentrated than vacuum.')
        print('    Low-index modes carry excess spectral weight.')
        print('    Consistent with a low-mode controller shaping large-scale flow.')
    else:
        print('  → H[h*] ≈ H[h₀]: kernel close to vacuum (diffuse).')
        print('    Network behaves as a passive abiotic D8 drainage system.')
    if delta_b1 > 0:
        print(f'  → Δβ₁ = +{delta_b1}: excess topological loops beyond D8 null.')
        print('    Candidate signature of a hydraulic controller (geomorphic or biotic).')
    else:
        print('  → Δβ₁ = 0: topology consistent with abiotic D8 drainage tree.')
    print()
    print('  Next steps for Mars 2020 context:')
    print('  1. Compute cross-kernel ‖k_cross‖_HS coupling DEM graph ⊗ PIXL chemistry graph')
    print('  2. Apply phase-transition sweep along Neretva inlet channel')
    print('  3. Compare with CRISM spectral map for mineralogy–topology correlation')
    print('=' * 65)


if __name__ == '__main__':
    main()
