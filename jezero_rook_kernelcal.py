#!/usr/bin/env python3
"""
jezero_rook_kernelcal.py
========================
Jezero Crater delta — rook-adjacency kernelcal WITH ARTIFACT FILTERING.

Key improvements over the first rook version:
  1. DEM pre-filter: E-W directional median removes N-S swath-seam step edges
     (HRSC/HiRISE DEMs have 1–10 m height offsets at mosaicking seams that
      D8 routing mistakes for channel valleys → straight N-S artifact chains)
  2. Post-hoc linear-chain filter: removes any remaining near-vertical connected
     components (col_std < 2 px AND len ≥ 12) that survive the DEM filter
  3. LCC analysis: reports kernelcal on the largest connected component alone
     (MaxCal theory is most meaningful on a connected graph)
  4. Parameter sweep: β₁ and ΔH over 4 acc-thresholds × 4 N-values to test
     robustness and show where Jezero and AZ Plateau intervals separate
  5. Component size distribution: quantifies how many of β₀ components are
     single pixels vs. substantive fragments

Figures → ./jezero_rook_figures/
  fig1_overview.png          — DEM hillshade + flow acc + channel mask
  fig2_artifact_filter.png   — before/after artifact removal (NEW)
  fig3_rook_graph.png        — rook graph with artifact-clean nodes
  fig4_junction_loops.png    — confluence zoom panels
  fig5_spectra.png           — eigenspectrum + h*(λ) for full & LCC
  fig6_component_dist.png    — β₀ component-size histogram (NEW)
  fig7_parameter_sweep.png   — β₁ and ΔH across parameter space (NEW)
  fig8_phase_space.png       — phase space: AZ / Jezero-clean / cities
"""

from __future__ import annotations
import math, sys
from pathlib import Path
from collections import deque

import numpy as np
from scipy.linalg import eigh as scipy_eigh
from scipy.ndimage import median_filter as ndimage_median_filter
from scipy.stats import linregress
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

KCAL_ROOT = Path(__file__).parent
sys.path.insert(0, str(KCAL_ROOT))
from kernelcal.terrain.diagnostics import (
    spectral_entropy, fixed_point_kernel, fiedler_mode_gap,
)

try:
    import rasterio, rasterio.crs, rasterio.transform
    from rasterio.warp import reproject, Resampling
    HAS_RASTERIO = True
except ImportError:
    raise RuntimeError('rasterio is required')

# ── paths & config ────────────────────────────────────────────────────────────
DEM_PATH = Path.home() / 'Downloads' / 'swjez-DEM_1m.tif'
FIG_DIR  = KCAL_ROOT / 'jezero_rook_figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

CELL_SIZE      = 20        # downsample 1 m → 20 m working grid
ACC_THR_FACTOR = 0.0004    # default flow-accumulation threshold fraction
N_MAX          = 3000      # default max channel nodes
MU2            = 2.0
SIGMA2         = 1.0

# Artifact filter parameters
EW_FILTER_WIDTH = 9         # E-W median kernel width (pixels) to suppress step edges
ARTIFACT_COL_STD_MAX = 2.0  # max column std-dev (pixels) → near-vertical chain
ARTIFACT_MIN_SIZE    = 12   # min chain length to flag as artifact

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9.5,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 7.5, 'legend.framealpha': 0.95,
    'figure.dpi': 150, 'savefig.dpi': 250, 'savefig.bbox': 'tight',
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
})

C_JEZ   = '#CC79A7'   # Jezero pink
C_AZ    = '#009E73'   # AZ green
C_CITY  = '#0072B2'   # city blue
C_ART   = '#D55E00'   # artifact orange


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & DOWNSAMPLE DEM
# ══════════════════════════════════════════════════════════════════════════════

def load_dem() -> tuple[np.ndarray, object]:
    print(f'  Reading {DEM_PATH.name}  ({DEM_PATH.stat().st_size // 1024**2} MB)…')
    with rasterio.open(DEM_PATH) as src:
        dem_1m = src.read(1).astype(float)
        dem_1m[dem_1m == src.nodata] = np.nan
        transform_1m = src.transform
        crs = src.crs

    rows, cols = dem_1m.shape
    rc = (rows // CELL_SIZE) * CELL_SIZE
    cc = (cols // CELL_SIZE) * CELL_SIZE
    ds = np.nanmean(
        dem_1m[:rc, :cc].reshape(rc // CELL_SIZE, CELL_SIZE,
                                 cc // CELL_SIZE, CELL_SIZE),
        axis=(1, 3))
    t = transform_1m
    from rasterio.transform import Affine
    ds_transform = Affine(t.a * CELL_SIZE, t.b, t.c,
                          t.d, t.e * CELL_SIZE, t.f)
    print(f'  DEM {ds.shape}  px={CELL_SIZE} m  '
          f'elev {np.nanmin(ds):.0f}–{np.nanmax(ds):.0f} m')
    return ds, ds_transform


# ══════════════════════════════════════════════════════════════════════════════
# 2. ARTIFACT FILTERING
# ══════════════════════════════════════════════════════════════════════════════

def prefilter_swath_artifacts(dem: np.ndarray,
                              ew_width: int = EW_FILTER_WIDTH) -> np.ndarray:
    """E-W directional median filter to suppress N-S swath-boundary step edges.

    HRSC/HiRISE DEMs mosaicked from push-broom strips often have 1–10 m height
    offsets at swath seams.  These seams run N-S (along-track).  D8 routing
    interprets the step as a valley → all pixels drain toward the seam →
    artificial high-accumulation line running N-S.

    A 1 × ew_width median filter blurs the step in the cross-track (E-W)
    direction while largely preserving real N-S channel walls (which are
    narrow relative to the filter width).
    """
    valid    = np.isfinite(dem)
    fill_val = float(np.nanmedian(dem))
    dem_fill = np.where(valid, dem, fill_val)
    dem_smooth = ndimage_median_filter(dem_fill, size=(1, ew_width))
    dem_filtered = np.where(valid, dem_smooth, np.nan)
    # Compute max step removed
    diff = np.nanmax(np.abs(dem - dem_filtered))
    print(f'  E-W median filter (width={ew_width}): max step suppressed = {diff:.1f} m')
    return dem_filtered


def _find_components(n: int, edges: list[tuple[int, int]]) -> list[np.ndarray]:
    """BFS connected-component labelling on an edge list."""
    adj = [[] for _ in range(n)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    visited = np.zeros(n, dtype=bool)
    components = []
    for start in range(n):
        if not visited[start]:
            comp = []
            q = deque([start])
            visited[start] = True
            while q:
                node = q.popleft()
                comp.append(node)
                for nb in adj[node]:
                    if not visited[nb]:
                        visited[nb] = True
                        q.append(nb)
            components.append(np.array(comp, dtype=int))
    return components


def filter_linear_chains(ri: np.ndarray, ci: np.ndarray,
                         edges: list[tuple[int, int]],
                         min_size: int = ARTIFACT_MIN_SIZE,
                         col_std_max: float = ARTIFACT_COL_STD_MAX
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Remove near-vertical connected chains — residual swath artifacts.

    Criterion: a connected component with
        • ≥ min_size nodes
        • column std-dev < col_std_max (stays within ~40 m horizontally)
    is almost certainly a seam artifact, not a real channel.

    Real N-S channels are dendritic and have tributaries entering from
    varying columns.  Pure seam artifacts are column-invariant chains.

    Returns keep_mask (bool array, True = node retained).
    """
    n = len(ri)
    components = _find_components(n, edges)

    artifact_nodes = set()
    n_artifacts = 0
    for comp in components:
        if len(comp) < min_size:
            continue
        col_std = float(np.std(ci[comp].astype(float)))
        if col_std < col_std_max:
            artifact_nodes.update(comp.tolist())
            n_artifacts += 1

    if artifact_nodes:
        print(f'  Linear-chain filter: removed {len(artifact_nodes)} nodes '
              f'from {n_artifacts} near-vertical chains '
              f'(col_std < {col_std_max} px, size ≥ {min_size})')
    else:
        print('  Linear-chain filter: no artifact chains detected')

    keep = np.array([i not in artifact_nodes for i in range(n)], dtype=bool)
    return keep


# ══════════════════════════════════════════════════════════════════════════════
# 3. D8 FLOW ACCUMULATION
# ══════════════════════════════════════════════════════════════════════════════

NBR8  = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
DIST8 = [math.sqrt(2),1,math.sqrt(2),1,1,math.sqrt(2),1,math.sqrt(2)]

def d8_flow_dir(dem: np.ndarray) -> np.ndarray:
    rows, cols = dem.shape
    padded = np.pad(np.where(np.isfinite(dem), dem, np.nanmin(dem)), 1, mode='edge')
    drops = np.full((rows, cols, 8), -np.inf)
    for k, (dr, dc) in enumerate(NBR8):
        nbr = padded[1+dr:1+dr+rows, 1+dc:1+dc+cols]
        drops[:, :, k] = (np.where(np.isfinite(dem), dem, np.nan) - nbr) / DIST8[k]
    return np.argmax(drops, axis=2).astype(np.int8)

def flow_accum(fdir: np.ndarray) -> np.ndarray:
    rows, cols = fdir.shape
    acc    = np.ones((rows, cols), dtype=np.int32)
    in_deg = np.zeros_like(acc)
    nbr    = np.array(NBR8, dtype=int)
    for r in range(rows):
        for c in range(cols):
            k = fdir[r, c]; nr = r + nbr[k,0]; nc = c + nbr[k,1]
            if 0 <= nr < rows and 0 <= nc < cols and (nr != r or nc != c):
                in_deg[nr, nc] += 1
    q = deque((r, c) for r in range(rows) for c in range(cols)
              if in_deg[r, c] == 0)
    while q:
        r, c = q.popleft()
        k = fdir[r, c]; nr = r + nbr[k,0]; nc = c + nbr[k,1]
        if 0 <= nr < rows and 0 <= nc < cols and (nr != r or nc != c):
            acc[nr, nc] += acc[r, c]
            in_deg[nr, nc] -= 1
            if in_deg[nr, nc] == 0:
                q.append((nr, nc))
    return acc


# ══════════════════════════════════════════════════════════════════════════════
# 4. ROOK-ADJACENCY GRAPH
# ══════════════════════════════════════════════════════════════════════════════

def channel_mask(acc, dem, thr_factor=ACC_THR_FACTOR):
    thr = max(int(thr_factor * acc.max()), 5)
    return (acc >= thr) & np.isfinite(dem)

def pixel_xy(rows_idx, cols_idx, transform):
    px = abs(transform.a)
    x0 = transform.c + px / 2
    y0 = transform.f - px / 2
    xs = x0 + cols_idx * px
    ys = y0 - rows_idx * px
    return np.column_stack([xs, ys])

def subsample_top_acc(rows_idx, cols_idx, acc, n_max):
    vals = acc[rows_idx, cols_idx]
    if len(vals) > n_max:
        top = np.argsort(vals)[::-1][:n_max]
        return rows_idx[top], cols_idx[top]
    return rows_idx, cols_idx

def build_rook_graph(rows_idx, cols_idx, dem, px_size):
    n = len(rows_idx)
    rc_idx = {(int(r), int(c)): i
              for i, (r, c) in enumerate(zip(rows_idx, cols_idx))}
    elev = np.array([dem[r, c] for r, c in zip(rows_idx, cols_idx)])
    sigma_e = max(float(np.nanmedian(np.abs(np.diff(elev)))), 0.1)

    W = np.zeros((n, n), dtype=float)
    edges = []
    for i, (r, c) in enumerate(zip(rows_idx, cols_idx)):
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            j = rc_idx.get((int(r)+dr, int(c)+dc))
            if j is not None and j > i:
                d_e = abs(float(dem[r,c]) - float(dem[r+dr,c+dc]))
                w   = math.exp(-d_e / (sigma_e + 1e-6))
                W[i,j] = max(W[i,j], w); W[j,i] = W[i,j]
                edges.append((i, j))

    L = np.diag(W.sum(1)) - W
    print(f'  Rook graph: {n} nodes  {len(edges)} edges  '
          f'(mean deg={2*len(edges)/n:.2f})')
    return L, W, edges


def subgraph(ri, ci, dem, transform, keep_mask):
    """Re-index a subset of nodes and rebuild the rook graph."""
    ri_k = ri[keep_mask]; ci_k = ci[keep_mask]
    L, W, edges = build_rook_graph(ri_k, ci_k, dem, abs(transform.a))
    xy = pixel_xy(ri_k, ci_k, transform)
    return ri_k, ci_k, xy, L, W, edges


def extract_lcc_indices(n: int, edges: list) -> np.ndarray:
    """Return node indices belonging to the largest connected component."""
    components = _find_components(n, edges)
    lcc = max(components, key=len)
    return lcc


# ══════════════════════════════════════════════════════════════════════════════
# 5. KERNELCAL
# ══════════════════════════════════════════════════════════════════════════════

def run_kernelcal(L, W, label='jezero'):
    n = L.shape[0]
    ev, _ = scipy_eigh(L)
    ev    = np.maximum(ev, 0.0)
    n_zero = int(np.sum(ev < 1e-6))
    w_modes = ev.copy()
    w_modes[:n_zero] = ev[n_zero] if n_zero < n else 1e-3

    h0     = np.maximum(np.exp(-ev), 1e-10)
    h_star, _ = fixed_point_kernel(L, h0=h0, mu2=MU2, sigma2=SIGMA2, w=w_modes)
    h_star = np.maximum(h_star, 1e-8)

    H_obs = spectral_entropy(h_star); H_vac = spectral_entropy(h0)
    dH    = H_obs - H_vac
    n_edges = int((W > 0).sum()) // 2
    beta0   = n_zero
    beta1   = max(0, n_edges - (n - beta0))
    lam_f   = float(ev[n_zero]) if n_zero < n else 0.0

    print(f'  [{label:12s}]  ΔH={dH:+.4f}  β₀={beta0}  β₁={beta1}  '
          f'Δβ₁/N={beta1/n:+.4f}  λ_f={lam_f:.5f}')
    return dict(ev=ev, h0=h0, h_star=h_star, dH=dH,
                beta0=beta0, beta1=beta1, db1_N=beta1/n,
                n=n, n_edges=n_edges, lam_f=lam_f, label=label)


def component_size_distribution(n: int, edges: list) -> dict:
    """Report size distribution of connected components."""
    components = _find_components(n, edges)
    sizes = sorted([len(c) for c in components], reverse=True)
    sizes_arr = np.array(sizes)
    n_singleton = int((sizes_arr == 1).sum())
    n_small     = int(((sizes_arr > 1) & (sizes_arr <= 10)).sum())
    n_medium    = int(((sizes_arr > 10) & (sizes_arr <= 50)).sum())
    n_large     = int((sizes_arr > 50).sum())
    lcc_size    = sizes_arr[0] if len(sizes_arr) > 0 else 0

    print(f'  Component distribution: β₀={len(components)}  '
          f'singleton={n_singleton}  2-10={n_small}  '
          f'11-50={n_medium}  >50={n_large}  LCC={lcc_size}')
    return dict(all=sizes, n_singleton=n_singleton, n_small=n_small,
                n_medium=n_medium, n_large=n_large, lcc_size=lcc_size,
                total=len(components))


# ══════════════════════════════════════════════════════════════════════════════
# 6. PARAMETER SWEEP
# ══════════════════════════════════════════════════════════════════════════════

def parameter_sweep(dem_filtered, transform,
                    acc_fracs=(0.001, 0.003, 0.005, 0.010),
                    n_vals=(1000, 2000, 3000, 5000)):
    """Sweep accumulation threshold and N to test β₁ and ΔH robustness."""
    print('\n  Parameter sweep: accumulation threshold × N …')
    fdir = d8_flow_dir(dem_filtered)
    acc  = flow_accum(fdir)

    rows = []
    for frac in acc_fracs:
        for n_max in n_vals:
            mask = channel_mask(acc, dem_filtered, thr_factor=frac)
            ri_s, ci_s = np.where(mask)
            ri_s, ci_s = subsample_top_acc(ri_s, ci_s, acc, n_max)
            n_actual = len(ri_s)
            if n_actual < 50:
                rows.append(dict(frac=frac, n_max=n_max, n_actual=n_actual,
                                 dH=np.nan, beta1=np.nan, db1_N=np.nan))
                continue
            L_s, W_s, edges_s = build_rook_graph(ri_s, ci_s, dem_filtered, abs(transform.a))
            gd = run_kernelcal(L_s, W_s, label=f't={frac:.3f}/N={n_max}')
            rows.append(dict(frac=frac, n_max=n_max, n_actual=n_actual,
                             dH=gd['dH'], beta1=gd['beta1'], db1_N=gd['db1_N']))
            print(f'    thr={frac:.3f}  N={n_max}  n={n_actual}  '
                  f'β₁={gd["beta1"]}  ΔH={gd["dH"]:+.4f}')
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# HELPER UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _hillshade(dem, px):
    dem_f = np.where(np.isfinite(dem), dem, np.nanmin(dem))
    dy, dx = np.gradient(dem_f, px, px)
    slope  = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dy, dx)
    hs = np.cos(math.radians(45))*np.cos(slope) + \
         np.sin(math.radians(45))*np.sin(slope)*np.cos(math.radians(315) - aspect)
    return np.clip(hs, 0, 1)

def _extent(dem, transform):
    px = abs(transform.a)
    r, c = dem.shape
    x0 = transform.c; y0 = transform.f - r * px
    return [x0, x0 + c*px, y0, y0 + r*px]


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def fig1_overview(dem, dem_filt, acc, mask_raw, mask_filt, transform):
    """Side-by-side: raw DEM channel mask vs. artifact-filtered mask."""
    hs  = _hillshade(dem, abs(transform.a))
    ext = _extent(dem, transform)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.subplots_adjust(wspace=0.06)

    axes[0].imshow(hs, cmap='gray', extent=ext, origin='upper', aspect='equal')
    im0 = axes[0].imshow(np.where(np.isfinite(dem), dem, np.nan),
                          cmap='gist_earth', extent=ext, origin='upper',
                          aspect='equal', alpha=0.55)
    plt.colorbar(im0, ax=axes[0], fraction=0.04, pad=0.02).set_label('Elev (m)')
    axes[0].set_title('(a) Hillshade + Elevation', fontweight='bold', pad=4)
    axes[0].set_xlabel('Easting (m)'); axes[0].set_ylabel('Northing (m)')

    # Raw mask — shows artifact chains
    axes[1].imshow(hs, cmap='gray', extent=ext, origin='upper', aspect='equal', alpha=0.65)
    rgba_raw = np.zeros((*mask_raw.shape, 4))
    rgba_raw[mask_raw] = [0.83, 0.30, 0.40, 0.80]   # red = includes artifacts
    axes[1].imshow(rgba_raw, extent=ext, origin='upper', aspect='equal')
    axes[1].set_title(f'(b) Channel mask (RAW)\n'
                      f'{mask_raw.sum():,} px — includes swath artifacts',
                      fontweight='bold', pad=4, color='#AA2222')
    axes[1].set_xlabel('Easting (m)'); axes[1].set_yticklabels([])

    # Filtered mask — artifact-clean
    axes[2].imshow(hs, cmap='gray', extent=ext, origin='upper', aspect='equal', alpha=0.65)
    rgba_filt = np.zeros((*mask_filt.shape, 4))
    rgba_filt[mask_filt] = [0.80, 0.47, 0.65, 0.85]  # pink = artifact-free
    axes[2].imshow(rgba_filt, extent=ext, origin='upper', aspect='equal')
    axes[2].set_title(f'(c) Channel mask (FILTERED)\n'
                      f'{mask_filt.sum():,} px — E-W median + chain removal',
                      fontweight='bold', pad=4, color='#663399')
    axes[2].set_xlabel('Easting (m)'); axes[2].set_yticklabels([])

    fig.suptitle('Jezero Crater — Swath Artifact Detection and Removal\n'
                 f'E-W median filter (width={EW_FILTER_WIDTH} px) + linear-chain removal',
                 fontsize=10.5, fontweight='bold', y=1.02)
    out = FIG_DIR / 'fig1_overview.png'
    fig.savefig(out); plt.close(fig)
    print(f'  Saved {out.name}')


def fig2_artifact_filter(dem, dem_raw, transform,
                         ri_raw, ci_raw, edges_raw,
                         ri_filt, ci_filt, edges_filt):
    """Before / after: rook graph highlighting removed artifact nodes."""
    hs  = _hillshade(dem, abs(transform.a))
    ext = _extent(dem, transform)
    xy_raw  = pixel_xy(ri_raw,  ci_raw,  transform)
    xy_filt = pixel_xy(ri_filt, ci_filt, transform)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.subplots_adjust(wspace=0.08)

    for ax, xy, edges, title, col in [
        (axes[0], xy_raw,  edges_raw,
         f'(a) Before filtering\n{len(xy_raw)} nodes  {len(edges_raw)} edges',
         '#DD3333'),
        (axes[1], xy_filt, edges_filt,
         f'(b) After artifact removal\n{len(xy_filt)} nodes  {len(edges_filt)} edges',
         C_JEZ),
    ]:
        ax.imshow(hs, cmap='gray', extent=ext, origin='upper', aspect='equal', alpha=0.65)
        if edges:
            segs = [[xy[i], xy[j]] for i, j in edges]
            lc = LineCollection(segs, linewidths=0.5, color=col, alpha=0.65, zorder=4)
            ax.add_collection(lc)
        ax.scatter(xy[:, 0], xy[:, 1], s=2, color=col, alpha=0.7, zorder=5)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_title(title, fontweight='bold', pad=4, color=col)
        ax.set_xlabel('Easting (m)')

    axes[0].set_ylabel('Northing (m)')
    axes[1].set_yticklabels([])

    n_removed = len(ri_raw) - len(ri_filt)
    fig.suptitle(f'Swath Artifact Removal — {n_removed:,} nodes eliminated\n'
                 f'Straight N-S chains (col_std < {ARTIFACT_COL_STD_MAX} px, '
                 f'size ≥ {ARTIFACT_MIN_SIZE}) removed',
                 fontsize=10, fontweight='bold', y=1.02)
    out = FIG_DIR / 'fig2_artifact_filter.png'
    fig.savefig(out); plt.close(fig)
    print(f'  Saved {out.name}')


def fig3_rook_graph(dem, transform, xy, edges, W, label='Jezero (filtered)'):
    hs  = _hillshade(dem, abs(transform.a))
    ext = _extent(dem, transform)
    deg = (W > 0).sum(1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.subplots_adjust(wspace=0.08)

    for ai, (ax, zoom) in enumerate(zip(axes, [False, True])):
        ax.imshow(hs, cmap='gray', extent=ext, origin='upper', aspect='equal', alpha=0.65)
        segs = [[xy[i], xy[j]] for i, j in edges]
        if segs:
            lc = LineCollection(segs, linewidths=0.6 if not zoom else 1.4,
                                color=C_JEZ, alpha=0.7, zorder=4)
            ax.add_collection(lc)
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=deg, cmap='YlOrRd',
                        s=4 if not zoom else 16, vmin=1, vmax=4,
                        edgecolors='none', zorder=5)
        if zoom:
            cx, cy = xy[:, 0].mean(), xy[:, 1].mean()
            half = 5000
            ax.set_xlim(cx-half, cx+half); ax.set_ylim(cy-half, cy+half)
            ax.set_title('(b) 10 km zoom — confluences visible',
                         fontweight='bold', pad=4, color=C_JEZ)
        else:
            ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
            ax.set_title(f'(a) Rook graph — {label}\n'
                         f'{len(xy)} nodes  {len(edges)} edges',
                         fontweight='bold', pad=4, color=C_JEZ)
        ax.set_xlabel('Easting (m)')
        if ai == 0: ax.set_ylabel('Northing (m)')
        else:       ax.set_yticklabels([])

    plt.colorbar(sc, ax=axes[-1], fraction=0.035, pad=0.02).set_label('Node degree')
    fig.suptitle('Jezero Channel Network — Rook Adjacency Graph (Artifact-Cleaned)\n'
                 'Edges exist only where channel pixels share a physical boundary',
                 fontsize=10, fontweight='bold', y=1.02)
    out = FIG_DIR / 'fig3_rook_graph.png'
    fig.savefig(out); plt.close(fig)
    print(f'  Saved {out.name}')


def fig4_junctions(dem, transform, xy, W, edges):
    hs  = _hillshade(dem, abs(transform.a))
    ext = _extent(dem, transform)
    deg = (W > 0).sum(1)
    junc = deg >= 3
    jxy  = xy[junc]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.subplots_adjust(wspace=0.08)

    if len(jxy) >= 3:
        from scipy.spatial.distance import cdist
        sel = [0]
        for _ in range(2):
            dists = cdist(jxy[sel], jxy).min(0)
            sel.append(int(np.argmax(dists)))
        centres = [jxy[s] for s in sel]
    else:
        centres = [xy.mean(0)] * 3

    half = 3000

    for ai, (ax, ctr) in enumerate(zip(axes, centres)):
        ax.imshow(hs, cmap='gray', extent=ext, origin='upper', aspect='equal', alpha=0.7)
        inbox = ((xy[:,0] >= ctr[0]-half) & (xy[:,0] <= ctr[0]+half) &
                 (xy[:,1] >= ctr[1]-half) & (xy[:,1] <= ctr[1]+half))
        box_s = set(np.where(inbox)[0])
        segs  = [[xy[i],xy[j]] for i,j in edges if i in box_s and j in box_s]
        if segs:
            lc = LineCollection(segs, linewidths=1.4, color=C_JEZ, alpha=0.85, zorder=4)
            ax.add_collection(lc)
        sc = ax.scatter(xy[inbox,0], xy[inbox,1], c=deg[inbox],
                        cmap='YlOrRd', s=22, vmin=1, vmax=4,
                        edgecolors='black', linewidths=0.3, zorder=5)
        junc_in = inbox & junc
        if junc_in.any():
            ax.scatter(xy[junc_in,0], xy[junc_in,1],
                       s=80, color='gold', marker='*',
                       edgecolors='black', linewidths=0.5, zorder=6,
                       label='Junction (degree ≥ 3)')
        ax.set_xlim(ctr[0]-half, ctr[0]+half)
        ax.set_ylim(ctr[1]-half, ctr[1]+half)
        ax.set_title(f'({chr(97+ai)})  6 km × 6 km patch\n★ = physical junction',
                     fontweight='bold', pad=4, fontsize=9)
        ax.set_xlabel('Easting (m)')
        if ai == 0: ax.set_ylabel('Northing (m)')
        else:       ax.set_yticklabels([])

    plt.colorbar(sc, ax=axes[-1], fraction=0.04, pad=0.02).set_label('Node degree')
    n_junc = int(junc.sum())
    fig.suptitle(f'Jezero — Physical Confluences (artifact-cleaned rook graph)\n'
                 f'{n_junc} junction nodes (degree ≥ 3) — each is a real channel bifurcation',
                 fontsize=10, fontweight='bold', y=1.02)
    out = FIG_DIR / 'fig4_junction_loops.png'
    fig.savefig(out); plt.close(fig)
    print(f'  Saved {out.name}')


def fig5_spectra(gd_full, gd_lcc):
    """Eigenspectrum and kernel comparison: full filtered graph vs. LCC."""
    AZ_DH = -0.0265

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.subplots_adjust(hspace=0.45, wspace=0.35)

    for row, (gd, tag, col) in enumerate([
        (gd_full, 'Full filtered graph', C_JEZ),
        (gd_lcc,  'Largest connected component (LCC)', '#8B0000'),
    ]):
        ev = np.sort(gd['ev'])
        n_sh = min(80, len(ev))

        ax = axes[row, 0]
        ax.bar(range(n_sh), ev[:n_sh], color=col, width=1.0,
               edgecolor='none', alpha=0.85)
        ax.set_xlabel('Mode $l$'); ax.set_ylabel('$\\lambda_l$')
        ax.set_title(f'Eigenspectrum — {tag}', fontweight='bold', pad=4)
        ax.text(0.97, 0.97,
                f'$N={gd["n"]}$\n'
                f'$\\beta_0 = {gd["beta0"]}$\n'
                f'$\\beta_1 = {gd["beta1"]}$\n'
                f'$\\Delta H = {gd["dH"]:+.3f}$',
                transform=ax.transAxes, ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc'))

        ax2 = axes[row, 1]
        sort = np.argsort(gd['ev'])
        lam  = gd['ev'][sort][:n_sh]
        h0s  = gd['h0'][sort][:n_sh]
        hss  = gd['h_star'][sort][:n_sh]
        ax2.fill_between(lam, h0s, hss, where=(hss >= h0s),
                         color='#009E73', alpha=0.25, label='amplified')
        ax2.fill_between(lam, h0s, hss, where=(hss < h0s),
                         color='#D55E00', alpha=0.25, label='suppressed')
        ax2.plot(lam, h0s, '--', color='#888888', lw=1.1, label='$h_0$ vacuum')
        ax2.plot(lam, hss,  '-', color=col, lw=2.0, label='$h^*$')
        ax2.set_xlabel('$\\lambda_l$'); ax2.set_ylabel('$h(\\lambda)$')
        ax2.set_title(f'Fixed-point kernel — {tag}', fontweight='bold', pad=4)
        ax2.text(0.05, 0.06,
                 f'$\\Delta H = {gd["dH"]:+.3f}$ nats\n'
                 f'(AZ null: $\\Delta H = {AZ_DH:+.3f}$)',
                 transform=ax2.transAxes, va='bottom', fontsize=9, color=col,
                 fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc'))
        ax2.legend(fontsize=7, loc='upper right')

    fig.suptitle('Jezero — Eigenspectra and Fixed-point Kernels\n'
                 'Full artifact-filtered graph vs. Largest Connected Component',
                 fontsize=11, fontweight='bold')
    out = FIG_DIR / 'fig5_spectra.png'
    fig.savefig(out); plt.close(fig)
    print(f'  Saved {out.name}')


def fig6_component_dist(comp_dist):
    """Bar chart of connected-component size distribution."""
    sizes = np.array(comp_dist['all'])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.subplots_adjust(wspace=0.35)

    # Full distribution (log scale)
    ax = axes[0]
    bins = np.logspace(0, np.log10(max(sizes)+1), 30)
    ax.hist(sizes, bins=bins, color=C_JEZ, edgecolor='white', linewidth=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('Component size (nodes)'); ax.set_ylabel('Count')
    ax.set_title('(a) Component size distribution\n(log-log scale)',
                 fontweight='bold', pad=4)
    ax.text(0.97, 0.97,
            f'Total β₀ = {comp_dist["total"]}\n'
            f'Singletons: {comp_dist["n_singleton"]}\n'
            f'Size 2–10:  {comp_dist["n_small"]}\n'
            f'Size 11–50: {comp_dist["n_medium"]}\n'
            f'Size > 50:  {comp_dist["n_large"]}\n'
            f'LCC size:   {comp_dist["lcc_size"]}',
            transform=ax.transAxes, ha='right', va='top', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc'))

    # Pie breakdown by category
    ax2 = axes[1]
    labels  = ['Singletons\n(threshold noise)',
               'Size 2–10\n(small fragments)',
               'Size 11–50\n(medium clusters)',
               'Size > 50\n(substantive)']
    vals = [comp_dist['n_singleton'], comp_dist['n_small'],
            comp_dist['n_medium'],    comp_dist['n_large']]
    colors = ['#DDDDDD', '#AAAAAA', C_JEZ, '#6600CC']
    vals_clean = [max(v, 0) for v in vals]
    wedges, texts, autotexts = ax2.pie(
        vals_clean, labels=labels, colors=colors,
        autopct='%1.0f%%', startangle=90,
        textprops={'fontsize': 8},
        wedgeprops={'edgecolor': 'white', 'linewidth': 1})
    ax2.set_title('(b) β₀ breakdown by component size\n'
                  '(most isolated components are threshold noise)',
                  fontweight='bold', pad=4)

    fig.suptitle('Jezero — Connected Component Size Distribution\n'
                 'Most β₀ components are single-pixel threshold noise, '
                 'not independent drainage basins',
                 fontsize=10, fontweight='bold', y=1.02)
    out = FIG_DIR / 'fig6_component_dist.png'
    fig.savefig(out); plt.close(fig)
    print(f'  Saved {out.name}')


def fig7_parameter_sweep(sweep_rows, gd_full, gd_lcc):
    """β₁ and ΔH heatmaps over accumulation threshold × N."""
    import pandas as pd  # only used for pivoting

    fracs = sorted(set(r['frac'] for r in sweep_rows))
    ns    = sorted(set(r['n_max'] for r in sweep_rows))

    def make_mat(key):
        mat = np.full((len(fracs), len(ns)), np.nan)
        for r in sweep_rows:
            fi = fracs.index(r['frac'])
            ni = ns.index(r['n_max'])
            mat[fi, ni] = r[key]
        return mat

    mat_b1  = make_mat('beta1')
    mat_dH  = make_mat('dH')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.subplots_adjust(wspace=0.4)

    # β₁ heatmap
    ax = axes[0]
    im = ax.imshow(mat_b1, aspect='auto', cmap='YlOrRd',
                   vmin=0, vmax=np.nanmax(mat_b1) + 1)
    ax.set_xticks(range(len(ns)));  ax.set_xticklabels([str(n) for n in ns])
    ax.set_yticks(range(len(fracs))); ax.set_yticklabels([f'{f:.3f}' for f in fracs])
    ax.set_xlabel('N (channel nodes)'); ax.set_ylabel('Acc. threshold fraction')
    ax.set_title('(a)  β₁ across parameter space\n(real confluences)',
                 fontweight='bold', pad=4)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label('β₁')
    for fi in range(len(fracs)):
        for ni in range(len(ns)):
            v = mat_b1[fi, ni]
            if not np.isnan(v):
                ax.text(ni, fi, f'{int(v)}', ha='center', va='center',
                        fontsize=8.5, fontweight='bold', color='white' if v > 5 else 'black')

    # ΔH heatmap
    ax2 = axes[1]
    vmin = min(np.nanmin(mat_dH), -0.35)
    im2 = ax2.imshow(mat_dH, aspect='auto', cmap='Blues_r',
                     vmin=vmin, vmax=0)
    ax2.set_xticks(range(len(ns)));  ax2.set_xticklabels([str(n) for n in ns])
    ax2.set_yticks(range(len(fracs))); ax2.set_yticklabels([f'{f:.3f}' for f in fracs])
    ax2.set_xlabel('N (channel nodes)'); ax2.set_ylabel('Acc. threshold fraction')
    ax2.set_title('(b)  ΔH across parameter space\n(spectral concentration)',
                  fontweight='bold', pad=4)
    plt.colorbar(im2, ax=ax2, fraction=0.04, pad=0.02).set_label('ΔH (nats)')
    for fi in range(len(fracs)):
        for ni in range(len(ns)):
            v = mat_dH[fi, ni]
            if not np.isnan(v):
                ax2.text(ni, fi, f'{v:+.2f}', ha='center', va='center',
                         fontsize=7.5, fontweight='bold',
                         color='white' if v < -0.15 else 'black')

    # Add reference lines: AZ Plateau β₁ = 3
    axes[0].axhline(-0.5, color='white', lw=0)  # dummy for spacing
    fig.suptitle('Jezero Parameter Sensitivity: Accumulation Threshold × Channel Node Count\n'
                 f'Default config: thr={ACC_THR_FACTOR:.4f}, N={N_MAX}  '
                 f'→ β₁={gd_full["beta1"]}  ΔH={gd_full["dH"]:+.3f}  '
                 f'(LCC: β₁={gd_lcc["beta1"]}  ΔH={gd_lcc["dH"]:+.3f})',
                 fontsize=9.5, fontweight='bold', y=1.04)
    out = FIG_DIR / 'fig7_parameter_sweep.png'
    fig.savefig(out); plt.close(fig)
    print(f'  Saved {out.name}')


def fig8_phase_space(gd_full, gd_lcc):
    """Phase space with full, LCC, AZ, and cities."""
    AZ_ROOK = dict(dH=-0.0265, db1_N=0.001)
    CITIES  = [
        dict(name='Barcelona', dH=-0.339, db1_N=0.608),
        dict(name='Phoenix',   dH=-0.285, db1_N=0.482),
        dict(name='Venice',    dH=-0.237, db1_N=0.190),
        dict(name='Marrakech', dH=-0.238, db1_N=0.247),
        dict(name='Houston',   dH=-0.301, db1_N=0.570),
    ]
    city_colors = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#D55E00']

    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    ax.axhspan(-0.02,  0.05, color='#EFF7FF', alpha=0.5, zorder=0)
    ax.axhspan( 0.05,  0.40, color='#FFFAEC', alpha=0.5, zorder=0)
    ax.axhspan( 0.40,  2.00, color='#EFFFEF', alpha=0.5, zorder=0)
    ax.text(-0.58, -0.010, 'Abiotic tier',      fontsize=8, color='#336699', style='italic')
    ax.text(-0.58,  0.06,  'Fossil/weak ctrl.',  fontsize=8, color='#888800', style='italic')
    ax.text(-0.58,  0.42,  'Active ctrl.',        fontsize=8, color='#006633', style='italic')

    # AZ Plateau
    ax.scatter(AZ_ROOK['dH'], AZ_ROOK['db1_N'], s=140,
               color=C_AZ, marker='o', edgecolors='black', linewidths=1.0, zorder=7)
    ax.annotate('AZ Plateau (rook)',
                (AZ_ROOK['dH'], AZ_ROOK['db1_N']),
                xytext=(-8, 16), textcoords='offset points',
                fontsize=8.5, color=C_AZ, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C_AZ, lw=1.0))

    # Jezero — full filtered graph
    ax.scatter(gd_full['dH'], gd_full['db1_N'], s=200,
               color=C_JEZ, marker='*', edgecolors='black', linewidths=0.9, zorder=8)
    ax.annotate(f'Jezero (filtered)\nΔH={gd_full["dH"]:+.3f}  β₁={gd_full["beta1"]}',
                (gd_full['dH'], gd_full['db1_N']),
                xytext=(12, -20), textcoords='offset points',
                fontsize=8.5, color=C_JEZ, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C_JEZ, lw=1.0))

    # Jezero LCC
    ax.scatter(gd_lcc['dH'], gd_lcc['db1_N'], s=160,
               color='#8B0000', marker='*', edgecolors='black', linewidths=0.9, zorder=8,
               alpha=0.85)
    ax.annotate(f'Jezero LCC (N={gd_lcc["n"]})\nΔH={gd_lcc["dH"]:+.3f}',
                (gd_lcc['dH'], gd_lcc['db1_N']),
                xytext=(-55, 16), textcoords='offset points',
                fontsize=8, color='#8B0000',
                arrowprops=dict(arrowstyle='->', color='#8B0000', lw=0.9))

    # Cities
    for c, col in zip(CITIES, city_colors):
        ax.scatter(c['dH'], c['db1_N'], s=70, color=col,
                   marker='D', edgecolors='black', linewidths=0.7, zorder=6)
        ax.annotate(c['name'], (c['dH'], c['db1_N']),
                    xytext=(5, 3), textcoords='offset points',
                    fontsize=7.5, color=col)

    ax.axhline(0, color='#cccccc', lw=0.7, ls=':')
    ax.axvline(0, color='#cccccc', lw=0.7, ls=':')

    ax.set_xlim(-0.62, 0.06); ax.set_ylim(-0.025, 1.4)
    ax.set_xlabel(r'$\Delta H = H[h^*] - H[h_0]$  (nats)', fontsize=10)
    ax.set_ylabel(r'$\Delta\beta_1 / N$  (topological excess per node)', fontsize=10)
    ax.set_title('Phase Space — Artifact-Cleaned Jezero vs Abiotic Null vs Cities\n'
                 'Full filtered graph (★ pink) and LCC only (★ dark red)',
                 fontweight='bold', pad=6, fontsize=10.5)

    leg = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor=C_AZ,
               markeredgecolor='black', markersize=10, label='AZ Plateau (rook)'),
        Line2D([0],[0], marker='*', color='w', markerfacecolor=C_JEZ,
               markeredgecolor='black', markersize=13, label='Jezero full (filtered)'),
        Line2D([0],[0], marker='*', color='w', markerfacecolor='#8B0000',
               markeredgecolor='black', markersize=13, label='Jezero LCC only'),
        Line2D([0],[0], marker='D', color='w', markerfacecolor='#888888',
               markeredgecolor='black', markersize=8,  label='Cities (OSM street network)'),
    ]
    ax.legend(handles=leg, fontsize=8.5, loc='upper right', framealpha=0.97)

    out = FIG_DIR / 'fig8_phase_space.png'
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f'  Saved {out.name}')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print('=' * 70)
    print('Jezero Crater — rook-adjacency kernelcal WITH ARTIFACT FILTERING')
    print('=' * 70)

    # ── 1. Load DEM ────────────────────────────────────────────────────────
    print('\n[1] Load DEM …')
    dem, transform = load_dem()
    px = abs(transform.a)

    # ── 2. Pre-filter swath artifacts ──────────────────────────────────────
    print('\n[2] Pre-filter swath artifacts (E-W median) …')
    dem_filt = prefilter_swath_artifacts(dem, ew_width=EW_FILTER_WIDTH)

    # ── 3. D8 on raw (for comparison) and filtered DEM ────────────────────
    print('\n[3] D8 flow accumulation (filtered DEM) …')
    fdir_filt = d8_flow_dir(dem_filt)
    acc_filt  = flow_accum(fdir_filt)

    print('    D8 flow accumulation (raw DEM, for comparison) …')
    fdir_raw  = d8_flow_dir(dem)
    acc_raw   = flow_accum(fdir_raw)

    mask_raw  = channel_mask(acc_raw,  dem,      ACC_THR_FACTOR)
    mask_filt = channel_mask(acc_filt, dem_filt, ACC_THR_FACTOR)
    print(f'  Raw channel pixels:      {mask_raw.sum():,}')
    print(f'  Filtered channel pixels: {mask_filt.sum():,}')

    # ── 4. Extract nodes from filtered DEM ────────────────────────────────
    print('\n[4] Extract channel nodes (filtered DEM) …')
    ri_filt_all, ci_filt_all = np.where(mask_filt)
    ri_filt_all, ci_filt_all = subsample_top_acc(ri_filt_all, ci_filt_all, acc_filt, N_MAX)

    # Also build raw graph for comparison figure
    ri_raw_all, ci_raw_all = np.where(mask_raw)
    ri_raw_all, ci_raw_all = subsample_top_acc(ri_raw_all, ci_raw_all, acc_raw, N_MAX)

    # ── 5. Build rook graphs ───────────────────────────────────────────────
    print('\n[5] Build rook graphs …')
    print('    Raw nodes:')
    L_raw, W_raw, edges_raw = build_rook_graph(ri_raw_all, ci_raw_all, dem, px)
    xy_raw = pixel_xy(ri_raw_all, ci_raw_all, transform)

    print('    Filtered-DEM nodes (before chain removal):')
    L_pre, W_pre, edges_pre = build_rook_graph(ri_filt_all, ci_filt_all, dem_filt, px)

    # ── 6. Post-hoc linear chain removal ──────────────────────────────────
    print('\n[6] Post-hoc linear-chain removal …')
    keep = filter_linear_chains(ri_filt_all, ci_filt_all, edges_pre)
    ri_clean = ri_filt_all[keep]
    ci_clean = ci_filt_all[keep]

    print('    Rebuild rook graph after chain removal:')
    L_clean, W_clean, edges_clean = build_rook_graph(ri_clean, ci_clean, dem_filt, px)
    xy_clean = pixel_xy(ri_clean, ci_clean, transform)

    # ── 7. kernelcal on full cleaned graph ────────────────────────────────
    print('\n[7] kernelcal on full artifact-cleaned graph …')
    gd_full = run_kernelcal(L_clean, W_clean, label='Jezero-clean')

    # ── 8. Component size distribution ────────────────────────────────────
    print('\n[8] Component size distribution …')
    comp_dist = component_size_distribution(len(ri_clean), edges_clean)

    # ── 9. LCC extraction and kernelcal ───────────────────────────────────
    print('\n[9] Largest connected component …')
    lcc_idx  = extract_lcc_indices(len(ri_clean), edges_clean)
    ri_lcc   = ri_clean[lcc_idx]
    ci_lcc   = ci_clean[lcc_idx]
    xy_lcc   = pixel_xy(ri_lcc, ci_lcc, transform)
    print(f'  LCC size: {len(lcc_idx)} nodes (of {len(ri_clean)} total)')
    L_lcc, W_lcc, edges_lcc = build_rook_graph(ri_lcc, ci_lcc, dem_filt, px)
    gd_lcc = run_kernelcal(L_lcc, W_lcc, label='Jezero-LCC')

    # ── 10. Summary table ─────────────────────────────────────────────────
    print('\n  ── Jezero summary table ────────────────────────────────────────')
    print(f'  {"System":30s}  {"N":>6}  {"ΔH":>8}  {"β₀":>6}  {"β₁":>5}  {"Δβ₁/N":>8}')
    print('  ' + '─' * 68)
    print(f'  {"AZ Plateau (rook, abiotic)":30s}  {3000:6d}  {-0.0265:+8.4f}  '
          f'{2504:6d}  {3:5d}  {0.001:+8.4f}')
    print(f'  {"Jezero full (raw DEM, prev)":30s}  {3000:6d}  {-0.263:+8.4f}  '
          f'{952:6d}  {7:5d}  {0.0023:+8.4f}')
    for gd, tag in [(gd_full, 'Jezero full (filtered)'),
                    (gd_lcc,  'Jezero LCC only')]:
        print(f'  {tag:30s}  {gd["n"]:6d}  {gd["dH"]:+8.4f}  '
              f'{gd["beta0"]:6d}  {gd["beta1"]:5d}  {gd["db1_N"]:+8.4f}')
    print()

    # ── 11. Parameter sweep ───────────────────────────────────────────────
    print('\n[11] Parameter sweep (4 thresholds × 4 N-values) …')
    sweep = parameter_sweep(dem_filt, transform)

    b1_vals = [r['beta1'] for r in sweep if not np.isnan(r.get('beta1', np.nan))]
    dH_vals = [r['dH']    for r in sweep if not np.isnan(r.get('dH', np.nan))]
    if b1_vals:
        print(f'\n  β₁ range across sweep: {int(min(b1_vals))} – {int(max(b1_vals))}')
        print(f'  ΔH range across sweep:  {min(dH_vals):+.3f} – {max(dH_vals):+.3f}')
        print(f'  AZ Plateau β₁ = 3, ΔH = −0.027  (for comparison)')

    # ── 12. Figures ───────────────────────────────────────────────────────
    print('\n[12] Figures …')
    fig1_overview(dem, dem_filt, acc_raw, mask_raw, mask_filt, transform)
    fig2_artifact_filter(dem, dem_filt, transform,
                         ri_raw_all, ci_raw_all, edges_raw,
                         ri_clean, ci_clean, edges_clean)
    fig3_rook_graph(dem_filt, transform, xy_clean, edges_clean, W_clean)
    fig4_junctions(dem_filt, transform, xy_clean, W_clean, edges_clean)
    fig5_spectra(gd_full, gd_lcc)
    fig6_component_dist(comp_dist)
    fig7_parameter_sweep(sweep, gd_full, gd_lcc)
    fig8_phase_space(gd_full, gd_lcc)
    print(f'\nAll figures → {FIG_DIR}')

    return gd_full, gd_lcc, sweep, comp_dist


if __name__ == '__main__':
    main()
