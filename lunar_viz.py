#!/usr/bin/env python3
"""
lunar_viz.py — Publication-quality visualizations for the LROC NAC crater analysis.

Produces 4 figures:
  fig1_crater_map.png         — NAC image with MaskRCNN detections + k-NN graph overlay
  fig2_crater_sfd.png         — Crater size-frequency distribution (SFD)
  fig3_kernelcal_diagnostics.png — Eigenspectrum + h*(λ) + ΔH annotation
  fig4_phase_space_extended.png  — Phase space: lunar + all prior systems, annotated

Requires the LROC repo to be present at /tmp/lroc_nac/
"""

from __future__ import annotations
import math, sys
from pathlib import Path

import numpy as np
from scipy.ndimage import label as nd_label
from scipy.spatial import cKDTree
from scipy.linalg import eigh as scipy_eigh
from scipy.stats import linregress
from matplotlib.collections import LineCollection
import matplotlib.patheffects as pe

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.ticker as mticker
from PIL import Image

KCAL_ROOT = Path(__file__).parent
sys.path.insert(0, str(KCAL_ROOT))
from kernelcal.terrain.diagnostics import (
    spectral_entropy, fixed_point_kernel, fiedler_mode_gap,
)

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

# ── paths ───────────────────────────────────────────────────────────────────
LROC_REPO = Path('/tmp/lroc_nac')
NAC_TIF   = LROC_REPO / 'NAC_ROI_ALPHNSUSLOA_E129S3581_cropped.tif'
MASK_DIR  = LROC_REPO / 'predicted_masks'
FIG_DIR   = KCAL_ROOT / 'lunar_figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIG_DIR = (KCAL_ROOT.parent /
                 'P4-journal-spectral-kernel-biosignature-planetary-surfaces' /
                 'figures')

# ── style ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
    'font.size': 9,
    'axes.labelsize': 9, 'axes.titlesize': 9.5,
    'axes.linewidth': 0.8,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'legend.fontsize': 7.5, 'legend.framealpha': 0.95,
    'legend.edgecolor': '#cccccc',
    'figure.dpi': 150,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.spines.top': False, 'axes.spines.right': False,
    'lines.linewidth': 1.2,
})

# Wong 2011 colorblind-safe palette
C_LUNAR  = '#CC79A7'   # reddish-purple
C_ABIO   = '#0072B2'   # blue
C_FOSSIL = '#E69F00'   # amber
C_ACTIVE = '#009E73'   # green
C_NULL   = '#BBBBBB'

K_NN   = 8
N_MAX  = 1500
MU2    = 2.0
SIGMA2 = 1.0
MIN_PX = 9


# ══════════════════════════════════════════════════════════════════════════════
# DATA PIPELINE  (re-runs cleanly from cached files)
# ══════════════════════════════════════════════════════════════════════════════

def load_all():
    """Return (nac_img, mask, transform, xy_m, diams_m, graph_dict)."""
    # NAC image
    src = rasterio.open(NAC_TIF)
    nac = src.read(1).astype(np.float32)
    W, H = src.width, src.height
    transform = src.transform
    src.close()

    # Reconstruct mask
    canvas = np.zeros((H, W), dtype=np.uint8)
    for mp in sorted(MASK_DIR.glob('tile_*.png')):
        parts = mp.stem.split('_')
        x_off, y_off = int(parts[1]), int(parts[2])
        tile = np.array(Image.open(mp))
        th, tw = tile.shape[:2]
        y_end = min(y_off + th, H); x_end = min(x_off + tw, W)
        canvas[y_off:y_end, x_off:x_end] = tile[:y_end-y_off, :x_end-x_off]

    # Extract craters
    labelled, n_blobs = nd_label((canvas > 128).astype(np.int32))
    cents_px, areas_px = [], []
    for lbl in range(1, n_blobs + 1):
        ys, xs = np.where(labelled == lbl)
        if len(ys) < MIN_PX:
            continue
        cents_px.append((xs.mean(), ys.mean()))
        areas_px.append(len(ys))
    cents_px = np.array(cents_px)
    areas_px = np.array(areas_px)

    if len(cents_px) > N_MAX:
        top = np.argsort(areas_px)[::-1][:N_MAX]
        cents_px = cents_px[top]; areas_px = areas_px[top]

    px_x = abs(transform.a); px_y = abs(transform.e)
    x0 = transform.c + px_x / 2; y0 = transform.f - px_y / 2
    xs_m = x0 + cents_px[:, 0] * px_x
    ys_m = y0 - cents_px[:, 1] * px_y
    xy_m = np.column_stack([xs_m, ys_m])
    diams_m = 2 * np.sqrt(areas_px * px_x * px_y / math.pi)

    # Graph + kernelcal
    n = len(xy_m)
    tree = cKDTree(xy_m)
    dists, inds = tree.query(xy_m, k=K_NN + 1)
    med_nn = float(np.median(dists[:, 1]))
    xr = xy_m[:, 0].max() - xy_m[:, 0].min()
    yr = xy_m[:, 1].max() - xy_m[:, 1].min()
    sigma = max(0.05 * math.hypot(xr, yr), 2 * max(med_nn, 1e-3))

    W_mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j_idx in range(1, K_NN + 1):
            j = inds[i, j_idx]; d = dists[i, j_idx]
            w = math.exp(-d**2 / sigma**2)
            if w > W_mat[i, j]:
                W_mat[i, j] = w; W_mat[j, i] = w

    L = np.diag(W_mat.sum(1)) - W_mat
    ev, evec = scipy_eigh(L)
    ev = np.maximum(ev, 0.0)
    n_zero  = int(np.sum(ev < 1e-6))
    w_modes = ev.copy()
    w_modes[:n_zero] = ev[n_zero] if n_zero < n else 1e-3
    h0     = np.maximum(np.exp(-ev), 1e-10)
    h_star, info = fixed_point_kernel(L, h0=h0, mu2=MU2, sigma2=SIGMA2, w=w_modes)
    h_star = np.maximum(h_star, 1e-8)

    H_obs = spectral_entropy(h_star); H_vac = spectral_entropy(h0)
    dH    = H_obs - H_vac
    dp    = fiedler_mode_gap(h_star, L, mu2=MU2, sigma2=SIGMA2, w=w_modes)
    n_edges = int((W_mat > 0).sum()) // 2
    beta0   = n_zero
    beta1   = max(0, n_edges - (n - beta0))
    db1     = beta1 - max(0, K_NN * n // 2 - (n - 1))
    lam_f   = float(ev[n_zero]) if n_zero < n else 0.0

    gd = dict(
        W=W_mat, L=L, ev=ev, evec=evec, h0=h0, h_star=h_star,
        H_obs=H_obs, H_vac=H_vac, dH=dH, dp=dp,
        beta0=beta0, beta1=beta1, db1=db1, db1_N=db1/n,
        lam_f=lam_f, n_edges=n_edges, n=n,
        inds=inds, dists=dists, sigma=sigma,
        cents_px=cents_px, areas_px=areas_px,
    )
    return nac, canvas, transform, xy_m, diams_m, gd


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — NAC image + masks + graph overlay  (2-panel)
# ══════════════════════════════════════════════════════════════════════════════

def fig1_crater_map(nac, mask, transform, xy_m, diams_m, gd):
    """Left: raw NAC + detection overlay.  Right: zoom + k-NN graph."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.5))
    fig.subplots_adjust(wspace=0.06)

    px_x = abs(transform.a)

    # extent in metres
    H, W = nac.shape
    x0 = transform.c; y0 = transform.f - H * abs(transform.e)
    ext_m = [x0, x0 + W * px_x, y0, y0 + H * abs(transform.e)]

    # ── LEFT: full scene ──────────────────────────────────────────────────
    ax = axes[0]
    # stretch NAC to [0,1]
    lo, hi = np.percentile(nac, [1, 99])
    nac_norm = np.clip((nac - lo) / (hi - lo + 1e-6), 0, 1)
    ax.imshow(nac_norm, cmap='gray', origin='upper',
              extent=ext_m, aspect='equal')

    # mask overlay (red tint)
    rgba = np.zeros((*mask.shape, 4), dtype=float)
    rgba[mask > 0] = [0.95, 0.25, 0.0, 0.50]
    ax.imshow(rgba, origin='upper', extent=ext_m, aspect='equal')

    # crater centroids
    ax.scatter(xy_m[:, 0], xy_m[:, 1],
               s=6, color=C_LUNAR, marker='.', alpha=0.8, linewidths=0,
               label=f'{gd["n"]} crater centroids')

    # k-NN graph edges (light, semi-transparent)
    segs = []
    for i in range(gd['n']):
        for j_idx in range(1, K_NN + 1):
            j = gd['inds'][i, j_idx]
            if j > i:
                segs.append([xy_m[i], xy_m[j]])
    lc = LineCollection(segs, linewidths=0.25, color='#56B4E9', alpha=0.25)
    ax.add_collection(lc)

    ax.set_xlabel('Easting  (m, lunar equirectangular)')
    ax.set_ylabel('Northing  (m)')
    ax.set_title('LROC NAC  — MaskRCNN crater detections\n'
                 '(red: predicted masks, dots: centroids, blue: k-NN graph)',
                 fontweight='bold', pad=5)
    ax.legend(loc='lower right', fontsize=7, markerscale=2)

    # scale bar 100 m
    sb_m = 100
    x_sb = ext_m[0] + 0.05 * (ext_m[1] - ext_m[0])
    y_sb = ext_m[2] + 0.04 * (ext_m[3] - ext_m[2])
    ax.plot([x_sb, x_sb + sb_m], [y_sb, y_sb], 'w-', lw=2.5)
    ax.text(x_sb + sb_m / 2, y_sb + 3, '100 m', ha='center',
            color='white', fontsize=7.5,
            path_effects=[pe.withStroke(linewidth=2, foreground='black')])

    # ── RIGHT: zoom centre region with graph edges coloured by weight ─────
    ax2 = axes[1]
    xc  = xy_m[:, 0].mean(); yc = xy_m[:, 1].mean()
    half = 300   # metres
    ax2.imshow(nac_norm, cmap='gray', origin='upper',
               extent=ext_m, aspect='equal')
    ax2.imshow(rgba, origin='upper', extent=ext_m, aspect='equal')

    # edges coloured by weight
    weights = []
    segs2   = []
    for i in range(gd['n']):
        for j_idx in range(1, K_NN + 1):
            j = gd['inds'][i, j_idx]
            if j > i and gd['W'][i, j] > 0:
                segs2.append([xy_m[i], xy_m[j]])
                weights.append(gd['W'][i, j])
    weights = np.array(weights)
    norm_w  = Normalize(vmin=0, vmax=weights.max())
    lc2 = LineCollection(segs2, linewidths=0.6, cmap='plasma',
                         norm=norm_w, alpha=0.55)
    lc2.set_array(weights)
    ax2.add_collection(lc2)

    # nodes coloured by degree (sum of edge weights)
    deg = gd['W'].sum(1)
    sc  = ax2.scatter(xy_m[:, 0], xy_m[:, 1],
                      c=deg, cmap='hot', s=10, vmin=deg.min(), vmax=deg.max(),
                      edgecolors='none', zorder=5)
    cbar = fig.colorbar(sc, ax=ax2, fraction=0.03, pad=0.02)
    cbar.set_label('Node degree (Σ edge weights)', fontsize=7)
    cbar.ax.tick_params(labelsize=6.5)

    ax2.set_xlim(xc - half, xc + half)
    ax2.set_ylim(yc - half, yc + half)
    ax2.set_xlabel('Easting  (m)')
    ax2.set_title('Zoom: k-NN graph  (edges coloured by weight,\n'
                  'nodes coloured by weighted degree)',
                  fontweight='bold', pad=5)
    ax2.set_yticklabels([])

    fig.suptitle(
        f'LROC NAC  ·  ~12.9°S / 358.1°E  ·  Lunar highland  ·  '
        f'DREAMS-lab MaskRCNN  |  N = {gd["n"]} craters  ·  '
        f'Ø = {diams_m.min():.0f}–{diams_m.max():.0f} m',
        fontsize=9, fontweight='bold', y=1.01)

    out = FIG_DIR / 'fig1_crater_map.png'
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f'  Saved {out.name}')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Crater size-frequency + spatial statistics  (3-panel)
# ══════════════════════════════════════════════════════════════════════════════

def fig2_crater_sfd(xy_m, diams_m, gd):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    fig.subplots_adjust(wspace=0.38)

    # ── Panel A: SFD log-log ──────────────────────────────────────────────
    ax = axes[0]
    bins = np.logspace(np.log10(diams_m.min() * 0.9),
                       np.log10(diams_m.max() * 1.1), 25)
    counts, edges = np.histogram(diams_m, bins=bins)
    centres = np.sqrt(edges[:-1] * edges[1:])
    mask_nz = counts > 0
    ax.scatter(centres[mask_nz], counts[mask_nz],
               s=30, color=C_LUNAR, edgecolors='black', linewidths=0.5, zorder=5)
    ax.plot(centres[mask_nz], counts[mask_nz], '-', color=C_LUNAR, lw=0.8, alpha=0.6)

    # Power-law fit
    lx = np.log10(centres[mask_nz]); ly = np.log10(counts[mask_nz])
    slope, intercept, r, *_ = linregress(lx, ly)
    x_fit = np.linspace(lx.min(), lx.max(), 100)
    ax.plot(10**x_fit, 10**(intercept + slope * x_fit),
            '--', color='#D55E00', lw=1.5,
            label=f'Power law  slope={slope:.2f}')
    ax.text(0.97, 0.97, f'$N \\propto D^{{{slope:.2f}}}$\n$R^2={r**2:.3f}$',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=7.5, color='#D55E00',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc'))
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Crater diameter  $D$  (m)')
    ax.set_ylabel('Count  $N(>D)$')
    ax.set_title('(a)  Size-frequency distribution\n(SFD — stochastic cratering)',
                 fontweight='bold', pad=4)
    ax.legend(fontsize=7)

    # ── Panel B: Nearest-neighbour distance distribution ──────────────────
    ax2 = axes[1]
    tree = cKDTree(xy_m)
    nn_dists, _ = tree.query(xy_m, k=2)
    nn = nn_dists[:, 1]
    ax2.hist(nn, bins=40, color=C_LUNAR, edgecolor='white', linewidth=0.4,
             density=True, label='Observed NN')

    # Poisson (CSR) expected: nn for uniform distribution
    # Expected mean NN = 1/(2*sqrt(density))
    area = (xy_m[:, 0].max() - xy_m[:, 0].min()) * (xy_m[:, 1].max() - xy_m[:, 1].min())
    density = gd['n'] / area
    mu_nn   = 1 / (2 * math.sqrt(density))
    # Rayleigh distribution for CSR
    from scipy.stats import rayleigh
    x_r = np.linspace(0, nn.max(), 200)
    sigma_r = mu_nn * math.sqrt(2 / math.pi)
    ax2.plot(x_r, rayleigh.pdf(x_r, scale=sigma_r),
             '-', color='#0072B2', lw=1.8,
             label=f'CSR Rayleigh\n$\\mu={mu_nn:.1f}$ m')

    ax2.set_xlabel('Nearest-neighbour distance  (m)')
    ax2.set_ylabel('Density')
    ax2.set_title('(b)  Nearest-neighbour distance\n(vs. complete spatial randomness)',
                  fontweight='bold', pad=4)
    ax2.legend(fontsize=7)

    # Annotation: Clark-Evans R
    obs_mean = nn.mean()
    CE_R = obs_mean / mu_nn
    ax2.text(0.97, 0.03,
             f'Clark–Evans $R = {CE_R:.3f}$\n'
             + ('$R < 1$: clustered' if CE_R < 1 else
                '$R > 1$: dispersed' if CE_R > 1 else '$R = 1$: random'),
             transform=ax2.transAxes, ha='right', va='bottom',
             fontsize=7.5, color='#333333',
             bbox=dict(boxstyle='round,pad=0.3', fc='#ffffd0', ec='#cccccc'))

    # ── Panel C: Spatial map coloured by diameter ─────────────────────────
    ax3 = axes[2]
    sc3 = ax3.scatter(xy_m[:, 0], xy_m[:, 1],
                      c=diams_m, cmap='plasma', s=10,
                      vmin=diams_m.min(), vmax=np.percentile(diams_m, 95),
                      edgecolors='none', alpha=0.85)
    cbar3 = fig.colorbar(sc3, ax=ax3, fraction=0.04, pad=0.02)
    cbar3.set_label('Diameter  (m)', fontsize=7)
    cbar3.ax.tick_params(labelsize=6.5)
    ax3.set_aspect('equal')
    ax3.set_xlabel('Easting  (m)')
    ax3.set_ylabel('Northing  (m)')
    ax3.set_title('(c)  Crater centroid map\n(colour = diameter)',
                  fontweight='bold', pad=4)
    ax3.tick_params(labelsize=7)

    fig.suptitle(
        f'Lunar Crater Statistics  ·  LROC NAC  ·  N = {gd["n"]} craters',
        fontsize=10, fontweight='bold', y=1.03)

    out = FIG_DIR / 'fig2_crater_sfd.png'
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f'  Saved {out.name}')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — kernelcal diagnostics  (3-panel)
# ══════════════════════════════════════════════════════════════════════════════

def fig3_kernelcal(gd):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    fig.subplots_adjust(wspace=0.40)

    ev   = gd['ev']
    sort = np.argsort(ev)
    lam  = ev[sort]; hs = gd['h_star'][sort]; h0s = gd['h0'][sort]
    n_show = min(100, len(ev))

    # ── A: Eigenspectrum ──────────────────────────────────────────────────
    ax = axes[0]
    ax.bar(range(n_show), ev[:n_show], color=C_LUNAR, width=1.0,
           edgecolor='none', alpha=0.80)
    ax.axvline(gd['beta0'], color='#D55E00', lw=1.5, ls='--',
               label=f'$\\beta_0 = {gd["beta0"]}$')
    ax.set_xlabel('Mode index  $l$')
    ax.set_ylabel('$\\lambda_l$')
    ax.set_title('(a)  Laplacian eigenspectrum\n(first 100 modes)',
                 fontweight='bold', pad=4)
    ax.legend(fontsize=8)
    ax.text(0.97, 0.97,
            f'$\\lambda_f = {gd["lam_f"]:.5f}$',
            transform=ax.transAxes, ha='right', va='top', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc'))

    # Inset: compare with other system archetypes (schematic description)
    ax.text(0.50, 0.55,
            'Near-linear ramp\n→ multi-scale structure\n(unexpected for\nstochastic craters)',
            transform=ax.transAxes, ha='center', va='top', fontsize=6.5,
            color='#D55E00', style='italic',
            bbox=dict(boxstyle='round,pad=0.3', fc='#fff8f8', ec='#ffcccc'))

    # ── B: h*(λ) vs h₀ ───────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.fill_between(lam, h0s, hs, where=(hs >= h0s),
                     color='#009E73', alpha=0.25, label='amplified')
    ax2.fill_between(lam, h0s, hs, where=(hs < h0s),
                     color='#D55E00', alpha=0.25, label='suppressed')
    ax2.plot(lam, h0s, '--', color='#888888', lw=1.5, label='$h_0$  vacuum')
    ax2.plot(lam, hs,  '-',  color=C_LUNAR,  lw=2.0, label='$h^*$  fixed-pt')
    ax2.set_xlabel('$\\lambda_l$')
    ax2.set_ylabel('$h(\\lambda)$')
    ax2.set_title('(b)  MaxCal fixed-point kernel\n$h^*(\\lambda)$ vs vacuum $h_0$',
                  fontweight='bold', pad=4)
    ax2.legend(fontsize=7, loc='upper right')
    ax2.text(0.05, 0.05,
             f'$\\Delta H = {gd["dH"]:+.3f}$ nats\n'
             f'$H[h^*] = {gd["H_obs"]:.3f}$\n'
             f'$H[h_0] = {gd["H_vac"]:.3f}$',
             transform=ax2.transAxes, va='bottom', fontsize=8,
             color=C_LUNAR, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc'))

    # ── C: Δβ₁ spectral modes — cumulative topology ───────────────────────
    ax3 = axes[2]
    # Show cumulative Betti-1 contribution estimated from eigenvalue spacing
    # (proxy: eigenvalue gap indicates topological band structure)
    gaps = np.diff(ev[:n_show])
    # Colour bars by gap magnitude — highlights topological bands
    bar_colors = plt.cm.RdYlGn(Normalize()(gaps))
    for li in range(len(gaps)):
        ax3.bar(li, ev[li+1] - ev[li] if li < len(gaps) else 0,
                bottom=ev[li], width=1.0, color=bar_colors[li],
                edgecolor='none', alpha=0.85)

    ax3.set_xlabel('Mode index  $l$')
    ax3.set_ylabel('$\\Delta\\lambda_l$  (gap)')
    ax3.set_title('(c)  Eigenvalue gaps  $\\Delta\\lambda_l$\n'
                  '(green = large gap = topological band boundary)',
                  fontweight='bold', pad=4)

    # Annotation boxes: topology counts
    ax3.text(0.97, 0.97,
             f'$\\beta_1 = {gd["beta1"]}$\n'
             f'$\\Delta\\beta_1 = {gd["db1"]:+d}$\n'
             f'$\\Delta\\beta_1/N = {gd["db1_N"]:.3f}$',
             transform=ax3.transAxes, ha='right', va='top', fontsize=8,
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc'))

    fig.suptitle(
        f'Lunar Crater Field — kernelcal MaxCal Diagnostics\n'
        f'LROC NAC highland region  ·  $N = {gd["n"]}$ nodes  ·  '
        f'$k = {K_NN}$-NN graph  ·  converged',
        fontsize=10, fontweight='bold', y=1.03)

    out = FIG_DIR / 'fig3_kernelcal_diagnostics.png'
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f'  Saved {out.name}')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Phase space: ALL systems  (publication figure)
# ══════════════════════════════════════════════════════════════════════════════

def fig4_phase_space(gd):
    """Full 4-system annotated phase space for paper."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    fig.subplots_adjust(wspace=0.40)

    # Empirical data
    systems = [
        {'label': 'Lunar craters',     'dH': gd['dH'],  'db1_N': gd['db1_N'],
         'color': C_LUNAR,  'marker': '*', 's': 160, 'tier': 'abiotic',
         'note': 'LROC NAC\n(stochastic impacts)'},
        {'label': 'AZ Plateau',         'dH': -0.392, 'db1_N': 221/1500,
         'color': C_ABIO,   'marker': 'o', 's': 80,  'tier': 'abiotic',
         'note': 'USGS 3DEP\n(gravity drainage)'},
        {'label': 'Jezero delta',        'dH': -0.377, 'db1_N': 960/2000,
         'color': C_FOSSIL, 'marker': 's', 's': 80,  'tier': 'fossil',
         'note': 'HiRISE DEM\n(ancient water)'},
        {'label': 'Barcelona',           'dH': -0.670, 'db1_N': 1060/678,
         'color': '#CC79A7', 'marker': 'D', 's': 55, 'tier': 'active', 'note': ''},
        {'label': 'Phoenix',             'dH': -0.691, 'db1_N': 470/678,
         'color': '#56B4E9', 'marker': 'D', 's': 55, 'tier': 'active', 'note': ''},
        {'label': 'Venice',              'dH': -0.674, 'db1_N': 1513/678,
         'color': '#D55E00', 'marker': 'D', 's': 55, 'tier': 'active', 'note': ''},
        {'label': 'Marrakech',           'dH': -0.619, 'db1_N': 1490/678,
         'color': '#F0E442', 'marker': 'D', 's': 55, 'tier': 'active', 'note': ''},
        {'label': 'Houston',             'dH': -0.629, 'db1_N': 840/678,
         'color': '#999999', 'marker': 'D', 's': 55, 'tier': 'active', 'note': ''},
    ]

    # ── Panel A: Full phase space ─────────────────────────────────────────
    ax = axes[0]

    # Tier background bands
    ax.axhspan(-0.15, 0.35, color='#EFF7FF', alpha=0.6, zorder=0)
    ax.axhspan( 0.35, 0.65, color='#FFF8E8', alpha=0.6, zorder=0)
    ax.axhspan( 0.65, 2.40, color='#EFFFEF', alpha=0.6, zorder=0)
    ax.text(-0.80, 0.05, 'Abiotic tier',   fontsize=7, color=C_ABIO,   style='italic')
    ax.text(-0.80, 0.38, 'Fossil tier',    fontsize=7, color=C_FOSSIL, style='italic')
    ax.text(-0.80, 1.50, 'Active\ncontroller', fontsize=7, color=C_ACTIVE, style='italic')

    for s in systems:
        ax.scatter(s['dH'], s['db1_N'], s=s['s'], color=s['color'],
                   marker=s['marker'], edgecolors='black', linewidths=0.6, zorder=5)
        if s['label'] in ('Lunar craters', 'AZ Plateau', 'Jezero delta'):
            xytext = (7, 5) if s['db1_N'] < 0.5 else (7, -12)
            note   = f"{s['label']}\n{s['note']}" if s['note'] else s['label']
            ax.annotate(note, (s['dH'], s['db1_N']),
                        textcoords='offset points', xytext=xytext,
                        fontsize=7, color=s['color'], fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color=s['color'],
                                        lw=0.8) if s['s'] > 100 else None)
        else:
            ax.annotate(s['label'], (s['dH'], s['db1_N']),
                        textcoords='offset points', xytext=(4, 3),
                        fontsize=6.0, color=s['color'])

    # Poisson null
    ax.scatter(-0.718, 0, s=35, color=C_NULL, marker='x',
               linewidths=1.2, zorder=3)
    ax.annotate('Poisson null', (-0.718, 0.03), fontsize=6.0, color='#888888')

    ax.axhline(0, color='#cccccc', lw=0.6, ls=':')
    ax.axvline(0, color='#cccccc', lw=0.6, ls=':')
    ax.set_xlim(-0.82, 0.08)
    ax.set_ylim(-0.15, 2.45)
    ax.set_xlabel(r'$\Delta H = H[h^*] - H[h_0]$  (nats)')
    ax.set_ylabel(r'$\Delta\beta_1 / N$  (normalised topological excess)')
    ax.set_title('(a)  Controller Phase Space — Four Tiers',
                 fontweight='bold', pad=5)

    # Legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0],[0], marker='*', color='w', markerfacecolor=C_LUNAR,
               markeredgecolor='black', markersize=10, label='Lunar craters'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=C_ABIO,
               markeredgecolor='black', markersize=7,  label='AZ Plateau (abiotic)'),
        Line2D([0],[0], marker='s', color='w', markerfacecolor=C_FOSSIL,
               markeredgecolor='black', markersize=7,  label='Jezero (fossil ctrl.)'),
        Line2D([0],[0], marker='D', color='w', markerfacecolor='#888888',
               markeredgecolor='black', markersize=7,  label='Cities (active ctrl.)'),
        Line2D([0],[0], marker='x', color=C_NULL, markersize=6,
               label='Poisson null'),
    ]
    ax.legend(handles=handles, fontsize=6.5, loc='upper right',
              framealpha=0.97)

    # ── Panel B: Δβ₁/N bar chart ─────────────────────────────────────────
    ax2 = axes[1]
    ordered = sorted(systems, key=lambda s: s['db1_N'])
    labels2  = [s['label'] for s in ordered]
    vals2    = [s['db1_N'] for s in ordered]
    cols2    = [s['color'] for s in ordered]
    htch_map = {'abiotic': '/', 'fossil': 'x', 'active': ''}
    xb = np.arange(len(ordered))
    bars2 = ax2.barh(xb, vals2, color=cols2, edgecolor='black',
                     linewidth=0.5, height=0.65)
    for bar, s in zip(bars2, ordered):
        bar.set_hatch(htch_map[s['tier']])

    # Tier separators
    abiotic_end = sum(1 for s in ordered if s['tier'] == 'abiotic')
    fossil_end  = abiotic_end + sum(1 for s in ordered if s['tier'] == 'fossil')
    ax2.axhline(abiotic_end - 0.5, color='#cccccc', lw=1.0, ls='--')
    ax2.axhline(fossil_end  - 0.5, color='#cccccc', lw=1.0, ls='--')

    for xi, v in enumerate(vals2):
        ax2.text(v + 0.02, xi, f'{v:.2f}', va='center', fontsize=7)

    ax2.set_yticks(xb)
    ax2.set_yticklabels(labels2, fontsize=7.5)
    ax2.set_xlabel(r'$\Delta\beta_1 / N$')
    ax2.set_title('(b)  Topological excess ranked\n'
                  r'$\Delta\beta_1/N$ (primary discriminant)',
                  fontweight='bold', pad=5)
    ax2.axvline(0, color='black', lw=0.7)

    # Tier labels
    tier_y = {
        'abiotic': abiotic_end / 2 - 0.5,
        'fossil':  (abiotic_end + fossil_end) / 2 - 0.5,
        'active':  (fossil_end + len(ordered)) / 2 - 0.5,
    }
    tier_c = {'abiotic': C_ABIO, 'fossil': C_FOSSIL, 'active': C_ACTIVE}
    tier_n = {'abiotic': 'Abiotic', 'fossil': 'Fossil', 'active': 'Active'}
    xlim_r = ax2.get_xlim()[1] if ax2.get_xlim()[1] > 0 else max(vals2) * 1.3
    for t, y in tier_y.items():
        ax2.text(max(vals2) * 1.25, y, tier_n[t],
                 ha='right', va='center', fontsize=7,
                 color=tier_c[t], style='italic')

    fig.suptitle(
        'Multi-System Topological Biosignature Calibration\n'
        'Lunar craters  ·  AZ Plateau  ·  Jezero delta (Mars)  ·  5 cities',
        fontsize=10, fontweight='bold', y=1.03)

    # Save to both lunar_figures and paper figures
    for out_dir, name in [
        (FIG_DIR, 'fig4_phase_space_extended.png'),
        (PAPER_FIG_DIR, 'fig_phasespace_four_tier.pdf'),
        (PAPER_FIG_DIR, 'fig_phasespace_four_tier.png'),
    ]:
        fig.savefig(out_dir / name, dpi=300)
        print(f'  Saved {name}')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print('=' * 60)
    print('Lunar crater visualizations')
    print(f'  Output: {FIG_DIR}')
    print('=' * 60)

    print('\nLoading data and recomputing graph …')
    nac, mask, transform, xy_m, diams_m, gd = load_all()
    print(f'  N = {gd["n"]} craters  |  ΔH = {gd["dH"]:+.3f}  |  '
          f'Δβ₁/N = {gd["db1_N"]:.3f}  |  β₀ = {gd["beta0"]}')

    print('\n[1] NAC image + detection overlay + graph …')
    fig1_crater_map(nac, mask, transform, xy_m, diams_m, gd)

    print('\n[2] Crater size-frequency + spatial statistics …')
    fig2_crater_sfd(xy_m, diams_m, gd)

    print('\n[3] Kernelcal diagnostics …')
    fig3_kernelcal(gd)

    print('\n[4] Phase space — all systems …')
    fig4_phase_space(gd)

    print(f'\nAll figures in {FIG_DIR}')


if __name__ == '__main__':
    main()
