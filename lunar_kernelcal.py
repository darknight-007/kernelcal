#!/usr/bin/env python3
"""
lunar_kernelcal.py — Lunar crater field as abiotic null for the controller-detection framework.

Data source: DREAMS-lab/LROC_NAC_MaskRCNN_Prediction_Pipeline
  NAC image:   NAC_ROI_ALPHNSUSLOA_E129S3581_cropped.tif
               (LROC NAC, ~12.9°S / 358.1°E lunar highland region)
  Masks:       6 × 350×350 predicted crater masks (binary, already in repo)

Pipeline:
  1. Reconstruct full mask from tiles (using pixel-offset filenames)
  2. Connected-component labelling → individual craters
  3. Extract centroids (pixel → lunar metres via rasterio transform)
  4. k-NN proximity graph on crater centroids
  5. Laplacian eigenpairs + MaxCal fixed-point kernel (kernelcal)
  6. Compare diagnostics against the predicted abiotic signature:
       ΔH ≈ 0,  Δβ₁/N ≈ 0,  β₀ large,  λ_fiedler very small

Predicted signature (pure abiotic null):
  β₀   = many    (isolated impact craters, no integrated drainage)
  Δβ₁  ≈ 0       (radial bowl drainage, no loop-forming process)
  ΔH   ≈ 0       (stochastic cratering, no scale-preferred controller)
  λ_f  ≈ 0       (crater rims fragment domain, poor global connectivity)
"""

from __future__ import annotations
import math, sys, os
from pathlib import Path

import numpy as np
from scipy.ndimage import label as nd_label, find_objects
from scipy.spatial import cKDTree
from scipy.linalg import eigh as scipy_eigh

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

KCAL_ROOT = Path(__file__).parent
sys.path.insert(0, str(KCAL_ROOT))
from kernelcal.terrain.diagnostics import (
    spectral_entropy, fixed_point_kernel, fiedler_mode_gap,
    stability_conservation_tradeoff,
)

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

# ── CONFIG ──────────────────────────────────────────────────────────────────
LROC_REPO   = Path('/tmp/lroc_nac')
NAC_TIF     = LROC_REPO / 'NAC_ROI_ALPHNSUSLOA_E129S3581_cropped.tif'
MASK_DIR    = LROC_REPO / 'predicted_masks'
FIG_DIR     = KCAL_ROOT / 'lunar_figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

K_NN   = 8
N_MAX  = 1500
MU2    = 2.0
SIGMA2 = 1.0
MIN_CRATER_PX = 9    # discard blobs < 3×3 px (noise / partial detections)

# Comparison data from prior runs
PRIOR = {
    'AZ Plateau\n(abiotic)': {'dH': -0.392, 'db1_N': 221/1500, 'color': '#0072B2', 'marker': 'o'},
    'Jezero\n(fossil ctrl)': {'dH': -0.377, 'db1_N': 960/2000, 'color': '#E69F00', 'marker': 's'},
    'Cities\n(median)':      {'dH': -0.645, 'db1_N': 960/678,  'color': '#009E73', 'marker': 'D'},
}

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 9,
    'axes.labelsize': 9, 'axes.titlesize': 9,
    'axes.linewidth': 0.8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'legend.fontsize': 7.5, 'figure.dpi': 150,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})
BG_W = 'white'


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: RECONSTRUCT FULL MASK FROM TILES
# ══════════════════════════════════════════════════════════════════════════════

def reconstruct_mask() -> tuple[np.ndarray, object]:
    """Paste tile masks back onto a canvas matching the NAC image."""
    if not HAS_RASTERIO:
        raise ImportError('rasterio required')
    src = rasterio.open(NAC_TIF)
    W, H = src.width, src.height
    transform = src.transform
    src.close()

    canvas = np.zeros((H, W), dtype=np.uint8)

    for mask_path in sorted(MASK_DIR.glob('tile_*.png')):
        stem = mask_path.stem           # tile_X_Y
        parts = stem.split('_')
        x_off = int(parts[1])           # pixel column in source
        y_off = int(parts[2])           # pixel row in source

        from PIL import Image
        tile = np.array(Image.open(mask_path))  # (350,350) uint8, 0/255
        th, tw = tile.shape[:2]

        # Clip to canvas bounds
        y_end = min(y_off + th, H)
        x_end = min(x_off + tw, W)
        canvas[y_off:y_end, x_off:x_end] = tile[:y_end-y_off, :x_end-x_off]

    print(f'  Full mask: {W}×{H} px  |  crater pixels: {(canvas>0).sum():,}'
          f'  ({100*(canvas>0).mean():.1f}%)')
    return canvas, transform


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: EXTRACT CRATER CENTROIDS
# ══════════════════════════════════════════════════════════════════════════════

def extract_craters(mask: np.ndarray, transform) -> tuple[np.ndarray, np.ndarray]:
    """Label connected components → individual craters → centroids in metres."""
    binary = (mask > 128).astype(np.int32)
    labelled, n_blobs = nd_label(binary)
    print(f'  Connected components (raw): {n_blobs}')

    centroids_px = []
    areas_px     = []
    for lbl in range(1, n_blobs + 1):
        ys, xs = np.where(labelled == lbl)
        if len(ys) < MIN_CRATER_PX:
            continue
        centroids_px.append((xs.mean(), ys.mean()))   # (col, row)
        areas_px.append(len(ys))

    centroids_px = np.array(centroids_px)       # (N, 2) in pixel space
    areas_px     = np.array(areas_px)
    print(f'  Craters after size filter (≥{MIN_CRATER_PX} px): {len(centroids_px)}')

    # Convert pixel → lunar metres via rasterio affine transform
    # transform.a = pixel width (m), transform.e = pixel height (m, negative)
    # transform.c = x origin,  transform.f = y origin
    px_size_x = abs(transform.a)
    px_size_y = abs(transform.e)
    x0 = transform.c + px_size_x / 2
    y0 = transform.f - px_size_y / 2

    xs_m = x0 + centroids_px[:, 0] * abs(transform.a)
    ys_m = y0 - centroids_px[:, 1] * abs(transform.e)

    xy_m = np.column_stack([xs_m, ys_m])
    diameters_m = 2 * np.sqrt(areas_px * px_size_x * px_size_y / math.pi)

    # Subsample to N_MAX (keep largest craters — most spatially significant)
    if len(xy_m) > N_MAX:
        top = np.argsort(areas_px)[::-1][:N_MAX]
        xy_m = xy_m[top]
        diameters_m = diameters_m[top]
        print(f'  Subsampled to top-{N_MAX} by area')

    print(f'  Diameter range: {diameters_m.min():.1f} – {diameters_m.max():.1f} m')
    return xy_m, diameters_m


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: GRAPH + KERNELCAL
# ══════════════════════════════════════════════════════════════════════════════

def build_and_run(xy: np.ndarray) -> dict:
    n    = len(xy)
    tree = cKDTree(xy)
    k_q  = min(K_NN + 1, n - 1)
    dists, inds = tree.query(xy, k=k_q + 1)

    med_nn = float(np.median(dists[:, 1]))
    xr = xy[:, 0].max() - xy[:, 0].min()
    yr = xy[:, 1].max() - xy[:, 1].min()
    sigma = max(0.05 * math.hypot(xr, yr), 2 * max(med_nn, 1e-3))

    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j_idx in range(1, min(K_NN + 1, k_q + 1)):
            j = inds[i, j_idx]
            d = dists[i, j_idx]
            w = math.exp(-d**2 / sigma**2)
            if w > W[i, j]:
                W[i, j] = w; W[j, i] = w

    L = np.diag(W.sum(1)) - W
    ev, _ = scipy_eigh(L)
    ev = np.maximum(ev, 0.0)

    n_zero  = int(np.sum(ev < 1e-6))
    w_modes = ev.copy()
    w_modes[:n_zero] = ev[n_zero] if n_zero < n else 1e-3

    h0     = np.maximum(np.exp(-ev), 1e-10)
    h_star, info = fixed_point_kernel(L, h0=h0, mu2=MU2, sigma2=SIGMA2, w=w_modes)
    h_star = np.maximum(h_star, 1e-8)

    H_obs = spectral_entropy(h_star)
    H_vac = spectral_entropy(h0)
    dH    = H_obs - H_vac
    dp    = fiedler_mode_gap(h_star, L, mu2=MU2, sigma2=SIGMA2, w=w_modes)

    n_edges  = int((W > 0).sum()) // 2
    beta0    = n_zero
    beta1    = max(0, n_edges - (n - beta0))
    e_null   = K_NN * n // 2
    db1      = beta1 - max(0, e_null - (n - 1))
    lam_f    = float(ev[n_zero]) if n_zero < n else 0.0

    return dict(
        L=L, W=W, eigvals=ev, h0=h0, h_star=h_star,
        H_obs=H_obs, H_vac=H_vac, delta_H=dH, delta_prime=dp,
        beta0=beta0, beta1=beta1, delta_beta1=db1, db1_N=db1/n,
        lam_fiedler=lam_f, n_edges=n_edges, n=n,
        converged=info['converged'], n_iter=info['n_iter'],
    )


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES (white background, publication-ready)
# ══════════════════════════════════════════════════════════════════════════════

def make_figures(mask: np.ndarray, xy: np.ndarray, diams: np.ndarray, d: dict):
    fig = plt.figure(figsize=(14, 10), facecolor=BG_W)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38)

    ev   = d['eigvals']
    sort = np.argsort(ev)
    lam  = ev[sort]; hs = d['h_star'][sort]; h0s = d['h0'][sort]

    # ── A: NAC image with crater overlay ─────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    src  = rasterio.open(NAC_TIF)
    img  = src.read(1).astype(float); src.close()
    ax_a.imshow(img, cmap='gray', origin='upper', aspect='equal')
    # overlay mask in colour
    overlay = np.zeros((*mask.shape, 4))
    overlay[mask > 0] = [1.0, 0.3, 0.0, 0.45]
    ax_a.imshow(overlay, origin='upper', aspect='equal')
    ax_a.scatter(xy[:, 0] / abs(src.transform.a),
                 (xy[:, 1] - src.transform.f) / src.transform.e,
                 s=4, c='cyan', marker='+', linewidths=0.5, alpha=0.7)
    ax_a.set_title('LROC NAC — predicted craters\n(orange mask, cyan centroids)',
                   fontsize=8, fontweight='bold', pad=4)
    ax_a.axis('off')
    ax_a.text(0.02, 0.02, f'N={d["n"]} craters', transform=ax_a.transAxes,
              color='white', fontsize=7.5, va='bottom',
              bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.6))

    # ── B: Crater size-frequency distribution ────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    bins = np.logspace(np.log10(diams.min()+1e-3), np.log10(diams.max()), 30)
    ax_b.hist(diams, bins=bins, color='#0072B2', edgecolor='white', linewidth=0.4)
    ax_b.set_xscale('log'); ax_b.set_yscale('log')
    ax_b.set_xlabel('Crater diameter (m)'); ax_b.set_ylabel('Count')
    ax_b.set_title('Crater size-frequency\n(SFD — should be power law)',
                   fontsize=8, fontweight='bold', pad=4)

    # ── C: Eigenspectrum ─────────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])
    n_show = min(80, len(ev))
    ax_c.bar(range(n_show), ev[:n_show], color='#56B4E9', width=1.0,
             edgecolor='none', alpha=0.80)
    ax_c.axvline(d['beta0'], color='#D55E00', lw=1.5, ls='--',
                 label=f'$\\beta_0={d["beta0"]}$')
    ax_c.set_xlabel('Mode index $l$'); ax_c.set_ylabel('$\\lambda_l$')
    ax_c.set_title('Laplacian eigenspectrum', fontsize=8, fontweight='bold', pad=4)
    ax_c.legend(fontsize=7)
    ax_c.text(0.97, 0.97,
              f'$\\lambda_f={d["lam_fiedler"]:.4f}$',
              transform=ax_c.transAxes, ha='right', va='top', fontsize=7.5,
              bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#cccccc'))

    # ── D: h*(λ) vs h₀ ───────────────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 0])
    ax_d.fill_between(lam, h0s, hs, where=(hs >= h0s),
                      color='#009E73', alpha=0.25, label='amplified')
    ax_d.fill_between(lam, h0s, hs, where=(hs < h0s),
                      color='#D55E00', alpha=0.25, label='suppressed')
    ax_d.plot(lam, h0s, '--', color='#888888', lw=1.2, label='$h_0$ vacuum')
    ax_d.plot(lam, hs,  '-',  color='#56B4E9', lw=1.8, label='$h^*$ fixed-pt')
    ax_d.set_xlabel('$\\lambda_l$'); ax_d.set_ylabel('$h(\\lambda)$')
    ax_d.set_title('Fixed-point kernel\n$h^*(\\lambda)$ vs vacuum',
                   fontsize=8, fontweight='bold', pad=4)
    ax_d.legend(fontsize=6.5)
    ax_d.text(0.03, 0.97, f'$\\Delta H = {d["delta_H"]:+.3f}$ nats',
              transform=ax_d.transAxes, va='top', fontsize=9,
              color='#0072B2', fontweight='bold',
              bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc'))

    # ── E: Controller phase space — ALL systems ───────────────────────────
    ax_e = fig.add_subplot(gs[1, 1])
    # Prior systems
    colors_prior = ['#0072B2', '#E69F00', '#009E73']
    markers_prior = ['o', 's', 'D']
    labels_prior  = ['AZ Plateau\n(abiotic)', 'Jezero (fossil)', 'Cities (active)']
    dH_prior  = [-0.392, -0.377, -0.645]
    db1N_prior = [221/1500, 960/2000, 960/678]
    for dh, db, col, mrk, lab in zip(dH_prior, db1N_prior,
                                      colors_prior, markers_prior, labels_prior):
        ax_e.scatter(dh, db, s=70, color=col, marker=mrk,
                     edgecolors='black', linewidths=0.6, zorder=5)
        ax_e.annotate(lab, (dh, db), textcoords='offset points',
                      xytext=(5, 4), fontsize=6.5, color=col)

    # Lunar result
    ax_e.scatter(d['delta_H'], d['db1_N'],
                 s=110, color='#CC79A7', marker='*', zorder=6,
                 edgecolors='black', linewidths=0.8)
    ax_e.annotate('Lunar\ncrater field', (d['delta_H'], d['db1_N']),
                  textcoords='offset points', xytext=(6, 4),
                  fontsize=8, color='#CC79A7', fontweight='bold')

    # Predicted "origin" box
    ax_e.add_patch(mpatches.FancyBboxPatch(
        (-0.15, -0.10), 0.3, 0.20,
        boxstyle='round,pad=0.02', fc='#ffffcc', ec='#aaaaaa',
        lw=0.8, ls='--', zorder=0, alpha=0.7
    ))
    ax_e.text(0, 0, 'Predicted\nnull zone', ha='center', va='center',
              fontsize=6.5, color='#888888', style='italic')

    ax_e.axhline(0, color='#cccccc', lw=0.7, ls=':')
    ax_e.axvline(0, color='#cccccc', lw=0.7, ls=':')
    ax_e.set_xlabel(r'$\Delta H$  (nats)')
    ax_e.set_ylabel(r'$\Delta\beta_1 / N$')
    ax_e.set_title('Controller Phase Space\n(all systems)',
                   fontsize=8, fontweight='bold', pad=4)

    # ── F: Summary bar comparison ─────────────────────────────────────────
    ax_f = fig.add_subplot(gs[1, 2])
    systems  = ['Lunar\ncraters', 'AZ\nPlateau', 'Jezero', 'Cities\n(med)']
    db1_vals = [d['db1_N'], 221/1500, 960/2000, 960/678]
    cols_bar = ['#CC79A7', '#0072B2', '#E69F00', '#009E73']
    types_b  = ['abiotic', 'abiotic', 'fossil', 'active']
    htch     = {'abiotic': '/', 'fossil': 'x', 'active': ''}
    xb = np.arange(len(systems))
    bars = ax_f.bar(xb, db1_vals, color=cols_bar, edgecolor='black',
                    linewidth=0.5, width=0.6)
    for bar, t in zip(bars, types_b):
        bar.set_hatch(htch[t])
    ax_f.axvline(1.5, color='#cccccc', lw=0.8, ls='--')
    ax_f.axvline(2.5, color='#cccccc', lw=0.8, ls='--')
    for xi, v in enumerate(db1_vals):
        ax_f.text(xi, v + 0.02, f'{v:.2f}', ha='center', fontsize=7)
    ax_f.set_xticks(xb); ax_f.set_xticklabels(systems, fontsize=7.5)
    ax_f.set_ylabel(r'$\Delta\beta_1 / N$')
    ax_f.set_title(r'Topological Excess $\Delta\beta_1/N$',
                   fontsize=8, fontweight='bold', pad=4)
    ax_f.set_ylim(-0.05, max(db1_vals) * 1.25)

    fig.suptitle(
        f'Lunar Crater Field — Abiotic Null  ·  kernelcal MaxCal\n'
        f'LROC NAC  |  $N={d["n"]}$ craters  |  '
        f'$\\Delta H={d["delta_H"]:+.3f}$  '
        f'$\\Delta\\beta_1/N={d["db1_N"]:.3f}$  '
        f'$\\beta_0={d["beta0"]}$  '
        f'$\\lambda_f={d["lam_fiedler"]:.4f}$',
        fontsize=10, fontweight='bold', y=1.02
    )
    out = FIG_DIR / 'lunar_crater_kernelcal.png'
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f'\n  Saved {out}')

    # ── Also save a clean paper figure for the biosig comparison ─────────
    fig2, ax2 = plt.subplots(figsize=(3.5, 3.2))
    for dh, db, col, mrk, lab in zip(dH_prior, db1N_prior,
                                      colors_prior, markers_prior, labels_prior):
        ax2.scatter(dh, db, s=60, color=col, marker=mrk,
                    edgecolors='black', linewidths=0.6, zorder=5)
        ax2.annotate(lab.replace('\n', ' '), (dh, db),
                     textcoords='offset points', xytext=(5, 3),
                     fontsize=6.5, color=col)
    ax2.scatter(d['delta_H'], d['db1_N'],
                s=100, color='#CC79A7', marker='*', zorder=6,
                edgecolors='black', linewidths=0.7)
    ax2.annotate('Lunar craters\n(★ abiotic null)', (d['delta_H'], d['db1_N']),
                 textcoords='offset points', xytext=(6, 4),
                 fontsize=7, color='#CC79A7', fontweight='bold')
    ax2.add_patch(mpatches.FancyBboxPatch(
        (-0.12, -0.08), 0.24, 0.16,
        boxstyle='round,pad=0.02', fc='#ffffcc', ec='#aaaaaa',
        lw=0.8, ls='--', zorder=0, alpha=0.7
    ))
    ax2.text(0, 0, 'Predicted null', ha='center', va='center',
             fontsize=6, color='#888888', style='italic')
    ax2.axhline(0, color='#cccccc', lw=0.6, ls=':')
    ax2.axvline(0, color='#cccccc', lw=0.6, ls=':')
    ax2.set_xlabel(r'$\Delta H$  (nats)')
    ax2.set_ylabel(r'$\Delta\beta_1 / N$')
    ax2.set_title('Controller Phase Space — Four Systems',
                  fontsize=8.5, fontweight='bold', pad=5)
    fig2.tight_layout()
    out2 = (KCAL_ROOT.parent /
            'journal-spectral-kernel-biosignature-planetary-surfaces-p4' /
            'figures' / 'fig_phasespace_with_lunar.pdf')
    fig2.savefig(out2, dpi=300)
    out2p = out2.with_suffix('.png')
    fig2.savefig(out2p, dpi=300)
    plt.close(fig2)
    print(f'  Saved {out2.name}  (paper figure)')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print('=' * 65)
    print('Lunar Crater Field — Abiotic Null  ·  kernelcal MaxCal')
    print(f'  NAC image: {NAC_TIF.name}')
    print(f'  Mask dir:  {MASK_DIR}')
    print('=' * 65)

    if not HAS_RASTERIO:
        print('ERROR: rasterio required'); return
    if not NAC_TIF.exists():
        print(f'ERROR: clone the DREAMS-lab repo to {LROC_REPO}'); return

    print('\n[1] Reconstructing full crater mask from tiles …')
    mask, transform = reconstruct_mask()

    print('\n[2] Extracting crater centroids …')
    xy, diams = extract_craters(mask, transform)
    print(f'  Final N: {len(xy)}')

    print('\n[3] Building k-NN graph and running kernelcal …')
    d = build_and_run(xy)

    print('\n' + '=' * 65)
    print('LUNAR CRATER FIELD — RESULTS')
    print('=' * 65)
    print(f'  N craters (graph nodes): {d["n"]}')
    print(f'  N edges:                 {d["n_edges"]}')
    print(f'  β₀ (disconnected comp.): {d["beta0"]}')
    print(f'  β₁ (independent loops):  {d["beta1"]}')
    print(f'  Δβ₁:                     {d["delta_beta1"]:+d}')
    print(f'  Δβ₁/N:                   {d["db1_N"]:.4f}')
    print(f'  λ_fiedler:               {d["lam_fiedler"]:.6f}')
    print(f'  H[h₀]:                   {d["H_vac"]:.4f} nats')
    print(f'  H[h*]:                   {d["H_obs"]:.4f} nats')
    print(f'  ΔH:                      {d["delta_H"]:+.4f} nats')
    print(f"  Δ':                      {d['delta_prime']:.4f}")
    print(f'  Converged in {d["n_iter"]} iterations')
    print()

    print('COMPARISON WITH PREDICTED ABIOTIC SIGNATURE')
    print(f'  {"Diagnostic":<20} {"Predicted":>14} {"Observed":>14}  {"Match?":>8}')
    print('  ' + '-' * 60)

    def _check(name, pred_fn, obs_val, pred_str):
        ok = pred_fn(obs_val)
        print(f'  {name:<20} {pred_str:>14} {obs_val:>14.4f}  {"✓" if ok else "✗":>8}')

    _check('ΔH ≈ 0',        lambda v: abs(v) < 0.20,  d['delta_H'],    '≈ 0')
    _check('Δβ₁/N ≈ 0',     lambda v: abs(v) < 0.30,  d['db1_N'],      '≈ 0')
    _check('λ_fiedler ≈ 0', lambda v: v < 0.01,       d['lam_fiedler'],'< 0.01')
    _check('β₀ > 1',        lambda v: v > 1,           float(d['beta0']), '> 1')

    print()
    print('FOUR-SYSTEM CONTROLLER NARRATIVE')
    print(f'  {"System":<22} {"ΔH":>7} {"Δβ₁/N":>8} {"β₀":>6}  Controller tier')
    print('  ' + '-' * 60)
    rows = [
        ('Lunar craters',    d['delta_H'], d['db1_N'],  d['beta0'],  'Abiotic (stochastic impacts)'),
        ('AZ Plateau',      -0.392,        221/1500,    22,           'Abiotic (gravity drainage)'),
        ('Jezero delta',    -0.377,        960/2000,    36,           'Fossil controller (water)'),
        ('Cities (median)', -0.645,        960/678,     1,            'Active controller (planning)'),
    ]
    for lab, dh, db, b0, ctrl in rows:
        print(f'  {lab:<22} {dh:>+7.3f} {db:>8.3f} {b0:>6d}  {ctrl}')
    print('=' * 65)

    print('\n[4] Generating figures …')
    make_figures(mask, xy, diams, d)


if __name__ == '__main__':
    main()
