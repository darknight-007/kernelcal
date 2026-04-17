#!/usr/bin/env python3
"""
bf_kernelcal_plots.py
=====================
Visualization of kernelcal spectral diagnostics on the Bobcat Fire (BF)
stream-channel vector time series — AZ site, Tonto National Forest.

Produces a 6-panel figure saved to figures/bf_kernelcal_analysis.png
and individual panel PNGs in figures/.

Panels
------
  A  Centroid maps (4 timestamps side-by-side) coloured by local degree
  B  k-NN graph overlay on Aug-2020 and Feb-2021 (before/after)
  C  Eigenvalue spectrum (Laplacian eigenvalues for all 4 timestamps)
  D  Fixed-point kernel h*(lambda) for all 4 timestamps
  E  Spectral entropy H[h] temporal evolution + polygon count
  F  beta1 and Delta' temporal evolution (dual axis)
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
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.spatial import cKDTree

# ── kernelcal ──────────────────────────────────────────────────────────────
KCAL_ROOT = Path(__file__).parent
sys.path.insert(0, str(KCAL_ROOT))
from kernelcal.terrain.diagnostics import (
    spectral_entropy_from_laplacian,
    fixed_point_kernel,
    fiedler_mode_gap,
    stability_conservation_tradeoff,
)

# ── CONFIG ──────────────────────────────────────────────────────────────────
DATA_DIR  = Path(__file__).parent / 'datasets' / 'bf_mbtiles'
FIG_DIR   = Path(__file__).parent / 'figures'
FIG_DIR.mkdir(exist_ok=True)

N_MAX    = 600
K_NN     = 6
SIGMA_M  = 8.0
DEDUP_M  = 0.5
MU2      = 2.0
SIGMA2   = 1.0
TARGET_Z = 20

TIMESTAMPS = [
    ('Aug 2020', DATA_DIR / 'bf_aug_2020.mbtiles'),
    ('Oct 2020', DATA_DIR / 'bf_oct_2020.mbtiles'),
    ('Dec 2020', DATA_DIR / 'bf_dec_2020_vector.mbtiles'),
    ('Feb 2021', DATA_DIR / 'bf_feb_2021_3d.mbtiles'),
]

COLORS = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']   # blue, orange, green, pink


# ══════════════════════════════════════════════════════════════════════════════
# MVT DECODER  (identical to bf_kernelcal_demo.py)
# ══════════════════════════════════════════════════════════════════════════════

def _varint(data, pos):
    result = shift = 0
    while True:
        b = data[pos]; pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80): return result, pos
        shift += 7

def _zigzag(n): return (n >> 1) ^ -(n & 1)

def _decode_geometry(cmds):
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
            l, pos = _varint(data, pos)
            blob = data[pos:pos+l]; pos += l
            if f == 4:
                ip = 0
                while ip < len(blob): v, ip = _varint(blob, ip); geom.append(v)
        elif w == 5: pos += 4
        elif w == 1: pos += 8
        else: break
    return geom

def _parse_layer(data):
    extent, feat_blobs, pos, N = 4096, [], 0, len(data)
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
        elif w == 5: pos += 4
        elif w == 1: pos += 8
        else: break
    out = []
    for fb in feat_blobs:
        cmds = _parse_feature(fb)
        if not cmds: continue
        for ring in _decode_geometry(cmds):
            if ring:
                out.append((float(np.mean([p[0] for p in ring])) / extent,
                             float(np.mean([p[1] for p in ring])) / extent))
    return out

def _parse_tile(raw):
    try: raw = gzip.decompress(raw)
    except Exception: pass
    out, pos, N = [], 0, len(raw)
    while pos < N:
        tag, pos = _varint(raw, pos)
        w, f = tag & 7, tag >> 3
        if w == 0: _, pos = _varint(raw, pos)
        elif w == 2:
            l, pos = _varint(raw, pos)
            blob = raw[pos:pos+l]; pos += l
            if f == 3: out.extend(_parse_layer(blob))
        elif w == 5: pos += 4
        elif w == 1: pos += 8
        else: break
    return out

def tile_bbox(z, x, y_tms):
    y = (1 << z) - 1 - y_tms; n = 1 << z
    lw = x/n*360 - 180; le = (x+1)/n*360 - 180
    ln = math.degrees(math.atan(math.sinh(math.pi*(1 - 2*y/n))))
    ls = math.degrees(math.atan(math.sinh(math.pi*(1 - 2*(y+1)/n))))
    return lw, ls, le, ln

def extract_centroids(path, target_zoom=TARGET_Z):
    con = sqlite3.connect(str(path))
    cur = con.cursor()
    cur.execute('SELECT DISTINCT zoom_level FROM tiles ORDER BY zoom_level DESC')
    zooms = [r[0] for r in cur.fetchall()]
    zoom = target_zoom if target_zoom in zooms else zooms[0]
    cur.execute('SELECT tile_column, tile_row, tile_data FROM tiles WHERE zoom_level=?', (zoom,))
    rows = cur.fetchall(); con.close()
    ll = []
    for tx, ty, blob in rows:
        lw, ls, le, ln = tile_bbox(zoom, tx, ty)
        for fx, fy in _parse_tile(bytes(blob)):
            ll.append((lw + fx*(le-lw), ln - fy*(ln-ls)))
    return np.array(ll) if ll else np.empty((0, 2))

def lonlat_to_metres(ll):
    lon0, lat0 = ll[:, 0].mean(), ll[:, 1].mean()
    R = 6_371_000.0; c = math.cos(math.radians(lat0))
    E = (ll[:, 0]-lon0)*c*(math.pi/180)*R
    N = (ll[:, 1]-lat0)*(math.pi/180)*R
    return np.column_stack([E, N])

def deduplicate(xy, r):
    if not len(xy): return xy
    tree = cKDTree(xy); kept = np.ones(len(xy), bool)
    for i in range(len(xy)):
        if not kept[i]: continue
        for j in tree.query_ball_point(xy[i], r):
            if j != i: kept[j] = False
    return xy[kept]

def subsample(xy, n, seed=42):
    if len(xy) <= n: return xy
    idx = np.random.default_rng(seed).choice(len(xy), n, replace=False)
    return xy[idx]

def build_laplacian_and_edges(xy, k, sigma):
    N = len(xy); tree = cKDTree(xy)
    dists, idxs = tree.query(xy, k=k+1)
    A = np.zeros((N, N)); edges = []
    for i in range(N):
        for r in range(1, k+1):
            j = idxs[i, r]
            w = math.exp(-dists[i, r]**2 / (2*sigma**2))
            if A[i, j] == 0:
                edges.append((i, j, w))
            A[i, j] += w; A[j, i] += w
    A = np.minimum(A, 1.0)
    return np.diag(A.sum(1)) - A, edges

def betti(L):
    ev = np.linalg.eigvalsh(L)
    b0 = int(np.sum(np.abs(ev) < 1e-6))
    V = L.shape[0]; A = np.diag(np.diag(L)) - L
    E = int(np.sum(A > 1e-10)) // 2
    return b0, max(0, E - V + b0)


# ══════════════════════════════════════════════════════════════════════════════
# COMPUTE ALL RESULTS
# ══════════════════════════════════════════════════════════════════════════════

print('Computing diagnostics for all timestamps...')
records = []
for label, path in TIMESTAMPS:
    print(f'  {label}...', end=' ', flush=True)
    ll   = extract_centroids(path)
    n_raw = len(ll)
    xy   = lonlat_to_metres(ll)
    xy   = deduplicate(xy, DEDUP_M)
    xy_s = subsample(xy, N_MAX)
    N    = len(xy_s)

    L, edges = build_laplacian_and_edges(xy_s, K_NN, SIGMA_M)
    eigvals  = np.linalg.eigvalsh(L)
    b0, b1   = betti(L)
    H        = spectral_entropy_from_laplacian(L, tau=1.0)
    h_star, info = fixed_point_kernel(L, mu2=MU2, sigma2=SIGMA2)
    dp       = fiedler_mode_gap(h_star, L, mu2=MU2, sigma2=SIGMA2)

    records.append(dict(
        label=label, n_raw=n_raw, N=N,
        xy=xy_s, xy_all=xy, edges=edges,
        L=L, eigvals=eigvals, h_star=h_star,
        H=H, dp=dp, b0=b0, b1=b1,
    ))
    print(f'H={H:.3f}  b1={b1}  dp={dp:.4f}')

print('Done. Generating figures...')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE: COMPREHENSIVE 3x2 PANEL
# ══════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor('#FFFFFF')
gs = gridspec.GridSpec(3, 4, figure=fig,
                       hspace=0.45, wspace=0.35,
                       left=0.05, right=0.97, top=0.93, bottom=0.07)

DARK_BG  = '#FFFFFF'
PANEL_BG = '#FFFFFF'
TEXT_COL = '#222222'
GRID_COL = '#CCCCCC'

def style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax.tick_params(colors=TEXT_COL, labelsize=8)
    ax.xaxis.label.set_color(TEXT_COL)
    ax.yaxis.label.set_color(TEXT_COL)
    if title: ax.set_title(title, color=TEXT_COL, fontsize=9, fontweight='bold', pad=6)
    if xlabel: ax.set_xlabel(xlabel, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, color=GRID_COL, linewidth=0.5, alpha=0.6)


# ── PANEL A: Centroid maps (4 timestamps) ────────────────────────────────────
print('  Panel A: centroid maps...')
for ti, rec in enumerate(records):
    ax = fig.add_subplot(gs[0, ti])
    xy = rec['xy']
    # colour points by local density (proxy for channel intensity)
    tree = cKDTree(xy)
    density = np.array([len(tree.query_ball_point(xy[i], 15.0)) for i in range(len(xy))])
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=density, s=3,
                    cmap='plasma', alpha=0.85, linewidths=0)
    style_ax(ax, title=f'A{ti+1}. {rec["label"]}  (N={rec["n_raw"]:,})',
             xlabel='East [m]', ylabel='North [m]' if ti == 0 else '')
    if ti > 0: ax.set_yticklabels([])
    cb = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label('local density\n(r=15 m)', color=TEXT_COL, fontsize=7)
    cb.ax.tick_params(colors=TEXT_COL, labelsize=7)
    ax.set_aspect('equal')


# ── PANEL B: Graph overlay — Aug vs Feb ───────────────────────────────────────
print('  Panel B: graph overlays...')
for ti, tidx in enumerate([0, 3]):
    ax = fig.add_subplot(gs[1, ti*2:ti*2+2])
    rec = records[tidx]
    xy  = rec['xy']
    col = COLORS[tidx]

    # Draw edges (thin, low alpha)
    for i, j, w in rec['edges']:
        ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]],
                color=col, alpha=float(w)*0.3, linewidth=0.4)

    # Draw nodes coloured by degree
    deg = np.array([sum(1 for _, jj, _ in rec['edges'] if _ > 0.1 and (i == _ or jj == i))
                    for i in range(len(xy))])
    # simpler: degree from Laplacian diagonal
    deg = np.diag(rec['L'])
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=deg, s=8,
                    cmap='YlOrRd', zorder=3, linewidths=0, alpha=0.9)
    style_ax(ax,
             title=f'B{ti+1}. k-NN graph — {rec["label"]}  '
                   f'b0={rec["b0"]}  b1={rec["b1"]}  H={rec["H"]:.3f}',
             xlabel='East [m]', ylabel='North [m]')
    cb = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.03)
    cb.set_label('degree (sum wts)', color=TEXT_COL, fontsize=7)
    cb.ax.tick_params(colors=TEXT_COL, labelsize=7)
    ax.set_aspect('equal')


# ── PANEL C: Eigenvalue spectrum ──────────────────────────────────────────────
print('  Panel C: eigenvalue spectrum...')
ax = fig.add_subplot(gs[2, 0:2])
for rec, col in zip(records, COLORS):
    ev = np.sort(rec['eigvals'])
    ax.plot(ev[:80], color=col, linewidth=1.4, label=rec['label'], alpha=0.9)
    # Mark Fiedler value (lambda_1)
    fiedler = ev[1] if len(ev) > 1 else 0
    ax.axvline(x=1, color=col, linewidth=0.6, linestyle=':', alpha=0.5)
style_ax(ax,
         title='C. Laplacian eigenvalue spectrum  (first 80 modes)',
         xlabel='Mode index l', ylabel='Eigenvalue lambda_l')
ax.set_yscale('log')
ax.set_ylim(bottom=1e-6)
legend = ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COL,
                   labelcolor=TEXT_COL, loc='lower right')


# ── PANEL D: Fixed-point kernel h*(lambda) ────────────────────────────────────
print('  Panel D: fixed-point kernel...')
ax = fig.add_subplot(gs[2, 2:4])
for rec, col in zip(records, COLORS):
    ev     = np.sort(np.linalg.eigvalsh(rec['L']))[:80]
    h_vals = rec['h_star'][:80]
    # Sort h* by eigenvalue
    order  = np.argsort(np.linalg.eigvalsh(rec['L']))[:80]
    h_ord  = rec['h_star'][order]
    ax.plot(range(80), h_ord, color=col, linewidth=1.4,
            label=f"{rec['label']}  dp={rec['dp']:.4f}", alpha=0.9)
# Overlay the vacuum h0 = exp(-lambda)
ev0 = np.sort(records[0]['eigvals'])[:80]
ax.plot(range(80), np.exp(-ev0), color='white', linewidth=0.8,
        linestyle='--', alpha=0.4, label='vacuum h0=exp(-lambda)')
style_ax(ax,
         title='D. Fixed-point kernel h*(lambda_l)  (sorted by eigenvalue)',
         xlabel='Mode index l  (sorted by lambda_l)',
         ylabel='h*(lambda_l)')
ax.set_yscale('log')
legend2 = ax.legend(fontsize=7, facecolor=PANEL_BG, edgecolor=GRID_COL,
                    labelcolor=TEXT_COL, loc='upper right')


# ── PANEL E-F: Temporal evolution (H, beta1, Delta') ─────────────────────────
print('  Panels E-F: temporal evolution...')
dates  = [r['label'] for r in records]
Hs     = [r['H']     for r in records]
b1s    = [r['b1']    for r in records]
dps    = [r['dp']    for r in records]
nraws  = [r['n_raw'] for r in records]
xs     = list(range(len(records)))

# ── E: H[h] and polygon count
ax_e = fig.add_subplot(gs[1, 2])
lns1 = ax_e.plot(xs, Hs, 'o-', color='#64B5F6', linewidth=2,
                 markersize=7, label='H[h] (nats)', zorder=3)
ax_e.fill_between(xs, [h*0.98 for h in Hs], [h*1.02 for h in Hs],
                  alpha=0.15, color='#64B5F6')
ax_e2 = ax_e.twinx()
lns2 = ax_e2.plot(xs, nraws, 's--', color='#FFB74D', linewidth=1.5,
                  markersize=6, label='N polygons', alpha=0.85)
ax_e2.set_ylabel('Polygon count  N_raw', color='#FFB74D', fontsize=8)
ax_e2.tick_params(colors='#FFB74D', labelsize=8)
ax_e2.yaxis.label.set_color('#FFB74D')
style_ax(ax_e,
         title='E. Spectral entropy  H[h]  &  polygon count',
         xlabel='Timestamp', ylabel='H[h]  [nats]')
ax_e.set_xticks(xs); ax_e.set_xticklabels(dates, rotation=15, ha='right', fontsize=7)
all_lns = lns1 + lns2
ax_e.legend(all_lns, [l.get_label() for l in all_lns],
            fontsize=7, facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL)

# ── F: beta1 and Delta' (dual axis)
ax_f = fig.add_subplot(gs[1, 3])
lns3 = ax_f.plot(xs, b1s, 'o-', color='#81C784', linewidth=2,
                 markersize=7, label='beta1 (loops)', zorder=3)
ax_f.fill_between(xs, [max(0, b-20) for b in b1s], [b+20 for b in b1s],
                  alpha=0.15, color='#81C784')
ax_f2 = ax_f.twinx()
lns4 = ax_f2.plot(xs, dps, 's--', color='#F06292', linewidth=1.5,
                  markersize=6, label="Delta' (stab.)", alpha=0.85)
ax_f2.set_ylabel("Delta'  (Hessian gap)", color='#F06292', fontsize=8)
ax_f2.tick_params(colors='#F06292', labelsize=8)
ax_f2.yaxis.label.set_color('#F06292')
style_ax(ax_f,
         title="F. beta1  (loops)  &  Delta'  (stability gap)",
         xlabel='Timestamp', ylabel='beta1')
ax_f.set_xticks(xs); ax_f.set_xticklabels(dates, rotation=15, ha='right', fontsize=7)
all_lns2 = lns3 + lns4
ax_f.legend(all_lns2, [l.get_label() for l in all_lns2],
            fontsize=7, facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL)


# ── Title and annotation ───────────────────────────────────────────────────────
fig.suptitle(
    'Kernelcal  |  Bobcat Fire channel network — spectral kernel dynamics\n'
    'Site: -111.265 W, 33.782 N  |  Tonto NF, AZ  |  Aug 2020 – Feb 2021  '
    '|  k=6, sigma=8 m, N_sub=600',
    color=TEXT_COL, fontsize=11, fontweight='bold', y=0.98
)

out_path = FIG_DIR / 'bf_kernelcal_analysis.png'
fig.savefig(out_path, dpi=150, facecolor=DARK_BG, bbox_inches='tight')
print(f'\nFigure saved -> {out_path}')
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: SPECTRAL ENTROPY EVOLUTION DETAIL
# ══════════════════════════════════════════════════════════════════════════════

fig2, axes = plt.subplots(1, 4, figsize=(18, 4.5))
fig2.patch.set_facecolor(DARK_BG)
fig2.suptitle(
    'Heat kernel  h(lambda) = exp(-tau * lambda)  —  spectral weight distribution  '
    '|  tau=1.0  |  Bobcat Fire AZ',
    color=TEXT_COL, fontsize=10, fontweight='bold', y=1.02
)

for ax, rec, col in zip(axes, records, COLORS):
    ev   = np.sort(np.maximum(rec['eigvals'], 0))
    h_hk = np.exp(-1.0 * ev)                     # heat kernel weights
    h_bar = h_hk / h_hk.sum()                    # normalised for entropy

    # Bar chart of spectral weights (first 60 modes)
    n_show = min(60, len(ev))
    bars = ax.bar(range(n_show), h_bar[:n_show], color=col, alpha=0.75, width=0.8)

    # Overlay cumulative
    ax2 = ax.twinx()
    ax2.plot(range(n_show), np.cumsum(h_bar[:n_show]), color='white',
             linewidth=1.2, alpha=0.6)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel('cumulative weight', color=TEXT_COL, fontsize=7)
    ax2.tick_params(colors=TEXT_COL, labelsize=7)
    ax2.set_facecolor(PANEL_BG)

    # Annotate entropy
    H_val = float(-np.sum(h_bar[h_bar > 0] * np.log(h_bar[h_bar > 0])))
    ax.text(0.97, 0.97, f'H = {H_val:.4f} nats\nb1 = {rec["b1"]}',
            transform=ax.transAxes, ha='right', va='top',
            color=TEXT_COL, fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=DARK_BG, alpha=0.7))

    style_ax(ax,
             title=f'{rec["label"]}  (N_raw={rec["n_raw"]:,})',
             xlabel='Mode index l', ylabel='Norm. spectral weight h_bar_l')
    ax.set_facecolor(PANEL_BG)

    # Mark zero-eigenvalue modes (beta0 indicator)
    zero_modes = int(np.sum(ev < 1e-6))
    if zero_modes > 0:
        ax.axvspan(0, zero_modes - 0.5, color='yellow', alpha=0.08, label=f'beta0={zero_modes}')

out2 = FIG_DIR / 'bf_spectral_weight_distribution.png'
fig2.tight_layout()
fig2.savefig(out2, dpi=150, facecolor=DARK_BG, bbox_inches='tight')
print(f'Figure saved -> {out2}')
plt.close(fig2)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: FIXED-POINT KERNEL EVOLUTION — h* ACROSS ALL TIMESTAMPS
# ══════════════════════════════════════════════════════════════════════════════

fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
fig3.patch.set_facecolor(DARK_BG)
fig3.suptitle(
    "Fixed-point kernel h*(lambda_l)  vs  vacuum h0 = exp(-lambda_l)\n"
    "Difference from vacuum = MaxCal correction from channel graph geometry",
    color=TEXT_COL, fontsize=10, fontweight='bold'
)

for ax, rec, col in zip(axes3.flat, records, COLORS):
    ev_sorted = np.sort(np.maximum(rec['eigvals'], 0))
    h0        = np.exp(-ev_sorted)
    order     = np.argsort(rec['eigvals'])
    h_star    = rec['h_star'][order]
    n_show    = min(120, len(ev_sorted))

    ax.semilogy(range(n_show), h0[:n_show], '--', color='#888',
                linewidth=1.0, alpha=0.6, label='vacuum h0=exp(-lam)')
    ax.semilogy(range(n_show), h_star[:n_show], '-', color=col,
                linewidth=1.6, alpha=0.9, label='h*(lambda_l)')

    # Shade the MaxCal correction region
    ax.fill_between(range(n_show),
                    np.minimum(h0[:n_show], h_star[:n_show]),
                    np.maximum(h0[:n_show], h_star[:n_show]),
                    alpha=0.15, color=col, label='MaxCal correction')

    ax.text(0.97, 0.97,
            f"H = {rec['H']:.4f}\nDelta' = {rec['dp']:.5f}\nb1 = {rec['b1']}",
            transform=ax.transAxes, ha='right', va='top',
            color=TEXT_COL, fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=DARK_BG, alpha=0.7))

    style_ax(ax, title=f"{rec['label']}  (N={rec['n_raw']:,} polygons)",
             xlabel='Mode index l  (sorted by lambda_l)',
             ylabel='kernel weight  h_l*')
    ax.set_facecolor(PANEL_BG)
    legend = ax.legend(fontsize=7, facecolor=PANEL_BG,
                       edgecolor=GRID_COL, labelcolor=TEXT_COL)

out3 = FIG_DIR / 'bf_fixedpoint_kernel_evolution.png'
fig3.tight_layout()
fig3.savefig(out3, dpi=150, facecolor=DARK_BG, bbox_inches='tight')
print(f'Figure saved -> {out3}')
plt.close(fig3)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: TEMPORAL DYNAMICS SUMMARY (publication-style)
# ══════════════════════════════════════════════════════════════════════════════

fig4, axes4 = plt.subplots(1, 3, figsize=(15, 4.5))
fig4.patch.set_facecolor(DARK_BG)
fig4.suptitle(
    'Kernelcal temporal diagnostics  |  Bobcat Fire channel network  |  AZ  2020-2021',
    color=TEXT_COL, fontsize=11, fontweight='bold'
)

xs   = list(range(len(records)))
xlbl = [r['label'] for r in records]

# Left: spectral entropy
ax = axes4[0]
ax.plot(xs, Hs, 'o-', color='#64B5F6', linewidth=2.5, markersize=9, zorder=3)
for xi, yi, lab in zip(xs, Hs, xlbl):
    ax.annotate(f'{yi:.3f}', (xi, yi), textcoords='offset points',
                xytext=(0, 9), ha='center', color=TEXT_COL, fontsize=8)
ax.fill_between(xs, [min(Hs)-0.05]*len(xs), Hs, alpha=0.12, color='#64B5F6')
style_ax(ax, title='Spectral entropy  H[h]  (nats)',
         xlabel='Timestamp', ylabel='H[h]  [nats]')
ax.set_xticks(xs); ax.set_xticklabels(xlbl, rotation=15, ha='right', fontsize=8)
ax.set_facecolor(PANEL_BG)

# Middle: beta1
ax = axes4[1]
ax.plot(xs, b1s, 'o-', color='#81C784', linewidth=2.5, markersize=9, zorder=3)
for xi, yi in zip(xs, b1s):
    ax.annotate(str(yi), (xi, yi), textcoords='offset points',
                xytext=(0, 9), ha='center', color=TEXT_COL, fontsize=8)
ax.fill_between(xs, [min(b1s)-10]*len(xs), b1s, alpha=0.12, color='#81C784')
style_ax(ax, title='beta1  (independent loops in channel graph)',
         xlabel='Timestamp', ylabel='beta1')
ax.set_xticks(xs); ax.set_xticklabels(xlbl, rotation=15, ha='right', fontsize=8)
ax.set_facecolor(PANEL_BG)

# Right: Delta' (stability margin)
ax = axes4[2]
ax.plot(xs, dps, 'o-', color='#F06292', linewidth=2.5, markersize=9, zorder=3)
for xi, yi in zip(xs, dps):
    ax.annotate(f'{yi:.5f}', (xi, yi), textcoords='offset points',
                xytext=(0, 9), ha='center', color=TEXT_COL, fontsize=7)
ax.fill_between(xs, [min(dps)-0.001]*len(xs), dps, alpha=0.12, color='#F06292')
style_ax(ax, title="Delta'  (Hessian stability margin)\n= conservation deficit per mode",
         xlabel='Timestamp', ylabel="Delta'")
ax.set_xticks(xs); ax.set_xticklabels(xlbl, rotation=15, ha='right', fontsize=8)
ax.set_facecolor(PANEL_BG)

out4 = FIG_DIR / 'bf_temporal_dynamics.png'
fig4.tight_layout()
fig4.savefig(out4, dpi=150, facecolor=DARK_BG, bbox_inches='tight')
print(f'Figure saved -> {out4}')
plt.close(fig4)

print(f'\nAll figures written to {FIG_DIR}/')
print('  bf_kernelcal_analysis.png          (6-panel comprehensive)')
print('  bf_spectral_weight_distribution.png (heat kernel weights per timestamp)')
print('  bf_fixedpoint_kernel_evolution.png  (h* vs vacuum h0)')
print('  bf_temporal_dynamics.png           (H, beta1, Delta\' summary)')
