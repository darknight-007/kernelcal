#!/usr/bin/env python3
"""
Standalone demo: five-way bloom-following strategy comparison (no ROS2 required).

No agent has prior knowledge of the bloom field.  Each rover carries an online
GP surrogate trained solely on its own noisy observations at VISITED positions.
Bloom concentration and gradient at UNVISITED candidate positions are estimated
from this GP posterior — no oracle access to the ground-truth field.

Five rovers follow the same double-gyre advecting Gaussian bloom:

  MaxCal       — p*(x) ∝ q(x)exp(−λ·f(x)), λ fitted to GP-utility constraint
  Adaptive-q   — Gaussian prior, σ_q adapted from bloom EMA, λ≡0
  Greedy       — argmax GP-mean utility over candidates; zero entropy
  Gradient     — follow GP gradient direction at current position
  Static-q     — fixed proximity prior q, λ≡0  (null reference)

Early in the mission the GP is uninformed → all strategies explore similarly.
Differentiation emerges once the GP accumulates observations near bloom patches.

Key comparisons:
  MaxCal vs Greedy    → value of entropy maintenance vs argmax of GP mean
  MaxCal vs Gradient  → MaxCal vs classical GP-gradient ascent
  Adaptive-q vs Static → bandwidth adaptation alone (no constraint opt.)
  Greedy vs Static    → value of any bloom information (GP vs none)

Usage
-----
    python demo_bloom_maxcal.py                 # live 5-panel animation
    python demo_bloom_maxcal.py --steps 300 --no-plot   # headless benchmark
    python demo_bloom_maxcal.py --steps 300 --save      # save PNG + GIF
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_MANUSCRIPT = os.path.abspath(os.path.join(_HERE, '..', '..', '..', '..'))
for p in [_HERE, _MANUSCRIPT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from bloom_maxcal_sim.bloom_field import AlgaeBloomField
from bloom_maxcal_sim.rover_model import DifferentialDriveRover, RoverConfig
from bloom_maxcal_sim.maxcal_bloom_follower import MaxCalBloomFollower, MaxCalConfig


# ---------------------------------------------------------------------------
# Strategy definitions
# ---------------------------------------------------------------------------

STRATEGIES = [
    ('maxcal',     'MaxCal',      'steelblue',      '-'),
    ('adaptive_q', 'Adaptive-q',  'mediumseagreen',  '-.'),
    ('greedy',     'Greedy',      'crimson',         '--'),
    ('gradient',   'Gradient ∇b', 'darkorchid',      ':'),
    ('static',     'Static-q',    'darkorange',      (0,(3,1,1,1))),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--steps',       type=int,   default=300)
    p.add_argument('--dt',          type=float, default=1.0)
    p.add_argument('--rover-steps', type=int,   default=10)
    p.add_argument('--save',        action='store_true')
    p.add_argument('--no-plot',     action='store_true')
    p.add_argument('--seed',        type=int,   default=42)
    return p.parse_args()


def make_cfg(mode: str) -> MaxCalConfig:
    return MaxCalConfig(
        n_candidates=32,
        lookahead_min=3.0, lookahead_max=10.0,
        sigma_q=6.0,
        bloom_target_quantile=0.65,
        kernel_length_scale=5.0,
        v_max=1.2, k_omega=1.8, arrival_radius=2.5,
        domain_x=(0.0, 100.0), domain_y=(0.0, 100.0),
        mode=mode,
        adaptive_q_ema_alpha=0.15,
        adaptive_q_sigma_min=2.5,
        adaptive_q_sigma_max=12.0,
    )


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def setup_figure(strategies):
    matplotlib.use('TkAgg')
    plt.ion()
    fig = plt.figure(figsize=(19, 10))
    fig.suptitle(
        "Bloom-following (no prior knowledge — GP surrogate from visited positions only)\n"
        "MaxCal · Adaptive-q · Greedy (GP) · Gradient ∇b (GP) · Static-q   ·   arXiv:2603.27880",
        fontsize=10, fontweight='bold',
    )
    gs = GridSpec(3, 3, figure=fig, hspace=0.58, wspace=0.42)
    ax_bloom  = fig.add_subplot(gs[:, :2])
    ax_bobs   = fig.add_subplot(gs[0, 2])
    ax_cum    = fig.add_subplot(gs[1, 2])
    ax_diag   = fig.add_subplot(gs[2, 2])

    ax_bloom.set_xlabel("x (m)"); ax_bloom.set_ylabel("y (m)")
    ax_bobs.set_title("Instantaneous bloom obs"); ax_bobs.set_xlabel("step")
    ax_cum.set_title("Cumulative bloom  ∫b dt");  ax_cum.set_xlabel("step")
    ax_diag.set_title("MaxCal |λ|, H  |  adaptive σ_q"); ax_diag.set_xlabel("step")

    im = ax_bloom.imshow(np.zeros((10,10)), origin='lower', cmap='YlGn',
                         vmin=0., vmax=1., aspect='auto', extent=[0,100,0,100])
    fig.colorbar(im, ax=ax_bloom, label='Bloom conc.')
    quiv = [None]

    lines_traj, dots, lines_bobs, lines_cum = {}, {}, {}, {}
    for key, label, color, ls in strategies:
        lt, = ax_bloom.plot([], [], color=color, lw=1.4, alpha=0.82,
                            label=label, linestyle=ls)
        dt, = ax_bloom.plot([], [], 'o', color=color, ms=7)
        lb, = ax_bobs.plot([], [], color=color, lw=1.3, label=label, linestyle=ls)
        lc, = ax_cum.plot([], [],  color=color, lw=1.3, label=label, linestyle=ls)
        lines_traj[key] = lt;  dots[key] = dt
        lines_bobs[key] = lb;  lines_cum[key] = lc

    ax_bloom.legend(loc='upper right', fontsize=8)
    ax_bobs.legend(fontsize=7); ax_cum.legend(fontsize=7)

    ln_lam, = ax_diag.plot([], [], color='steelblue',      lw=1.3, label='MaxCal |λ|')
    ln_H,   = ax_diag.plot([], [], color='steelblue',      lw=1.0, ls=':', label='MaxCal H')
    ln_sq,  = ax_diag.plot([], [], color='mediumseagreen', lw=1.3, ls='-.', label='Adapt. σ_q')
    ax_diag.legend(fontsize=7)

    return dict(
        fig=fig, im=im, quiv=quiv,
        ax_bloom=ax_bloom, ax_bobs=ax_bobs, ax_cum=ax_cum, ax_diag=ax_diag,
        lines_traj=lines_traj, dots=dots, lines_bobs=lines_bobs, lines_cum=lines_cum,
        ln_lam=ln_lam, ln_H=ln_H, ln_sq=ln_sq,
    )


def update_figure(H, bloom, followers, data, step):
    grid = bloom.field_grid().astype(np.float32)
    H['im'].set_data(grid); H['im'].set_clim(0., max(float(grid.max()), 0.01))

    sk = 10
    U, V = bloom.advection_field_grid()
    XX, YY = np.meshgrid(bloom.xs[::sk], bloom.ys[::sk])
    if H['quiv'][0] is not None: H['quiv'][0].remove()
    H['quiv'][0] = H['ax_bloom'].quiver(XX, YY, U[::sk,::sk], V[::sk,::sk],
                                         scale=0.6, alpha=0.25, color='navy', width=0.002)

    for key, _, _, _ in STRATEGIES:
        H['lines_traj'][key].set_data(data['traj_x'][key], data['traj_y'][key])
        pos = data['pos'][key]
        H['dots'][key].set_data([pos[0]], [pos[1]])

    H['ax_bloom'].set_xlim(0,100); H['ax_bloom'].set_ylim(0,100)
    bvals = {k: (data['bloom'][k][-1] if data['bloom'][k] else 0.) for k,*_ in STRATEGIES}
    ref = max(bvals['static'], 1e-9)
    title = f"t={bloom.t:.0f}s  step={step}\n" + \
            "  ".join(f"{lbl}={bvals[k]:.3f}({bvals[k]/ref:.1f}×)"
                      for k, lbl, *_ in STRATEGIES)
    H['ax_bloom'].set_title(title, fontsize=8)

    steps = list(range(len(data['bloom']['maxcal'])))
    for key, *_ in STRATEGIES:
        H['lines_bobs'][key].set_data(steps, data['bloom'][key])
        H['lines_cum'][key].set_data(steps, data['cum'][key])

    all_b = sum((data['bloom'][k] for k,*_ in STRATEGIES), [])
    H['ax_bobs'].set_xlim(0, max(1,len(steps)))
    H['ax_bobs'].set_ylim(0, max(max(all_b, default=0.05)*1.15, 0.05))
    all_c = sum((data['cum'][k] for k,*_ in STRATEGIES), [])
    H['ax_cum'].set_xlim(0, max(1,len(steps)))
    H['ax_cum'].set_ylim(0, max(max(all_c, default=0.05)*1.05, 1.))

    if data['lambda_mc']:
        n = len(data['lambda_mc'])
        H['ln_lam'].set_data(list(range(n)), data['lambda_mc'])
        H['ln_H'].set_data(list(range(len(data['entropy_mc']))), data['entropy_mc'])
        H['ln_sq'].set_data(list(range(len(data['sigma_q_aq']))), data['sigma_q_aq'])
        all_d = data['lambda_mc'] + data['entropy_mc'] + data['sigma_q_aq']
        H['ax_diag'].set_xlim(0, max(1,n)); H['ax_diag'].set_ylim(0, max(max(all_d)*1.15, 0.1))

    H['fig'].canvas.draw_idle(); H['fig'].canvas.flush_events()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(data, strategies):
    print()
    print("═"*74)
    print("  Five-strategy bloom-following comparison")
    print("═"*74)
    header = f"  {'Metric':<32}"
    for _, lbl, *_ in strategies:
        header += f"  {lbl:>10}"
    print(header)
    print("  " + "─"*32 + ("  " + "─"*10) * len(strategies))

    def row(label, vals, fmt='.4f'):
        s = f"  {label:<32}"
        for v in vals: s += f"  {v:>10{fmt}}"
        print(s)

    bm = {k: np.array(data['bloom'][k]) for k,*_ in strategies}
    row("Mean bloom obs",     [bm[k].mean()          for k,*_ in strategies])
    row("Max bloom obs",      [bm[k].max()           for k,*_ in strategies])
    row("Cumulative bloom",   [data['cum'][k][-1] if data['cum'][k] else 0.
                               for k,*_ in strategies], fmt='.1f')
    row("Frac obs > 0.5",     [(bm[k]>0.5).mean()   for k,*_ in strategies])
    print()
    ref = max(bm['static'].mean(), 1e-9)
    row("Gain vs Static-q",   [bm[k].mean()/ref      for k,*_ in strategies], fmt='.2f')
    ref_gr = max(bm['greedy'].mean(), 1e-9)
    row("Gain vs Greedy",     [bm[k].mean()/ref_gr   for k,*_ in strategies], fmt='.2f')
    print("═"*74 + "\n")


# ---------------------------------------------------------------------------
# σ_q readout for adaptive_q
# ---------------------------------------------------------------------------

def sigma_q_from_ema(ema: float, cfg: MaxCalConfig) -> float:
    b = float(np.clip(ema, 0., 2.))
    return cfg.adaptive_q_sigma_min + (cfg.adaptive_q_sigma_max - cfg.adaptive_q_sigma_min) * math.exp(-3.*b)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    bloom = AlgaeBloomField(seed=args.seed)

    rover_cfg = RoverConfig(x0=50., y0=50., theta0=0.,
                            v_max=1.5, omega_max=1.2,
                            sigma_obs=0.02, sigma_dir=0.15)
    rovers    = {k: DifferentialDriveRover(config=rover_cfg, seed=args.seed+i+1)
                 for i,(k,*_) in enumerate(STRATEGIES)}
    followers = {k: MaxCalBloomFollower(config=make_cfg(k))
                 for k,*_ in STRATEGIES}
    rngs      = {k: np.random.default_rng(args.seed + i*10 + 10)
                 for i,(k,*_) in enumerate(STRATEGIES)}

    rover_dt = args.dt / args.rover_steps

    data = {
        'traj_x': {k: [] for k,*_ in STRATEGIES},
        'traj_y': {k: [] for k,*_ in STRATEGIES},
        'pos':    {k: (50., 50.) for k,*_ in STRATEGIES},
        'bloom':  {k: [] for k,*_ in STRATEGIES},
        'cum':    {k: [] for k,*_ in STRATEGIES},
        'lambda_mc': [], 'entropy_mc': [], 'sigma_q_aq': [],
    }
    frames = []

    show = not args.no_plot
    H = setup_figure(STRATEGIES) if show else None

    keys = [k for k,*_ in STRATEGIES]
    labels = {k: lbl for k,lbl,*_ in STRATEGIES}
    print(f"\nRunning {args.steps} steps — 5-way comparison")
    print(f"{'step':>5}  " + "  ".join(f"{labels[k]:>9}" for k in keys) +
          "  " + "  ".join(f"/{labels[k][:3]:>4}" for k in ['greedy','static']))
    t0 = time.time()

    for step in range(args.steps):
        for k in keys:
            v, w = followers[k].update(rovers[k], bloom, args.dt, rng=rngs[k])
            for _ in range(args.rover_steps):
                rovers[k].step(v, w, rover_dt)

        bloom.step(args.dt)

        for k in keys:
            x, y, _ = rovers[k].pose()
            data['traj_x'][k].append(x); data['traj_y'][k].append(y)
            data['pos'][k] = (x, y)
            rec = followers[k].last_record()
            b = rec.bloom_obs if rec else 0.
            data['bloom'][k].append(b)
            data['cum'][k].append((data['cum'][k][-1] if data['cum'][k] else 0.) + b)

        rec_mc = followers['maxcal'].last_record()
        rec_aq = followers['adaptive_q'].last_record()
        if rec_mc:
            data['lambda_mc'].append(float(np.linalg.norm(rec_mc.lagrange_multipliers)))
            data['entropy_mc'].append(rec_mc.entropy_nats)
        data['sigma_q_aq'].append(sigma_q_from_ema(followers['adaptive_q']._bloom_ema,
                                                    followers['adaptive_q'].cfg))

        if step % 50 == 0 or step == args.steps - 1:
            bv  = {k: data['bloom'][k][-1] for k in keys}
            ref_gr = max(bv['greedy'], 1e-9)
            ref_sk = max(bv['static'], 1e-9)
            elapsed = time.time() - t0
            print(f"{step:5d}  " +
                  "  ".join(f"{bv[k]:9.4f}" for k in keys) +
                  f"  {bv['maxcal']/ref_gr:>5.2f}×  {bv['maxcal']/ref_sk:>5.2f}×  "
                  f"({elapsed:.1f}s)")

        if show and step % 5 == 0:
            update_figure(H, bloom, followers, data, step)
            if args.save:
                H['fig'].canvas.draw()
                buf = np.frombuffer(H['fig'].canvas.tostring_rgb(), dtype=np.uint8)
                frames.append(buf.reshape(H['fig'].canvas.get_width_height()[::-1]+(3,)))

    print_summary(data, STRATEGIES)

    if show:
        plt.ioff(); plt.tight_layout()
        if args.save:
            plt.savefig('bloom_5way_comparison.png', dpi=150)
            if frames:
                try:
                    from PIL import Image
                    imgs = [Image.fromarray(f) for f in frames]
                    imgs[0].save('bloom_5way_comparison.gif', save_all=True,
                                 append_images=imgs[1:], duration=150, loop=0)
                    print("Saved bloom_5way_comparison.gif")
                except ImportError:
                    print("Pillow not installed; skip GIF.")
        plt.show()


if __name__ == '__main__':
    main()
