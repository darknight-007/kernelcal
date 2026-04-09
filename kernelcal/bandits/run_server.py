"""
Server-optimised DDK-GPUCB runner.

Usage
-----
    python3 -m kernelcal.bandits.run_server [options]

    or directly:
    python3 kernelcal/bandits/run_server.py --T 1000 --seeds 20 --n_jobs 4

Options
-------
--T         int     Number of rounds per trial (default 500)
--seeds     int     Number of random seeds to run (default 10)
--n_arms_x  int     Spatial arm grid dimension (default 5)
--n_arms_t  int     Temporal arm grid dimension (default 6)
--sigma_n   float   Observation noise std (default 0.08)
--kernel_step float LML gradient step size (default 0.03)
--landauer  float   Thermodynamic cost weight (default 0.005)
--rho       float   Kernel consensus weight (default 0.3)
--adapt_every int   Rounds between kernel updates (default 3)
--n_jobs    int     Parallel workers; -1 = all cores (default 1)
--out       str     Output directory for results and figures (default "results")
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Allow running as script or as module
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kernelcal.bandits.field import SpatiotemporalField
from kernelcal.bandits.kernels import (AnisotropicSEKernel, SEPeriodicKernel,
                                        MixtureKernel)
from kernelcal.bandits.agents import (MixtureKernelAgent, StaticGPUCBAgent,
                                       DDUCBAgent)
from kernelcal.bandits.network import GossipNetwork, grid_adjacency


# ---------------------------------------------------------------------------
# One seed
# ---------------------------------------------------------------------------

def run_one_seed(seed: int, args) -> dict:
    """Run all methods for one seed.  Returns a metrics dict."""
    f = SpatiotemporalField(
        n_arms_x=args.n_arms_x,
        n_arms_t=args.n_arms_t,
        sigma_n=args.sigma_n,
        seed=seed,
    )
    K = f.K_arms
    arm_locs = f.arm_locations
    net = GossipNetwork(grid_adjacency(2, 2))

    per_idx = [0, 1]  # agents in periodic region (left half)
    smo_idx = [2, 3]  # agents in smooth region (right half)

    # ---- Agents ----
    mix = [MixtureKernelAgent(
               agent_id=i, arm_locations=arm_locs,
               kernel_step=args.kernel_step,
               landauer_pen=args.landauer)
           for i in range(4)]

    se  = [StaticGPUCBAgent(agent_id=i, arm_locations=arm_locs,
                             ell=0.25, sigma_n=args.sigma_n)
           for i in range(4)]

    sep = [StaticGPUCBAgent(agent_id=i, arm_locations=arm_locs,
                             ell=0.30, sigma_n=args.sigma_n)
           for i in range(4)]
    for a in sep:
        a._kernel = SEPeriodicKernel(log_sigma_n=np.log(args.sigma_n))

    dd  = [DDUCBAgent(agent_id=i, K=K, sigma=args.sigma_n)
           for i in range(4)]

    reg  = {m: np.zeros(args.T)
            for m in ["Mixture-DDK", "Static-SE", "Static-SEPer", "DDUCB"]}
    w_per, w_smo = [], []

    def _gossip(agents):
        ms = [a._m.copy() for a in agents]
        ns = [a._n.copy() for a in agents]
        for i, a in enumerate(agents):
            nb = net.neighbors(i); wts = net.gossip_weights(i)
            a.receive_gossip([ms[j] for j in nb], [ns[j] for j in nb], wts)

    for t in range(args.T):
        # Mixture
        ch = [a.select_arm() for a in mix]
        for a, arm in zip(mix, ch): a.add_local_observation(arm, f.pull(arm))
        reg["Mixture-DDK"][t] = sum(f.suboptimality_gap(arm) for arm in ch)
        _gossip(mix)
        if t % args.adapt_every == 0:
            for a in mix: a.adapt_kernel()
            w_per.append(np.mean([mix[i].mixing_weight for i in per_idx]))
            w_smo.append(np.mean([mix[i].mixing_weight for i in smo_idx]))

        # Static SE
        ch = [a.select_arm() for a in se]
        for a, arm in zip(se, ch): a.add_local_observation(arm, f.pull(arm))
        reg["Static-SE"][t] = sum(f.suboptimality_gap(arm) for arm in ch)
        _gossip(se)

        # Static SE×Per
        ch = [a.select_arm() for a in sep]
        for a, arm in zip(sep, ch): a.add_local_observation(arm, f.pull(arm))
        reg["Static-SEPer"][t] = sum(f.suboptimality_gap(arm) for arm in ch)
        _gossip(sep)

        # DDUCB
        ch = [a.select_arm() for a in dd]
        for a, arm in zip(dd, ch): a.add_local_observation(arm, f.pull(arm))
        reg["DDUCB"][t] = sum(f.suboptimality_gap(arm) for arm in ch)
        _gossip(dd)

    return {
        "seed": seed,
        "cumulative_regret": {m: np.cumsum(reg[m]).tolist() for m in reg},
        "w_per": w_per,
        "w_smo": w_smo,
        "w_per_final": float(np.mean([mix[i].mixing_weight for i in per_idx])),
        "w_smo_final": float(np.mean([mix[i].mixing_weight for i in smo_idx])),
        "K": K,
        "best_arm": f.best_arm,
        "best_reward": f.best_reward,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DDK-GPUCB server runner")
    parser.add_argument("--T",           type=int,   default=500)
    parser.add_argument("--seeds",       type=int,   default=10)
    parser.add_argument("--n_arms_x",    type=int,   default=5)
    parser.add_argument("--n_arms_t",    type=int,   default=6)
    parser.add_argument("--sigma_n",     type=float, default=0.08)
    parser.add_argument("--kernel_step", type=float, default=0.03)
    parser.add_argument("--landauer",    type=float, default=0.005)
    parser.add_argument("--rho",         type=float, default=0.3)
    parser.add_argument("--adapt_every", type=int,   default=3)
    parser.add_argument("--n_jobs",      type=int,   default=1)
    parser.add_argument("--out",         type=str,   default="results")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"DDK-GPUCB  T={args.T}  seeds={args.seeds}  "
          f"K={args.n_arms_x}×{args.n_arms_t}={args.n_arms_x*args.n_arms_t}  "
          f"jobs={args.n_jobs}")

    t0 = time.time()

    if args.n_jobs == 1:
        results = []
        for s in range(args.seeds):
            r = run_one_seed(s, args)
            results.append(r)
            w_p, w_s = r["w_per_final"], r["w_smo_final"]
            R_mix = r["cumulative_regret"]["Mixture-DDK"][-1]
            R_se  = r["cumulative_regret"]["Static-SE"][-1]
            ok = "✓" if w_p > 0.55 and w_s < 0.45 else "~"
            print(f"  seed {s:3d}  R_mix={R_mix:6.0f}  R_se={R_se:6.0f}"
                  f"  w_per={w_p:.2f}  w_smo={w_s:.2f}  {ok}")
    else:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(run_one_seed)(s, args) for s in range(args.seeds)
        )
        results.sort(key=lambda r: r["seed"])
        for r in results:
            w_p, w_s = r["w_per_final"], r["w_smo_final"]
            R_mix = r["cumulative_regret"]["Mixture-DDK"][-1]
            ok = "✓" if w_p > 0.55 and w_s < 0.45 else "~"
            print(f"  seed {r['seed']:3d}  R_mix={R_mix:6.0f}"
                  f"  w_per={w_p:.2f}  w_smo={w_s:.2f}  {ok}")

    elapsed = time.time() - t0

    # ---- Save raw results ----
    out_file = Path(args.out) / "results.json"
    with open(out_file, "w") as fh:
        json.dump({"args": vars(args), "results": results}, fh, indent=2)
    print(f"\nSaved  {out_file}  ({elapsed:.1f}s)")

    # ---- Summary ----
    methods = ["Mixture-DDK", "Static-SE", "Static-SEPer", "DDUCB"]
    print("\n=== Final R(T) ===")
    for m in methods:
        vals = [r["cumulative_regret"][m][-1] for r in results]
        print(f"  {m:20s}: {np.mean(vals):.1f} ± {np.std(vals):.1f}")

    w_per_f = [r["w_per_final"] for r in results]
    w_smo_f = [r["w_smo_final"] for r in results]
    print(f"\n=== Mixing weight at t={args.T} ===")
    print(f"  w_per: {np.mean(w_per_f):.3f} ± {np.std(w_per_f):.3f}  (target >0.6)")
    print(f"  w_smo: {np.mean(w_smo_f):.3f} ± {np.std(w_smo_f):.3f}  (target <0.4)")
    correct = sum(p > 0.55 and s < 0.45 for p, s in zip(w_per_f, w_smo_f))
    print(f"  Both correct: {correct}/{args.seeds} seeds")

    # ---- Quick matplotlib plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        T = args.T
        rounds = np.arange(T)
        colors = {"Mixture-DDK":"#2166ac","Static-SE":"#d73027",
                  "Static-SEPer":"#1a9641","DDUCB":"#f46d43"}
        ls_map = {"Mixture-DDK":"-","Static-SE":"--",
                  "Static-SEPer":"-.","DDUCB":":"}

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        ax = axes[0]
        for m in methods:
            mu = np.mean([r["cumulative_regret"][m] for r in results], axis=0)
            sd = np.std( [r["cumulative_regret"][m] for r in results], axis=0)
            ax.plot(rounds, mu, color=colors[m], ls=ls_map[m], lw=2, label=m)
            ax.fill_between(rounds, mu-sd, mu+sd, color=colors[m], alpha=0.12)
        ax.set_xlabel("Round $t$"); ax.set_ylabel("Cumulative regret $R(t)$")
        ax.set_title(f"Regret — {args.seeds} seeds × T={T}")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes[1]
        adapt_rounds = np.arange(len(results[0]["w_per"])) * args.adapt_every
        mean_wp = np.mean([r["w_per"] for r in results], axis=0)
        std_wp  = np.std( [r["w_per"] for r in results], axis=0)
        mean_ws = np.mean([r["w_smo"] for r in results], axis=0)
        std_ws  = np.std( [r["w_smo"] for r in results], axis=0)
        ax.plot(adapt_rounds, mean_wp, "#2166ac", lw=2,
                label="periodic agents ($w\\to1$)")
        ax.fill_between(adapt_rounds, mean_wp-std_wp, mean_wp+std_wp,
                        color="#2166ac", alpha=0.15)
        ax.plot(adapt_rounds, mean_ws, "#d73027", lw=2, ls="--",
                label="smooth agents ($w\\to0$)")
        ax.fill_between(adapt_rounds, mean_ws-std_ws, mean_ws+std_ws,
                        color="#d73027", alpha=0.15)
        ax.axhline(0.5, color="gray", lw=0.7, ls=":")
        ax.set_ylim(-0.05, 1.05); ax.set_xlabel("Round $t$")
        ax.set_ylabel("Mixing weight $w$")
        ax.set_title("Kernel class detection")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig_path = Path(args.out) / "results.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Saved  {fig_path}")
    except ImportError:
        print("matplotlib not available; skipping plot")


if __name__ == "__main__":
    main()
