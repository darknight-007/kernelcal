"""
End-to-end DDK-GPUCB simulation experiment.

Scenario
--------
A 2-D anisotropic field over a unit square lake cross-section.
N=4 agents are arranged on a 2×2 grid graph, each responsible for
one quadrant.  The field has different axis-aligned anisotropy in
different spatial regions -- no kernel that is uniform in both
lengthscales can perform well everywhere.

Metrics collected per round:
  - Instantaneous regret per agent
  - Cumulative network regret
  - Hilbert-Schmidt kernel divergence between agents
  - Fisher-Rao distance between agent kernels
  - GP inference time per agent

Baselines compared:
  1. DDK-GPUCB (our method, with kernel consensus)
  2. DDK-GPUCB-NoConsensus (local kernel adaptation, no consensus)
  3. StaticGPUCB (fixed isotropic kernel)
  4. DDUCB (no GP structure)

Usage
-----
    from kernelcal.bandits import run_experiment, ExperimentConfig
    results = run_experiment(ExperimentConfig(T=200, N=4, seed=0))
    results.plot()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .field import AnisotropicField
from .kernels import AnisotropicSEKernel, hs_distance, fisher_rao_distance
from .agents import DDKGPUCBAgent, DDUCBAgent, StaticGPUCBAgent
from .network import GossipNetwork, grid_adjacency, ring_adjacency


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Simulation parameters.

    Parameters
    ----------
    T          : number of rounds
    N          : number of agents (must be square for grid, any for ring)
    topology   : 'grid' or 'ring'
    n_arms_x, n_arms_y : arm grid dimensions
    sigma_n    : observation noise std
    kernel_step: kernel adaptation learning rate
    landauer_pen: thermodynamic cost weight
    consensus_rho: kernel consensus weight (0 to disable)
    seed       : random seed
    """

    T: int = 300
    N: int = 4
    topology: str = "grid"       # 'grid' or 'ring'
    n_arms_x: int = 5
    n_arms_y: int = 4
    sigma_n: float = 0.05
    kernel_step: float = 0.05
    landauer_pen: float = 0.01
    consensus_rho: float = 0.3
    seed: int = 42
    adapt_every: int = 5         # kernel adaptation every N rounds
    consensus_every: int = 5     # kernel consensus every N rounds


# ---------------------------------------------------------------------------
# Results container
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResults:
    """Collected metrics from one simulation run."""

    config: ExperimentConfig
    # Cumulative regret: (T, n_methods)
    cumulative_regret: Dict[str, np.ndarray] = field(default_factory=dict)
    # HS kernel divergence between agents: (T,) for DDK-GPUCB
    hs_divergence: np.ndarray = field(default_factory=lambda: np.array([]))
    # Fisher-Rao distance: (T,)
    fr_distance: np.ndarray = field(default_factory=lambda: np.array([]))
    # Per-agent kernel theta histories: list of (T/adapt_every, 4) arrays
    kernel_histories: List[np.ndarray] = field(default_factory=list)
    # GP inference wall-clock time per round (seconds)
    inference_time: np.ndarray = field(default_factory=lambda: np.array([]))

    # ------------------------------------------------------------------

    def summary(self) -> str:
        lines = ["=" * 60, "DDK-GPUCB Experiment Summary", "=" * 60]
        T = self.config.T
        for method, creg in self.cumulative_regret.items():
            lines.append(f"  {method:30s}  R(T={T}) = {creg[-1]:.2f}")
        if len(self.hs_divergence) > 0:
            lines.append(f"\n  Final HS kernel divergence : "
                         f"{self.hs_divergence[-1]:.4f}")
        if len(self.fr_distance) > 0:
            lines.append(f"  Final Fisher-Rao distance  : "
                         f"{self.fr_distance[-1]:.4f}")
        if len(self.inference_time) > 0:
            lines.append(f"  Mean GP inference time (s) : "
                         f"{self.inference_time.mean():.4f}")
        return "\n".join(lines)

    def plot(self, save_path: Optional[str] = None) -> None:  # pragma: no cover
        """Plot regret curves and kernel divergence."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available; skipping plot")
            return

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        # --- Cumulative regret ---
        ax = axes[0]
        styles = {
            "DDK-GPUCB":            dict(color="steelblue", lw=2.0),
            "DDK-GPUCB-NoConsensus":dict(color="cornflowerblue", lw=1.5, ls="--"),
            "StaticGPUCB":          dict(color="darkorange", lw=1.5, ls="-."),
            "DDUCB":                dict(color="firebrick", lw=1.5, ls=":"),
        }
        for method, creg in self.cumulative_regret.items():
            style = styles.get(method, {})
            ax.plot(creg, label=method, **style)
        ax.set_xlabel("Round $t$")
        ax.set_ylabel("Cumulative regret $R(t)$")
        ax.set_title("Network cumulative regret")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # --- Kernel divergence ---
        ax = axes[1]
        if len(self.hs_divergence) > 0:
            ax.plot(self.hs_divergence, color="steelblue", lw=2, label="HS distance")
        if len(self.fr_distance) > 0:
            ax.plot(self.fr_distance, color="darkorange", lw=2,
                    ls="--", label="FR distance (log-space)")
        ax.set_xlabel("Round $t$")
        ax.set_ylabel("Kernel divergence")
        ax.set_title("Inter-agent kernel divergence")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # --- Kernel hyperparameter trajectories ---
        ax = axes[2]
        label_map = {0: r"$\ell_x$", 1: r"$\ell_y$",
                     2: r"$\sigma_f$", 3: r"$\sigma_n$"}
        colors = ["steelblue", "darkorange", "seagreen", "firebrick"]
        for i, hist in enumerate(self.kernel_histories):
            if len(hist) == 0:
                continue
            h = np.array(hist)
            for dim in range(min(2, h.shape[1])):  # plot ell_x and ell_y
                ax.plot(
                    np.exp(h[:, dim]),
                    color=colors[dim],
                    alpha=0.5 + 0.1 * i,
                    lw=1.2,
                    label=f"Agent {i} {label_map[dim]}" if i == 0 else "",
                )
        ax.set_xlabel("Adaptation step")
        ax.set_ylabel("Lengthscale")
        ax.set_title(r"Kernel lengthscale $\ell_x, \ell_y$ per agent")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()


# ---------------------------------------------------------------------------
# Agent initialisation helpers
# ---------------------------------------------------------------------------

def _agent_arm_subsets(N: int, K: int) -> List[List[int]]:
    """Divide K arms roughly equally among N agents."""
    arm_ids = list(range(K))
    subsets = []
    for i in range(N):
        subsets.append(arm_ids[i::N])
    return subsets


def _build_network(config: ExperimentConfig) -> GossipNetwork:
    N = config.N
    if config.topology == "grid":
        rows = int(np.round(np.sqrt(N)))
        cols = int(np.ceil(N / rows))
        A = grid_adjacency(rows, cols)[:N, :N]
    else:
        A = ring_adjacency(N)
    return GossipNetwork(A)


# ---------------------------------------------------------------------------
# Single-round step for DDK-GPUCB agents
# ---------------------------------------------------------------------------

def _step_ddk(
    agents: List[DDKGPUCBAgent],
    field: AnisotropicField,
    network: GossipNetwork,
    t: int,
    config: ExperimentConfig,
) -> Tuple[float, float]:
    """One round of DDK-GPUCB: select, pull, gossip, adapt.

    Returns
    -------
    inst_regret : summed instantaneous regret across agents
    inf_time    : total GP inference wall-clock time (seconds)
    """
    # 1. Arm selection (timed)
    t0 = time.perf_counter()
    chosen = [agent.select_arm() for agent in agents]
    inf_time = time.perf_counter() - t0

    # 2. Pull rewards
    rewards = [field.pull(arm) for arm in chosen]

    # 3. Compute instantaneous regret
    inst_regret = sum(field.suboptimality_gap(arm) for arm in chosen)

    # 4. Local observation update
    for agent, arm, reward in zip(agents, chosen, rewards):
        agent.add_local_observation(arm, reward)

    # 5. Gossip: exchange reward sums and pull counts with neighbours
    new_m = [agent._m.copy() for agent in agents]
    new_n = [agent._n.copy() for agent in agents]
    for i, agent in enumerate(agents):
        nb = network.neighbors(i)
        w  = network.gossip_weights(i)
        agent.receive_gossip(
            [new_m[j] for j in nb],
            [new_n[j] for j in nb],
            w,
        )

    # 6. Kernel adaptation (every adapt_every rounds)
    if t % config.adapt_every == 0:
        for agent in agents:
            agent.adapt_kernel()

    return inst_regret, inf_time


def _step_ddk_noconsensus(
    agents: List[DDKGPUCBAgent],
    field: AnisotropicField,
    network: GossipNetwork,
    t: int,
    config: ExperimentConfig,
) -> float:
    """DDK-GPUCB without the kernel consensus step."""
    chosen = [agent.select_arm() for agent in agents]
    rewards = [field.pull(arm) for arm in chosen]
    inst_regret = sum(field.suboptimality_gap(arm) for arm in chosen)
    for agent, arm, reward in zip(agents, chosen, rewards):
        agent.add_local_observation(arm, reward)
    new_m = [a._m.copy() for a in agents]
    new_n = [a._n.copy() for a in agents]
    for i, agent in enumerate(agents):
        nb = network.neighbors(i)
        w  = network.gossip_weights(i)
        agent.receive_gossip([new_m[j] for j in nb], [new_n[j] for j in nb], w)
    if t % config.adapt_every == 0:
        for agent in agents:
            agent.adapt_kernel()
    return inst_regret


def _step_static(
    agents: List[StaticGPUCBAgent],
    field: AnisotropicField,
    network: GossipNetwork,
) -> float:
    chosen = [agent.select_arm() for agent in agents]
    rewards = [field.pull(arm) for arm in chosen]
    inst_regret = sum(field.suboptimality_gap(arm) for arm in chosen)
    for agent, arm, reward in zip(agents, chosen, rewards):
        agent.add_local_observation(arm, reward)
    new_m = [a._m.copy() for a in agents]
    new_n = [a._n.copy() for a in agents]
    for i, agent in enumerate(agents):
        nb = network.neighbors(i)
        w  = network.gossip_weights(i)
        agent.receive_gossip([new_m[j] for j in nb], [new_n[j] for j in nb], w)
    return inst_regret


def _step_dducb(
    agents: List[DDUCBAgent],
    field: AnisotropicField,
    network: GossipNetwork,
) -> float:
    chosen = [agent.select_arm() for agent in agents]
    rewards = [field.pull(arm) for arm in chosen]
    inst_regret = sum(field.suboptimality_gap(arm) for arm in chosen)
    for agent, arm, reward in zip(agents, chosen, rewards):
        agent.add_local_observation(arm, reward)
    new_m = [a._m.copy() for a in agents]
    new_n = [a._n.copy() for a in agents]
    for i, agent in enumerate(agents):
        nb = network.neighbors(i)
        w  = network.gossip_weights(i)
        agent.receive_gossip([new_m[j] for j in nb], [new_n[j] for j in nb], w)
    return inst_regret


# Needed for type hint inside function
from typing import Tuple


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment(config: ExperimentConfig) -> ExperimentResults:
    """Run all four methods on the same anisotropic field and return metrics.

    Parameters
    ----------
    config : ExperimentConfig

    Returns
    -------
    ExperimentResults with cumulative regret, kernel divergence, and timing.
    """
    # ---- Field ----
    f = AnisotropicField(
        n_arms_x=config.n_arms_x,
        n_arms_y=config.n_arms_y,
        sigma_n=config.sigma_n,
        seed=config.seed,
    )
    K = f.K_arms
    arm_locs = f.arm_locations
    subsets = _agent_arm_subsets(config.N, K)

    # ---- Network ----
    net = _build_network(config)
    print(repr(net))

    # ---- DDK-GPUCB agents ----
    ddk_agents = [
        DDKGPUCBAgent(
            agent_id=i,
            arm_locations=arm_locs,
            arm_subset=subsets[i],
            kernel_step=config.kernel_step,
            landauer_pen=config.landauer_pen,
            consensus_rho=config.consensus_rho,
        )
        for i in range(config.N)
    ]
    # DDK-GPUCB without consensus (separate agent instances)
    ddk_nc_agents = [
        DDKGPUCBAgent(
            agent_id=i,
            arm_locations=arm_locs,
            arm_subset=subsets[i],
            kernel_step=config.kernel_step,
            landauer_pen=config.landauer_pen,
            consensus_rho=0.0,  # no consensus
        )
        for i in range(config.N)
    ]
    # Static GP-UCB agents
    static_agents = [
        StaticGPUCBAgent(agent_id=i, arm_locations=arm_locs)
        for i in range(config.N)
    ]
    # DDUCB agents
    dducb_agents = [
        DDUCBAgent(agent_id=i, K=K, sigma=f.sigma_n)
        for i in range(config.N)
    ]

    # ---- Storage ----
    reg = {
        "DDK-GPUCB":             np.zeros(config.T),
        "DDK-GPUCB-NoConsensus": np.zeros(config.T),
        "StaticGPUCB":           np.zeros(config.T),
        "DDUCB":                 np.zeros(config.T),
    }
    hs_div = np.zeros(config.T)
    fr_div = np.zeros(config.T)
    inf_times = np.zeros(config.T)

    # ---- Main loop ----
    for t in range(config.T):
        # DDK-GPUCB
        inst, it = _step_ddk(ddk_agents, f, net, t, config)
        reg["DDK-GPUCB"][t] = inst
        inf_times[t] = it

        # Kernel consensus (every consensus_every rounds)
        if t % config.consensus_every == 0 and config.consensus_rho > 0:
            kernels = [a.kernel for a in ddk_agents]
            for i, agent in enumerate(ddk_agents):
                nb = net.neighbors(i)
                w  = net.gossip_weights(i)
                agent.consensus_step([kernels[j] for j in nb], w)

        # Kernel divergence between agents
        ref_X = arm_locs
        kernels = [a.kernel for a in ddk_agents]
        hs_vals = []
        fr_vals = []
        for i in range(config.N):
            for j in range(i + 1, config.N):
                hs_vals.append(hs_distance(kernels[i], kernels[j], ref_X))
                fr_vals.append(fisher_rao_distance(kernels[i], kernels[j]))
        hs_div[t] = np.mean(hs_vals) if hs_vals else 0.0
        fr_div[t] = np.mean(fr_vals) if fr_vals else 0.0

        # DDK-GPUCB no consensus
        reg["DDK-GPUCB-NoConsensus"][t] = _step_ddk_noconsensus(
            ddk_nc_agents, f, net, t, config
        )

        # Static GP-UCB
        reg["StaticGPUCB"][t] = _step_static(static_agents, f, net)

        # DDUCB
        reg["DDUCB"][t] = _step_dducb(dducb_agents, f, net)

    # Convert instantaneous to cumulative
    for method in reg:
        reg[method] = np.cumsum(reg[method])

    kernel_histories = [a.kernel_theta_history for a in ddk_agents]

    results = ExperimentResults(
        config=config,
        cumulative_regret=reg,
        hs_divergence=hs_div,
        fr_distance=fr_div,
        kernel_histories=kernel_histories,
        inference_time=inf_times,
    )
    print(results.summary())
    return results
