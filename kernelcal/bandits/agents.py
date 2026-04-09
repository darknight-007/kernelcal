"""
DDK-GPUCB agent and baseline agents.

Agents
------
DDKGPUCBAgent   -- Decentralised Dynamic-Kernel GP-UCB (our method)
DDUCBAgent      -- DDUCB baseline (no GP structure, stationary arms)
StaticGPUCBAgent -- Decentralised GP-UCB with a fixed isotropic kernel
OracleGPUCBAgent -- Centralised GP-UCB with the true non-stationary kernel
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .kernels import (AnisotropicSEKernel, SEPeriodicKernel,
                      MixtureKernel, kernel_consensus)


# ---------------------------------------------------------------------------
# Shared UCB utility
# ---------------------------------------------------------------------------

def ucb_score(mu: np.ndarray, std: np.ndarray, beta: float) -> np.ndarray:
    """GP-UCB acquisition: mu + sqrt(beta) * std."""
    return mu + np.sqrt(beta) * std


def beta_t(t: int, K: int, delta: float = 0.1) -> float:
    """Exploration parameter schedule (Srinivas et al. 2010)."""
    return 2.0 * np.log(K * (t ** 2) * (np.pi ** 2) / (6.0 * delta))


# ---------------------------------------------------------------------------
# DDK-GPUCB: Decentralised Dynamic-Kernel GP-UCB
# ---------------------------------------------------------------------------

@dataclass
class DDKGPUCBAgent:
    """Decentralised Dynamic-Kernel GP-UCB agent.

    Each agent maintains:
      - A local GP posterior with an anisotropic SE kernel.
      - Gossip buffers for pooling reward sums and pull counts.
      - A kernel that adapts via marginal-likelihood gradient ascent.
      - A kernel consensus step that averages hyperparameters with neighbours.

    Parameters
    ----------
    agent_id     : int
    arm_locations: (K, 2) array of arm positions in 2-D space
    arm_subset   : list of arm indices this agent is responsible for sampling
                   (all agents can pull all arms, but each agent's local GP
                   is initialised near the centroid of its subset)
    kernel_step  : learning rate for kernel hyperparameter update
    landauer_pen : thermodynamic cost weight (L2 penalty in log-space)
    consensus_rho: weight for kernel consensus step (0 = no consensus)
    """

    agent_id: int
    arm_locations: np.ndarray
    arm_subset: List[int] = field(default_factory=list)
    kernel_step: float = 0.05
    landauer_pen: float = 0.01
    consensus_rho: float = 0.3
    delta: float = 0.1

    def __post_init__(self) -> None:
        K = len(self.arm_locations)
        # Gossip buffers: reward sums and pull counts (running consensus)
        self._m = np.zeros(K)   # estimated network-wide reward sum per arm
        self._n = np.zeros(K)   # estimated network-wide pull count per arm
        # Local kernel: initialise with isotropic SE, will adapt
        self._kernel = AnisotropicSEKernel(
            log_ell_x=np.log(0.2),
            log_ell_y=np.log(0.2),
            log_sigma_f=np.log(0.8),
            log_sigma_n=np.log(0.1),
        )
        # Local observation buffer
        self._X_obs: list = []
        self._y_obs: list = []
        # Round counter
        self._t: int = 0
        # History for metrics
        self.kernel_theta_history: list = []
        self.lml_history: list = []

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def kernel(self) -> AnisotropicSEKernel:
        return self._kernel

    @property
    def n_obs(self) -> int:
        return len(self._y_obs)

    # ------------------------------------------------------------------
    # Gossip buffer update (called once per round by the simulator)
    # ------------------------------------------------------------------

    def receive_gossip(
        self,
        m_neighbors: List[np.ndarray],
        n_neighbors: List[np.ndarray],
        weights: np.ndarray,
    ) -> None:
        """Running consensus update for reward sums and pull counts."""
        m_new = np.zeros_like(self._m)
        n_new = np.zeros_like(self._n)
        for m_j, n_j, w in zip(m_neighbors, n_neighbors, weights):
            m_new += w * m_j
            n_new += w * n_j
        # Self-weight
        self_w = 1.0 - np.sum(weights)
        self._m = self_w * self._m + m_new
        self._n = self_w * self._n + n_new

    def add_local_observation(self, arm: int, reward: float) -> None:
        """Add a local (arm, reward) pair to buffers and gossip state."""
        self._m[arm] += reward
        self._n[arm] += 1.0
        self._X_obs.append(self.arm_locations[arm])
        self._y_obs.append(reward)

    # ------------------------------------------------------------------
    # Arm selection: GP-UCB on local posterior
    # ------------------------------------------------------------------

    def select_arm(self) -> int:
        """Return arm index with highest GP-UCB score."""
        self._t += 1
        K = len(self.arm_locations)

        if self.n_obs < 2:
            # cold start: optimistic initialisation
            return int(np.argmin(self._n))

        X_obs = np.array(self._X_obs)
        y_obs = np.array(self._y_obs)
        mu, std = self._kernel.posterior(X_obs, y_obs, self.arm_locations)
        beta = beta_t(self._t, K, self.delta)
        scores = ucb_score(mu, std, beta)
        return int(np.argmax(scores))

    # ------------------------------------------------------------------
    # Kernel adaptation: marginal-likelihood gradient + Landauer penalty
    # ------------------------------------------------------------------

    def adapt_kernel(self) -> float:
        """One MaxCal-inspired gradient step on log-marginal-likelihood.

        Returns the LML before the step.
        """
        if self.n_obs < 3:
            return 0.0
        X = np.array(self._X_obs)
        y = np.array(self._y_obs)
        lml = self._kernel.adapt(
            X, y,
            step=self.kernel_step,
            landauer_penalty=self.landauer_pen,
        )
        self.kernel_theta_history.append(self._kernel.theta.copy())
        self.lml_history.append(lml)
        return lml

    # ------------------------------------------------------------------
    # Kernel consensus: average hyperparameters with neighbours
    # ------------------------------------------------------------------

    def consensus_step(
        self,
        neighbor_kernels: List[AnisotropicSEKernel],
        weights: np.ndarray,
    ) -> None:
        """Gossip-weighted average of kernel hyperparameters."""
        if not neighbor_kernels:
            return
        self._kernel = kernel_consensus(
            self._kernel, neighbor_kernels, weights, rho=self.consensus_rho
        )

    # ------------------------------------------------------------------
    # Estimated mean rewards (for arm selection in DDUCB-style fallback)
    # ------------------------------------------------------------------

    def estimated_means(self) -> np.ndarray:
        """Ratio estimator: m_hat / n_hat per arm."""
        safe_n = np.where(self._n > 0, self._n, 1.0)
        return self._m / safe_n


# ---------------------------------------------------------------------------
# DDUCB baseline: no GP, standard UCB on gossip-pooled statistics
# ---------------------------------------------------------------------------

@dataclass
class DDUCBAgent:
    """Decentralised Delayed UCB agent (Martínez-Rubio et al. 2019).

    Uses sample-mean UCB without any GP structure.  Arms are treated
    as independent with unknown means.
    """

    agent_id: int
    K: int
    sigma: float = 1.0
    eta: float = 2.0
    delta: float = 0.1

    def __post_init__(self) -> None:
        self._m = np.zeros(self.K)
        self._n = np.zeros(self.K)
        self._t: int = 0

    def receive_gossip(
        self,
        m_neighbors: List[np.ndarray],
        n_neighbors: List[np.ndarray],
        weights: np.ndarray,
    ) -> None:
        m_new = n_new = np.zeros(self.K)
        for mj, nj, w in zip(m_neighbors, n_neighbors, weights):
            m_new = m_new + w * mj
            n_new = n_new + w * nj
        self_w = 1.0 - np.sum(weights)
        self._m = self_w * self._m + m_new
        self._n = self_w * self._n + n_new

    def add_local_observation(self, arm: int, reward: float) -> None:
        self._m[arm] += reward
        self._n[arm] += 1.0

    def select_arm(self) -> int:
        self._t += 1
        safe_n = np.where(self._n > 0, self._n, 1e-9)
        means = self._m / safe_n
        bonus = np.sqrt(2 * self.eta * self.sigma ** 2 * np.log(self._t + 1) / safe_n)
        return int(np.argmax(means + bonus))


# ---------------------------------------------------------------------------
# MixtureKernelAgent: DDK-GPUCB with SE + SE×Periodic mixture (Class switching)
# ---------------------------------------------------------------------------

@dataclass
class MixtureKernelAgent:
    """DDK-GPUCB agent with a learnable SE + SE×Periodic mixture kernel.

    This is the "different lens" agent.  It starts with equal weight on
    SE and SE×Periodic (w=0.5) and learns the mixing weight from data.

    In periodic regions: w → 1 (SE×Periodic wins)
    In smooth regions:   w → 0 (SE wins)

    The mixing weight trajectory directly shows whether the agent has
    detected the structural difference between its region and the alternative.
    """

    agent_id: int
    arm_locations: np.ndarray
    arm_subset: List[int] = field(default_factory=list)
    kernel_step: float = 0.05
    landauer_pen: float = 0.01
    delta: float = 0.1

    def __post_init__(self) -> None:
        K = len(self.arm_locations)
        self._m = np.zeros(K)
        self._n = np.zeros(K)
        self._kernel = MixtureKernel(
            kernel_se=AnisotropicSEKernel(
                log_ell_x=np.log(0.25), log_ell_y=np.log(0.25)),
            kernel_per=SEPeriodicKernel(
                log_ell_x=np.log(0.25), log_period=np.log(0.35)),
            logit_w=0.0,   # start at w=0.5
        )
        self._X_obs: list = []
        self._y_obs: list = []
        self._t: int = 0
        self.w_history: list = []
        self.lml_history: list = []

    @property
    def kernel(self) -> MixtureKernel:
        return self._kernel

    @property
    def mixing_weight(self) -> float:
        """Current w: fraction of SE×Periodic in the mixture."""
        return self._kernel.w

    def receive_gossip(self, m_neighbors, n_neighbors, weights):
        m_new = n_new = np.zeros(len(self._m))
        for mj, nj, w in zip(m_neighbors, n_neighbors, weights):
            m_new = m_new + w * mj
            n_new = n_new + w * nj
        self_w = 1.0 - np.sum(weights)
        self._m = self_w * self._m + m_new
        self._n = self_w * self._n + n_new

    def add_local_observation(self, arm: int, reward: float) -> None:
        self._m[arm] += reward
        self._n[arm] += 1.0
        self._X_obs.append(self.arm_locations[arm])
        self._y_obs.append(reward)

    def select_arm(self) -> int:
        self._t += 1
        K = len(self.arm_locations)
        if len(self._y_obs) < 2:
            return int(np.argmin(self._n))
        X = np.array(self._X_obs)
        y = np.array(self._y_obs)
        mu, std = self._kernel.posterior(X, y, self.arm_locations)
        b = beta_t(self._t, K, self.delta)
        return int(np.argmax(ucb_score(mu, std, b)))

    def adapt_kernel(self) -> float:
        if len(self._y_obs) < 3:
            return 0.0
        X = np.array(self._X_obs)
        y = np.array(self._y_obs)
        lml = self._kernel.adapt(X, y, step=self.kernel_step,
                                 landauer_penalty=self.landauer_pen)
        self.w_history.append(self._kernel.w)
        self.lml_history.append(lml)
        return lml


# ---------------------------------------------------------------------------
# StaticGPUCBAgent: decentralised GP-UCB with fixed isotropic kernel
# ---------------------------------------------------------------------------

@dataclass
class StaticGPUCBAgent:
    """Decentralised GP-UCB with a fixed isotropic SE kernel.

    The kernel hyperparameters are set at initialisation and never updated.
    This is the baseline that shows the cost of kernel misspecification.
    """

    agent_id: int
    arm_locations: np.ndarray
    ell: float = 0.2         # fixed isotropic lengthscale
    sigma_f: float = 1.0
    sigma_n: float = 0.1
    delta: float = 0.1

    def __post_init__(self) -> None:
        self._kernel = AnisotropicSEKernel(
            log_ell_x=np.log(self.ell),
            log_ell_y=np.log(self.ell),
            log_sigma_f=np.log(self.sigma_f),
            log_sigma_n=np.log(self.sigma_n),
        )
        self._X_obs: list = []
        self._y_obs: list = []
        self._m = np.zeros(len(arm_locations := self.arm_locations))
        self._n = np.zeros(len(arm_locations))
        self._t: int = 0

    def receive_gossip(
        self,
        m_neighbors: List[np.ndarray],
        n_neighbors: List[np.ndarray],
        weights: np.ndarray,
    ) -> None:
        m_new = n_new = np.zeros(len(self._m))
        for mj, nj, w in zip(m_neighbors, n_neighbors, weights):
            m_new = m_new + w * mj
            n_new = n_new + w * nj
        self_w = 1.0 - np.sum(weights)
        self._m = self_w * self._m + m_new
        self._n = self_w * self._n + n_new

    def add_local_observation(self, arm: int, reward: float) -> None:
        self._m[arm] += reward
        self._n[arm] += 1.0
        self._X_obs.append(self.arm_locations[arm])
        self._y_obs.append(reward)

    def select_arm(self) -> int:
        self._t += 1
        K = len(self.arm_locations)
        if len(self._y_obs) < 2:
            return int(np.argmin(self._n))
        X = np.array(self._X_obs)
        y = np.array(self._y_obs)
        mu, std = self._kernel.posterior(X, y, self.arm_locations)
        b = beta_t(self._t, K, self.delta)
        return int(np.argmax(ucb_score(mu, std, b)))
