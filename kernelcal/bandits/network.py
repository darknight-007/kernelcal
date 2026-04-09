"""
Gossip network with Chebyshev-accelerated consensus.

Implements the communication layer from DDUCB
(Martínez-Rubio, Kanade, Rebeschini, NeurIPS 2019).

Classes
-------
GossipNetwork
    Builds and holds the gossip matrix P for an arbitrary graph.
    Provides Chebyshev-accelerated mixing and raw mixing steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Pre-built graph topologies
# ---------------------------------------------------------------------------

def ring_adjacency(n: int) -> np.ndarray:
    """Adjacency matrix for a ring of n nodes."""
    A = np.zeros((n, n))
    for i in range(n):
        A[i, (i + 1) % n] = 1
        A[(i + 1) % n, i] = 1
    return A


def grid_adjacency(rows: int, cols: int) -> np.ndarray:
    """Adjacency matrix for a rows×cols 2-D grid (4-connected)."""
    n = rows * cols
    A = np.zeros((n, n))
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    j = nr * cols + nc
                    A[i, j] = 1
    return A


def complete_adjacency(n: int) -> np.ndarray:
    """Adjacency matrix for a complete graph of n nodes."""
    return np.ones((n, n)) - np.eye(n)


def gossip_matrix_metropolis(A: np.ndarray) -> np.ndarray:
    """Metropolis-Hastings gossip matrix from adjacency matrix A.

    P_ij = 1 / (1 + max(d_i, d_j))  for i ≠ j, {i,j} ∈ E
    P_ii = 1 - sum_{j ≠ i} P_ij
    """
    n = A.shape[0]
    d = A.sum(axis=1)
    P = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and A[i, j] > 0:
                P[i, j] = 1.0 / (1.0 + max(d[i], d[j]))
        P[i, i] = 1.0 - P[i].sum()
    return P


# ---------------------------------------------------------------------------
# GossipNetwork
# ---------------------------------------------------------------------------

@dataclass
class GossipNetwork:
    """Communication graph and gossip matrix for N agents.

    Parameters
    ----------
    adjacency : (N, N) binary adjacency matrix
    eps       : target mixing accuracy for stage-length computation
    """

    adjacency: np.ndarray
    eps: float = 0.01

    def __post_init__(self) -> None:
        self.N = self.adjacency.shape[0]
        self.P = gossip_matrix_metropolis(self.adjacency)
        eigs = np.linalg.eigvalsh(self.P)
        eigs_sorted = np.sort(np.abs(eigs))[::-1]
        self._lambda2 = float(eigs_sorted[1]) if self.N > 1 else 0.0
        self._spectral_gap = 1.0 - self._lambda2

    @property
    def spectral_gap(self) -> float:
        """1 - |lambda_2(P)|."""
        return self._spectral_gap

    @property
    def lambda2(self) -> float:
        """|lambda_2(P)|."""
        return self._lambda2

    @property
    def stage_length(self) -> int:
        """Chebyshev stage length C needed for eps-accurate mixing."""
        if self._lambda2 >= 1.0:
            return 1
        return max(1, int(np.ceil(
            np.log(2 * self.N / self.eps) /
            np.sqrt(2 * np.log(1.0 / max(self._lambda2, 1e-9)))
        )))

    def neighbors(self, i: int) -> List[int]:
        """Indices of neighbors of agent i."""
        return list(np.where(self.adjacency[i] > 0)[0])

    def gossip_weights(self, i: int) -> np.ndarray:
        """P[i, j] for all j ≠ i that are neighbors."""
        nb = self.neighbors(i)
        return self.P[i, nb]

    # ------------------------------------------------------------------
    # Plain mixing step (one round)
    # ------------------------------------------------------------------

    def mix_step(self, values: np.ndarray) -> np.ndarray:
        """Apply one round of gossip: values <- P @ values.

        Parameters
        ----------
        values : (N, d) array — one d-dimensional value per agent
        """
        return self.P @ values

    # ------------------------------------------------------------------
    # Chebyshev-accelerated mixing (one stage of C rounds)
    # ------------------------------------------------------------------

    def chebyshev_mix(self, values: np.ndarray,
                      C: Optional[int] = None) -> np.ndarray:
        """Apply C rounds of Chebyshev-accelerated gossip.

        Implements the recurrence from Lemma 3.1 of Martínez-Rubio et al.:

            y_0 = values / 2
            y_{r+1} = (2/|λ2|) * P @ y_r - (w_{r-1}/w_{r+1}) * y_{r-1}

        where w_r are Chebyshev polynomial coefficients.

        Returns the mixed values (same shape as input).
        """
        if C is None:
            C = self.stage_length
        if C <= 1 or self._lambda2 < 1e-9:
            return self.mix_step(values)

        lam = max(self._lambda2, 1e-9)
        y_prev = np.zeros_like(values, dtype=float)
        y_curr = values.astype(float) / 2.0
        w_prev, w_curr = 0.0, 0.5

        for r in range(C):
            y_new_raw = (2.0 / lam) * (self.P @ y_curr)
            w_new = 2.0 * w_curr / lam - w_prev
            if abs(w_new) < 1e-12:
                break
            coeff = w_prev / w_new if w_new != 0 else 0.0
            y_new = y_new_raw - coeff * y_prev
            y_prev, y_curr = y_curr, y_new
            w_prev, w_curr = w_curr, w_new

        # Rescale: the Chebyshev recurrence includes a factor of 2
        return 2.0 * y_curr

    # ------------------------------------------------------------------
    # Running consensus update (DDUCB style)
    # ------------------------------------------------------------------

    def running_consensus_update(
        self,
        x_mixed: np.ndarray,
        x_new: np.ndarray,
    ) -> np.ndarray:
        """Gossip step that incorporates a new observation vector.

        x_mixed_{t+1} = P @ x_mixed_t + x_new

        This is the running consensus from Eq. (1) of the paper.
        """
        return self.P @ x_mixed + x_new

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"GossipNetwork(N={self.N}, "
            f"spectral_gap={self.spectral_gap:.4f}, "
            f"stage_length={self.stage_length})"
        )
