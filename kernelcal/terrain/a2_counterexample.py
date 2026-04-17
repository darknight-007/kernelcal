"""Worked A2 counterexample on short-cycle graphs.

This module provides a small, reproducible simulation for the manuscript caveat:
the topology floor k_min = beta0 + beta1 can fail to preserve cycle information
when the L0 low-frequency subspace does not contain a beta1-dimensional
cycle-carrying subspace (Assumption A2 in the paper).

The implementation here is intentionally minimal and deterministic:

1) Long-cycle control case:
   Two separated 4-cycles connected by a bridge path.
   Empirically, the first k_min L0 modes retain two independent cycle signals.

2) Short-cycle counterexample:
   A figure-eight graph formed by two short triangles sharing one articulation.
   At the same k_min = beta0 + beta1, projected cycle signals collapse to rank 1.

This is a calibration experiment, not a proof of necessity/sufficiency.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json

import numpy as np


@dataclass(frozen=True)
class A2CaseResult:
    """Per-graph result for the A2 proxy check."""

    name: str
    n_vertices: int
    n_edges: int
    beta0: int
    beta1: int
    k_min: int
    projected_rank: int
    singular_values: np.ndarray
    sigma_min: float
    spectral_gap_ratio: float
    a2_proxy_holds: bool
    # Parametric quantities from the A2-failure criterion (seed note).
    ell_min: int                 # shortest cycle length
    ell_max: int                 # longest cycle length
    gamma: float                 # cycle-length ratio ell_max / ell_min
    delta_k: float               # absolute spectral gap lambda_{k+1} - lambda_k
    rho_k: float                 # normalized cycle-subspace fidelity in [0, 1]
    bound_proxy: float           # 1 / (ell_min^2 * delta_k), the dominant term


@dataclass(frozen=True)
class A2CounterexampleResult:
    """Paired control/counterexample result."""

    long_cycle_case: A2CaseResult
    short_cycle_case: A2CaseResult

    @property
    def counterexample_confirmed(self) -> bool:
        """True when long-cycle passes and short-cycle fails under same floor."""
        return (
            self.long_cycle_case.a2_proxy_holds
            and (not self.short_cycle_case.a2_proxy_holds)
            and self.long_cycle_case.k_min == self.short_cycle_case.k_min
        )


@dataclass(frozen=True)
class A2SweepPoint:
    """One sampled point in a cycle-ratio sweep."""

    family: str
    short_cycle_len: int
    long_cycle_len: int
    length_ratio: float
    bridge_len: int
    beta1: int
    k_min: int
    projected_rank: int
    sigma_min: float
    spectral_gap_ratio: float
    a2_proxy_holds: bool
    # Parametric criterion quantities.
    ell_min: int
    ell_max: int
    gamma: float
    delta_k: float
    rho_k: float
    bound_proxy: float
    # Augmentation recovery (k += augment_delta_k).
    augment_delta_k: int
    rho_after_augment: float
    projected_rank_after_augment: int


@dataclass(frozen=True)
class A2SweepResult:
    """Aggregated results for the A2 cycle-ratio sweep."""

    points: list[A2SweepPoint]

    def rows(self) -> list[dict]:
        """Return JSON/CSV-friendly rows."""
        out: list[dict] = []
        for p in self.points:
            out.append(
                {
                    "family": p.family,
                    "short_cycle_len": p.short_cycle_len,
                    "long_cycle_len": p.long_cycle_len,
                    "length_ratio": p.length_ratio,
                    "bridge_len": p.bridge_len,
                    "beta1": p.beta1,
                    "k_min": p.k_min,
                    "projected_rank": p.projected_rank,
                    "sigma_min": p.sigma_min,
                    "spectral_gap_ratio": p.spectral_gap_ratio,
                    "ell_min": p.ell_min,
                    "ell_max": p.ell_max,
                    "gamma": p.gamma,
                    "delta_k": p.delta_k,
                    "rho_k": p.rho_k,
                    "bound_proxy": p.bound_proxy,
                    "augment_delta_k": p.augment_delta_k,
                    "rho_after_augment": p.rho_after_augment,
                    "projected_rank_after_augment": p.projected_rank_after_augment,
                    "a2_proxy_holds": p.a2_proxy_holds,
                    "a2_proxy_fails": (not p.a2_proxy_holds),
                }
            )
        return out


def _laplacian_from_edges(n_vertices: int, edges: list[tuple[int, int]]) -> np.ndarray:
    A = np.zeros((n_vertices, n_vertices), dtype=float)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    D = np.diag(A.sum(axis=1))
    return D - A


def _cycle_indicator_matrix(
    n_vertices: int,
    cycles: list[set[int]],
) -> np.ndarray:
    """Node-space cycle proxy signals (demeaned indicators)."""
    cols = []
    for cyc in cycles:
        x = np.zeros(n_vertices, dtype=float)
        x[list(cyc)] = 1.0
        x -= x.mean()
        cols.append(x)
    return np.stack(cols, axis=1)


def _normalized_cycle_columns(C: np.ndarray) -> np.ndarray:
    """Return C with each column normalized to unit 2-norm (zero columns unchanged)."""
    norms = np.linalg.norm(C, axis=0)
    norms_safe = np.where(norms > 1e-12, norms, 1.0)
    return C / norms_safe


def _rho_and_rank(
    Phi_k: np.ndarray,
    C_norm: np.ndarray,
    tol: float,
) -> tuple[float, int, np.ndarray]:
    """Compute rho = sigma_min(Phi_k^T C_norm), rank, and singular values."""
    M = Phi_k.T @ C_norm
    svals = np.linalg.svd(M, compute_uv=False)
    rho = float(svals[-1]) if svals.size else 0.0
    rank = int(np.sum(svals > tol))
    return rho, rank, svals


def _evaluate_case(
    *,
    name: str,
    n_vertices: int,
    edges: list[tuple[int, int]],
    cycles: list[set[int]],
    tol: float,
    augment_delta_k: int = 0,
) -> A2CaseResult:
    L = _laplacian_from_edges(n_vertices, edges)
    eigvals, eigvecs = np.linalg.eigh(L)

    beta0 = 1  # all fixtures here are connected
    beta1 = len(edges) - n_vertices + beta0
    k_min = beta0 + beta1

    # Raw (unnormalized) projection for backward-compatible sigma_min.
    Phi_k = eigvecs[:, :k_min]
    C = _cycle_indicator_matrix(n_vertices, cycles)
    C_proj = Phi_k @ (Phi_k.T @ C)
    singular_values = np.linalg.svd(C_proj, compute_uv=False)
    sigma_min = float(singular_values[-1]) if singular_values.size else 0.0
    projected_rank = int(np.sum(singular_values > tol))
    a2_proxy_holds = projected_rank >= beta1

    # Normalized cycle fidelity rho(k) in [0, 1].
    C_norm = _normalized_cycle_columns(C)
    rho_k, _, _ = _rho_and_rank(Phi_k, C_norm, tol)

    # Spectral gap (absolute) and ratio.
    if k_min < n_vertices - 1:
        lam_k = float(eigvals[k_min])
        lam_kp1 = float(eigvals[k_min + 1])
        delta_k_abs = max(lam_kp1 - lam_k, 0.0)
        denom = max(lam_kp1, 1e-12)
        spectral_gap_ratio = delta_k_abs / denom
    else:
        delta_k_abs = 0.0
        spectral_gap_ratio = 0.0

    # Cycle lengths (from the provided cycles list).
    cycle_lengths = [len(c) for c in cycles] or [0]
    ell_min = int(min(cycle_lengths))
    ell_max = int(max(cycle_lengths))
    gamma = float(ell_max) / float(ell_min) if ell_min > 0 else 0.0

    # Dominant bound term from the seed proposition.
    denom_bound = max(ell_min * ell_min * max(delta_k_abs, 1e-12), 1e-12)
    bound_proxy = 1.0 / denom_bound

    return A2CaseResult(
        name=name,
        n_vertices=n_vertices,
        n_edges=len(edges),
        beta0=beta0,
        beta1=beta1,
        k_min=k_min,
        projected_rank=projected_rank,
        singular_values=singular_values,
        sigma_min=sigma_min,
        spectral_gap_ratio=float(spectral_gap_ratio),
        a2_proxy_holds=a2_proxy_holds,
        ell_min=ell_min,
        ell_max=ell_max,
        gamma=float(gamma),
        delta_k=float(delta_k_abs),
        rho_k=float(rho_k),
        bound_proxy=float(bound_proxy),
    )


def rho_with_augmentation(
    *,
    n_vertices: int,
    edges: list[tuple[int, int]],
    cycles: list[set[int]],
    augment_delta_k: int,
    tol: float = 1e-6,
) -> tuple[float, int]:
    """Recompute normalized rho and rank when retaining k_min + augment_delta_k modes."""
    L = _laplacian_from_edges(n_vertices, edges)
    eigvals, eigvecs = np.linalg.eigh(L)
    del eigvals

    beta0 = 1
    beta1 = len(edges) - n_vertices + beta0
    k = beta0 + beta1 + max(0, int(augment_delta_k))
    k = min(k, n_vertices)

    Phi_k = eigvecs[:, :k]
    C = _cycle_indicator_matrix(n_vertices, cycles)
    C_norm = _normalized_cycle_columns(C)
    rho, rank, _ = _rho_and_rank(Phi_k, C_norm, tol)
    return rho, rank


def _make_figure8_cycle_graph(
    short_cycle_len: int,
    long_cycle_len: int,
) -> tuple[int, list[tuple[int, int]], list[set[int]]]:
    """Two cycles sharing one articulation node (interleaved short-cycle family)."""
    if short_cycle_len < 3 or long_cycle_len < 3:
        raise ValueError("Cycle lengths must be >= 3.")

    # Shared articulation vertex 0.
    short_nodes = [0] + list(range(1, short_cycle_len))
    long_start = short_cycle_len
    long_nodes = [0] + list(range(long_start, long_start + long_cycle_len - 1))

    edges: set[tuple[int, int]] = set()

    def add_cycle(nodes: list[int]) -> None:
        m = len(nodes)
        for i in range(m):
            a = nodes[i]
            b = nodes[(i + 1) % m]
            edges.add((min(a, b), max(a, b)))

    add_cycle(short_nodes)
    add_cycle(long_nodes)

    n_vertices = short_cycle_len + long_cycle_len - 1
    cycles = [set(short_nodes), set(long_nodes)]
    return n_vertices, sorted(edges), cycles


def _make_separated_cycle_graph(
    short_cycle_len: int,
    long_cycle_len: int,
    bridge_len: int,
) -> tuple[int, list[tuple[int, int]], list[set[int]]]:
    """Two disjoint cycles connected by a bridge path (control family)."""
    if short_cycle_len < 3 or long_cycle_len < 3:
        raise ValueError("Cycle lengths must be >= 3.")
    if bridge_len < 1:
        raise ValueError("bridge_len must be >= 1.")

    c1 = list(range(short_cycle_len))
    c2_start = short_cycle_len
    c2 = list(range(c2_start, c2_start + long_cycle_len))

    edges: set[tuple[int, int]] = set()

    def add_cycle(nodes: list[int]) -> None:
        m = len(nodes)
        for i in range(m):
            a = nodes[i]
            b = nodes[(i + 1) % m]
            edges.add((min(a, b), max(a, b)))

    add_cycle(c1)
    add_cycle(c2)

    # Bridge from c1[0] to c2[0] with bridge_len edges.
    # bridge_len=1 means direct edge.
    prev = c1[0]
    next_free = c2_start + long_cycle_len
    for _ in range(bridge_len - 1):
        mid = next_free
        next_free += 1
        edges.add((min(prev, mid), max(prev, mid)))
        prev = mid
    edges.add((min(prev, c2[0]), max(prev, c2[0])))

    n_vertices = next_free
    cycles = [set(c1), set(c2)]
    return n_vertices, sorted(edges), cycles


def run_worked_a2_counterexample(tol: float = 1e-6) -> A2CounterexampleResult:
    """Run long-cycle control and short-cycle A2-failure counterexample.

    Returns
    -------
    A2CounterexampleResult
        Includes both case results and a boolean summary
        ``counterexample_confirmed``.
    """
    # Control: two separated loops + bridge path (beta1 = 2, k_min = 3)
    long_edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # first 4-cycle
        (4, 5), (5, 6), (6, 7), (7, 4),  # second 4-cycle
        (3, 8), (8, 9), (9, 4),           # bridge path
    ]
    long_cycles = [{0, 1, 2, 3}, {4, 5, 6, 7}]

    # Counterexample: figure-eight of two short triangles sharing one node
    # (beta1 = 2, k_min = 3)
    short_edges = [
        (0, 1), (1, 2), (2, 0),  # left triangle
        (2, 3), (3, 4), (4, 2),  # right triangle
    ]
    short_cycles = [{0, 1, 2}, {2, 3, 4}]

    long_case = _evaluate_case(
        name="long_cycle_control",
        n_vertices=10,
        edges=long_edges,
        cycles=long_cycles,
        tol=tol,
    )
    short_case = _evaluate_case(
        name="short_cycle_figure8_counterexample",
        n_vertices=5,
        edges=short_edges,
        cycles=short_cycles,
        tol=tol,
    )
    return A2CounterexampleResult(
        long_cycle_case=long_case,
        short_cycle_case=short_case,
    )


def _sweep_point_from_case(
    *,
    family: str,
    short_len: int,
    long_len: int,
    bridge_len: int,
    ratio: float,
    n_vertices: int,
    edges: list[tuple[int, int]],
    cycles: list[set[int]],
    case: A2CaseResult,
    augment_delta_k: int,
    tol: float,
) -> A2SweepPoint:
    rho_aug, rank_aug = rho_with_augmentation(
        n_vertices=n_vertices,
        edges=edges,
        cycles=cycles,
        augment_delta_k=augment_delta_k,
        tol=tol,
    )
    return A2SweepPoint(
        family=family,
        short_cycle_len=short_len,
        long_cycle_len=long_len,
        length_ratio=ratio,
        bridge_len=bridge_len,
        beta1=case.beta1,
        k_min=case.k_min,
        projected_rank=case.projected_rank,
        sigma_min=case.sigma_min,
        spectral_gap_ratio=case.spectral_gap_ratio,
        a2_proxy_holds=case.a2_proxy_holds,
        ell_min=case.ell_min,
        ell_max=case.ell_max,
        gamma=case.gamma,
        delta_k=case.delta_k,
        rho_k=case.rho_k,
        bound_proxy=case.bound_proxy,
        augment_delta_k=int(augment_delta_k),
        rho_after_augment=float(rho_aug),
        projected_rank_after_augment=int(rank_aug),
    )


def run_a2_cycle_ratio_sweep(
    *,
    short_cycle_lengths: tuple[int, ...] = (3, 4, 5),
    long_cycle_lengths: tuple[int, ...] = (3, 4, 5, 6, 8, 10, 12),
    bridge_len: int = 2,
    tol: float = 1e-6,
    augment_delta_k: int = 2,
) -> A2SweepResult:
    """Sweep A2 proxy over cycle-length ratios and graph families.

    For each pair (short_len, long_len) with short_len <= long_len, we evaluate:
      - figure-eight family (short cycles spectrally interleaved),
      - separated-cycle control family (same lengths, bridge-connected).

    The parametric quantities (ell_min, gamma, delta_k, rho_k, bound_proxy)
    and the augmentation recovery (rho after retaining k_min + augment_delta_k
    modes) are recorded for each point.
    """
    points: list[A2SweepPoint] = []

    for s in short_cycle_lengths:
        for l in long_cycle_lengths:
            if s > l:
                continue
            ratio = float(s) / float(l)

            n_v_f8, e_f8, cyc_f8 = _make_figure8_cycle_graph(s, l)
            r_f8 = _evaluate_case(
                name=f"figure8_s{s}_l{l}",
                n_vertices=n_v_f8,
                edges=e_f8,
                cycles=cyc_f8,
                tol=tol,
            )
            points.append(
                _sweep_point_from_case(
                    family="figure8",
                    short_len=s,
                    long_len=l,
                    bridge_len=0,
                    ratio=ratio,
                    n_vertices=n_v_f8,
                    edges=e_f8,
                    cycles=cyc_f8,
                    case=r_f8,
                    augment_delta_k=augment_delta_k,
                    tol=tol,
                )
            )

            n_v_ctl, e_ctl, cyc_ctl = _make_separated_cycle_graph(s, l, bridge_len)
            r_ctl = _evaluate_case(
                name=f"control_s{s}_l{l}_b{bridge_len}",
                n_vertices=n_v_ctl,
                edges=e_ctl,
                cycles=cyc_ctl,
                tol=tol,
            )
            points.append(
                _sweep_point_from_case(
                    family="separated_control",
                    short_len=s,
                    long_len=l,
                    bridge_len=bridge_len,
                    ratio=ratio,
                    n_vertices=n_v_ctl,
                    edges=e_ctl,
                    cycles=cyc_ctl,
                    case=r_ctl,
                    augment_delta_k=augment_delta_k,
                    tol=tol,
                )
            )

    return A2SweepResult(points=points)


def fit_bound_constants(
    result: A2SweepResult,
    *,
    family: str = "separated_control",
) -> dict:
    """Empirically fit (C1, C2) in rho >= 1 - C1*bound_proxy - C2*(gamma - 1).

    Uses least squares on the rows of ``family``. Returns fitted constants
    and residual statistics; meant as a numerically-observed sanity check
    on the parametric A2-failure criterion, not a theoretical constant.
    """
    rows = [p for p in result.points if p.family == family]
    if len(rows) < 3:
        return {
            "n_points": len(rows),
            "C1": float("nan"),
            "C2": float("nan"),
            "rmse": float("nan"),
            "family": family,
        }

    # y = 1 - rho, predictors = [bound_proxy, gamma - 1]
    y = np.array([1.0 - p.rho_k for p in rows], dtype=float)
    X = np.array(
        [[p.bound_proxy, p.gamma - 1.0] for p in rows], dtype=float
    )
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    return {
        "family": family,
        "n_points": len(rows),
        "C1": float(coef[0]),
        "C2": float(coef[1]),
        "rmse": rmse,
    }


def write_a2_sweep_json(result: A2SweepResult, path: str | Path) -> None:
    """Write sweep rows to JSON."""
    p = Path(path)
    p.write_text(json.dumps(result.rows(), indent=2))


def write_a2_sweep_csv(result: A2SweepResult, path: str | Path) -> None:
    """Write sweep rows to CSV."""
    rows = result.rows()
    p = Path(path)
    if not rows:
        p.write_text("")
        return
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

