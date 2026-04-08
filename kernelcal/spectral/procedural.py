"""
Procedural graph generation + spectral diagnostics.

This module provides a stable, image-free path for reproducible examples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .graph import SpectralGraph
from .source import GaussianMISource
from .dynamics import (
    SpectralKernelDynamics,
    FixedPointResult,
    StabilityResult,
    spectral_entropy,
    field_equation_residual,
)


@dataclass(frozen=True)
class ProceduralSpectralDiagnostics:
    """End-to-end diagnostics on a procedurally generated graph."""

    graph: SpectralGraph
    dynamics: SpectralKernelDynamics
    fixed_point: FixedPointResult
    stability: StabilityResult
    spectral_entropy_value: float
    residual_inf_norm: float


def procedural_graph_spectral_diagnostics(
    *,
    graph_kind: str = "path",
    N: int = 8,
    sigma2: float = 1.0,
    mu2: float = 1.0,
    eigenvalue_aware_source: bool = True,
    weak_edge_i: Optional[int] = None,
    weak_edge_j: Optional[int] = None,
    weak_edge_epsilon: float = 0.25,
    h0: Optional[np.ndarray] = None,
    fp_max_iter: int = 500,
    fp_tol: float = 1e-10,
) -> ProceduralSpectralDiagnostics:
    """
    Canonical procedural pipeline:
      graph factory -> SpectralGraph -> fixed-point + stability diagnostics.

    Parameters
    ----------
    graph_kind : {"path", "cycle", "weak_path"}
        Graph family to instantiate.
    N : int
        Number of nodes.
    weak_edge_i, weak_edge_j, weak_edge_epsilon :
        Used only when graph_kind="weak_path".
    """
    kind = graph_kind.strip().lower()
    if kind == "path":
        graph = SpectralGraph.path_graph(N)
    elif kind == "cycle":
        graph = SpectralGraph.cycle_graph(N)
    elif kind == "weak_path":
        if weak_edge_i is None or weak_edge_j is None:
            if N < 4:
                raise ValueError("weak_path requires N >= 4.")
            weak_edge_i, weak_edge_j = (N // 2 - 1, N // 2)
        graph = SpectralGraph.path_graph_with_weak_edge(
            N, int(weak_edge_i), int(weak_edge_j), float(weak_edge_epsilon)
        )
    else:
        raise ValueError(
            f"Unsupported graph_kind={graph_kind!r}. Use 'path', 'cycle', or 'weak_path'."
        )

    src = GaussianMISource(
        sigma2=sigma2,
        mu2=mu2,
        eigenvalues=(graph.eigenvalues if eigenvalue_aware_source else None),
    )
    dyn = SpectralKernelDynamics(graph=graph, source=src, h0=h0)
    fp = dyn.fixed_point_iteration(max_iter=fp_max_iter, tol=fp_tol)
    stab = dyn.stability_analysis(fp.h_star)

    T_vals = src.T(fp.h_star)
    residual = field_equation_residual(fp.h_star, dyn.h0, T_vals)
    residual_inf = float(np.max(np.abs(residual)))

    return ProceduralSpectralDiagnostics(
        graph=graph,
        dynamics=dyn,
        fixed_point=fp,
        stability=stab,
        spectral_entropy_value=float(spectral_entropy(fp.h_star)),
        residual_inf_norm=residual_inf,
    )
