"""
Canonical end-to-end pipelines for spectral kernel dynamics.

This module wires together:
    channel image -> extracted channel graph -> SpectralGraph ->
    MaxCal spectral dynamics diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .channel_image import (
    ChannelGraphExtraction,
    FlowTopologyAnalysis,
    analyze_channel_network_image,
)
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
class ChannelImageSpectralDiagnostics:
    """Output of the canonical channel-image spectral pipeline."""

    extraction: ChannelGraphExtraction
    flow_topology: FlowTopologyAnalysis
    spectral_graph: SpectralGraph
    dynamics: SpectralKernelDynamics
    fixed_point: FixedPointResult
    stability: StabilityResult
    spectral_entropy_value: float
    residual_inf_norm: float


def _laplacian_from_weighted_edges(
    n_nodes: int,
    edges: Tuple[Tuple[int, int, float], ...],
) -> np.ndarray:
    """Build combinatorial Laplacian from weighted undirected edges."""
    L = np.zeros((n_nodes, n_nodes), dtype=float)
    for u, v, w in edges:
        if u == v:
            continue
        ww = float(max(w, 1e-12))
        L[u, u] += ww
        L[v, v] += ww
        L[u, v] -= ww
        L[v, u] -= ww
    return L


def channel_image_to_spectral_diagnostics(
    image_path: str,
    *,
    sigma2: float = 1.0,
    mu2: float = 1.0,
    eigenvalue_aware_source: bool = True,
    h0: Optional[np.ndarray] = None,
    fp_max_iter: int = 500,
    fp_tol: float = 1e-10,
    source_supply: float = 1.0,
    capacity_width_power: float = 1.25,
    capacity_length_power: float = 1.0,
    min_component_size: int = 32,
    include_dark: bool = False,
    coarsen_bin_px: int = 1,
) -> ChannelImageSpectralDiagnostics:
    """
    Canonical pipeline:
      1) extract channel topology + flow diagnostics from image,
      2) build SpectralGraph from extracted weighted topology,
      3) run fixed-point and stability diagnostics.

    Returns
    -------
    ChannelImageSpectralDiagnostics
        Unified object carrying all intermediate and final diagnostics.
    """
    extraction, flow = analyze_channel_network_image(
        image_path=image_path,
        source_supply=source_supply,
        capacity_width_power=capacity_width_power,
        capacity_length_power=capacity_length_power,
        min_component_size=min_component_size,
        include_dark=include_dark,
        coarsen_bin_px=coarsen_bin_px,
    )

    if flow.n_nodes < 2:
        raise ValueError(
            "Extracted graph has fewer than 2 nodes after cleanup; "
            "cannot build a meaningful SpectralGraph."
        )

    weighted_edges = []
    for e in extraction.edges:
        # Use the same physically motivated proxy used in flow capacity.
        w = (e.mean_width_px ** capacity_width_power) / (
            e.length_px ** capacity_length_power + 1e-9
        )
        weighted_edges.append((e.u, e.v, float(max(w, 1e-12))))

    if not weighted_edges:
        raise ValueError("No edges extracted from image; cannot run spectral diagnostics.")

    # Normalize edge weights for numerical conditioning (median scale = 1).
    w_vals = np.array([w for _, _, w in weighted_edges], dtype=float)
    w_med = float(np.median(w_vals[w_vals > 0])) if np.any(w_vals > 0) else 1.0
    if w_med <= 0:
        w_med = 1.0
    weighted_edges = [(u, v, w / w_med) for (u, v, w) in weighted_edges]

    L = _laplacian_from_weighted_edges(flow.n_nodes, tuple(weighted_edges))
    graph = SpectralGraph(L)

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

    return ChannelImageSpectralDiagnostics(
        extraction=extraction,
        flow_topology=flow,
        spectral_graph=graph,
        dynamics=dyn,
        fixed_point=fp,
        stability=stab,
        spectral_entropy_value=float(spectral_entropy(fp.h_star)),
        residual_inf_norm=residual_inf,
    )
