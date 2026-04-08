"""
kernelcal.spectral
==================
Spectral kernel dynamics on finite graphs via the MaxCal field equation.

Maps to Section 3 of: "Spectral Kernel Dynamics via Maximum Caliber:
Fixed Points, Geodesics, and Phase Transitions" (Das, 2026).

Public API
----------
SpectralGraph            — Laplacian eigendecomposition and graph-filter kernel construction.
GaussianMISource         — Concrete source functional 𝒯_l from Gaussian mutual information.
SpectralKernelDynamics   — Coordinator: ℛ_l, fixed-point iteration, geodesics, stability, diagnostics.
FixedPointResult         — Dataclass returned by fixed_point_iteration().
StabilityResult          — Dataclass returned by stability_analysis().
run_all_experiments      — One-call runner producing all six verification figures.
                           Loaded lazily to avoid import-cycle warnings when the module
                           is run directly (python -m kernelcal.spectral.experiments).

Quick-start
-----------
>>> from kernelcal.spectral import SpectralGraph, GaussianMISource, SpectralKernelDynamics
>>> g = SpectralGraph.path_graph(5)
>>> src = GaussianMISource(sigma2=1.0, mu2=1.0)
>>> dyn = SpectralKernelDynamics(g, src)
>>> result = dyn.fixed_point_iteration()
>>> print(result.converged, result.h_star)

>>> from kernelcal.spectral import run_all_experiments
>>> run_all_experiments(output_dir="figures/spectral")
"""

from .graph import SpectralGraph
from .source import GaussianMISource, CoupledGaussianMISource
from .procedural import (
    ProceduralSpectralDiagnostics,
    procedural_graph_spectral_diagnostics,
)
from .dynamics import (
    SpectralKernelDynamics,
    FixedPointResult,
    StabilityResult,
    geometric_functional,
    vacuum_solution,
    geodesic,
    spectral_entropy,
    hessian_matrix,
    hessian_gap,
    fiedler_gap,
    coupling_entropy,
    contraction_bound,
    field_equation_residual,
)

# Lazy import: avoids RuntimeWarning when experiments.py is run as __main__
# while it is also reachable via the package.
def __getattr__(name: str):
    if name == "run_all_experiments":
        from .experiments import run_all_experiments
        return run_all_experiments
    if name == "run_procedural_examples":
        from .procedural_examples import run_procedural_examples
        return run_procedural_examples
    # Dormant image pipeline: still reachable explicitly, but not part of
    # the default public API while procedural examples are preferred.
    if name in {
        "ChannelEdge",
        "ChannelGraphExtraction",
        "FlowTopologyAnalysis",
        "ChannelVerificationArtifacts",
        "extract_channel_graph_from_image",
        "analyze_channel_network_image",
        "save_channel_extraction_verification",
    }:
        from . import channel_image as _ch
        return getattr(_ch, name)
    if name in {
        "ChannelImageSpectralDiagnostics",
        "channel_image_to_spectral_diagnostics",
    }:
        from . import pipeline as _pl
        return getattr(_pl, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # core classes
    "SpectralGraph",
    "GaussianMISource",
    "CoupledGaussianMISource",
    "ProceduralSpectralDiagnostics",
    "SpectralKernelDynamics",
    # result dataclasses
    "FixedPointResult",
    "StabilityResult",
    # pure functions (usable without the coordinator class)
    "geometric_functional",
    "vacuum_solution",
    "geodesic",
    "spectral_entropy",
    "hessian_matrix",
    "hessian_gap",
    "fiedler_gap",
    "coupling_entropy",
    "contraction_bound",
    "field_equation_residual",
    "procedural_graph_spectral_diagnostics",
    # experiment runner (lazy)
    "run_all_experiments",
    "run_procedural_examples",
]
