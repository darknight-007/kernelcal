"""
kernelcal.fluid
===============
Discrete fluid-dynamics-inspired models for distributions over kernel space.

Main entry point:
    run_twenty_node_experiment()
"""

from .dynamics import (
    FluidGraph,
    PotentialLandscape,
    FluidSimulationConfig,
    FluidSimulationResult,
    ring_distance,
    gaussian_bump_on_ring,
    make_twenty_node_reference_landscape,
    simulate_kernel_fluid,
    save_timeseries_csv,
)
from .sparse import (
    SparseFluidGraph,
    continuity_drho,
    edge_flux,
    edge_gradient,
    edge_laplacian_smoothing,
    node_signed_inflow,
    simulate_kernel_fluid_sparse,
)


def __getattr__(name: str):
    if name == "run_twenty_node_experiment":
        from .experiments import run_twenty_node_experiment
        return run_twenty_node_experiment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "FluidGraph",
    "PotentialLandscape",
    "FluidSimulationConfig",
    "FluidSimulationResult",
    "ring_distance",
    "gaussian_bump_on_ring",
    "make_twenty_node_reference_landscape",
    "simulate_kernel_fluid",
    "save_timeseries_csv",
    "run_twenty_node_experiment",
    # PR-A.0 sparse-Laplacian solver
    "SparseFluidGraph",
    "continuity_drho",
    "edge_flux",
    "edge_gradient",
    "edge_laplacian_smoothing",
    "node_signed_inflow",
    "simulate_kernel_fluid_sparse",
]

