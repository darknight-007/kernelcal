"""kernelcal.terrain — Planetary terrain analysis and topological biosignature detection.

Submodules
----------
dem          DEM processing: graph construction, D8 flow routing, slope/curvature
craters      Crater rim detection, Betti numbers, abiotic null model
channels     Drainage network graphs, Strahler ordering, triple spectral diagnostic
biosig       Topological biosignature (Δβ₁), cross-kernel test, plume entropy
diagnostics  Spectral entropy, fixed-point kernel, phase-transition sweep

Quick-start
-----------
>>> import numpy as np
>>> from kernelcal.terrain.dem import synthetic_crater_dem, dem_to_graph
>>> from kernelcal.terrain.craters import crater_betti_numbers, abiotic_beta1_craters
>>> from kernelcal.terrain.channels import drainage_network_graph, triple_spectral_diagnostic
>>> from kernelcal.terrain.biosig import topological_biosignature, detection_threshold
>>> from kernelcal.terrain.diagnostics import fixed_point_kernel, spectral_entropy

Paper references
----------------
P1 : Das (2026) "Spectral Kernel Dynamics via Maximum Caliber"
P2 : Das (2026) "Spectral Kernel Dynamics for Planetary Twins"
P3 : Das (2026) "Spectral Kernel Dynamics for Terrestrial Environmental Networks"
P4 : Das (2026) "Spectral Kernel Dynamics as a Biosignature Framework"
"""

from .dem        import (dem_to_graph, terrain_graph_laplacian,
                          dem_to_point_cloud,
                          slope, curvature_planform, curvature_profile,
                          d8_flow_direction, flow_accumulation, channel_mask,
                          synthetic_crater_dem, synthetic_channel_dem,
                          TerrainGraph)
from .craters    import (detect_craters, crater_rim_mask, crater_rim_graph,
                          crater_betti_numbers, abiotic_beta1_craters,
                          crater_spectral_signature, CraterCandidate)
from .channels   import (drainage_network_graph, drainage_graph_laplacian,
                          triple_spectral_diagnostic, curl_energy,
                          abiotic_beta1_channels, topology_budget,
                          DrainageGraph, ChannelDiagnostic)
from .biosig     import (topological_biosignature, detection_threshold,
                          cross_kernel, cross_kernel_norm, factorization_test,
                          spectral_kernel_from_laplacian,
                          chemical_affinity_graph, plume_spectral_entropy,
                          BiosignatureReport, TopologicalBiosignature)
from .diagnostics import (spectral_entropy, spectral_entropy_from_laplacian,
                           fixed_point_kernel, fiedler_mode_gap,
                           stability_conservation_tradeoff,
                           phase_transition_sweep, observability_ratio,
                           bandwidth_optimal_modes,
                           PhaseSweepResult)

# Convenience alias: dem_to_flow_graph may not exist yet; guard gracefully
try:
    from .dem import dem_to_flow_graph
except ImportError:
    pass

__all__ = [
    # dem
    "dem_to_graph", "terrain_graph_laplacian", "dem_to_point_cloud",
    "slope", "curvature_planform", "curvature_profile",
    "d8_flow_direction", "flow_accumulation", "channel_mask",
    "synthetic_crater_dem", "synthetic_channel_dem", "TerrainGraph",
    # craters
    "detect_craters", "crater_rim_mask", "crater_rim_graph",
    "crater_betti_numbers", "abiotic_beta1_craters",
    "crater_spectral_signature", "CraterCandidate",
    # channels
    "drainage_network_graph", "drainage_graph_laplacian",
    "triple_spectral_diagnostic", "curl_energy",
    "abiotic_beta1_channels", "topology_budget",
    "DrainageGraph", "ChannelDiagnostic",
    # biosig
    "topological_biosignature", "detection_threshold",
    "cross_kernel", "cross_kernel_norm", "factorization_test",
    "spectral_kernel_from_laplacian",
    "chemical_affinity_graph", "plume_spectral_entropy",
    "BiosignatureReport", "TopologicalBiosignature",
    # diagnostics
    "spectral_entropy", "spectral_entropy_from_laplacian",
    "fixed_point_kernel", "fiedler_mode_gap",
    "stability_conservation_tradeoff",
    "phase_transition_sweep", "observability_ratio",
    "bandwidth_optimal_modes", "PhaseSweepResult",
]
