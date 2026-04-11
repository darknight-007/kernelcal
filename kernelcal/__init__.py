"""
kernelcal
=========
Python library implementing Kernel Dynamics under Maximum Caliber (MaxCal),
with integration hooks for DeepGIS-XR.  Each subpackage corresponds to one
of the seven integration threads documented in deepgis_maxcal_integration.md.

Companion to: "Kernel Dynamics under Path Entropy Maximization" (Das, ASU)
Repository:   https://github.com/darknight-007/kernelcal

Subpackages
-----------
kernel
    Hilbert-Schmidt geometry on kernel space K, kernel trajectories, and
    fixed-point detection.  (Paper §3-4)

maxcal
    Core MaxCal functional (path entropy, Lagrange dual, fitting) and the
    MaxCalSampler drop-in for the DeepGIS World Sampler.  (Paper §2, §5)

ntk
    NTK tracker for monitoring representational drift during fine-tuning,
    and the Hellinger kernel / NTK–Hellinger comparison (Conjecture 3).

assembly
    RKHS complexity metrics, per-tile complexity maps, and the assembly-
    theory-motivated reward signal for the World Sampler.  (Paper §6)

thermodynamics
    Landauer bound checks, GPU power monitoring, and thermodynamic
    efficiency accumulation.  (Paper Theorem 1)

models
    MaxCal multi-model kernel selector over the five DeepGIS AI backends.

prompts
    Self-consistent prompt iteration for Grounding DINO.

fluid
    Discrete kernel-fluid dynamics on graph-structured kernel space, including
    a runnable 20-node two-phase experiment with diagnostics.

spectral
    Spectral kernel dynamics on finite graphs: SpectralGraph, GaussianMISource,
    SpectralKernelDynamics, and a six-experiment verification suite that makes
    every Section 3 claim of the companion paper numerically visible.
    (Paper §3: Propositions 1–3, Corollaries 1–3, Remarks 4, 8, Q6)

Quick-start
-----------
>>> from kernelcal.maxcal import MaxCalSampler
>>> from kernelcal.models import ModelKernelSelector
>>> from kernelcal.thermodynamics import PowerMonitor
>>> from kernelcal.ntk import NTKTracker, compare_ntk_to_hellinger
>>> from kernelcal.assembly import complexity_map, assembly_reward_signal
>>> from kernelcal.prompts import PromptKernelIterator
>>> from kernelcal.kernel import KernelTrajectory, FixedPointDetector
"""

from .kernel import (
    hilbert_schmidt_distance,
    hilbert_schmidt_norm,
    is_psd,
    project_to_psd,
    KernelTrajectory,
    FixedPointDetector,
)
from .maxcal import (
    path_entropy,
    fit_lagrange_multipliers,
    MaxCalSampler,
)
from .ntk import (
    NTKTracker,
    compute_empirical_ntk,
    hellinger_kernel_matrix,
    compare_ntk_to_hellinger,
)
from .assembly import (
    rkhs_norm,
    spectral_complexity,
    complexity_map,
    assembly_reward_signal,
    assembly_index_lower_bound,
)
from .thermodynamics import (
    landauer_bound,
    kernel_mutual_information_change,
    check_landauer_bound,
    PowerMonitor,
    ThermodynamicEfficiency,
)
try:
    from .models import ModelKernelSelector, ModelRecord
except ImportError:
    pass
try:
    from .prompts import PromptKernelIterator, prompt_kernel_distance
except ImportError:
    pass
try:
    from .navigation import (
        SemanticSLAMKernelTracker,
        InformativePathPlanner,
        HumanPilotDemonstrationLearner,
    )
except ImportError:
    pass
from .attention import (
    AttentionKernel,
    AttentionKernelResult,
    AttentionKernelTracker,
    run_attention_experiment,
    AttentionExperimentResult,
)
from .spectral import (
    SpectralGraph,
    GaussianMISource,
    CoupledGaussianMISource,
    ProceduralSpectralDiagnostics,
    procedural_graph_spectral_diagnostics,
    SpectralKernelDynamics,
    FixedPointResult,
    StabilityResult,
    spectral_entropy,
    hessian_gap,
    coupling_entropy,
)
from .video import (
    CompressedDepthFrame,
    DepthStreamCodec,
    DepthStreamConfig,
    depth_image_to_xyz,
    pointcloud2_to_xyz,
)
from .geo3d import (
    CompressedMeshGeometry,
    CompressedSpectralKernel,
    CompressionBounds,
    HodgeSpectralBasis,
    PersistencePair,
    PersistenceResult,
    TemporalKernelSummary,
    adjacency_to_laplacian,
    betti_numbers,
    build_hodge_basis,
    compress_dae,
    compress_mesh_geometry,
    compress_mesh_roundtrip,
    compress_point_cloud,
    compress_temporal_clouds,
    compression_ratio_formula,
    compression_ratio_vs_modes,
    decompress_dae,
    decompress_mesh_roundtrip,
    decompress_to_kernel,
    distortion_from_eigenvalues,
    estimate_compression_bounds,
    hodge_decompose,
    knn_symmetric_adjacency,
    mesh_combinatorial_laplacian,
    mesh_persistence,
    mode_count_for_distortion,
    mode_count_for_topology,
    subsample_points,
    vietoris_rips_persistence,
)

__version__ = "0.4.0"

__all__ = [
    # kernel
    "hilbert_schmidt_distance",
    "hilbert_schmidt_norm",
    "is_psd",
    "project_to_psd",
    "KernelTrajectory",
    "FixedPointDetector",
    # maxcal
    "path_entropy",
    "fit_lagrange_multipliers",
    "MaxCalSampler",
    # ntk
    "NTKTracker",
    "compute_empirical_ntk",
    "hellinger_kernel_matrix",
    "compare_ntk_to_hellinger",
    # assembly
    "rkhs_norm",
    "spectral_complexity",
    "complexity_map",
    "assembly_reward_signal",
    "assembly_index_lower_bound",
    # thermodynamics
    "landauer_bound",
    "kernel_mutual_information_change",
    "check_landauer_bound",
    "PowerMonitor",
    "ThermodynamicEfficiency",
    # models
    "ModelKernelSelector",
    "ModelRecord",
    # prompts
    "PromptKernelIterator",
    "prompt_kernel_distance",
    # navigation
    "SemanticSLAMKernelTracker",
    "InformativePathPlanner",
    "HumanPilotDemonstrationLearner",
    # attention
    "AttentionKernel",
    "AttentionKernelResult",
    "AttentionKernelTracker",
    "run_attention_experiment",
    "AttentionExperimentResult",
    # fluid
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
    # spectral
    "SpectralGraph",
    "GaussianMISource",
    "CoupledGaussianMISource",
    "ProceduralSpectralDiagnostics",
    "procedural_graph_spectral_diagnostics",
    "SpectralKernelDynamics",
    "FixedPointResult",
    "StabilityResult",
    "spectral_entropy",
    "hessian_gap",
    "coupling_entropy",
    # video
    "CompressedDepthFrame",
    "DepthStreamCodec",
    "DepthStreamConfig",
    "depth_image_to_xyz",
    "pointcloud2_to_xyz",
    # geo3d
    "subsample_points",
    "knn_symmetric_adjacency",
    "adjacency_to_laplacian",
    "CompressedSpectralKernel",
    "CompressedMeshGeometry",
    "CompressedSpectralKernel",
    "CompressionBounds",
    "HodgeSpectralBasis",
    "PersistencePair",
    "PersistenceResult",
    "compress_point_cloud",
    "decompress_to_kernel",
    "mesh_combinatorial_laplacian",
    "compress_mesh_geometry",
    "compress_mesh_roundtrip",
    "decompress_mesh_roundtrip",
    "compress_dae",
    "decompress_dae",
    "TemporalKernelSummary",
    "compress_temporal_clouds",
    # hodge
    "betti_numbers",
    "build_hodge_basis",
    "hodge_decompose",
    # topology
    "mesh_persistence",
    "vietoris_rips_persistence",
    # bounds
    "compression_ratio_formula",
    "compression_ratio_vs_modes",
    "distortion_from_eigenvalues",
    "estimate_compression_bounds",
    "mode_count_for_topology",
    "mode_count_for_distortion",
    "run_all_experiments",  # lazy via kernelcal.spectral.__getattr__
    "run_procedural_examples",  # lazy via kernelcal.spectral.__getattr__
]


def __getattr__(name: str):
    if name == "run_all_experiments":
        from .spectral.experiments import run_all_experiments
        return run_all_experiments
    if name == "run_procedural_examples":
        from .spectral.procedural_examples import run_procedural_examples
        return run_procedural_examples
    if name == "run_twenty_node_experiment":
        from .fluid.experiments import run_twenty_node_experiment
        return run_twenty_node_experiment
    # Backward-compatible access to dormant image pipeline symbols.
    if name in {
        "ChannelEdge",
        "ChannelGraphExtraction",
        "FlowTopologyAnalysis",
        "ChannelVerificationArtifacts",
        "ChannelImageSpectralDiagnostics",
        "extract_channel_graph_from_image",
        "analyze_channel_network_image",
        "save_channel_extraction_verification",
        "channel_image_to_spectral_diagnostics",
    }:
        from . import spectral as _sp
        return getattr(_sp, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
