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

graph_explorer
    Shared exploration-policy primitives used by both the drone-DEM
    explorer and the bishop-rocks k-NN graph explorer: ``BettiWeights``,
    ``Candidate``, ``score_betti_candidate``, ``choose_best_candidate``
    for the canonical ``w_beta1·β₁/n − w_beta0·β₀/n + w_unseen·unseen``
    score with revisit penalty and cyclic tie-break; plus
    ``CameraModel`` (``altitude + fov → footprint_side_m``) and
    ``CoverageRaster`` (bool visited-mask over a metric bbox for
    ``1 − mean(visited[target])`` unseen semantics).  (Paper §P4 Δβ₁
    biosignature signal)

blender
    Blender/bpy integration for spectral digital twins.  Strict separation of
    concerns: Blender owns geometry and ground truth; kernelcal.geo3d owns all
    spectral and topological computation.  Modules:
      terrain_gen      — procedural terrain with controlled β₁ (bpy required)
      twin_receiver    — synthesized-twin visualiser in Blender (bpy required)
      q10_pipeline     — Nyström β₁ topology error experiment (pure Python)
      run_q10_experiment.sh — end-to-end orchestrator (Blender → kernelcal)
    Note: bpy modules are only importable inside Blender's own Python runtime.

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

__version__ = "0.9.2"

# ---------------------------------------------------------------------------
# Core (always-available) re-exports.
#
# These subpackages have no optional dependencies beyond numpy/scipy and are
# imported eagerly.  Optional subpackages (models, prompts, navigation,
# control) and the heavy geo3d subpackage are exposed lazily via
# ``__getattr__`` below so that ``import kernelcal`` stays light.
# ---------------------------------------------------------------------------
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
    CowanFarquharSource,
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

# ---------------------------------------------------------------------------
# Optional eager imports — still probed eagerly, but any failures leave the
# matching names absent rather than polluting the namespace.  Symbols are
# registered in ``__all__`` only when the import succeeds, so ``__all__``
# never lies about what's actually importable from the top-level namespace.
# ---------------------------------------------------------------------------
_OPTIONAL_EAGER: dict[str, tuple[str, ...]] = {
    ".models": ("ModelKernelSelector", "ModelRecord"),
    ".prompts": ("PromptKernelIterator", "prompt_kernel_distance"),
    ".navigation": (
        "SemanticSLAMKernelTracker",
        "InformativePathPlanner",
        "HumanPilotDemonstrationLearner",
    ),
    ".control": (
        "fit_riccati_analytic",
        "fit_riccati_residual",
        "care_residual",
        "estimate_A_log_OU",
        "ard_to_observation_matrix",
        "coupling_entropy_off_diagonal",
        "off_diagonal_frobenius",
        "riccati_conjecture_test",
        "landauer_R_lower_bound",
        "RiccatiAnalysisResult",
        "RiccatiConjectureTest",
        "OUIdentificationResult",
        "PlantPhenotypingCAREAnalyzer",
        "CAREAnalyzerConfig",
        "RotationInput",
        "CAREAnalyzerState",
    ),
}

_optional_loaded: set[str] = set()
for _mod_path, _names in _OPTIONAL_EAGER.items():
    try:
        _mod = __import__(__name__ + _mod_path, fromlist=list(_names))
    except ImportError:
        continue
    for _name in _names:
        globals()[_name] = getattr(_mod, _name)
        _optional_loaded.add(_name)
del _mod_path, _names

# ---------------------------------------------------------------------------
# Lazy subpackages.  geo3d is heavy (gudhi/trimesh/scipy.sparse machinery),
# fluid pulls SciPy ODE machinery, and spectral.experiments is a matplotlib-
# heavy driver; all three are loaded on first attribute access rather than
# on ``import kernelcal``.
# ---------------------------------------------------------------------------
_LAZY_GEO3D: frozenset[str] = frozenset({
    "CompressedMeshGeometry",
    "CompressedSpectralKernel",
    "CompressionBounds",
    "HodgeSpectralBasis",
    "PersistencePair",
    "PersistenceResult",
    "TemporalKernelSummary",
    "adjacency_to_laplacian",
    "betti_numbers",
    "build_hodge_basis",
    "compress_dae",
    "compress_mesh_geometry",
    "compress_mesh_roundtrip",
    "compress_point_cloud",
    "compress_temporal_clouds",
    "compression_ratio_formula",
    "compression_ratio_vs_modes",
    "decompress_dae",
    "decompress_mesh_roundtrip",
    "decompress_to_kernel",
    "distortion_from_eigenvalues",
    "estimate_compression_bounds",
    "hodge_decompose",
    "knn_symmetric_adjacency",
    "mesh_combinatorial_laplacian",
    "mesh_persistence",
    "mode_count_for_distortion",
    "mode_count_for_topology",
    "subsample_points",
    "vietoris_rips_persistence",
})

_LAZY_FLUID: frozenset[str] = frozenset({
    "FluidGraph",
    "PotentialLandscape",
    "FluidSimulationConfig",
    "FluidSimulationResult",
    "ring_distance",
    "gaussian_bump_on_ring",
    "make_twenty_node_reference_landscape",
    "simulate_kernel_fluid",
    "save_timeseries_csv",
})

_LAZY_CHANNEL_IMAGE: frozenset[str] = frozenset({
    "ChannelEdge",
    "ChannelGraphExtraction",
    "FlowTopologyAnalysis",
    "ChannelVerificationArtifacts",
    "ChannelImageSpectralDiagnostics",
    "extract_channel_graph_from_image",
    "analyze_channel_network_image",
    "save_channel_extraction_verification",
    "channel_image_to_spectral_diagnostics",
})


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
    if name in _LAZY_FLUID:
        from . import fluid as _fl
        return getattr(_fl, name)
    if name in _LAZY_GEO3D:
        from . import geo3d as _g
        return getattr(_g, name)
    if name in _LAZY_CHANNEL_IMAGE:
        from . import spectral as _sp
        return getattr(_sp, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Stable core symbols, always present.
_CORE_ALL: tuple[str, ...] = (
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
    # attention
    "AttentionKernel",
    "AttentionKernelResult",
    "AttentionKernelTracker",
    "run_attention_experiment",
    "AttentionExperimentResult",
    # spectral
    "SpectralGraph",
    "GaussianMISource",
    "CoupledGaussianMISource",
    "CowanFarquharSource",
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
    # lazy-top-level entry points
    "run_all_experiments",
    "run_procedural_examples",
    "run_twenty_node_experiment",
)

# Optional + lazy symbols are advertised in __all__ as well, because they're
# reachable from ``kernelcal.<name>`` at runtime; however we only list
# optional names that the current process could actually import.
__all__: list[str] = list(_CORE_ALL)
__all__.extend(sorted(_optional_loaded))
__all__.extend(sorted(_LAZY_FLUID))
__all__.extend(sorted(_LAZY_GEO3D))
# NOTE: _LAZY_CHANNEL_IMAGE is intentionally omitted from __all__ — those are
# back-compat aliases only, not a recommended top-level API.
