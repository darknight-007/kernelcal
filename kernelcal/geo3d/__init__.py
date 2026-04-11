"""Geo3D compression with spectral graph kernels, Hodge topology, and bounds.

This subpackage provides compact spectral encodings for:
- point clouds (k-NN graph Laplacian)
- triangle meshes (connectivity Laplacian)
- temporal scan sequences (Hilbert–Schmidt path diagnostics)
- Hodge Laplacian complex (L₀/L₁/L₂), Betti numbers, Hodge decomposition
- persistent homology (0D/1D) for topology-preserving compression
- compression ratio bounds and rate–distortion analysis
"""

from .graph3d import adjacency_to_laplacian, knn_symmetric_adjacency, subsample_points
from .hodge import (
    HodgeSpectralBasis,
    betti_numbers,
    boundary_1,
    boundary_2,
    build_hodge_basis,
    hodge_decompose,
    hodge_laplacian_0,
    hodge_laplacian_1,
    hodge_laplacian_2,
)
from .topology import (
    PersistencePair,
    PersistenceResult,
    mesh_persistence,
    persistence_0d,
    persistence_1d,
    vietoris_rips_persistence,
)
from .bounds import (
    CompressionBounds,
    compression_ratio_formula,
    compression_ratio_vs_modes,
    distortion_from_eigenvalues,
    distortion_upper_bound,
    estimate_compression_bounds,
    mode_count_for_distortion,
    mode_count_for_topology,
)
from .mesh import (
    CompressedMeshGeometry,
    compress_dae,
    compress_mesh_geometry,
    compress_mesh_roundtrip,
    decompress_dae,
    decompress_mesh_roundtrip,
    mesh_combinatorial_laplacian,
)
from .spectral_codec import (
    CompressedSpectralKernel,
    compress_point_cloud,
    decompress_to_kernel,
)
from .temporal import TemporalKernelSummary, compress_temporal_clouds
from .large_mesh import (
    LargeMeshCompressed,
    compress_large_mesh,
    compress_obj,
    decompress_large_mesh,
    decompress_obj,
    large_mesh_bounds,
    load_obj,
    sparse_combinatorial_laplacian,
)

__all__ = [
    # graph3d
    "subsample_points",
    "knn_symmetric_adjacency",
    "adjacency_to_laplacian",
    # spectral_codec
    "CompressedSpectralKernel",
    "compress_point_cloud",
    "decompress_to_kernel",
    # mesh
    "mesh_combinatorial_laplacian",
    "compress_mesh_geometry",
    "CompressedMeshGeometry",
    "compress_mesh_roundtrip",
    "decompress_mesh_roundtrip",
    "compress_dae",
    "decompress_dae",
    # temporal
    "TemporalKernelSummary",
    "compress_temporal_clouds",
    # hodge
    "boundary_1",
    "boundary_2",
    "hodge_laplacian_0",
    "hodge_laplacian_1",
    "hodge_laplacian_2",
    "betti_numbers",
    "build_hodge_basis",
    "hodge_decompose",
    "HodgeSpectralBasis",
    # topology
    "PersistencePair",
    "PersistenceResult",
    "persistence_0d",
    "persistence_1d",
    "vietoris_rips_persistence",
    "mesh_persistence",
    # large_mesh
    "LargeMeshCompressed",
    "compress_large_mesh",
    "compress_obj",
    "decompress_large_mesh",
    "decompress_obj",
    "large_mesh_bounds",
    "load_obj",
    "sparse_combinatorial_laplacian",
    # bounds
    "CompressionBounds",
    "compression_ratio_formula",
    "compression_ratio_vs_modes",
    "distortion_from_eigenvalues",
    "distortion_upper_bound",
    "estimate_compression_bounds",
    "mode_count_for_topology",
    "mode_count_for_distortion",
]
