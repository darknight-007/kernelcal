"""Legacy import path for the former standalone ``deepgis-kernelcal`` distribution.

All implementations live in :mod:`kernelcal.geo3d`. New code should use::

    from kernelcal.geo3d import compress_point_cloud, ...

This package exists so ``pip install -e .`` from the kernelcal tree continues to
provide ``import deepgis_kernelcal`` for existing notebooks and scripts.
"""

from __future__ import annotations

from kernelcal.geo3d import (
    CompressedSpectralKernel,
    TemporalKernelSummary,
    compress_mesh_geometry,
    compress_point_cloud,
    compress_temporal_clouds,
    decompress_to_kernel,
    mesh_combinatorial_laplacian,
)

__all__ = [
    "CompressedSpectralKernel",
    "TemporalKernelSummary",
    "compress_mesh_geometry",
    "compress_point_cloud",
    "compress_temporal_clouds",
    "decompress_to_kernel",
    "mesh_combinatorial_laplacian",
]

# Matches the last standalone deepgis-kernelcal release; prefer kernelcal's version.
__version__ = "0.1.0"
