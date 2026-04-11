"""Compression ratio bounds and distortion estimates for spectral mesh coding.

Notation
--------
V = n_vertices,  E = n_edges,  F = n_faces,  k = n_modes retained.

Storage model
-------------
The payload stored on disk is:

  If ``coeff_only=True``  (recompute Laplacian basis at decode time):
      vertex_coeffs  :  k × 3  float64  =  24k bytes
      eigenvalues    :  k      float64  =   8k bytes
      faces          :  F × 3  int32    =  12F bytes
      ──────────────────────────────────────────────
      Total compressed:  32k + 12F  bytes

  If ``coeff_only=False`` (eigenvectors stored, no recomputation):
      eigenvectors   :  V × k  float64  =   8Vk bytes
      vertex_coeffs  :  k × 3  float64  =  24k  bytes
      eigenvalues    :  k      float64  =   8k  bytes
      faces          :  F × 3  int32    =  12F  bytes
      ──────────────────────────────────────────────
      Total compressed:  8Vk + 32k + 12F  bytes

Raw storage (float64 vertices, int32 faces):
      vertices  :  V × 3  float64  =  24V  bytes
      faces     :  F × 3  int32    =  12F  bytes
      ──────────────────────────────────────────
      Total raw:  24V + 12F  bytes

Distortion model
----------------
The reconstruction error for vertex coordinates is:

    ||V - Φ_k Φ_kᵀ V||²_F = ||V||²_F - ||Φ_kᵀ V||²_F
                           = Σ_{l>k} ||Φ_lᵀ V||²_F    (spectral tail energy)

Upper bound via Laplacian smoothness (Poincaré / Cheeger):
    RMS vertex error  ≤  sqrt(spectral_tail_energy / n_vertices)
    Normalised error  =  spectral_tail_energy / ||V||²_F

Topology preservation
---------------------
β₀ connected components are preserved iff k ≥ β₀ (zero modes kept).
β₁ independent 1-cycles are preserved iff k ≥ β₀ + β₁ (enough low-freq modes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CompressionBounds:
    """All compression metrics for a single mesh + mode-count pair.

    Attributes
    ----------
    n_vertices, n_edges, n_faces : mesh dimensions
    n_modes                      : k (number of retained modes)
    coeff_only                   : True ⇒ eigenvectors are recomputed at decode
    raw_bytes                    : uncompressed storage (float64 v + int32 f)
    compressed_bytes             : compressed storage
    compression_ratio            : raw_bytes / compressed_bytes
    bits_per_vertex              : (compressed_bytes × 8) / n_vertices
    spectral_tail_energy         : ||V - Φ_k Φ_kᵀ V||²_F
    total_energy                 : ||V||²_F
    relative_distortion          : spectral_tail_energy / total_energy
    rms_error                    : sqrt(tail / n_vertices)
    betti                        : (β₀, β₁, β₂) if computed, else None
    topology_preserved           : True iff k ≥ β₀ + β₁
    """

    n_vertices: int
    n_edges: int
    n_faces: int
    n_modes: int
    coeff_only: bool
    raw_bytes: int
    compressed_bytes: int
    compression_ratio: float
    bits_per_vertex: float
    spectral_tail_energy: float
    total_energy: float
    relative_distortion: float
    rms_error: float
    betti: tuple[int, int, int] | None = None
    topology_preserved: bool | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Mesh:  V={self.n_vertices}, E={self.n_edges}, F={self.n_faces}",
            f"Modes: k={self.n_modes}  (coeff_only={self.coeff_only})",
            f"Storage:  {self.raw_bytes:,} → {self.compressed_bytes:,} bytes",
            f"Compression ratio:  {self.compression_ratio:.2f}×",
            f"Bits / vertex:      {self.bits_per_vertex:.1f} bpv",
            f"Relative distortion:{self.relative_distortion:.4f}",
            f"RMS vertex error:   {self.rms_error:.6f}",
        ]
        if self.betti is not None:
            b0, b1, b2 = self.betti
            lines.append(f"Betti (β₀,β₁,β₂):  ({b0}, {b1}, {b2})")
            lines.append(f"Topology preserved: {self.topology_preserved}")
        return "\n".join(lines)


def _storage(
    n_vertices: int,
    n_faces: int,
    n_modes: int,
    coeff_only: bool,
    float_bytes: int = 8,
    int_bytes: int = 4,
) -> tuple[int, int]:
    """Return (raw_bytes, compressed_bytes)."""
    raw = float_bytes * 3 * n_vertices + int_bytes * 3 * n_faces
    if coeff_only:
        compressed = float_bytes * (n_modes * 3 + n_modes) + int_bytes * 3 * n_faces
    else:
        compressed = (
            float_bytes * n_vertices * n_modes
            + float_bytes * (n_modes * 3 + n_modes)
            + int_bytes * 3 * n_faces
        )
    return raw, compressed


def compression_ratio_formula(
    n_vertices: int,
    n_faces: int,
    n_modes: int,
    coeff_only: bool = True,
) -> float:
    """Closed-form compression ratio (raw / compressed), geometry only.

    coeff_only=True  :  ratio ≈ 3V / (4k + F)         (faces lossless)
    coeff_only=False :  ratio ≈ 3V / (kV + 4k + F)    (eigenvectors stored)
    """
    raw, comp = _storage(n_vertices, n_faces, n_modes, coeff_only)
    return raw / comp if comp > 0 else float("inf")


def compression_ratio_vs_modes(
    n_vertices: int,
    n_faces: int,
    modes: list[int] | np.ndarray | None = None,
    coeff_only: bool = True,
) -> np.ndarray:
    """Compression ratio for a range of mode counts.

    Returns 2-column array [k, ratio].
    """
    if modes is None:
        modes = np.arange(1, min(n_vertices, 512) + 1)
    ks = np.asarray(modes, dtype=int)
    ratios = np.array(
        [compression_ratio_formula(n_vertices, n_faces, int(k), coeff_only) for k in ks]
    )
    return np.stack([ks.astype(float), ratios], axis=1)


def distortion_from_eigenvalues(
    vertices_xyz: np.ndarray,
    eigenvectors: np.ndarray,
    eigenvalues: np.ndarray,
    n_modes: int,
) -> tuple[float, float, float]:
    """Compute spectral distortion from Laplacian eigenpairs.

    Parameters
    ----------
    vertices_xyz  : (V, 3) original vertex positions
    eigenvectors  : (V, N) full eigenvector matrix
    eigenvalues   : (N,) Laplacian eigenvalues
    n_modes       : k modes retained

    Returns
    -------
    (spectral_tail_energy, total_energy, relative_distortion)
    """
    v = np.asarray(vertices_xyz, dtype=float)
    Phi = np.asarray(eigenvectors, dtype=float)
    k = min(int(n_modes), Phi.shape[1])
    Phi_k = Phi[:, :k]
    v_hat = Phi_k @ (Phi_k.T @ v)
    tail = float(np.sum((v - v_hat) ** 2))
    total = float(np.sum(v ** 2))
    rel = tail / total if total > 0 else 0.0
    return tail, total, rel


def distortion_upper_bound(
    eigenvalues: np.ndarray,
    vertex_coeffs: np.ndarray,
    n_modes: int,
) -> float:
    """Poincaré-style upper bound on spectral tail energy.

    Bound: Σ_{l>k} λ_l × ||c_l||²  where c_l = Φ_lᵀ V.

    This holds because ||Φ_lᵀ V||² ≤ ||c_l||² and the Laplacian
    weights higher-frequency components by λ_l.
    """
    lam = np.asarray(eigenvalues, dtype=float)
    coeffs = np.asarray(vertex_coeffs, dtype=float)  # (N, 3) or (N,)
    k = int(n_modes)
    if k >= len(lam):
        return 0.0
    lam_tail = lam[k:]
    if coeffs.ndim == 2:
        c_sq = np.sum(coeffs[k:] ** 2, axis=1)
    else:
        c_sq = coeffs[k:] ** 2
    return float(np.dot(lam_tail, c_sq))


def estimate_compression_bounds(
    vertices_xyz: np.ndarray,
    faces: np.ndarray,
    eigenvectors: np.ndarray,
    eigenvalues: np.ndarray,
    n_modes: int,
    coeff_only: bool = True,
    betti: tuple[int, int, int] | None = None,
    n_edges: int | None = None,
) -> CompressionBounds:
    """Compute all compression metrics for a compressed mesh.

    Parameters
    ----------
    vertices_xyz : (V, 3) original vertices
    faces        : (F, 3) integer face array
    eigenvectors : (V, ≥n_modes) Laplacian eigenvectors
    eigenvalues  : (≥n_modes,) Laplacian eigenvalues
    n_modes      : k retained modes
    coeff_only   : whether eigenvectors are excluded from payload
    betti        : (β₀, β₁, β₂) if available
    n_edges      : edge count if known (otherwise estimated from faces)
    """
    v = np.asarray(vertices_xyz, dtype=float)
    f = np.asarray(faces, dtype=int)
    V, F = v.shape[0], f.shape[0]

    # Edge count estimate via Euler: E ≈ 3/2 F for a manifold mesh
    E = n_edges if n_edges is not None else int(round(3 * F / 2))

    raw, comp = _storage(V, F, n_modes, coeff_only)
    ratio = raw / comp if comp > 0 else float("inf")
    bpv = (comp * 8) / V if V > 0 else float("inf")

    tail, total, rel = distortion_from_eigenvalues(v, eigenvectors, eigenvalues, n_modes)
    rms = float(np.sqrt(tail / V)) if V > 0 else 0.0

    topo_preserved: bool | None = None
    if betti is not None:
        b0, b1, _ = betti
        topo_preserved = n_modes >= (b0 + b1)

    return CompressionBounds(
        n_vertices=V,
        n_edges=E,
        n_faces=F,
        n_modes=n_modes,
        coeff_only=coeff_only,
        raw_bytes=raw,
        compressed_bytes=comp,
        compression_ratio=ratio,
        bits_per_vertex=bpv,
        spectral_tail_energy=tail,
        total_energy=total,
        relative_distortion=rel,
        rms_error=rms,
        betti=betti,
        topology_preserved=topo_preserved,
        meta={
            "eigenvalue_at_k": float(eigenvalues[n_modes - 1]) if n_modes <= len(eigenvalues) else None,
            "eigenvalue_kp1": float(eigenvalues[n_modes]) if n_modes < len(eigenvalues) else None,
            "spectral_gap_at_k": (
                float(eigenvalues[n_modes] - eigenvalues[n_modes - 1])
                if n_modes < len(eigenvalues) and n_modes >= 1
                else None
            ),
        },
    )


def mode_count_for_topology(betti: tuple[int, int, int]) -> int:
    """Minimum modes needed to preserve β₀ and β₁ topological invariants."""
    b0, b1, _ = betti
    return b0 + b1


def mode_count_for_distortion(
    eigenvalues: np.ndarray,
    vertex_coeffs: np.ndarray,
    target_rel_distortion: float,
) -> int:
    """Find minimum k such that relative distortion ≤ target_rel_distortion.

    Parameters
    ----------
    eigenvalues       : (N,) full spectrum (descending stability useful but ascending ok)
    vertex_coeffs     : (N, 3) or (N,) spectral coefficients Φᵀ V
    target_rel_distortion : desired fraction of total energy remaining after compression

    Returns
    -------
    Minimum k ∈ {1, ..., N} satisfying the target, or N if never met.
    """
    lam = np.asarray(eigenvalues, dtype=float)
    coeffs = np.asarray(vertex_coeffs, dtype=float)
    N = len(lam)
    if coeffs.ndim == 2:
        c_sq = np.sum(coeffs ** 2, axis=1)
    else:
        c_sq = coeffs ** 2
    total = float(c_sq.sum())
    if total == 0.0:
        return 1

    cumulative = np.cumsum(c_sq)
    captured_ratio = cumulative / total
    idx = np.searchsorted(captured_ratio, 1.0 - target_rel_distortion)
    return int(np.clip(idx + 1, 1, N))
