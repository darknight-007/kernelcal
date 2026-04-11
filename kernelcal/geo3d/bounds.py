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


# ============================================================================
# Self-introspection: CompressionScore
# ============================================================================

@dataclass
class CompressionScore:
    """Multi-dimensional quality report for a spectral mesh compression.

    Each field measures a different aspect of what was *lost* by keeping
    only k Laplacian modes.  The framework identifies four orthogonal
    loss channels:

    Geometry  — how much do vertex positions drift?
    Spectral  — how much of the kernel's frequency structure is preserved?
    Kernel    — how much of the full kernel's inner-product norm is retained?
    Topology  — do the mesh's handles and components survive?

    ``overall_loss ∈ [0, 1]`` is a weighted composite; 0 is lossless,
    1 is total destruction.  ``bottleneck`` names the dominant channel.

    Attributes
    ----------
    n_vertices, n_faces, n_modes
        Mesh dimensions and retained mode count.
    compression_ratio, bits_per_vertex
        Storage efficiency (raw_bytes / compressed_bytes).
    relative_distortion, rms_vertex_error
        Geometry loss.  ``None`` when no vertex coordinates are available.
    spectral_entropy_compressed
        H[h_k] = −∑ h̄_l log h̄_l for the k retained modes.
    spectral_entropy_max
        log(k): maximum achievable entropy for k modes (uniform weights).
    spectral_entropy_retention
        H[h_k] / log(k) ∈ [0, 1].  Near 1 means spectral weight is spread
        evenly across retained modes; near 0 means all weight concentrates
        on the lowest-frequency mode.
    kernel_hs_norm
        ||K̂||_HS = sqrt(∑_l h_l²): Hilbert–Schmidt norm of compressed kernel.
    kernel_hs_relative
        ||K̂||_HS / ||K_full||_HS ∈ (0, 1].  Requires ``eigenvalues_full``.
        Near 1 means the compressed kernel captures most of the full kernel's
        norm; near 0 means most spectral mass was truncated.
    spectral_gap_ratio
        (λ_{k+1} − λ_k) / max(λ_k, ε).  Large gap = natural truncation
        point (eigenmodes cluster neatly below k); near 0 = cutting
        mid-cluster.  Requires ``eigenvalues_full``.
    topology_preserved
        True iff k ≥ β₀ + β₁ (connected components and independent cycles
        are preserved in the compressed representation).
    topology_margin
        k − (β₀ + β₁).  Positive = safety margin; negative = topological
        information is being lost.
    overall_loss
        Weighted composite loss ∈ [0, 1].  Lower is better.
    bottleneck
        Which channel contributes most to ``overall_loss``.
    """

    # Identification
    n_vertices: int
    n_faces: int | None
    n_modes: int

    # Rate
    compression_ratio: float
    bits_per_vertex: float

    # Geometry
    relative_distortion: float | None
    rms_vertex_error: float | None

    # Spectral
    spectral_entropy_compressed: float
    spectral_entropy_max: float
    spectral_entropy_retention: float

    # Kernel norm
    kernel_hs_norm: float
    kernel_hs_relative: float | None

    # Spectral gap at truncation
    spectral_gap_ratio: float | None

    # Topology
    topology_preserved: bool | None
    topology_margin: int | None

    # Overall
    overall_loss: float
    bottleneck: str

    def grade(self) -> str:
        """Letter-grade based on overall_loss."""
        loss = self.overall_loss
        if loss < 0.05:
            return "Excellent"
        if loss < 0.15:
            return "Good"
        if loss < 0.35:
            return "Fair"
        return "Poor"

    def summary(self) -> str:
        lines = [
            f"── Compression Score ──────────────────────────────────",
            f"  Grade:             {self.grade()}  (loss={self.overall_loss:.4f})",
            f"  Bottleneck:        {self.bottleneck}",
            f"",
            f"  Modes / vertices:  k={self.n_modes}  /  V={self.n_vertices}"
            + (f"  F={self.n_faces}" if self.n_faces is not None else ""),
            f"  Compression ratio: {self.compression_ratio:.2f}×"
            f"  ({self.bits_per_vertex:.1f} bpv)",
            f"",
            f"  ── Geometry ──",
        ]
        if self.relative_distortion is not None:
            lines += [
                f"  Relative distortion:     {self.relative_distortion:.4f}",
                f"  RMS vertex error:        {self.rms_vertex_error:.6f}",
            ]
        else:
            lines.append("  (no vertex coordinates provided)")
        lines += [
            f"",
            f"  ── Spectral ──",
            f"  Entropy (compressed): {self.spectral_entropy_compressed:.4f}",
            f"  Entropy (max k modes):{self.spectral_entropy_max:.4f}",
            f"  Entropy retention:    {self.spectral_entropy_retention:.4f}",
        ]
        if self.kernel_hs_relative is not None:
            lines.append(f"  Kernel HS retention:  {self.kernel_hs_relative:.4f}")
        if self.spectral_gap_ratio is not None:
            lines.append(
                f"  Spectral gap at k:    {self.spectral_gap_ratio:.4f}"
                + ("  ✓ natural cut" if self.spectral_gap_ratio > 0.5 else "  ✗ mid-cluster")
            )
        lines.append("")
        lines.append("  ── Topology ──")
        if self.topology_preserved is not None:
            status = "✓ preserved" if self.topology_preserved else "✗ LOST"
            lines.append(
                f"  Topology:             {status}"
                f"  (margin={self.topology_margin:+d})"
            )
        else:
            lines.append("  (no Betti numbers provided)")
        lines.append("────────────────────────────────────────────────────")
        return "\n".join(lines)


def _spectral_entropy(h: np.ndarray) -> float:
    """H[h] = −∑ h̄_l log h̄_l, zero-safe."""
    h = np.asarray(h, dtype=float)
    total = h.sum()
    if total <= 0:
        return 0.0
    h_bar = h / total
    h_bar = np.where(h_bar > 0, h_bar, 1.0)
    return float(-np.sum(h_bar * np.log(h_bar)))


def score_compression(
    compressed: Any,
    *,
    vertices_original: np.ndarray | None = None,
    eigenvalues_full: np.ndarray | None = None,
    betti: tuple[int, int, int] | None = None,
    coeff_only: bool = True,
) -> "CompressionScore":
    """Compute a multi-dimensional quality score for a compressed mesh.

    Parameters
    ----------
    compressed
        Any of ``CompressedMeshGeometry``, ``LargeMeshCompressed``, or
        ``CompressedSpectralKernel``.  Must expose ``.eigenvalues``,
        ``.eigenvectors``, ``.h``, and ``.meta``.
    vertices_original : (V, 3) array, optional
        Original vertex positions.  Required for geometry metrics.
    eigenvalues_full : (N,) array, optional
        Full Laplacian spectrum (all N modes).  Required for
        ``kernel_hs_relative`` and ``spectral_gap_ratio``.
    betti : (β₀, β₁, β₂), optional
        Betti numbers of the original mesh (from ``betti_numbers()``).
    coeff_only : bool, default True
        Storage model: True = eigenvectors re-derived at decode time.

    Returns
    -------
    CompressionScore
        Call ``.summary()`` for a human-readable report.
    """
    # ── Extract common fields via duck typing ─────────────────────────────
    try:
        eigenvalues = np.asarray(compressed.eigenvalues, dtype=float)
        eigenvectors = np.asarray(compressed.eigenvectors, dtype=float)
        h = np.asarray(compressed.h, dtype=float)
        meta = compressed.meta if hasattr(compressed, "meta") else {}
    except AttributeError as exc:
        raise TypeError(
            "compressed must expose .eigenvalues, .eigenvectors, .h, .meta"
        ) from exc

    k = len(h)
    V = int(meta.get("n_vertices", eigenvectors.shape[0]))
    F = meta.get("n_faces") or meta.get("n_faces", None)
    F = int(F) if F is not None else None

    # ── Compression rate ──────────────────────────────────────────────────
    n_faces_for_storage = F if F is not None else 0
    raw, comp = _storage(V, n_faces_for_storage, k, coeff_only)
    ratio = raw / comp if comp > 0 else float("inf")
    bpv = (comp * 8) / V if V > 0 else float("inf")

    # ── Geometry loss ─────────────────────────────────────────────────────
    rel_dist: float | None = None
    rms_err: float | None = None
    if vertices_original is not None:
        v = np.asarray(vertices_original, dtype=float)
        if v.shape == (V, 3) and eigenvectors.shape == (V, k):
            tail, total, rel_dist = distortion_from_eigenvalues(
                v, eigenvectors, eigenvalues, k
            )
            rms_err = float(np.sqrt(tail / V)) if V > 0 else 0.0

    # ── Spectral loss ─────────────────────────────────────────────────────
    h_entropy = _spectral_entropy(h)
    h_entropy_max = float(np.log(k)) if k > 1 else 0.0
    h_entropy_retention = (h_entropy / h_entropy_max) if h_entropy_max > 0 else 1.0

    # ── Kernel HS norm and relative retention ─────────────────────────────
    kernel_hs_norm = float(np.sqrt(np.sum(h ** 2)))
    kernel_hs_relative: float | None = None
    if eigenvalues_full is not None:
        lam_full = np.asarray(eigenvalues_full, dtype=float)
        tau = meta.get("heat_tau", 1.0) or 1.0
        h_full = np.exp(-lam_full * float(tau))
        h_full = np.maximum(h_full, 1e-12)
        hs_full = float(np.sqrt(np.sum(h_full ** 2)))
        kernel_hs_relative = kernel_hs_norm / hs_full if hs_full > 0 else None

    # ── Spectral gap at truncation point ─────────────────────────────────
    spectral_gap_ratio: float | None = None
    if eigenvalues_full is not None and len(eigenvalues_full) > k:
        lam_k = float(eigenvalues_full[k - 1])
        lam_kp1 = float(eigenvalues_full[k])
        denom = max(lam_k, 1e-12)
        spectral_gap_ratio = (lam_kp1 - lam_k) / denom

    # ── Topology ─────────────────────────────────────────────────────────
    topo_preserved: bool | None = None
    topo_margin: int | None = None
    if betti is not None:
        b0, b1, _ = betti
        topo_margin = k - (b0 + b1)
        topo_preserved = topo_margin >= 0

    # ── Overall loss (weighted composite) ────────────────────────────────
    # Geometry:  weight 0.5 if available, else proxy via spectral loss
    # Spectral:  weight 0.3 (entropy retention loss)
    # Topology:  weight 0.2 (binary penalty; 0 if topology preserved)
    spectral_loss = 1.0 - h_entropy_retention
    topology_penalty = 0.0
    if topo_preserved is not None and not topo_preserved:
        # Scale penalty by how far below the topology threshold k is
        deficit = -(topo_margin or 0)
        b_sum = (betti[0] + betti[1]) if betti else 1  # type: ignore[index]
        topology_penalty = min(1.0, deficit / max(b_sum, 1))

    if rel_dist is not None:
        geo_loss = float(min(rel_dist, 1.0))
        overall = 0.5 * geo_loss + 0.3 * spectral_loss + 0.2 * topology_penalty
        components = {"geometry": 0.5 * geo_loss, "spectral": 0.3 * spectral_loss,
                      "topology": 0.2 * topology_penalty}
    else:
        # No geometry available: re-weight remaining channels
        overall = 0.6 * spectral_loss + 0.4 * topology_penalty
        components = {"spectral": 0.6 * spectral_loss, "topology": 0.4 * topology_penalty}

    overall = float(np.clip(overall, 0.0, 1.0))
    bottleneck = max(components, key=lambda c: components[c])

    return CompressionScore(
        n_vertices=V,
        n_faces=F,
        n_modes=k,
        compression_ratio=ratio,
        bits_per_vertex=bpv,
        relative_distortion=rel_dist,
        rms_vertex_error=rms_err,
        spectral_entropy_compressed=h_entropy,
        spectral_entropy_max=h_entropy_max,
        spectral_entropy_retention=h_entropy_retention,
        kernel_hs_norm=kernel_hs_norm,
        kernel_hs_relative=kernel_hs_relative,
        spectral_gap_ratio=spectral_gap_ratio,
        topology_preserved=topo_preserved,
        topology_margin=topo_margin,
        overall_loss=overall,
        bottleneck=bottleneck,
    )
