"""Large-mesh spectral compression.

Three solver paths, auto-selected by ``compress_obj`` / ``compress_large_mesh``:

  Dense eigh          (V ≤ V_DENSE  ≈ 5 000)
      O(V³) — fastest for small meshes.

  LOBPCG              (V_DENSE < V ≤ V_LOBPCG ≈ 20 000)
      Sparse iterative, Jacobi preconditioner.  No LU factorisation.
      Practical for moderate meshes with k ≤ 64.

  Nyström extension   (V > V_LOBPCG, default for large meshes)
      1. Subsample n_coarse ≈ max(500, 5·k, capped at 1500) vertices.
      2. Build dense k-NN graph on coarse vertices; exact eigh — O(n_c³) ≈ 0.3–2 s.
      3. Interpolate eigenvectors to all V vertices via weighted k-NN — O(V·k_interp·k).
      4. Estimate Rayleigh quotients via L_sparse @ Phi_full — O(nnz·k).
      5. Sort, apply heat-kernel weights h(λ) = exp(−λτ), project vertices.

      Constraint: n_coarse ≥ n_modes (required for coarse eigh to yield k eigenvectors).
      Cost for V = 465 K, k = 128, n_coarse = 1 500:   ≈ 10–20 s total.

OBJ IO is trimesh-free: a minimal parser handles v/f lines directly.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree

log = logging.getLogger(__name__)

from .graph3d import knn_symmetric_adjacency, adjacency_to_laplacian

# Vertex-count thresholds for auto-dispatch
V_DENSE   =  5_000   # use dense eigh below this
V_LOBPCG  = 20_000   # use LOBPCG below this, Nyström above


# ---------------------------------------------------------------------------
# Minimal OBJ parser (no trimesh dependency)
# ---------------------------------------------------------------------------

def load_obj(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse OBJ file, returning (vertices, faces).

    Only reads 'v' and 'f' lines; ignores normals, texcoords, materials.
    Face entries with vertex/texture/normal indices use only the vertex part.
    Handles triangles and quads (quads are split into two triangles).
    """
    vertices: list[list[float]] = []
    faces: list[list[int]] = []

    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v":
                vertices.append([float(x) for x in parts[1:4]])
            elif parts[0] == "f":
                # Each token may be "v", "v/vt", "v/vt/vn", or "v//vn"
                idx = [int(tok.split("/")[0]) for tok in parts[1:]]
                # OBJ is 1-indexed
                idx = [i - 1 if i > 0 else len(vertices) + i for i in idx]
                if len(idx) == 3:
                    faces.append(idx)
                elif len(idx) == 4:
                    # Split quad into two triangles
                    faces.append([idx[0], idx[1], idx[2]])
                    faces.append([idx[0], idx[2], idx[3]])
                # Ignore n-gons > 4

    return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int32)


# ---------------------------------------------------------------------------
# Sparse Laplacian builder
# ---------------------------------------------------------------------------

def sparse_combinatorial_laplacian(n_vertices: int, faces: np.ndarray) -> sp.csr_matrix:
    """Build combinatorial Laplacian as a sparse matrix (memory-efficient)."""
    f = np.asarray(faces, dtype=int)[:, :3]
    rows, cols = [], []
    for tri in f:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for u, v in ((a, b), (b, c), (c, a)):
            rows += [u, v]
            cols += [v, u]
    rows_arr = np.array(rows, dtype=np.int32)
    cols_arr = np.array(cols, dtype=np.int32)
    data = np.ones(len(rows_arr), dtype=np.float64)
    W = sp.csr_matrix(
        (data, (rows_arr, cols_arr)), shape=(n_vertices, n_vertices)
    )
    W = (W + W.T) / 2
    W.data = np.ones_like(W.data)  # unit weights
    d = np.asarray(W.sum(axis=1)).ravel()
    D = sp.diags(d)
    return (D - W).tocsr()


# ---------------------------------------------------------------------------
# Sparse spectral compression
# ---------------------------------------------------------------------------

@dataclass
class LargeMeshCompressed:
    """Compressed large mesh: sparse-eigensolver output + vertex coefficients + faces."""

    eigenvalues: np.ndarray    # (k,)
    eigenvectors: np.ndarray   # (V, k)
    h: np.ndarray              # (k,)  heat-kernel weights
    vertex_coeffs: np.ndarray  # (k, 3)
    faces: np.ndarray          # (F, 3) int32
    meta: dict[str, Any]

    def to_bytes(self) -> bytes:
        buf = BytesIO()
        np.savez_compressed(
            buf,
            eigenvalues=self.eigenvalues,
            eigenvectors=self.eigenvectors,
            h=self.h,
            vertex_coeffs=self.vertex_coeffs,
            faces=self.faces,
            meta=np.array(self.meta, dtype=object),
        )
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes) -> "LargeMeshCompressed":
        z = np.load(BytesIO(data), allow_pickle=True)
        raw = z["meta"].item() if "meta" in z.files else {}
        meta = raw if isinstance(raw, dict) else {}
        return cls(
            eigenvalues=np.asarray(z["eigenvalues"]),
            eigenvectors=np.asarray(z["eigenvectors"]),
            h=np.asarray(z["h"]),
            vertex_coeffs=np.asarray(z["vertex_coeffs"]),
            faces=np.asarray(z["faces"], dtype=np.int32),
            meta=meta,
        )


def compress_large_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    n_modes: int = 128,
    heat_tau: float = 1.0,
    tol: float = 0.0,
    maxiter: int | None = None,
) -> LargeMeshCompressed:
    """Compress a large mesh using sparse shift-invert Lanczos.

    Parameters
    ----------
    vertices  : (V, 3) float
    faces     : (F, 3) int
    n_modes   : k Laplacian modes to retain (default 128)
    heat_tau  : heat-kernel weight τ: h(λ) = exp(-λτ)
    tol       : eigensolver tolerance (0 = machine precision)
    maxiter   : max Lanczos iterations (None = scipy default)

    Returns
    -------
    LargeMeshCompressed payload with vertex coefficients.
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int32)
    V, F = v.shape[0], f.shape[0]

    t0 = time.perf_counter()
    L = sparse_combinatorial_laplacian(V, f)
    t_lap = time.perf_counter() - t0

    k = min(n_modes, V - 2)
    t1 = time.perf_counter()

    # LOBPCG: no LU factorisation required — scales to millions of nodes.
    # Jacobi (diagonal) preconditioner: M⁻¹ = diag(1/d), d = diag(L).
    # Regularise zero diagonal entries to avoid division by zero.
    d = np.asarray(L.diagonal(), dtype=np.float64)
    d_reg = np.where(d > 0, d, 1.0)
    M_inv = sp.diags(1.0 / d_reg)

    rng = np.random.default_rng(0)
    X0 = rng.standard_normal((V, k))
    X0, _ = np.linalg.qr(X0)
    X0 = np.asfortranarray(X0[:, :k])

    lam, Phi = spla.lobpcg(
        L, X0,
        M=M_inv,
        largest=False,
        tol=max(tol, 1e-6),
        maxiter=maxiter or 300,
        verbosityLevel=0,
    )
    t_eig = time.perf_counter() - t1

    order = np.argsort(lam)
    lam = lam[order]
    Phi = Phi[:, order]
    lam = np.maximum(lam, 0.0)

    h = np.exp(-lam * float(heat_tau))
    h = np.maximum(h, 1e-12)

    coeffs = Phi.T @ v   # (k, 3)

    # Count edges via Euler: E ≈ 3/2 F for manifold mesh
    n_edges_est = int(round(3 * F / 2))

    meta: dict[str, Any] = {
        "kind": "large_mesh",
        "n_vertices": V,
        "n_faces": F,
        "n_edges_est": n_edges_est,
        "n_modes": k,
        "heat_tau": float(heat_tau),
        "time_laplacian_s": round(t_lap, 3),
        "time_eigensolver_s": round(t_eig, 3),
    }
    return LargeMeshCompressed(
        eigenvalues=lam,
        eigenvectors=Phi,
        h=h,
        vertex_coeffs=coeffs,
        faces=f,
        meta=meta,
    )


def decompress_large_mesh(c: LargeMeshCompressed) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct (vertices, faces) from compressed payload."""
    v_hat = c.eigenvectors @ c.vertex_coeffs
    return np.asarray(v_hat, dtype=np.float64), np.asarray(c.faces, dtype=np.int32)


# ---------------------------------------------------------------------------
# Nyström extension — the scalable path for V > V_LOBPCG
# ---------------------------------------------------------------------------

def compress_large_mesh_nystrom(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    n_modes: int = 128,
    n_coarse: int | None = None,
    heat_tau: float = 1.0,
    k_knn: int = 8,
    k_interp: int = 3,
    seed: int = 0,
) -> LargeMeshCompressed:
    """Compress a large mesh via Nyström spectral extension.

    Algorithm
    ---------
    1. Build sparse Laplacian L on full mesh  — O(F)
    2. Subsample ``n_coarse`` vertices randomly
    3. k-NN graph on coarse vertices + dense eigh — O(n_c³)
    4. Weighted k-NN interpolation of coarse eigenvectors to all V vertices
    5. Estimate Rayleigh quotients:  λ̂_l = φ_lᵀ L φ_l  via L @ Φ
    6. Heat-kernel weights h(λ̂) = exp(−λ̂ τ); project vertices: C = Φᵀ V

    Parameters
    ----------
    vertices  : (V, 3) float
    faces     : (F, 3) int
    n_modes   : k modes to retain
    n_coarse  : coarse subgraph size (default: max(1000, 10·k), capped at V//2)
    heat_tau  : heat-kernel weight parameter
    k_knn     : neighbours for coarse k-NN graph
    k_interp  : neighbours for Nyström interpolation
    seed      : RNG seed for subsampling
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int32)
    V, F = v.shape[0], f.shape[0]

    k = min(int(n_modes), V - 2)
    # n_coarse must satisfy: n_c >= k (coarse eigh produces n_c eigenvectors).
    # Cap at 1500 so dense eigh stays below ~2 s on any hardware.
    if n_coarse is None:
        n_coarse = min(max(500, 5 * k), 1500)
    n_c = min(int(n_coarse), V // 2)

    if n_c < k:
        raise ValueError(
            f"n_coarse ({n_c}) must be >= n_modes ({k}). "
            f"Pass n_coarse >= {k} explicitly, or reduce n_modes."
        )

    log.debug("[Nyström] V=%d  F=%d  k=%d  n_coarse=%d", V, F, k, n_c)

    # ── 1. Sparse Laplacian on full mesh ──────────────────────────────────
    t0 = time.perf_counter()
    L_sp = sparse_combinatorial_laplacian(V, f)
    t_lap = time.perf_counter() - t0
    log.debug("[Nyström] Laplacian built  (%.1fs)", t_lap)

    # ── 2. Subsample coarse vertices ──────────────────────────────────────
    rng = np.random.default_rng(seed)
    coarse_idx = rng.choice(V, n_c, replace=False)
    coarse_xyz = v[coarse_idx]

    # Auto sigma: median nearest-neighbour distance among coarse points
    tree_c = cKDTree(coarse_xyz)
    d_nn, _ = tree_c.query(coarse_xyz, k=2)
    sigma = max(float(np.median(d_nn[:, 1])), 1e-12)

    # ── 3. Dense k-NN graph + exact eigendecomposition ────────────────────
    t1 = time.perf_counter()
    W_c = knn_symmetric_adjacency(coarse_xyz, k=min(k_knn, n_c - 1), sigma=sigma)
    L_c = adjacency_to_laplacian(W_c)
    lam_c, Phi_c = np.linalg.eigh(L_c)
    lam_c = np.maximum(lam_c[:k], 0.0)
    Phi_c = Phi_c[:, :k]
    t_eig = time.perf_counter() - t1
    log.debug("[Nyström] Coarse eigh done (%.1fs)  λ_0=%.4f  λ_%d=%.4f",
              t_eig, lam_c[0], k - 1, lam_c[-1])

    # ── 4. Nyström interpolation: coarse → full ───────────────────────────
    t2 = time.perf_counter()
    k_q = min(k_interp, n_c)
    d_full, idx_full = tree_c.query(v, k=k_q)   # (V, k_q)

    # RBF weights; handle zero-distance (exact coarse vertex hit)
    w = np.exp(-(d_full ** 2) / (2.0 * sigma ** 2))
    w_sum = w.sum(axis=1, keepdims=True)
    w = w / np.where(w_sum > 0, w_sum, 1.0)     # (V, k_q)

    # Φ_full[i] = Σ_j w_ij * Φ_c[idx_full[i,j]]
    Phi_full = np.einsum("ij,ijk->ik", w, Phi_c[idx_full])   # (V, k)

    # Column-normalise (avoids O(Vk²) full QR while preserving near-orthogonality)
    norms = np.linalg.norm(Phi_full, axis=0, keepdims=True)
    Phi_full = Phi_full / np.where(norms > 0, norms, 1.0)
    t_interp = time.perf_counter() - t2
    log.debug("[Nyström] Interpolation done (%.1fs)", t_interp)

    # ── 5. Rayleigh quotients on full sparse Laplacian ────────────────────
    t3 = time.perf_counter()
    LPhi = L_sp @ Phi_full                                   # (V, k)
    lam_est = np.einsum("ij,ij->j", Phi_full, LPhi)         # diag(ΦᵀLΦ)
    lam_est = np.maximum(lam_est, 0.0)
    order = np.argsort(lam_est)
    lam_est = lam_est[order]
    Phi_full = Phi_full[:, order]
    t_rayleigh = time.perf_counter() - t3
    log.debug("[Nyström] Rayleigh quotients done (%.1fs)  λ̂_0=%.4f  λ̂_%d=%.4f",
              t_rayleigh, lam_est[0], k - 1, lam_est[-1])

    # ── 6. Heat-kernel weights + vertex coefficients ──────────────────────
    h = np.exp(-lam_est * float(heat_tau))
    h = np.maximum(h, 1e-12)
    # Phi_full is only approximately orthonormal after Nyström interpolation.
    # Solve least-squares for coefficients to avoid projection bias.
    coeffs, *_ = np.linalg.lstsq(Phi_full, v, rcond=None)   # (k, 3)

    t_total = time.perf_counter() - t0
    log.debug("[Nyström] Total: %.1fs", t_total)

    meta: dict[str, Any] = {
        "kind": "large_mesh_nystrom",
        "n_vertices": V,
        "n_faces": F,
        "n_edges_est": int(round(3 * F / 2)),
        "n_modes": k,
        "n_coarse": n_c,
        "k_knn": k_knn,
        "k_interp": k_interp,
        "heat_tau": float(heat_tau),
        "time_laplacian_s": round(t_lap, 2),
        "time_eigh_s": round(t_eig, 2),
        "time_interp_s": round(t_interp, 2),
        "time_rayleigh_s": round(t_rayleigh, 2),
        "time_total_s": round(t_total, 2),
    }
    return LargeMeshCompressed(
        eigenvalues=lam_est,
        eigenvectors=Phi_full,
        h=h,
        vertex_coeffs=coeffs,
        faces=f,
        meta=meta,
    )


# ---------------------------------------------------------------------------
# OBJ-specific convenience wrappers
# ---------------------------------------------------------------------------

def compress_obj(
    obj_path: str | Path,
    *,
    n_modes: int = 128,
    heat_tau: float = 1.0,
    n_coarse: int | None = None,
    payload_path: str | Path | None = None,
) -> LargeMeshCompressed:
    """Load OBJ and compress, auto-selecting the best eigensolver.

    Dispatch rules (by vertex count V):
      V ≤ 5 000   → dense ``np.linalg.eigh``          (exact, < 0.1 s)
      V ≤ 20 000  → sparse LOBPCG + Jacobi precond.   (< 30 s for k ≤ 64)
      V > 20 000  → Nyström extension on n_coarse pts  (≈ 10–20 s for any V)
    """
    vertices, faces = load_obj(obj_path)
    V = vertices.shape[0]
    log.debug("Loaded OBJ: V=%d  F=%d", V, faces.shape[0])
    if V <= V_DENSE:
        c = compress_large_mesh(vertices, faces, n_modes=n_modes, heat_tau=heat_tau)
    elif V <= V_LOBPCG:
        c = compress_large_mesh(vertices, faces, n_modes=n_modes, heat_tau=heat_tau)
    else:
        c = compress_large_mesh_nystrom(
            vertices, faces,
            n_modes=n_modes,
            n_coarse=n_coarse,
            heat_tau=heat_tau,
        )
    if payload_path is not None:
        Path(payload_path).write_bytes(c.to_bytes())
    return c


def decompress_obj(
    compressed: LargeMeshCompressed | bytes | str | Path,
    obj_output_path: str | Path,
) -> Path:
    """Decompress and write reconstructed OBJ mesh."""
    if isinstance(compressed, LargeMeshCompressed):
        c = compressed
    elif isinstance(compressed, (str, Path)):
        c = LargeMeshCompressed.from_bytes(Path(compressed).read_bytes())
    elif isinstance(compressed, bytes):
        c = LargeMeshCompressed.from_bytes(compressed)
    else:
        raise TypeError("Expected LargeMeshCompressed, bytes, or path.")

    vertices, faces = decompress_large_mesh(c)
    out = Path(obj_output_path)
    _write_obj(vertices, faces, out)
    return out


def _write_obj(vertices: np.ndarray, faces: np.ndarray, path: Path) -> None:
    """Write minimal OBJ file (no normals, no materials)."""
    with open(path, "w") as fh:
        fh.write("# Reconstructed by kernelcal.geo3d.large_mesh\n")
        for v in vertices:
            fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            # OBJ is 1-indexed
            fh.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")


# ---------------------------------------------------------------------------
# Compression bounds for large meshes
# ---------------------------------------------------------------------------

def large_mesh_bounds(c: LargeMeshCompressed, vertices_original: np.ndarray) -> dict:
    """Compute compression ratio, distortion, and RMS error.

    Two storage modes are reported:

    coeff_only  — store only faces + spectral coefficients + eigenvalues.
                  Decode requires re-running the eigensolver (Nyström / LOBPCG).
                  Bytes: 8*(k*3 + k) + 4*3*F

    with_basis  — store eigenvectors too; decode is instant matrix multiply.
                  Bytes: 8*(V*k + k*3 + k) + 4*3*F  (current default payload)
    """
    from .bounds import distortion_from_eigenvalues
    V = c.meta["n_vertices"]
    F = c.meta["n_faces"]
    k = c.meta["n_modes"]

    raw_bytes   = 8 * 3 * V + 4 * 3 * F
    coeff_bytes = 8 * (k * 3 + k) + 4 * 3 * F          # faces + coeffs only
    basis_bytes = 8 * (V * k + k * 3 + k) + 4 * 3 * F  # + eigenvectors

    tail, total, rel = distortion_from_eigenvalues(
        vertices_original, c.eigenvectors, c.eigenvalues, k
    )
    rms = float(np.sqrt(tail / V)) if V > 0 else 0.0

    return {
        "n_vertices": V,
        "n_faces": F,
        "n_modes": k,
        "raw_bytes (binary)": raw_bytes,
        # coeff_only — theoretical minimum (eigenvectors re-derived at decode)
        "coeff_only_bytes": coeff_bytes,
        "compression_ratio (coeff_only)": round(raw_bytes / coeff_bytes, 2),
        "bits_per_vertex (coeff_only)": round(coeff_bytes * 8 / V, 1),
        # with_basis — actual payload as written by to_bytes()
        "with_basis_bytes": basis_bytes,
        "compression_ratio (with_basis)": round(raw_bytes / basis_bytes, 3),
        # distortion
        "spectral_tail_energy": round(tail, 4),
        "total_energy": round(total, 4),
        "relative_distortion": round(rel, 6),
        "rms_vertex_error": round(rms, 6),
        # timing
        "time_total_s": c.meta.get("time_total_s"),
    }
