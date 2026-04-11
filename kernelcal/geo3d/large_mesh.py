"""Large-mesh spectral compression using sparse iterative eigensolvers.

For meshes with V > ~5000 vertices, full dense eigendecomposition
(O(V³)) is infeasible.  This module uses:

  scipy.sparse.linalg.eigsh  with shift-invert (sigma=0)
      — finds the k smallest eigenvalues in O(k · V · nnz) time
      — nnz ≈ 6V for typical triangle meshes (average degree ~6)

OBJ loading is done without trimesh: a minimal parser handles
'v' and 'f' lines directly.  Face entries may be:
    f v1 v2 v3
    f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


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
# OBJ-specific convenience wrappers
# ---------------------------------------------------------------------------

def compress_obj(
    obj_path: str | Path,
    *,
    n_modes: int = 128,
    heat_tau: float = 1.0,
    payload_path: str | Path | None = None,
) -> LargeMeshCompressed:
    """Load OBJ, compress, optionally write payload bytes to disk."""
    vertices, faces = load_obj(obj_path)
    c = compress_large_mesh(vertices, faces, n_modes=n_modes, heat_tau=heat_tau)
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
    """Compute compression ratio, distortion and RMS error."""
    from .bounds import compression_ratio_formula, distortion_from_eigenvalues
    V = c.meta["n_vertices"]
    F = c.meta["n_faces"]
    k = c.meta["n_modes"]

    raw_bytes = 8 * 3 * V + 4 * 3 * F
    # coeff_only payload: k×3 coeffs + k eigenvalues + F×3 faces
    comp_bytes = 8 * (k * 3 + k) + 4 * 3 * F
    ratio = raw_bytes / comp_bytes

    tail, total, rel = distortion_from_eigenvalues(
        vertices_original, c.eigenvectors, c.eigenvalues, k
    )
    rms = float(np.sqrt(tail / V)) if V > 0 else 0.0

    return {
        "n_vertices": V,
        "n_faces": F,
        "n_modes": k,
        "raw_bytes": raw_bytes,
        "compressed_bytes (coeff_only)": comp_bytes,
        "compression_ratio (coeff_only)": round(ratio, 2),
        "bits_per_vertex (coeff_only)": round(comp_bytes * 8 / V, 1),
        "spectral_tail_energy": round(tail, 6),
        "total_energy": round(total, 6),
        "relative_distortion": round(rel, 6),
        "rms_vertex_error": round(rms, 6),
        "time_laplacian_s": c.meta.get("time_laplacian_s"),
        "time_eigensolver_s": c.meta.get("time_eigensolver_s"),
    }
