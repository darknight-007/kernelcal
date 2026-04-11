"""Mesh connectivity Laplacian and spectral mesh/DAE compression."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

from ..spectral import SpectralGraph
from .graph3d import adjacency_to_laplacian
from .spectral_codec import CompressedSpectralKernel


def _edges_from_triangles(faces: np.ndarray) -> set[tuple[int, int]]:
    """Extract undirected edge set from triangle faces."""
    f = np.asarray(faces, dtype=int)
    if f.ndim != 2 or f.shape[1] < 3:
        raise ValueError("faces must have shape (T, 3) or more.")
    edges: set[tuple[int, int]] = set()
    for tri in f[:, :3]:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for u, v in ((a, b), (b, c), (c, a)):
            if u > v:
                u, v = v, u
            edges.add((u, v))
    return edges


def mesh_combinatorial_laplacian(n_vertices: int, faces: np.ndarray) -> np.ndarray:
    """Build combinatorial Laplacian from mesh connectivity."""
    if n_vertices < 2:
        raise ValueError("n_vertices must be at least 2.")
    edges = _edges_from_triangles(faces)
    W = np.zeros((n_vertices, n_vertices), dtype=float)
    for u, v in edges:
        W[u, v] = 1.0
        W[v, u] = 1.0
    return adjacency_to_laplacian(W)


def compress_mesh_geometry(
    vertices_xyz: np.ndarray,
    faces: np.ndarray,
    *,
    n_modes: int = 48,
    heat_tau: float | None = 0.5,
) -> CompressedSpectralKernel:
    """Compress mesh graph into truncated spectral kernel coordinates."""
    v = np.asarray(vertices_xyz, dtype=float)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError("vertices_xyz must have shape (V, 3).")

    L = mesh_combinatorial_laplacian(v.shape[0], faces)
    sg = SpectralGraph(L)
    k = min(int(n_modes), sg.N)
    if k < 1:
        raise ValueError("n_modes must be at least 1.")
    lam = sg.eigenvalues[:k]
    Phi = sg.eigenvectors[:, :k]
    if heat_tau is not None and heat_tau > 0:
        h = np.exp(-lam * float(heat_tau))
    else:
        h = np.ones(k)
    h = np.maximum(h, 1e-12)
    meta: dict[str, Any] = {
        "kind": "mesh",
        "n_vertices": int(v.shape[0]),
        "n_faces": int(np.asarray(faces).shape[0]),
        "n_modes": int(k),
        "heat_tau": None if heat_tau is None else float(heat_tau),
    }
    return CompressedSpectralKernel(eigenvalues=lam, eigenvectors=Phi, h=h, meta=meta)


@dataclass
class CompressedMeshGeometry:
    """Compact mesh representation suitable for DAE round-trip."""

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    h: np.ndarray
    vertex_coeffs: np.ndarray  # (k, 3)
    faces: np.ndarray  # (F, 3) int
    meta: dict[str, Any]

    def to_bytes(self) -> bytes:
        """Serialize compressed mesh payload."""
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
    def from_bytes(cls, payload: bytes) -> "CompressedMeshGeometry":
        """Deserialize compressed mesh payload."""
        z = np.load(BytesIO(payload), allow_pickle=True)
        raw = z["meta"].item() if "meta" in z.files else {}
        meta = raw if isinstance(raw, dict) else {}
        return cls(
            eigenvalues=np.asarray(z["eigenvalues"]),
            eigenvectors=np.asarray(z["eigenvectors"]),
            h=np.asarray(z["h"]),
            vertex_coeffs=np.asarray(z["vertex_coeffs"]),
            faces=np.asarray(z["faces"], dtype=int),
            meta=meta,
        )


def compress_mesh_roundtrip(
    vertices_xyz: np.ndarray,
    faces: np.ndarray,
    *,
    n_modes: int = 48,
    heat_tau: float | None = 0.5,
) -> CompressedMeshGeometry:
    """Compress mesh for geometry reconstruction using spectral coefficients."""
    v = np.asarray(vertices_xyz, dtype=float)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError("vertices_xyz must have shape (V, 3).")
    f = np.asarray(faces, dtype=int)
    if f.ndim != 2 or f.shape[1] < 3:
        raise ValueError("faces must have shape (F, 3) or more.")
    if np.min(f[:, :3]) < 0 or np.max(f[:, :3]) >= v.shape[0]:
        raise ValueError("faces contain invalid vertex indices.")

    L = mesh_combinatorial_laplacian(v.shape[0], f[:, :3])
    sg = SpectralGraph(L)
    k = min(int(n_modes), sg.N)
    if k < 1:
        raise ValueError("n_modes must be at least 1.")
    lam = sg.eigenvalues[:k]
    Phi = sg.eigenvectors[:, :k]

    if heat_tau is not None and heat_tau > 0:
        h = np.exp(-lam * float(heat_tau))
    else:
        h = np.ones(k)
    h = np.maximum(h, 1e-12)

    # Spectral coefficients for vertex coordinates.
    coeffs = Phi.T @ v
    meta: dict[str, Any] = {
        "kind": "mesh_roundtrip",
        "n_vertices": int(v.shape[0]),
        "n_faces": int(f.shape[0]),
        "n_modes": int(k),
        "heat_tau": None if heat_tau is None else float(heat_tau),
    }
    return CompressedMeshGeometry(
        eigenvalues=lam,
        eigenvectors=Phi,
        h=h,
        vertex_coeffs=coeffs,
        faces=f[:, :3],
        meta=meta,
    )


def decompress_mesh_roundtrip(c: CompressedMeshGeometry) -> tuple[np.ndarray, np.ndarray]:
    """Decode compressed mesh into vertices and faces."""
    v_hat = c.eigenvectors @ c.vertex_coeffs
    return np.asarray(v_hat, dtype=float), np.asarray(c.faces, dtype=int)


def _load_trimesh():
    try:
        import trimesh  # type: ignore
    except Exception as e:  # pragma: no cover - environment dependent
        raise ImportError(
            "DAE IO requires trimesh (+ pycollada). "
            "Install with: pip install trimesh pycollada"
        ) from e
    return trimesh


def compress_dae(
    dae_path: str | Path,
    *,
    n_modes: int = 48,
    heat_tau: float | None = 0.5,
    payload_path: str | Path | None = None,
) -> CompressedMeshGeometry:
    """Load a DAE mesh, compress to spectral payload, optionally save payload bytes."""
    trimesh = _load_trimesh()
    loaded = trimesh.load(str(dae_path), force="mesh")
    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.dump(concatenate=True)
    else:
        mesh = loaded
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    compressed = compress_mesh_roundtrip(vertices, faces, n_modes=n_modes, heat_tau=heat_tau)

    if payload_path is not None:
        Path(payload_path).write_bytes(compressed.to_bytes())
    return compressed


def decompress_dae(
    compressed: CompressedMeshGeometry | bytes | str | Path,
    dae_output_path: str | Path,
) -> Path:
    """Decode compressed mesh and export reconstructed geometry to DAE."""
    trimesh = _load_trimesh()

    if isinstance(compressed, CompressedMeshGeometry):
        c = compressed
    elif isinstance(compressed, (str, Path)):
        c = CompressedMeshGeometry.from_bytes(Path(compressed).read_bytes())
    elif isinstance(compressed, bytes):
        c = CompressedMeshGeometry.from_bytes(compressed)
    else:
        raise TypeError("compressed must be CompressedMeshGeometry, bytes, or path.")

    vertices, faces = decompress_mesh_roundtrip(c)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    out_path = Path(dae_output_path)
    mesh.export(str(out_path), file_type="dae")
    return out_path
