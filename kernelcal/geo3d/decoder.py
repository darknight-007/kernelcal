"""Digital twin decoder — receiving end of the spectral kernel pipeline.

Three-stage pipeline corresponding to the paper (§5–6):

  Stage 1 — Skeleton reconstruction
      Reconstruct base geometry from k >= beta_0 + beta_1 spectral modes.
      Guarantees topology-preserving decompression (Theorem 1).

  Stage 2 — Conservation-deficit triage (D_m gate)
      Compute per-mode conservation residuals D_{m,t} from the transmitted
      telemetry.  High |D_t| triggers a detail-request flag or more
      aggressive procedural synthesis.

  Stage 3 — Detail synthesis dispatch
      Route to detail_synthesis.py based on available diagnostics:
        - Spectral entropy  H[h*]  → procedural noise roughness
        - Curl energy  E_curl       → flow/channel texture activation
        - Latent scene code         → pre-cached texture library
        - Sparse landmark points    → Poisson-style surface pinning

All spectral/topological computation is in kernelcal; geometry rendering
and visualization are delegated to Blender (twin_receiver.py) or MeshLab
(via OBJ export) or ROS2 (via DigitalTwinUpdate messages).

Paper references
----------------
  D_m = -Delta'               Proposition 2 (stability-conservation tradeoff)
  k_min = beta_0 + beta_1     Theorem 1 (topological conservation)
  H[h*], E_curl               Proposition 4 (channel spectral signature)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from .large_mesh import (
    LargeMeshCompressed,
    decompress_large_mesh,
    _write_obj,
)
from .hodge import boundary_1, boundary_2, hodge_decompose

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Telemetry packet (what the rover/encoder transmits)
# ---------------------------------------------------------------------------

@dataclass
class SpectralTelemetry:
    """All fields transmitted by the rover encoder per observation cycle.

    Mandatory
    ---------
    compressed : LargeMeshCompressed
        Spectral eigenpairs, heat-kernel weights, vertex coefficients, faces.
    betti : tuple[int, int, int]
        (beta_0, beta_1, beta_2) computed on-board.

    Optional diagnostics (paper §5–6)
    ----------------------------------
    D_m_residuals : per-mode conservation residuals |D_{m,t}|  shape (k,)
    spectral_entropy : H[h*_t] scalar
    delta_prime : stability margin Delta' = -D_m (uniform in mode-separable case)
    curl_energy : E_curl = ||B2 omega*||^2 scalar
    latent_code : integer index into scene-class library (None = unknown)
    landmark_xyz : sparse high-confidence LiDAR points (N_lm, 3)
    """

    compressed: LargeMeshCompressed
    betti: tuple[int, int, int]

    # diagnostics — all optional, decoder degrades gracefully without them
    D_m_residuals: np.ndarray | None = None        # (k,) per-mode |D_{m,t}|
    spectral_entropy: float | None = None          # H[h*]
    delta_prime: float | None = None               # stability margin |Delta'|
    curl_energy: float | None = None               # E_curl
    latent_code: int | None = None                 # scene-class index
    landmark_xyz: np.ndarray | None = None         # (N_lm, 3) pin points
    timestamp: float | None = None
    frame_id: str = "map"


# ---------------------------------------------------------------------------
# Decoder output
# ---------------------------------------------------------------------------

class DetailLevel(Enum):
    """Decoder decision on required detail synthesis."""
    SKELETON_ONLY  = "skeleton_only"    # k < k_min or missing diagnostics
    LOW            = "low"              # D_t small, H low → light noise
    MEDIUM         = "medium"           # D_t moderate or curl active
    HIGH           = "high"             # D_t large → aggressive synthesis + patch request
    PATCH_REQUEST  = "patch_request"    # D_t exceeds hard threshold — request re-transmission


@dataclass
class DecodedTwin:
    """Output of the decoder, ready for rendering or ROS2 publication.

    vertices_base  : (V, 3) reconstructed from spectral modes (Stage 1)
    faces          : (F, 3) triangle connectivity
    detail_level   : DetailLevel enum (Stage 2 D_m gate)
    detail_params  : dict passed to detail_synthesis.synthesize()
    diagnostics    : per-frame scalar summary for logging / RViz overlays
    request_patch  : True if decoder requests a high-frequency re-transmission
    """

    vertices_base: np.ndarray
    faces: np.ndarray
    detail_level: DetailLevel
    detail_params: dict[str, Any]
    diagnostics: dict[str, Any]
    request_patch: bool = False
    telemetry: SpectralTelemetry | None = None


# ---------------------------------------------------------------------------
# Thresholds (tuneable; match paper §7 calibration targets)
# ---------------------------------------------------------------------------

D_T_THRESHOLD_LOW    = 1.0    # |D_t| per mode: below this → low detail
D_T_THRESHOLD_MED    = 5.0    # between low and high
D_T_THRESHOLD_PATCH  = 20.0   # above this → request re-transmission
ENTROPY_HIGH         = 3.5    # H[h*] above this → complex scene
CURL_ACTIVE          = 0.05   # E_curl fraction triggering flow textures


# ---------------------------------------------------------------------------
# Stage 1 — Skeleton reconstruction
# ---------------------------------------------------------------------------

def reconstruct_skeleton(tel: SpectralTelemetry) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct (vertices, faces) from compressed spectral payload.

    Implements k >= beta_0 + beta_1 guard (Theorem 1):
    if the transmitted k is below k_min, logs a topology-loss warning.
    """
    beta0, beta1, _ = tel.betti
    k_min = beta0 + beta1
    k_actual = tel.compressed.meta.get("n_modes", 0)

    if k_actual < k_min:
        log.warning(
            "Topology loss: transmitted k=%d < k_min=%d (beta_0=%d beta_1=%d). "
            "Reconstructed mesh may be missing %d independent cycle(s).",
            k_actual, k_min, beta0, beta1, k_min - k_actual,
        )

    vertices, faces = decompress_large_mesh(tel.compressed)
    log.debug("Skeleton: V=%d  F=%d  k=%d  k_min=%d",
              vertices.shape[0], faces.shape[0], k_actual, k_min)
    return vertices, faces


# ---------------------------------------------------------------------------
# Stage 2 — D_m conservation-deficit gate
# ---------------------------------------------------------------------------

def _aggregate_D_t(tel: SpectralTelemetry) -> float:
    """Aggregate per-mode residuals into scalar D_t diagnostic."""
    if tel.D_m_residuals is not None and len(tel.D_m_residuals) > 0:
        return float(np.sum(np.abs(tel.D_m_residuals)))
    # Fallback: use delta_prime * k as a proxy
    if tel.delta_prime is not None:
        k = tel.compressed.meta.get("n_modes", 1)
        return abs(float(tel.delta_prime)) * k
    return 0.0


def triage_detail_level(tel: SpectralTelemetry) -> tuple[DetailLevel, dict[str, Any]]:
    """Apply D_m gate to decide detail level and synthesis parameters.

    Returns
    -------
    level        : DetailLevel enum
    detail_params: dict consumed by detail_synthesis.synthesize()
    """
    D_t = _aggregate_D_t(tel)
    H   = tel.spectral_entropy if tel.spectral_entropy is not None else 1.0
    E_c = tel.curl_energy      if tel.curl_energy      is not None else 0.0
    lc  = tel.latent_code

    # Noise roughness scales with spectral entropy (high H → rough terrain)
    roughness = min(1.0, H / ENTROPY_HIGH)
    # Curl activation: flow/channel textures where curl energy is elevated
    curl_active = E_c > CURL_ACTIVE

    params: dict[str, Any] = {
        "roughness": roughness,
        "curl_active": curl_active,
        "curl_energy": E_c,
        "latent_code": lc,
        "D_t": D_t,
        "spectral_entropy": H,
        "landmark_xyz": tel.landmark_xyz,
        "betti": tel.betti,
    }

    if D_t > D_T_THRESHOLD_PATCH:
        level = DetailLevel.PATCH_REQUEST
        log.warning("D_t=%.2f exceeds patch threshold — flagging re-transmission.", D_t)
    elif D_t > D_T_THRESHOLD_MED:
        level = DetailLevel.HIGH
    elif D_t > D_T_THRESHOLD_LOW:
        level = DetailLevel.MEDIUM
    elif tel.compressed.meta.get("n_modes", 0) < (tel.betti[0] + tel.betti[1]):
        level = DetailLevel.SKELETON_ONLY
    else:
        level = DetailLevel.LOW

    return level, params


# ---------------------------------------------------------------------------
# Stage 3 — Curl energy computation from reconstructed mesh
# ---------------------------------------------------------------------------

def compute_curl_energy(
    vertices: np.ndarray,
    faces: np.ndarray,
    drainage_signal: np.ndarray | None = None,
) -> float:
    """Compute E_curl = ||B2 omega*||^2 for a synthetic drainage signal.

    If no drainage signal is provided, uses vertex height as a proxy
    (gradient of z-coordinate as a scalar potential).

    Returns
    -------
    curl_energy : float  E_curl / E_total  (fractional curl energy)
    """
    n_v = vertices.shape[0]
    try:
        B1 = boundary_1(n_v, faces)
        B2 = boundary_2(faces)
    except Exception as e:
        log.debug("Curl energy fallback (boundary ops failed): %s", e)
        return 0.0

    # Default drainage signal: z-gradient across edges.
    # B1 has shape (n_V, n_E); column e has exactly two nonzero entries:
    # B1[tail, e] = -1  and  B1[head, e] = +1.
    if drainage_signal is None:
        z = vertices[:, 2]
        n_e = B2.shape[0]
        drainage_signal = np.zeros(n_e, dtype=float)
        B1_csc = B1.tocsc()
        for eid in range(n_e):
            col = B1_csc.getcol(eid)
            rows = col.nonzero()[0]
            if len(rows) == 2:
                data = np.asarray(col[rows, 0]).ravel()
                tail = int(rows[np.argmin(data)])  # entry = -1
                head = int(rows[np.argmax(data)])  # entry = +1
                drainage_signal[eid] = z[head] - z[tail]

    try:
        grad, curl, harmonic = hodge_decompose(drainage_signal, B1, B2)
        E_total = float(np.dot(drainage_signal, drainage_signal))
        E_curl  = float(np.dot(curl, curl))
        return E_curl / E_total if E_total > 1e-12 else 0.0
    except Exception as e:
        log.debug("Curl decomposition failed: %s", e)
        return 0.0


# ---------------------------------------------------------------------------
# Main decoder entry point
# ---------------------------------------------------------------------------

def decode(
    tel: SpectralTelemetry,
    *,
    compute_curl: bool = True,
) -> DecodedTwin:
    """Full three-stage decoder.

    Parameters
    ----------
    tel          : SpectralTelemetry from the rover encoder
    compute_curl : if True and tel.curl_energy is None, estimate it locally

    Returns
    -------
    DecodedTwin  ready for detail_synthesis.synthesize() or direct export
    """
    # Stage 1
    vertices, faces = reconstruct_skeleton(tel)

    # Optionally estimate curl energy if not transmitted
    if compute_curl and tel.curl_energy is None:
        tel.curl_energy = compute_curl_energy(vertices, faces)
        log.debug("Estimated curl energy: %.4f", tel.curl_energy)

    # Stage 2
    level, params = triage_detail_level(tel)

    diagnostics = {
        "D_t": params["D_t"],
        "spectral_entropy": params["spectral_entropy"],
        "curl_energy": params["curl_energy"],
        "curl_active": params["curl_active"],
        "detail_level": level.value,
        "n_vertices": int(vertices.shape[0]),
        "n_faces": int(faces.shape[0]),
        "k_transmitted": tel.compressed.meta.get("n_modes"),
        "k_min": tel.betti[0] + tel.betti[1],
        "beta": tel.betti,
        "latent_code": params["latent_code"],
        "timestamp": tel.timestamp,
        "frame_id": tel.frame_id,
    }

    return DecodedTwin(
        vertices_base=vertices,
        faces=faces,
        detail_level=level,
        detail_params=params,
        diagnostics=diagnostics,
        request_patch=(level == DetailLevel.PATCH_REQUEST),
        telemetry=tel,
    )


# ---------------------------------------------------------------------------
# OBJ export helper (for MeshLab / downstream consumers)
# ---------------------------------------------------------------------------

def export_decoded_obj(twin: DecodedTwin, path: str) -> None:
    """Write base-geometry OBJ for MeshLab or RViz mesh display."""
    from pathlib import Path
    _write_obj(twin.vertices_base, twin.faces, Path(path))
    log.info("Decoded OBJ written: %s", path)
