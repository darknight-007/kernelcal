"""
kernelcal.control.analyzer
==========================
High-level per-campaign CARE analyzer for plant phenotyping.

Consumes a rotation-by-rotation stream of kernelcal fixed-point outputs
plus GP ARD trajectories and returns the full Riccati-gain biosignature
time series predicted by Table tab:care_phases of the manuscript.

Intended consumer: the ROS 2 `care_analysis_node` of the
`circular_plant_phenotyping` package, which feeds this class with one
RotationInput per arm rotation and publishes the resulting
CAREAnalyzerState on /care/state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .care import (
    RiccatiAnalysisResult,
    RiccatiConjectureTest,
    OUIdentificationResult,
    ard_to_observation_matrix,
    estimate_A_log_OU,
    fit_riccati_analytic,
    fit_riccati_residual,
    riccati_conjecture_test,
    landauer_R_lower_bound,
)


@dataclass
class CAREAnalyzerConfig:
    """Static configuration for the per-campaign analyzer."""

    # Dimensionality of the kernel-mode state (number of Laplacian modes kept).
    n_modes: int

    # Number of control channels (LED + irrigation).  B has shape (n_modes, n_controls).
    n_controls: int = 4

    # Fisher-Rao state-cost weight in log-coordinates; default from the paper.
    Q_fisher_rao: float = 0.5

    # Minimum number of rotations to accumulate before attempting OU ID.
    ou_min_samples: int = 6

    # Force diagonal A during OU identification (mode-separable case).
    diagonal_A: bool = False

    # Ridge regularization for OU regression.
    ou_regularization: float = 1e-6

    # Landauer temperature for the sanity-check R_ctrl lower bound.
    temperature_kelvin: float = 300.0

    # Per-channel unit Landauer delta_I_k (nats per unit control action).
    # Only used for the physical lower-bound reporting in CAREAnalyzerState.
    # The CARE is solved with R_ctrl = R_ctrl_scale * I (see below); the
    # Landauer floor is compared to this scale and a warning is attached
    # if the scale is below it.
    landauer_delta_I_per_channel: float = 1.0e-3

    # Unit-normalized control cost used in the CARE solve, R_ctrl = scale * I.
    # Choose this so the Q/R ratio places Q and R on comparable energetic
    # scales (Q = 1/2 here).  Landauer bounds the *lower edge* of this
    # scale; the CARE solve itself uses R_ctrl_scale because the Landauer
    # bound is typically 1e-20 smaller than any realistic controller cost.
    R_ctrl_scale: float = 1.0

    # Default actuation matrix B (n_modes, n_controls).  If None, it is
    # assumed identified jointly with A during OU-ID.
    B_default: Optional[np.ndarray] = None

    # Conjecture-test tolerance; matches the paper's reporting convention.
    p_m_target: float = 2.0
    p_m_tolerance: float = 0.10

    # If False, skip the analytic CARE path and always use residual
    # minimization (useful when A is known to be noisy).
    try_analytic_first: bool = True


@dataclass
class RotationInput:
    """Per-rotation input bundle fed to the analyzer."""
    rotation_index: int
    timestamp: float                          # seconds since campaign start
    h_star: np.ndarray                        # (N,) MaxCal fixed-point weights
    D_m: np.ndarray                           # (N,) conservation deficit per mode
    delta_prime: np.ndarray                   # (N,) Hessian margins per mode
    ard_lengthscales: np.ndarray              # (M,) per-band GP length-scales
    control_input: Optional[np.ndarray] = None  # (n_controls,) applied u
    source_jacobian: Optional[np.ndarray] = None  # (N, N) dT/dh at h^*; optional
    mode_basis: Optional[np.ndarray] = None   # (M, N) band -> mode projector


@dataclass
class CAREAnalyzerState:
    """State emitted once per rotation after ingest()."""
    rotation_index: int
    timestamp: float
    ou: Optional[OUIdentificationResult]
    riccati: Optional[RiccatiAnalysisResult]
    conjecture: Optional[RiccatiConjectureTest]
    C_obs: np.ndarray
    R_ctrl_floor: float
    status: str                               # 'warmup' | 'fitted' | 'degraded'
    note: str = ""


class PlantPhenotypingCAREAnalyzer:
    """Rotation-driven CARE analyzer.

    Usage
    -----
    >>> cfg = CAREAnalyzerConfig(n_modes=8, n_controls=4)
    >>> analyzer = PlantPhenotypingCAREAnalyzer(cfg)
    >>> for rot_input in campaign:
    ...     state = analyzer.ingest(rot_input)
    ...     publish(state)
    """

    def __init__(self, cfg: CAREAnalyzerConfig) -> None:
        self.cfg = cfg
        self._log_h_history: List[np.ndarray] = []
        self._control_history: List[np.ndarray] = []
        self._timestamps: List[float] = []
        self._log_h_reference: Optional[np.ndarray] = None
        self._last_P: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Reference fixed-point anchor (pre-stress window)
    # ------------------------------------------------------------------

    def set_reference_fixed_point(self, h_star_ref: np.ndarray) -> None:
        """Pin the log-coordinate anchor log h^* used for delta_ell.

        Typically set from the rolling mean of h_star during the
        pre-stress window (days 1-3 of the drought campaign).
        """
        h_star_ref = np.asarray(h_star_ref, dtype=float)
        self._log_h_reference = np.log(np.maximum(h_star_ref, 1e-12))

    # ------------------------------------------------------------------
    # Per-rotation ingest
    # ------------------------------------------------------------------

    def ingest(self, rot: RotationInput) -> CAREAnalyzerState:
        """Ingest one rotation; return the updated analyzer state."""
        cfg = self.cfg
        h_star = np.asarray(rot.h_star, dtype=float)
        if h_star.shape != (cfg.n_modes,):
            raise ValueError(
                f"h_star must have shape ({cfg.n_modes},); got {h_star.shape}."
            )

        if self._log_h_reference is None:
            # First rotation: anchor the log-coordinate frame here.
            self._log_h_reference = np.log(np.maximum(h_star, 1e-12))

        delta_ell = (
            np.log(np.maximum(h_star, 1e-12)) - self._log_h_reference
        )
        self._log_h_history.append(delta_ell)
        self._timestamps.append(float(rot.timestamp))
        if rot.control_input is not None:
            u = np.asarray(rot.control_input, dtype=float)
            if u.shape != (cfg.n_controls,):
                raise ValueError(
                    f"control_input must have shape ({cfg.n_controls},); "
                    f"got {u.shape}."
                )
            self._control_history.append(u)
        else:
            self._control_history.append(np.zeros(cfg.n_controls, dtype=float))

        C_obs = ard_to_observation_matrix(
            rot.ard_lengthscales, mode_basis=rot.mode_basis, normalize=True,
        )

        total_dI = max(
            cfg.landauer_delta_I_per_channel
            * float(np.linalg.norm(self._control_history[-1], 1)),
            1e-9,
        )
        R_floor = landauer_R_lower_bound(total_dI, cfg.temperature_kelvin)
        R_ctrl_matrix = cfg.R_ctrl_scale * np.eye(cfg.n_controls)

        if len(self._log_h_history) < cfg.ou_min_samples:
            return CAREAnalyzerState(
                rotation_index=rot.rotation_index,
                timestamp=rot.timestamp,
                ou=None,
                riccati=None,
                conjecture=None,
                C_obs=C_obs,
                R_ctrl_floor=R_floor,
                status="warmup",
                note=(
                    f"Accumulating samples: "
                    f"{len(self._log_h_history)}/{cfg.ou_min_samples}."
                ),
            )

        X = np.array(self._log_h_history, dtype=float)
        U = np.array(self._control_history, dtype=float)
        dt = self._mean_dt()

        ou = estimate_A_log_OU(
            X,
            dt=dt,
            control=U if cfg.B_default is None else None,
            diagonal_only=cfg.diagonal_A,
            regularization=cfg.ou_regularization,
        )
        A = ou.A
        B = ou.B if cfg.B_default is None else np.asarray(cfg.B_default)

        Q = cfg.Q_fisher_rao * np.eye(cfg.n_modes)
        R_ctrl = R_ctrl_matrix

        riccati = self._solve_care(A, B, Q, R_ctrl)
        conj = riccati_conjecture_test(
            riccati.P,
            p_m_target=cfg.p_m_target,
            tolerance=cfg.p_m_tolerance,
        )
        self._last_P = riccati.P

        status = "fitted" if riccati.converged else "degraded"
        note = ""
        if not riccati.converged:
            note = (
                f"Residual {riccati.residual_frobenius:.3e}; check stress "
                f"onset, mode coupling, or numerical conditioning."
            )
        return CAREAnalyzerState(
            rotation_index=rot.rotation_index,
            timestamp=rot.timestamp,
            ou=ou,
            riccati=riccati,
            conjecture=conj,
            C_obs=C_obs,
            R_ctrl_floor=R_floor,
            status=status,
            note=note,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _mean_dt(self) -> float:
        if len(self._timestamps) < 2:
            return 1.0
        dts = np.diff(self._timestamps)
        dts = dts[dts > 0]
        if dts.size == 0:
            return 1.0
        return float(np.mean(dts))

    def _solve_care(
        self,
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R_ctrl: np.ndarray,
    ) -> RiccatiAnalysisResult:
        if self.cfg.try_analytic_first:
            try:
                P = fit_riccati_analytic(A, B, Q, R_ctrl)
                from .care import (
                    care_residual,
                    off_diagonal_frobenius,
                    coupling_entropy_off_diagonal,
                )
                res_mat = care_residual(P, A, B, Q, R_ctrl)
                res_fro = float(np.linalg.norm(res_mat, ord="fro"))
                return RiccatiAnalysisResult(
                    P=P,
                    residual_frobenius=res_fro,
                    method="analytic",
                    converged=res_fro < 1e-6,
                    diagonal=np.diag(P).copy(),
                    off_diagonal_mass=off_diagonal_frobenius(P),
                    coupling_entropy=coupling_entropy_off_diagonal(P),
                    eigvals=np.linalg.eigvalsh(P),
                )
            except Exception:
                pass

        return fit_riccati_residual(
            A, B, Q, R_ctrl,
            P_init=self._last_P,
            enforce_psd=True,
        )
