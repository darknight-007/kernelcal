"""
kernelcal.control
=================
CARE (Continuous Algebraic Riccati Equation) solvers and MaxCal-optimal
controller identification for kernel-space dynamics.

Maps to Section IV-J of the plant-phenotyping manuscript and open problem
Q12 of the companion spectral-kernel-dynamics papers.  The primary use
cases are:

* Solving the geometric CARE in log-coordinates around a MaxCal fixed
  point, with Fisher-Rao Q = (1/2) I and Landauer-bounded R_ctrl.
* Identifying the OU mean-reversion matrix A from a log-coordinate kernel
  trajectory produced by `kernelcal.spectral.SpectralKernelDynamics`.
* Converting a Gaussian-process ARD length-scale trajectory into an
  empirical observation matrix C_obs (Section IV-J, Eq. 3 of the paper).
* Reporting the off-diagonal Riccati-gain biosignature:
  the off-diagonal Frobenius mass of P, the coupling entropy
  S_coup(P), and the mode-wise conformance to the p_m = 2 conjecture.

Quick-start
-----------
>>> from kernelcal.control import (
...     fit_riccati_analytic, fit_riccati_residual,
...     estimate_A_log_OU, ard_to_observation_matrix,
...     riccati_conjecture_test, RiccatiAnalysisResult,
...     PlantPhenotypingCAREAnalyzer,
... )
"""

from .care import (
    fit_riccati_analytic,
    fit_riccati_residual,
    care_residual,
    estimate_A_log_OU,
    ard_to_observation_matrix,
    coupling_entropy_off_diagonal,
    off_diagonal_frobenius,
    riccati_conjecture_test,
    landauer_R_lower_bound,
    RiccatiAnalysisResult,
    RiccatiConjectureTest,
    OUIdentificationResult,
)
from .analyzer import (
    PlantPhenotypingCAREAnalyzer,
    CAREAnalyzerConfig,
    RotationInput,
    CAREAnalyzerState,
)

__all__ = [
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
]
