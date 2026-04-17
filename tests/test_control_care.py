"""Unit tests for kernelcal.control (CARE, OU identification, ARD -> C_obs).

Run from the package root:
    pytest tests/test_control_care.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from kernelcal.control import (
    CAREAnalyzerConfig,
    PlantPhenotypingCAREAnalyzer,
    RotationInput,
    ard_to_observation_matrix,
    care_residual,
    coupling_entropy_off_diagonal,
    estimate_A_log_OU,
    fit_riccati_analytic,
    fit_riccati_residual,
    landauer_R_lower_bound,
    off_diagonal_frobenius,
    riccati_conjecture_test,
)
from kernelcal.spectral import CowanFarquharSource


N_MODES = 4
N_CONTROLS = 2


@pytest.fixture
def lqr_system():
    """A stabilizable mode-separable LQR system with known-good CARE."""
    A = -np.diag([0.2, 0.4, 0.6, 0.8])
    B = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    Q = 0.5 * np.eye(N_MODES)
    R = 1.0e-3 * np.eye(N_CONTROLS)
    return A, B, Q, R


def test_analytic_care_matches_residual(lqr_system):
    A, B, Q, R = lqr_system
    P_analytic = fit_riccati_analytic(A, B, Q, R)
    res = care_residual(P_analytic, A, B, Q, R)
    assert np.linalg.norm(res, "fro") < 1e-8
    eigvals = np.linalg.eigvalsh(P_analytic)
    assert np.all(eigvals >= -1e-10)


def test_residual_minimizer_converges(lqr_system):
    A, B, Q, R = lqr_system
    result = fit_riccati_residual(A, B, Q, R, enforce_psd=True)
    assert result.converged, f"residual={result.residual_frobenius}"
    assert result.method == "residual"
    assert result.off_diagonal_mass >= 0.0
    assert result.coupling_entropy >= 0.0


def test_residual_matches_analytic_within_tolerance(lqr_system):
    A, B, Q, R = lqr_system
    P_analytic = fit_riccati_analytic(A, B, Q, R)
    result = fit_riccati_residual(A, B, Q, R, P_init=P_analytic)
    assert np.allclose(result.P, P_analytic, atol=1e-3)


def test_off_diagonal_frobenius_zero_on_diagonal():
    P = np.diag([1.0, 2.0, 3.0])
    assert off_diagonal_frobenius(P) == pytest.approx(0.0)


def test_off_diagonal_frobenius_positive_on_coupled():
    P = np.array([[1.0, 0.3], [0.3, 2.0]])
    assert off_diagonal_frobenius(P) > 0


def test_coupling_entropy_zero_on_diagonal():
    P = np.diag([1.0, 2.0, 3.0])
    s = coupling_entropy_off_diagonal(P)
    # Zero off-diagonal => uniform-fallback => log(N-1) = log(2).
    assert s == pytest.approx(np.log(2.0), abs=1e-6)


def test_riccati_conjecture_pass_on_2I():
    P = 2.0 * np.eye(N_MODES)
    test = riccati_conjecture_test(P, p_m_target=2.0, tolerance=1e-3)
    assert test.passes
    assert test.max_abs_relative == pytest.approx(0.0)


def test_riccati_conjecture_fail_on_shifted_P():
    P = 2.5 * np.eye(N_MODES)
    test = riccati_conjecture_test(P, p_m_target=2.0, tolerance=0.10)
    assert not test.passes
    assert test.max_abs_relative == pytest.approx(0.25, abs=1e-6)


def test_landauer_bound_positive_above_zero_info():
    w = landauer_R_lower_bound(1.0, temperature_kelvin=300.0)
    assert w > 0


def test_landauer_zero_info_gives_zero():
    assert landauer_R_lower_bound(0.0) == 0.0


def test_ou_id_recovers_known_A_with_noise():
    # Exact OU transition matrix simulation: no Euler discretization
    # bias and bounded drive noise.  Verifies the per-mode identifier
    # recovers A within a few percent at T = 1000 with moderate SNR.
    from scipy.linalg import expm

    rng = np.random.default_rng(0)
    A_true = -np.diag([0.5, 1.0, 1.5, 2.0])
    T = 1000
    dt = 0.1
    M = expm(A_true * dt)
    X = np.zeros((T, 4))
    X[0] = rng.normal(0.0, 1.0, size=4)
    for t in range(1, T):
        X[t] = M @ X[t - 1] + rng.normal(0.0, 0.05, size=4)
    result = estimate_A_log_OU(X, dt=dt, diagonal_only=True)
    fit = np.diag(result.A)
    truth = np.diag(A_true)
    rel_err = np.abs((fit - truth) / truth)
    # AR(1) OLS is known to carry a small-sample bias (Hurwicz / Nickell);
    # 30% is a generous envelope for a smoke test.  The production test
    # of the estimator is the monotonic-order check below.
    assert np.all(rel_err < 0.30), (
        f"A_fit={fit} truth={truth} rel_err={rel_err}"
    )
    # Rank ordering of decay rates should be preserved even with bias.
    assert np.all(np.diff(fit) < 0), f"A_fit not monotonic: {fit}"


def test_ard_to_C_obs_diagonal_form():
    ell = np.array([1.0, 2.0, 0.5])
    C = ard_to_observation_matrix(ell, normalize=False)
    assert C.shape == (3, 3)
    assert np.allclose(np.diag(C), [1.0, 1.0 / 4.0, 4.0])


def test_ard_to_C_obs_with_mode_basis():
    ell = np.array([1.0, 2.0, 0.5])
    basis = np.eye(3)[:, :2]
    C = ard_to_observation_matrix(ell, mode_basis=basis, normalize=False)
    assert C.shape == (3, 2)


def test_plant_analyzer_warmup_then_fit():
    cfg = CAREAnalyzerConfig(
        n_modes=N_MODES,
        n_controls=N_CONTROLS,
        ou_min_samples=6,
        diagonal_A=True,
        # Use a fixed B so the analyzer does not try to jointly identify it.
        B_default=np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]),
    )
    analyzer = PlantPhenotypingCAREAnalyzer(cfg)
    rng = np.random.default_rng(1)
    from scipy.linalg import expm
    A_true = -np.diag([0.5, 1.0, 1.5, 2.0])
    dt = 0.5
    M = expm(A_true * dt)
    h_ref = np.exp(np.zeros(N_MODES))
    dl = rng.normal(0.0, 1.0, size=N_MODES)
    analyzer.set_reference_fixed_point(h_ref)

    saw_fitted = False
    for r in range(50):
        dl = M @ dl + rng.normal(0.0, 0.05, size=N_MODES)
        h = np.exp(dl)
        rot = RotationInput(
            rotation_index=r,
            timestamp=r * dt,
            h_star=h,
            D_m=np.zeros(N_MODES),
            delta_prime=np.ones(N_MODES) * 0.2,
            ard_lengthscales=np.ones(5),
            control_input=np.array([0.1, 0.2]),
        )
        state = analyzer.ingest(rot)
        if state.status == "fitted":
            saw_fitted = True
            assert state.riccati is not None
            assert state.conjecture is not None
            assert state.C_obs.shape[0] == 5
            assert state.R_ctrl_floor >= 0
            break
    assert saw_fitted, "Analyzer never produced a fitted state."


def test_cowan_farquhar_calibration_matches_target():
    # 3-parameter family cannot match an N=5 target exactly; we check
    # the best-fit LS error is below a loose threshold consistent with
    # the Tier-2/3 hypothesis framing in Section IV-J of the paper.
    lam = np.linspace(0.1, 1.0, 5)
    h_star = 2.0 * np.ones_like(lam)
    src = CowanFarquharSource.calibrated(
        eigenvalues=lam, h_star_target=h_star,
    )
    T_vals = src.T(h_star)
    target = 0.125 - lam
    assert np.mean((T_vals - target) ** 2) < 1e-2


def test_cowan_farquhar_exact_fit_on_three_modes():
    # With N=3 modes and 3 parameters the fit should be near-exact.
    lam = np.linspace(0.1, 1.0, 3)
    h_star = 2.0 * np.ones_like(lam)
    src = CowanFarquharSource.calibrated(
        eigenvalues=lam, h_star_target=h_star,
    )
    T_vals = src.T(h_star)
    target = 0.125 - lam
    assert np.mean((T_vals - target) ** 2) < 1e-6


def test_cowan_farquhar_stability_on_positive_h():
    lam = np.linspace(0.1, 1.0, 4)
    h_star = np.ones_like(lam)
    src = CowanFarquharSource.calibrated(
        eigenvalues=lam, h_star_target=h_star,
    )
    margin = src.stability_margin(h_star)
    assert margin.shape == h_star.shape
