"""Regression tests for Q19: sigma_m on P_8.

Locks in the numerical answer produced by sigma_m_p8.py so that any
future change to the fixed-point iteration, the Hessian-to-OU mapping,
or the CARE solvers is caught.

Maps to:
  Note 62b Section 4        -- sigma_m = 1/2 dual Riccati conjecture
  Note 62c Section 3 step 1 -- "compute sigma_m from existing kernelcal
                               data on P_8" (this file IS that step)
  arXiv:2604.09745          -- P_8 parameters sigma^2 = 1, mu_2 = 2

Run:  pytest tests/test_sigma_m_p8.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from kernelcal.spectral import (
    CowanFarquharSource,
    GaussianMISource,
    SpectralGraph,
    SpectralKernelDynamics,
)

# Import from the repo-root one-pager.
from sigma_m_p8 import (
    MU2_DEFAULT,
    N_MODES_DEFAULT,
    P_M_TARGET,
    Q_FISHER_RAO,
    R_CTRL_SCALE_DEFAULT,
    SIGMA2_DEFAULT,
    SIGMA_M_TARGET,
    evaluate_sigma_m,
    find_duality_scale,
    run_q19_report,
)


# ---------------------------------------------------------------------------
# Canonical P_8 fixed-point fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def p8_graph():
    return SpectralGraph.path_graph(N_MODES_DEFAULT)


@pytest.fixture(scope="module")
def gmi_source(p8_graph):
    return GaussianMISource(sigma2=SIGMA2_DEFAULT, mu2=MU2_DEFAULT,
                            eigenvalues=None)


@pytest.fixture(scope="module")
def gmi_result(p8_graph, gmi_source):
    return evaluate_sigma_m(
        p8_graph, gmi_source,
        source_label="gmi_canonical",
        R_ctrl_scale=R_CTRL_SCALE_DEFAULT,
    )


@pytest.fixture(scope="module")
def cf_calibrated_source(p8_graph, gmi_source):
    dyn = SpectralKernelDynamics(p8_graph, gmi_source)
    h_star = dyn.fixed_point_iteration().h_star
    return CowanFarquharSource.calibrated(
        eigenvalues=p8_graph.eigenvalues,
        h_star_target=h_star,
        eigenvalue_weighted=False,
    )


@pytest.fixture(scope="module")
def cf_result(p8_graph, cf_calibrated_source):
    return evaluate_sigma_m(
        p8_graph, cf_calibrated_source,
        source_label="cf_calibrated",
        R_ctrl_scale=R_CTRL_SCALE_DEFAULT,
    )


# ---------------------------------------------------------------------------
# Fixed-point sanity: reproduce the arXiv P_8 Gaussian MI result
# ---------------------------------------------------------------------------


def test_p8_fixed_point_is_uniform(gmi_result):
    """On P_8 with w_l = 1 (mode-blind), h* is uniform across modes."""
    h = gmi_result.h_star
    assert gmi_result.fixed_point_converged
    assert gmi_result.field_residual < 1e-8
    assert h.shape == (N_MODES_DEFAULT,)
    assert np.allclose(h, h[0], atol=1e-10), (
        f"P_8 fixed point must be mode-uniform; got spread {h.max()-h.min():.3e}"
    )


def test_p8_fixed_point_value(gmi_result):
    """h_l* ~ 0.1547 for sigma^2 = 1, mu_2 = 2.

    This value is the self-consistent solution of
        h = exp(-1 - mu_2 / (2 (sigma^2 + h)))
    for mu_2 = 2, sigma^2 = 1, and matches route3_conservation_test.py.
    """
    h = float(gmi_result.h_star[0])
    assert h == pytest.approx(0.15469, abs=1e-4), h


def test_p8_hessian_log_is_diagonal_and_uniform(gmi_result):
    """For a mode-separable source with w_l = 1, H_log = diag(h*^2 H_h)
    and is uniform in the mode-blind Gaussian MI case."""
    A = gmi_result.A_log
    off_diag_mass = np.linalg.norm(A - np.diag(np.diag(A)), ord="fro")
    assert off_diag_mass < 1e-12, off_diag_mass
    diag = np.diag(A)
    assert np.allclose(diag, diag[0], atol=1e-12)
    assert float(diag[0]) == pytest.approx(-0.13684, abs=1e-4)


# ---------------------------------------------------------------------------
# Primal LQR: p_m measurement
# ---------------------------------------------------------------------------


def test_gmi_p_m_is_uniform(gmi_result):
    """p_m is the same across all 8 modes (mode-separable symmetry)."""
    p = gmi_result.p_m
    assert p.shape == (N_MODES_DEFAULT,)
    assert np.allclose(p, p[0], atol=1e-10)


def test_gmi_p_m_value_and_conjecture_deviation(gmi_result):
    """Empirical Q19 closure: locks in p_m ~ 0.5834 under self-dual CARE.

    The p_m = 2 primal Riccati conjecture does NOT hold for the generic
    Gaussian MI source on P_8 at R_ctrl_scale = 1 (self-dual B=C=I).
    The measured p_m is ~0.5834, which is ~71 % below the conjecture.
    """
    p = float(gmi_result.p_m[0])
    assert p == pytest.approx(0.58339, abs=1e-3), p
    # Conjecture fails at the 10 % tolerance inherited from the analyzer.
    assert not gmi_result.p_m_passes
    assert gmi_result.p_m_max_abs_relative == pytest.approx(0.7083, abs=1e-2)


# ---------------------------------------------------------------------------
# Dual LQE: sigma_m measurement
# ---------------------------------------------------------------------------


def test_gmi_sigma_m_equals_p_m_under_self_dual(gmi_result):
    """For symmetric A with B = C = I, Q = W, R = V the primal and dual
    CAREs are identical, so diag(Sigma) = diag(P).  This is a structural
    sanity check on the self-dual setup."""
    assert np.allclose(gmi_result.sigma_m, gmi_result.p_m, atol=1e-10)


def test_gmi_sigma_m_value_and_conjecture_deviation(gmi_result):
    """Empirical Q19 closure: sigma_m ~ 0.5834, vs conjectured 1/2.

    The dual Riccati conjecture sigma_m = 1/2 holds to ~17 % relative
    for the Gaussian MI source on P_8 under the self-dual convention.
    This is the concrete numerical answer requested in Note 62c section 3.
    """
    s = float(gmi_result.sigma_m[0])
    assert s == pytest.approx(0.58339, abs=1e-3), s
    assert gmi_result.sigma_m_max_abs_relative == pytest.approx(0.1668, abs=1e-2)


# ---------------------------------------------------------------------------
# LQR-LQE duality check
# ---------------------------------------------------------------------------


def test_gmi_self_dual_does_not_realize_duality(gmi_result):
    """At R_ctrl_scale = 1 the operational duality P * Sigma = I fails
    by a large margin; p_m * sigma_m = 0.3404 instead of 1.  This
    justifies the R_ctrl_scale sweep in sigma_m_p8.find_duality_scale."""
    assert gmi_result.duality_residual_fro > 1.0
    assert gmi_result.p_times_sigma_mean == pytest.approx(0.3404, abs=1e-2)


def test_gmi_duality_scale_scan_is_in_expected_range(p8_graph, gmi_source):
    """The duality-minimizing R* lies in (1, 10) for Gaussian MI on P_8
    and drives || P Sigma - I ||_F below 0.2."""
    r_best, err_best = find_duality_scale(p8_graph, gmi_source)
    assert 1.0 < r_best < 10.0, r_best
    assert err_best < 0.2, err_best


# ---------------------------------------------------------------------------
# Cowan-Farquhar instrumentation source
# ---------------------------------------------------------------------------


def test_cf_calibrated_spreads_p_m_across_modes(cf_result):
    """Unlike GMI, the calibrated Cowan-Farquhar source breaks the
    mode-uniformity: p_m varies across modes because the source
    T_l(h*) = 1/8 - lambda_l is eigenvalue-dependent."""
    p = cf_result.p_m
    assert p.max() - p.min() > 0.3
    # Not uniform: standard deviation well above GMI's ~0.
    assert float(np.std(p)) > 0.1


def test_cf_calibrated_fixed_point_scales_with_lambda(cf_result, p8_graph):
    """Cowan-Farquhar h* grows with lambda (opposite of the mode-blind GMI
    case).  Mode 0 (lambda = 0) has the smallest h*; last mode has the
    largest."""
    h = cf_result.h_star
    lam = p8_graph.eigenvalues
    # Spearman-style: sort by lambda, h* should be non-decreasing.
    order = np.argsort(lam)
    h_sorted = h[order]
    assert np.all(np.diff(h_sorted) >= -1e-6)


# ---------------------------------------------------------------------------
# End-to-end report smoke test
# ---------------------------------------------------------------------------


def test_run_q19_report_silent_returns_both_sources():
    """run_q19_report(verbose=False) returns a dict with both labels."""
    results = run_q19_report(verbose=False)
    assert len(results) == 2
    labels = list(results.keys())
    assert any("Gaussian MI" in lab for lab in labels)
    assert any("Cowan-Farquhar" in lab for lab in labels)
    for res in results.values():
        assert res.N == N_MODES_DEFAULT
        assert res.fixed_point_converged
        # Every result carries a valid summary string.
        text = res.summary()
        assert "Q19" in text
        assert "p_m" in text
        assert "sigma_m" in text
