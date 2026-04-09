"""
Minimal regression tests for kernelcal.attention.

Run: pytest tests/test_attention_core.py -v
"""

import numpy as np
import pytest


class TestAttentionKernel:
    def test_synthetic_converges(self):
        from kernelcal.attention.kernel import AttentionKernel
        ak = AttentionKernel.synthetic(seq_len=16, seed=0)
        result = ak.analyse()
        assert result.converged
        assert result.residual_inf_norm < 1e-6

    def test_spectral_entropy_positive(self):
        from kernelcal.attention.kernel import AttentionKernel
        ak = AttentionKernel.synthetic(seq_len=16, seed=42)
        result = ak.analyse()
        assert result.spectral_entropy > 0

    def test_fiedler_gap_above_vacuum(self):
        from kernelcal.attention.kernel import AttentionKernel
        ak = AttentionKernel.synthetic(seq_len=16, seed=7)
        result = ak.analyse()
        assert result.fiedler_gap > np.e * 0.9  # near or above vacuum bound

    def test_identity_attention(self):
        """Pure identity attention (uniform diagonal) should converge."""
        from kernelcal.attention.kernel import AttentionKernel
        A = np.eye(8) * 0.8 + np.ones((8, 8)) * 0.025
        ak = AttentionKernel(A)
        result = ak.analyse()
        assert result.converged


class TestCKA:
    def test_identical_matrices_cka_one(self):
        from kernelcal.attention.landauer import centered_kernel_alignment
        K = np.random.default_rng(0).standard_normal((10, 10))
        K = K @ K.T
        cka = centered_kernel_alignment(K, K)
        assert abs(cka - 1.0) < 1e-6

    def test_different_matrices_cka_less_than_one(self):
        from kernelcal.attention.landauer import centered_kernel_alignment
        rng = np.random.default_rng(0)
        K1 = rng.standard_normal((10, 10)); K1 = K1 @ K1.T
        K2 = rng.standard_normal((10, 10)); K2 = K2 @ K2.T
        cka = centered_kernel_alignment(K1, K2)
        assert cka < 1.0


class TestEnergyMonitor:
    def test_auto_detect_creates_monitor(self):
        from kernelcal.attention.energy import EnergyMonitor
        m = EnergyMonitor.auto_detect()
        assert "flops_estimate" in m._sources

    def test_flops_estimate(self):
        from kernelcal.attention.energy import EnergyMonitor
        est = EnergyMonitor.estimate_flops_energy(
            n_params=1_000_000, n_steps=1000, batch_tokens=512
        )
        assert est['total_flops'] > 0
        assert est['joules'] > 0

    def test_start_stop(self):
        from kernelcal.attention.energy import EnergyMonitor
        m = EnergyMonitor.auto_detect()
        m.start()
        m.add_training_step(n_params=100_000, batch_tokens=64)
        report = m.stop()
        assert report.elapsed_s >= 0
        assert report.flops_estimate > 0


class TestPerturbationFit:
    def test_exponential_decay_fitting(self):
        from kernelcal.attention.perturbation import _fit_exponential_decay
        steps = np.arange(0, 100, 5)
        true_alpha = 0.05
        displacements = 1.0 * np.exp(-true_alpha * steps) + 1e-4
        alpha, r2 = _fit_exponential_decay(steps, displacements)
        assert abs(alpha - true_alpha) < 0.01
        assert r2 > 0.95
