"""
Q40 sleep-EEG spectral-entropy pipeline: synthetic-data pytest.

Validates ``kernelcal.bio.sleep_eeg`` without requiring external
polysomnography.  We synthesise a multichannel time series whose
channel-channel covariance kernel has a *known* spectral entropy
profile, then check that the pipeline recovers it.

Two synthetic regimes:

1. **Task-engaged (low-entropy source)**
   Signal is a rank-1 latent driving every channel plus iid noise:
   one dominant eigenmode => normalised entropy much less than 1.

2. **Quiescent (high-entropy source, "vacuum")**
   Signal is an isotropic iid process across all channels:
   roughly uniform eigenspectrum => normalised entropy approaches 1.

The P0.5 "sleep as beam cooling" prediction (Q40) maps NREM -> regime
2 and wake -> regime 1, so the pipeline is validated iff it assigns
*higher* normalised spectral entropy to regime-2 windows than to
regime-1 windows.  A small number of tests also check structural
properties (eigenvalue ordering, bounds on entropy, Welch-t contrast
sign) that must hold independently of the cooling story, so breakage
of either class is diagnostic.

Companion documents
-------------------
- P0.5 Remarks 6.12-6.13 (sleep as beam cooling)
- misc-field-notes/80_brain_as_representational_particle_accelerator.txt (Q40)
- misc-field-notes/81_reflexivity_tattoos_heisenberg_representational_thermodynamics.txt
- kernelcal/bio/sleep_eeg.py
- q40_sleep_eeg.py  (CLI for real EDF+hypnogram runs)
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pytest

from kernelcal.bio import (
    SleepEEGWindow,
    channel_covariance_kernel,
    kernel_spectral_entropy_timeseries,
    stage_contrast,
    summarise_by_stage,
)


# ---------------------------------------------------------------------------
# Synthetic generators
# ---------------------------------------------------------------------------

def _rank1_signal(
    n_channels: int,
    duration_sec: float,
    sample_rate: float,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """(C, T) signal: single latent drives all channels + iid noise.

    Channel-channel covariance has one eigenvalue >> all others, so
    normalised spectral entropy is far below 1.
    """
    T = int(round(duration_sec * sample_rate))
    latent = rng.standard_normal(T)
    loadings = rng.standard_normal(n_channels).reshape(n_channels, 1)
    signal = loadings @ latent.reshape(1, T)
    signal = signal + noise_std * rng.standard_normal((n_channels, T))
    return signal


def _isotropic_signal(
    n_channels: int,
    duration_sec: float,
    sample_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """(C, T) iid Gaussian: covariance ~= identity -> vacuum spectrum."""
    T = int(round(duration_sec * sample_rate))
    return rng.standard_normal((n_channels, T))


def _make_recording_with_stages(
    n_channels: int,
    sample_rate: float,
    wake_epochs: int,
    nrem_epochs: int,
    epoch_sec: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, List[str]]:
    """Concatenate wake (rank-1) and NREM (isotropic) blocks.

    Returns signal of shape (C, T) and a per-epoch stage list.
    """
    wake = _rank1_signal(
        n_channels,
        duration_sec=wake_epochs * epoch_sec,
        sample_rate=sample_rate,
        noise_std=0.3,
        rng=rng,
    )
    nrem = _isotropic_signal(
        n_channels,
        duration_sec=nrem_epochs * epoch_sec,
        sample_rate=sample_rate,
        rng=rng,
    )
    signal = np.concatenate([wake, nrem], axis=1)
    stages = ["W"] * wake_epochs + ["N3"] * nrem_epochs
    return signal, stages


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_recording():
    rng = np.random.default_rng(seed=20260416)
    sample_rate = 100.0
    epoch_sec = 30.0
    signal, stages = _make_recording_with_stages(
        n_channels=8,
        sample_rate=sample_rate,
        wake_epochs=20,
        nrem_epochs=20,
        epoch_sec=epoch_sec,
        rng=rng,
    )
    windows = channel_covariance_kernel(
        signal,
        sample_rate=sample_rate,
        window_sec=epoch_sec,
        stride_sec=epoch_sec,
        stages=stages,
        stage_epoch_sec=epoch_sec,
    )
    return {
        "signal": signal,
        "sample_rate": sample_rate,
        "stages": stages,
        "windows": windows,
        "epoch_sec": epoch_sec,
        "n_channels": 8,
    }


# ---------------------------------------------------------------------------
# Structural tests (must hold regardless of cooling prediction)
# ---------------------------------------------------------------------------

class TestPipelineStructural:
    """Shape, bounds, and ordering invariants of the pipeline."""

    def test_window_count_matches_epoch_count(self, synthetic_recording):
        windows = synthetic_recording["windows"]
        stages = synthetic_recording["stages"]
        assert len(windows) == len(stages), (
            f"expected one window per epoch: got {len(windows)} windows "
            f"vs {len(stages)} stage epochs"
        )

    def test_covariance_matrices_are_symmetric_psd(self, synthetic_recording):
        for w in synthetic_recording["windows"]:
            assert np.allclose(w.covariance, w.covariance.T, atol=1e-10), (
                f"covariance at t={w.t_center:g} is not symmetric"
            )
            min_eig = w.eigvals.min()
            assert min_eig >= -1e-10, (
                f"covariance at t={w.t_center:g} has negative eigenvalue {min_eig:g}"
            )

    def test_eigvals_descending(self, synthetic_recording):
        for w in synthetic_recording["windows"]:
            diffs = np.diff(w.eigvals)
            assert np.all(diffs <= 1e-12), (
                f"eigvals not descending at t={w.t_center:g}: {w.eigvals}"
            )

    def test_normalised_entropy_in_unit_interval(self, synthetic_recording):
        for w in synthetic_recording["windows"]:
            H = w.spectral_entropy_normalised
            assert 0.0 <= H <= 1.0 + 1e-12, (
                f"normalised entropy {H:g} outside [0, 1] at t={w.t_center:g}"
            )

    def test_stage_labels_attached_to_all_windows(self, synthetic_recording):
        labels = {w.stage for w in synthetic_recording["windows"]}
        assert labels == {"W", "N3"}, (
            f"expected exactly {{W, N3}}, got {labels}"
        )

    def test_timeseries_extractor_returns_aligned_arrays(self, synthetic_recording):
        t, H, stages = kernel_spectral_entropy_timeseries(
            synthetic_recording["windows"]
        )
        assert t.shape == H.shape
        assert len(stages) == t.size
        assert np.all(np.diff(t) > 0), "window centres must be strictly increasing"


# ---------------------------------------------------------------------------
# Q40 prediction tests: quiescent regime has higher spectral entropy
# ---------------------------------------------------------------------------

class TestSleepAsBeamCoolingOnSyntheticEEG:
    """Operationalises the P0.5 sleep conjecture as an EEG-shaped test.

    If these fail, the pipeline is not producing the predicted
    contrast on a ground-truth synthetic recording, and no claim
    about real Sleep-EDF data should be based on this pipeline
    until the failure is explained.
    """

    def test_nrem_mean_entropy_exceeds_wake(self, synthetic_recording):
        summary = summarise_by_stage(synthetic_recording["windows"])
        H_wake = summary["W"].mean_entropy
        H_nrem = summary["N3"].mean_entropy
        assert H_nrem > H_wake, (
            f"Q40 prediction violated on synthetic recording: "
            f"NREM mean entropy {H_nrem:.4f} !> Wake mean entropy {H_wake:.4f}"
        )

    def test_contrast_has_large_effect_size(self, synthetic_recording):
        """Rank-1 vs. isotropic is an extreme contrast; Cohen's d must be huge."""
        contrast = stage_contrast(
            synthetic_recording["windows"], stage_a="N3", stage_b="W"
        )
        assert contrast.delta_mean > 0.0, (
            f"delta_mean sign wrong: {contrast.delta_mean:+g}"
        )
        assert contrast.cohens_d > 2.0, (
            f"expected large effect (d > 2) on synthetic rank-1 vs. isotropic; "
            f"got d={contrast.cohens_d:g}"
        )

    def test_nrem_entropy_approaches_vacuum(self, synthetic_recording):
        """Isotropic blocks drive normalised entropy toward 1 (vacuum)."""
        summary = summarise_by_stage(synthetic_recording["windows"])
        H_nrem = summary["N3"].mean_entropy
        assert H_nrem > 0.90, (
            f"isotropic (vacuum-like) source should yield H/log C -> 1; "
            f"got {H_nrem:g} (expected > 0.90)"
        )

    def test_wake_entropy_far_from_vacuum(self, synthetic_recording):
        """Rank-1 block must concentrate on one mode -> low normalised entropy."""
        summary = summarise_by_stage(synthetic_recording["windows"])
        H_wake = summary["W"].mean_entropy
        assert H_wake < 0.60, (
            f"rank-1 source should yield low H/log C; got {H_wake:g}"
        )


# ---------------------------------------------------------------------------
# Seed stability: the sign of the contrast must not depend on RNG seed
# ---------------------------------------------------------------------------

class TestSeedStability:
    """Q40 contrast sign is a property of the generating process, not the seed."""

    @pytest.mark.parametrize("seed", [11, 137, 2026, 43951, 99991])
    def test_nrem_over_wake_across_seeds(self, seed):
        rng = np.random.default_rng(seed)
        fs = 100.0
        epoch = 30.0
        signal, stages = _make_recording_with_stages(
            n_channels=8,
            sample_rate=fs,
            wake_epochs=10,
            nrem_epochs=10,
            epoch_sec=epoch,
            rng=rng,
        )
        windows = channel_covariance_kernel(
            signal, sample_rate=fs, window_sec=epoch, stride_sec=epoch,
            stages=stages, stage_epoch_sec=epoch,
        )
        c = stage_contrast(windows, stage_a="N3", stage_b="W")
        assert c.delta_mean > 0.0, (
            f"seed={seed}: NREM mean entropy should exceed Wake; "
            f"got delta={c.delta_mean:+g}"
        )


# ---------------------------------------------------------------------------
# Degenerate-input guards
# ---------------------------------------------------------------------------

class TestDegenerateInputs:
    """The pipeline should fail loudly on inputs that are not analysable."""

    def test_single_channel_rejected(self):
        signal = np.random.default_rng(0).standard_normal((1, 1000))
        with pytest.raises(ValueError, match="at least 2 channels"):
            channel_covariance_kernel(signal, sample_rate=100.0, window_sec=1.0)

    def test_zero_sample_rate_rejected(self):
        signal = np.random.default_rng(0).standard_normal((4, 1000))
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            channel_covariance_kernel(signal, sample_rate=0.0, window_sec=1.0)

    def test_too_short_window_rejected(self):
        signal = np.random.default_rng(0).standard_normal((4, 1000))
        with pytest.raises(ValueError, match="window/stride too small"):
            channel_covariance_kernel(signal, sample_rate=100.0, window_sec=0.0)

    def test_zero_eigenvalue_signal_returns_zero_entropy(self):
        """All-zero signal -> eigvals all zero -> entropy defined as 0."""
        signal = np.zeros((4, 1000))
        windows = channel_covariance_kernel(
            signal, sample_rate=100.0, window_sec=1.0,
        )
        assert all(w.spectral_entropy_normalised == 0.0 for w in windows)

    def test_stage_contrast_with_too_few_samples_returns_nan(self):
        windows = [
            SleepEEGWindow(0.0, np.eye(2), np.array([1.0, 1.0]), 1.0, "W"),
        ]
        c = stage_contrast(windows, stage_a="N3", stage_b="W")
        assert np.isnan(c.delta_mean)


# ---------------------------------------------------------------------------
# Summariser correctness
# ---------------------------------------------------------------------------

class TestSummariser:

    def test_means_match_direct_computation(self, synthetic_recording):
        windows = synthetic_recording["windows"]
        summary = summarise_by_stage(windows, stages=["W", "N3"])
        for stage in ("W", "N3"):
            direct = np.mean([
                w.spectral_entropy_normalised for w in windows if w.stage == stage
            ])
            assert np.isclose(summary[stage].mean_entropy, direct, atol=1e-12)

    def test_absent_stage_reports_nan_and_zero_n(self, synthetic_recording):
        summary = summarise_by_stage(
            synthetic_recording["windows"], stages=["W", "N1", "N2", "N3", "R"]
        )
        assert summary["R"].n_windows == 0
        assert np.isnan(summary["R"].mean_entropy)
