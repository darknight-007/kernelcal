"""Tests for the PR-A deduplications.

Covers three previously duplicated primitives that now delegate to a
single canonical implementation:

1. ``spectral_entropy`` — one helper in :mod:`kernelcal.spectral.entropy`,
   re-exported from :mod:`kernelcal.spectral.dynamics` and
   :mod:`kernelcal.terrain.diagnostics`, used by
   :mod:`kernelcal.bio.sleep_eeg` via ``normalize=True``.
2. ``hs_distance`` in :mod:`kernelcal.bandits.kernels` now delegates to
   :func:`kernelcal.kernel.space.hilbert_schmidt_distance`.
3. ``LocalFrame`` — one dataclass in :mod:`kernelcal.geo3d.local_frame`,
   re-exported by ``bishop_rocks_graph_explorer`` for back-compat.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# spectral_entropy parity + behavior
# ---------------------------------------------------------------------------


class TestSpectralEntropyUnified:
    def test_reexports_are_same_callable(self):
        """Every historical import site must resolve to the canonical helper."""
        from kernelcal.spectral.entropy import spectral_entropy as canonical
        from kernelcal.spectral.dynamics import spectral_entropy as via_dynamics
        from kernelcal.spectral import spectral_entropy as via_spectral
        from kernelcal.terrain.diagnostics import spectral_entropy as via_terrain
        from kernelcal.terrain import spectral_entropy as via_terrain_root

        assert via_dynamics is canonical
        assert via_spectral is canonical
        assert via_terrain is canonical
        assert via_terrain_root is canonical

    def test_uniform_vector_hits_log_N(self):
        from kernelcal.spectral.entropy import spectral_entropy

        for N in (2, 5, 16):
            h = np.ones(N, dtype=float)
            assert spectral_entropy(h) == pytest.approx(math.log(N), abs=1e-12)
            assert spectral_entropy(h, normalize=True) == pytest.approx(1.0, abs=1e-12)

    def test_single_mode_concentrated(self):
        from kernelcal.spectral.entropy import spectral_entropy

        h = np.zeros(7, dtype=float)
        h[3] = 1.0
        assert spectral_entropy(h) == pytest.approx(0.0, abs=1e-12)
        assert spectral_entropy(h, normalize=True) == pytest.approx(0.0, abs=1e-12)

    def test_all_zero_input_is_zero(self):
        from kernelcal.spectral.entropy import spectral_entropy

        assert spectral_entropy(np.zeros(5)) == 0.0
        assert spectral_entropy(np.zeros(5), normalize=True) == 0.0
        assert spectral_entropy([], normalize=True) == 0.0

    def test_nonfinite_sum_returns_zero(self):
        from kernelcal.spectral.entropy import spectral_entropy

        bad = np.array([np.inf, 1.0, 2.0])
        assert spectral_entropy(bad) == 0.0

    def test_negative_sum_returns_zero(self):
        from kernelcal.spectral.entropy import spectral_entropy

        assert spectral_entropy(np.array([-1.0, -2.0])) == 0.0

    def test_zero_interleaving_equivalent_to_filtered_version(self):
        """Former terrain behavior (filter zeros first) must match the new
        canonical behavior (mask zeros via log(1))."""
        from kernelcal.spectral.entropy import spectral_entropy

        rng = np.random.default_rng(42)
        dense = rng.uniform(0.1, 3.0, size=5)
        padded = np.concatenate([dense, np.zeros(4)])  # insert zeros
        rng.shuffle(padded)

        H_dense = spectral_entropy(dense)
        H_padded = spectral_entropy(padded)
        assert H_padded == pytest.approx(H_dense, abs=1e-12)

    def test_normalize_true_matches_sleep_eeg_private_helper(self):
        """The private ``_normalised_spectral_entropy`` in bio/sleep_eeg
        must delegate to the canonical helper with ``normalize=True``."""
        from kernelcal.bio.sleep_eeg import _normalised_spectral_entropy
        from kernelcal.spectral.entropy import spectral_entropy

        rng = np.random.default_rng(0)
        for trial in range(5):
            eigvals = rng.uniform(0.0, 5.0, size=rng.integers(2, 20))
            a = _normalised_spectral_entropy(eigvals)
            b = spectral_entropy(eigvals, normalize=True)
            assert a == pytest.approx(b, abs=1e-12)

    def test_matches_legacy_formula_on_random_vectors(self):
        """Sanity check: old ``spectral.dynamics`` formula and old
        ``terrain.diagnostics`` formula are numerically identical for any
        nonnegative vector — so the unified implementation must agree with
        both."""
        from kernelcal.spectral.entropy import spectral_entropy

        rng = np.random.default_rng(1)
        for _ in range(10):
            h = rng.uniform(0.0, 10.0, size=rng.integers(2, 12)).astype(float)

            legacy_dynamics = self._legacy_dynamics(h)
            legacy_terrain = self._legacy_terrain(h)
            new = spectral_entropy(h)

            assert legacy_dynamics == pytest.approx(legacy_terrain, abs=1e-12)
            assert new == pytest.approx(legacy_dynamics, abs=1e-12)

    # ----- helpers reproducing the pre-PR-A formulas verbatim -----
    @staticmethod
    def _legacy_dynamics(h):
        h = np.asarray(h, dtype=float)
        s = h.sum()
        if s <= 0:
            return 0.0
        h_bar = h / s
        h_bar = np.where(h_bar > 0, h_bar, 1.0)
        return float(-np.sum(h_bar * np.log(h_bar)))

    @staticmethod
    def _legacy_terrain(h):
        h = np.asarray(h, dtype=float)
        pos = h[h > 0]
        if len(pos) == 0:
            return 0.0
        h_bar = pos / pos.sum()
        return float(-np.sum(h_bar * np.log(h_bar)))


# ---------------------------------------------------------------------------
# hs_distance delegation
# ---------------------------------------------------------------------------


class TestBanditsHSDistanceDelegation:
    def _kernels(self):
        from kernelcal.bandits.kernels import AnisotropicSEKernel

        # Kernel hyperparameters are stored in log-space (log_ell_x,
        # log_ell_y, log_sigma_f, log_sigma_n) — see
        # ``AnisotropicSEKernel.__init__`` in ``kernelcal/bandits/kernels.py``.
        k1 = AnisotropicSEKernel(
            log_ell_x=math.log(1.0),
            log_ell_y=math.log(1.0),
            log_sigma_f=math.log(1.0),
            log_sigma_n=math.log(0.1),
        )
        k2 = AnisotropicSEKernel(
            log_ell_x=math.log(2.0),
            log_ell_y=math.log(0.5),
            log_sigma_f=math.log(1.2),
            log_sigma_n=math.log(0.1),
        )
        return k1, k2

    def test_matches_hilbert_schmidt_distance_of_gram_matrices(self):
        from kernelcal.bandits.kernels import hs_distance
        from kernelcal.kernel.space import hilbert_schmidt_distance

        rng = np.random.default_rng(7)
        X_ref = rng.uniform(-2.0, 2.0, size=(24, 2))

        k1, k2 = self._kernels()
        K1 = k1.K(X_ref, add_noise=False)
        K2 = k2.K(X_ref, add_noise=False)

        assert hs_distance(k1, k2, X_ref) == pytest.approx(
            hilbert_schmidt_distance(K1, K2), abs=1e-12
        )

    def test_self_distance_is_zero(self):
        from kernelcal.bandits.kernels import hs_distance

        rng = np.random.default_rng(3)
        X_ref = rng.uniform(-1.0, 1.0, size=(10, 2))
        k1, _ = self._kernels()

        assert hs_distance(k1, k1, X_ref) == pytest.approx(0.0, abs=1e-12)

    def test_symmetry(self):
        from kernelcal.bandits.kernels import hs_distance

        rng = np.random.default_rng(11)
        X_ref = rng.uniform(-1.0, 1.0, size=(15, 2))
        k1, k2 = self._kernels()

        d12 = hs_distance(k1, k2, X_ref)
        d21 = hs_distance(k2, k1, X_ref)
        assert d12 == pytest.approx(d21, abs=1e-12)


# ---------------------------------------------------------------------------
# LocalFrame — package-level canonical + script-level back-compat alias
# ---------------------------------------------------------------------------


class TestLocalFrame:
    def test_script_reexports_canonical_class(self):
        """``bishop_rocks_graph_explorer.LocalFrame`` must be the same
        object as the canonical package class, so existing callers that
        reach into the script continue to work."""
        REPO_ROOT = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(REPO_ROOT))
        try:
            bishop = pytest.importorskip("bishop_rocks_graph_explorer")
        finally:
            sys.path.remove(str(REPO_ROOT))

        from kernelcal.geo3d import LocalFrame as canonical

        assert bishop.LocalFrame is canonical

    def test_origin_maps_to_zero(self):
        from kernelcal.geo3d import LocalFrame

        frame = LocalFrame(lon0=-118.44, lat0=37.45)
        x, y = frame.to_xy(np.array([-118.44]), np.array([37.45]))
        assert abs(float(x[0])) < 1e-6
        assert abs(float(y[0])) < 1e-6

    def test_one_degree_latitude_is_about_111km(self):
        from kernelcal.geo3d import LocalFrame

        frame = LocalFrame(lon0=-118.44, lat0=37.45)
        _, y = frame.to_xy(np.array([-118.44]), np.array([38.45]))
        assert 111_000.0 < float(y[0]) < 111_500.0

    def test_longitude_scales_with_cos_latitude(self):
        from kernelcal.geo3d import LocalFrame

        frame = LocalFrame(lon0=0.0, lat0=60.0)
        x, _ = frame.to_xy(np.array([1.0]), np.array([60.0]))
        # 111,320 * cos(60°) ≈ 55,660 m
        assert 55_000.0 < float(x[0]) < 56_500.0

    def test_scalar_inputs_work(self):
        from kernelcal.geo3d import LocalFrame

        frame = LocalFrame(lon0=10.0, lat0=20.0)
        x, y = frame.to_xy(10.5, 20.5)
        assert float(x) > 0.0
        assert float(y) > 0.0

    def test_round_trip_to_lonlat(self):
        from kernelcal.geo3d import LocalFrame

        frame = LocalFrame(lon0=-118.44, lat0=37.45)
        lon = np.array([-118.5, -118.4, -118.3])
        lat = np.array([37.4, 37.5, 37.6])
        x, y = frame.to_xy(lon, lat)
        lon_back, lat_back = frame.to_lonlat(x, y)
        np.testing.assert_allclose(lon_back, lon, atol=1e-12)
        np.testing.assert_allclose(lat_back, lat, atol=1e-12)

    def test_frozen_dataclass(self):
        from kernelcal.geo3d import LocalFrame

        frame = LocalFrame(lon0=1.0, lat0=2.0)
        with pytest.raises(Exception):
            frame.lon0 = 3.0  # type: ignore[misc]
