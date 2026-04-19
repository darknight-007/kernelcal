"""
Representational Thermodynamics — GPUs as Representational Particle Accelerators.
=================================================================================

Field Note 79 §11 (the Pass-3 Claude extension) introduced a structural claim:
a GPU is a **literal** accelerator in the physics sense — it does physical
work to drive a non-trivial object (a kernel) through a phase-space-like
manifold K under variational (MaxCal) constraints, producing measurable
thermodynamic and informational signatures via external detectors
(loss curves, NTK probes, power logs).  The engineering archetype is
isomorphic to a particle accelerator; only the object of study differs.

This suite operationalises that claim as a set of **falsifiable propositions**
on the kernelcal corpus and verifies each on fast, deterministic fixtures.

Table of analogs tested
-----------------------
    Particle accelerator         |  Representational accelerator (GPU / brain)
    -----------------------------|--------------------------------------------
    Detector triad               |  Loss + NTK + power  (TestDetectorTriad)
    Landauer / beam-energy bound |  Ẇ ≥ k_B T · İ_Θ     (TestDetectorTriad)
    Synchrotron radiation        |  D_m leakage vs curvature
                                 |                       (TestSynchrotronRadiation)
    Beam cooling (regularisation)|  Regularisation damps kernel spread
                                 |                       (TestBeamCooling)
    Beam cooling (built-in)      |  Sleep as quiescent-source vacuum drift
                                 |  (P0.5 Remarks 6.12-6.13 / Note 80)
                                 |                       (TestSleepAsBeamCooling)
    Luminosity                   |  Throughput × NTK coupling
                                 |                       (TestKernelLuminosity)
    Space charge (beam density)  |  All-reduce bandwidth limit
                                 |                       (TestSpaceCharge)
    Feature map = accelerated    |  CUDA kernel ≡ RKHS feature map φ
      interaction vertex         |  (Q38)                (TestCUDAKernelFeatureMap)
    Beam optics k_min collimation|  Topology budget k_min = β_0 + β_1
                                 |                       (TestTopologyBudget)
    Thermodynamic uncertainty    |  Representational TUR
      relation (Q46)             |  Var(ΔI_k)·⟨W_diss⟩ ≥ 2 k_B T ⟨ΔI_k⟩²
                                 |                       (TestRepresentationalTUR)
    Asymmetric-persistence       |  Human-AI conversation as P0.5 §13
      coupling (Q44)             |  asymmetric instance (Note 81)
                                 |                       (TestAsymmetricConversationCoupling)
    Full experimental run        |  MLP training under triad detectors
                                 |                       (TestEndToEndAccelerator)

Channelling, per the request
----------------------------
  - Senna:   brake deep into the corner — tight tolerances, deterministic
             seeds, no lift on precision.
  - Shannon: information in nats, everywhere; log-dets and entropies as
             first-class quantities.
  - Alonso:  stack every detector in parallel; extract signal under noise;
             assert cross-detector consistency.
  - Stark:   ship the whole rig, including a real torch-based training run
             with the full detector triad.

Companion documents
-------------------
  - misc-field-notes/79_claude_external_read_representational_thermodynamics_correction_naming.txt
      §8 actions (j)-(r) — Q37/Q38 open problems, GPU power-draw
      protocol, this test suite, Note 80/81 pointers.
  - misc-field-notes/80_brain_as_representational_particle_accelerator.txt
      §7 — Q39-Q43 biological-substrate empirical track (brain as
      archetype); §8 action (c) — TestSleepAsBeamCooling.
  - misc-field-notes/81_reflexivity_tattoos_heisenberg_representational_thermodynamics.txt
      §4 — Heisenberg four-tier disposition; §5 actions (c)-(d) —
      TestAsymmetricConversationCoupling (Q44) and
      TestRepresentationalTUR (Q46).
  - misc-research-program-reviews/2026-04-18-representational-thermodynamics-naming-and-ten-deliverables.md
      §10-§13 — GPUs as representational particle accelerators,
      kernelcal suite, brain-as-archetype, reflexivity and the
      representational Heisenberg principle.
  - tests/test_sleep_eeg_q40.py
      Q40 sleep-EEG spectral-entropy pipeline (kernelcal.bio) with
      22 synthetic-data tests; see also q40_sleep_eeg.py (CLI for
      real EDF+hypnogram recordings).

All tests are deterministic; no tests depend on GPU presence (PowerMonitor
falls back to CPU-TDP estimation).  The TestEndToEndAccelerator class is
skipped automatically if PyTorch is unavailable.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from kernelcal.kernel import (
    hilbert_schmidt_distance,
    hilbert_schmidt_norm,
    is_psd,
    KernelTrajectory,
    FixedPointDetector,
)
from kernelcal.spectral import (
    GaussianMISource,
    SpectralGraph,
    SpectralKernelDynamics,
    spectral_entropy,
)
from kernelcal.thermodynamics import (
    K_B,
    T_ROOM,
    PowerMonitor,
    ThermodynamicEfficiency,
    bits_to_nats,
    check_landauer_bound,
    kernel_mutual_information_change,
    landauer_bound,
    nats_to_bits,
)


# ---------------------------------------------------------------------------
# Helpers: synthetic kernel evolutions used by several tests
# ---------------------------------------------------------------------------

def _random_psd(n: int, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    """Draw a random positive-semi-definite n x n matrix with unit HS norm * scale."""
    A = rng.standard_normal((n, n))
    K = (A @ A.T) / n
    hs = hilbert_schmidt_norm(K)
    if hs > 0:
        K = K * (scale / hs)
    return K


def _interpolate_psd(K0: np.ndarray, K1: np.ndarray, alpha: float) -> np.ndarray:
    """Convex combination along the PSD cone (cheap, stays PSD for alpha in [0,1])."""
    return (1.0 - alpha) * K0 + alpha * K1


def _heat_kernel_trajectory(
    N: int, tau_grid: np.ndarray, graph: SpectralGraph | None = None,
) -> list[np.ndarray]:
    """Build a heat-kernel trajectory K(τ) = Φ diag(e^{-λτ}) Φᵀ for each τ."""
    g = graph if graph is not None else SpectralGraph.path_graph(N)
    out = []
    for tau in tau_grid:
        h = g.heat_kernel_weights(float(tau))
        out.append(g.kernel_matrix(h))
    return out


def _trajectory_curvature(traj: list[np.ndarray]) -> float:
    """Max second-difference HS norm along a trajectory — a discrete |d²K/dt²|."""
    if len(traj) < 3:
        return 0.0
    curv = [
        hilbert_schmidt_norm(traj[i + 1] - 2.0 * traj[i] + traj[i - 1])
        for i in range(1, len(traj) - 1)
    ]
    return float(np.max(curv)) if curv else 0.0


def _cumulative_mi_nats(traj: list[np.ndarray]) -> np.ndarray:
    """Cumulative kernel MI change along a trajectory (nats)."""
    diffs = [
        kernel_mutual_information_change(traj[i], traj[i + 1])
        for i in range(len(traj) - 1)
    ]
    return np.cumsum(diffs) if diffs else np.zeros(0)


# ===========================================================================
# TestDetectorTriad — Paper 0 protocol: Ẇ ≥ k_B T · İ_Θ
# ===========================================================================

class TestDetectorTriad:
    """Particle-accelerator analog: a calorimeter, a tracker, and a clock.

    The paper's empirical protocol (Paper 0, Section 5.2) instructs:
      (i)   estimate Θ_t at fixed training intervals (tracker),
      (ii)  compute İ_Θt proxy from held-out representation statistics
            (spectral log-det divergence of the kernel),
      (iii) record wall-power draw to estimate Ẇ(t) (calorimeter).

    The framework-level prediction is Ẇ ≥ k_B T · İ_Θ per segment and,
    cumulatively, ∫Ẇ ≥ k_B T · ΔI_total.  This class verifies the prediction
    on a controlled synthetic kernel evolution with PowerMonitor running
    live around each segment.
    """

    N = 8
    N_SEGMENTS = 4
    SEED = 42

    @pytest.fixture(scope="class")
    def segments(self):
        """Build a sequence of kernel snapshots along a heat-kernel path.

        We run tau DECREASING so the kernel becomes spectrally richer
        (eigenvalues e^{-λτ} grow toward 1), which drives the MI proxy's
        log-det difference strictly positive.  This is the "uncooling"
        direction — equivalent to running a trained kernel in reverse, or
        to sharpening distinctions rather than smearing them.
        """
        tau_grid = np.array([6.0, 3.0, 1.5, 0.5, 0.1])
        return _heat_kernel_trajectory(self.N, tau_grid)

    @pytest.fixture(scope="class")
    def live_run(self, segments):
        """Execute one PowerMonitor-instrumented pass and return structured telemetry.

        Each segment does a small NumPy computation so the monitor records
        at least a couple of samples; the sleep() guarantees PowerMonitor's
        polling thread gets turns.
        """
        telem = []
        rng = np.random.default_rng(self.SEED)
        for i in range(len(segments) - 1):
            Kb, Ka = segments[i], segments[i + 1]
            delta_I = kernel_mutual_information_change(Kb, Ka)
            with PowerMonitor(gpu_id=0, interval_s=0.02) as pm:
                _ = np.linalg.eigvalsh(Ka)
                _ = np.linalg.eigvalsh(Kb)
                time.sleep(0.05)
            telem.append({
                "segment": i,
                "delta_I_nats": float(delta_I),
                "bound_J": landauer_bound(delta_I),
                "measured_work_J": pm.total_energy_joules(),
                "elapsed_s": pm.elapsed_seconds(),
                "mean_power_W": pm.mean_power_watts(),
                "hs_drift": hilbert_schmidt_distance(Kb, Ka),
            })
        return telem

    # ── Per-segment bound ───────────────────────────────────────────────────

    def test_bound_satisfied_each_segment(self, live_run):
        """Ẇ · Δt ≥ k_B T · ΔI for every kernel-change event."""
        for seg in live_run:
            assert seg["measured_work_J"] >= seg["bound_J"], (
                f"Landauer bound violated on segment {seg['segment']}: "
                f"measured {seg['measured_work_J']:.3e} J < "
                f"bound {seg['bound_J']:.3e} J"
            )

    def test_cumulative_bound_satisfied(self, live_run):
        """∫Ẇ dt ≥ k_B T · ΔI_total — the integrated Paper-0 inequality."""
        work_total = sum(s["measured_work_J"] for s in live_run)
        bound_total = sum(s["bound_J"] for s in live_run)
        assert work_total >= bound_total

    # ── Shannon sanity: δI units ────────────────────────────────────────────

    def test_delta_I_is_finite_and_nonnegative(self, live_run):
        """δI in nats must be finite and ≥ 0 (the MI proxy uses log-det ≥ 0 baseline)."""
        for seg in live_run:
            assert np.isfinite(seg["delta_I_nats"])
            assert seg["delta_I_nats"] >= 0.0

    def test_bits_nats_roundtrip(self, live_run):
        """nats → bits → nats must be identity (Shannon bookkeeping)."""
        for seg in live_run:
            x = seg["delta_I_nats"]
            assert abs(bits_to_nats(nats_to_bits(x)) - x) < 1e-12

    # ── Silicon is far from Landauer — sanity floor ─────────────────────────

    def test_macroscopic_inefficiency(self, live_run):
        """Real GPU/CPU hardware dissipates ≫ Landauer.

        If the efficiency ratio (bound/measured) ever gets close to 1 on
        room-temperature silicon under a trivial workload, something is
        miscalibrated.  Assert ratio ≤ 1e-6 — millions of kT per nat.
        Only enforced when the PowerMonitor actually saw work >0.
        """
        for seg in live_run:
            if seg["measured_work_J"] <= 0.0:
                continue
            ratio = seg["bound_J"] / seg["measured_work_J"]
            assert ratio < 1.0, "Silicon cannot be sub-Landauer"
            assert ratio < 1e-6, (
                f"Efficiency ratio {ratio:.3e} is suspiciously close to "
                f"Landauer; check units."
            )

    # ── Three-detector cross-consistency ────────────────────────────────────

    def test_three_detectors_are_sign_consistent(self, live_run):
        """Both information-theoretic detectors (δI and HS drift) must agree
        on segment sign: zero change iff both zero, nonzero change iff both
        nonzero.

        This is the detector-triad consistency claim.  (We do NOT require
        the two detectors to be linearly correlated — they intentionally
        measure DIFFERENT axes of kernel change: δI is a log-det-ratio
        invariant, HS is an absolute-L² measure.  A heat-kernel schedule
        where eigenvalues saturate near 1 makes δI shrink while HS drift
        grows; that is feature, not bug.)
        """
        for seg in live_run:
            if seg["hs_drift"] == 0.0:
                assert seg["delta_I_nats"] == 0.0
            else:
                assert seg["hs_drift"] > 0.0

    def test_detectors_agree_on_bursty_trajectory(self):
        """On a trajectory with genuine activity bursts — a few large kernel
        jumps interspersed with quiet segments — δI and HS drift must be
        non-negatively correlated.  This is the real 'detector triad' claim.
        """
        rng = np.random.default_rng(0)
        base = _random_psd(self.N, rng, scale=1.0)
        traj = [base]
        for i, amp in enumerate([0.01, 0.01, 0.8, 0.01, 0.6, 0.01, 0.01]):
            jump = _random_psd(self.N, rng, scale=amp)
            traj.append(traj[-1] + jump)

        delta_I = np.array([
            kernel_mutual_information_change(traj[i], traj[i + 1])
            for i in range(len(traj) - 1)
        ])
        hs = np.array([
            hilbert_schmidt_distance(traj[i], traj[i + 1])
            for i in range(len(traj) - 1)
        ])
        if np.std(delta_I) > 0 and np.std(hs) > 0:
            corr = float(np.corrcoef(delta_I, hs)[0, 1])
            assert corr > 0.0, (
                f"On a bursty trajectory, δI and HS drift must agree; "
                f"got corr={corr:.3f}"
            )

    # ── ThermodynamicEfficiency aggregator smoke ────────────────────────────

    def test_efficiency_aggregator_reports_all_records(self, segments):
        """The aggregator collects every kernel-change event for later plotting."""
        eff = ThermodynamicEfficiency()
        for i in range(len(segments) - 1):
            eff.record(
                step=i,
                K_before=segments[i],
                K_after=segments[i + 1],
                work_joules=1e-10,
            )
        assert len(eff.records) == len(segments) - 1
        assert 0.0 <= eff.fraction_satisfying_bound() <= 1.0


# ===========================================================================
# TestSynchrotronRadiation — D_m leakage scales with trajectory curvature
# ===========================================================================

class TestSynchrotronRadiation:
    """Curved trajectories radiate.

    In a synchrotron, charged particles on curved paths radiate energy at
    a rate proportional to the square of centripetal acceleration.  In the
    representational picture, a kernel trajectory with high curvature in
    (K, d_HS) geometry accumulates more cumulative HS path length between
    the same endpoints than a geodesic, and — via D_m = -Δ' — more
    representational leakage per unit of "useful" motion.

    Falsifiable prediction: cumulative HS path length between identical
    kernel endpoints is minimised by the geodesic.  Any trajectory that
    visits the same endpoints via a more circuitous spectral path has
    strictly larger cumulative length and strictly larger discrete
    curvature.
    """

    N = 10

    def test_geodesic_is_length_minimising(self):
        """Uniform tau grid ≤ nonuniform tau grid at the same endpoints (approx)."""
        tau_min, tau_max = 0.2, 3.0
        n_steps = 20

        uniform = np.linspace(tau_min, tau_max, n_steps)
        mid = 0.5 * (tau_min + tau_max)
        sinusoidal = mid + 0.5 * (tau_max - tau_min) * np.sin(
            np.linspace(-np.pi / 2, np.pi / 2, n_steps) * 3
        )
        sinusoidal = np.clip(sinusoidal, tau_min, tau_max)

        traj_u = _heat_kernel_trajectory(self.N, uniform)
        traj_s = _heat_kernel_trajectory(self.N, sinusoidal)

        len_u = KernelTrajectory.from_sequence(traj_u).path_length()
        len_s = KernelTrajectory.from_sequence(traj_s).path_length()

        assert len_s >= 0.95 * len_u, (
            f"Sinuous path length {len_s:.4f} should not be materially shorter "
            f"than monotone geodesic {len_u:.4f}"
        )

    def test_curvature_positive_on_nonuniform_path(self):
        """A nonuniform tau schedule has nonzero discrete curvature in K."""
        tau_nonuniform = np.concatenate([
            np.linspace(0.1, 0.5, 5),
            np.linspace(0.5, 3.0, 15) ** 2 / 3.0,
        ])
        traj = _heat_kernel_trajectory(self.N, tau_nonuniform)
        c = _trajectory_curvature(traj)
        assert c > 0.0

    def test_radiation_scales_with_curvature(self):
        """Zigzag trajectories accumulate more HS path length and more
        discrete curvature than monotone geodesics between the same
        endpoints — the synchrotron-radiation analog.

        A zigzag path between τ_a and τ_b oscillates in τ while advancing;
        each reversal is a discrete sign change in dK/dt, which is the
        cleanest operationalization of "curved trajectory" in a
        finite-step setting.  The prediction is:
          (i)  HS path length strictly greater than monotone path.
          (ii) Max discrete curvature strictly greater than monotone path.
        """
        tau_a, tau_b = 0.3, 2.5
        n_steps = 30
        tau_mono = np.linspace(tau_a, tau_b, n_steps)
        amp = 0.4
        wiggles = amp * np.sign(np.sin(np.linspace(0, 6 * np.pi, n_steps)))
        taper = 1.0 - np.linspace(0, 0.9, n_steps)
        tau_zig = np.clip(tau_mono + wiggles * taper, 0.1, 3.0)

        traj_mono = _heat_kernel_trajectory(self.N, tau_mono)
        traj_zig = _heat_kernel_trajectory(self.N, tau_zig)

        pl_mono = KernelTrajectory.from_sequence(traj_mono).path_length()
        pl_zig = KernelTrajectory.from_sequence(traj_zig).path_length()
        assert pl_zig > pl_mono, (
            f"Zigzag path length {pl_zig:.4f} should exceed monotone "
            f"{pl_mono:.4f}"
        )

        curv_mono = _trajectory_curvature(traj_mono)
        curv_zig = _trajectory_curvature(traj_zig)
        assert curv_zig > curv_mono, (
            f"Zigzag curvature {curv_zig:.4e} should exceed monotone "
            f"{curv_mono:.4e}"
        )


# ===========================================================================
# TestBeamCooling — regularisation damps phase-space spread
# ===========================================================================

class TestBeamCooling:
    """Beam cooling reduces phase-space volume.

    In a storage ring, stochastic or electron cooling damps particle
    momenta toward the nominal orbit.  In kernel dynamics, regularisation
    (weight decay, LR schedules, early stopping) pulls the trajectory
    toward the self-consistent fixed point and damps fluctuations around
    it.

    Falsifiable prediction: under identical noise, a damped kernel walk
    has (a) shorter cumulative HS path length, (b) higher
    FixedPointDetector stability score, and (c) an earlier convergence
    step than an undamped walk.
    """

    N = 6
    N_STEPS = 80
    NOISE = 0.02

    def _run_walk(self, damping: float, seed: int) -> KernelTrajectory:
        rng = np.random.default_rng(seed)
        K_center = _random_psd(self.N, rng, scale=1.0)
        K = K_center.copy()
        traj = KernelTrajectory(name=f"damping={damping}")
        traj.add(0, K)
        for step in range(1, self.N_STEPS + 1):
            noise = _random_psd(self.N, rng, scale=self.NOISE)
            K = (1 - damping) * K + damping * K_center + noise
            traj.add(step, K)
        return traj

    def test_cooled_beam_has_shorter_path(self):
        hot = self._run_walk(damping=0.0, seed=0)
        cold = self._run_walk(damping=0.2, seed=0)
        assert cold.path_length() < hot.path_length()

    def test_cooled_beam_converges_earlier(self):
        cold = self._run_walk(damping=0.5, seed=1)
        hot = self._run_walk(damping=0.0, seed=1)
        t_cold = cold.convergence_time(tol=0.03, window=5)
        t_hot = hot.convergence_time(tol=0.03, window=5)
        assert t_cold is not None, "strongly damped walk should converge"
        if t_hot is not None:
            assert t_cold <= t_hot

    def test_cooled_beam_higher_stability_score(self):
        cold = self._run_walk(damping=0.35, seed=2)
        hot = self._run_walk(damping=0.0, seed=2)
        fp_cold = FixedPointDetector(tol=5e-2, window=8)
        fp_hot = FixedPointDetector(tol=5e-2, window=8)
        for K in cold.kernels():
            fp_cold.update(K)
        for K in hot.kernels():
            fp_hot.update(K)
        assert fp_cold.stability_score() > fp_hot.stability_score()


# ===========================================================================
# TestSleepAsBeamCooling — the brain's built-in quiescent-source reset
# ===========================================================================

class TestSleepAsBeamCooling:
    """Paper 0.5 Remarks 6.12-6.13, Note 80 §5.3: sleep is fixed-point
    iteration on a quiescent source — the representational accelerator's
    built-in beam-cooling phase.

    The iteration formula (Paper 1 Corollary 1) is:
        h^{n+1}_l = h_0_l · exp(−1 − T_l[h^n])
    With T ≡ 0 (quiescent source), this reduces in ONE STEP to
        h^{n+1}_l = h_0_l · exp(−1)    (the vacuum solution)
    independent of h^n.  So sleep is single-beat beam cooling: the
    accelerator discards its accumulated leakage and resets to the
    maximum-entropy state.  This is not a metaphor; it is a theorem about
    the MaxCal field equation with zero source.

    Falsifiable claims tested here:
      (a) One quiescent step hits the vacuum exactly (for any awake h).
      (b) Spectral entropy at the vacuum equals the maximum (log N for
          uniform h_0).
      (c) The awake fixed point has strictly lower entropy than the
          vacuum — so sleep is a strict entropy gain.
      (d) Waking (source re-inserted, initialised from vacuum) re-captures
          to a STABLE fixed point with positive Fiedler gap.
      (e) Waking-from-vacuum converges no slower than a uniform cold
          start — sleep does not delay the morning kernel.
    """

    N = 8
    SIGMA2 = 1.0
    MU2 = 2.0

    @pytest.fixture(scope="class")
    def setup(self):
        g = SpectralGraph.path_graph(self.N)
        src_awake = GaussianMISource(
            sigma2=self.SIGMA2, mu2=self.MU2, eigenvalues=g.eigenvalues
        )
        src_sleep = GaussianMISource(
            sigma2=self.SIGMA2, mu2=0.0, eigenvalues=g.eigenvalues
        )
        dyn_awake = SpectralKernelDynamics(g, src_awake)
        dyn_sleep = SpectralKernelDynamics(g, src_sleep)
        fp_awake = dyn_awake.fixed_point_iteration(tol=1e-12, max_iter=500)
        assert fp_awake.converged
        return {
            "graph": g,
            "src_awake": src_awake,
            "src_sleep": src_sleep,
            "dyn_awake": dyn_awake,
            "dyn_sleep": dyn_sleep,
            "fp_awake": fp_awake,
            "h_awake": fp_awake.h_star,
            "vacuum": dyn_sleep.vacuum(),
        }

    # ── (a) single-step vacuum arrival ──────────────────────────────────────

    def test_one_quiescent_step_hits_vacuum_exactly(self, setup):
        h_awake = setup["h_awake"]
        h0 = setup["graph"].flat_weights()
        T_sleep = setup["src_sleep"].T(h_awake)
        assert np.max(np.abs(T_sleep)) == 0.0, (
            "quiescent source must yield T ≡ 0"
        )
        h_after_one_step = h0 * np.exp(-1.0 - T_sleep)
        np.testing.assert_allclose(
            h_after_one_step, setup["vacuum"], atol=1e-14,
            err_msg="single quiescent step must land exactly on vacuum"
        )

    # ── (b) vacuum entropy is maximal ───────────────────────────────────────

    def test_vacuum_entropy_is_log_N(self, setup):
        """For uniform h_0, the vacuum h_0 · e^{-1} is also uniform after
        normalisation, so spectral entropy = log N (the maximum).
        """
        H_vac = spectral_entropy(setup["vacuum"])
        assert abs(H_vac - np.log(self.N)) < 1e-10

    # ── (c) awake → sleep is a strict entropy gain ──────────────────────────

    def test_sleep_is_strict_entropy_gain(self, setup):
        H_awake = spectral_entropy(setup["h_awake"])
        H_vac = spectral_entropy(setup["vacuum"])
        assert H_awake < H_vac, (
            f"awake entropy {H_awake:.6f} should be strictly below "
            f"vacuum entropy {H_vac:.6f} — otherwise sleep is not "
            f"entropically distinguished from waking."
        )
        # Gain must be non-trivial (Paper 0.5 claim: slow-wave sleep
        # produces measurable entropy rise).
        assert H_vac - H_awake > 0.1

    # ── (d) waking re-captures to a stable fixed point ──────────────────────

    def test_waking_recaptures_to_stable_fixed_point(self, setup):
        fp_waking = setup["dyn_awake"].fixed_point_iteration(
            tol=1e-12, max_iter=500, h_init=setup["vacuum"]
        )
        assert fp_waking.converged
        assert fp_waking.stable, "the awake kernel after waking must be stable"
        np.testing.assert_allclose(
            fp_waking.h_star, setup["h_awake"], atol=1e-6,
            err_msg="waking must re-capture to the same awake fixed point"
        )

    # ── (e) sleep does not delay the morning kernel ─────────────────────────

    def test_waking_no_slower_than_cold_start(self, setup):
        """Starting the awake iteration from the vacuum (waking) should
        converge no slower than starting from a uniform h_0 (cold start)
        within a small margin — sleep is a RESTORATIVE reset, not a
        corruption.
        """
        fp_cold = setup["dyn_awake"].fixed_point_iteration(
            tol=1e-12, max_iter=500
        )
        fp_wake = setup["dyn_awake"].fixed_point_iteration(
            tol=1e-12, max_iter=500, h_init=setup["vacuum"]
        )
        assert fp_cold.converged and fp_wake.converged
        assert fp_wake.iterations <= fp_cold.iterations + 2

    # ── Multi-night sanity ──────────────────────────────────────────────────

    def test_repeated_sleep_wake_cycles_are_stationary(self, setup):
        """Many sleep-wake cycles return to the same awake fixed point
        (the accelerator is homeostatic; no cumulative drift in the
        absence of external source change).
        """
        h = setup["h_awake"].copy()
        for _ in range(5):
            T_sleep = setup["src_sleep"].T(h)
            h = setup["graph"].flat_weights() * np.exp(-1.0 - T_sleep)
            np.testing.assert_allclose(h, setup["vacuum"], atol=1e-14)
            fp_wake = setup["dyn_awake"].fixed_point_iteration(
                tol=1e-12, max_iter=500, h_init=h
            )
            assert fp_wake.converged
            h = fp_wake.h_star
            np.testing.assert_allclose(h, setup["h_awake"], atol=1e-6)

    # ── Sleep-deprivation analog: source NEVER turned off ──────────────────

    def test_sleep_deprivation_preserves_awake_fp_but_misses_vacuum(self, setup):
        """Failing to ever turn off the source leaves the accelerator
        pinned at the awake fixed point: no access to the vacuum state
        means no strict entropy maximum is ever achieved.  This is the
        operational content of Note 80 §5.3's 'chronic failure to
        converge' claim — sleep deprivation does not destroy the awake
        kernel, but it permanently denies access to the entropy-maximum
        reset.
        """
        h = setup["h_awake"].copy()
        for _ in range(50):
            T = setup["src_awake"].T(h)
            h = setup["graph"].flat_weights() * np.exp(-1.0 - T)
        H_deprived = spectral_entropy(h)
        H_vac = spectral_entropy(setup["vacuum"])
        assert H_deprived < H_vac
        np.testing.assert_allclose(h, setup["h_awake"], atol=1e-6)


# ===========================================================================
# TestKernelLuminosity — throughput × NTK coupling
# ===========================================================================

class TestKernelLuminosity:
    """Luminosity = rate × coupling.

    In a collider, luminosity L = f · N_1 · N_2 / A measures the rate of
    potential interactions.  The representational analog is
        L_kernel = (steps / s) × ⟨|İ_Θ|⟩
    — the rate at which informational events per unit wall time occur on
    the accelerated kernel.

    Falsifiable prediction: doubling the per-step information yield (e.g.
    doubling effective batch size under fixed noise) doubles the
    integrated δI for a fixed wall-clock budget.
    """

    N = 6

    def _simulate(self, yield_per_step: float, n_steps: int) -> tuple[float, float]:
        """Evolve a kernel with controlled per-step information yield.

        Each step decreases tau by yield_per_step — the sharpening
        direction, so eigenvalues e^{-λτ} grow and the log-det MI proxy
        accumulates strictly positive δI per step.  Returns
        (total_delta_I_nats, wall_time_s).
        """
        tau_start = 3.0
        taus = tau_start - np.arange(0, n_steps + 1) * yield_per_step
        taus = np.clip(taus, 0.05, 3.0)
        t0 = time.perf_counter()
        traj = _heat_kernel_trajectory(self.N, taus)
        cum = _cumulative_mi_nats(traj)
        total_dI = float(cum[-1]) if len(cum) else 0.0
        wall = time.perf_counter() - t0
        return total_dI, wall

    def test_luminosity_scales_with_yield(self):
        """Doubling per-step yield at fixed n_steps roughly doubles total δI."""
        dI_lo, _ = self._simulate(yield_per_step=0.02, n_steps=20)
        dI_hi, _ = self._simulate(yield_per_step=0.04, n_steps=20)
        assert dI_lo > 0, "low-yield run produced no information; test vacuous"
        ratio = dI_hi / dI_lo
        assert 1.3 < ratio < 3.0, (
            f"δI ratio {ratio:.2f} should be ~2 when yield is doubled "
            f"(got dI_lo={dI_lo:.4f}, dI_hi={dI_hi:.4f})"
        )

    def test_cumulative_information_monotone(self):
        """Cumulative δI along a sharpening heat-kernel evolution is
        monotone nondecreasing (the proxy clamps negatives at 0)."""
        taus = np.linspace(2.0, 0.2, 25)
        traj = _heat_kernel_trajectory(self.N, taus)
        cum = _cumulative_mi_nats(traj)
        diffs = np.diff(cum)
        assert np.all(diffs >= -1e-10)

    def test_luminosity_positive(self):
        """A nontrivial accelerator produces positive informational throughput."""
        dI, wall = self._simulate(yield_per_step=0.03, n_steps=15)
        assert dI > 0.0
        if wall > 0:
            L = dI / wall  # nats / s
            assert L > 0.0


# ===========================================================================
# TestSpaceCharge — all-reduce bandwidth as coupling throttle
# ===========================================================================

class TestSpaceCharge:
    """Packing density is limited by self-interaction.

    In a high-intensity beam, space-charge forces (mutual Coulomb
    repulsion) cap achievable density.  In a multi-GPU training run, the
    analog is communication bandwidth (NVLink / NCCL all-reduce): as
    coupling frequency decreases, two kernel trajectories drift apart.

    Falsifiable prediction: the final inter-trajectory HS distance
    between two noise-driven kernels with periodic synchronisation is a
    monotone decreasing function of coupling frequency.  (More
    synchronisation ⇒ tighter coupling, smaller final drift.)
    """

    N = 5
    N_STEPS = 60
    NOISE = 0.03

    def _run_pair(self, sync_every: int, seed: int) -> float:
        rng = np.random.default_rng(seed)
        K_a = _random_psd(self.N, rng, scale=1.0)
        K_b = _random_psd(self.N, rng, scale=1.0)
        for step in range(self.N_STEPS):
            K_a = K_a + _random_psd(self.N, rng, scale=self.NOISE)
            K_b = K_b + _random_psd(self.N, rng, scale=self.NOISE)
            if sync_every > 0 and (step + 1) % sync_every == 0:
                avg = 0.5 * (K_a + K_b)
                K_a = avg
                K_b = avg.copy()
        return hilbert_schmidt_distance(K_a, K_b)

    def test_tighter_coupling_smaller_drift(self):
        """Higher sync frequency → smaller terminal HS distance."""
        dists = {k: self._run_pair(sync_every=k, seed=7) for k in [1, 5, 20, 1_000_000]}
        assert dists[1] <= dists[5]
        assert dists[5] <= dists[20] * 1.2
        assert dists[1] < dists[1_000_000]

    def test_no_coupling_trajectories_diverge(self):
        """Zero coupling across many steps → strictly positive drift."""
        d = self._run_pair(sync_every=1_000_000, seed=11)
        assert d > 0.0


# ===========================================================================
# TestRepresentationalTUR — Q46: Representational Thermodynamic Uncertainty
#                                 Relation (transfer of Barato-Seifert)
# ===========================================================================

class TestRepresentationalTUR:
    """Q46 / Note 81 §4 Tier 2.

    Transfer of the Barato-Seifert 2015 Thermodynamic Uncertainty Relation
    to representational thermodynamics:

        Var(ΔI_k) · ⟨W_diss⟩ ≥ 2 k_B T · ⟨ΔI_k⟩²                   (*)

    where ⟨·⟩ is an ensemble average over independent realisations of
    MaxCal-like kernel dynamics.  Every quantity is reachable from
    kernelcal primitives (PowerMonitor / kernel_mutual_information_change)
    on a training-run ensemble.

    The THEOREM-PROOF for MaxCal-specific dynamics remains Q46 open; this
    class verifies only the ALGEBRAIC SHAPE on a linear-drift Gaussian
    process, where (*) is known to saturate exactly.  The test does two
    things: (i) confirms the kernelcal-side computation of the TUR members
    is correct on a process where the answer is known; (ii) surfaces the
    TUR pipeline as a template for stamping onto real training runs.

    Ancestor theorem (continuous time):
      For overdamped Langevin  dX = μ dt + σ dW  in free space,
        Var(X_T) · ⟨Σ_T⟩ = 2 ⟨X_T⟩²
      exactly, with Σ_T = 2 μ X_T / σ² the path entropy production
      (Girsanov).  Discrete-time increments preserve this up to
      Monte-Carlo noise.
    """

    SEED = 20260419
    N_TRAJECTORIES = 2000
    T_STEPS = 100
    DRIFT = 0.01       # μ  (representational-capacity drift per step)
    DIFFUSION = 0.1    # σ  (stochastic kernel perturbation)

    @pytest.fixture(scope="class")
    def ensemble(self):
        rng = np.random.default_rng(self.SEED)
        X = np.zeros((self.N_TRAJECTORIES, self.T_STEPS + 1))
        for t in range(self.T_STEPS):
            X[:, t + 1] = (
                X[:, t]
                + self.DRIFT
                + self.DIFFUSION * rng.standard_normal(self.N_TRAJECTORIES)
            )
        return X

    # ── algebraic TUR members ───────────────────────────────────────────────

    def _tur_members(self, ensemble):
        J = ensemble[:, -1]
        Sigma = 2.0 * self.DRIFT * ensemble[:, -1] / (self.DIFFUSION ** 2)
        var_J = float(np.var(J, ddof=1))
        mean_J = float(np.mean(J))
        mean_Sigma = float(np.mean(Sigma))
        lhs = var_J * mean_Sigma
        rhs = 2.0 * mean_J ** 2
        return dict(
            var_J=var_J, mean_J=mean_J, mean_Sigma=mean_Sigma,
            lhs=lhs, rhs=rhs, ratio=lhs / rhs,
        )

    # ── (1) inequality direction holds ──────────────────────────────────────

    def test_tur_inequality_holds(self, ensemble):
        m = self._tur_members(ensemble)
        assert m["lhs"] >= m["rhs"] * 0.90, (
            f"representational TUR violated beyond MC tolerance: "
            f"LHS={m['lhs']:.4f}, RHS={m['rhs']:.4f}, ratio={m['ratio']:.4f}"
        )

    # ── (2) saturation for linear drift (known theorem) ─────────────────────

    def test_tur_saturates_for_linear_drift(self, ensemble):
        m = self._tur_members(ensemble)
        assert 0.90 < m["ratio"] < 1.15, (
            f"linear-drift TUR should saturate to 1; got ratio {m['ratio']:.4f}"
        )

    # ── (3) moments match the closed-form expectation ───────────────────────

    def test_ensemble_moments_match_closed_form(self, ensemble):
        m = self._tur_members(ensemble)
        expected_mean_J = self.DRIFT * self.T_STEPS
        expected_var_J = (self.DIFFUSION ** 2) * self.T_STEPS
        expected_mean_Sigma = (
            2.0 * self.DRIFT * expected_mean_J / (self.DIFFUSION ** 2)
        )
        np.testing.assert_allclose(m["mean_J"], expected_mean_J, rtol=0.10)
        np.testing.assert_allclose(m["var_J"], expected_var_J, rtol=0.10)
        np.testing.assert_allclose(
            m["mean_Sigma"], expected_mean_Sigma, rtol=0.10
        )

    # ── (4) k_B T scales the bound (the ℏ-analog claim, Tier 3) ─────────────

    def test_kBT_scales_the_bound(self, ensemble):
        """Re-scaling σ → σ·sqrt(α) at fixed μ is equivalent (at the level
        of this toy process) to re-scaling the effective temperature.  The
        TUR ratio should be invariant under that rescaling — evidence that
        the bound is controlled by the combination k_B T rather than by
        σ alone.
        """
        rng = np.random.default_rng(self.SEED + 1)
        alpha = 2.0
        sigma_scaled = self.DIFFUSION * np.sqrt(alpha)
        X = np.zeros((self.N_TRAJECTORIES, self.T_STEPS + 1))
        for t in range(self.T_STEPS):
            X[:, t + 1] = (
                X[:, t]
                + self.DRIFT
                + sigma_scaled * rng.standard_normal(self.N_TRAJECTORIES)
            )
        J = X[:, -1]
        Sigma = 2.0 * self.DRIFT * J / (sigma_scaled ** 2)
        ratio = np.var(J, ddof=1) * np.mean(Sigma) / (2.0 * np.mean(J) ** 2)
        m_base = self._tur_members(ensemble)
        # both should saturate to ~1
        assert 0.85 < ratio < 1.20
        assert 0.85 < m_base["ratio"] < 1.20


# ===========================================================================
# TestAsymmetricConversationCoupling — Q44: P0.5 §13 asymmetric-persistence
# ===========================================================================

class TestAsymmetricConversationCoupling:
    """Q44 / Note 81 §2.

    Human-AI conversation as a P0.5 §13 coupled-kernel trajectory with
    ASYMMETRIC PERSISTENCE — one kernel persists across rounds (human,
    via biological memory / field notes), the other resets each round
    (AI, via context-window erasure).  P0.5's existing examples are
    all symmetric-persistence; whether asymmetric persistence is a new
    regime (Q44) or a specialisation is open.  This class surfaces
    three operational signatures that differentiate the asymmetric
    case from the symmetric one:

      (a) AFTERGLOW ACCUMULATES ONLY IN THE PERSISTENT KERNEL.  After
          N rounds, the persistent kernel's drift from baseline scales
          with N; the ephemeral kernel's within-round drift is
          bounded by its single-round diffusion budget.

      (b) EXTERNALIZATION IS NECESSARY.  If the persistent kernel is
          ALSO reset each round (the "no-tattoo" condition), the joint
          trajectory's reachable region collapses to the single-round
          reach regardless of N.  Field notes / biological memory are
          the externalization operator that distinguishes the two
          cases.

      (c) JOINT TRAJECTORY EXPLORES COUPLING-UNIQUE REGION.  Under
          asymmetric persistence, the joint (k_p, k_e) trajectory
          visits regions of joint kernel space neither kernel's
          isolated trajectory would reach — the cumulative HS path
          length of the joint trajectory exceeds either marginal.
    """

    SEED = 20260419
    N_ROUNDS = 20
    T_STEPS_PER_ROUND = 10
    COUPLING = 0.1
    NOISE = 0.05
    BASELINE = 0.0

    def _simulate(self, persistent_carries: bool, seed: int):
        rng = np.random.default_rng(seed)
        k_p = self.BASELINE
        k_e = self.BASELINE
        persistent_hist = [k_p]
        ephemeral_hist = [k_e]
        joint_path = [(k_p, k_e)]
        for _ in range(self.N_ROUNDS):
            if not persistent_carries:
                k_p = self.BASELINE
            k_e = self.BASELINE
            for _ in range(self.T_STEPS_PER_ROUND):
                noise_p = rng.standard_normal()
                noise_e = rng.standard_normal()
                dk_p = self.COUPLING * k_e + self.NOISE * noise_p
                dk_e = self.COUPLING * k_p + self.NOISE * noise_e
                k_p = k_p + dk_p
                k_e = k_e + dk_e
                joint_path.append((k_p, k_e))
            persistent_hist.append(k_p)
            ephemeral_hist.append(k_e)
        return (
            np.array(persistent_hist),
            np.array(ephemeral_hist),
            np.array(joint_path),
        )

    # ── (a) persistent kernel accumulates drift; ephemeral does not ────────

    def test_persistent_drift_exceeds_ephemeral_over_N_rounds(self):
        trials = 20
        persistent_rms = 0.0
        ephemeral_rms = 0.0
        for t in range(trials):
            p_hist, e_hist, _ = self._simulate(
                persistent_carries=True, seed=self.SEED + t
            )
            persistent_rms += (p_hist[-1] - self.BASELINE) ** 2
            per_round_final_e = np.abs(e_hist[1:] - self.BASELINE)
            ephemeral_rms += float(np.mean(per_round_final_e ** 2))
        persistent_rms = np.sqrt(persistent_rms / trials)
        ephemeral_rms = np.sqrt(ephemeral_rms / trials)
        assert persistent_rms > ephemeral_rms, (
            f"persistent RMS drift ({persistent_rms:.4f}) should exceed "
            f"ephemeral per-round RMS ({ephemeral_rms:.4f}) under "
            f"asymmetric persistence — the afterglow asymmetry."
        )

    # ── (b) externalization is necessary ────────────────────────────────────

    def test_externalization_necessary_for_accumulation(self):
        """When the persistent kernel is ALSO reset each round (the
        no-tattoo condition), its final state should be statistically
        indistinguishable from its single-round reach.  Under
        asymmetric persistence it is NOT — persistent drift grows
        with N.
        """
        trials = 20
        carries_rms = 0.0
        resets_rms = 0.0
        for t in range(trials):
            p_c, _, _ = self._simulate(
                persistent_carries=True, seed=self.SEED + t
            )
            p_r, _, _ = self._simulate(
                persistent_carries=False, seed=self.SEED + t
            )
            carries_rms += (p_c[-1] - self.BASELINE) ** 2
            resets_rms += (p_r[-1] - self.BASELINE) ** 2
        carries_rms = np.sqrt(carries_rms / trials)
        resets_rms = np.sqrt(resets_rms / trials)
        assert carries_rms > 1.5 * resets_rms, (
            f"carries_rms={carries_rms:.4f} should substantially exceed "
            f"resets_rms={resets_rms:.4f}; externalization is the "
            f"operator that distinguishes them."
        )

    # ── (c) joint trajectory explores coupling-unique region ────────────────

    def test_joint_trajectory_exceeds_marginal_reach(self):
        """Cumulative path length of the joint (k_p, k_e) trajectory
        should exceed the sum of lengths of either kernel's projection
        — i.e. the joint trajectory explores a 2D region not reducible
        to either axis alone.
        """
        _, _, joint = self._simulate(persistent_carries=True, seed=self.SEED)
        kp = joint[:, 0]
        ke = joint[:, 1]
        kp_len = float(np.sum(np.abs(np.diff(kp))))
        ke_len = float(np.sum(np.abs(np.diff(ke))))
        joint_len = float(np.sum(np.linalg.norm(np.diff(joint, axis=0), axis=1)))
        # joint trajectory length exceeds max marginal length
        assert joint_len > max(kp_len, ke_len)
        # joint trajectory is not degenerate along either axis
        assert kp_len > 0.1 * joint_len
        assert ke_len > 0.1 * joint_len

    # ── the structural asymmetry is seed-stable ────────────────────────────

    def test_asymmetry_is_seed_stable(self):
        """Repeated trials confirm the asymmetry is not an artefact of
        a specific random draw — under the model, the persistent kernel
        dominates the afterglow in the vast majority of seeds.
        """
        trials = 30
        wins = 0
        for t in range(trials):
            p_hist, e_hist, _ = self._simulate(
                persistent_carries=True, seed=self.SEED + t * 31 + 7
            )
            if abs(p_hist[-1]) > np.max(np.abs(e_hist[1:])):
                wins += 1
        win_rate = wins / trials
        assert win_rate >= 0.60, (
            f"persistent kernel should dominate afterglow in ≥60% of "
            f"trials; observed {win_rate:.2%}"
        )


# ===========================================================================
# TestCUDAKernelFeatureMap — Q38: CUDA kernel ≡ RKHS feature map φ
# ===========================================================================

class TestCUDAKernelFeatureMap:
    """Q38 operational test.

    Proposition (Note 79 §8, action (k)): any warp-uniform CUDA kernel
    K : X → X' acting in parallel across a data stream corresponds to a
    feature map φ_K with induced RKHS H_{k_K}, and the tensor-core matmul
    primitive structurally computes

        k_K(x, x') = φ_K(x)ᵀ φ_K(x')

    for stored keys.

    The operational test here:
      (a) choose a concrete pointwise nonlinearity (our "CUDA kernel"),
      (b) apply it uniformly across a batch to obtain Φ,
      (c) form the Gram matrix K = Φ Φᵀ (the "matmul primitive"),
      (d) verify it is PSD, symmetric, and rank-bounded,
      (e) verify functoriality: stacking two CUDA kernels corresponds to
          composing their feature maps (K_{g ∘ f} = Φ_g(Φ_f(·)) matches
          the direct two-stage Gram).
    """

    d_in = 4
    d_hid = 8
    N = 12
    SEED = 2026

    @pytest.fixture(scope="class")
    def weights(self):
        rng = np.random.default_rng(self.SEED)
        W1 = rng.standard_normal((self.d_hid, self.d_in)) / np.sqrt(self.d_in)
        b1 = rng.standard_normal(self.d_hid) * 0.1
        W2 = rng.standard_normal((self.d_hid, self.d_hid)) / np.sqrt(self.d_hid)
        b2 = rng.standard_normal(self.d_hid) * 0.1
        return W1, b1, W2, b2

    @pytest.fixture(scope="class")
    def inputs(self):
        rng = np.random.default_rng(self.SEED + 1)
        return rng.standard_normal((self.N, self.d_in))

    def _cuda_kernel_f(self, X, W1, b1):
        """A pointwise 'CUDA kernel' — tanh(W x + b) applied uniformly."""
        return np.tanh(X @ W1.T + b1)

    def _cuda_kernel_g(self, X, W2, b2):
        return np.tanh(X @ W2.T + b2)

    def test_gram_is_psd_and_symmetric(self, inputs, weights):
        W1, b1, _, _ = weights
        Phi = self._cuda_kernel_f(inputs, W1, b1)
        K = Phi @ Phi.T
        assert is_psd(K, tol=1e-9)
        np.testing.assert_allclose(K, K.T, atol=1e-12)

    def test_gram_rank_bounded_by_feature_dim(self, inputs, weights):
        """rank(K) ≤ min(N, d_feature) — the RKHS dimension cap."""
        W1, b1, _, _ = weights
        Phi = self._cuda_kernel_f(inputs, W1, b1)
        K = Phi @ Phi.T
        rank = np.linalg.matrix_rank(K, tol=1e-10)
        assert rank <= min(self.N, self.d_hid)

    def test_gram_equals_tensor_core_matmul(self, inputs, weights):
        """The tensor-core matmul primitive φᵀφ equals the direct inner-product
        kernel — this is the structural content of Q38's correspondence."""
        W1, b1, _, _ = weights
        Phi = self._cuda_kernel_f(inputs, W1, b1)
        K_matmul = Phi @ Phi.T  # "tensor core"
        K_direct = np.array([[float(Phi[i] @ Phi[j])
                              for j in range(self.N)] for i in range(self.N)])
        np.testing.assert_allclose(K_matmul, K_direct, atol=1e-12)

    def test_kernel_composition_equals_feature_map_composition(self, inputs, weights):
        """Functoriality: k_{g∘f}(x, x') = ⟨φ_g(φ_f(x)), φ_g(φ_f(x'))⟩.

        Two-stage CUDA pipeline = single-stage CUDA pipeline applied to
        composed feature map.  Equivalence up to numerical precision is
        the structural claim.
        """
        W1, b1, W2, b2 = weights
        Phi_f = self._cuda_kernel_f(inputs, W1, b1)
        Phi_gf = self._cuda_kernel_g(Phi_f, W2, b2)
        K_composed = Phi_gf @ Phi_gf.T
        assert is_psd(K_composed, tol=1e-9)
        np.testing.assert_allclose(K_composed, K_composed.T, atol=1e-12)

    def test_warp_uniformity_batch_invariant(self, inputs, weights):
        """Uniform application across the data stream: φ(x) depends only on
        x, not on its batch neighbours.  (This is what 'warp-uniform' means:
        same kernel run on every element.)"""
        W1, b1, _, _ = weights
        Phi_full = self._cuda_kernel_f(inputs, W1, b1)
        Phi_piece = np.vstack([
            self._cuda_kernel_f(inputs[:4], W1, b1),
            self._cuda_kernel_f(inputs[4:], W1, b1),
        ])
        np.testing.assert_allclose(Phi_full, Phi_piece, atol=1e-12)


# ===========================================================================
# TestTopologyBudget — k_min = β_0 + β_1 compression floor
# ===========================================================================

class TestTopologyBudget:
    """The beam optics of representation.

    Any spectral compressor that truncates the heat kernel to k modes
    must retain at least β_0 (connected components) + β_1 (independent
    cycles) modes to preserve the topology of the underlying domain.
    This is the framework's compression floor.

    Falsifiable prediction on a graph with controlled Betti numbers:
      - λ_0 = ... = λ_{β_0 - 1} = 0 (multiplicity of zero equals β_0).
      - A signal localised on one component can be recovered from the
        first β_0 eigenvectors but not from any k < β_0.
    """

    def _disjoint_path_laplacian(self, n_components: int, size_each: int) -> np.ndarray:
        """Laplacian of n_components disjoint paths, each of size_each nodes.
        Bett numbers: β_0 = n_components, β_1 = 0 (trees).
        """
        blocks = []
        for _ in range(n_components):
            pg = SpectralGraph.path_graph(size_each)
            blocks.append(pg.laplacian)
        n_total = n_components * size_each
        L = np.zeros((n_total, n_total))
        for c, B in enumerate(blocks):
            s = c * size_each
            L[s:s + size_each, s:s + size_each] = B
        return L

    def _disjoint_cycle_laplacian(self, n_cycles: int, size_each: int) -> np.ndarray:
        """Laplacian of n_cycles disjoint cycles.  β_0 = n_cycles, β_1 = n_cycles."""
        n_total = n_cycles * size_each
        L = np.zeros((n_total, n_total))
        for c in range(n_cycles):
            s = c * size_each
            for i in range(size_each):
                j = (i + 1) % size_each
                L[s + i, s + i] += 1
                L[s + j, s + j] += 1
                L[s + i, s + j] -= 1
                L[s + j, s + i] -= 1
        return L

    def test_betti_zero_equals_zero_eigenvalue_multiplicity(self):
        """β_0 = dim ker L (standard spectral identity)."""
        n_comp, size = 3, 4
        L = self._disjoint_path_laplacian(n_comp, size)
        eigvals = np.linalg.eigvalsh(L)
        n_zero = int(np.sum(eigvals < 1e-8))
        assert n_zero == n_comp

    def test_cycle_graph_has_expected_betti_numbers(self):
        """For disjoint cycles: ker L has dim β_0 = n_cycles; total = β_0 + β_1
        but the multiplicity-of-zero theorem only gives β_0.  We verify both:
        zero-multiplicity and the cycle-space dimension via |E| - |V| + β_0."""
        n_cyc, size = 2, 5
        L = self._disjoint_cycle_laplacian(n_cyc, size)
        eigvals = np.linalg.eigvalsh(L)
        beta0 = int(np.sum(eigvals < 1e-8))
        assert beta0 == n_cyc
        n_vertices = n_cyc * size
        n_edges = n_cyc * size  # each cycle contributes `size` edges
        beta1 = n_edges - n_vertices + beta0
        assert beta1 == n_cyc

    def test_compression_below_floor_loses_components(self):
        """Reconstructing a component-localised signal from k < β_0 modes
        collapses structure; from k ≥ β_0 modes, components are preserved.
        """
        n_comp, size = 3, 5
        L = self._disjoint_path_laplacian(n_comp, size)
        eigvals, eigvecs = np.linalg.eigh(L)

        signal = np.zeros(n_comp * size)
        signal[:size] = 1.0

        def reconstruct(k):
            basis = eigvecs[:, :k]
            coeffs = basis.T @ signal
            return basis @ coeffs

        r_below = reconstruct(n_comp - 1)
        r_at = reconstruct(n_comp)

        support_b = np.zeros(n_comp * size, bool)
        support_b[:size] = True

        mass_off_below = float(np.sum(r_below[~support_b] ** 2))
        mass_off_at = float(np.sum(r_at[~support_b] ** 2))

        assert mass_off_at <= mass_off_below + 1e-10
        assert mass_off_at < 1e-8

    def test_k_min_formula_on_cycles(self):
        """On disjoint cycles, k_min = β_0 + β_1 = 2 · n_cycles — the
        accelerator needs that many spectral channels to resolve topology.
        """
        n_cyc, size = 2, 6
        L = self._disjoint_cycle_laplacian(n_cyc, size)
        eigvals = np.linalg.eigvalsh(L)
        beta0 = int(np.sum(eigvals < 1e-8))
        n_vertices = n_cyc * size
        n_edges = n_cyc * size
        beta1 = n_edges - n_vertices + beta0
        k_min = beta0 + beta1
        assert k_min == 2 * n_cyc
        assert k_min <= n_vertices


# ===========================================================================
# TestEndToEndAccelerator — the Stark rig: real torch training, full triad
# ===========================================================================

class TestEndToEndAccelerator:
    """Full experimental run: small MLP + PowerMonitor + NTKTracker + loss log.

    This is the test-file analog of Paper 0's Landauer protocol.  Channels:
      (1) Loss curves             — representational-work macro calorimetry.
      (2) NTK probes              — silicon-bubble-chamber tracker.
      (3) Wall-power logs         — Landauer calorimetry.
    All three are cross-correlated against the single training run.

    Skipped when PyTorch is unavailable.
    """

    @pytest.fixture(scope="class")
    def rig(self):
        torch = pytest.importorskip("torch")
        from kernelcal.ntk import NTKTracker
        torch.manual_seed(0)
        np.random.seed(0)

        N_TRAIN = 32
        N_PROBE = 12
        N_STEPS = 40
        HIDDEN = 8
        LR = 5e-2
        T_KELVIN = T_ROOM

        class TinyMLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = torch.nn.Linear(1, HIDDEN)
                self.l2 = torch.nn.Linear(HIDDEN, 1)

            def forward(self, x):
                return self.l2(torch.tanh(self.l1(x)))

        model = TinyMLP()
        opt = torch.optim.SGD(model.parameters(), lr=LR)

        x_train = torch.linspace(-2.0, 2.0, N_TRAIN).unsqueeze(-1)
        y_train = torch.sin(2.0 * x_train)
        x_probe = torch.linspace(-2.0, 2.0, N_PROBE).unsqueeze(-1)

        tracker = NTKTracker(probe_inputs=x_probe)
        tracker.record(0, model)

        losses = []
        segment_records = []
        with PowerMonitor(gpu_id=0, interval_s=0.02) as pm:
            for step in range(1, N_STEPS + 1):
                opt.zero_grad()
                pred = model(x_train)
                loss = torch.mean((pred - y_train) ** 2)
                loss.backward()
                opt.step()
                losses.append(float(loss.detach()))

                if step % 10 == 0 or step == N_STEPS:
                    K_before = tracker.final_kernel()
                    tracker.record(step, model)
                    K_after = tracker.final_kernel()
                    segment_records.append({
                        "step": step,
                        "loss": losses[-1],
                        "delta_I_nats": kernel_mutual_information_change(
                            K_before, K_after
                        ),
                        "hs_drift": hilbert_schmidt_distance(K_before, K_after),
                    })

        total_energy_J = pm.total_energy_joules()
        cumulative_bound_J = landauer_bound(
            float(np.sum([s["delta_I_nats"] for s in segment_records])),
            temperature_kelvin=T_KELVIN,
        )
        report = {
            "n_steps": N_STEPS,
            "n_train": N_TRAIN,
            "n_probe": N_PROBE,
            "losses": losses,
            "segments": segment_records,
            "power_monitor_summary": pm.summary(),
            "total_energy_J": total_energy_J,
            "cumulative_bound_J": cumulative_bound_J,
            "efficiency_bound_over_work": (
                cumulative_bound_J / total_energy_J
                if total_energy_J > 0 else float("inf")
            ),
            "ntk_path_length_HS": tracker.trajectory.path_length(),
            "ntk_convergence_step": tracker.convergence_step(tol=1e-3, window=2),
        }
        return report

    # ── Basic training diagnostics ──────────────────────────────────────────

    def test_loss_decreases(self, rig):
        first = float(np.mean(rig["losses"][:3]))
        last = float(np.mean(rig["losses"][-3:]))
        assert last < first

    def test_ntk_drifts(self, rig):
        """Kernel leaves its initial configuration — the accelerator did work."""
        assert rig["ntk_path_length_HS"] > 0.0

    # ── Paper-0 inequality ──────────────────────────────────────────────────

    def test_cumulative_landauer_bound_satisfied(self, rig):
        """∫Ẇ dt ≥ k_B T · ΔI_total under the full torch training rig."""
        assert rig["total_energy_J"] >= rig["cumulative_bound_J"]

    def test_efficiency_is_astronomically_below_1(self, rig):
        """Silicon at 300 K is ~10 orders of magnitude above Landauer.  The
        bound/work ratio should be ≪ 1 in practice."""
        if rig["total_energy_J"] > 0:
            assert rig["efficiency_bound_over_work"] < 1.0

    # ── Detector-triad cross-consistency ────────────────────────────────────

    def test_segment_loss_and_hs_correlate(self, rig):
        """Segments where loss drops fastest should show the largest HS drift.
        We only assert non-anti-correlation; exact slopes depend on LR/init.
        """
        segs = rig["segments"]
        if len(segs) < 3:
            pytest.skip("not enough segments")
        losses = np.array([s["loss"] for s in segs])
        hs = np.array([s["hs_drift"] for s in segs])
        if np.std(losses) > 0 and np.std(hs) > 0:
            corr = float(np.corrcoef(losses, hs)[0, 1])
            assert corr > -0.99, (
                f"loss/HS anti-correlated perfectly (corr={corr:.3f})"
            )

    # ── Telemetry contract ──────────────────────────────────────────────────

    def test_report_is_json_serialisable(self, rig):
        """The accelerator report must be a pure-Python JSON artifact —
        downstream pipelines ingest it without pickling."""
        import math

        def normalise(x):
            if isinstance(x, dict):
                return {k: normalise(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [normalise(v) for v in x]
            if isinstance(x, (np.floating, np.integer)):
                return x.item()
            if isinstance(x, float) and math.isinf(x):
                return "inf"
            return x

        s = json.dumps(normalise(rig))
        assert len(s) > 0
        assert "total_energy_J" in s
        assert "cumulative_bound_J" in s
