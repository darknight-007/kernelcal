"""
MaxCal-based bloom-following controller.

Maps the algae bloom field to a velocity command for the 2D rover by selecting
the next waypoint from a Maximum Caliber distribution over candidate positions,
then driving toward it with a proportional velocity controller.

MaxCal formulation
------------------
State space  Γ = {x_1, …, x_N}  — N candidate next-positions sampled around
the rover's current location.

Reference measure:

    q(x_i) ∝ exp(−d(x_i, x_now)² / (2 ℓ_q²))

a Gaussian proximity prior that penalises energetically expensive moves.

Constraints on the path distribution p:

    ⟨b(x_i)⟩_p          ≥ B_target    (bloom concentration — seek the bloom)
    ⟨∥∇b(x_i)∥⟩_p       ≥ G_target    (gradient magnitude — track bloom fronts)
    ⟨d(x_i, x_now)⟩_p   ≤ D_target    (energy budget)

MaxCal solution (Eq. 2 of the paper):

    log p(x_i) = log q(x_i) − λ_B f_B(x_i) − λ_G f_G(x_i) − λ_D f_D(x_i) − log Z

where {λ_i} are found by minimising the Lagrange dual L(λ) = log Z(λ) − λ·F,
using kernelcal.maxcal.functional.fit_lagrange_multipliers.

The kernel k_t over candidate positions is an RBF kernel on ℝ²:

    k(x_i, x_j) = exp(−∥x_i − x_j∥² / (2 ℓ_k²))

Kernel trajectory tracking — HS distance between successive k_t matrices is
monitored via kernelcal.kernel.space.hilbert_schmidt_distance, giving a
quantitative measure of how rapidly the bloom-following strategy is evolving
(cf. Section VI.C of the paper).

Thermodynamic accounting
------------------------
Each step acquires mutual information

    ΔI_t = log(Var_prior / Var_posterior)   (proxy from concentration belief)

which lower-bounds the thermodynamic work cost by the Landauer relation
(Eq. 13 of the paper, via kernelcal.thermodynamics.bounds.landauer_bound).
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.special import logsumexp

# kernelcal imports — resolved via sys.path manipulation in the ROS2 node;
# for standalone use the package must be installed or on PYTHONPATH.
try:
    from kernelcal.kernel.space import rbf_kernel, hilbert_schmidt_distance
    from kernelcal.thermodynamics.bounds import landauer_bound
    _KERNELCAL_AVAILABLE = True
except ImportError:
    _KERNELCAL_AVAILABLE = False

    def rbf_kernel(X, Y=None, length_scale=1.0):  # type: ignore
        X = np.asarray(X)
        Y = X if Y is None else np.asarray(Y)
        d2 = np.sum((X[:, None] - Y[None, :]) ** 2, axis=-1)
        return np.exp(-d2 / (2 * length_scale ** 2))

    def hilbert_schmidt_distance(K1, K2):  # type: ignore
        return float(np.linalg.norm(K1 - K2, "fro") / np.sqrt(K1.shape[0]))

    def landauer_bound(delta_I_nats, T_kelvin=298.15):  # type: ignore
        kB = 1.380649e-23
        return float(kB * T_kelvin * delta_I_nats)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MaxCalConfig:
    """Parameters for the MaxCal bloom-following controller."""
    # Candidate generation
    n_candidates: int = 32          # number of candidate next-positions
    lookahead_min: float = 3.0      # minimum lookahead radius (m)
    lookahead_max: float = 10.0     # maximum lookahead radius (m)
    n_angle_bins: int = 16          # angular bins for candidate generation

    # Reference measure (proximity prior)
    sigma_q: float = 6.0            # Gaussian prior width (m)

    # Constraint targets (set dynamically if None)
    bloom_conc_target: Optional[float] = None   # B_target; auto if None
    gradient_target: Optional[float] = None     # G_target; auto if None
    distance_target: Optional[float] = None     # D_target; auto if None

    # How aggressively to set auto targets (fraction of observed range)
    bloom_target_quantile: float = 0.65
    gradient_target_quantile: float = 0.55
    distance_target_fraction: float = 0.50

    # RBF kernel length scale for HS-distance tracking
    kernel_length_scale: float = 5.0

    # Thermodynamic temperature for Landauer bound
    temperature_kelvin: float = 298.15

    # --- Strategy mode ---
    # 'maxcal'     : full MaxCal — λ fitted to bloom utility constraint (default)
    # 'static'     : static kernel — proximity prior q only, λ ≡ 0
    #                (path distribution fixed regardless of bloom)
    # 'adaptive_q' : adaptive-kernel baseline — same Gaussian kernel class as
    #                'static', but σ_q is updated each step from a running EMA of
    #                bloom concentration.  High bloom → narrow σ_q (exploit);
    #                low bloom → wide σ_q (explore).  No MaxCal optimisation.
    # 'greedy'     : practical baseline — always select the candidate with the
    #                highest bloom utility f(x) = b(x) + 0.4|∇b(x)|, irrespective
    #                of distance.  Same feature access as MaxCal; tests whether
    #                entropy maintenance (distributional selection) beats argmax.
    # 'gradient'   : classical heuristic — select the candidate whose direction
    #                from the rover best aligns with the local ∇b at the current
    #                position, scaled by the bloom gradient magnitude.
    #                Independent of the MaxCal framework entirely.
    mode: str = 'maxcal'

    # Backward-compatible alias (overrides mode when False)
    use_maxcal: bool = True

    # Adaptive-q hyper-parameters
    adaptive_q_ema_alpha: float = 0.15    # EMA smoothing factor for bloom obs
    adaptive_q_sigma_min: float = 2.5     # σ_q floor (m) — tight exploitation
    adaptive_q_sigma_max: float = 12.0    # σ_q ceiling (m) — wide exploration

    # Velocity controller gains
    v_max: float = 1.2              # m/s
    k_omega: float = 1.8            # rad/s per rad
    k_v_theta: float = 2.5          # heading-error speed gate sharpness
    arrival_radius: float = 2.5     # m

    # Domain bounds (clipping for candidates)
    domain_x: Tuple[float, float] = (0.0, 100.0)
    domain_y: Tuple[float, float] = (0.0, 100.0)


# ---------------------------------------------------------------------------
# Online GP surrogate  — no oracle access to unvisited positions
# ---------------------------------------------------------------------------

class BloomGP:
    """Incremental Gaussian-process surrogate for bloom concentration.

    Built solely from the rover's own noisy observations at visited positions.
    Provides posterior mean and std at unvisited candidate locations, replacing
    any direct queries to the ground-truth bloom field at those locations.

    Model
    -----
    Prior:  f ~ GP(0, k_f)  with  k_f(x,x') = σ_f² exp(−‖x−x'‖²/2ℓ²)
    Likelihood:  y = f(x) + ε,  ε ~ N(0, σ_n²)
    Posterior mean:  μ*(X*) = K(X*,X) [K(X,X)+σ_n²I]⁻¹ y
    Posterior var:   σ²*(X*) = σ_f² − diag(K(X*,X) [K(X,X)+σ_n²I]⁻¹ K(X,X*))

    A sliding window of at most ``max_obs`` recent observations is kept to
    bound inference cost at O(max_obs²) per prediction.
    """

    def __init__(
        self,
        length_scale: float = 12.0,   # m — roughly 1 bloom-patch sigma
        noise_std: float = 0.02,       # observation noise (matches rover sensor)
        prior_std: float = 1.0,        # prior amplitude
        max_obs: int = 80,             # sliding window size
    ) -> None:
        self.ell = length_scale
        self.sigma_n2 = noise_std ** 2
        self.sigma_f2 = prior_std ** 2
        self.max_obs = max_obs
        self._X: List[List[float]] = []
        self._y: List[float] = []
        # Cached factorisation — invalidated on each update
        self._cache: Optional[dict] = None

    @property
    def n_obs(self) -> int:
        return len(self._X)

    def update(self, x: float, y: float, b_obs: float) -> None:
        """Add an observation and invalidate the cached Cholesky factor."""
        self._X.append([x, y])
        self._y.append(float(b_obs))
        if len(self._X) > self.max_obs:
            self._X = self._X[-self.max_obs:]
            self._y = self._y[-self.max_obs:]
        self._cache = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rbf(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        d2 = np.sum((A[:, None] - B[None, :]) ** 2, axis=-1)
        return self.sigma_f2 * np.exp(-d2 / (2.0 * self.ell ** 2))

    def _chol_factor(self) -> Optional[dict]:
        """Lazily compute and cache the Cholesky factor of K(X,X)+σ_n²I."""
        if self._cache is not None:
            return self._cache
        if self.n_obs == 0:
            return None
        X = np.array(self._X)
        y = np.array(self._y)
        K = self._rbf(X, X) + self.sigma_n2 * np.eye(len(X))
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
            self._cache = {'X': X, 'L': L, 'alpha': alpha}
        except np.linalg.LinAlgError:
            self._cache = None
        return self._cache

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self, candidates: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """GP posterior (mean, std) at candidate positions.

        Returns non-negative mean (bloom conc. is ≥ 0) and std ≥ 0.
        Falls back to (0, prior_std) when no observations are available.
        """
        fac = self._chol_factor()
        if fac is None:
            return (
                np.zeros(len(candidates)),
                np.full(len(candidates), math.sqrt(self.sigma_f2)),
            )
        X, L, alpha = fac['X'], fac['L'], fac['alpha']
        K_xX = self._rbf(candidates, X)          # (M, N)
        mean = np.clip(K_xX @ alpha, 0.0, None)

        v = np.linalg.solve(L, K_xX.T)           # (N, M)
        var = np.clip(self.sigma_f2 - np.sum(v ** 2, axis=0), 1e-9, None)
        return mean, np.sqrt(var)

    def predict_gradient(
        self, x: float, y: float, h: float = 1.0
    ) -> Tuple[float, float]:
        """Finite-difference gradient of GP posterior mean at (x, y).

        Uses four query points in a single batched GP call.
        Returns (0, 0) when insufficient observations are available.
        """
        if self.n_obs < 3:
            return 0.0, 0.0
        pts = np.array([[x + h, y], [x - h, y], [x, y + h], [x, y - h]])
        m, _ = self.predict(pts)
        gx = (m[0] - m[1]) / (2.0 * h)
        gy = (m[2] - m[3]) / (2.0 * h)
        return float(gx), float(gy)

    def candidate_grad_magnitudes(
        self, candidates: np.ndarray, h: float = 1.0
    ) -> np.ndarray:
        """Batch ‖∇GP-mean‖ at all candidate positions (efficient single call)."""
        M = len(candidates)
        pts = np.vstack([
            candidates + [h, 0],
            candidates - [h, 0],
            candidates + [0, h],
            candidates - [0, h],
        ])
        m, _ = self.predict(pts)
        gx = (m[:M] - m[M:2 * M]) / (2.0 * h)
        gy = (m[2 * M:3 * M] - m[3 * M:]) / (2.0 * h)
        return np.hypot(gx, gy)


# ---------------------------------------------------------------------------
# Diagnostic record
# ---------------------------------------------------------------------------

@dataclass
class MaxCalStepRecord:
    """Per-step diagnostic information."""
    step: int
    t: float
    rover_x: float
    rover_y: float
    waypoint_x: float
    waypoint_y: float
    bloom_obs: float
    gradient_mag: float
    v_cmd: float
    omega_cmd: float
    entropy_nats: float
    hs_distance: float          # HS distance from previous kernel
    lagrange_multipliers: List[float]
    landauer_bound_nJ: float    # Landauer lower bound (nanojoules)
    arrived: bool


# ---------------------------------------------------------------------------
# Core controller
# ---------------------------------------------------------------------------

class MaxCalBloomFollower:
    """MaxCal-based bloom-following velocity controller.

    At each call to ``update()``:
    1. Generate N candidate next-positions around the rover.
    2. Evaluate bloom concentration b(x_i) and gradient ||∇b(x_i)|| at each.
    3. Build the MaxCal distribution p over candidates via kernelcal.
    4. Select the highest-probability candidate as the next waypoint.
    5. Compute (v, ω) commands to drive toward it.
    6. Track kernel trajectory in Hilbert-Schmidt metric and Landauer bound.

    Parameters
    ----------
    config : MaxCalConfig

    Usage
    -----
    >>> follower = MaxCalBloomFollower()
    >>> v, omega = follower.update(rover, bloom_field, dt=0.1)
    """

    def __init__(self, config: Optional[MaxCalConfig] = None) -> None:
        self.cfg = config or MaxCalConfig()

        # Resolve back-compat: use_maxcal=False overrides mode to 'static'
        if not self.cfg.use_maxcal and self.cfg.mode == 'maxcal':
            self.cfg.mode = 'static'

        self._step: int = 0
        self._t: float = 0.0
        self._waypoint: Optional[np.ndarray] = None
        self._lambdas: np.ndarray = np.zeros(3)
        self._prev_kernel: Optional[np.ndarray] = None
        self._history: List[MaxCalStepRecord] = []

        # Running belief for Landauer bound proxy
        self._bloom_variance: float = 0.1

        # Adaptive-q: exponential moving average of observed bloom concentration
        self._bloom_ema: float = 0.15

        # Online GP surrogate — built from the rover's own observations only.
        # No ground-truth bloom queries at unvisited candidate positions.
        self._gp = BloomGP(
            length_scale=self.cfg.kernel_length_scale * 2.0,
            noise_std=0.02,
            prior_std=1.0,
            max_obs=80,
        )

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------

    def _generate_candidates(self, x: float, y: float, rng: np.random.Generator) -> np.ndarray:
        """Generate N candidate positions around (x, y).

        Uses a multi-ring scheme: angular bins at several radii, plus
        the current position (so the rover can "stay" if optimal).
        """
        cfg = self.cfg
        angles = np.linspace(0.0, 2.0 * np.pi, cfg.n_angle_bins, endpoint=False)
        radii = np.linspace(cfg.lookahead_min, cfg.lookahead_max, 3)

        cands = []
        for r in radii:
            for a in angles:
                cx = x + r * math.cos(a)
                cy = y + r * math.sin(a)
                cands.append([cx, cy])

        # Add a few random candidates for diversity
        n_random = max(1, cfg.n_candidates - len(cands))
        for _ in range(n_random):
            a = rng.uniform(0, 2 * np.pi)
            r = rng.uniform(cfg.lookahead_min, cfg.lookahead_max)
            cands.append([x + r * math.cos(a), y + r * math.sin(a)])

        candidates = np.array(cands, dtype=float)
        # Clip to domain
        candidates[:, 0] = np.clip(candidates[:, 0], *cfg.domain_x)
        candidates[:, 1] = np.clip(candidates[:, 1], *cfg.domain_y)
        return candidates  # (M, 2)

    # ------------------------------------------------------------------
    # MaxCal distribution  (correct dual formulation)
    # ------------------------------------------------------------------

    def _build_maxcal_distribution(
        self,
        candidates: np.ndarray,
        bloom_field,
        rover_x: float,
        rover_y: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute MaxCal log-probabilities over candidate positions.

        Feature evaluation uses the rover's online GP surrogate — no direct
        queries to the ground-truth bloom field at unvisited candidate positions.

        The GP is trained only on (position, observation) pairs from the rover's
        own trajectory.  Early in the mission when the GP is uncertain, all
        candidates look similar and the prior q dominates (correct behaviour).

        bloom_field is still passed for the 'gradient' mode, which may use the
        rover's on-board gradient sensor at the current (visited) position.

        Returns
        -------
        log_p      : (M,) normalised log-probabilities
        log_q      : (M,) reference log-weights
        bloom_vals : (M,) GP posterior mean at each candidate
        grad_vals  : (M,) GP gradient magnitude at each candidate
        dist_vals  : (M,) Euclidean distance from current rover position
        """
        cfg = self.cfg

        # ------------------------------------------------------------------
        # Feature evaluation — GP surrogate ONLY, no oracle bloom access
        # ------------------------------------------------------------------
        bloom_vals, _bloom_std = self._gp.predict(candidates)
        grad_vals = self._gp.candidate_grad_magnitudes(candidates)
        dist_vals = np.linalg.norm(candidates - np.array([rover_x, rover_y]), axis=1)

        # Reference measure: proximity prior (energy model) on the bounded ring
        log_q = -dist_vals ** 2 / (2.0 * cfg.sigma_q ** 2)
        log_q -= logsumexp(log_q)

        mode = cfg.mode

        # ------------------------------------------------------------------
        # Mode: 'static'
        # Path distribution = reference measure q.  λ ≡ 0.
        # No bloom information enters waypoint selection.
        # ------------------------------------------------------------------
        if mode == 'static':
            self._lambdas = np.zeros(3)
            return log_q.copy(), log_q, bloom_vals, grad_vals, dist_vals

        # ------------------------------------------------------------------
        # Mode: 'greedy'
        # Select the candidate with the highest bloom utility, regardless of
        # distance.  Puts all probability mass on argmax f(x_i).
        # This is the strongest practical baseline: same candidate information
        # as MaxCal, zero entropy, no proximity prior.
        # ------------------------------------------------------------------
        if mode == 'greedy':
            utility_g = bloom_vals + 0.4 * grad_vals
            best = int(np.argmax(utility_g))
            log_p_g = np.full(len(candidates), -1e30)
            log_p_g[best] = 0.0          # all mass on argmax
            self._lambdas = np.zeros(3)
            return log_p_g, log_q, bloom_vals, grad_vals, dist_vals

        # ------------------------------------------------------------------
        # Mode: 'gradient'
        # Classical bloom-front following: pick the candidate whose bearing
        # most closely aligns with the GP-estimated ∇b at the current position.
        # The rover's GP is built from its own past observations — no oracle.
        # Falls back to greedy-on-GP when gradient estimate is unreliable
        # (< 3 observations).
        # ------------------------------------------------------------------
        if mode == 'gradient':
            gx, gy = self._gp.predict_gradient(rover_x, rover_y)
            grad_mag_local = math.hypot(gx, gy)
            if grad_mag_local < 1e-9:
                # Insufficient GP data — fall back to greedy-on-GP
                utility_fb = bloom_vals + 0.4 * grad_vals
                best_fb = int(np.argmax(utility_fb))
                log_p_fb = np.full(len(candidates), -1e30)
                log_p_fb[best_fb] = 0.0
                self._lambdas = np.zeros(3)
                return log_p_fb, log_q, bloom_vals, grad_vals, dist_vals
            dx = candidates[:, 0] - rover_x
            dy = candidates[:, 1] - rover_y
            norms = np.hypot(dx, dy) + 1e-12
            cos_align = (dx * gx + dy * gy) / (norms * grad_mag_local)
            log_p_gr = 3.0 * cos_align
            log_p_gr -= logsumexp(log_p_gr)
            self._lambdas = np.zeros(3)
            return log_p_gr, log_q, bloom_vals, grad_vals, dist_vals

        # ------------------------------------------------------------------
        # Mode: 'adaptive_q'
        # Same Gaussian kernel class as 'static', but σ_q is rescaled each
        # step by a running EMA of the rover's bloom observations:
        #
        #   σ_q_eff = σ_q_min + (σ_q_max − σ_q_min) · exp(−k · b̄)
        #
        # where b̄ is the EMA-smoothed bloom and k=3 maps [0,1] → [σ_max, σ_min].
        # High bloom → σ_q_eff ≈ σ_q_min  (tight exploitation)
        # Low  bloom → σ_q_eff ≈ σ_q_max  (wide exploration)
        # λ ≡ 0 throughout — no MaxCal optimisation.
        # ------------------------------------------------------------------
        if mode == 'adaptive_q':
            b_ema = float(np.clip(self._bloom_ema, 0.0, 2.0))
            k_decay = 3.0
            alpha = math.exp(-k_decay * b_ema)   # 1 when b_ema=0, →0 when b_ema large
            sigma_q_eff = cfg.adaptive_q_sigma_min + (
                cfg.adaptive_q_sigma_max - cfg.adaptive_q_sigma_min
            ) * alpha
            log_q_eff = -dist_vals ** 2 / (2.0 * sigma_q_eff ** 2)
            log_q_eff -= logsumexp(log_q_eff)
            self._lambdas = np.zeros(3)
            return log_q_eff.copy(), log_q_eff, bloom_vals, grad_vals, dist_vals

        # ------------------------------------------------------------------
        # Mode: 'maxcal'  (default)
        # Full MaxCal: fit λ to maximise bloom utility constraint.
        # ------------------------------------------------------------------
        # Feature matrix  (M × 1): [utility]
        # Combined utility weights bloom more heavily; bloom fronts also rewarded.
        utility = bloom_vals + 0.4 * grad_vals   # single informative feature
        F_mat = utility.reshape(-1, 1)

        # Constraint target: mean utility under proximity prior ⟨u⟩_q,
        # shifted upward by a fraction of the std to seek above-average utility.
        u_q = float(np.exp(log_q) @ utility)          # reference mean utility
        u_std = float(np.std(utility))
        shift_fraction = 1.0 + cfg.bloom_target_quantile * 2.0  # adaptive shift
        B_target = u_q + shift_fraction * max(u_std, 1e-6)
        B_target = float(np.clip(B_target, u_q + 1e-6, utility.max() * 0.98))
        F_target = np.array([B_target])

        # Correct MaxCal dual minimisation
        log_p, lambdas = _correct_maxcal(log_q, F_mat, F_target, self._lambdas[:1])
        self._lambdas = np.pad(lambdas, (0, max(0, 3 - len(lambdas))), constant_values=0.0)

        return log_p, log_q, bloom_vals, grad_vals, dist_vals

    # ------------------------------------------------------------------
    # Kernel trajectory tracking
    # ------------------------------------------------------------------

    def _update_kernel_tracking(self, candidates: np.ndarray, log_p: np.ndarray) -> float:
        """Update the RBF kernel over candidates weighted by p; return HS distance."""
        cfg = self.cfg
        K_base = rbf_kernel(candidates, length_scale=cfg.kernel_length_scale)
        p = np.exp(log_p)
        K_t = np.outer(p, p) * K_base   # probability-weighted kernel matrix

        if self._prev_kernel is None or self._prev_kernel.shape != K_t.shape:
            hs_dist = 0.0
        else:
            hs_dist = hilbert_schmidt_distance(K_t, self._prev_kernel)

        self._prev_kernel = K_t
        return float(hs_dist)

    # ------------------------------------------------------------------
    # Waypoint velocity controller
    # ------------------------------------------------------------------

    def _velocity_toward_waypoint(
        self,
        rover_x: float,
        rover_y: float,
        rover_theta: float,
        waypoint: np.ndarray,
    ) -> Tuple[float, float, bool]:
        """Proportional controller: drive toward *waypoint*.

        Returns (v, omega, arrived).
        """
        cfg = self.cfg
        dx = waypoint[0] - rover_x
        dy = waypoint[1] - rover_y
        dist = math.hypot(dx, dy)

        if dist < cfg.arrival_radius:
            return 0.0, 0.0, True

        desired_heading = math.atan2(dy, dx)
        heading_error = _wrap_angle(desired_heading - rover_theta)

        omega = float(np.clip(cfg.k_omega * heading_error, -1.2, 1.2))
        speed_gate = math.exp(-cfg.k_v_theta * heading_error ** 2)
        # slow down near waypoint
        proximity_factor = min(1.0, dist / (2.0 * cfg.arrival_radius))
        v = float(np.clip(cfg.v_max * speed_gate * proximity_factor, 0.0, cfg.v_max))

        return v, omega, False

    # ------------------------------------------------------------------
    # Landauer bound proxy
    # ------------------------------------------------------------------

    def _landauer_step(self, bloom_obs: float) -> float:
        """Estimate nanojoules lower-bounded by Landauer for this step's info gain.

        Uses a running Bayesian variance update:  posterior var ← prior var × ρ
        where ρ < 1 represents information acquired from the observation.
        This is a simplified proxy; a full GP would give exact values.
        """
        sigma_obs = 0.02  # rover observation noise
        prior_var = self._bloom_variance
        posterior_var = 1.0 / (1.0 / prior_var + 1.0 / sigma_obs ** 2)
        delta_I_nats = 0.5 * math.log(prior_var / (posterior_var + 1e-300))
        delta_I_nats = max(delta_I_nats, 0.0)
        self._bloom_variance = posterior_var + 1e-4  # relax toward prior
        bound_J = landauer_bound(delta_I_nats, self.cfg.temperature_kelvin)
        return float(bound_J * 1e9)  # nanojoules

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(
        self,
        rover,
        bloom_field,
        dt: float,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[float, float]:
        """Compute and return (v, ω) velocity commands for the rover.

        Parameters
        ----------
        rover : DifferentialDriveRover
        bloom_field : AlgaeBloomField
        dt : float — time step (s); rover.step() is NOT called here
        rng : optional RNG for candidate generation

        Returns
        -------
        v : float — linear velocity command (m/s)
        omega : float — angular velocity command (rad/s)
        """
        if rng is None:
            rng = np.random.default_rng(self._step)

        x, y, theta = rover.pose()

        # 1. Observe bloom at CURRENT position (the rover is here — this is
        #    a sensor reading, not oracle access to an unvisited location).
        bloom_obs = rover.observe_bloom(bloom_field)
        grad_mag  = rover.observe_gradient_magnitude(bloom_field)

        # 2. Update the GP surrogate and EMA with this observation BEFORE
        #    selecting the next waypoint. Candidates will be evaluated using
        #    the updated GP posterior — no ground-truth bloom queries elsewhere.
        self._gp.update(x, y, float(bloom_obs))
        alpha_ema = self.cfg.adaptive_q_ema_alpha
        self._bloom_ema = (1.0 - alpha_ema) * self._bloom_ema + alpha_ema * float(bloom_obs)

        # 3. Generate candidate next-positions
        candidates = self._generate_candidates(x, y, rng)

        # 4. Strategy-specific distribution over candidates (GP-based features)
        log_p, log_q, bloom_vals, grad_vals, dist_vals = self._build_maxcal_distribution(
            candidates, bloom_field, x, y
        )
        p = np.exp(log_p)

        # 5. Select best candidate (highest probability)
        best_idx = int(np.argmax(p))
        waypoint = candidates[best_idx]
        self._waypoint = waypoint

        # 6. Kernel tracking (HS distance from previous step)
        hs_dist = self._update_kernel_tracking(candidates, log_p)

        # 7. Velocity command toward waypoint
        v_cmd, omega_cmd, arrived = self._velocity_toward_waypoint(x, y, theta, waypoint)

        # 8. Landauer bound from information acquired this step
        landauer_nJ = self._landauer_step(bloom_obs)

        # Entropy of MaxCal distribution (nats)
        entropy = float(-np.sum(p * np.log(np.clip(p, 1e-300, None))))

        # 7. Record diagnostics
        self._history.append(MaxCalStepRecord(
            step=self._step,
            t=self._t,
            rover_x=x,
            rover_y=y,
            waypoint_x=float(waypoint[0]),
            waypoint_y=float(waypoint[1]),
            bloom_obs=float(bloom_obs),
            gradient_mag=float(grad_mag),
            v_cmd=float(v_cmd),
            omega_cmd=float(omega_cmd),
            entropy_nats=entropy,
            hs_distance=hs_dist,
            lagrange_multipliers=self._lambdas.tolist(),
            landauer_bound_nJ=landauer_nJ,
            arrived=arrived,
        ))

        self._step += 1
        self._t += dt
        return float(v_cmd), float(omega_cmd)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def current_waypoint(self) -> Optional[np.ndarray]:
        """Return the most recently selected target waypoint, or None."""
        return self._waypoint

    def last_record(self) -> Optional[MaxCalStepRecord]:
        """Return the most recent diagnostic record, or None."""
        return self._history[-1] if self._history else None

    def history(self) -> List[MaxCalStepRecord]:
        """Return all recorded step diagnostics."""
        return list(self._history)

    def kernel_evolution(self) -> np.ndarray:
        """Return (N,) array of HS distances across all recorded steps."""
        return np.array([r.hs_distance for r in self._history])

    def entropy_evolution(self) -> np.ndarray:
        """Return (N,) array of MaxCal distribution entropies (nats)."""
        return np.array([r.entropy_nats for r in self._history])

    def bloom_trace(self) -> np.ndarray:
        """Return (N,) array of observed bloom concentrations over time."""
        return np.array([r.bloom_obs for r in self._history])

    def lagrange_trace(self) -> np.ndarray:
        """Return (N, 3) array of Lagrange multipliers [λ_bloom, λ_grad, λ_dist]."""
        return np.array([r.lagrange_multipliers for r in self._history])

    def statistics(self) -> Dict:
        """Summary diagnostics dictionary."""
        if not self._history:
            return {}
        bt = self.bloom_trace()
        ke = self.kernel_evolution()
        ee = self.entropy_evolution()
        return {
            "mode": self.cfg.mode,
            "steps": self._step,
            "kernelcal_available": _KERNELCAL_AVAILABLE,
            "mean_bloom_obs": float(np.mean(bt)),
            "max_bloom_obs": float(np.max(bt)),
            "mean_hs_distance": float(np.mean(ke)),
            "max_hs_distance": float(np.max(ke)),
            "mean_entropy_nats": float(np.mean(ee)),
            "total_landauer_nJ": float(
                sum(r.landauer_bound_nJ for r in self._history)
            ),
            "lagrange_mean": self.lagrange_trace().mean(axis=0).tolist(),
            "bloom_ema_final": float(self._bloom_ema),
        }

    def __repr__(self) -> str:
        return (
            f"MaxCalBloomFollower("
            f"steps={self._step}, "
            f"kernelcal={'yes' if _KERNELCAL_AVAILABLE else 'fallback'}, "
            f"waypoint={self._waypoint})"
        )


# ---------------------------------------------------------------------------
# Correct MaxCal dual solver
# ---------------------------------------------------------------------------

def _correct_maxcal(
    log_q: np.ndarray,
    F_mat: np.ndarray,
    F_target: np.ndarray,
    lambda0: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the correct MaxCal distribution by minimising the dual

        D(λ) = log Z(λ) + λ · F_target
        Z(λ) = Σ_i q_i exp(−λ · f_i)

    so that ⟨f_i⟩_{p*} = F_target_i at the minimum.

    Gradient (used by scipy.minimize):

        ∂D/∂λ_k = F_target_k − ⟨f_k⟩_{p(λ)}

    Parameters
    ----------
    log_q    : (N,) log-reference weights (need not be normalised)
    F_mat    : (N, C) feature matrix  [f_1(x_i), …, f_C(x_i)]
    F_target : (C,) constraint targets
    lambda0  : (C,) initial Lagrange multipliers (warm start)

    Returns
    -------
    log_p   : (N,) normalised log-probabilities of MaxCal distribution
    lambdas : (C,) optimal Lagrange multipliers
    """
    from scipy.optimize import minimize as _sp_minimize

    C = F_mat.shape[1]
    lam0 = np.zeros(C) if lambda0 is None or len(lambda0) != C else lambda0.copy()

    def dual(lam: np.ndarray) -> float:
        log_unnorm = log_q - F_mat @ lam
        return float(logsumexp(log_unnorm) + lam @ F_target)

    def dual_grad(lam: np.ndarray) -> np.ndarray:
        log_unnorm = log_q - F_mat @ lam
        log_p = log_unnorm - logsumexp(log_unnorm)
        p = np.exp(log_p)
        return F_target - F_mat.T @ p  # ← correct: F − ⟨f⟩_p

    result = _sp_minimize(
        dual, lam0, jac=dual_grad,
        method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8},
    )
    lam_opt = result.x
    log_unnorm = log_q - F_mat @ lam_opt
    log_p = log_unnorm - logsumexp(log_unnorm)
    return log_p, lam_opt


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _wrap_angle(a: float) -> float:
    return float((a + math.pi) % (2.0 * math.pi) - math.pi)
