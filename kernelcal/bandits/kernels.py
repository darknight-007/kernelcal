"""
Kernel classes for the spatiotemporal DDK-GPUCB experiment.

Two structurally distinct kernel classes are defined:

  Class A — AnisotropicSEKernel
    k(z, z') = σ² exp(-(x-x')²/2ℓ_x² - (t-t')²/2ℓ_t²)
    "Same lens": treats both spatial and temporal axes with SE covariance.
    Cannot represent periodicity regardless of hyperparameter tuning.

  Class B — SEPeriodicKernel
    k(z, z') = σ² exp(-(x-x')²/2ℓ_x²) × exp(-2 sin²(π(t-t')/p) / ℓ_p²)
    "Different lens": product of SE (spatial) and Periodic (temporal).
    Structurally capable of capturing temporal periodicity; SE alone is not.

  MixtureKernel
    k = (1-w) * k_SE  +  w * k_SEPer
    Smooth interpolation between classes.  The mixing weight w ∈ [0,1]
    is learned from data via gradient on log-marginal-likelihood.
    Agents in periodic regions should converge to w → 1;
    agents in aperiodic regions should stay at w → 0.

This distinction is the core experimental claim:
  - Adjusting ℓ_x, ℓ_t within SE is "refocusing the same lens."
  - Switching to SE×Periodic is "a different product kernel" that
    captures structure the SE family cannot represent at any lengthscale.

The kernel is parameterised by log-hyperparameters for unconstrained
gradient optimisation:

    theta = [log(ell_x), log(ell_y), log(sigma_f), log(sigma_n)]

This ensures positivity without projection during gradient steps.

Functions
---------
AnisotropicSEKernel   -- kernel matrix + log-marginal-likelihood + gradient
hs_distance           -- Hilbert-Schmidt distance between two kernel matrices
kernel_consensus      -- Gossip-weighted average of hyperparameter vectors
fisher_rao_distance   -- Approximate Fisher-Rao distance via HS proxy
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Optional, Tuple

import numpy as np

from ..kernel.space import hilbert_schmidt_distance


# ---------------------------------------------------------------------------
# Kernel matrix utilities
# ---------------------------------------------------------------------------

def _aniso_se_K(
    X: np.ndarray,
    X2: Optional[np.ndarray],
    ell_x: float,
    ell_y: float,
    sigma_f: float,
    sigma_n: float,
    add_noise: bool = True,
) -> np.ndarray:
    """Compute anisotropic SE covariance matrix.

    k(x, x') = sigma_f^2 * exp(-0.5 * [(x1-x1')^2/ell_x^2 + (x2-x2')^2/ell_y^2])

    Parameters
    ----------
    X   : (n, 2) training inputs
    X2  : (m, 2) test inputs, or None to use X
    add_noise : if True and X2 is None, add sigma_n^2 * I to the diagonal
    """
    if X2 is None:
        X2 = X
        same = True
    else:
        same = False

    diff1 = X[:, 0:1] - X2[:, 0:1].T   # (n, m)
    diff2 = X[:, 1:2] - X2[:, 1:2].T   # (n, m)
    K = sigma_f ** 2 * np.exp(
        -0.5 * (diff1 ** 2 / ell_x ** 2 + diff2 ** 2 / ell_y ** 2)
    )
    if same and add_noise:
        K += sigma_n ** 2 * np.eye(len(X))
    return K


# ---------------------------------------------------------------------------
# Kernel class
# ---------------------------------------------------------------------------

@dataclass
class AnisotropicSEKernel:
    """Anisotropic squared-exponential kernel with axis-aligned lengthscales.

    Hyperparameters stored in log-space for unconstrained optimisation.

    Parameters
    ----------
    log_ell_x, log_ell_y : float  -- log lengthscales along x and y axes
    log_sigma_f          : float  -- log signal amplitude
    log_sigma_n          : float  -- log noise standard deviation
    """

    log_ell_x: float = np.log(0.2)
    log_ell_y: float = np.log(0.2)
    log_sigma_f: float = np.log(1.0)
    log_sigma_n: float = np.log(0.1)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def ell_x(self) -> float:
        return float(np.exp(self.log_ell_x))

    @property
    def ell_y(self) -> float:
        return float(np.exp(self.log_ell_y))

    @property
    def sigma_f(self) -> float:
        return float(np.exp(self.log_sigma_f))

    @property
    def sigma_n(self) -> float:
        return float(np.exp(self.log_sigma_n))

    @property
    def theta(self) -> np.ndarray:
        """Hyperparameter vector [log_ell_x, log_ell_y, log_sigma_f, log_sigma_n]."""
        return np.array([
            self.log_ell_x, self.log_ell_y, self.log_sigma_f, self.log_sigma_n
        ])

    @theta.setter
    def theta(self, value: np.ndarray) -> None:
        self.log_ell_x, self.log_ell_y, self.log_sigma_f, self.log_sigma_n = value

    # ------------------------------------------------------------------
    # Kernel matrix
    # ------------------------------------------------------------------

    def K(self, X: np.ndarray, X2: Optional[np.ndarray] = None,
          add_noise: bool = True) -> np.ndarray:
        return _aniso_se_K(X, X2, self.ell_x, self.ell_y,
                           self.sigma_f, self.sigma_n, add_noise=add_noise)

    # ------------------------------------------------------------------
    # GP posterior
    # ------------------------------------------------------------------

    def posterior(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return GP posterior mean and standard deviation at X_test.

        Returns
        -------
        mu  : (m,) posterior mean
        std : (m,) posterior standard deviation
        """
        K_train = self.K(X_train, add_noise=True)
        K_star  = self.K(X_train, X_test, add_noise=False)   # (n, m)
        K_ss    = self.K(X_test, add_noise=False)             # (m, m)

        K_train += 1e-8 * np.eye(len(X_train))  # numerical jitter
        try:
            L = np.linalg.cholesky(K_train)
        except np.linalg.LinAlgError:
            K_train += 1e-4 * np.eye(len(X_train))
            L = np.linalg.cholesky(K_train)

        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        mu    = K_star.T @ alpha

        v     = np.linalg.solve(L, K_star)
        var   = np.diag(K_ss) - np.sum(v ** 2, axis=0)
        std   = np.sqrt(np.maximum(var, 1e-12))
        return mu, std

    # ------------------------------------------------------------------
    # Log-marginal-likelihood and gradient
    # ------------------------------------------------------------------

    def log_marginal_likelihood(
        self, X: np.ndarray, y: np.ndarray
    ) -> float:
        """Compute log p(y | X, theta)."""
        K = self.K(X, add_noise=True) + 1e-8 * np.eye(len(X))
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return -1e9
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        lml = (
            -0.5 * y @ alpha
            - np.sum(np.log(np.diag(L)))
            - 0.5 * len(y) * np.log(2 * np.pi)
        )
        return float(lml)

    def lml_gradient(
        self, X: np.ndarray, y: np.ndarray, eps: float = 1e-4
    ) -> np.ndarray:
        """Numerical gradient of log-marginal-likelihood w.r.t. theta.

        Uses central differences.  Each evaluation costs one Cholesky.
        """
        grad = np.zeros(4)
        theta0 = self.theta.copy()
        for i in range(4):
            tp = theta0.copy(); tp[i] += eps; self.theta = tp
            fp = self.log_marginal_likelihood(X, y)
            tm = theta0.copy(); tm[i] -= eps; self.theta = tm
            fm = self.log_marginal_likelihood(X, y)
            grad[i] = (fp - fm) / (2 * eps)
        self.theta = theta0
        return grad

    def adapt(
        self,
        X: np.ndarray,
        y: np.ndarray,
        step: float = 0.05,
        landauer_penalty: float = 0.0,
        clip: float = 3.0,
    ) -> float:
        """One gradient-ascent step on log-marginal-likelihood.

        The Landauer penalty discourages large kernel updates (thermodynamic
        cost of changing the distinction structure):

            theta <- theta + step * (grad_lml - landauer_penalty * theta)

        The penalty pulls hyperparameters toward zero (log-space), i.e.
        toward unit lengthscales and unit amplitude.

        Parameters
        ----------
        step             : learning rate for hyperparameter update
        landauer_penalty : weight on L2 regularisation in log-space
        clip             : max norm of the gradient step (stability)

        Returns
        -------
        lml : log-marginal-likelihood before the step (for monitoring)
        """
        if len(X) < 2:
            return 0.0
        lml = self.log_marginal_likelihood(X, y)
        grad = self.lml_gradient(X, y)
        grad -= landauer_penalty * self.theta  # regularise toward origin
        # clip gradient norm
        gnorm = np.linalg.norm(grad)
        if gnorm > clip:
            grad = grad * clip / gnorm
        self.theta = self.theta + step * grad
        # hard bounds: keep log-hyperparameters in a reasonable range
        self.theta = np.clip(self.theta, -4.0, 2.0)
        return lml

    def copy(self) -> "AnisotropicSEKernel":
        k = AnisotropicSEKernel()
        k.theta = self.theta.copy()
        return k


# ---------------------------------------------------------------------------
# Class B: SE(spatial) × Periodic(temporal) product kernel
# ---------------------------------------------------------------------------

@dataclass
class SEPeriodicKernel:
    """Product kernel: SE along spatial axis × Periodic along temporal axis.

    k((x,t),(x',t')) = σ²
        · exp(-(x-x')² / 2ℓ_x²)          [spatial SE]
        · exp(-2 sin²(π(t-t')/p) / ℓ_p²)  [temporal Periodic]

    This kernel is structurally capable of capturing temporal periodicity.
    The SE kernel (Class A) cannot represent this regardless of its
    lengthscale parameters — it is a different lens, not just a different
    focal length.

    Hyperparameters stored in log-space:
        theta = [log(ℓ_x), log(p), log(ℓ_p), log(σ), log(σ_n)]
    where p is the period and ℓ_p is the periodic lengthscale.
    """

    log_ell_x: float = np.log(0.3)
    log_period: float = np.log(0.5)   # period along t axis (0.5 = 2 full cycles)
    log_ell_p: float = np.log(1.0)    # periodic lengthscale (smoothness of periodic comp)
    log_sigma_f: float = np.log(1.0)
    log_sigma_n: float = np.log(0.1)

    @property
    def ell_x(self) -> float:   return float(np.exp(self.log_ell_x))
    @property
    def period(self) -> float:  return float(np.exp(self.log_period))
    @property
    def ell_p(self) -> float:   return float(np.exp(self.log_ell_p))
    @property
    def sigma_f(self) -> float: return float(np.exp(self.log_sigma_f))
    @property
    def sigma_n(self) -> float: return float(np.exp(self.log_sigma_n))

    @property
    def theta(self) -> np.ndarray:
        return np.array([self.log_ell_x, self.log_period,
                         self.log_ell_p, self.log_sigma_f, self.log_sigma_n])

    @theta.setter
    def theta(self, v: np.ndarray) -> None:
        (self.log_ell_x, self.log_period,
         self.log_ell_p, self.log_sigma_f, self.log_sigma_n) = v

    def K(self, X: np.ndarray, X2: Optional[np.ndarray] = None,
          add_noise: bool = True) -> np.ndarray:
        """X columns: [x_spatial, t_temporal]."""
        if X2 is None:
            X2 = X; same = True
        else:
            same = False

        # Spatial SE component
        dx = X[:, 0:1] - X2[:, 0:1].T
        k_se = np.exp(-0.5 * dx**2 / self.ell_x**2)

        # Temporal Periodic component
        dt = X[:, 1:2] - X2[:, 1:2].T
        k_per = np.exp(-2.0 * np.sin(np.pi * dt / self.period)**2 / self.ell_p**2)

        K = self.sigma_f**2 * k_se * k_per
        if same and add_noise:
            K += self.sigma_n**2 * np.eye(len(X))
        return K

    def posterior(self, X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        K_train = self.K(X_train, add_noise=True) + 1e-8 * np.eye(len(X_train))
        K_star  = self.K(X_train, X_test, add_noise=False)
        K_ss    = self.K(X_test,  add_noise=False)
        try:
            L = np.linalg.cholesky(K_train)
        except np.linalg.LinAlgError:
            K_train += 1e-4 * np.eye(len(X_train))
            L = np.linalg.cholesky(K_train)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        mu    = K_star.T @ alpha
        v     = np.linalg.solve(L, K_star)
        std   = np.sqrt(np.maximum(np.diag(K_ss) - np.sum(v**2, axis=0), 1e-12))
        return mu, std

    def log_marginal_likelihood(self, X: np.ndarray, y: np.ndarray) -> float:
        K = self.K(X, add_noise=True) + 1e-8 * np.eye(len(X))
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return -1e9
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        return float(-0.5 * y @ alpha
                     - np.sum(np.log(np.diag(L)))
                     - 0.5 * len(y) * np.log(2 * np.pi))

    def lml_gradient(self, X: np.ndarray, y: np.ndarray,
                     eps: float = 1e-4) -> np.ndarray:
        grad = np.zeros(5)
        t0 = self.theta.copy()
        for i in range(5):
            tp = t0.copy(); tp[i] += eps; self.theta = tp
            fp = self.log_marginal_likelihood(X, y)
            tm = t0.copy(); tm[i] -= eps; self.theta = tm
            fm = self.log_marginal_likelihood(X, y)
            grad[i] = (fp - fm) / (2 * eps)
        self.theta = t0
        return grad

    def adapt(self, X: np.ndarray, y: np.ndarray,
              step: float = 0.05, landauer_penalty: float = 0.0,
              clip: float = 3.0) -> float:
        if len(X) < 2:
            return 0.0
        lml  = self.log_marginal_likelihood(X, y)
        grad = self.lml_gradient(X, y) - landauer_penalty * self.theta
        gnorm = np.linalg.norm(grad)
        if gnorm > clip:
            grad *= clip / gnorm
        self.theta = np.clip(self.theta + step * grad, -4.0, 2.0)
        return lml

    def copy(self) -> "SEPeriodicKernel":
        k = SEPeriodicKernel(); k.theta = self.theta.copy(); return k


# ---------------------------------------------------------------------------
# Mixture kernel: (1-w)*SE + w*(SE×Periodic)
# ---------------------------------------------------------------------------

@dataclass
class MixtureKernel:
    """Smooth mixture of AnisotropicSEKernel (class A) and SEPeriodicKernel (class B).

    k_mix = (1-w) * k_SE  +  w * k_SEPer,   w ∈ [0,1].

    The mixing weight w is learned from data.  It starts at 0.5 (equal mix)
    and adapts via gradient ascent on the log-marginal-likelihood.

    This is the "dynamic kernel" in the full sense:
    - w ≈ 0  means the agent has concluded its region is aperiodic (SE wins)
    - w ≈ 1  means the agent has detected periodicity (SE×Periodic wins)

    The weight is stored as logit(w) for unconstrained optimisation.
    theta = concat(theta_SE, theta_SEPer, [logit_w])
    """

    kernel_se:  AnisotropicSEKernel = None
    kernel_per: SEPeriodicKernel    = None
    logit_w: float = 0.0            # logit(0.5) = 0 → w = 0.5 initially

    def __post_init__(self):
        if self.kernel_se  is None: self.kernel_se  = AnisotropicSEKernel()
        if self.kernel_per is None: self.kernel_per = SEPeriodicKernel()

    @property
    def w(self) -> float:
        """Mixing weight w = sigmoid(logit_w)."""
        return float(1.0 / (1.0 + np.exp(-self.logit_w)))

    def K(self, X: np.ndarray, X2: Optional[np.ndarray] = None,
          add_noise: bool = True) -> np.ndarray:
        w   = self.w
        Kse  = self.kernel_se.K(X,  X2, add_noise=False)
        Kper = self.kernel_per.K(X, X2, add_noise=False)
        K    = (1.0 - w) * Kse + w * Kper
        if (X2 is None) and add_noise:
            sn = (1 - w) * self.kernel_se.sigma_n + w * self.kernel_per.sigma_n
            K += sn**2 * np.eye(len(X))
        return K

    def posterior(self, X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        K_train = self.K(X_train, add_noise=True) + 1e-8 * np.eye(len(X_train))
        K_star  = self.K(X_train, X_test, add_noise=False)
        K_ss    = self.K(X_test,  add_noise=False)
        try:
            L = np.linalg.cholesky(K_train)
        except np.linalg.LinAlgError:
            K_train += 1e-4 * np.eye(len(X_train))
            L = np.linalg.cholesky(K_train)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        mu    = K_star.T @ alpha
        v     = np.linalg.solve(L, K_star)
        std   = np.sqrt(np.maximum(np.diag(K_ss) - np.sum(v**2, axis=0), 1e-12))
        return mu, std

    def log_marginal_likelihood(self, X: np.ndarray, y: np.ndarray) -> float:
        K = self.K(X, add_noise=True) + 1e-8 * np.eye(len(X))
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return -1e9
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        return float(-0.5 * y @ alpha
                     - np.sum(np.log(np.diag(L)))
                     - 0.5 * len(y) * np.log(2 * np.pi))

    def adapt(self, X: np.ndarray, y: np.ndarray,
              step: float = 0.05, landauer_penalty: float = 0.0,
              clip: float = 3.0) -> float:
        """Jointly adapt all hyperparameters plus the mixing weight."""
        if len(X) < 3:
            return 0.0
        lml0 = self.log_marginal_likelihood(X, y)
        eps  = 1e-4
        # Gradient w.r.t. logit_w
        self.logit_w += eps
        fp = self.log_marginal_likelihood(X, y)
        self.logit_w -= 2 * eps
        fm = self.log_marginal_likelihood(X, y)
        self.logit_w += eps
        dw = (fp - fm) / (2 * eps) - landauer_penalty * self.logit_w
        self.logit_w = float(np.clip(self.logit_w + step * dw, -6.0, 6.0))
        # Also adapt the component hyperparameters
        self.kernel_se.adapt(X,  y, step=step * 0.5,
                             landauer_penalty=landauer_penalty)
        self.kernel_per.adapt(X, y, step=step * 0.5,
                              landauer_penalty=landauer_penalty)
        return lml0

    def copy(self) -> "MixtureKernel":
        m = MixtureKernel(
            kernel_se=self.kernel_se.copy(),
            kernel_per=self.kernel_per.copy(),
            logit_w=self.logit_w,
        )
        return m


# ---------------------------------------------------------------------------
# Distance and consensus utilities
# ---------------------------------------------------------------------------

def hs_distance(k1: AnisotropicSEKernel, k2: AnisotropicSEKernel,
                X_ref: np.ndarray) -> float:
    """Hilbert-Schmidt distance between two kernels evaluated on reference points.

    Builds the noise-free Gram matrices ``K1``, ``K2`` at ``X_ref`` and
    delegates to :func:`kernelcal.kernel.space.hilbert_schmidt_distance`,
    i.e. ``|| K1 - K2 ||_F / sqrt(n)`` with ``n = len(X_ref)``.
    """
    K1 = k1.K(X_ref, add_noise=False)
    K2 = k2.K(X_ref, add_noise=False)
    return hilbert_schmidt_distance(K1, K2)


def fisher_rao_distance(k1: AnisotropicSEKernel,
                        k2: AnisotropicSEKernel) -> float:
    """Approximate Fisher-Rao distance via log-hyperparameter Euclidean distance.

    For the SE kernel family, the Fisher information metric in log-hyperparameter
    space is approximately diagonal, so the geodesic length is approximated by
    the Euclidean distance in log-space.  This is exact only in a neighbourhood
    of the identity but is numerically tractable.
    """
    return float(np.linalg.norm(k1.theta - k2.theta))


def kernel_consensus(
    kernel_i: AnisotropicSEKernel,
    neighbor_kernels: list,
    weights: np.ndarray,
    rho: float = 0.3,
) -> AnisotropicSEKernel:
    """Gossip-weighted consensus step in log-hyperparameter space.

    theta_i <- (1 - rho) * theta_i + rho * sum_j P_ij * theta_j

    This is a Euclidean averaging in log-space.  For SE kernels,
    this corresponds to geometric averaging of the raw hyperparameters.

    Parameters
    ----------
    kernel_i         : current kernel of agent i
    neighbor_kernels : list of kernels from neighbors j
    weights          : gossip weights P_ij for each neighbor
    rho              : consensus strength in [0, 1]
    """
    if not neighbor_kernels:
        return kernel_i.copy()

    neighbor_avg = np.zeros(4)
    for k, w in zip(neighbor_kernels, weights):
        neighbor_avg += w * k.theta
    total_w = np.sum(weights)
    if total_w > 0:
        neighbor_avg /= total_w

    new_theta = (1.0 - rho) * kernel_i.theta + rho * neighbor_avg
    out = kernel_i.copy()
    out.theta = new_theta
    return out
