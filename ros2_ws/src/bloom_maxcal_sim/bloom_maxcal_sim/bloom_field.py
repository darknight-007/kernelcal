"""
Spatiotemporally varying advecting Gaussian algae bloom field.

Mathematical model
------------------
The bloom concentration field  b : Ω × ℝ≥0 → [0, ∞)  is a superposition of
N anisotropic Gaussian patches that advect under a double-gyre stream function —
a canonical benchmark for chaotic transport in oceanography.

Patch field
~~~~~~~~~~~
    b(x, y, t) = Σ_{i=1}^{N}  A_i(t) · G_i(x, y, t)

where each G_i is a rotated anisotropic Gaussian:

    G_i(x, y, t) = exp(-Q_i(x − μx_i(t), y − μy_i(t)))

    Q_i(δx, δy) = (δx cos θ_i + δy sin θ_i)² / (2 σ_{x,i}(t)²)
                + (−δx sin θ_i + δy cos θ_i)² / (2 σ_{y,i}(t)²)

Double-gyre advection field  (Shadden et al., 2005)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
On the normalised domain [0,1]×[0,1], the stream function is:

    ψ(x̃, ỹ, t) = A · sin(π g(x̃, t)) · sin(π ỹ)

    g(x̃, t)    = ε sin(ω t) x̃² + (1 − 2ε sin(ω t)) x̃

Velocity components in physical units (m, m/s):

    u(x, y, t) = −(π A / Ly)  sin(π g(x/Lx, t))  cos(π y/Ly)
    v(x, y, t) = +(π A / Lx)  cos(π g(x/Lx, t))  sin(π y/Ly)
                              · g'(x/Lx, t)

where  g'(x̃, t) = 2ε sin(ωt) x̃ + (1 − 2ε sin(ωt)).

Patch dynamics  (Euler-Maruyama, step dt)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    μx_i ← μx_i + u(μx_i, μy_i, t) Δt + σ_w √Δt ξ_x
    μy_i ← μy_i + v(μx_i, μy_i, t) Δt + σ_w √Δt ξ_y
    (reflecting boundaries on Ω = [0, Lx] × [0, Ly])

Logistic amplitude dynamics:

    A_i(t + Δt) = A_i(t) + r_i A_i(t)(1 − A_i(t)/K_i) Δt

    r_i  — intrinsic growth rate (s⁻¹)
    K_i  — carrying capacity (peak concentration, dimensionless or mg m⁻³)

Slow diffusive broadening of each patch:

    σ_{x,i} ← (σ_{x,i}² + 2 D_i Δt)^{1/2}
    σ_{y,i} ← (σ_{y,i}² + 2 D_i Δt)^{1/2}

References
----------
Shadden, S. C., Lekien, F. & Marsden, J. E. (2005). Definition and properties of
Lagrangian coherent structures from finite-time Lyapunov exponents in two-dimensional
aperiodic flows. Physica D, 212, 271–304.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BloomPatch:
    """Single anisotropic Gaussian algal patch."""
    mu_x: float        # centre x (m)
    mu_y: float        # centre y (m)
    sigma_x: float     # semi-axis length along principal direction (m)
    sigma_y: float     # semi-axis length along minor direction (m)
    theta: float       # orientation of principal axis (rad, CCW from +x)
    amplitude: float   # peak concentration (e.g. mg chl m⁻³, or dimensionless ∈ [0,1])
    growth_rate: float # r_i (s⁻¹)  — logistic intrinsic growth
    carrying_cap: float  # K_i — maximum amplitude
    diffusivity: float   # D_i (m² s⁻¹) — Fickian patch spreading


@dataclass
class DoubleGyreParams:
    """Double-gyre current parameters."""
    A: float = 5.0           # stream-function amplitude → u_max ≈ π·A/Ly ≈ 0.16 m/s
    eps: float = 0.30        # gyre oscillation amplitude (dimensionless, ε)
    omega: float = 2 * np.pi / 25.0   # oscillation angular frequency — 25-second period
    Lx: float = 100.0        # domain width (m)
    Ly: float = 100.0        # domain height (m)


@dataclass
class BloomFieldConfig:
    """Top-level configuration for the bloom simulation."""
    gyre: DoubleGyreParams = field(default_factory=DoubleGyreParams)
    sigma_wind: float = 0.25     # turbulent noise amplitude (m s⁻¹ / sqrt(s)) — adds stochastic drift
    # Grid resolution for rasterised field evaluation
    nx: int = 120
    ny: int = 120


# ---------------------------------------------------------------------------
# Default bloom patches  (three biologically distinct patches)
# ---------------------------------------------------------------------------

def _default_patches(Lx: float = 100.0, Ly: float = 100.0) -> List[BloomPatch]:
    """Three representative algal patches in the gyre domain.

    Timescales are tuned so that all three processes — advection, growth, and
    diffusive spreading — produce noticeable change within ~30 simulation seconds:

        advection  : u_max ≈ 0.16 m/s  →  ~5 m shift in 30 s (≈ ½ patch width)
        growth     : r ≈ 0.015 s⁻¹     →  ~10–20 % amplitude change in 30 s
        diffusion  : D ≈ 1.2 m²/s      →  σ grows 10 → 13 m (30 %) in 30 s
    """
    return [
        BloomPatch(
            mu_x=0.30 * Lx, mu_y=0.55 * Ly,
            sigma_x=10.0, sigma_y=7.0,
            theta=np.pi / 6,
            amplitude=0.75,
            growth_rate=0.015,      # was 2e-4  → 75× faster
            carrying_cap=1.0,
            diffusivity=1.2,        # was 0.025 → 48× faster spread
        ),
        BloomPatch(
            mu_x=0.68 * Lx, mu_y=0.45 * Ly,
            sigma_x=8.0, sigma_y=12.0,
            theta=-np.pi / 5,
            amplitude=0.55,
            growth_rate=0.012,
            carrying_cap=0.90,
            diffusivity=0.90,
        ),
        BloomPatch(
            mu_x=0.50 * Lx, mu_y=0.25 * Ly,
            sigma_x=7.5, sigma_y=7.5,
            theta=0.0,
            amplitude=0.40,
            growth_rate=0.020,      # fastest grower — most responsive
            carrying_cap=0.85,
            diffusivity=0.70,
        ),
    ]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class AlgaeBloomField:
    """Spatiotemporally varying advecting Gaussian algae bloom.

    Parameters
    ----------
    config : BloomFieldConfig
        Global simulation configuration.
    patches : list of BloomPatch, optional
        Initial patch state.  Defaults to three representative patches.
    seed : int
        RNG seed for turbulent noise.

    Usage
    -----
    >>> bloom = AlgaeBloomField()
    >>> c = bloom.concentration_at(50.0, 50.0)  # sample at a point
    >>> bloom.step(dt=1.0)                       # advance 1 second
    >>> grid = bloom.field_grid()                # (ny, nx) concentration array
    """

    def __init__(
        self,
        config: Optional[BloomFieldConfig] = None,
        patches: Optional[List[BloomPatch]] = None,
        seed: int = 42,
    ) -> None:
        self.cfg = config or BloomFieldConfig()
        self.rng = np.random.default_rng(seed)
        self.t: float = 0.0

        Lx, Ly = self.cfg.gyre.Lx, self.cfg.gyre.Ly
        self.patches: List[BloomPatch] = (
            patches if patches is not None else _default_patches(Lx, Ly)
        )

        # Pre-compute evaluation grid
        self.xs = np.linspace(0.0, Lx, self.cfg.nx)
        self.ys = np.linspace(0.0, Ly, self.cfg.ny)
        self.XX, self.YY = np.meshgrid(self.xs, self.ys)  # each (ny, nx)

    # ------------------------------------------------------------------
    # Double-gyre velocity field
    # ------------------------------------------------------------------

    def _g(self, x_tilde: np.ndarray) -> np.ndarray:
        p = self.cfg.gyre
        return p.eps * np.sin(p.omega * self.t) * x_tilde ** 2 + (
            1.0 - 2.0 * p.eps * np.sin(p.omega * self.t)
        ) * x_tilde

    def _g_prime(self, x_tilde: np.ndarray) -> np.ndarray:
        p = self.cfg.gyre
        return 2.0 * p.eps * np.sin(p.omega * self.t) * x_tilde + (
            1.0 - 2.0 * p.eps * np.sin(p.omega * self.t)
        )

    def advection_velocity(self, x: float, y: float) -> Tuple[float, float]:
        """Double-gyre velocity (u, v) in m/s at physical position (x, y)."""
        p = self.cfg.gyre
        x_t = x / p.Lx
        y_t = y / p.Ly
        g = float(self._g(np.array([x_t]))[0])
        gp = float(self._g_prime(np.array([x_t]))[0])
        u = -(np.pi * p.A / p.Ly) * np.sin(np.pi * g) * np.cos(np.pi * y_t)
        v = +(np.pi * p.A / p.Lx) * np.cos(np.pi * g) * np.sin(np.pi * y_t) * gp
        return float(u), float(v)

    def advection_field_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """Velocity field (U, V) on the evaluation grid, each of shape (ny, nx)."""
        p = self.cfg.gyre
        X_t = self.XX / p.Lx
        Y_t = self.YY / p.Ly
        G = self._g(X_t)
        Gp = self._g_prime(X_t)
        U = -(np.pi * p.A / p.Ly) * np.sin(np.pi * G) * np.cos(np.pi * Y_t)
        V = +(np.pi * p.A / p.Lx) * np.cos(np.pi * G) * np.sin(np.pi * Y_t) * Gp
        return U, V

    # ------------------------------------------------------------------
    # Gaussian patch helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _patch_gaussian(
        XX: np.ndarray, YY: np.ndarray,
        mu_x: float, mu_y: float,
        sigma_x: float, sigma_y: float,
        theta: float,
    ) -> np.ndarray:
        """Rotated anisotropic Gaussian evaluated on a grid."""
        dx = XX - mu_x
        dy = YY - mu_y
        ct, st = np.cos(theta), np.sin(theta)
        dx_r = ct * dx + st * dy
        dy_r = -st * dx + ct * dy
        return np.exp(-(dx_r ** 2 / (2.0 * sigma_x ** 2) + dy_r ** 2 / (2.0 * sigma_y ** 2)))

    @staticmethod
    def _patch_at_point(
        x: float, y: float,
        mu_x: float, mu_y: float,
        sigma_x: float, sigma_y: float,
        theta: float,
        amplitude: float,
    ) -> float:
        dx, dy = x - mu_x, y - mu_y
        ct, st = np.cos(theta), np.sin(theta)
        dx_r = ct * dx + st * dy
        dy_r = -st * dx + ct * dy
        g = np.exp(-(dx_r ** 2 / (2.0 * sigma_x ** 2) + dy_r ** 2 / (2.0 * sigma_y ** 2)))
        return float(amplitude * g)

    # ------------------------------------------------------------------
    # Field queries
    # ------------------------------------------------------------------

    def concentration_at(self, x: float, y: float) -> float:
        """Sample total bloom concentration at a single point (m)."""
        total = sum(
            self._patch_at_point(x, y, p.mu_x, p.mu_y, p.sigma_x, p.sigma_y, p.theta, p.amplitude)
            for p in self.patches
        )
        return float(np.clip(total, 0.0, None))

    def gradient_at(self, x: float, y: float) -> Tuple[float, float]:
        """Analytical spatial gradient ∇b = (∂b/∂x, ∂b/∂y) at (x, y)."""
        gx = gy = 0.0
        for p in self.patches:
            dx, dy = x - p.mu_x, y - p.mu_y
            ct, st = np.cos(p.theta), np.sin(p.theta)
            dx_r = ct * dx + st * dy
            dy_r = -st * dx + ct * dy
            g = np.exp(
                -(dx_r ** 2 / (2.0 * p.sigma_x ** 2) + dy_r ** 2 / (2.0 * p.sigma_y ** 2))
            )
            # chain rule: ∂G/∂x = G · (∂Q/∂x) with sign flip
            dq_dx_r = dx_r / p.sigma_x ** 2
            dq_dy_r = dy_r / p.sigma_y ** 2
            # rotate back: ∂Q/∂x = (∂Q/∂dx_r)(∂dx_r/∂x) + (∂Q/∂dy_r)(∂dy_r/∂x)
            dg_dx = -g * (dq_dx_r * ct + dq_dy_r * (-st))
            dg_dy = -g * (dq_dx_r * st + dq_dy_r * ct)
            gx += p.amplitude * dg_dx
            gy += p.amplitude * dg_dy
        return float(gx), float(gy)

    def gradient_magnitude_at(self, x: float, y: float) -> float:
        """||∇b|| at (x, y)."""
        gx, gy = self.gradient_at(x, y)
        return float(np.hypot(gx, gy))

    def field_grid(self) -> np.ndarray:
        """Full concentration field on the evaluation grid, shape (ny, nx)."""
        grid = np.zeros_like(self.XX)
        for p in self.patches:
            grid += p.amplitude * self._patch_gaussian(
                self.XX, self.YY, p.mu_x, p.mu_y, p.sigma_x, p.sigma_y, p.theta
            )
        return np.clip(grid, 0.0, None)

    def gradient_magnitude_grid(self) -> np.ndarray:
        """||∇b|| on the evaluation grid, shape (ny, nx)."""
        # Compute via finite differences on the field grid for speed
        b = self.field_grid()
        dx = self.xs[1] - self.xs[0]
        dy = self.ys[1] - self.ys[0]
        gbx = np.gradient(b, dx, axis=1)
        gby = np.gradient(b, dy, axis=0)
        return np.hypot(gbx, gby)

    # ------------------------------------------------------------------
    # Time integration
    # ------------------------------------------------------------------

    def step(self, dt: float) -> None:
        """Advance the bloom field by *dt* seconds (Euler-Maruyama)."""
        p_cfg = self.cfg.gyre
        sw = self.cfg.sigma_wind
        for p in self.patches:
            # Advect patch centre
            u, v = self.advection_velocity(p.mu_x, p.mu_y)
            noise_x = sw * self.rng.standard_normal() * np.sqrt(dt)
            noise_y = sw * self.rng.standard_normal() * np.sqrt(dt)
            p.mu_x += u * dt + noise_x
            p.mu_y += v * dt + noise_y
            # Reflecting boundary
            p.mu_x = float(np.clip(p.mu_x, 0.0, p_cfg.Lx))
            p.mu_y = float(np.clip(p.mu_y, 0.0, p_cfg.Ly))

            # Logistic amplitude growth
            r, K, A = p.growth_rate, p.carrying_cap, p.amplitude
            p.amplitude = float(np.clip(A + r * A * (1.0 - A / K) * dt, 0.01, K * 1.1))

            # Fickian broadening: σ² → σ² + 2D·dt
            p.sigma_x = float(np.sqrt(max(p.sigma_x ** 2 + 2.0 * p.diffusivity * dt, 1.0)))
            p.sigma_y = float(np.sqrt(max(p.sigma_y ** 2 + 2.0 * p.diffusivity * dt, 1.0)))

        self.t += dt

    # ------------------------------------------------------------------
    # Derived statistics
    # ------------------------------------------------------------------

    def bloom_center_of_mass(self) -> Tuple[float, float]:
        """Concentration-weighted centre of mass of the full bloom field."""
        g = self.field_grid()
        total = g.sum() + 1e-12
        cx = float((self.XX * g).sum() / total)
        cy = float((self.YY * g).sum() / total)
        return cx, cy

    def peak_location(self) -> Tuple[float, float]:
        """(x, y) of the maximum concentration grid point."""
        g = self.field_grid()
        iy, ix = np.unravel_index(np.argmax(g), g.shape)
        return float(self.xs[ix]), float(self.ys[iy])

    def bloom_front_points(self, threshold: float = 0.25) -> np.ndarray:
        """(x, y) grid points on the high-gradient bloom front above *threshold*.

        Returns array of shape (M, 2).
        """
        g = self.field_grid()
        gmag = self.gradient_magnitude_grid()
        mask = (g > threshold) & (gmag > 0.4 * gmag.max())
        return np.column_stack([self.XX[mask], self.YY[mask]])

    def domain(self) -> Tuple[float, float, float, float]:
        """Return (x_min, x_max, y_min, y_max) of the domain."""
        p = self.cfg.gyre
        return 0.0, p.Lx, 0.0, p.Ly

    def __repr__(self) -> str:
        return (
            f"AlgaeBloomField(t={self.t:.1f}s, "
            f"n_patches={len(self.patches)}, "
            f"domain=[0,{self.cfg.gyre.Lx}]×[0,{self.cfg.gyre.Ly}])"
        )
