"""
2D differential-drive rover kinematics with velocity control.

State
-----
    s = (x, y, θ)   — position (m) and heading (rad, CCW from +x axis)

Control inputs
--------------
    u = (v, ω)       — linear velocity (m/s) and angular velocity (rad/s)

Kinematics (unicycle model, Euler integration)
----------------------------------------------
    ẋ = v cos θ
    ẏ = v sin θ
    θ̇ = ω

Constraints
-----------
    |v|   ≤ v_max       (forward and reverse)
    |ω|   ≤ omega_max   (turning rate)
    a_v   = (v - v_prev) / dt ≤ a_max     (acceleration limit)
    a_ω   = (ω - ω_prev) / dt ≤ alpha_max (angular acceleration)

Sensor model
------------
The rover reports a noisy observation of the bloom concentration:

    z_t = b(x_t, y_t) + ε_t,   ε_t ~ N(0, σ_obs²)

and the bloom gradient direction:

    ψ_t = atan2(∂b/∂y, ∂b/∂x) + δ_t,   δ_t ~ N(0, σ_dir²)

These are used by the MaxCal controller as the information signal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RoverConfig:
    """Physical and sensing parameters of the simulated rover."""
    # Kinematics limits
    v_max: float = 1.5          # m/s — maximum linear speed
    omega_max: float = 1.2      # rad/s — maximum angular rate
    a_max: float = 0.8          # m/s² — linear acceleration limit
    alpha_max: float = 1.5      # rad/s² — angular acceleration limit
    # Wheel-base (used only for display, kinematics use unicycle model)
    wheelbase: float = 0.5      # m
    # Observation noise
    sigma_obs: float = 0.02     # concentration observation noise (same units as bloom)
    sigma_dir: float = 0.15     # gradient direction noise (rad)
    # Initial state
    x0: float = 50.0            # m
    y0: float = 50.0            # m
    theta0: float = 0.0         # rad


@dataclass
class RoverState:
    """Full state of the simulated rover at one time step."""
    x: float
    y: float
    theta: float
    v: float      # current linear velocity (m/s)
    omega: float  # current angular velocity (rad/s)
    t: float      # simulation time (s)


# ---------------------------------------------------------------------------
# Rover
# ---------------------------------------------------------------------------

class DifferentialDriveRover:
    """2D differential-drive rover with unicycle kinematics.

    Parameters
    ----------
    config : RoverConfig
    seed : int
        RNG seed for observation noise.

    Usage
    -----
    >>> rover = DifferentialDriveRover()
    >>> rover.step(v=1.0, omega=0.1, dt=0.1)
    >>> x, y, theta = rover.pose()
    """

    def __init__(
        self,
        config: Optional[RoverConfig] = None,
        seed: int = 123,
    ) -> None:
        self.cfg = config or RoverConfig()
        self.rng = np.random.default_rng(seed)

        self.x: float = self.cfg.x0
        self.y: float = self.cfg.y0
        self.theta: float = self.cfg.theta0
        self.v: float = 0.0
        self.omega: float = 0.0
        self.t: float = 0.0

        self._history: list[RoverState] = []
        self._record()

    # ------------------------------------------------------------------
    # Kinematics
    # ------------------------------------------------------------------

    def step(self, v_cmd: float, omega_cmd: float, dt: float) -> None:
        """Advance rover state by *dt* seconds given commanded (v, ω).

        Applies acceleration limits, then integrates unicycle kinematics.

        Parameters
        ----------
        v_cmd : float
            Desired linear velocity (m/s).  Clamped to [-v_max, v_max].
        omega_cmd : float
            Desired angular velocity (rad/s).  Clamped to [-omega_max, omega_max].
        dt : float
            Time step (s).
        """
        cfg = self.cfg
        # Acceleration-limit smoothing
        dv_max = cfg.a_max * dt
        dw_max = cfg.alpha_max * dt
        v_cmd = float(np.clip(v_cmd, -cfg.v_max, cfg.v_max))
        omega_cmd = float(np.clip(omega_cmd, -cfg.omega_max, cfg.omega_max))

        v_new = float(np.clip(v_cmd, self.v - dv_max, self.v + dv_max))
        w_new = float(np.clip(omega_cmd, self.omega - dw_max, self.omega + dw_max))
        v_new = float(np.clip(v_new, -cfg.v_max, cfg.v_max))
        w_new = float(np.clip(w_new, -cfg.omega_max, cfg.omega_max))

        # Unicycle integration
        self.x += v_new * math.cos(self.theta) * dt
        self.y += v_new * math.sin(self.theta) * dt
        self.theta = _wrap_angle(self.theta + w_new * dt)
        self.v = v_new
        self.omega = w_new
        self.t += dt

        self._record()

    # ------------------------------------------------------------------
    # Sensing
    # ------------------------------------------------------------------

    def observe_bloom(self, bloom_field) -> float:
        """Return noisy bloom concentration at current position.

        Parameters
        ----------
        bloom_field : AlgaeBloomField
        """
        c_true = bloom_field.concentration_at(self.x, self.y)
        noise = self.rng.normal(0.0, self.cfg.sigma_obs)
        return float(np.clip(c_true + noise, 0.0, None))

    def observe_gradient_direction(self, bloom_field) -> float:
        """Return noisy angle of ∇b at current position (rad, CCW from +x)."""
        gx, gy = bloom_field.gradient_at(self.x, self.y)
        direction = math.atan2(gy, gx)
        noise = self.rng.normal(0.0, self.cfg.sigma_dir)
        return _wrap_angle(direction + noise)

    def observe_gradient_magnitude(self, bloom_field) -> float:
        """Return noisy gradient magnitude ||∇b|| at current position."""
        gx, gy = bloom_field.gradient_at(self.x, self.y)
        return float(np.hypot(gx, gy))  # no noise — used only as feature

    # ------------------------------------------------------------------
    # Pose accessors
    # ------------------------------------------------------------------

    def pose(self) -> Tuple[float, float, float]:
        """Return (x, y, θ) current pose."""
        return self.x, self.y, self.theta

    def position(self) -> np.ndarray:
        """Return np.array([x, y])."""
        return np.array([self.x, self.y])

    def velocity(self) -> Tuple[float, float]:
        """Return (v, ω) current velocity commands."""
        return self.v, self.omega

    def state(self) -> RoverState:
        """Full current state."""
        return RoverState(
            x=self.x, y=self.y, theta=self.theta,
            v=self.v, omega=self.omega, t=self.t,
        )

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def _record(self) -> None:
        self._history.append(self.state())

    def trajectory(self) -> np.ndarray:
        """Return (N, 2) array of (x, y) positions over time."""
        return np.array([[s.x, s.y] for s in self._history])

    def reset(self) -> None:
        """Reset rover to initial configuration."""
        self.x = self.cfg.x0
        self.y = self.cfg.y0
        self.theta = self.cfg.theta0
        self.v = 0.0
        self.omega = 0.0
        self.t = 0.0
        self._history.clear()
        self._record()

    def __repr__(self) -> str:
        return (
            f"DifferentialDriveRover("
            f"x={self.x:.2f}, y={self.y:.2f}, θ={math.degrees(self.theta):.1f}°, "
            f"v={self.v:.2f} m/s, ω={self.omega:.3f} rad/s)"
        )


# ---------------------------------------------------------------------------
# Velocity controllers
# ---------------------------------------------------------------------------

class WaypointVelocityController:
    """Proportional waypoint follower for the 2D rover.

    Given a target waypoint, produces (v, ω) commands using a simple
    proportional law that blends heading correction and forward motion.

    Control law
    -----------
    Heading error:  Δθ = wrap(atan2(dy, dx) − θ)
    Angular rate:   ω  = k_ω · Δθ
    Linear speed:   v  = v_max · σ(Δθ)  (reduces speed for large errors)

    where σ(Δθ) = exp(−k_v_theta · Δθ²) is a heading-aligned gate.

    Parameters
    ----------
    v_max : float — maximum forward speed (m/s)
    k_omega : float — proportional gain for angular velocity
    k_v_theta : float — sharpness of speed reduction with heading error
    arrival_radius : float — distance (m) at which waypoint is declared reached
    """

    def __init__(
        self,
        v_max: float = 1.2,
        k_omega: float = 1.8,
        k_v_theta: float = 2.5,
        arrival_radius: float = 2.0,
    ) -> None:
        self.v_max = v_max
        self.k_omega = k_omega
        self.k_v_theta = k_v_theta
        self.arrival_radius = arrival_radius

    def compute(
        self,
        rover: DifferentialDriveRover,
        waypoint: np.ndarray,
    ) -> Tuple[float, float, bool]:
        """Compute (v, ω) command toward *waypoint*.

        Returns
        -------
        v : float — linear velocity command (m/s)
        omega : float — angular velocity command (rad/s)
        arrived : bool — True when within arrival_radius of waypoint
        """
        dx = waypoint[0] - rover.x
        dy = waypoint[1] - rover.y
        dist = math.hypot(dx, dy)

        if dist < self.arrival_radius:
            return 0.0, 0.0, True

        desired_heading = math.atan2(dy, dx)
        heading_error = _wrap_angle(desired_heading - rover.theta)

        omega = float(np.clip(
            self.k_omega * heading_error,
            -rover.cfg.omega_max,
            rover.cfg.omega_max,
        ))
        speed_gate = math.exp(-self.k_v_theta * heading_error ** 2)
        v = float(np.clip(self.v_max * speed_gate, 0.0, rover.cfg.v_max))

        # Slow down when close to waypoint
        if dist < 5.0:
            v *= dist / 5.0

        return v, omega, False


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _wrap_angle(a: float) -> float:
    """Wrap angle to (−π, π]."""
    return float((a + math.pi) % (2.0 * math.pi) - math.pi)
