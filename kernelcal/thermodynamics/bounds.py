"""
Information-thermodynamic bounds: Landauer principle for kernel change.

Maps to Theorem 1 of the paper:

    δW ≥ k_B T · δI_k

where δI_k is the mutual information newly unlocked by the updated kernel,
measured in nats (use × ln2 for bits).

PowerMonitor provides GPU power logging via nvidia-smi subprocess (no pynvml
dependency required).  It measures the physical work W during a kernel-change
event (e.g., a fine-tuning step), enabling an empirical test of the bound.

All energy units are joules; all information units are nats by default.
"""

from __future__ import annotations

import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np

from ..kernel.space import hilbert_schmidt_distance

# Boltzmann constant in J/K
K_B: float = 1.380649e-23
# Room temperature (K)
T_ROOM: float = 298.15


# ---------------------------------------------------------------------------
# Landauer bound
# ---------------------------------------------------------------------------

def landauer_bound(
    delta_I_nats: float,
    temperature_kelvin: float = T_ROOM,
) -> float:
    """Minimum work required to realise δI nats of new mutual information.

    δW_min = k_B T · δI_k  (in joules)

    Parameters
    ----------
    delta_I_nats : float
        Change in mutual information in nats (use bits × ln2 to convert).
    temperature_kelvin : float
        System temperature (default 298.15 K ≈ 25°C).

    Returns
    -------
    float : minimum work in joules.
    """
    return K_B * temperature_kelvin * delta_I_nats


def bits_to_nats(bits: float) -> float:
    return bits * np.log(2)


def nats_to_bits(nats: float) -> float:
    return nats / np.log(2)


def joules_to_kbt_units(joules: float,
                        temperature_kelvin: float = T_ROOM) -> float:
    """Express energy in units of k_B T."""
    return joules / (K_B * temperature_kelvin)


# ---------------------------------------------------------------------------
# Mutual information change from kernel matrices
# ---------------------------------------------------------------------------

def kernel_mutual_information_change(
    K1: np.ndarray,
    K2: np.ndarray,
    method: str = "spectral",
    eps: float = 1e-10,
) -> float:
    """Estimate δI_k = I(A;E|k₂) − I(A;E|k₁) from the change in kernel.

    This is a proxy estimator.  The paper's bound requires the true mutual
    information; here we use the log-ratio of effective dimensions as a
    conservative lower-bound proxy.

    'spectral' method: δI ≈ Δ[log det(I + K)] using the log-determinant
    divergence between kernel matrices, which equals the KL divergence
    between the corresponding kernel-PCA Gaussian distributions.

    Parameters
    ----------
    K1, K2 : (N, N) kernel matrices before and after change.
    method : 'spectral' (default) or 'hs_proxy'.
    eps : eigenvalue floor.

    Returns
    -------
    delta_I : float in nats (≥ 0 if the new kernel is more expressive).
    """
    K1 = np.asarray(K1, dtype=float)
    K2 = np.asarray(K2, dtype=float)
    n = K1.shape[0]

    if method == "spectral":
        ev1 = np.maximum(np.linalg.eigvalsh(K1), eps)
        ev2 = np.maximum(np.linalg.eigvalsh(K2), eps)
        # Log-det divergence (one-sided KL between Gaussians with covariance K)
        logdet1 = np.sum(np.log(ev1))
        logdet2 = np.sum(np.log(ev2))
        # Proxy: difference in log-partition function
        delta_I = float((logdet2 - logdet1) / n)
        return max(delta_I, 0.0)

    elif method == "hs_proxy":
        # Crude lower bound: HS distance as information proxy
        d = hilbert_schmidt_distance(K1, K2)
        # Convert via: 1 nat ≈ 1 unit of HS distance at unit scale
        return float(d)

    else:
        raise ValueError(f"Unknown method {method!r}. Use 'spectral' or 'hs_proxy'.")


# ---------------------------------------------------------------------------
# Bound verification
# ---------------------------------------------------------------------------

def check_landauer_bound(
    measured_work_joules: float,
    K1: np.ndarray,
    K2: np.ndarray,
    temperature_kelvin: float = T_ROOM,
    method: str = "spectral",
) -> dict:
    """Check whether the measured work satisfies the Landauer bound δW ≥ k_BT δI_k.

    Returns a diagnostic dict useful for logging fine-tuning experiments.
    """
    delta_I = kernel_mutual_information_change(K1, K2, method=method)
    bound = landauer_bound(delta_I, temperature_kelvin)
    satisfied = measured_work_joules >= bound
    efficiency = bound / (measured_work_joules + 1e-300)

    return {
        "delta_I_nats": delta_I,
        "delta_I_bits": nats_to_bits(delta_I),
        "landauer_bound_joules": bound,
        "measured_work_joules": measured_work_joules,
        "bound_satisfied": bool(satisfied),
        "thermodynamic_efficiency": float(efficiency),
        "temperature_K": temperature_kelvin,
        "hs_distance": float(hilbert_schmidt_distance(K1, K2)),
    }


# ---------------------------------------------------------------------------
# GPU Power Monitor
# ---------------------------------------------------------------------------

@dataclass
class _PowerSample:
    timestamp: float
    power_watts: float


class PowerMonitor:
    """Context manager that polls GPU power via nvidia-smi.

    Usage
    -----
    with PowerMonitor(gpu_id=0, interval_s=0.1) as pm:
        train_step(model, batch)

    print(pm.total_energy_joules())
    print(pm.mean_power_watts())

    Falls back to CPU time × TDP estimate when nvidia-smi is unavailable.
    """

    def __init__(
        self,
        gpu_id: int = 0,
        interval_s: float = 0.1,
        cpu_tdp_watts: float = 65.0,
    ):
        self.gpu_id = gpu_id
        self.interval_s = interval_s
        self.cpu_tdp_watts = cpu_tdp_watts

        self._samples: List[_PowerSample] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._nvidia_available: bool = self._check_nvidia()

    def _check_nvidia(self) -> bool:
        try:
            subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, timeout=2
            )
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _poll_power(self) -> None:
        while not self._stop_event.is_set():
            t = time.time()
            power = self._read_power_watts()
            self._samples.append(_PowerSample(timestamp=t, power_watts=power))
            time.sleep(self.interval_s)

    def _read_power_watts(self) -> float:
        if not self._nvidia_available:
            return self.cpu_tdp_watts
        try:
            result = subprocess.run(
                ["nvidia-smi",
                 f"--id={self.gpu_id}",
                 "--query-gpu=power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=1
            )
            return float(result.stdout.strip())
        except Exception:
            return self.cpu_tdp_watts

    def __enter__(self) -> "PowerMonitor":
        self._samples.clear()
        self._stop_event.clear()
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._poll_power, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args) -> None:
        self._end_time = time.time()
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)

    def total_energy_joules(self) -> float:
        """Trapezoidal integration of power samples → energy in joules."""
        if len(self._samples) < 2:
            elapsed = (self._end_time or time.time()) - (self._start_time or 0)
            return elapsed * self.cpu_tdp_watts
        times = np.array([s.timestamp for s in self._samples])
        powers = np.array([s.power_watts for s in self._samples])
        return float(np.trapz(powers, times))

    def mean_power_watts(self) -> float:
        if not self._samples:
            return 0.0
        return float(np.mean([s.power_watts for s in self._samples]))

    def elapsed_seconds(self) -> float:
        if self._start_time is None:
            return 0.0
        end = self._end_time or time.time()
        return end - self._start_time

    def summary(self) -> dict:
        return {
            "elapsed_s": self.elapsed_seconds(),
            "mean_power_W": self.mean_power_watts(),
            "total_energy_J": self.total_energy_joules(),
            "n_samples": len(self._samples),
            "nvidia_available": self._nvidia_available,
        }


# ---------------------------------------------------------------------------
# Thermodynamic efficiency aggregator
# ---------------------------------------------------------------------------

@dataclass
class ThermodynamicEfficiency:
    """Accumulates Landauer-bound checks across multiple kernel-change events.

    Use to build an efficiency curve over a fine-tuning run.
    """
    records: List[dict] = field(default_factory=list)

    def record(
        self,
        step: int,
        K_before: np.ndarray,
        K_after: np.ndarray,
        work_joules: float,
        temperature_kelvin: float = T_ROOM,
    ) -> dict:
        result = check_landauer_bound(
            work_joules, K_before, K_after, temperature_kelvin
        )
        result["step"] = step
        self.records.append(result)
        return result

    def efficiency_curve(self) -> Tuple[np.ndarray, np.ndarray]:
        """steps, efficiencies arrays."""
        steps = np.array([r["step"] for r in self.records])
        effs = np.array([r["thermodynamic_efficiency"] for r in self.records])
        return steps, effs

    def mean_efficiency(self) -> float:
        if not self.records:
            return 0.0
        return float(np.mean([r["thermodynamic_efficiency"] for r in self.records]))

    def fraction_satisfying_bound(self) -> float:
        if not self.records:
            return 0.0
        return float(np.mean([r["bound_satisfied"] for r in self.records]))
