"""
Automatic energy monitoring for training experiments.

Aggregates all available hardware power sources — no manual kWh entry.
Priority: GPU hardware counter > GPU polling > Intel RAPL > FLOPs estimate.

When wifi wall-power meters are available, pass wall_watts_callback to
capture system-level power alongside component-level readings.

Usage:
    monitor = EnergyMonitor.auto_detect()
    monitor.start()
    # ... training loop ...
    report = monitor.stop()
    print(f"Total energy: {report.total_joules:.2f} J = {report.total_wh:.6f} Wh")
"""

from __future__ import annotations

import os
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class EnergyReport:
    """Energy consumption report from a monitored interval."""
    elapsed_s: float
    gpu_joules: float
    cpu_joules: float
    flops_estimate: int
    flops_joules: float
    sources_used: List[str]
    wall_joules: float = 0.0
    n_samples: int = 0

    @property
    def total_joules(self) -> float:
        if self.wall_joules > 0:
            return self.wall_joules
        return self.gpu_joules + self.cpu_joules + self.flops_joules

    @property
    def total_wh(self) -> float:
        return self.total_joules / 3600.0

    @property
    def mean_power_watts(self) -> float:
        if self.elapsed_s <= 0:
            return 0.0
        return self.total_joules / self.elapsed_s

    def summary(self) -> dict:
        return {
            "elapsed_s": self.elapsed_s,
            "total_joules": self.total_joules,
            "total_wh": self.total_wh,
            "mean_power_watts": self.mean_power_watts,
            "gpu_joules": self.gpu_joules,
            "cpu_joules": self.cpu_joules,
            "flops_estimate": self.flops_estimate,
            "flops_joules": self.flops_joules,
            "wall_joules": self.wall_joules,
            "sources": self.sources_used,
            "n_samples": self.n_samples,
        }


class _GPUProbe:
    """GPU energy via pynvml hardware counter or polling."""

    def __init__(self, device_id: int = 0, poll_interval: float = 0.5):
        self._device_id = device_id
        self._poll_interval = poll_interval
        self._nvml = None
        self._handle = None
        self._readings: List[Tuple[float, float]] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._energy_start_mj: Optional[float] = None
        self.available = False
        self.method = "none"
        self._init()

    def _init(self):
        try:
            import pynvml as nvml
            nvml.nvmlInit()
            self._handle = nvml.nvmlDeviceGetHandleByIndex(self._device_id)
            self._nvml = nvml
            self.available = True
            try:
                nvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
                self.method = "nvml_hw_counter"
            except Exception:
                self.method = "nvml_poll"
        except Exception:
            pass

    def _read_watts(self) -> float:
        if self._nvml is None:
            return 0.0
        try:
            return self._nvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
        except Exception:
            return 0.0

    def _read_energy_mj(self) -> Optional[float]:
        if self._nvml is None:
            return None
        try:
            return float(self._nvml.nvmlDeviceGetTotalEnergyConsumption(self._handle))
        except Exception:
            return None

    def _poll_loop(self):
        while self._running:
            self._readings.append((time.time(), self._read_watts()))
            time.sleep(self._poll_interval)

    def start(self):
        self._readings.clear()
        self._energy_start_mj = self._read_energy_mj()
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

        end_mj = self._read_energy_mj()
        if self._energy_start_mj is not None and end_mj is not None:
            return (end_mj - self._energy_start_mj) / 1000.0  # mJ → J

        if len(self._readings) < 2:
            return 0.0
        t = np.array([r[0] for r in self._readings])
        w = np.array([r[1] for r in self._readings])
        return float(np.trapz(w, t))

    @property
    def n_samples(self) -> int:
        return len(self._readings)


class _RAPLProbe:
    """CPU/DRAM energy via Intel RAPL (Linux sysfs)."""

    RAPL_BASE = Path("/sys/class/powercap/intel-rapl")

    def __init__(self):
        self.available = False
        self._domains: List[Path] = []
        self._start_uj: Dict[str, int] = {}
        self._init()

    def _init(self):
        if not self.RAPL_BASE.exists():
            return
        for d in sorted(self.RAPL_BASE.glob("intel-rapl:*")):
            ej = d / "energy_uj"
            if ej.exists() and os.access(str(ej), os.R_OK):
                self._domains.append(ej)
                self.available = True

    def _read_uj(self) -> Dict[str, int]:
        out = {}
        for p in self._domains:
            try:
                out[str(p)] = int(p.read_text().strip())
            except (OSError, ValueError):
                pass
        return out

    def start(self):
        self._start_uj = self._read_uj()

    def stop(self) -> float:
        end_uj = self._read_uj()
        total_uj = 0
        for key in self._start_uj:
            if key in end_uj:
                delta = end_uj[key] - self._start_uj[key]
                if delta < 0:
                    delta += 2**32  # counter wrap
                total_uj += delta
        return total_uj / 1e6  # µJ → J


class _FLOPSEstimator:
    """Estimate energy from FLOPs when no hardware monitoring is available."""

    JOULES_PER_FLOP = {
        "gpu_fp16": 0.5e-12,   # ~0.5 pJ/FLOP for modern GPU FP16
        "gpu_fp32": 1.0e-12,   # ~1 pJ/FLOP for GPU FP32
        "cpu_fp32": 5.0e-12,   # ~5 pJ/FLOP rough CPU estimate
    }

    def __init__(self):
        self.total_flops: int = 0
        self._mode = "gpu_fp16"

    def set_mode(self, mode: str):
        if mode in self.JOULES_PER_FLOP:
            self._mode = mode

    def add_training_step(self, n_params: int, batch_tokens: int):
        """Approximate FLOPs for one forward+backward pass: ~6 * n_params * tokens."""
        self.total_flops += 6 * n_params * batch_tokens

    def add_flops(self, flops: int):
        self.total_flops += flops

    def energy_joules(self) -> float:
        return self.total_flops * self.JOULES_PER_FLOP[self._mode]

    def reset(self):
        self.total_flops = 0


class EnergyMonitor:
    """Unified energy monitor: auto-detects all available hardware sources.

    Priority for total_joules:
      1. Wall power (if wall_watts_callback provided)
      2. GPU hw counter or polling (pynvml)
      3. Intel RAPL (CPU+DRAM)
      4. FLOPs estimate (always available as fallback)

    GPU + RAPL readings are additive (different subsystems).
    Wall power, when available, replaces component readings as the
    authoritative total.
    """

    def __init__(
        self,
        gpu_device_id: int = 0,
        poll_interval: float = 0.5,
        wall_watts_callback: Optional[Callable[[], float]] = None,
    ):
        self._gpu = _GPUProbe(gpu_device_id, poll_interval)
        self._rapl = _RAPLProbe()
        self._flops = _FLOPSEstimator()
        self._wall_callback = wall_watts_callback
        self._wall_readings: List[Tuple[float, float]] = []
        self._wall_running = False
        self._wall_thread: Optional[threading.Thread] = None
        self._poll_interval = poll_interval
        self._start_time = 0.0
        self._end_time = 0.0

    @classmethod
    def auto_detect(cls, gpu_device_id: int = 0, **kwargs) -> "EnergyMonitor":
        """Create an EnergyMonitor with auto-detected backends."""
        m = cls(gpu_device_id=gpu_device_id, **kwargs)
        sources = []
        if m._gpu.available:
            sources.append(f"gpu:{m._gpu.method}")
        if m._rapl.available:
            sources.append("rapl")
        sources.append("flops_estimate")
        if m._wall_callback:
            sources.append("wall_meter")
        m._sources = sources
        return m

    def add_training_step(self, n_params: int, batch_tokens: int):
        self._flops.add_training_step(n_params, batch_tokens)

    def add_flops(self, flops: int):
        self._flops.add_flops(flops)

    def set_flops_mode(self, mode: str):
        self._flops.set_mode(mode)

    def _wall_poll_loop(self):
        while self._wall_running:
            try:
                w = self._wall_callback()
                self._wall_readings.append((time.time(), w))
            except Exception:
                pass
            time.sleep(self._poll_interval)

    def start(self):
        self._start_time = time.time()
        self._flops.reset()
        if self._gpu.available:
            self._gpu.start()
        if self._rapl.available:
            self._rapl.start()
        if self._wall_callback:
            self._wall_readings.clear()
            self._wall_running = True
            self._wall_thread = threading.Thread(
                target=self._wall_poll_loop, daemon=True
            )
            self._wall_thread.start()

    def stop(self) -> EnergyReport:
        self._end_time = time.time()
        elapsed = self._end_time - self._start_time

        gpu_j = self._gpu.stop() if self._gpu.available else 0.0
        cpu_j = self._rapl.stop() if self._rapl.available else 0.0
        flops_j = self._flops.energy_joules()

        wall_j = 0.0
        if self._wall_callback:
            self._wall_running = False
            if self._wall_thread:
                self._wall_thread.join(timeout=2.0)
            if len(self._wall_readings) >= 2:
                t = np.array([r[0] for r in self._wall_readings])
                w = np.array([r[1] for r in self._wall_readings])
                wall_j = float(np.trapz(w, t))

        sources = getattr(self, "_sources", ["flops_estimate"])

        return EnergyReport(
            elapsed_s=elapsed,
            gpu_joules=gpu_j,
            cpu_joules=cpu_j,
            flops_estimate=self._flops.total_flops,
            flops_joules=flops_j,
            sources_used=sources,
            wall_joules=wall_j,
            n_samples=self._gpu.n_samples,
        )

    @staticmethod
    def estimate_flops_energy(
        n_params: int,
        n_steps: int,
        batch_tokens: int,
        mode: str = "gpu_fp16",
    ) -> dict:
        """Quick offline estimate without running anything."""
        flops = 6 * n_params * batch_tokens * n_steps
        j_per_flop = _FLOPSEstimator.JOULES_PER_FLOP.get(mode, 1e-12)
        joules = flops * j_per_flop
        return {
            "total_flops": flops,
            "joules": joules,
            "wh": joules / 3600,
            "mode": mode,
        }
