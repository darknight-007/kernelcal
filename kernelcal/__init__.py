"""
kernelcal
=========
Python library implementing Kernel Dynamics under Maximum Caliber (MaxCal),
with integration hooks for DeepGIS-XR.  Each subpackage corresponds to one
of the seven integration threads documented in deepgis_maxcal_integration.md.

Companion to: "Kernel Dynamics under Path Entropy Maximization" (Das, ASU)
Repository:   https://github.com/darknight-007/kernelcal

Subpackages
-----------
kernel
    Hilbert-Schmidt geometry on kernel space K, kernel trajectories, and
    fixed-point detection.  (Paper §3-4)

maxcal
    Core MaxCal functional (path entropy, Lagrange dual, fitting) and the
    MaxCalSampler drop-in for the DeepGIS World Sampler.  (Paper §2, §5)

ntk
    NTK tracker for monitoring representational drift during fine-tuning,
    and the Hellinger kernel / NTK–Hellinger comparison (Conjecture 3).

assembly
    RKHS complexity metrics, per-tile complexity maps, and the assembly-
    theory-motivated reward signal for the World Sampler.  (Paper §6)

thermodynamics
    Landauer bound checks, GPU power monitoring, and thermodynamic
    efficiency accumulation.  (Paper Theorem 1)

models
    MaxCal multi-model kernel selector over the five DeepGIS AI backends.

prompts
    Self-consistent prompt iteration for Grounding DINO.

Quick-start
-----------
>>> from kernelcal.maxcal import MaxCalSampler
>>> from kernelcal.models import ModelKernelSelector
>>> from kernelcal.thermodynamics import PowerMonitor
>>> from kernelcal.ntk import NTKTracker, compare_ntk_to_hellinger
>>> from kernelcal.assembly import complexity_map, assembly_reward_signal
>>> from kernelcal.prompts import PromptKernelIterator
>>> from kernelcal.kernel import KernelTrajectory, FixedPointDetector
"""

from .kernel import (
    hilbert_schmidt_distance,
    hilbert_schmidt_norm,
    is_psd,
    project_to_psd,
    KernelTrajectory,
    FixedPointDetector,
)
from .maxcal import (
    path_entropy,
    fit_lagrange_multipliers,
    MaxCalSampler,
)
from .ntk import (
    NTKTracker,
    compute_empirical_ntk,
    hellinger_kernel_matrix,
    compare_ntk_to_hellinger,
)
from .assembly import (
    rkhs_norm,
    spectral_complexity,
    complexity_map,
    assembly_reward_signal,
    assembly_index_lower_bound,
)
from .thermodynamics import (
    landauer_bound,
    kernel_mutual_information_change,
    check_landauer_bound,
    PowerMonitor,
    ThermodynamicEfficiency,
)
from .models import ModelKernelSelector, ModelRecord
from .prompts import PromptKernelIterator, prompt_kernel_distance

__version__ = "0.1.0"

__all__ = [
    # kernel
    "hilbert_schmidt_distance",
    "hilbert_schmidt_norm",
    "is_psd",
    "project_to_psd",
    "KernelTrajectory",
    "FixedPointDetector",
    # maxcal
    "path_entropy",
    "fit_lagrange_multipliers",
    "MaxCalSampler",
    # ntk
    "NTKTracker",
    "compute_empirical_ntk",
    "hellinger_kernel_matrix",
    "compare_ntk_to_hellinger",
    # assembly
    "rkhs_norm",
    "spectral_complexity",
    "complexity_map",
    "assembly_reward_signal",
    "assembly_index_lower_bound",
    # thermodynamics
    "landauer_bound",
    "kernel_mutual_information_change",
    "check_landauer_bound",
    "PowerMonitor",
    "ThermodynamicEfficiency",
    # models
    "ModelKernelSelector",
    "ModelRecord",
    # prompts
    "PromptKernelIterator",
    "prompt_kernel_distance",
]
