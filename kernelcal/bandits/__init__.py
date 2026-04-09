"""
kernelcal.bandits
=================
Decentralised Dynamic-Kernel GP-UCB (DDK-GPUCB) simulation suite.

Corresponds to the manuscript
  "Decentralised Gaussian Process Bandits with Dynamic Kernels under MaxCal"

Sub-modules
-----------
field       -- Anisotropic 2-D GP field with spatially varying lengthscales
kernels     -- Anisotropic SE kernel, HS/Fisher-Rao distance, kernel consensus
agents      -- DDK-GPUCB agent; DDUCB and static-GP-UCB baselines
network     -- Gossip matrix, Chebyshev-accelerated mixing
experiment  -- End-to-end runner, metrics, plotting
"""

from .field import AnisotropicField
from .kernels import AnisotropicSEKernel, hs_distance, kernel_consensus
from .agents import DDKGPUCBAgent, DDUCBAgent, StaticGPUCBAgent
from .network import GossipNetwork
from .experiment import run_experiment, ExperimentConfig

__all__ = [
    "AnisotropicField",
    "AnisotropicSEKernel",
    "hs_distance",
    "kernel_consensus",
    "DDKGPUCBAgent",
    "DDUCBAgent",
    "StaticGPUCBAgent",
    "GossipNetwork",
    "run_experiment",
    "ExperimentConfig",
]
