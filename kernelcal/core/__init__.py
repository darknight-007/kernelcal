"""Stable core API facade for widely cited kernelcal primitives.

This module provides a narrow import surface intended to remain stable across
minor releases, even while subpackages evolve. Prefer these imports in papers,
tutorials, and downstream integrations that need API continuity.
"""

from __future__ import annotations

from ..kernel import FixedPointDetector, KernelTrajectory
from ..maxcal import MaxCalSampler

__all__ = [
    "FixedPointDetector",
    "KernelTrajectory",
    "MaxCalSampler",
]
