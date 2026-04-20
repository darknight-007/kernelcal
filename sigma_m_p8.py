"""
Back-compat shim for the former top-level ``sigma_m_p8`` module.

The canonical implementation now lives at
``kernelcal.thermodynamics.sigma_m_p8``; this shim simply re-exports every
public attribute from that module so that scripts and tests which used
``from sigma_m_p8 import ...`` continue to work after the package
promotion (PR-E reorg).

For new code, prefer::

    from kernelcal.thermodynamics.sigma_m_p8 import ...
"""

from __future__ import annotations

import importlib

_canonical = importlib.import_module("kernelcal.thermodynamics.sigma_m_p8")


def __getattr__(name: str):
    # PEP 562 — defer every attribute lookup to the canonical module so
    # additions there are visible here automatically, and nothing needs
    # a manual re-export block.
    return getattr(_canonical, name)


def __dir__():
    return sorted(set(list(globals().keys()) + dir(_canonical)))


if __name__ == "__main__":
    _canonical.run_q19_report(verbose=True)
