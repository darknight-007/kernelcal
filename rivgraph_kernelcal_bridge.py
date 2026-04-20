#!/usr/bin/env python3
"""Back-compat shim for the former top-level ``rivgraph_kernelcal_bridge``.

The implementation moved to :mod:`kernelcal.integrations.rivgraph`. New
code should import from there::

    from kernelcal.integrations.rivgraph import main, rivgraph_to_terrain_graph

This shim is kept so existing scripts that do
``import rivgraph_kernelcal_bridge as bridge``, ``from rivgraph_kernelcal_bridge
import X``, or ``python rivgraph_kernelcal_bridge.py --flag ...`` continue to
work unchanged. Attribute access is forwarded via PEP 562 ``__getattr__``, so
every public name in the canonical module is reachable here.

Prefer one of:
  * ``kernelcal-rivgraph-bridge ...``              (console-script entry point)
  * ``python -m kernelcal.integrations.rivgraph ...``

Running this file directly still works too::

    python rivgraph_kernelcal_bridge.py --rivgraph-repo /path/to/rivgraph --plot
"""

from __future__ import annotations

from kernelcal.integrations import rivgraph as _impl
from kernelcal.integrations.rivgraph import main

__all__ = ["main"]


def __getattr__(name: str):
    """Forward any other attribute lookup to the canonical module."""
    return getattr(_impl, name)


if __name__ == "__main__":
    main()
