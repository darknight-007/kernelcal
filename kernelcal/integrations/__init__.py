"""Third-party integrations for kernelcal.

This subpackage is the canonical home for glue code that bridges
kernelcal's spectral / topological diagnostics to *external* libraries
and datasets. It exists so top-level experiment scripts don't have to
keep reinventing the wrapping logic, and so each integration has a
single, testable module path.

Modules
-------
:mod:`kernelcal.integrations.rivgraph`
    RivGraph (https://github.com/VeinsOfTheEarth/RivGraph) bridge:
    georeferenced channel-mask → RivGraph skeleton / network →
    kernelcal Laplacian diagnostics. Also exposes a ``main()``
    console-script entry point (``kernelcal-rivgraph-bridge``).

Back-compat
-----------
A legacy ``rivgraph_kernelcal_bridge`` module at the repo root forwards
all attribute access to :mod:`kernelcal.integrations.rivgraph` via
``__getattr__`` (PEP 562), so existing callers that do
``import rivgraph_kernelcal_bridge as bridge`` keep working during the
transition. New code should import directly from this subpackage.

Design note: modules here may depend on heavy optional packages (e.g.
``rivgraph``, ``networkx``). Imports are therefore deferred to function
scope inside each submodule so that ``import kernelcal.integrations``
itself stays cheap.
"""

from __future__ import annotations

__all__: list[str] = []
