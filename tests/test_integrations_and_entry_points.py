"""Tests for PR-C: integrations subpackage + console-script entry points.

Covers:

1. ``kernelcal.integrations.rivgraph`` imports cleanly and exposes
   ``main``, ``rivgraph_to_terrain_graph``, and ``terrain_graph_from_graphiphy``
   (names that existing external callers import).
2. The root-level ``rivgraph_kernelcal_bridge`` shim forwards attribute
   access to the canonical module via PEP 562 ``__getattr__`` — so
   ``import rivgraph_kernelcal_bridge as bridge; bridge.main`` and
   ``from rivgraph_kernelcal_bridge import rivgraph_to_terrain_graph``
   both resolve to the canonical callable.
3. ``kernelcal.attention.landauer.main`` exists and ``_main`` is a
   back-compat alias for it.
4. The ``[project.scripts]`` entries declared in ``pyproject.toml``
   resolve to real callables.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

# ``tomllib`` is in the stdlib from Python 3.11. On 3.9/3.10 fall back to
# the PyPI ``tomli`` package (already pulled in transitively by pip). The
# pyproject-parsing tests skip cleanly if neither is available.
try:  # pragma: no cover - trivial import shim
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# kernelcal.integrations.rivgraph — canonical module
# ---------------------------------------------------------------------------


class TestRivgraphIntegration:
    def test_module_imports(self):
        mod = importlib.import_module("kernelcal.integrations.rivgraph")
        assert mod is not None

    def test_main_is_callable(self):
        from kernelcal.integrations.rivgraph import main

        assert callable(main)

    def test_public_helpers_are_exposed(self):
        """Names external callers historically imported from the bridge."""
        from kernelcal.integrations.rivgraph import (
            rivgraph_to_terrain_graph,
            terrain_graph_from_graphiphy,
            resolve_laplacian_terrain_graph,
        )

        for fn in (
            rivgraph_to_terrain_graph,
            terrain_graph_from_graphiphy,
            resolve_laplacian_terrain_graph,
        ):
            assert callable(fn)

    def test_integrations_package_has_docstring(self):
        import kernelcal.integrations as pkg

        assert pkg.__doc__ is not None
        assert "rivgraph" in pkg.__doc__.lower()


# ---------------------------------------------------------------------------
# Back-compat shim at repo root
# ---------------------------------------------------------------------------


class TestRivgraphBackcompatShim:
    def _load_shim(self):
        # Make the repo root importable even when this test is run from an
        # installed package path where the shim wouldn't otherwise be on
        # sys.path. (``testpaths = ['tests']`` + ``pythonpath = ['.']`` in
        # pyproject.toml already covers this in CI, but belt-and-braces.)
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        # Force a fresh import so a cached stale copy doesn't hide regressions.
        sys.modules.pop("rivgraph_kernelcal_bridge", None)
        return importlib.import_module("rivgraph_kernelcal_bridge")

    def test_shim_main_is_canonical_main(self):
        shim = self._load_shim()
        from kernelcal.integrations.rivgraph import main as canonical_main

        assert shim.main is canonical_main

    def test_shim_forwards_via_getattr(self):
        """PEP 562 ``__getattr__`` forwards any lookup to the impl module."""
        shim = self._load_shim()
        from kernelcal.integrations import rivgraph as canonical

        assert shim.rivgraph_to_terrain_graph is canonical.rivgraph_to_terrain_graph
        assert shim.terrain_graph_from_graphiphy is canonical.terrain_graph_from_graphiphy

    def test_shim_from_import_works(self):
        """``from rivgraph_kernelcal_bridge import X`` works for forwarded names."""
        self._load_shim()
        from rivgraph_kernelcal_bridge import rivgraph_to_terrain_graph  # noqa: F401
        from kernelcal.integrations.rivgraph import (
            rivgraph_to_terrain_graph as canonical_fn,
        )

        assert rivgraph_to_terrain_graph is canonical_fn

    def test_shim_has_docstring_pointing_to_new_location(self):
        shim = self._load_shim()
        assert shim.__doc__ is not None
        assert "kernelcal.integrations.rivgraph" in shim.__doc__


# ---------------------------------------------------------------------------
# Landauer main alias
# ---------------------------------------------------------------------------


class TestLandauerMainAlias:
    def test_main_exists_and_is_callable(self):
        from kernelcal.attention.landauer import main

        assert callable(main)

    def test_private_alias_is_same_object(self):
        """``_main`` is kept as a back-compat alias for ``main``."""
        from kernelcal.attention.landauer import _main, main

        assert _main is main


# ---------------------------------------------------------------------------
# pyproject.toml [project.scripts] — each target must resolve
# ---------------------------------------------------------------------------


@pytest.mark.skipif(tomllib is None, reason="Neither stdlib tomllib (3.11+) nor tomli is available")
class TestProjectScripts:
    @pytest.fixture(scope="class")
    def scripts(self):
        with (REPO_ROOT / "pyproject.toml").open("rb") as f:
            data = tomllib.load(f)
        return data["project"].get("scripts", {})

    def test_expected_entry_points_declared(self, scripts):
        assert "kernelcal-rivgraph-bridge" in scripts
        assert "kernelcal-landauer" in scripts

    @pytest.mark.parametrize(
        "ep_name,ep_target",
        [
            ("kernelcal-rivgraph-bridge", "kernelcal.integrations.rivgraph:main"),
            ("kernelcal-landauer", "kernelcal.attention.landauer:main"),
        ],
    )
    def test_entry_point_target_matches_expected(self, scripts, ep_name, ep_target):
        assert scripts[ep_name] == ep_target

    def test_every_entry_point_target_resolves(self, scripts):
        """Each ``module:function`` target must import and be callable."""
        for ep_name, spec in scripts.items():
            module_name, _, attr = spec.partition(":")
            assert module_name and attr, (
                f"{ep_name!r} target {spec!r} is not in 'module:function' form"
            )
            mod = importlib.import_module(module_name)
            fn = getattr(mod, attr, None)
            assert callable(fn), (
                f"{ep_name!r} -> {spec!r} did not resolve to a callable "
                f"(got {fn!r})"
            )
