# Navigation demos

Standalone runnable scripts that exercise the `kernelcal.navigation` stack
end-to-end. They are **not** pytest modules — they used to live under
`tests/` which caused pytest to collect (and warn about) them; they were
moved here in the PR-B refactor so the distinction between *unit tests*
(`tests/`) and *executable examples* (`examples/`) stays sharp.

## Scripts

| Script | Purpose |
|---|---|
| [`toy_navigation_2d.py`](toy_navigation_2d.py) | 10×10 grid scenario that chains `SemanticSLAMKernelTracker` → `InformativePathPlanner` → `HumanPilotDemonstrationLearner` through SLAM warm-up, autonomous nav, pilot demonstrations, and λ-transfer. Records wall-clock, CPU, memory, and (if available) GPU telemetry per kernelcal call. Produces 8 diagnostic figures. |
| [`demo_velocity_control.py`](demo_velocity_control.py) | 12×12 Earth Rover traversal demonstrating `TerrainKernelVelocityController`: high-complexity terrain patches, a "tracking lost" event, pilot bias, and MaxCal planner integration. Produces 6 diagnostic figures (velocity heat-map, per-factor breakdown, SLAM kernel trace, path comparison, spatial speed map). Referenced from `VELOCITY_CONTROL.md`. |

## Running

From the repository root:

```bash
python examples/navigation/toy_navigation_2d.py
python examples/navigation/demo_velocity_control.py
```

Each script inserts the repo root on `sys.path` so `import kernelcal`
resolves without `pip install -e .`.

## Output

Both scripts write PNGs to `examples/navigation/figures/` (created on
first run). That directory is listed in `.gitignore` — the figures are
regenerable artifacts, and the *source of truth* is the script itself.
