# Contributing to kernelcal

Thanks for contributing. This repository supports active research and applied
verification workflows, so reproducibility and clear provenance matter.

## Development setup

1. Clone the repository.
2. Use a virtual environment.
3. Install in editable mode with dev dependencies:

```bash
pip install -e ".[dev]"
```

## Run tests

Use the full suite before opening a PR:

```bash
python -m pytest tests/ -q
```

## Contribution guidelines

- Keep changes scoped to one concern (docs, tests, feature, refactor).
- Add or update tests when behavior changes.
- Prefer deterministic scripts and document data assumptions.
- Avoid committing large local data caches (`datasets/`, `cache/`, `urban_cache/`).
- Keep public APIs backward compatible when possible. If breaking changes are
  required, document migration notes in `README.md`.

## Stable API expectation

Downstream projects should prefer imports from `kernelcal.core` for these
widely cited primitives:

- `FixedPointDetector`
- `KernelTrajectory`
- `MaxCalSampler`

Changes to this facade should be rare and must be paired with tests.

## Pull request checklist

- [ ] Tests pass locally (`python -m pytest tests/ -q`)
- [ ] New/changed behavior has tests
- [ ] Docs updated if user-facing behavior changed
- [ ] No large binary artifacts or local caches added
