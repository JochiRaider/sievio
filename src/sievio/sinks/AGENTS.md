# AGENTS.md – sinks/ guidance

This directory inherits the root `AGENTS.md`. These rules apply to sink implementations.

## Expectations

- Sinks own output layout (naming, compression, sidecars); do not scatter file writing elsewhere.
- Keep output compatibility stable; avoid changing record schema or summary formats without explicit scope.
- Ensure `finalize()` behavior preserves existing JSONL/sidecars and summary record order.

## Targeted tests

- `PYTHONPATH=src pytest tests/test_runner_finalizers.py`
- `PYTHONPATH=src pytest tests/test_pipeline_middlewares.py`
- `PYTHONPATH=src pytest tests/test_config_builder_pipeline.py`
