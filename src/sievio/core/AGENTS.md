# AGENTS.md – core/ guidance

This directory inherits the root `AGENTS.md`. These rules are stricter for core orchestration, QC, and safety boundaries.

## Scope

- Read core orchestration modules; modify only when the task explicitly requires it.
- Prefer registries, lifecycle hooks, or middlewares over edits to `PipelineEngine`.

## Review notes (P0/P1)

P0 (explicit scope required):
- `core/pipeline.py`, `core/builder.py`, `core/registries.py`, `core/safe_http.py`
- `core/qc_controller.py`, `core/qc_post.py`, `core/records.py`, `core/factories_*.py`

P1 (risk notes + tests required):
- New network access paths, dependency additions, or changes to QC/dedup utilities.

## Targeted tests

- `PYTHONPATH=src pytest tests/test_pipeline_middlewares.py`
- `PYTHONPATH=src pytest tests/test_config_builder_pipeline.py`
- `PYTHONPATH=src pytest tests/test_qc_controller.py tests/test_qc_post.py`
- `PYTHONPATH=src pytest tests/test_safe_http.py`
- `PYTHONPATH=src pytest tests/test_records.py tests/test_schema_validation.py`
