# AGENTS.md – sources/ guidance

This directory inherits the root `AGENTS.md`. These rules apply to source implementations.

## Expectations

- New sources MUST be registered via `SourceRegistry` and a factory; avoid ad-hoc wiring.
- Remote sources MUST use `SafeHttpClient` (via `HttpConfig.build_client()` or `make_http_client`).
- Preserve record metadata shape; add fields only in an additive, backward-compatible way.

## Targeted tests

- `PYTHONPATH=src pytest tests/test_sources_fs.py`
- `PYTHONPATH=src pytest tests/test_jsonl_source.py`
- `PYTHONPATH=src pytest tests/test_sqlite_source_security.py tests/test_sqlite_source_validation.py`
- `PYTHONPATH=src pytest -k "source"`
