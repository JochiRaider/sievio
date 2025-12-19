# Contributing to Sievio

Use this guide for day-to-day development. For architecture rules, read `AGENTS.md` and `LLMS.md`; for high-level theory of operation, see `docs/TECHNICAL_MANUAL.md`.

## Quickstart
- Python 3.11+.
- Install in editable mode (extras as needed): `pip install -e ".[dev,tok,pdf,evtx,parquet,qc,langid]"`.
- Required checks: `PYTHONPATH=src pytest`, `ruff check .`, `mypy --config-file pyproject.toml src`.
- Sample scripts for smoke tests: `scripts/manual_test_github_toml.py`, `scripts/manual_test_web_pdf.py`.

## Conventions for contributors
- **File layout:** New sources under `src/sievio/sources/`, sinks under `src/sievio/sinks/`, QC helpers under `src/sievio/core/extras/` or `src/sievio/core/`, orchestration/CLI helpers under `src/sievio/cli/`.
- **Registries, not ad-hoc wiring:** Prefer `SourceRegistry`, `SinkRegistry`, `BytesHandlerRegistry`, and `QualityScorerRegistry` (optionally via `sievio.plugins`) over hard-coded branching. Keep configs runtime-free (no live clients/scorers/extractors).
- **HTTP/safety:** Use `SafeHttpClient` via `HttpConfig.build_client()` or `safe_http.get_global_http_client()` for remote access (GitHub, PDFs, SQLite downloads).
- **Logging:** Use `get_logger(__name__)`; respect `LoggingConfig`. Avoid print in core code.
- **QC defaults:** Keep `qc.enabled` defaulting to `false` in examples; implement new QC logic as `QualityScorer`/heuristics, not inside sources/sinks.
- **Tests/style:** Add tests under `tests/` with `pytest`; keep typing/style via `mypy` and `ruff`.

## Extending Sievio
- **New Source/Sink:** Implement the `Source`/`Sink` protocol, register a factory with `SourceRegistry`/`SinkRegistry` (core or plugin), and add a `[[sources.specs]]` / `[[sinks.specs]]` example.
- **New bytes handler:** Register `(sniff, handler)` with `BytesHandlerRegistry` for binary formats. Handlers return iterable records given bytes, relative path, optional `RepoContext`, and optional `ChunkPolicy`.
- **Custom QC scorer:** Implement `QualityScorer` or a factory (`id`, `build(cfg: QCConfig)`), register with `quality_scorer_registry` or via a plugin. Leave `qc.scorer` unset in declarative configs; use registries instead.
- **Plugins:** Publish an entry point under `sievio.plugins` that receives registries and performs registrations. For finer control, pass a `RegistryBundle` to `build_pipeline_plan`.

### Extension points
- **Registries/plugins:** Register custom sources/sinks/bytes handlers/QC or safety scorers via `core/registries.py` or plugins.
- **Record/file middlewares:** Implement `RecordMiddleware` / `FileMiddleware` or simple functions; inject via `PipelineOverrides.record_middlewares` / `file_middlewares` (wired by `build_engine`).
- **Run lifecycle hooks:** Implement `RunLifecycleHook` for run-level artifacts (summaries, dataset cards) and register via the builder or registries.

Example middleware:
```python
from sievio import load_config_from_path, convert, PipelineOverrides

def add_source_tag(record):
    meta = record.setdefault("meta", {})
    meta.setdefault("tags", []).append("my-source")
    return record

cfg = load_config_from_path("config.toml")
overrides = PipelineOverrides(record_middlewares=[add_source_tag])
stats = convert(cfg, overrides=overrides)
```

### Contributor cheat sheet
- **New source type:** Implement `Source`, add `SourceFactory`, register (core or plugin), add `[[sources.specs]]` example.
- **New sink:** Implement `Sink`, add `SinkFactory`, register, document `[[sinks.specs]]`.
- **New binary handler:** Implement `(sniff, handler)`, register with `BytesHandlerRegistry`, optionally gate behind an extra.
- **Tweak metadata:** Use `RecordMeta`, `RepoContext.as_meta_seed()`, `ensure_meta_dict`, `merge_meta_defaults`; put QC signals under `meta["extra"]["qc_signals"]`.
- **Change QC:** Start from `QCConfig`/`QCHeuristics`, implement `QualityScorer.score_record`, register via `quality_scorer_registry`; use `core/qc_post.py` for post-QC/CSV export.
- **Touch runner/high-level API:** Keep `convert` thin (build plan → run engine → return stats). New profile helpers should build a `SievioConfig` via existing factories/registries and set metadata/dataset_card sensibly.
- **Dataset cards:** See `src/sievio/core/dataset_card.py` (`CardFragment`, `DatasetCardFields`, `write_card_fragment_for_run`, `build_dataset_card_from_fragments`). Add new stats via `PipelineStats` and fragment aggregation.

## Development & testing
- Dev deps: `[project.optional-dependencies].dev` (`pytest`, `pytest-cov`, `ruff`, `mypy`, `build`, `twine`).
- Commands: `PYTHONPATH=src pytest`; `ruff check .`; `mypy --config-file pyproject.toml src`.
- Minimum Python: 3.11.
- Smoke tests: `scripts/manual_test_github_toml.py`, `scripts/manual_test_web_pdf.py` (prepend `src/` to `sys.path`).
