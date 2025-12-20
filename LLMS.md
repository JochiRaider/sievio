# Sievio - LLM Guide

This file gives LLMs a compact routing + architecture map so they can work in the
right modules and avoid touching core pipeline wiring by accident.

For operational rules, invariants, and "how to ask an AI for help," see `AGENTS.md`.
This `LLMS.md` focuses on architecture, module responsibilities, and where to make
changes for different tasks.

---

## 1. High-level overview

Sievio is a library-first, configuration-driven ingestion pipeline that turns
repositories and related artifacts into normalized JSONL / Parquet datasets
suitable for LLM pre-training, fine-tuning, and analysis.

At a high level:

- You describe a run with `SievioConfig` (Python or TOML).
- The builder turns that config into a `PipelinePlan` and `PipelineRuntime`.
- The pipeline engine coordinates sources -> decode -> chunk -> record
  construction -> sinks.
- Optional subsystems (QC, dataset cards, language ID, safety/QC extras, CLI
  runners, plugins) sit around the core rather than reimplementing it.

---

## 2. First 60 seconds: where to start by task

- Add a new source: start with `src/sievio/sources/`,
  `src/sievio/core/interfaces.py`, `src/sievio/core/factories_sources.py`,
  `src/sievio/core/registries.py`, `src/sievio/core/config.py`.
  Avoid: editing the engine loop in `src/sievio/core/pipeline.py` unless the
  change is truly orchestration-level.
- Add a new sink/output format: start with `src/sievio/sinks/sinks.py`,
  `src/sievio/sinks/parquet.py`, `src/sievio/core/interfaces.py`,
  `src/sievio/core/factories_sinks.py`, `src/sievio/core/registries.py`.
  Avoid: editing the engine loop in `src/sievio/core/pipeline.py` unless the
  change is truly orchestration-level.
- Add a bytes handler (binary format): start with
  `src/sievio/core/factories_sources.py`, `src/sievio/core/registries.py`,
  `src/sievio/core/interfaces.py`, and patterns in `src/sievio/sources/pdfio.py`,
  `src/sievio/sources/evtxio.py`, `src/sievio/sources/parquetio.py`.
  Avoid: building a custom Source unless location rules are special.
- Add or modify QC/safety scoring: start with `src/sievio/core/interfaces.py`,
  `src/sievio/core/extras/qc.py`, `src/sievio/core/extras/safety.py`,
  `src/sievio/core/factories_qc.py`, `src/sievio/core/qc_controller.py`,
  `src/sievio/core/qc_post.py`. Avoid: embedding QC logic in sources/sinks.
- Add middleware or lifecycle hooks: start with `src/sievio/core/interfaces.py`,
  `src/sievio/core/pipeline.py`, `src/sievio/core/hooks.py`,
  `src/sievio/core/builder.py`. Avoid: forking the engine for small behavior.
- Change config knobs/default merging: start with `src/sievio/core/config.py`
  (`build_config_from_defaults_and_options`) and
  `src/sievio/core/factories_sources.py` / `src/sievio/core/factories_sinks.py`.
  Avoid: reading raw `spec.options` except constructor-only identifiers.
- Adjust concurrency/executor selection: start with
  `src/sievio/core/concurrency.py`, `src/sievio/core/config.py`,
  `src/sievio/core/pipeline.py`. Avoid: per-source executor logic.
- Debug pipeline behavior, stats, dataset cards: start with
  `src/sievio/core/pipeline.py`, `src/sievio/core/records.py`,
  `src/sievio/core/hooks.py`, `src/sievio/core/dataset_card.py`,
  `src/sievio/core/qc_controller.py`. Avoid: sinks for record schema changes.

---

## 3. Key types and layering (Config -> Plan -> Runtime -> Engine -> Overrides)

Key types (paths):
- `SievioConfig` in `src/sievio/core/config.py` (declarative-only spec).
- `PipelinePlan`, `PipelineRuntime`, `PipelineOverrides` in
  `src/sievio/core/builder.py`.
- `PipelineEngine` in `src/sievio/core/pipeline.py`.
- `RunLifecycleHook`, `RecordMiddleware`, `FileMiddleware`, `InlineScreener` in
  `src/sievio/core/interfaces.py`.

Flow (text diagram):
- `SievioConfig` -> `build_pipeline_plan()` -> `PipelinePlan(spec + runtime)`
  -> `PipelineEngine(plan)` -> `apply_overrides_to_engine()` -> `engine.run()`.

What belongs where:
- Config: typed, serializable knobs only (no live clients, scorers, sources,
  sinks, or hooks). See `HttpConfig`, `QCConfig`, `SinkConfig`, `SourceConfig`.
- Plan/Runtime: constructed objects (sources, sinks, HTTP clients, bytes
  handlers, scorers, hooks, language detectors) derived from config + registries.
- Engine: executes the per-record loop and applies middlewares/hooks.
- Overrides: runtime-only replacements for config-driven wiring (HTTP client,
  QC/safety scorers, extractors, bytes handlers, middlewares).

Per-record loop (hot path) summary:
- `PipelineEngine.run()` opens sources/sinks, builds an extractor, resolves
  executor settings, and iterates source items.
- `DefaultExtractor` in `src/sievio/core/convert.py` handles decode -> chunk ->
  record building.
- File middleware runs after extraction; record middleware runs before sink
  writes inside the unified record chain.
- Lifecycle hooks (`RunLifecycleHook`) run on start, per-record, end, and
  on-artifacts.

Config normalization rule:
- `build_config_from_defaults_and_options` in `src/sievio/core/config.py` is the
  canonical merge helper for defaults + per-spec options; factories should only
  read constructor-only identifiers from `spec.options`.

Configuration (`SievioConfig`) highlights:
- `sources`: per-kind defaults in `[sources.defaults.<kind>]` plus declarative
  `[[sources.specs]]` entries. Factories consume defaults + specs.
- `decode`: Unicode normalization, control stripping, mojibake repair, and byte
  caps (`DecodeConfig` in `src/sievio/core/config.py`).
- `chunk`: tokenizer selection and `ChunkPolicy` for chunk sizes/overlap.
- `language` / `code_lang`: human and code language detectors and extension
  hints.
- `pipeline`: concurrency (`max_workers`, `executor_kind`, `submit_window`,
  `fail_fast`).
- `sinks`: output defaults and `[[sinks.specs]]` for primary JSONL and extras.
- `http`: Safe HTTP client settings (`HttpConfig` -> `SafeHttpClient`).
- `qc`: quality + safety screening config (`QCConfig`, `SafetyConfig`).
- `logging`: `LoggingConfig` for logger name/level/format.
- `metadata`: run identity info for summaries and dataset cards.
- `dataset_card`: dataset card behavior and fields.

Sources (built-ins, via registries/factories):
- `local_dir` (`src/sievio/sources/fs.py`)
- `github_zip` (`src/sievio/sources/githubio.py`)
- `web_pdf_list` / `web_page_pdf` (`src/sievio/sources/sources_webpdf.py`)
- `csv_text` (`src/sievio/sources/csv_source.py`)
- `sqlite` (`src/sievio/sources/sqlite_source.py`)
- Optional bytes handlers: PDF/EVTX/Parquet (`src/sievio/sources/pdfio.py`,
  `src/sievio/sources/evtxio.py`, `src/sievio/sources/parquetio.py`).

Decode and chunk:
- Decode lives in `src/sievio/core/decode.py` (`decode_bytes`, `read_text`).
- Chunking lives in `src/sievio/core/chunk.py` (`ChunkPolicy`, split helpers);
  token counts use `tiktoken` when `[tok]` is installed.

Extractors and records:
- `DefaultExtractor` in `src/sievio/core/convert.py` is the standard pipeline
  extractor (decode -> chunk -> record).
- Records are dicts with `"text"` and `"meta"` built by
  `src/sievio/core/records.py`; summary records include run metadata and QC
  summaries.

Sinks:
- JSONL/Prompt sinks in `src/sievio/sinks/sinks.py`.
- Parquet sink in `src/sievio/sinks/parquet.py` (requires `[parquet]` extra).

Runtime overrides (advanced):
- `PipelineOverrides` can replace HTTP client, scorers, extractors, bytes
  handlers, and middlewares for a single run without mutating `SievioConfig`.

Distributed execution (sharded runs):
- Generate shard configs via `src/sievio/core/sharding.py` and merge stats via
  `src/sievio/core/stats_aggregate.py` (see CLI helpers in `sievio shard` and
  `sievio merge-stats`).

---

## 4. Core vs non-core

- Core (`src/sievio/core/`): configs, registries, builder, pipeline engine, QC,
  dataset cards, safe HTTP, concurrency. Read-mostly unless the task requires
  changes.
- Non-core: CLI (`src/sievio/cli/`), sources (`src/sievio/sources/`), sinks
  (`src/sievio/sinks/`), extras, tests, scripts. Prefer changes here.

---

## 5. Extension points first

### Registries (preferred extension point)

- `SourceRegistry`, `SinkRegistry`, `BytesHandlerRegistry`,
  `QualityScorerRegistry`, `SafetyScorerRegistry` live in
  `src/sievio/core/registries.py`.
- Built-ins are registered via `default_source_registry()` and
  `default_sink_registry()` and bundled by `default_registries()`.
- Prefer adding new components via registries or plugins rather than modifying
  the builder/pipeline.

### Plugins

- Plugin discovery lives in `src/sievio/core/plugins.py` and loads entry points
  under the `sievio.plugins` group.
- A plugin function receives registries and registers sources/sinks/scorers.

### Overrides

- `PipelineOverrides` in `src/sievio/core/builder.py` allows runtime-only
  replacement of HTTP clients, scorers, extractors, bytes handlers, and
  middlewares.
- Prefer overrides for one-off or per-run customization; prefer registries for
  reusable components.

### Middlewares

- Record middleware: `process(record) -> Optional[Record]` (return `None` to
  drop). Protocol in `src/sievio/core/interfaces.py` and wiring in
  `src/sievio/core/pipeline.py`.
- File middleware: `process(item, records) -> Optional[Iterable[Record]]` (return
  `None` to drop all records for a file). Protocol in
  `src/sievio/core/interfaces.py` and wiring in `src/sievio/core/pipeline.py`.
- Prefer middlewares over changing the engine for cross-cutting behavior.

### Lifecycle hooks

- `RunLifecycleHook` in `src/sievio/core/interfaces.py` defines `on_run_start`,
  `on_record`, `on_run_end`, and `on_artifacts`.
- Built-ins: `RunSummaryHook` and `LanguageTaggingMiddleware` in
  `src/sievio/core/hooks.py`, `DatasetCardHook` in
  `src/sievio/core/dataset_card.py`, `InlineQCHook` in
  `src/sievio/core/qc_controller.py`, `PostQCHook`/`PostSafetyHook` in
  `src/sievio/core/qc_post.py`.
- Hooks should annotate metadata or emit sidecar artifacts; avoid writing output
  files directly outside sinks.

---

## 6. Module map

Maintenance rules:
- Keep this list aligned with `project_files.md` and the on-disk tree.
- Mark core modules as CORE and treat them as read-only unless a task requires
  edits.
- Replace placeholders with a one-line summary or a TODO pointing at specific
  symbols.

### 6.1 Core package - `src/sievio/core/` (CORE, read-only unless required)

- `__init__.py`
  Core package initializer and exports for shared constants/helpers.
- `config.py`
  Configuration models and helpers for defining `SievioConfig` and related
  config sections (sources, sinks, QC, safety, chunking, etc.).
- `records.py`
  Construction and normalization of output record dictionaries, including run
  headers and consistent metadata fields. Hosts schema-version helpers
  (`check_record_schema`).
- `naming.py`
  Utilities for building safe, normalized output filenames and extensions based
  on config and repo context.
- `licenses.py`
  Helpers for detecting and normalizing license information from repositories
  and archives.
- `decode.py`
  Decoding bytes into normalized text using encoding detection and heuristics,
  plus text normalization helpers.
- `chunk.py`
  Token-aware document/code chunking utilities and the `ChunkPolicy` machinery.
- `convert.py`
  High-level helpers that take file inputs (paths/bytes), run decode + chunking,
  and produce extractor/record dictionaries.
- `factories.py`
  Facade re-exporting factory helpers split across sinks, sources, QC, context.
- `factories_sinks.py`
  Sink and output-path factories (`OutputPaths`, JSONL/Prompt sinks, Parquet
  dataset sink).
- `factories_sources.py`
  Source factories (`LocalDirSourceFactory`, `GitHubZipSourceFactory`, etc.) and
  bytes-handler wiring (`make_bytes_handlers`, `UnsupportedBinary`).
- `factories_qc.py`
  QC and safety scorer factory helpers (`make_qc_scorer`, `make_safety_scorer`).
- `factories_context.py`
  Repo context inference and HTTP client construction helpers
  (`make_repo_context_from_git`, `make_http_client`).
- `log.py`
  Logging configuration helpers, package logger setup, and temporary level
  context manager.
- `registries.py`
  Registries for sources, sinks, bytes handlers, quality scorers, safety
  scorers, and lifecycle hooks; central place to register built-ins and plugins.
- `plugins.py`
  Plugin discovery and registration helpers (entry-point based), wiring external
  sources/sinks/scorers into registries.
- `sharding.py`
  Helpers to generate per-shard configs with isolated outputs from a base config
  + target list.
- `stats_aggregate.py`
  Utilities to merge multiple `PipelineStats.as_dict()` outputs (counts/flags
  only) for distributed runs; validates QC config consistency and clears
  non-additive QC fields like per-screener `signal_stats` and `top_dup_families`.
- `interfaces.py`
  Protocols/typed interfaces shared across the system (sources, sinks, lifecycle
  hooks, quality/safety scorers, middlewares, etc.), plus core type aliases.
- `concurrency.py`
  Abstractions over thread/process executors with bounded submission windows,
  plus helpers to derive executor settings from `SievioConfig`.
- `builder.py`
  Orchestrates config -> `PipelinePlan`/`PipelineRuntime` construction: builds
  sources, sinks, bytes handlers, HTTP client, lifecycle hooks, QC wiring, and
  language detectors. Hosts `PipelineOverrides` and `build_engine` for runtime
  overrides.
- `hooks.py`
  Built-in pipeline lifecycle hooks for run summaries/finalizers. Hosts
  `LanguageTaggingMiddleware` wired from configured detectors.
- `pipeline.py`
  Pipeline engine that runs the ingestion loop: coordinates sources, decoding,
  chunking, record creation, sinks, hooks, middleware, and statistics. Defines
  `PipelineEngine`, record/file middleware adapters, and helpers like
  `run_pipeline` and `apply_overrides_to_engine`.
- `qc_utils.py`
  Low-level quality-control utilities (similarity hashing, duplicate detection,
  basic QC heuristics and summaries).
- `qc_controller.py`
  Inline screening controller and related helpers for advisory/inline gating
  during the main pipeline run (e.g., `InlineQCHook`), coordinating quality and
  safety screeners with `score_record`.
- `qc_post.py`
  Post-hoc screening lifecycle hooks and JSONL drivers (`run_qc_over_jsonl`,
  `run_safety_over_jsonl`) for scoring or CSV export after the main run.
- `dataset_card.py`
  Builders for dataset card fragments and rendering/assembling Hugging Face style
  dataset cards from run statistics and metadata.
- `safe_http.py`
  Stdlib-only HTTP client with IP/redirect safeguards and a module-level global
  client helper for simple use cases.
- `dedup_store.py`
  SQLite-backed global deduplication store for MinHash LSH signatures, used by
  the default quality scorer and seeding scripts.
- `language_id.py`
  Central hub for language identification: extension maps, doc-format hints, and
  detector factories for optional backends.
- `extras/__init__.py`
  Namespace for optional extractors and helpers (no side effects on import).
- `extras/md_kql.py`
  Optional helpers to detect and extract KQL blocks from Markdown content.
- `extras/qc.py`
  Optional quality-control scorers and CSV writers (default
  `JSONLQualityScorer` + factory) with optional global MinHash dedup store.
- `extras/safety.py`
  Stdlib-only baseline safety scorer (`RegexSafetyScorer`) and default factory.
- `extras/langid_lingua.py`
  Optional Lingua-based human language detector implementing
  `LanguageDetector`.
- `extras/langid_pygments.py`
  Optional Pygments-backed code-language detector implementing
  `CodeLanguageDetector`.

POST safety semantics:
- `qc.safety.mode="post"` triggers a safety pass after the main pipeline
  finishes. The pass reads the primary JSONL without rewriting it, evaluates
  safety decisions (respecting `annotate_only` for gates), and merges the safety
  screener summary into `PipelineStats.qc`.

### 6.2 Top-level package - `src/sievio/`

- `__init__.py`
  Public package surface for `sievio`, re-exporting primary configuration types
  and helpers such as `convert`, `convert_local_dir`, and `convert_github`.

### 6.3 CLI - `src/sievio/cli/`

- `__init__.py`
  CLI package marker; no runtime logic.
- `main.py`
  Command-line entrypoint invoked by the `sievio` console script; parses args
  and dispatches to runner helpers.
- `runner.py`
  High-level orchestration helpers used by the CLI and library: builds profiles,
  constructs configs/output paths, invokes the builder/engine, and provides
  convenience wrappers.

### 6.4 Sources - `src/sievio/sources/`

- `__init__.py`
  Sources package marker; no runtime logic.
- `fs.py`
  Local filesystem source that walks repositories with gitignore support and
  streaming hints.
- `githubio.py`
  GitHub zipball source utilities: parsing specs/URLs, API helpers, download and
  iterate archive members.
- `sources_webpdf.py`
  Web PDF sources for lists of URLs or in-page PDF links with download/extract
  logic.
- `pdfio.py`
  PDF reading/extraction helpers used by web-PDF sources.
- `csv_source.py`
  CSV-backed source emitting text rows with configurable column selection.
- `jsonl_source.py`
  JSONL source that streams records from existing JSONL files (not registered
  by default; wire via a custom factory or helper).
- `parquetio.py`
  Parquet bytes handler for reading Parquet payloads into records when the
  `[parquet]` extra is installed.
- `sqlite_source.py`
  SQLite source for reading text columns from database files.
- `evtxio.py`
  Windows Event Log (EVTX) bytes handler for `.evtx` files when the `[evtx]`
  extra is installed.

### 6.5 Sinks - `src/sievio/sinks/`

- `__init__.py`
  Sinks package marker; no runtime logic.
- `sinks.py`
  Built-in sink implementations (JSONL, gzip JSONL, prompt text, no-op) and sink
  protocol helpers.
- `parquet.py`
  Parquet dataset sink for writing records to Parquet files.

### 6.6 Scripts - `scripts/`

Manual, developer-focused scripts for end-to-end testing and experiments (not
part of the public API).

- `scripts/manual_test_github.py`
  Manual smoke-test script to run a GitHub -> JSONL/Parquet pipeline.
- `scripts/manual_test_github_toml.py`
  Manual test driving GitHub conversion from a TOML config file.
- `scripts/manual_test_web_pdf.py`
  Manual test for web PDF ingestion and conversion through the pipeline.
- `scripts/seed_dedup_db.py`
  CLI helper for seeding a global MinHash deduplication SQLite DB from JSONL
  files; parameters must match QC MinHash settings.

### 6.7 Tests - `tests/`

Key tests to start with:
- `tests/test_builder_runtime_layering.py`
- `tests/test_pipeline_middlewares.py`
- `tests/test_qc_controller.py`
- `tests/test_qc_post.py`
- `tests/test_safe_http.py`
- `tests/test_config_builder_pipeline.py`
- `tests/test_concurrency.py`
- `tests/test_records.py`
- `tests/test_dataset_card.py`
- `tests/test_plugins_and_registries.py`

Full suite: see `tests/`.

### 6.8 Repository root

Top-level files are summarized in `project_files.md`. Key ones:

- `README.md`
  Project overview and usage instructions.
- `docs/TECHNICAL_MANUAL.md`
  Detailed installation, configuration, and operational notes.
- `AGENTS.md`
  Operational rules and invariants for AI coding assistants.
- `LLMS.md`
  This architecture and module-responsibility guide.
- `example_config.toml`, `manual_test_github.toml`
  Example `SievioConfig` TOML files for documentation and manual tests.
- `pyproject.toml`
  Packaging configuration, dependencies, and tool settings.

---

## 7. Trust boundaries

- All remote access must go through `SafeHttpClient` in
  `src/sievio/core/safe_http.py` (built via `HttpConfig.build_client()` in
  `src/sievio/core/config.py` or `make_http_client()` in
  `src/sievio/core/factories_context.py`).
- Remote-capable sources (`src/sievio/sources/githubio.py`,
  `src/sievio/sources/sources_webpdf.py`, `src/sievio/sources/sqlite_source.py`)
  must use the safe HTTP client and respect its redirect/IP safeguards.
- Timeouts, redirect limits, and allowed redirect suffixes live in `HttpConfig`.
- Config remains declarative; live clients and resources belong in runtime.

---

## 8. Common pitfalls

- Stashing live clients/scorers/sources in `SievioConfig` instead of runtime.
- Bypassing registries with ad-hoc wiring in builder or pipeline code.
- Confusing QC mode vs safety mode (especially `annotate_only`).
- Using record middleware for file-level decisions (or vice versa).
- Using process executors with non-picklable callables/resources.
- Writing output files outside sinks or mutating record schema in sinks.
- Reading raw `spec.options` instead of using
  `build_config_from_defaults_and_options`.

---

## 9. Maintenance contract

- Update this file whenever core modules, extension points, or key symbols are
  added, removed, or renamed.
- Verify references by grepping symbols and checking `project_files.md`.
- `rg` every symbol mentioned in Sections 2–7.
- Verify every referenced path exists.
- Keep this doc task-first and concise; link to README/technical manual for
  broader narrative.
- Reviewer checklist for core-wiring PRs: confirm registry usage, safe HTTP
  boundary, QC/safety semantics, and record schema compatibility.
