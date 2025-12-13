# Sievio – LLM Guide

This file gives LLMs a compact map of Sievio so they can work in the right modules and avoid touching core pipeline wiring by accident.

> Rename note: the project previously shipped as RepoCapsule. All imports/CLIs now use `sievio`; no compatibility shim exists. See `MIGRATION.md` for details.

For operational rules, invariants, and “how to ask an AI for help,” see `agents.md`.
This `llms.md` focuses on architecture, module responsibilities, and where to make
changes for different tasks.

---

## 1. High-level overview

Sievio is a **library-first, configuration-driven ingestion pipeline** that turns repositories and related artifacts into normalized JSONL / Parquet datasets suitable for LLM pre-training, fine-tuning and analysis.

At a high level:

* You describe a run with `SievioConfig` (Python or TOML).
* The **builder** turns that config into a `PipelinePlan` and `PipelineRuntime`.
* The **pipeline engine** coordinates sources → decode → chunk → record construction → sinks.
* Optional subsystems (QC, dataset cards, language ID, safety/QC extras, CLI runners, plugins) sit **around** the core rather than reimplementing it.

### Config normalization

Per-kind defaults and per-spec options should always be merged into typed dataclasses before constructing sources/sinks/QC objects. This keeps factories declarative, makes new config knobs automatically tunable, and avoids hand-merging option dicts. `build_config_from_defaults_and_options` is the canonical helper for this pattern.
- Factories should only pull constructor-only identifiers (paths/URLs/ids) directly from `spec.options` and rely on the helper for everything else (sources, sinks, QC).

---

## 2. Core vs non-core

For the purposes of this guide:

* **Core files** are a curated shortlist.

  * They define configs and record schemas, pipeline planning/runtime, registries, language/QC/safety wiring, dataset cards, and top-level orchestration.
  * They are explicitly marked as **CORE** in **3** (“Module map”) and should be treated as the architectural reference for new sources, sinks, scorers, and plugins.
  * LLMs should usually **read but not modify** core orchestration logic (builder, pipeline engine, registries, interfaces, factories, QC/dataset-card hooks, concurrency) unless a task explicitly calls for it.

* **Non-core files** are everything else.

  * These include CLI entry points, concrete source/sink implementations, optional extras/plugins, tests, scripts, and other integration glue.
  * Non-core modules are the **preferred place** to:
    * add or tweak behavior for a specific integration,
    * implement new sources/sinks/scorers/safety checks that *use* the core APIs,
    * and experiment with features without destabilizing the architecture. 
  * When working with an LLM coding assistant, non-core files may not always be shared in full. This `llms.md` exists to give enough high-level context (responsibilities, key entry points, relationships to core) to avoid duplicated work and accidental reimplementation.

---

## 3. Module map

### 3.1 Core package – `src/sievio/core/`

These are treated as the “core” of the system.

* `__init__.py`
  Core package initializer and exports for shared constants/helpers.

* `config.py`
  Configuration models and helpers for defining `SievioConfig` and related config sections (sources, sinks, QC, safety, chunking, etc.).

* `records.py`
  Construction and normalization of output record dictionaries, including run headers and consistent metadata fields. Also hosts schema-version helpers (`check_record_schema`) to warn when ingesting JSONL produced by mismatched library versions.

* `naming.py`
  Utilities for building safe, normalized output filenames and extensions based on config and repo context.

* `licenses.py`
  Helpers for detecting and normalizing license information from repositories and archives.

* `decode.py`
  Decoding bytes into normalized text using encoding detection and simple heuristics, plus helpers for text normalization.

* `chunk.py`
  Token-aware document/code chunking utilities and the `ChunkPolicy` machinery; yields chunk dicts for downstream processing.

* `convert.py`
  High-level helpers that take file inputs (paths/bytes), run decode + chunking, and produce extractor/record dictionaries.

* `factories.py`
  Facade re-exporting factory helpers split across sinks, sources, QC, and context modules.

* `factories_sinks.py`
  Sink and output-path factories (`OutputPaths`, JSONL/Prompt sinks, Parquet dataset sink).

* `factories_sources.py`
  Source factories (`LocalDirSourceFactory`, `GitHubZipSourceFactory`, etc.) and bytes-handler wiring (`make_bytes_handlers`, `UnsupportedBinary`).

* `factories_qc.py`
  QC and safety scorer factory helpers (`make_qc_scorer`, `make_safety_scorer`).

* `factories_context.py`
  Repo context inference and HTTP client construction helpers (`make_repo_context_from_git`, `make_http_client`).

* `log.py`
  Logging configuration helpers, package logger setup, and temporary level context manager.

* `registries.py`
  Registries for sources, sinks, bytes handlers, quality scorers, safety scorers, and lifecycle hooks; central place to register built-ins and plugins, with `RegistryBundle` helpers for passing registry sets into the builder or plugin loader.

* `plugins.py`
  Plugin discovery and registration helpers (entry-point based), wiring external sources/sinks/scorers into the registries.

* `sharding.py`
  Helpers to generate per-shard configs with isolated outputs from a base config + target list.

* `stats_aggregate.py`
  Utilities to merge multiple `PipelineStats.as_dict()` outputs (counts/flags only) for distributed runs; validates QC config consistency and clears non-additive QC fields like `signal_stats` and `top_dup_families`.

* `interfaces.py`
  Protocols/typed interfaces shared across the system (sources, sinks, lifecycle hooks, quality/safety scorers, etc.), plus core type aliases.

* `concurrency.py`
  Abstractions over thread/process executors with bounded submission windows, plus helpers to derive executor settings from `SievioConfig`.

* `builder.py`
  Orchestrates config → `PipelinePlan`/`PipelineRuntime` construction: builds sources, sinks, bytes handlers, HTTP client, lifecycle hooks, QC wiring, and language detectors from a `SievioConfig`. Hosts `PipelineOverrides` and `build_engine`, which is the preferred entry point for wiring runtime-only overrides (HTTP/QC/safety scorers, language detectors, bytes handlers, file extractor, and record/file middlewares) into a `PipelineEngine`. If you are changing how configs become runtime objects or how overrides/hooks/QC wiring are applied, start here.

* `hooks.py`
  Built-in pipeline lifecycle hooks for run summaries/finalizers used by the builder to assemble runtime hooks. Hosts `LanguageTaggingMiddleware`, which the builder instantiates from configured detectors and carries on `PipelineRuntime`.

* `pipeline.py`
  Pipeline engine that runs the ingestion loop: coordinates sources, decoding/chunking/record creation, sinks, hooks, middleware, and statistics. Defines `PipelineEngine`, record/file middleware adapters (`add_record_middleware`, `add_file_middleware` plus `_FuncRecordMiddleware`/`_FuncFileMiddleware`), and helpers like `run_pipeline` and `apply_overrides_to_engine` that turn a config + `PipelineOverrides` into a fully wired engine. The engine consumes record middlewares carried on `PipelineRuntime` (for example, the language-tagging middleware from `hooks.py`). If you need to adjust overall run flow or middleware behavior, this is usually the right place – but prefer adding middlewares/hooks over modifying the engine itself.

* `qc_utils.py`
  Low-level quality-control utilities (similarity hashing, duplicate detection, basic QC heuristics and summaries).

* `qc_controller.py`
  Inline screening controller and related helpers for advisory/inline gating during the main pipeline run (e.g., `InlineQCHook`), coordinating both quality and safety screeners with record-level scorers (`score_record`) only. Hosts `QualitySignals`/`filter_qc_meta` and `QCSummaryTracker.signal_stats` for schema-aligned QC signals (len_tok, ascii_ratio, repetition, gopher_quality, etc.). Safety gating is driven by `SafetyConfig.mode` + `annotate_only` (independent of QC mode) and can run without QC; POST safety is handled by a separate hook/driver after the pipeline completes.

* `qc_post.py`
  Post-hoc screening lifecycle hooks and reusable JSONL drivers (`run_qc_over_jsonl`, `run_safety_over_jsonl`, `iter/collect_qc_rows_from_jsonl`) for scoring or CSV export after the main run completes, including optional QC/safety signal sidecars (CSV or Parquet). `QCMode.POST` enforces gates only in this pass; POST safety evaluates decisions and updates summaries but does not rewrite the primary JSONL. If you are changing QC/safety export/sidecar behavior or running QC/safety as a separate pass over JSONL, start here instead of modifying sinks.

* `dataset_card.py`
  Builders for dataset card fragments and rendering/assembling Hugging Face–style dataset cards from run statistics and metadata.

* `safe_http.py`
  Stdlib-only HTTP client with IP/redirect safeguards and a module-level global client helper for simple use-cases.

* `extras/__init__.py`
  Namespace for optional or higher-level “extras” built on top of core primitives.
  *Summary pending for individual extras modules.*

* `extras/md_kql.py`
  Optional helpers to detect and extract KQL blocks from Markdown content.

* `extras/qc.py`
  Optional quality-control scorers and CSV writers used when QC extras are installed, including the default `JSONLQualityScorer` (SimHash + MinHash LSH) and its factory. Supports an optional global MinHash dedup store shared across processes.

* `extras/safety.py`
  Stdlib-only baseline safety scorer (`RegexSafetyScorer`) plus a default factory registered with the safety scorer registry.

### POST safety semantics

`qc.safety.mode="post"` triggers a safety pass after the main pipeline finishes. The pass reads the primary JSONL without rewriting it, evaluates safety decisions (respecting `annotate_only` for gates), and merges the safety screener summary into `PipelineStats.qc` alongside legacy safety counters. Optional outputs include a compact CSV report (`_safety.csv` suffix by default) and an optional signals sidecar (`_safety_signals.(csv|parquet)`), derived from streamed scoring results rather than mutating records in place.

Example configuration:

```toml
[qc]
enabled = false

[qc.safety]
enabled = true
mode = "post"
write_csv = true
```

* `dedup_store.py`
  SQLite-backed global deduplication store for MinHash LSH signatures. Enforces LSH parameters (`n_perm`, `bands`, `jaccard_threshold`) via a `metadata` table and is designed to be used by `JSONLQualityScorer` (and the `seed_dedup_db.py` script) for cross-process/global near-duplicate tracking.

* `language_id.py`
  Central hub for language identification: extension maps (`CODE_EXTS`, `DOC_EXTS`, `EXT_LANG`), doc format hints, classify helpers, baseline human/code detectors, and factory functions to load optional backends (`extras/langid_lingua.py`, `extras/langid_pygments.py`). It is the single source of truth for code/doc language tags.

* `extras/langid_lingua.py`
  Optional Lingua-based human language detector implementing the `LanguageDetector` protocol; loaded via `make_language_detector("lingua")`.

* `extras/langid_pygments.py`
  Optional Pygments-backed code-language detector implementing the `CodeLanguageDetector` protocol; loaded via `make_code_language_detector("pygments")`.

> **Note:** Additional core modules under `core/` that are not listed here yet should be added with short summaries as they are introduced or become relevant.

---

### 3.2 Top-level package – `src/sievio/`

* `__init__.py`
  Defines the public package surface for `sievio`, re-exporting primary configuration types (e.g., `SievioConfig`) and high-level helpers such as `convert`, `convert_local_dir`, and `convert_github`. Non-exported symbols are considered expert/unstable.

---

### 3.3 CLI – `src/sievio/cli/`

* `__init__.py`
  CLI package initializer.
  *Summary pending (likely minimal; may just expose entrypoints or re-export runner helpers).*

* `main.py`
  Command-line entrypoint module invoked by the `sievio` console script; responsible for parsing CLI arguments and dispatching to runner helpers.
  *Summary pending.*

* `runner.py`
  High-level orchestration helpers used by the CLI and library:

  * Builds GitHub/local/web-PDF repo profiles.
  * Constructs configs/output paths using the core factories facade.
  * Invokes the builder and pipeline engine to execute runs.
  * Provides convenience wrappers (e.g., `convert_github`, `convert_local_dir`, `convert_web_pdf`).

---

### 3.4 Sources – `src/sievio/sources/`

* `__init__.py`
  Source package initializer.
  *Summary pending for individual source modules.*

* `fs.py`
  Local filesystem source that walks repositories with gitignore support and streaming hints.

* `githubio.py`
  GitHub zipball source utilities: parsing specs/URLs, API helpers, download/iterate archive members.

* `sources_webpdf.py`
  Web PDF sources for lists of URLs or in-page PDF links with download/extract logic.

* `pdfio.py`
  PDF reading/extraction helpers used by web-PDF sources.

* `csv_source.py`
  CSV-backed source emitting text rows with configurable column selection.

* `jsonl_source.py`
  JSONL source that streams records from existing JSONL files.

* `parquetio.py`
  Parquet source for ingesting Parquet datasets as records.

* `sqlite_source.py`
  SQLite source for reading text columns from database files.

* `evtxio.py`
  Windows Event Log (EVTX) source for ingesting event records.

---

### 3.5 Sinks – `src/sievio/sinks/`

* `__init__.py`
  Sink package initializer.
  *Summary pending for individual sink modules.*

* `sinks.py`
  Built-in sink implementations (JSONL, gzip JSONL, prompt text, no-op) and sink protocol helpers.

* `parquet.py`
  Parquet dataset sink for writing records to Parquet files.

---

### 3.6 Scripts – `scripts/`

Manual, developer-focused scripts for end-to-end testing and experiments (not part of the public API).

* `scripts/manual_test_github.py`
  Manual smoke-test script to run a GitHub → JSONL/Parquet pipeline using hard-coded parameters.

* `scripts/manual_test_github_toml.py`
  Manual test driving GitHub conversion from a TOML config file.

* `scripts/manual_test_web_pdf.py`
  Manual test for web PDF ingestion and conversion through the pipeline.

* `scripts/seed_dedup_db.py`
  CLI helper for seeding a global MinHash deduplication SQLite DB from existing JSONL/JSONL.GZ files. Uses the same doc_id and MinHash logic as `JSONLQualityScorer` and writes banded LSH data into `core/dedup_store.py`’s schema; parameters must match your QC scorer’s MinHash settings.

---

### 3.7 Tests – `tests/`

Pytest-based test suite validating core behavior and non-core wiring.

* `tests/test_builder_runtime_layering.py`
  Tests around separation of config vs runtime (`PipelinePlan`, `PipelineRuntime`) and layering behavior in the builder.

* `tests/test_chunk.py`
  Unit tests for chunking logic (`ChunkPolicy`, `iter_chunk_dicts`, token limits, etc.).

* `tests/test_cli_main.py`
  Tests for the CLI entrypoint (`cli.main`) and argument handling.

* `tests/test_concurrency.py`
  Tests for `concurrency.py` executors, bounded submission, and configuration via `SievioConfig`.

* `tests/test_config_builder_pipeline.py`
  End-to-end tests tying together config parsing, builder wiring, and pipeline execution.

* `tests/test_config_merge_helper.py`
  Tests the config/options merge helper (`build_config_from_defaults_and_options`) used by factories.

* `tests/test_convert.py`
  Tests for `convert.py` helpers (decode + chunk → records) and mode/format detection.

* `tests/test_convert_integration.py`
  Integration tests covering decode → chunk → record flows under representative configs.

* `tests/test_dataset_card.py`
  Tests dataset card generation and rendering behavior.

* `tests/test_decode.py`
  Tests for byte decoding, encoding detection, and normalization rules.

* `tests/test_decode_fallback.py`
  Tests fallback decode heuristics/codecs when preferred paths are unavailable.

* `tests/test_dedup_store.py`
  Tests deduplication store schema/metadata enforcement and LSH lookups.

* `tests/test_factories_sources_config_overlays.py`
  Tests source config overlay behavior (defaults vs per-spec options).

* `tests/test_hooks.py`
  Tests lifecycle hook wiring and behavior.

* `tests/test_log_and_naming.py`
  Tests for logging helpers and naming utilities (`naming.py`).

* `tests/test_pipeline_middlewares.py`
  Focused tests for `PipelineEngine` middleware behavior (record/file middleware adapters, QC hooks) and basic sink error handling. Middleware-related override wiring should be covered here when adding new behavior to `PipelineOverrides` or `build_engine`.

* `tests/test_plugins_and_registries.py`
  Tests plugin discovery and registry behavior for sources, sinks, and scorers.

* `tests/test_qc_controller.py`
  Tests inline QC controller behavior and interaction with the pipeline.

* `tests/test_qc_defaults.py`
  Tests default QC configuration and behavior (e.g., default scorers, thresholds).

* `tests/test_qc_integration.py`
  Integration tests that run QC wiring end-to-end through the pipeline.

* `tests/test_qc_post.py`
  Tests post-hoc QC driver behavior (`PostQCHook`, `run_qc_over_jsonl`).

* `tests/test_qc_safety_modes.py`
  Tests safety scorer modes (inline/advisory) and their interaction with QC and summary stats.

* `tests/test_qc_simhash_optimization.py`
  Tests QC SimHash/MinHash optimization and dedup behaviors.

* `tests/test_qc_utils.py`
  Tests low-level QC utilities (similarity hashing, duplicate detection, heuristics).

* `tests/test_records.py`
  Tests record construction, metadata propagation, and header generation.

* `tests/test_runner_finalizers.py`
  Tests that sinks and finalizers run correctly at the end of a pipeline run (run-summary, prompt files, etc.).

* `tests/test_safe_http.py`
  Tests `safe_http.py` HTTP client behavior, redirect/IP safety, and global client helpers.

* `tests/test_schema_validation.py`
  Tests config/schema validation behavior.

* `tests/test_sharding.py`
  Tests sharding helpers and config splitting logic.

* `tests/test_sqlite_source_security.py`
  Tests SQLite source download/allowlist safeguards and security constraints.

* `tests/test_sqlite_source_validation.py`
  Tests SQLite source validation rules.

* `tests/test_stats_aggregate.py`
  Tests stats aggregation behavior.

---

### 3.8 Repository root

Top-level files are summarized in `project_files.md`. Key ones:

* `README.md`
  Human-facing project overview and usage instructions.

* `agents.md`
  Operational rules and invariants for AI coding assistants.

* `llms.md`
  This architecture and module-responsibility guide.

* `example_config.toml`, `manual_test_github.toml`
  Example `SievioConfig` TOML files for documentation and manual tests.

* `pyproject.toml`
  Packaging configuration, dependencies, and project metadata.

---

### 3.9 Notes for future expansion

* As new modules are added (especially under `src/sievio/core/` and `src/sievio/sources` / `sinks`), they should be appended here with:

  * One-line purpose summary.
  * How they depend on or extend existing core modules.
* For planned-but-not-yet-implemented modules, add entries like:

  ```markdown
  - `src/sievio/core/FOO.py`  
    Summary pending – planned module for …
  ```

  and update them once the implementation lands.
