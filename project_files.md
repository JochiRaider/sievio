# Project File Tree

This file is a read-only index of files and tests in the repo for AI assistants
and humans. It does not contain source; use it to decide which files to open.

For architecture and module descriptions, see `LLMS.md`.
For rules, invariants, and AI usage guidelines, see `AGENTS.md`.

```text
.
├── AGENTS.md
├── CITATION.cff
├── CONTRIBUTING.md
├── LICENSE
├── LLMS.md
├── README.md
├── SECURITY.md
├── accel
│   ├── AGENTS.md
│   ├── Cargo.lock
│   ├── Cargo.toml
│   ├── LLMS.md
│   ├── README.md
│   ├── pyproject.toml
│   ├── python
│   │   └── sievio_accel
│   │       ├── __init__.py
│   │       └── qc.py
│   └── src
│       └── lib.rs
├── core_upload.md
├── docs
│   ├── CONFIGURATION.md
│   ├── CONTRIBUTING.md
│   ├── DEPLOYMENT.md
│   ├── QUALITY_CONTROL.md
│   ├── README.md
│   ├── TECHNICAL_MANUAL.md
│   ├── cookbook
│   │   ├── README.md
│   │   ├── custom_pii_scrubbing.md
│   │   ├── dedup_post_qc.md
│   │   └── pdf_ingestion.md
│   └── index.md
├── example_config.toml
├── manual_test_github.toml
├── mkdocs.yml
├── project_files.md
├── py.typed
├── pyproject.toml
├── sample.jsonl
├── scripts
│   ├── generate_configuration_md.py
│   ├── manual_test_github.py
│   ├── manual_test_github_toml.py
│   ├── manual_test_web_pdf.py
│   └── seed_dedup_db.py
├── src
│   └── sievio
│       ├── __init__.py
│       ├── cli
│       │   ├── __init__.py
│       │   ├── main.py
│       │   └── runner.py
│       ├── core
│       │   ├── AGENTS.md
│       │   ├── __init__.py
│       │   ├── accel.py
│       │   ├── builder.py
│       │   ├── chunk.py
│       │   ├── concurrency.py
│       │   ├── config.py
│       │   ├── convert.py
│       │   ├── dataset_card.py
│       │   ├── decode.py
│       │   ├── dedup_store.py
│       │   ├── extras
│       │   │   ├── __init__.py
│       │   │   ├── langid_lingua.py
│       │   │   ├── langid_pygments.py
│       │   │   ├── md_kql.py
│       │   │   ├── qc.py
│       │   │   └── safety.py
│       │   ├── factories.py
│       │   ├── factories_context.py
│       │   ├── factories_qc.py
│       │   ├── factories_sinks.py
│       │   ├── factories_sources.py
│       │   ├── hooks.py
│       │   ├── interfaces.py
│       │   ├── language_id.py
│       │   ├── licenses.py
│       │   ├── log.py
│       │   ├── naming.py
│       │   ├── pipeline.py
│       │   ├── plugins.py
│       │   ├── qc_controller.py
│       │   ├── qc_post.py
│       │   ├── qc_utils.py
│       │   ├── records.py
│       │   ├── registries.py
│       │   ├── registry_callables.py
│       │   ├── runner.py
│       │   ├── safe_http.py
│       │   ├── sharding.py
│       │   └── stats_aggregate.py
│       ├── sinks
│       │   ├── AGENTS.md
│       │   ├── __init__.py
│       │   ├── parquet.py
│       │   └── sinks.py
│       └── sources
│           ├── AGENTS.md
│           ├── __init__.py
│           ├── csv_source.py
│           ├── evtxio.py
│           ├── fs.py
│           ├── githubio.py
│           ├── jsonl_source.py
│           ├── parquetio.py
│           ├── pdfio.py
│           ├── sources_webpdf.py
│           └── sqlite_source.py
├── tests
│   ├── conftest.py
│   ├── test_builder_runtime_layering.py
│   ├── test_callable_registries.py
│   ├── test_chunk.py
│   ├── test_cli_main.py
│   ├── test_concurrency.py
│   ├── test_config_builder_pipeline.py
│   ├── test_config_merge_helper.py
│   ├── test_convert.py
│   ├── test_convert_integration.py
│   ├── test_dataset_card.py
│   ├── test_dataset_card_qc.py
│   ├── test_decode.py
│   ├── test_decode_fallback.py
│   ├── test_dedup_store.py
│   ├── test_factories_sources_config_overlays.py
│   ├── test_hooks.py
│   ├── test_jsonl_source.py
│   ├── test_language_customization.py
│   ├── test_log_and_naming.py
│   ├── test_pipeline_error_rate.py
│   ├── test_pipeline_middlewares.py
│   ├── test_pipeline_streaming.py
│   ├── test_plugins_and_registries.py
│   ├── test_qc_controller.py
│   ├── test_qc_defaults.py
│   ├── test_qc_integration.py
│   ├── test_qc_parallel_inline.py
│   ├── test_qc_post.py
│   ├── test_qc_safety_modes.py
│   ├── test_qc_simhash_optimization.py
│   ├── test_qc_utils.py
│   ├── test_qc_utils_accel_parity.py
│   ├── test_qc_utils_accel_shim.py
│   ├── test_records.py
│   ├── test_runner_finalizers.py
│   ├── test_safe_http.py
│   ├── test_schema_validation.py
│   ├── test_sharding.py
│   ├── test_sources_fs.py
│   ├── test_sqlite_source_security.py
│   ├── test_sqlite_source_validation.py
│   └── test_stats_aggregate.py
└── uv.lock
```
