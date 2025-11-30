# Project File Tree

This file is a read-only index of files and tests in the repo for AI assistants
and humans. It does not contain source; use it to decide which files to open.

For architecture and module descriptions, see `llms.md`.
For rules, invariants, and AI usage guidelines, see `agents.md`.

```text
.
├── .gitattributes
├── .gitignore
├── README.md
├── agents.md
├── example_config.toml
├── llms.md
├── manual_test_github.toml
├── project_files.md
├── pyproject.toml
├── sample.jsonl
├── scripts
│   ├── manual_test_github.py
│   ├── manual_test_github_toml.py
│   └── manual_test_web_pdf.py
├── src
│   └── repocapsule
│       ├── __init__.py
│       ├── cli
│       │   ├── __init__.py
│       │   ├── main.py
│       │   └── runner.py
│       ├── core
│       │   ├── __init__.py
│       │   ├── builder.py
│       │   ├── chunk.py
│       │   ├── concurrency.py
│       │   ├── config.py
│       │   ├── convert.py
│       │   ├── dataset_card.py
│       │   ├── decode.py
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
│       │   └── safe_http.py
│       ├── sinks
│       │   ├── __init__.py
│       │   ├── parquet.py
│       │   └── sinks.py
│       └── sources
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
└── tests
    ├── conftest.py
    ├── test_builder_runtime_layering.py
    ├── test_chunk.py
    ├── test_cli_main.py
    ├── test_concurrency.py
    ├── test_config_builder_pipeline.py
    ├── test_convert.py
    ├── test_dataset_card.py
    ├── test_decode.py
    ├── test_hooks.py
    ├── test_log_and_naming.py
    ├── test_pipeline_middlewares.py
    ├── test_plugins_and_registries.py
    ├── test_qc_controller.py
    ├── test_qc_defaults.py
    ├── test_qc_post.py
    ├── test_records.py
    ├── test_runner_finalizers.py
    └── test_safe_http.py
```
