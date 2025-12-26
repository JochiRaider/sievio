# Sievio

Sievio is a **library-first, configuration-driven ingestion pipeline** that converts repositories and related artifacts into **normalized datasets** (typically JSONL, optionally other formats) suitable for LLM pre-training, fine-tuning, and analysis.

It is designed to be:
- **Composable**: extend via registries/plugins/overrides instead of editing the engine loop.
- **Operationally safe**: remote access is intended to flow through a hardened HTTP boundary.
- **Data-quality aware**: optional QC + safety screening can run inline or post-hoc.

---

## How it works

Sievio’s “spine” is intentionally simple:

**Config → Builder (plan/runtime) → Pipeline engine (sources → decode → chunk → records → sinks)**

- **Sources** enumerate inputs (local repos, GitHub repos, web PDFs, etc.).
- **Decode** normalizes bytes to text (with guardrails for odd encodings).
- **Chunk** splits text into model-friendly segments.
- **Records** are emitted with `"text"` plus `"meta"` for provenance and run metadata.
- **Sinks** own output layout (file naming, sidecars, compression, additional artifacts).

---

## Quickstart

### 1) Install dev tooling (uv)
If you’re using `uv` and have `mkdocs` in your `dev` extra:

```bash
uv sync --extra dev
```

### 2) Run the docs site

```bash
uv run mkdocs serve
```

### 3) Run a minimal conversion (Python API)

#### Convert a local directory

```python
from sievio import convert_local_dir

stats = convert_local_dir(
    root_dir="path/to/repo",
    out_jsonl="out/repo.jsonl",
)
print(stats)
```

#### Convert a GitHub repository

```python
from sievio import convert_github

stats = convert_github(
    url="https://github.com/OWNER/REPO",
    out_jsonl="out/repo.jsonl",
)
print(stats)
```

#### Config-driven run

If you want full control, build a `SievioConfig` (or load one from TOML/JSON) and run it:

```python
from sievio import load_config_from_path, convert

cfg = load_config_from_path("example_config.toml")
stats = convert(cfg)
print(stats)
```

---

## What you get: outputs and record shape

Sievio’s primary outputs are dataset artifacts produced by your configured sinks. In most workflows this includes:

* A **primary JSONL** file with one record per chunk.
* Optional **prompt text** output (useful for inspection or certain fine-tuning formats).
* Optional sidecar artifacts depending on sinks and QC settings.

The guiding invariant is that metadata should be **stable and additive**: new metadata fields may be introduced, but existing fields should remain usable by downstream consumers.

---

## Quality Control and Safety

Sievio supports quality and safety screening as a **layer** around the core extraction pipeline.

Common patterns:

* **Inline QC/safety**: annotate (or optionally filter) records as they’re produced.
* **Post-hoc QC/safety**: rescore an existing JSONL after extraction, producing reports/diagnostics without re-running sources.

If you are introducing new quality/safety logic, prefer implementing it as:

* a scorer (registered via the appropriate registry or plugin), and/or
* a middleware/hook that annotates or filters records

…instead of embedding heuristics into sources or sinks.

See: **[QUALITY_CONTROL.md](QUALITY_CONTROL.md)**

---

## Extending Sievio

Sievio is meant to be extended without “forking the engine.” Prefer these extension points:

* **Registries**: register new Sources, Sinks, bytes handlers, or scorers.
* **Plugins**: package reusable registrations behind an entry point.
* **Overrides**: swap runtime wiring (HTTP client, scorers, extractors, middlewares) for a single run.
* **Middlewares & hooks**: add cross-cutting behaviors (tagging, filtering, sidecar artifacts) without modifying sinks or the engine.

If you are new to the internals, start with the architecture map:

* **[TECHNICAL_MANUAL.md](TECHNICAL_MANUAL.md)**

---

## Built-in capabilities (high level)

Out of the box, Sievio supports workflows such as:

* Local directory ingestion (git-aware context when available)
* GitHub repository ingestion (zipball-based)
* Web PDF ingestion patterns (depending on enabled sources/extras)
* JSONL output (plus optional prompt text output)
* Optional extras for tokenization, PDF extraction, EVTX handling, parquet, QC/safety scoring (when installed)

The authoritative list of knobs and defaults lives in:

* **[CONFIGURATION.md](CONFIGURATION.md)**

---

## Cookbook

Task-oriented, copy/paste friendly guides live under `cookbook/`:

* **[Cookbook overview](cookbook/README.md)**
* **[Custom PII scrubbing](cookbook/custom_pii_scrubbing.md)**
* **[Dedup post QC](cookbook/dedup_post_qc.md)**
* **[PDF ingestion](cookbook/pdf_ingestion.md)**

---

## Contributing and operations

* Contribution workflow and expectations:

  * **[CONTRIBUTING.md](CONTRIBUTING.md)**
* Deployment and publishing patterns:

  * **[DEPLOYMENT.md](DEPLOYMENT.md)**

Recommended local checks:

```bash
uv run pytest
uv run ruff check .
uv run mypy --config-file pyproject.toml src
```

---

## Where to go next

If you’re trying to:

* **Configure a run** → **[CONFIGURATION.md](CONFIGURATION.md)**
* **Understand architecture / extension points** → **[TECHNICAL_MANUAL.md](TECHNICAL_MANUAL.md)**
* **Add QC / safety scoring** → **[QUALITY_CONTROL.md](QUALITY_CONTROL.md)**
* **Ship docs / deploy** → **[DEPLOYMENT.md](DEPLOYMENT.md)**
* **Find worked examples** → **[cookbook/README.md](cookbook/README.md)**