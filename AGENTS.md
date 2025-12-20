# AGENTS.md – Guide for AI Coding Assistants

This file is for AI coding assistants (e.g. ChatGPT) working on this repo.
It is intentionally named `AGENTS.md` so agent tooling can auto-discover it.

The project is a **library-first, configuration-driven ingestion pipeline** that turns repositories and related artifacts into normalized datasets (JSONL/Parquet) for LLM pre-training, fine-tuning or analysis. A config describes a run, the builder turns it into a plan/runtime, and the pipeline engine coordinates sources → decode → chunk → records → sinks, with optional QC and dataset cards layered around the core.

For deeper architecture and module-by-module descriptions, see `LLMS.md`.
For the canonical file tree (what files and tests exist), see `project_files.md`.
Use this `AGENTS.md` as the primary rules and “how to work safely” guide.

---

## 1. Setup & core commands

Treat this section as the minimal "how to run checks" reference. Full installation and configuration details now live in `docs/TECHNICAL_MANUAL.md` (see `README.md` for the overview).

### Environment

* Python: 3.11+
* Use the project’s virtualenv for all commands (for example):

```bash
# From the repo root
source .venv/bin/activate
```

* Install (dev mode, with common extras):

```bash
pip install -e ".[dev,tok,pdf,evtx,parquet,qc]"
```

### Build / test / checks

Run these before proposing changes:

```bash
# Run tests
PYTHONPATH=src pytest

# Lint & style
ruff check .

# Type-check
mypy --config-file pyproject.toml src
```

Targeted test patterns:

```bash
# Run a single test file
PYTHONPATH=src pytest tests/test_pipeline_middlewares.py

# Run a single test function
PYTHONPATH=src pytest tests/test_pipeline_middlewares.py::test_record_middleware_adapter_sets_tag

# Run a keyword subset
PYTHONPATH=src pytest -k "qc and safety"
```

---

## 2. Mental model & project structure

### Core vs non-core

- **Core** (`src/sievio/core/`): config models, builder, pipeline engine, registries, QC, dataset cards, HTTP safety, logging, concurrency. Other code should **use these**, not reimplement them.
- **Non-core**:
  - CLI and runner helpers: `src/sievio/cli/`
  - Public package surface: `src/sievio/__init__.py`
  - Source/sink implementations: `src/sievio/sources/`, `src/sievio/sinks/`
  - Extras (optional modules): `src/sievio/core/extras/`
  - Tests: `tests/`

### Golden "spine" modules

These hold the main invariants and are more sensitive to changes:

- `core/config.py` – config models and options.
- `core/registries.py` – registries for sources, sinks, bytes handlers, quality/safety scorers.
- `core/factories_*.py` – construction of sources, sinks, QC, context/HTTP.
- `core/builder.py` – config → `PipelinePlan` / `PipelineRuntime`.
- `core/pipeline.py` – pipeline engine.
- `core/qc_utils.py`, `core/qc_controller.py`, `core/qc_post.py` – QC utilities, inline QC controller, post-hoc QC driver.
- `core/dedup_store.py` – SQLite-backed global MinHash LSH store used by QC extras and seeding scripts.
- `core/dataset_card.py` – dataset card fragments and rendering.
- `core/safe_http.py` – stdlib-only HTTP client with IP/redirect safeguards.
- `core/sharding.py`, `core/stats_aggregate.py` – sharded-run helpers (config splitting, stats merging).

When in doubt, prefer changing **non-core** code and wiring via registries/factories rather than patching these core modules.

### Config normalization

Per-kind defaults and per-spec options should converge into small dataclasses before construction so factories stay thin. This keeps sources/sinks/QC/scorers consistent and makes new config fields usable without touching factory code. Treat `build_config_from_defaults_and_options` as the canonical merge helper and prefer adding defaults in `[sources.defaults]` / `[sinks.defaults]` over ad-hoc lookups.
- Factories should only read constructor-only identifiers (paths/URLs/ids) directly from `spec.options`; everything else should flow through `build_config_from_defaults_and_options` so defaults + overrides remain aligned.

---

## 3. Working agreements (do-not-break invariants)

1. **Use registries, not ad-hoc wiring**
   - Register new Sources/Sinks/Scorers via the appropriate registry (e.g., `SourceRegistry`, `SinkRegistry`, `BytesHandlerRegistry`, `QualityScorerRegistry`) or plugins, rather than hard-coded conditionals.

2. **Keep config declarative**
   - Config objects should not hold live clients, scorers, or open resources. Runtime objects belong in the plan/runtime, not in config models.

3. **HTTP safety**
   - All remote access (GitHub, PDFs, downloads, APIs) must go through the safe HTTP client/wrapper (e.g., `SafeHttpClient`), not ad-hoc `requests`/`urllib` calls.

4. **QC as a layer, not baked-in**
   - New QC logic should be expressed as scorers and heuristics, not embedded inside sources/sinks. Use the QC interfaces and registries. Safety follows `SafetyConfig.mode` + `annotate_only` independently of `QCConfig.mode` (inline safety can run even when QC is post-only; POST safety runs after the pipeline to update summaries and optional reports without rewriting JSONL).
   - Global MinHash deduplication is implemented as an optional SQLite store (`core/dedup_store.GlobalDedupStore`) used by the default scorer when configured via `qc.scorer_options.global_dedup`. Do not reimplement LSH or store logic in sources/sinks; reuse the existing scorer + store APIs.

5. **Sinks own output layout**
   - Sinks decide file naming, compression, and sidecar artifacts. Avoid scattering file-writing logic into unrelated modules.

6. **Metadata is stable and additive**
   - Preserve existing record metadata fields and keys. Add new fields without breaking existing consumers (sinks, dataset cards, QC tooling).

7. **Small, focused diffs**
   - Prefer minimal, task-scoped changes. Avoid repo-wide refactors unless explicitly requested.

8. **No new dependencies by default**
   - Do not introduce new third-party packages unless explicitly requested or clearly required.

9. **Truncation / missing context**
   - If code or output looks truncated or scrambled, do not guess. Ask for the smallest missing snippet or file region instead of inventing content.

### Review strategy (for agents doing code review)

1. Fast pass: map the architecture, data flow, and trust boundaries using `LLMS.md` and the core modules.
2. Deep pass: focus on the highest-risk or most central files (pipeline, registries, safety/QC, sinks/sources involved in the task).

See “Review guidelines” for P0/P1 risk classification and required checks.

---

## 4. Review guidelines

### P0 / P1 severity

P0 (must be called out in review and only changed with explicit scope):
- HTTP safety boundary: `SafeHttpClient`, `safe_http.py`, `make_http_client`, or any bypass of safe HTTP usage.
- QC/safety gating semantics: `QCMode`, `SafetyConfig.annotate_only`, inline vs post hooks (`InlineQCHook`, `PostQCHook`), and `score_record` decisions.
- Record schema/metadata compatibility: `core/records.py`, QC metadata fields, or summary schema changes.
- Per-record loop/middleware ordering: `PipelineEngine`, `apply_overrides_to_engine`, `record_middlewares`, `file_middlewares`.

P1 (must be called out with risk notes and test evidence):
- New network access paths or remote sources.
- Dependency additions or new extras.
- Output layout changes in sinks that affect downstream consumers.

### Always verify

- `PYTHONPATH=src pytest`
- `ruff check .`
- `mypy --config-file pyproject.toml src`
- Schema remains additive, and Safe HTTP + QC/safety modes keep their existing semantics.

## 5. Code style

Source of truth: `pyproject.toml` under `[tool.ruff]` (line length 100; select `E`, `F`, `I`, `B`, `UP`).
- Run `ruff check .` (includes import sorting via `I`).
- No formatter is configured in-repo; avoid broad reformatting. If a formatter is added, follow the repo docs/`pyproject.toml`.
- Keep changes minimal and avoid whitespace churn.

## 6. Git workflow

- Follow the repo’s documented flow (README / `docs/CONTRIBUTING.md`): feature branch, commit, PR.
- PR summaries MUST include intent, key files touched, tests run (commands), and P0/P1 risk notes when applicable.
- Avoid incidental formatting or unrelated cleanup.

## 7. Hard boundaries

DO NOT change without explicit instruction:
- Generated or indexed docs: `docs/CONFIGURATION.md`, `project_files.md` (regenerate via scripts if needed).
- Core orchestration “golden spine” modules (`core/config.py`, `core/registries.py`, `core/factories_*.py`, `core/builder.py`, `core/pipeline.py`, `core/qc_*.py`, `core/records.py`, `core/safe_http.py`).
- Safe HTTP boundary: no new network access or direct `requests`/`urllib` usage.
- Secrets/PII: never commit secrets or log PII; avoid adding debug logging in hot paths.
- Lockfiles or build artifacts if added (e.g., `poetry.lock`, `uv.lock`, `dist/`, `build/`).
- Run outputs (JSONL/Parquet/*.card.json) should be regenerated, not edited by hand.

Directory-scoped overrides use `AGENTS.override.md` only when explicitly introduced.

## 8. Definition of done for agent-generated changes

A change is "done" when:

- [ ] Tests pass: `PYTHONPATH=src pytest`
- [ ] Lint and type-check are clean: `ruff check .`, `mypy --config-file pyproject.toml src`
- [ ] No core invariants are broken (registries, Safe HTTP, config semantics, metadata shape)
- [ ] Relevant docs/examples updated (`README.md`, `LLMS.md`, configs if needed)
- [ ] Diff is small, focused, and clearly explainable in a commit/PR message

---

## 9. Golden paths (common tasks)

### 9.1 New source type

1. Implement a `Source` under `src/sievio/sources/` (see `LLMS.md` §3.4).
2. Add a `SourceFactory` and register it with the source registry.
3. Add or update config examples to show usage.

### 9.2 New sink or output format

1. Implement a `Sink` under `src/sievio/sinks/` (see `LLMS.md` §3.5).
2. Add a `SinkFactory` and register it with the sink registry.
3. Add or update config examples and docs.

### 9.3 New bytes handler (binary format)

1. Implement a bytes handler + sniff function and wire it via the bytes handler registry (see `LLMS.md` §3.1 for factories/registries).
2. Reuse existing decode/chunk/record helpers where possible.
3. Add tests or examples for the new format.

### 9.4 Change or extend QC behavior

1. Start from the QC config/utilities and scorer interfaces (see `LLMS.md` §3.1 QC modules).
2. Implement or update a `QualityScorer` and register it via the QC registry.
3. Use inline QC controller and/or post-QC driver as appropriate, plus tests.

### 9.5 Dataset cards

1. Update `src/sievio/core/dataset_card.py` fragments/merging logic (see `LLMS.md` §3.1 dataset card).
2. Ensure pipeline stats feed any new signals into card fragments.
3. Keep output compatible with HF dataset card conventions (YAML front matter + Markdown).

### 9.6 Sharded runs / distributed execution

1. Show agents the sharding/stats helpers: `core/sharding.py`, `core/stats_aggregate.py`, configs in `core/config.py`, pipeline wiring in `core/pipeline.py`, QC summaries in `core/qc_controller.py`.
2. Ask the agent to generate shard configs via the CLI (`sievio shard ...`) or by calling `generate_shard_configs`.
3. After shards finish, merge stats JSON files with `sievio merge-stats` or `merge_pipeline_stats([...])`; combine JSONL/prompt outputs with `cat`/`zcat` or downstream tooling.

---

## 10. How humans should ask agents for help

When asking an AI assistant to change this repo, provide:

1. **Task** – bugfix / refactor / new feature / review.
2. **Constraints** – e.g., "stdlib only", "no new deps", "keep behavior backwards compatible".
3. **Scope** – files or modules you believe are involved.
4. **Evidence** – failing tests, stack traces, sample inputs/outputs, config snippets.
5. **Desired output** – patch/diff, review notes, test plan, or design sketch.

Example:

> Task: Add a new DB-like source that reads from a different backend.
> Constraints: stdlib-only if possible; no changes to the pipeline engine.
> Scope: `core/interfaces.py`, `sources/...`, registries.
> Evidence: config snippet and failing test.
> Desired output: patch + tests.

Agents should:
- Start with a fast, top-down scan (architecture, core modules, config impact).
- Then deep dive into the smallest set of files needed to implement the change safely.
