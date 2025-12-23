# Sievio

## Purpose & audience
This manual is an **operator-focused runbook** for running, tuning, and debugging Sievio pipelines (CLI or Python API). It is intentionally **behavior-centric**: it explains what happens at runtime and how to operate it safely at scale.

For developer internals (module/file map, where changes should live), use `../LLMS.md`. For contribution workflows and extension points, use `../CONTRIBUTING.md`. For the full, generated config schema, use `CONFIGURATION.md`.

## Where to go next
- Docs index: `README.md`
- Configuration reference (generated): `CONFIGURATION.md` (do not edit by hand)
- QC deep dive: `QUALITY_CONTROL.md`
- Deployment/sharding: `DEPLOYMENT.md`
- Cookbook recipes: `cookbook/`
- Architecture/module map (authoritative): `../LLMS.md`
- Contributor guide: `CONTRIBUTING.md` (stable link to repo-root `../CONTRIBUTING.md`)

## Table of contents
- [Installation](#installation)
- [Quickstart](#quickstart)
- [CLI](#cli)
- [Concepts & architecture](#concepts--architecture)
- [Quality control (QC)](#quality-control-qc)
- [Dataset cards (Hugging Face style)](#dataset-cards-hugging-face-style)
- [Sharding & distributed runs](#sharding--distributed-runs)
- [Configuration reference](#configuration-reference)
- [Configuration options in context](#configuration-options-in-context)
- [Troubleshooting](#troubleshooting)
- [Limitations & roadmap](#limitations--roadmap)

## Key features (operational view)
- **Config-driven runs:** `SievioConfig` (TOML/JSON/Python) describes sources, processing, sinks, QC/safety, logging, and metadata.
- **Stable outputs:** JSONL (optionally gzipped) as the primary corpus; optional prompt text; optional Parquet sink; optional dataset-card fragments.
- **Screening layer:** QC (quality) + safety can run inline (drop), advisory (annotate), or post-hoc (summary/reports without rewriting JSONL).
- **Safe remote ingestion:** GitHub/PDF/remote SQLite access is routed through a hardened stdlib HTTP client.
- **Sharding:** helper CLI generates per-shard configs and merges per-shard stats safely (counts-only; non-additive QC signal stats cleared).

### Core factories split
This section previously listed internal factory modules. That file map now lives in `../LLMS.md` to avoid duplication and drift.

## Installation
Sievio is typically installed from source in this repository.

```sh
# Core
pip install .

# Development (editable)
pip install -e .
```

Common optional extras:
- `pip install ".[tok]"` – token-aware chunking via `tiktoken`.
- `pip install ".[pdf]"` – PDF ingestion.
- `pip install ".[evtx]"` – Windows EVTX ingestion.
- `pip install ".[parquet]"` – Parquet sink/sidecars.
- `pip install ".[qc]"` – QC workflows/scoring (enables `sievio qc`).
- `pip install ".[langid]"` – optional language ID backends.

## Quickstart

### CLI
```bash
# Run from a config file (TOML/JSON) and print stats JSON to stdout
sievio run -c example_config.toml

# Validate and print the effective config (no run)
sievio run -c example_config.toml --dry-run
```

### Quickstart (library)
```python
from sievio import convert_local_dir

stats = convert_local_dir(
    root_dir="./repo",
    out_jsonl="out/repo.jsonl",
    out_prompt="out/repo.prompt.txt",  # optional
)
print(stats)
```

## CLI
Installable entry points: `sievio` and `sievio-qc` (alias).

General notes:
- Run commands print **stats JSON** to stdout (redirect if you need it for sharded merges).
- `--log-level` configures the CLI logger.

### Run commands
- `sievio run -c CONFIG [--override-max-workers N] [--override-executor-kind auto|thread|process] [--dry-run]`
  - Loads a TOML/JSON config.
  - `--dry-run` validates and prints the effective config, then exits `0`.
  - Without `--dry-run`, runs the pipeline and prints stats JSON.
- `sievio local ROOT_DIR OUT.jsonl [--prompt OUT.prompt] [--base-config CONFIG]`
  - Builds a small profile config for a local directory and runs it.
- `sievio github URL OUT.jsonl [--prompt OUT.prompt] [--base-config CONFIG]`
  - Builds a small profile config for a GitHub repo (best-effort metadata) and runs it.

### Post-hoc tools
- `sievio qc INPUT.jsonl [--csv OUT.csv] [--parallel] [--config CONFIG]`
  - Runs post-hoc QC over an existing JSONL and prints a QC summary JSON.
  - `--config` reuses QC settings from that config file.
  - `--csv` uses the QC extras’ default CSV helper and **does not** apply `qc.scorer_options` from the config (see `QUALITY_CONTROL.md`).
- `sievio card --fragments GLOB... --output README.md`
  - Merges `*.card.json` fragments into a dataset-card README (does not create parent directories).
- `sievio shard --targets targets.txt --base base_config.toml --kind KIND --shards N --out-dir shards/`
  - Generates per-shard JSON configs (refuses to write into a non-empty `--out-dir`).
- `sievio merge-stats STATS.json... [--output merged.json]`
  - Merges `PipelineStats.as_dict()` JSON outputs; see sharding notes below for limitations.

## Concepts & architecture

### Pipeline flow (runtime ordering)
At a high level, each run follows this flow:

```text
sources -> decode -> chunk -> records -> (inline QC/safety?) -> sinks
                                                     |
                                                     v
                                       run summary + optional card fragment
                                                     |
                                                     v
                                  optional post-QC / post-safety (reads JSONL)
```

Operationally important ordering notes:
- **Inline QC/safety** (when enabled) can drop or annotate records before sinks write them.
- **Post-QC/post-safety** run after the pipeline completes and operate on the **primary JSONL path**; they do not rewrite the JSONL (they update summaries and can emit reports/sidecars).
- A run appends a canonical **run summary record** to the primary JSONL at completion (last record; empty `text`, `meta` carries config/stats/QC summary).

### Configuration (`SievioConfig`) in practice
`SievioConfig` is the single source of truth for a run. The full field list is generated in `CONFIGURATION.md`; use `example_config.toml` for a curated “complete example”.

The sections operators most often tune:
- `sources.*`: what gets ingested and how it is enumerated.
- `decode.*` and `chunk.*`: how bytes become normalized text and how text becomes chunks/records.
- `pipeline.*`: concurrency and failure semantics.
- `sinks.*` and `metadata.*`: output layout and canonical primary paths.
- `qc.*`: quality/safety screening modes, thresholds, dedup, and post-pass behavior.
- `http.*`: remote fetch safety and timeouts/limits.

## Quality control (QC)
QC and safety are a screening layer over records. The deep dive lives in `QUALITY_CONTROL.md`; this section focuses on operator semantics.

Modes (quality):
- `qc.enabled=false` → QC off regardless of `qc.mode`.
- `qc.mode="inline"` → score during the run and drop records below gates.
- `qc.mode="advisory"` → score during the run and annotate only (no drops).
- `qc.mode="post"` → main pipeline runs without QC drops; QC runs post-hoc and updates summaries/reports.

Safety (`qc.safety.*`) is independent of `qc.mode` and can also run in post mode.

## Dataset cards (Hugging Face style)
When enabled, each run writes a `*.card.json` fragment next to the primary JSONL (file name: `{primary_jsonl}.card.json`). Merge fragments into a final README card with:

```bash
sievio card --fragments "out/*.card.json" --output out/README.md
```

## Sharding & distributed runs
For detailed guidance, see `DEPLOYMENT.md`. This section is the “gotchas-first” operator summary.

### Distributed execution / sharded runs
This is the recommended split-and-run workflow for distributed launches.

### Gotchas (read first)
- `sievio shard` writes **JSON shard configs** and refuses to write into a **non-empty** `--out-dir`.
- Shard configs are derived from the base config, but the helper rewrites outputs to avoid collisions:
  - For `default_jsonl_prompt` sinks it removes explicit `jsonl_path`/`prompt_path` and relies on `sinks.output_dir` + `sinks.jsonl_basename`.
  - It assigns each shard a unique `sinks.output_dir` and `sinks.jsonl_basename`.
- `sievio merge-stats` merges counts and flags, but clears non-additive QC fields (notably per-signal mean/stdev payloads).
- Stats merge expects consistent QC config across shards (mode/min_score/drop_near_dups); if inconsistent, it raises an error.

### End-to-end flow
1. Generate shard configs:
   ```bash
   sievio shard \
     --targets targets.txt \
     --base base_config.toml \
     --kind github_zip \
     --shards 8 \
     --out-dir shards/
   ```
   `targets.txt` is a newline-separated list; blank lines and `#` comments are ignored.

2. Run shards and capture stats:
   ```bash
   parallel 'sievio run -c {} > {.}.stats.json' ::: shards/*.json
   ```

3. Merge stats:
   ```bash
   sievio merge-stats shards/*.stats.json > merged_stats.json
   ```

4. Merge artifacts:
   - Concatenate JSONL shards (`cat`/`zcat`) as needed for downstream use.
   - Merge dataset-card fragments with `sievio card`.

### Data-consistency checklist
- Use a single base config and avoid per-shard overrides that change QC thresholds/modes.
- Ensure shard outputs do not collide (use the shard helper’s rewritten output layout).
- For global near-duplicate behavior, point all shards at the same dedup DB (`qc.scorer_options.global_dedup.path`) and seed it ahead of time when appropriate.

## Configuration reference
For exact field lists and defaults, use `CONFIGURATION.md` (generated from the config dataclasses). For a curated example that exercises most knobs, use `example_config.toml`.

## Configuration options in context
This section explains the *operator meaning* of major config groups and when to reach for them. For exact fields and defaults, use `CONFIGURATION.md`.

### Most common knobs
| Option | Effect | Use when |
| --- | --- | --- |
| `pipeline.executor_kind` (`auto|thread|process`) | Concurrency strategy | Runs are CPU-heavy (bias to `process`) or I/O-heavy (bias to `thread`) |
| `pipeline.max_workers` | Parallelism | Throughput tuning / resource caps |
| `pipeline.submit_window` | In-flight work bound | Memory pressure / large-file workloads |
| `chunk.policy.target_tokens` / `overlap_tokens` | Chunk size / redundancy | Match model context window / dataset sizing |
| `qc.mode` / `qc.min_score` | QC mode and thresholds | Gate vs annotate vs post-hoc scoring |
| `qc.parallel_post` | Parallel post scoring | Post-QC is slow and QC extras are installed |
| `qc.safety.mode` / `qc.safety.annotate_only` | Safety execution and drops | Run safety without dropping, or run post-hoc safety reports |
| `sinks.output_dir` / `sinks.jsonl_basename` | Output layout defaults | Deterministic file layout across runs/shards |
| `http.timeout` / caps | Remote fetch bounds | Untrusted/slow networks and hostile inputs |

### `sources.*` – What gets ingested
Controls *where data comes from* and how it is enumerated.

Use this when:
- You want to add/remove repositories, directories, or table/query inputs.
- You need to restrict extensions/sizes to manage performance and safety.

### `decode.*` – How bytes become text
Controls Unicode normalization, control stripping, mojibake repair, and byte caps.

Use this when:
- You see garbled characters, invalid UTF-8, or unexpected line breaks.
- You need hard bounds on file reads for safety.

### `chunk.*` – How text becomes records
Controls chunking policy (doc vs code), target size/overlap, and tokenizer behavior.

Use this when:
- You’re tuning record size for a specific model context window.
- You want different behavior for prose vs code.

### `pipeline.*` – Concurrency and failure behavior
Controls execution strategy, pool size, submission window, and fail-fast semantics.

Use this when:
- You need to trade throughput vs memory.
- You need deterministic failure behavior (fail fast) vs “best effort”.

### `sinks.*` and `metadata.*` – Outputs and canonical paths
Controls output layout (what is written and where) and the canonical `primary_jsonl`/`prompt_path` used by post passes and dataset cards.

Use this when:
- You are integrating with downstream tools that expect stable paths/layout.
- You are running shards and want deterministic per-shard output paths.

### `qc.*` – Screening layer (quality + safety)
Controls screening modes, thresholds, reports/sidecars, and global dedup settings.

Use this when:
- You want to gate datasets or generate scoring reports.
- You want safety annotation/drops aligned with policy.

## Troubleshooting
- **`sievio qc` says QC extras are required:** install `pip install ".[qc]"` (or `pip install "sievio[qc]"` if using a published package).
- **`sievio shard` refuses to write configs:** `--out-dir` must be empty; pick a new directory.
- **`sievio merge-stats` fails with “Inconsistent QC config”:** ensure shards used the same base config (QC mode/min_score/drop_near_dups).
- **Post-QC/post-safety doesn’t run:** post passes require a resolvable primary JSONL path and (for QC) the QC extras; see `QUALITY_CONTROL.md`.

## Limitations & roadmap
- Optional extras gate features (`[pdf]`, `[evtx]`, `[parquet]`, `[tok]`, `[qc]`, `[langid]`).
- Token counts fall back to heuristic estimates when `tiktoken` is absent.
- Executor selection is heuristic; extreme workloads may need manual tuning of `pipeline.executor_kind` and `pipeline.max_workers`.