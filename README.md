# Sievio

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Contributors](https://img.shields.io/github/contributors/jochiraider/sievio)](https://github.com/jochiraider/sievio/graphs/contributors)

> Library-first, config-driven ingestion pipeline that turns repositories, CSV/SQLite, web PDFs, and Windows EVTX event logs into normalized JSONL and Parquet datasets for LLM fine-tuning and analysis.

## Table of contents

- [Sievio](#sievio)
  - [Table of contents](#table-of-contents)
  - [About](#about)
  - [How it works](#how-it-works)
  - [Getting started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Documentation](#documentation)
  - [Usage](#usage)
    - [CLI](#cli)
    - [Python](#python)
  - [Roadmap](#roadmap)
  - [Contributing](#contributing)
  - [License](#license)
  - [Contact](#contact)

## About

Sievio is a practical ingestion toolkit for building LLM-ready datasets from heterogeneous sources. It normalizes content into a consistent record schema and streams results to JSONL (and optional Parquet), so you can stop maintaining one-off ingestion scripts per source.

Why Sievio?

- **One interface for many inputs:** repos, CSV/SQLite, PDF collections, and EVTX logs.
- **Reproducible runs:** define datasets in TOML/JSON or Python and rerun them deterministically.
- **Traceable outputs:** records carry provenance metadata (source, repository context, and other lineage fields).
- **Safety-minded remote ingestion:** remote fetching is routed through a stdlib-only HTTP client with IP/redirect safeguards.

## How it works

At a glance:

```text
Configuration & Plan                Pipeline Engine (The Loop)                        Outputs
    (Declarative -> Runtime)        (Iterate -> Process -> Filter -> Write)
┌─────────────────────────────┐    ┌────────────────────────────────────────┐    ┌─────────────────┐
│ SievioConfig (TOML/Py)      │    │                                        │    │                 │
│  + Registries (Src/Sink/QC) │─┐  │  1. Source (Iterate Items/Bytes)       │    │ Normalized Data │
│                             │ │  │     ↓                                  │ ┌─>│ [ .jsonl.gz   ] │
│ [ Builder ] -> PipelinePlan │ │  │  2. Decode (Mojibake/Charset)          │ │  │ [ .parquet    ] │
└──────────────┬──────────────┘ │  │     ↓                                  │ │  │                 │
               │                │  │  3. Chunk (Tokenize/Split)             │ │  └─────────────────┘
               │                │  │     ↓                                  │ │
  Inputs (Source Types)         │  │  4. Record Builder (Metadata/ID)       │ │
┌──────────────────────────┐    │  │     ↓                                  │ │
│ • Local Dir / Git Repo   │────┼─>│  5. Inline QC (Safety/Gating) ─────────┼─┘
│ • GitHub Zipball         │    │  │     (Drop or Annotate)                 │  
│ • Web PDFs / URLs        │    │  │     ↓                                  │    ┌─────────────────┐
│ • SQL / CSV / JSONL      │    │  │  6. Sinks (Write to Disk)              │    │ Artifacts       │
│ • Bytes (PDF/EVTX)       │    │  │     ↓                                  │    │                 │
└──────────────────────────┘    │  │  7. Stats Aggregation                  │───>│ [ Dataset Card] │
                                │  └────────────────────────────────────────┘    │ [ QC Summary  ] │
                                │                                                │                 │
                                │             Post-Run Hooks                     └─────────────────┘
                                └─────────────────────────────────────────> (Optional Post-QC/Safety)
```

Architecture overview (the “stable spine”):

**Config (`SievioConfig`) → Builder (plan/runtime) → Pipeline engine (sources → decode → chunk → records → sinks)**

If you want the full architecture and module map, start with `LLMS.md`. For an operator runbook (run/tune/debug), see `docs/TECHNICAL_MANUAL.md`.

## Getting started

### Prerequisites

* Python **3.11+**

### Installation

Sievio is typically installed from source in this repository.

```bash
# Core (from source)
pip install .

# Development (editable)
pip install -e .
```

Optional extras (install only what you need). A few common combinations:

```bash
# Common: PDF + Parquet + token-aware chunking
pip install ".[pdf,parquet,tok]"

# QC workflows and scoring (also enables `sievio qc`)
pip install ".[qc]"

# Full optional feature set for development/power users
pip install ".[tok,pdf,parquet,qc,evtx,langid,accel]"
```

Extras reference:

* `tok`: token-aware chunking via `tiktoken`
* `pdf`: PDF extraction via `pypdf`
* `parquet`: Parquet outputs via `pyarrow`
* `qc`: QC and scoring dependencies (for post-hoc scoring and heavier QC workflows)
* `evtx`: Windows Event Log support (`python-evtx`)
* `langid`: language ID backends (for more precise language tagging)
* `accel`: optional Rust acceleration (`sievio-accel`)

## Documentation

* Docs index: `docs/README.md`
* Technical manual: `docs/TECHNICAL_MANUAL.md`
* Configuration reference (generated): `docs/CONFIGURATION.md`
* Quality control: `docs/QUALITY_CONTROL.md`
* Deployment/sharding: `docs/DEPLOYMENT.md`
* Cookbook recipes: `docs/cookbook/`

## Usage

### CLI

```bash
# Run from a config file (TOML/JSON)
sievio run -c example_config.toml

# Local directory → JSONL
sievio local ./repo out.jsonl

# GitHub repository → JSONL
sievio github https://github.com/owner/name out.jsonl

# Build a dataset card README from per-run fragments
sievio card --fragments "out/*.card.json" --output README.md

# Post-hoc QC over an existing JSONL (requires `pip install ".[qc]"`)
sievio qc out.jsonl --csv out_quality.csv

# Generate shard configs from a base config + targets list
# Note: `--kind web_pdf_list` requires `pip install ".[pdf]"`
sievio shard --targets targets.txt --base config.toml --shards 8 --out-dir shards/ --kind web_pdf_list

# Run a shard and capture stats JSON (stdout)
sievio run -c shards/shard_0000.json > shards/shard_0000.stats.json

# Merge stats JSON files from multiple shards
sievio merge-stats shards/*.stats.json > merged_stats.json

# See all commands and options
sievio --help
```

### Python

Golden path for local directories:

```python
from sievio import convert_local_dir

stats = convert_local_dir(
    root_dir="./repo",
    out_jsonl="out/repo.jsonl",
)
print(stats)
```

Golden path for GitHub repositories:

```python
from sievio import convert_github

stats = convert_github(
    url="https://github.com/owner/name",
    out_jsonl="out/owner__name.jsonl",
)
print(stats)
```

Config-driven runs:

```python
from sievio import load_config_from_path, convert

cfg = load_config_from_path("example_config.toml")
stats = convert(cfg)
print(stats)
```

## Roadmap

* [ ] More connectors and structured sources
* [ ] More rust acceleration
* [ ] More QC reporting and workflows

## Contributing

Contributions are welcome.

**Attention AI agents:** Please read `AGENTS.md` before generating code.

* Project rules, invariants, and required checks: `AGENTS.md`
* Architecture/module map and “where changes should live”: `LLMS.md`

Typical workflow:

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/my-change`)
3. Commit your changes
4. Open a pull request

## License

Sievio is distributed under the MIT License. See `LICENSE` for details.

## Contact

* GitHub: [https://github.com/jochiraider/sievio](https://github.com/jochiraider/sievio)
* Issues: [https://github.com/jochiraider/sievio/issues](https://github.com/jochiraider/sievio/issues)
