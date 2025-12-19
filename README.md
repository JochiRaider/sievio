# Sievio

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Contributors](https://img.shields.io/github/contributors/jochiraider/sievio)](https://github.com/jochiraider/sievio/graphs/contributors)

> Library-first, config-driven ingestion pipeline that turns repositories, CSV/SQLite, web PDFs, and Windows EVTX event logs into normalized JSONL and Parquet datasets for LLM fine-tuning and analysis.

## Table of contents

- [Sievio](#sievio)
  - [Table of contents](#table-of-contents)
  - [About](#about)
  - [Built with](#built-with)
  - [Getting started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Usage](#usage)
    - [CLI](#cli)
    - [Python](#python)
  - [Roadmap](#roadmap)
  - [Contributing](#contributing)
  - [License](#license)
  - [Contact](#contact)

## About

Sievio is designed around a stable “spine”:

**Config (`SievioConfig`) → Builder (plan/runtime) → Pipeline engine (sources → decode → chunk → records → sinks)**

Key capabilities:

- **Config-first, reproducible runs** (Python or TOML)
- **Built-in source types**: local directories (gitignore-aware), GitHub zipballs, CSV/TSV, SQLite, and web-PDF URL lists/page scrapes (PDF/EVTX/Parquet handling activates via extras)
- **Built-in outputs**: JSONL (plain or `.jsonl.gz`) plus optional prompt text; Parquet sink requires the `parquet` extra
- **Extensible by design** via registries/factories/plugins (sources, sinks, bytes handlers, QC/safety, hooks)
- **Safe remote ingestion** (GitHub zipballs, web PDFs, SQLite downloads) routed through a stdlib-only HTTP client with IP/redirect safeguards
- **Language tagging** by default (baseline heuristics); optional `langid` backends via the `langid` extra
- **Optional quality control + safety screening** (inline/advisory/post); near-duplicate detection and `sievio qc` require the `qc` extra
- **Dataset card fragments** for downstream publishing workflows
- **Sharded runs** via `sievio shard` + `sievio merge-stats` helpers for distributed execution

## Built with

- [Python 3.11+](https://www.python.org/)
- Optional dependencies are installed via extras (see `pyproject.toml`):
  - `tok`: [tiktoken](https://github.com/openai/tiktoken) (token-aware chunking)
  - `parquet`: [PyArrow](https://arrow.apache.org/) (Parquet outputs)
  - `pdf`: [PyPDF](https://pypdf.readthedocs.io/) (PDF extraction)
  - `qc`: QC/scoring dependencies (torch/transformers)
  - `evtx`: Windows Event Log support (`python-evtx`)
  - `langid`: language ID backends (`lingua-language-detector`, `pygments`)

## Getting started

### Prerequisites

- Python **3.11+**

### Installation

```bash
# For standard usage (from source)
pip install .
# For development (editable mode)
pip install -e .
# Optional extras (install only what you need):
pip install -e ".[tok,pdf,parquet,qc,evtx,langid]"
```

## Documentation
- Docs index: `docs/README.md`
- Technical manual: `docs/TECHNICAL_MANUAL.md`
- Configuration reference (generated): `docs/CONFIGURATION.md`
- Quality control: `docs/QUALITY_CONTROL.md`
- Deployment/sharding: `docs/DEPLOYMENT.md`
- Cookbook recipes: `docs/cookbook/`

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

# Post-hoc QC over an existing JSONL (requires the `qc` extra)
sievio qc out.jsonl --csv out_quality.csv

# Generate shard configs from a base config + targets list
sievio shard --targets targets.txt --base config.toml --shards 8 --out-dir shards/ --kind web_pdf_list

# Merge stats JSON files from multiple shards
sievio merge-stats shards/*/stats.json > merged_stats.json

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

Documentation:

- Read the technical manual: `docs/TECHNICAL_MANUAL.md`

## Roadmap

- [ ] More connectors and structured sources
- [ ] More QC reporting + workflows
- [ ] More dataset card automation

## Contributing

Contributions are welcome.

**Attention AI agents:** Please read `AGENTS.md` before generating code.

- Project rules, invariants, and required checks: `AGENTS.md`
- Architecture/module map and “where changes should live”: `LLMS.md`

Typical workflow:

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/my-change`)
3. Commit your changes
4. Open a pull request

## License

Sievio is distributed under the MIT License. See `LICENSE` for details.

## Contact

- GitHub: https://github.com/jochiraider/sievio
- Issues: https://github.com/jochiraider/sievio/issues
