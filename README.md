<div id="top"></div>

[![license](https://img.shields.io/github/license/JochiRaider/RepoCapsule?style=for-the-badge)](LICENSE)
[![forks](https://img.shields.io/github/forks/JochiRaider/RepoCapsule?style=for-the-badge)](https://github.com/JochiRaider/RepoCapsule/network/members)
[![contributors](https://img.shields.io/github/contributors/JochiRaider/RepoCapsule?style=for-the-badge)](https://github.com/JochiRaider/RepoCapsule/graphs/contributors)
[![stars](https://img.shields.io/github/stars/JochiRaider/RepoCapsule?style=for-the-badge)](https://github.com/JochiRaider/RepoCapsule/stargazers)
[![issues](https://img.shields.io/github/issues/JochiRaider/RepoCapsule?style=for-the-badge)](https://github.com/JochiRaider/RepoCapsule/issues)


<div align="center">
  <h1>RepoCapsule</h1>
  <p>
    Repository → JSONL converter with robust decoding, structure‑aware chunking, Markdown→KQL extraction, and GitHub streaming helpers — ideal for pre‑training corpora and RAG.
  </p>
  <p>
    <a href="https://github.com/your-org/repocapsule"><strong>Explore the docs »</strong></a>
    ·
    <a href="https://github.com/your-org/repocapsule/issues">Report Bug</a>
    ·
    <a href="https://github.com/your-org/repocapsule/issues">Request Feature</a>
  </p>
</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#features">Features</a></li>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li><a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#configuration">Configuration</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a>
      <ul>
        <li><a href="#quick-start">Quick Start</a></li>
        <li><a href="#api-surface">API Surface</a></li>
        <li><a href="#kql-extraction">KQL Extraction from Markdown</a></li>
        <li><a href="#chunking">Chunking & Policies</a></li>
        <li><a href="#quality-scoring">Quality Scoring (optional)</a></li>
        <li><a href="#jsonl-schema">JSONL Schema</a></li>
      </ul>
    </li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>


## About The Project

RepoCapsule turns GitHub repositories (or local folders) into clean, structure‑aware JSONL corpora.
It is designed for LLM data pipelines: efficient conversion, robust Unicode decoding, and sensible chunking for both code and docs. Optional helpers extract KQL queries from Markdown, and an add‑on quality module scores the resulting chunks.

### Features

- **Stream‑safe GitHub ingestion**: Download zipballs in chunks (disk‑buffered), with zip‑bomb defenses and path safety checks.
- **Robust decoding**: BOM‑aware UTF‑8/16/32 detection, heuristic UTF‑16 guess, cp1252 fallback, newline/control cleanup.
- **Structure‑aware chunking**:
  - Markdown (ATX/Setext + fenced code) and reStructuredText (titles, directives, literal blocks).
  - Code line‑packer tuned for source files.
  - Target/min/overlap token policy with **tiktoken** when available, fast estimate otherwise.
- **Markdown → KQL**: Detect fenced/indented KQL blocks (or infer via heuristics), capture nearby headings as titles.
- **Canonical JSONL records**: Include path, language hint, license, stable `sha256`, and `tokens`/`bytes` size.
- **Optional QC**: Lightweight heuristics (+ optional HF perplexity) to sanity‑check corpora and write a CSV.

### Built With

- Python ≥ 3.11 (stdlib‑only core)
- Optional extras:
  - `tiktoken` for exact token counts
  - `torch` + `transformers` for perplexity in QC
  - `pyyaml` for YAML validation in QC


## Getting Started

### Prerequisites

- Python 3.11+ (3.12/3.13/3.14 also supported)
- (Optional) CUDA GPU if you plan to run QC with perplexity

### Installation

From source (editable) with optional extras:

```bash
pip install -e .[dev]
# add accurate tokenization
pip install -e .[tok]
# add quality scoring w/ perplexity
pip install -e .[qc]
```

### Configuration

- GitHub API: set `GITHUB_TOKEN` (or `GH_TOKEN`) to avoid low anonymous rate limits.

```bash
export GITHUB_TOKEN=ghp_...
```


## Usage

### Quick Start

Convert a GitHub repo to JSONL — and also a prompt‑friendly text file in the same pass:

```python
from pathlib import Path
from repocapsule import configure_logging, ChunkPolicy, convert_github_url_to_jsonl_autoname

configure_logging(level="INFO")
out_dir = Path("out"); out_dir.mkdir(parents=True, exist_ok=True)

jsonl_path, prompt_path = convert_github_url_to_jsonl_autoname(
    "https://github.com/owner/repo",
    out_dir,
    also_prompt_text=True,                   # stream JSONL + prompt together
    kql_from_markdown=False,                 # set True to extract only KQL from Markdown
    policy=ChunkPolicy(mode="auto", target_tokens=1200, overlap_tokens=150, min_tokens=250),
)
print(jsonl_path, prompt_path)
```

Convert a **local** folder to JSONL with default include‑exts:

```python
from repocapsule import convert_repo_to_jsonl, ChunkPolicy
convert_repo_to_jsonl(
    root="/path/to/repo",
    jsonl_path="/path/to/out/repo.jsonl",
    policy=ChunkPolicy(mode="doc", target_tokens=1700, overlap_tokens=40, min_tokens=400),
)
```

### API Surface

Public entry points (selection):

- GitHub I/O: `parse_github_url`, `get_repo_info`, `download_zipball_to_temp`, `iter_zip_members`, `build_output_basename`
- Converters: `convert_repo_to_jsonl`, `convert_repo_to_jsonl_autoname`,
  `convert_github_url_to_jsonl`, `convert_github_url_to_both`, `convert_github_url_to_jsonl_autoname`
- Chunking: `ChunkPolicy`, `chunk_text`, `split_doc_blocks`, `count_tokens`
- Markdown/KQL: `extract_kql_blocks_from_markdown`, `is_probable_kql`, `guess_kql_tables`, `derive_category_from_rel`
- Records: `build_record`, language/extension helpers
- Logging: `configure_logging`, `get_logger`

See the docstrings and source for full signatures.

### KQL Extraction

Extract only KQL blocks from Markdown files (good for hunting query repos):

```python
from repocapsule import convert_github_url_to_jsonl_autoname
convert_github_url_to_jsonl_autoname(
    "https://github.com/owner/repo-of-kql",
    "out",
    kql_from_markdown=True,
)
```

Heuristics accept fences labeled `kql`/`kusto`, or infer KQL by operators (e.g., `| where`, `summarize`).

### Chunking

`ChunkPolicy` controls chunk size and overlap. For prose, targets ~1500–2000 tokens work well; for code, slightly smaller targets reduce truncation risk.

- **Docs**: format‑aware splitters are applied before packing (Markdown, reStructuredText).
- **Code**: simple line‑based blocks that prefer blank lines as boundaries.
- **Token counting**: uses `tiktoken` if installed; otherwise a fast char/token estimate.

### Quality Scoring

Install the `qc` extra to compute lightweight scores and (optionally) perplexity:

```python
from repocapsule import score_jsonl_to_csv
csv_path = score_jsonl_to_csv(
    "/path/to/corpus.jsonl",
    lm_model_id="Qwen/Qwen2.5-1.5B",  # optional; omit to skip perplexity
    device="cuda",
    dtype="bfloat16",
)
print(csv_path)
```

Produces a CSV with columns like `score`, `tokens`, `lang`, `path`, `near_dup`, etc.

### JSONL Schema

Each line is an object:

```json
{
  "text": "<chunk>",
  "meta": {
    "source": "https://github.com/owner/repo",
    "repo": "owner/repo",
    "path": "sub/dir/file.py",
    "license": "Apache-2.0",
    "lang": "Python",
    "chunk_id": 1,
    "n_chunks": 3,
    "encoding": "utf-8",
    "had_replacement": false,
    "sha256": "...",
    "tokens": 1234,
    "bytes": 5678
  }
}
```


## Roadmap

- [ ] CLI entry point(s) for common conversions and QC
- [ ] Additional doc splitters (Asciidoc), more language hints
- [ ] More extractors (e.g., SPL/Sigma/YARA from docs)
- [ ] Optional parallelism for large local folders
- [ ] Dedup/near‑dup filtering helpers
- [ ] License classifier heuristics & SPDX enrichment

See open issues for current plans and discussion.


## Contributing

Contributions are welcome! Please open an issue to discuss significant changes.

1. Fork the project
2. Create a feature branch: `git checkout -b feat/awesome`
3. Commit your changes: `git commit -m "feat: add awesome"`
4. Push to the branch: `git push origin feat/awesome`
5. Open a Pull Request


## License

Distributed under the MIT License. See `LICENSE` for details.


## Contact

Open an issue on GitHub or start a discussion. You can also file bugs anonymously by emailing your CI or a group mailbox (replace this line with your preferred contact).


## Acknowledgments

- Inspired by real‑world data‑prep needs for LLM pre‑training and RAG
- Thanks to the maintainers of the Best‑README‑Template and the broader OSS ecosystem


---

[contributors-shield]: https://img.shields.io/github/contributors/your-org/repocapsule.svg?style=for-the-badge
[contributors-url]: https://github.com/your-org/repocapsule/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/your-org/repocapsule.svg?style=for-the-badge
[forks-url]: https://github.com/your-org/repocapsule/network/members
[stars-shield]: https://img.shields.io/github/stars/your-org/repocapsule.svg?style=for-the-badge
[stars-url]: https://github.com/your-org/repocapsule/stargazers
[issues-shield]: https://img.shields.io/github/issues/your-org/repocapsule.svg?style=for-the-badge
[issues-url]: https://github.com/your-org/repocapsule/issues
[license-shield]: https://img.shields.io/github/license/your-org/repocapsule.svg?style=for-the-badge
[license-url]: https://github.com/your-org/repocapsule/blob/main/LICENSE

