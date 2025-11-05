<div id="top"></div>

[![license](https://img.shields.io/github/license/JochiRaider/RepoCapsule?style=for-the-badge)](LICENSE)
[![forks](https://img.shields.io/github/forks/JochiRaider/RepoCapsule?style=for-the-badge)](https://github.com/JochiRaider/RepoCapsule/network/members)
[![contributors](https://img.shields.io/github/contributors/JochiRaider/RepoCapsule?style=for-the-badge)](https://github.com/JochiRaider/RepoCapsule/graphs/contributors)
[![stars](https://img.shields.io/github/stars/JochiRaider/RepoCapsule?style=for-the-badge)](https://github.com/JochiRaider/RepoCapsule/stargazers)
[![issues](https://img.shields.io/github/issues/JochiRaider/RepoCapsule?style=for-the-badge)](https://github.com/JochiRaider/RepoCapsule/issues)


<h1 align="center">RepoCapsule</h1>

<p align="center">
  Repository → JSONL converter with robust decoding, structure‑aware chunking, Markdown→KQL extraction, and GitHub streaming helpers — ideal for pre‑training corpora and RAG.
</p>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About the Project</a></li>
    <li><a href="#features">Features</a></li>
    <li><a href="#architecture">Architecture</a></li>
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
        <li><a href="#kql-extraction">KQL Extraction</a></li>
        <li><a href="#chunking">Chunking</a></li>
        <li><a href="#quality-scoring">Quality Scoring</a></li>
        <li><a href="#jsonl-schema">JSONL Schema</a></li>
      </ul>
    </li>
    <li><a href="#security-model">Security Model</a></li>
    <li><a href="#observability">Observability</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#known-limitations">Known Limitations</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## About the Project

RepoCapsule turns GitHub repositories (or local folders) into clean, structure‑aware JSONL corpora.
It is designed for LLM data pipelines: efficient conversion, robust Unicode decoding, and sensible chunking for both code and docs. Optional helpers extract KQL queries from Markdown, and an add‑on quality module scores the resulting chunks.

## Features

- **Stream‑safe GitHub ingestion**
  - Zipball downloads are buffered to disk and iterated **without extraction**, with anti‑zip‑slip and anti‑zip‑bomb checks.
  - Guards include: absolute/parent‑traversal rejection, per‑file and total uncompressed size caps, member count caps, and compression‑ratio screening.
- **Robust decoding**
  - BOM‑aware UTF‑8/16/32 detection, heuristic UTF‑16 guess, `cp1252` fallback, newline/control cleanup.
- **Structure‑aware chunking**
  - Markdown (ATX/Setext + fenced code) and reStructuredText (titles, directives, literal blocks).
  - Code line packer tuned for source files.
  - Policy‑driven targets/min/overlap; uses `tiktoken` if installed, else a fast estimate.
- **Markdown → KQL**
  - Detect fenced/indented KQL blocks (or infer via heuristics), capture nearby headings as titles. Use `KqlFromMarkdownExtractor` via the `extractors` parameter.
- **Canonical JSONL records**
  - Include `path`, language hint, license, stable `sha256`, and `meta.tokens`/`meta.bytes` for downstream accounting.
- **Optional QC**
  - Lightweight heuristics (+ optional LM perplexity) to sanity‑check corpora and write a CSV.

## Architecture

- `githubio.py` — GitHub URL parsing, metadata calls, zipball download, safe member iteration.
- `fs.py` — Filesystem traversal with light ignore handling.
- `decode.py` — Bytes→text with Unicode repair.
- `convert.py` / `pipeline.py` — Orchestrate repo→records conversion and streaming writes.
- `md_kql.py` — KQL extraction from Markdown.
- `records.py` — Record assembly (hashes, language hints, `meta.tokens`, `meta.bytes`).
- `qc.py` — Quality scoring utilities (heuristics; optional perplexity).
- `runner.py` — High‑level convenience functions for local/GitHub inputs.

## Getting Started

### Prerequisites

- Python 3.11+ (3.12/3.13 also fine)
- (Optional) CUDA‑capable GPU if you plan to run QC with perplexity

### Installation

From source with optional extras:

```bash
pip install -e .[dev]
# accurate tokenization
pip install -e .[tok]
# quality scoring w/ perplexity
pip install -e .[qc]
```

### Configuration

- **GitHub API token** — set `GITHUB_TOKEN` (or `GH_TOKEN`) to avoid anonymous rate limits.

```bash
export GITHUB_TOKEN=ghp_...
```

## Usage

### Quick Start

Convert a GitHub repo to JSONL:

```python
from pathlib import Path
from repocapsule import configure_logging, ChunkPolicy, convert_github_url_to_jsonl_autoname

configure_logging(level="INFO")
out_dir = Path("out"); out_dir.mkdir(parents=True, exist_ok=True)

jsonl_path = convert_github_url_to_jsonl_autoname(
    "https://github.com/owner/repo",
    out_dir,
    policy=ChunkPolicy(mode="auto", target_tokens=1200, overlap_tokens=150, min_tokens=250),
)
print(jsonl_path)
```python
from pathlib import Path
from repocapsule import configure_logging, ChunkPolicy, convert_github_url_to_jsonl_autoname

configure_logging(level="INFO")
out_dir = Path("out"); out_dir.mkdir(parents=True, exist_ok=True)

jsonl_path, prompt_path = convert_github_url_to_jsonl_autoname(
    "https://github.com/owner/repo",
    out_dir,
    # For JSONL only (auto-named), just call and capture the returned path
jsonl_path = convert_github_url_to_jsonl_autoname(
    "https://github.com/owner/repo",
    out_dir,
    policy=ChunkPolicy(mode="auto", target_tokens=1200, overlap_tokens=150, min_tokens=250),
)
print(jsonl_path)
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

Write **both JSONL and a prompt text** in one pass:

```python
from repocapsule import convert_github_url_to_both, ChunkPolicy

jsonl_path, prompt_path = convert_github_url_to_both(
    "https://github.com/owner/repo",
    jsonl_path="out/repo.jsonl",
    prompt_txt_path="out/repo.prompt.txt",
    policy=ChunkPolicy(mode="auto", target_tokens=1200, overlap_tokens=150, min_tokens=250),
)
print(jsonl_path, prompt_path)
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

- **GitHub I/O:** `parse_github_url`, `get_repo_info`, `download_zipball_to_temp`, `iter_zip_members`, `build_output_basename`
- **Converters:** `convert_repo_to_jsonl`, `convert_repo_to_jsonl_autoname`,
  `convert_github_url_to_jsonl`, `convert_github_url_to_both`, `convert_github_url_to_jsonl_autoname`
- **Chunking:** `ChunkPolicy`, `chunk_text`, `split_doc_blocks`, `count_tokens`
- **Markdown/KQL:** `extract_kql_blocks_from_markdown`, `is_probable_kql`, `guess_kql_tables`, `derive_category_from_rel`, and the `KqlFromMarkdownExtractor` class (import from `repocapsule.md_kql`).
- **Records:** `build_record`, language/extension helpers
- **Logging:** `configure_logging`, `get_logger`



Use the **extractors** parameter with the provided `KqlFromMarkdownExtractor`:

```python
from repocapsule import convert_github_url_to_jsonl_autoname, ChunkPolicy
from repocapsule.md_kql import KqlFromMarkdownExtractor

jsonl_path = convert_github_url_to_jsonl_autoname(
    "https://github.com/owner/repo-of-kql",
    "out",
    policy=ChunkPolicy(mode="doc", target_tokens=1500, overlap_tokens=100, min_tokens=200),
    extractors=[KqlFromMarkdownExtractor()],
)
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

- **Docs:** format‑aware splitters are applied before packing (Markdown, reStructuredText).
- **Code:** line‑based blocks that prefer blank lines as boundaries.
- **Token counting:** uses `tiktoken` if installed; otherwise a fast char/token estimate.

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

Outputs a CSV with columns like `score`, `tokens`, `lang`, `path`, `near_dup`, etc.

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

## Security Model

**Threat‑oriented ingestion** — RepoCapsule defends against common archive risks when consuming GitHub zipballs:

- **Path traversal (Zip Slip)**: rejects absolute paths, drive letters, and parent (`..`) segments; strips GitHub’s top‑folder prefix.
- **Zip bombs**: caps total uncompressed bytes, per‑file bytes, number of members, and flags extreme compression ratios.
- **Symlinks**: skips symlinked entries (based on Unix mode bits in zip metadata).

> **Note:** Default thresholds are conservative but adjustable at call time. If you ingest very large repos or binaries, tune caps accordingly.

## Observability

- Structured logs include info on processed/skipped members. For production, consider counters like `skipped_by_ratio`, `skipped_by_path`, `skipped_symlink`, and totals.

## Roadmap

- [ ] CLI entry points for common conversions and QC
- [ ] Additional doc splitters (Asciidoc), more language hints
- [ ] More extractors (e.g., SPL/Sigma/YARA from docs)
- [ ] Optional parallelism for large local folders
- [ ] Dedup/near‑dup filtering helpers
- [ ] License classifier heuristics & SPDX enrichment
- [ ] Test suite (unit + negative): archive safety, RFC‑6266 filenames, long‑text QC

## Known Limitations

- **Tests:** a full pytest suite is still being built; until then, treat the archive safety code as defense‑in‑depth and monitor logs.
- **Perplexity QC:** very long texts may emit warnings on small‑context models; use the stride parameters or avoid PPL for giant chunks.
- **Symlink detection:** relies on Unix mode bits in zip metadata; some archives (non‑GitHub) may encode links inconsistently.

## Contributing

Contributions are welcome! Please open an issue to discuss significant changes.

1. Fork the project
2. Create a feature branch: `git checkout -b feat/awesome`
3. Commit your changes: `git commit -m "feat: add awesome"`
4. Push to the branch: `git push origin feat/awesome`
5. Open a Pull Request

## License

Distributed under the MIT License. See [license-url] for details.

## Contact

Open an issue on GitHub or start a discussion.

## Acknowledgments

- Inspired by real‑world data‑prep needs for LLM pre‑training and RAG
- Thanks to the maintainers of the Best‑README‑Template and the broader OSS ecosystem

---

[contributors-shield]: https://img.shields.io/github/contributors/JochiRaider/repocapsule.svg?style=for-the-badge
[contributors-url]: https://github.com/your-org/repocapsule/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/JochiRaider/repocapsule.svg?style=for-the-badge
[forks-url]: https://github.com/your-org/repocapsule/network/members
[stars-shield]: https://img.shields.io/github/stars/JochiRaider/repocapsule.svg?style=for-the-badge
[stars-url]: https://github.com/your-org/repocapsule/stargazers
[issues-shield]: https://img.shields.io/github/issues/JochiRaider/repocapsule.svg?style=for-the-badge
[issues-url]: https://github.com/your-org/repocapsule/issues
[license-shield]: https://img.shields.io/github/license/JochiRaider/repocapsule.svg?style=for-the-badge
[license-url]: https://github.com/your-org/repocapsule/blob/main/LICENSE

