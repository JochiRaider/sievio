# RepoCapsule

**Repository → JSONL (PTD) converter with robust decoding, structure‑aware chunking, optional KQL‑from‑Markdown extraction, GitHub streaming helpers, and an optional quality‑scoring pass.**

Python 3.11+, standard library by default (optional extras for QC).

---

## Why?

RepoCapsule turns heterogeneous repositories into **clean JSONL corpora** for pre‑training, post‑training refinement, retrieval, or evaluation. It preserves structure (Markdown sections, fenced code, code lines), emits a consistent JSONL schema (with `meta.tokens` and `meta.bytes`), and can also write a **prompt‑text sidecar** in the same pass for quick prompting and spot QA.

---

## Features

- **Streamed GitHub ingestion** with defensive limits and license‑aware output names
- **Robust decoding**: UTF‑8 first, BOM handling, UTF‑16 heuristics, cp1252 fallback, mojibake repair
- **Structure‑aware chunking**:
  - Docs and Markdown → paragraph‑based with sentence, fence, and table awareness
  - Code → line‑aware packing with a lightweight token estimate (no external tokenizer required)
- **Markdown → KQL extraction** (opt‑in): pull only KQL code blocks from `.md`
- **JSONL schema** with `meta.lang`, `meta.tokens`, `meta.bytes`, `meta.sha256`, chunk indices, and license
- **Prompt text sidecar**: write `*.prompt.txt` while streaming JSONL, no giant in‑memory list
- **.gitignore‑aware** traversal for local directories
- **Optional QC**: heuristics and optional LM perplexity to produce a CSV quality report
- **Library‑friendly logging** helpers

---

## Install

```bash
# from repo root
python -m pip install -e .

# optional extras for quality scoring (torch, transformers, tiktoken, pyyaml)
python -m pip install -e .[qc]
```

Tip: For GitHub API calls, set a token to reduce throttling.

- Windows: `setx GITHUB_TOKEN "<your-token>"`
- macOS/Linux: `export GITHUB_TOKEN="<your-token>"`

---

## Quickstart

### 1) Convert a GitHub repo with auto‑named outputs

```python
from pathlib import Path
from repocapsule import convert_github_url_to_jsonl_autoname

out_dir = Path("out")
out_dir.mkdir(parents=True, exist_ok=True)

jsonl_path, prompt_path = convert_github_url_to_jsonl_autoname(
    "https://github.com/olafhartong/sysmon-modular",
    out_dir,
    include_exts={".yml", ".yaml", ".xml", ".md", ".txt"},
    also_prompt_text=True,       # stream‑write JSONL + prompt together
    kql_from_markdown=False,     # not needed for this repo
)
print(jsonl_path, prompt_path)
```

A ready‑made runner is included: `python scripts/test_sysmon_modular.py`.

### 2) Convert a local directory

```python
from repocapsule import convert_repo_to_jsonl_autoname

jsonl_path = convert_repo_to_jsonl_autoname(
    "path/to/repo",
    "out",
    include_exts={".py", ".md", ".yml", ".json", ".xml"},
    kql_from_markdown=True,   # only if you want KQL blocks from .md
)
print(jsonl_path)
```

---

## JSONL schema

Each line is one object:

```json
{
  "text": "<chunk>",
  "meta": {
    "source": "https://github.com/owner/repo",
    "repo": "owner/repo",
    "path": "sub/dir/file.ext",
    "license": "Apache-2.0",
    "lang": "Markdown",
    "chunk_id": 1,
    "n_chunks": 3,
    "encoding": "utf-8",
    "had_replacement": false,
    "sha256": "…",
    "tokens": 1234,
    "bytes": 5678
  }
}
```

`tokens` is a fast, symbols‑aware estimate. `bytes` is the UTF‑8 length of `text`.

---

## Prompt text sidecar

When `also_prompt_text=True` or when using `convert_github_url_to_both(...)`, the converter writes a `*.prompt.txt` file while streaming JSONL. Each chunk is prefixed like:

```
### path/to/file.ext [3/7] (lang=YAML)

<chunk text>
```

---

## Markdown → KQL extraction (opt‑in)

Set `kql_from_markdown=True` to emit records only for Markdown code blocks that look like KQL. Unlabeled fences are accepted if they match a small KQL heuristic (operators and shape). Useful for repos that interleave prose and hunting queries.

---

## Chunking policy

Use `ChunkPolicy` to tune chunk sizes.

```python
from repocapsule import ChunkPolicy, convert_repo_to_jsonl

policy = ChunkPolicy(
    mode="auto",       # "auto" | "doc" | "code"
    target_tokens=800,
    overlap_tokens=100,
    min_tokens=200
)

convert_repo_to_jsonl("repo_dir", "out/repo.jsonl", policy=policy)
```

Guidelines:

- Docs: paragraph‑aware packing that keeps fences, tables, and lists intact
- Code: line‑aware packing that is deterministic and diff‑friendly

---

## Quality scoring (optional)

Install extras with `.[qc]` and then:

```python
from repocapsule import score_jsonl_to_csv
csv_path = score_jsonl_to_csv("out/owner_repo__main__MIT.jsonl",
                              lm_model_id=None)  # or a local HF model id
print(csv_path)  # …_quality.csv
```

The CSV includes fields such as `score`, `perplexity` when a model is supplied, `parse_ok` (syntax checks), `repetition`, and `near_dup`.

---

## Safety and limits

- Defensive archive reading: member count caps, total uncompressed limits, per‑file caps, and compression‑ratio checks when iterating zip members
- `.gitignore` is honored for local traversal, and common junk or build paths are excluded by default

---

## Logging

Library‑friendly by default with no noisy root handlers. For scripts:

```python
from repocapsule import configure_logging
log = configure_logging(level="INFO")
```

A `temp_level(...)` context manager is also available for scoped verbosity.

---

## Troubleshooting

- API throttling: provide a GitHub token via `GITHUB_TOKEN` or `GH_TOKEN`
- If you hit 403 or 429, wait until the reset timestamp in rate‑limit headers
- Markdown basics: use headings, lists, code fences, and tables for readability
- SPDX license IDs: outputs may include an SPDX short identifier in the filename for provenance, for example `…__MIT.jsonl`

---

## Contributing

- Python 3.11+
- Lint and type check with dev extras (`ruff`, `mypy`, `pytest`)
- Open issues or PRs with focused changes and clear repro steps

---

## License

MIT (see `LICENSE`).


