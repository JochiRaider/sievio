# Recipe: Ingesting a Remote PDF List

Goal: Download a list of remote PDFs and emit JSONL (and optional prompt text). Requires the `[pdf]` extra.

## Prerequisites
- Install extras: `pip install -e ".[pdf]"` (add `parquet` if you also want Parquet output).
- Ensure `sievio` CLI is available (`pyproject.toml` exposes `sievio`).

## Minimal config (`pdf_run.toml`)
```toml
[sources.pdf]
max_pdf_bytes = 104857600      # 100 MiB cap
download_max_workers = 8
retries = 2

[[sources.specs]]
kind = "web_pdf_list"
options = { urls = [
  "https://example.com/a.pdf",
  "https://example.com/b.pdf",
], add_prefix = "batch1" }

[sinks]
output_dir = "out"
jsonl_basename = "pdf_corpus"

[[sinks.specs]]
kind = "default_jsonl_prompt"
[sinks.specs.options]
jsonl_path = "out/pdf_corpus.jsonl"
prompt_path = "out/pdf_corpus.prompt.txt"
```

## Run
```bash
sievio run -c pdf_run.toml
```

## Validate
- Check counts in `out/pdf_corpus.stats.json` (written by the run summary hook).
- Spot-check records: `python - <<'PY'\nimport json\nfrom pathlib import Path\nprint(json.loads(Path('out/pdf_corpus.stats.json').read_text())['files'])\nPY`

## Troubleshooting
- Non-PDF responses are skipped when `require_pdf=true` (default). Set `require_pdf=false` only if you accept ambiguous content.
- For large lists, increase `download_max_workers` and ensure `max_pdf_bytes` matches your workload.
