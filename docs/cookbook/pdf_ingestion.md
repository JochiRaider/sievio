# Recipe: Ingesting a Remote PDF List

Goal: Download a list of remote PDFs and emit JSONL (and optional prompt text). Requires the `[pdf]` extra.

## Prerequisites
- Install extras: `pip install -e ".[pdf]"` (add `parquet` if you also want Parquet output).
- Ensure `sievio` CLI is available (`pyproject.toml` exposes `sievio`).

## Minimal config (`pdf_run.toml`)
```toml
[sources.defaults.web_pdf_list]
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
- Capture stats JSON from stdout if you want a file (e.g., `sievio run -c pdf_run.toml > out/pdf_corpus.stats.json`).
- Spot-check counts from the run-summary record appended to the JSONL:
  ```bash
  python - <<'PY'
  import json
  from pathlib import Path

  path = Path("out/pdf_corpus.jsonl")
  with path.open(encoding="utf-8") as handle:
      for line in handle:
          pass
  meta = json.loads(line).get("meta", {})
  print(meta.get("stats", {}).get("files"))
  PY
  ```

## Troubleshooting
- Non-PDF responses are skipped when `require_pdf=true` (default). Set `require_pdf=false` only if you accept ambiguous content.
- For large lists, increase `download_max_workers` and ensure `max_pdf_bytes` matches your workload.
