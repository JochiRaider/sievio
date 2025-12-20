# Recipe: Custom PII Scrubbing via Middleware

Goal: redact PII-like patterns before they hit sinks/QC by adding a record middleware. Works without extras; optional QC/safety can run afterward.

## Prerequisites
- Editable install: `pip install -e .`
- Know your input path and output targets.

## Minimal script
```python
import re
from sievio import PipelineOverrides, convert, make_local_repo_config

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}")

def scrub_pii(record):
    text = record.get("text", "")
    if text:
        record["text"] = EMAIL_RE.sub("<redacted-email>", text)
    return record

overrides = PipelineOverrides(record_middlewares=[scrub_pii])
cfg = make_local_repo_config(
    root_dir="./repo",
    out_jsonl="out/repo_scrubbed.jsonl",
)
stats = convert(cfg, overrides=overrides)
print(stats)
```

Run it with `PYTHONPATH=src python3 script.py`. The middleware runs inside the pipeline engine before sinks/QC.

## Notes
- For more complex scrubbing, implement `SafetyScorer` (see `src/sievio/core/interfaces.py`) and register it with `safety_scorer_registry`; then set `qc.safety.scorer_id` to your factory id.
- Middlewares should be pure functions (no side effects); return the mutated record so downstream QC/sinks see the redacted text.
