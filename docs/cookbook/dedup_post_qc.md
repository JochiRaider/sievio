# Recipe: Deduplicating a Merged Dataset (Post-QC)

Goal: Run post-hoc QC with near-duplicate detection using a global MinHash store. Requires the `[qc]` extra.

## Prerequisites
- Install extras: `pip install -e ".[qc]"`.
- Produce or merge a primary JSONL (e.g., `merged.jsonl`).

## Step 1: Seed a global dedup DB (optional but recommended)
Use the bundled helper to pre-index existing JSONL/JSONL.GZ files:
```bash
python3 scripts/seed_dedup_db.py out/global_dedup.db data/*.jsonl.gz \
  --k 5 --perm 128 --bands 32 --threshold 0.82
```
Keep the MinHash parameters aligned with your QC settings.

## Step 2: QC config for post mode (`qc_post.toml`)
```toml
[qc]
enabled = true
mode = "post"
drop_near_dups = true
min_score = 60.0
parallel_post = true

[qc.scorer_options.global_dedup]
path = "out/global_dedup.db"
read_only = true

[qc.safety]
enabled = false  # enable + tune if you want safety scoring too
```

## Step 3: Run post-QC over your dataset
```bash
sievio qc merged.jsonl --config qc_post.toml --csv merged_quality.csv
```
This runs post-QC with the config settings for gating/summary. Note: `--csv` uses the default scorer settings (it does not apply `qc.scorer_options` such as `global_dedup`); use `run_qc_over_jsonl` programmatically if you need config-driven CSV output.

## Validate
- Inspect `merged_quality.csv` for drops/flags.
- Check the run summary emitted to stdout for `screeners.quality` counts.
- Confirm the dedup DB metadata matches your MinHash params if you change them.

## Troubleshooting
- Missing `[qc]` extras → install with `pip install -e ".[qc]"`.
- If you want to write signals sidecars, set `qc.write_signals_sidecar=true` and `qc.signals_format` to `csv` or `parquet`.
