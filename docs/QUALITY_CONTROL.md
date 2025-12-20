# Quality Control & Safety

Sievio’s screening layer evaluates records for quality and safety. It can annotate, gate inline, or run post-hoc without rewriting the primary JSONL.

## Modes
- `qc.enabled=false` → QC off regardless of mode.
- `mode="inline"` → score during extraction and drop records below `min_score`/near-dup gates.
- `mode="advisory"` → score inline, annotate only (no drops).
- `mode="post"` → main pipeline runs without QC drops; QC gates in a post pass (`sievio qc ...` or post hook) and can emit CSV/sidecars.
- Safety (`qc.safety.*`) is independent: supports `inline`/`advisory`/`post` with `annotate_only` to disable drops.

## Signals and metadata
- Inline QC writes signals under `meta["extra"]["qc_signals"]`; summaries live in `stats["qc"]["screeners"]["quality"]["signal_stats"]`.
- Default signals from `JSONLQualityScorer` (when `[qc]` extras installed): token counts (`len_tok`), ascii ratio, repetition, gopher_quality, perplexity (when a model is configured), near-duplicate hashes/ids, language hints.
- Safety scorer (`default_safety`) flags regex-based PII/toxicity/license issues; flags are reported in QC summaries and optional `_safety.csv`/`_safety_signals.*` sidecars when safety post mode is used.

## Configuration knobs (see `docs/CONFIGURATION.md` for full list)
- `qc.min_score`, `qc.drop_near_dups`, `qc.exact_dedup`, `qc.fail_on_error`.
- `qc.scorer_id` (None → registry default `jsonl_default`), `qc.scorer_options.heuristics` (tune thresholds), `qc.scorer_options.global_dedup` (MinHash store path/read_only).
- `qc.write_csv`, `qc.csv_suffix`, `qc.write_signals_sidecar`, `qc.signals_format` (`csv|parquet`), `qc.signals_suffix`.
- Post-pass concurrency: `qc.parallel_post`, `qc.post_executor_kind`, `qc.post_max_workers`, `qc.post_submit_window`.
- Safety: `qc.safety.enabled`, `qc.safety.mode`, `qc.safety.annotate_only`, `qc.safety.allowed_licenses`, `qc.safety.toxicity_threshold`, `qc.safety.write_csv`, `qc.safety.write_signals_sidecar`, `qc.safety.signals_format`.

## Running QC
- Inline/advisory: set `qc.enabled=true` and `mode` to `inline` or `advisory` in the main config; run via `sievio run ...`.
- Post-hoc: either set `qc.mode="post"` in the run config or use the CLI wrapper on an existing JSONL:
  ```bash
  sievio qc primary.jsonl --config qc_post.toml --csv primary_quality.csv
  ```
  Requires `[qc]` extras. Note: `--csv` uses `score_jsonl_to_csv` defaults and does not apply `qc.scorer_options` from the config.

## Deduplication
- Local near-dup detection uses SimHash/MinHash heuristics controlled by `qc.scorer_options.heuristics`.
- Global dedup: configure `[qc.scorer_options.global_dedup]` with a SQLite path (optionally seeded via `scripts/seed_dedup_db.py`); set `read_only=true` when reusing a seeded DB.

## Exports
- QC CSV report: `{primary_jsonl}{qc.csv_suffix}`.
- Signals sidecar: `{primary_jsonl}{qc.signals_suffix}` with `qc.signals_format`.
- Safety post mode: `_safety.csv` and optional `_safety_signals.(csv|parquet)`.
