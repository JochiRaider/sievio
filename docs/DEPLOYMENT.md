# Deployment & Sharding

Operational guidance for running Sievio at scale. For API details, see `docs/TECHNICAL_MANUAL.md`; for QC specifics, see `docs/QUALITY_CONTROL.md`.

## Sharded runs
1. **Generate shard configs**  
   ```bash
   sievio shard \
     --targets targets.txt \
     --base base_config.toml \
     --kind github_zip \
     --shards 4 \
     --out-dir shards
   ```
   - `targets.txt` is a newline-separated list of inputs for the chosen kind (e.g., GitHub URLs).
   - Each shard config is JSON and includes isolated outputs.

2. **Run shards in parallel**  
   Use your scheduler of choice (Slurm/K8s/parallel). Example:
   ```bash
   parallel 'sievio run -c {} > {.}.stats.json' ::: shards/*.json
   ```

3. **Merge stats**  
   ```bash
   sievio merge-stats shards/*.stats.json > merged_stats.json
   ```
   Stats merging keeps counts/flags and clears non-additive QC fields.

4. **Merge artifacts**  
   Concatenate JSONL/prompt outputs (`cat`/`zcat`), and merge dataset-card fragments with:
   ```bash
   sievio card --fragments "shards/**/*.card.json" --output out/README.md
   ```

## Observability
- CLI emits JSON stats to stdout; redirect per shard if you plan to merge stats.
- Run summary records are appended to the primary JSONL; the last record carries config/stats/QC summary.
- QC/safety post passes can emit CSV/Parquet sidecars (see `docs/QUALITY_CONTROL.md`).

## Runtime knobs
- Concurrency: `pipeline.executor_kind` (`auto|thread|process`), `max_workers`, and `submit_window`.
- Remote safety: `http.*` (timeouts, redirect safeguards) and PDF/SQLite download caps.
- QC post-processing: `qc.parallel_post` and `qc.post_*` override concurrency for post passes.

## Footnotes
- Sharding currently supports `github_zip`, `web_pdf_list`, `local_dir`, and `sqlite`; use `--kind` accordingly.
- For very large PDFs/EVTX workloads, set `pipeline.executor_kind="process"` in the base config to bias toward process workers.
