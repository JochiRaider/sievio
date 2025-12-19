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
   parallel 'sievio run -c {}' ::: shards/*.json
   ```

3. **Merge stats**  
   ```bash
   sievio merge-stats shards/*/stats.json > merged_stats.json
   ```
   Stats merging keeps counts/flags and clears non-additive QC fields.

4. **Merge artifacts**  
   Concatenate JSONL/prompt outputs (`cat`/`zcat`), and merge dataset-card fragments with:
   ```bash
   sievio card --fragments "shards/**/*.card.json" --output out/README.md
   ```

## Observability
- Each run writes a stats JSON (via run-summary hook) next to outputs; inspect counts/errors/QC summaries there.
- CLI emits JSON stats to stdout; capture logs for failures (`--log-level DEBUG` for more detail).
- QC/safety post passes can emit CSV/Parquet sidecars (see `docs/QUALITY_CONTROL.md`).

## Runtime knobs
- Concurrency: `pipeline.executor_kind` (`auto|thread|process`), `max_workers`, and `submit_window`.
- Remote safety: `http.*` (timeouts, redirect safeguards) and PDF/SQLite download caps.
- QC post-processing: `qc.parallel_post` and `qc.post_*` override concurrency for post passes.

## Footnotes
- Sharding currently targets source kinds registered in `SourceRegistry` (e.g., `github_zip`, `web_pdf_list`); use `--kind` accordingly.
- For very large PDFs/EVTX workloads, set `pipeline.executor_kind="process"` in the base config to bias toward process workers.
