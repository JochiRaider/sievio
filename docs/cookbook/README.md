# Sievio Cookbook

Task-focused examples grounded in existing APIs and configs. Recipes:

- `pdf_ingestion.md` – ingest a list of remote PDFs (requires `[pdf]` extra).
- `custom_pii_scrubbing.md` – add a lightweight record middleware to scrub PII-like patterns.
- `dedup_post_qc.md` – seed a global MinHash store and run post-QC deduplication (requires `[qc]` extra).

Run commands are regular `sievio` CLI invocations; ensure required extras are installed before trying a recipe.
