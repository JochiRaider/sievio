# Sievio Documentation

Start here to find the right doc for your role.

- **Technical manual:** `docs/TECHNICAL_MANUAL.md` – theory of operation, architecture overview, and end-to-end workflow.
- **Configuration reference:** `docs/CONFIGURATION.md` – generated from the actual `SievioConfig` dataclasses; run `python3 scripts/generate_configuration_md.py` to refresh.
- **Quality control:** `docs/QUALITY_CONTROL.md` – QC/safety signals, modes, tuning, and exports.
- **Deployment & sharding:** `docs/DEPLOYMENT.md` – running shards, merging stats, and operational tips.
- **Cookbook:** `docs/cookbook/` – task-focused recipes (PDF ingestion, custom scrubbing, dedup/QC workflows).
- **Contributing:** `docs/CONTRIBUTING.md` – links to the canonical `CONTRIBUTING.md` (see also `AGENTS.md` and `LLMS.md`).

Other references:
- Root README: `README.md` – quick intro and usage snippets.
- Example config: `example_config.toml` – comprehensive example with defaults.
- Architecture maps: `LLMS.md` – module responsibilities and core vs non-core guidance.
