# Contributing to Sievio

This file exists so documentation links remain stable.

The canonical contributor guide lives at `CONTRIBUTING.md` in the repo root. Please read that file for:
- Setup and required checks (`PYTHONPATH=src pytest`, `ruff check .`, `mypy --config-file pyproject.toml src`)
- Extension patterns (sources/sinks/bytes handlers/QC via registries/plugins)
- Safety boundaries (use `SafeHttpClient`; keep configs runtime-free)

If you are using an AI coding assistant, also read `AGENTS.md` and `LLMS.md` first.
