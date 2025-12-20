# manual_test_github.py
# SPDX-License-Identifier: MIT
"""
Minimal GitHub smoke test.

Run from the repo root without installing the package. Outputs
land in ./out/<license>/<repo>/ with prompt text enabled by default.

Set GITHUB_TOKEN or GH_TOKEN to avoid rate limiting.
"""

from __future__ import annotations

# Allow running from a source checkout without installing the package.
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from pathlib import Path

from sievio import SievioConfig, configure_logging, convert_github
from sievio.cli.runner import default_paths_for_github
from sievio.core.config import DEFAULT_QC_SCORER_ID

# Default repo to try; override with SIEVIO_GH_URL env var.
REPO_URL = "https://github.com/JochiRaider/URL_Research_Tool"
OUT_DIR = Path(__file__).resolve().parents[1] / "out" / "min_test"


def build_config() -> SievioConfig:
    """Return a minimally tweaked config suitable for smoke testing."""
    cfg = SievioConfig()
    cfg.sinks.prompt.include_prompt_file = True
    cfg.qc.enabled = True
    cfg.qc.mode = cfg.qc.normalize_mode()  # keep default inline gating
    cfg.qc.scorer_id = DEFAULT_QC_SCORER_ID
    cfg.qc.scorer_options["lm_model_id"] = None  # use default scorer without perplexity
    cfg.qc.write_csv = True
    return cfg


def main() -> None:
    log = configure_logging(level="INFO")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    jsonl_path, prompt_path, _ctx = default_paths_for_github(REPO_URL, out_dir=OUT_DIR, include_prompt=True)

    log.info("Converting repo %s → %s (prompt=%s)", REPO_URL, jsonl_path, bool(prompt_path))
    cfg = build_config()
    stats = convert_github(
        REPO_URL,
        str(jsonl_path),
        out_prompt=(str(prompt_path) if prompt_path else None),
        base_config=cfg,
    )

    print("\n=== Done ===")
    print("JSONL :", jsonl_path)
    print("Prompt:", prompt_path)
    print("Stats :", stats)


if __name__ == "__main__":
    main()
