# manual_test_github_toml.py
# SPDX-License-Identifier: MIT
"""
RepoCapsule - manual smoke test (GitHub)
========================================

Run this from your workspace root (no install required). The script:

- Validates the GitHub URL up front
- Autonames outputs with SPDX license + optional ref + timestamp
- Loads a TOML-based RepocapsuleConfig and applies a few runtime-only tweaks
- Builds a GitHub profile config and runs `convert(...)`
- Treats KQL-from-Markdown as an optional Extractor
- Lets the pipeline own sink open/close

Environment: set GITHUB_TOKEN or GH_TOKEN to avoid GitHub API rate limits.
"""

from __future__ import annotations

# Allow running from a source checkout without installing the package.
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from pathlib import Path
from typing import Optional, Sequence

from repocapsule.log import configure_logging
from repocapsule import load_config_from_path, convert, RepocapsuleConfig
from repocapsule.chunk import ChunkPolicy
from repocapsule.interfaces import RepoContext
from repocapsule.runner import default_paths_for_github, make_github_profile

# Optional extractor for KQL blocks inside Markdown
try:
    from repocapsule.md_kql import KqlFromMarkdownExtractor
except Exception:  # pragma: no cover - optional extra
    KqlFromMarkdownExtractor = None  # type: ignore[assignment]

try:  # optional QC extra
    from repocapsule.qc import JSONLQualityScorer
except Exception:  # pragma: no cover - keep import errors silent
    JSONLQualityScorer = None  # type: ignore[assignment]

# ──────────────────────────────────────────────────────────────────────────────
# User-editable knobs for this manual test
# ──────────────────────────────────────────────────────────────────────────────

# Example GitHub repo to test:
# URL = "https://github.com/pallets/flask/tree/main/docs"
# URL = "https://github.com/JochiRaider/URL_Research_Tool"
# URL = "https://github.com/chinapandaman/PyPDFForm"
URL = "https://github.com/SystemsApproach/book"
REF: Optional[str] = None  # e.g. "main", "v1.0.0", or a commit SHA (only used for naming if spec.ref is None)

# Workspace root and output directory for artifacts (portable path under the repo root):
REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "out"

# Path to the TOML config used as a base for this manual test.
CONFIG_PATH = REPO_ROOT / "manual_test_github.toml"

# Markdown → KQL extraction (via Extractor; this just toggles whether we add it)
ENABLE_KQL_MD_EXTRACTOR = False

# Chunking policy: tweak as needed (not expressed in TOML yet)
POLICY = ChunkPolicy(mode="doc")  # , target_tokens=1700, overlap_tokens=40, min_tokens=400

# Write prompt text too? (affects how we plan output paths)
ALSO_PROMPT_TEXT = True


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _plan_output_paths(
    url: str,
    out_dir: Path,
    *,
    ref_hint: Optional[str],
    with_prompt: bool,
) -> tuple[Path, Optional[Path], RepoContext]:
    jsonl_str, prompt_str, ctx = default_paths_for_github(
        url,
        out_dir=out_dir,
        include_prompt=with_prompt,
    )
    jsonl_path = Path(jsonl_str)
    prompt_path = Path(prompt_str) if prompt_str is not None else None
    return jsonl_path, prompt_path, ctx


def _build_config(
    base_cfg: RepocapsuleConfig,
    extractors: Sequence[object],
) -> RepocapsuleConfig:
    """
    Take a TOML-loaded RepocapsuleConfig and apply runtime-only tweaks.

    The TOML file controls stable, serializable knobs (QC, concurrency, HTTP, etc.).
    This helper stitches in objects that are hard to express in TOML, such as:
    - the ChunkPolicy instance
    - any custom Extractors (e.g., KqlFromMarkdownExtractor)
    """
    cfg = base_cfg

    # Runtime-only wiring that isn't TOML-friendly.
    cfg.chunk.policy = POLICY
    cfg.pipeline.extractors = tuple(extractors)
   
    if cfg.qc.enabled:
        if JSONLQualityScorer is None:
            raise RuntimeError(
                "QC is enabled in the config, but QC extras are not installed. "
                "Disable qc.enabled in the TOML or install the QC dependencies."
            )
        cfg.qc.scorer = JSONLQualityScorer(
            lm_model_id="Qwen/Qwen2.5-1.5B",
            device="cuda",        # or "cpu"
            dtype="bfloat16",     # or "float32" / "float16"
            local_files_only=False,
            heuristics=getattr(cfg.qc, "heuristics", None),
        )
    else:
        cfg.qc.scorer = None
    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    # Optional: set a token to avoid low GitHub rate limits
    # os.environ.setdefault("GITHUB_TOKEN", "<your token>")

    log = configure_logging(level="INFO")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load base config from TOML.
    base_cfg = load_config_from_path(CONFIG_PATH)
    if not isinstance(base_cfg, RepocapsuleConfig):
        raise TypeError(f"Expected RepocapsuleConfig from {CONFIG_PATH}, got {type(base_cfg)!r}")

    log.info(
        "Converting repo → %s (autoname=%s, prompt=%s, qc=%s[%s])",
        URL,
        True,
        ALSO_PROMPT_TEXT,
        "on" if getattr(base_cfg.qc, "enabled", False) else "off",
        getattr(base_cfg.qc, "mode", "off"),
    )

    jsonl_path, prompt_path, ctx = _plan_output_paths(
        URL,
        OUT_DIR,
        ref_hint=REF,
        with_prompt=ALSO_PROMPT_TEXT,
    )

    extractors: list[object] = []
    if ENABLE_KQL_MD_EXTRACTOR and KqlFromMarkdownExtractor is not None:
        extractors.append(KqlFromMarkdownExtractor())

    # Apply runtime-only wiring on top of the TOML-based config.
    run_cfg = _build_config(base_cfg, extractors)

    profile_cfg = make_github_profile(
        URL,
        str(jsonl_path),
        out_prompt=str(prompt_path) if prompt_path else None,
        base_config=run_cfg,
        repo_context=ctx,
    )
    stats = convert(profile_cfg)

    print("\n=== Done ===")
    print("JSONL :", jsonl_path)
    print("Prompt:", prompt_path)
    print("Stats :", stats)
    if getattr(base_cfg.qc, "enabled", False):
        print("QC Summary:", stats.get("qc", {}))


if __name__ == "__main__":
    main()
