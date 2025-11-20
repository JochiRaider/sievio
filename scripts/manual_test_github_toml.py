# manual_test_github_toml.py
# SPDX-License-Identifier: MIT
"""
RepoCapsule - manual smoke test (GitHub)
========================================

Run this from your workspace root (no install required). The script:

- Validates the GitHub URL up front
- Autonames outputs with SPDX license + optional ref + timestamp
- Loads a TOML-based RepocapsuleConfig and applies a few runtime-only tweaks
- Registers an optional QC scorer via the new registry system (keeps the config runtime-free)
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

from repocapsule import RepocapsuleConfig, convert, load_config_from_path
from repocapsule.cli.runner import default_paths_for_github, make_github_profile
from repocapsule.core.builder import build_pipeline_plan
from repocapsule.core.chunk import ChunkPolicy
from repocapsule.core.interfaces import RepoContext
from repocapsule.core.log import configure_logging
from repocapsule.core.pipeline import PipelineEngine
from repocapsule.core.registries import quality_scorer_registry

# Optional extractor for KQL blocks inside Markdown
try:
    from repocapsule.core.extras.md_kql import KqlFromMarkdownExtractor
except Exception:  # pragma: no cover - optional extra
    KqlFromMarkdownExtractor = None  # type: ignore[assignment]

# QC extras probe (used to auto-disable QC when dependencies are missing)
try:
    import importlib

    importlib.import_module("repocapsule.core.extras.qc")
    QC_EXTRAS_AVAILABLE = True
except Exception:
    QC_EXTRAS_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────────────
# User-editable knobs for this manual test
# ──────────────────────────────────────────────────────────────────────────────

# Example GitHub repo to test:
# URL = "https://github.com/pallets/flask/tree/main/docs"
URL = "https://github.com/JochiRaider/URL_Research_Tool"
# URL = "https://github.com/chinapandaman/PyPDFForm"
# URL = "https://github.com/SystemsApproach/book"
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

# Optional QC scorer override (keeps the config declarative; wired via registry)
USE_CUSTOM_QC_SCORER = True
QC_MODEL_ID = "Qwen/Qwen2.5-1.5B"
QC_DEVICE = "cuda"       # or "cpu"
QC_DTYPE = "bfloat16"    # or "float32" / "float16"
QC_LOCAL_ONLY = False


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _plan_output_paths(
    url: str,
    out_dir: Path,
    *,
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
    QC scorers are now registered via the registry system to keep the config runtime-free.
    """
    cfg = base_cfg

    # Runtime-only wiring that isn't TOML-friendly.
    cfg.chunk.policy = POLICY
    cfg.pipeline.extractors = tuple(extractors)
    cfg.qc.scorer = None
    return cfg


def _maybe_disable_qc(cfg: RepocapsuleConfig, log) -> bool:
    """
    Turn off QC if the optional QC extras are missing to avoid runtime errors.
    """
    if not getattr(cfg.qc, "enabled", False):
        return False
    if QC_EXTRAS_AVAILABLE:
        return False
    log.warning("QC enabled but QC extras are not installed; disabling QC for this run.")
    cfg.qc.enabled = False
    cfg.qc.mode = "off"
    cfg.qc.scorer = None
    return True


def _register_qc_factory(cfg: RepocapsuleConfig, log) -> None:
    """
    Register a QC scorer factory when QC is enabled.

    The registry keeps the declarative config free of runtime objects. Registering
    the factory before ``convert(...)`` ensures it wins over the default factory.
    """
    if not (getattr(cfg.qc, "enabled", False) and USE_CUSTOM_QC_SCORER):
        return
    if not QC_EXTRAS_AVAILABLE:
        log.warning("QC extras are not installed; skipping custom QC scorer registration.")
        return

    class ManualQualityScorerFactory:
        id = "jsonl_default"  # override the default factory

        def build(self, qc_cfg):
            try:
                from repocapsule.core.extras.qc import JSONLQualityScorer
            except Exception as exc:  # pragma: no cover - optional extra
                raise RuntimeError(
                    "QC is enabled in the config, but QC extras are not installed. "
                    "Disable qc.enabled in the TOML or install the QC dependencies."
                ) from exc
            return JSONLQualityScorer(
                lm_model_id=QC_MODEL_ID,
                device=QC_DEVICE,
                dtype=QC_DTYPE,
                local_files_only=QC_LOCAL_ONLY,
                heuristics=getattr(qc_cfg, "heuristics", None),
            )

    quality_scorer_registry.register(ManualQualityScorerFactory())


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

    extractors: list[object] = []
    if ENABLE_KQL_MD_EXTRACTOR and KqlFromMarkdownExtractor is not None:
        extractors.append(KqlFromMarkdownExtractor())

    # Apply runtime-only wiring on top of the TOML-based config.
    run_cfg = _build_config(base_cfg, extractors)
    _maybe_disable_qc(run_cfg, log)
    _register_qc_factory(run_cfg, log)

    qc_enabled = bool(getattr(run_cfg.qc, "enabled", False))
    qc_mode = getattr(run_cfg.qc, "mode", "off")

    log.info(
        "Converting repo → %s (autoname=%s, prompt=%s, qc=%s[%s])",
        URL,
        True,
        ALSO_PROMPT_TEXT,
        "on" if qc_enabled else "off",
        qc_mode,
    )

    jsonl_path, prompt_path, ctx = _plan_output_paths(
        URL,
        OUT_DIR,
        with_prompt=ALSO_PROMPT_TEXT,
    )

    profile_cfg = make_github_profile(
        URL,
        str(jsonl_path),
        out_prompt=str(prompt_path) if prompt_path else None,
        base_config=run_cfg,
        repo_context=ctx,
    )
    plan = build_pipeline_plan(profile_cfg, scorer_registry=quality_scorer_registry)
    engine = PipelineEngine(plan)
    stats = convert(engine)

    print("\n=== Done ===")
    print("JSONL :", jsonl_path)
    print("Prompt:", prompt_path)
    print("Stats :", stats)
    if qc_enabled:
        print("QC Summary:", stats.get("qc", {}))


if __name__ == "__main__":
    main()
