# manual_test_github_toml.py
# SPDX-License-Identifier: MIT
"""
Sievio - manual smoke test (GitHub)
========================================

Run this from your workspace root (no install required). The script:

- Validates the GitHub URL up front
- Autonames outputs with SPDX license + optional ref + timestamp
- Loads a TOML-based SievioConfig and applies a few runtime-only tweaks
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
from dataclasses import replace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from sievio import SievioConfig, convert, load_config_from_path
from sievio.core.config import DEFAULT_QC_SCORER_ID, QCHeuristics
from sievio.cli.runner import default_paths_for_github, make_github_profile
from sievio.core.builder import PipelineOverrides
from sievio.core.chunk import ChunkPolicy
from sievio.core.interfaces import RepoContext
from sievio.core.log import configure_logging
from sievio.core.registries import quality_scorer_registry
from sievio.core.convert import DefaultExtractor
from sievio.sources.githubio import parse_github_url

# Optional extractor for KQL blocks inside Markdown
try:
    from sievio.core.extras.md_kql import KqlFromMarkdownExtractor
except Exception:  # pragma: no cover - optional extra
    KqlFromMarkdownExtractor = None  # type: ignore[assignment]

# QC extras probe (used to auto-disable QC when dependencies are missing)
try:
    import importlib

    importlib.import_module("sievio.core.extras.qc")
    QC_EXTRAS_AVAILABLE = True
except Exception:
    QC_EXTRAS_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────────────
# User-editable knobs for this manual test
# ──────────────────────────────────────────────────────────────────────────────

# Example GitHub repo to test:
# URL = "https://github.com/pallets/flask/tree/main/docs"
URL = "https://github.com/JochiRaider/URL_Research_Tool"
# URL = "https://github.com/Bert-JanP/Hunting-Queries-Detection-Rules"
# URL = "https://github.com/chinapandaman/PyPDFForm"
# URL = "https://github.com/SystemsApproach/book"
REF: Optional[str] = None  # e.g. "main", "v1.0.0", or a commit SHA (applied when URL has no ref)

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

# Force threads to avoid pickling issues with process executors.
PIPELINE_EXECUTOR = "thread"  # "thread" | "process" | "auto"

# Optional QC scorer override (keeps the config declarative; wired via registry)
USE_CUSTOM_QC_SCORER = True
QC_MODEL_ID = "Qwen/Qwen2.5-1.5B"
QC_DEVICE = "cuda"       # or "cpu"
QC_DTYPE = "bfloat16"    # or "float32" / "float16"
QC_LOCAL_ONLY = False
# Disable dedup (exact + global) for this manual run.
DISABLE_DEDUP = True


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _apply_ref(url: str, ref: Optional[str]) -> str:
    """
    Add a ref to a GitHub URL when one is not already present.

    default_paths_for_github/make_github_profile already honor refs baked into
    the URL (e.g., .../tree/<ref>); this helper only patches in REF when the
    URL points at the repo root.
    """
    if not ref:
        return url
    spec = parse_github_url(url)
    if spec is None or spec.ref:
        return url
    return f"https://github.com/{spec.owner}/{spec.repo}/tree/{ref}"


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
    base_cfg: SievioConfig,
) -> SievioConfig:
    """
    Take a TOML-loaded SievioConfig and apply runtime-only tweaks.

    The TOML file controls stable, serializable knobs (QC, concurrency, HTTP, etc.).
    This helper stitches in objects that are hard to express in TOML, such as:
    - the ChunkPolicy instance
    QC scorers are now registered via the registry system to keep the config runtime-free.
    """
    cfg = base_cfg

    # Runtime-only wiring that isn't TOML-friendly.
    cfg.chunk.policy = POLICY
    cfg.qc.scorer = None
    cfg.pipeline.executor_kind = PIPELINE_EXECUTOR
    if DISABLE_DEDUP:
        cfg.qc.exact_dedup = False
        scorer_opts = dict(getattr(cfg.qc, "scorer_options", {}) or {})
        # Heuristics mapping: disable MinHash and loosen SimHash gating.
        heur = dict(scorer_opts.get("heuristics") or {})
        heur["enable_minhash"] = False
        if "simhash_hamm_thresh" not in heur:
            heur["simhash_hamm_thresh"] = None  # use constructor default; dedup disabled via drop_near_dups
        scorer_opts["heuristics"] = heur
        scorer_opts["exact_dedup"] = False
        scorer_opts.setdefault("global_dedup", {"path": None, "read_only": False})
        cfg.qc.scorer_options = scorer_opts
    return cfg


def _maybe_disable_qc(cfg: SievioConfig, log) -> bool:
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


def _register_qc_factory(cfg: SievioConfig, log) -> None:
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

        def build(self, options: Mapping[str, Any]):
            try:
                from sievio.core.extras.qc import JSONLQualityScorer
            except Exception as exc:  # pragma: no cover - optional extra
                raise RuntimeError(
                    "QC is enabled in the config, but QC extras are not installed. "
                    "Disable qc.enabled in the TOML or install the QC dependencies."
                ) from exc

            opts = dict(options or {})
            heur_opt = opts.get("heuristics")
            heuristics: QCHeuristics | None
            if heur_opt is None:
                heuristics = None
            elif isinstance(heur_opt, QCHeuristics):
                heuristics = heur_opt
            elif isinstance(heur_opt, Mapping):
                heuristics = QCHeuristics(**dict(heur_opt))
            else:
                raise TypeError(
                    f"heuristics must be QCHeuristics or a mapping when using {self.id}; got {type(heur_opt).__name__}"
                )
            if heuristics is not None:
                heuristics.validate()

            global_opts = opts.get("global_dedup", {}) or {}
            return JSONLQualityScorer(
                lm_model_id=opts.get("lm_model_id", QC_MODEL_ID),
                device=opts.get("device", QC_DEVICE),
                dtype=opts.get("dtype", QC_DTYPE),
                local_files_only=bool(opts.get("local_files_only", QC_LOCAL_ONLY)),
                heuristics=heuristics,
                global_dedup_path=global_opts.get("path"),
                global_dedup_read_only=bool(global_opts.get("read_only", False)),
                exact_dedup=bool(opts.get("exact_dedup", True)),
            )

    quality_scorer_registry.register(ManualQualityScorerFactory())


class _RuntimeExtractor(DefaultExtractor):
    """
    FileExtractor that injects runtime-only Extractors without mutating the spec.
    """

    def __init__(self, extractors: Sequence[object]):
        super().__init__()
        self._runtime_extractors = tuple(extractors)

    def extract(self, item, *, config, context=None):
        try:
            pipeline_cfg = replace(config.pipeline, extractors=self._runtime_extractors)
            cfg = replace(config, pipeline=pipeline_cfg)
        except Exception:
            cfg = config
        return super().extract(item, config=cfg, context=context)


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
    if not isinstance(base_cfg, SievioConfig):
        raise TypeError(f"Expected SievioConfig from {CONFIG_PATH}, got {type(base_cfg)!r}")

    extractors: list[object] = []
    if ENABLE_KQL_MD_EXTRACTOR and KqlFromMarkdownExtractor is not None:
        extractors.append(KqlFromMarkdownExtractor())

    # Apply runtime-only wiring on top of the TOML-based config.
    run_cfg = _build_config(base_cfg)
    _maybe_disable_qc(run_cfg, log)
    _register_qc_factory(run_cfg, log)
    if getattr(run_cfg.qc, "enabled", False) and not getattr(run_cfg.qc, "scorer_id", None):
        run_cfg.qc.scorer_id = DEFAULT_QC_SCORER_ID

    # Allow repo_context to populate repo_url instead of the static TOML value.
    run_cfg.metadata.repo_url = None

    qc_enabled = bool(getattr(run_cfg.qc, "enabled", False))
    qc_mode = getattr(run_cfg.qc, "mode", "off")
    target_url = _apply_ref(URL, REF)

    log.info(
        "Converting repo → %s (autoname=%s, prompt=%s, qc=%s[%s])",
        target_url,
        True,
        ALSO_PROMPT_TEXT,
        "on" if qc_enabled else "off",
        qc_mode,
    )

    jsonl_path, prompt_path, ctx = _plan_output_paths(
        target_url,
        OUT_DIR,
        with_prompt=ALSO_PROMPT_TEXT,
    )

    profile_cfg = make_github_profile(
        target_url,
        str(jsonl_path),
        out_prompt=str(prompt_path) if prompt_path else None,
        base_config=run_cfg,
        repo_context=ctx,
    )
    profile_cfg.metadata.primary_jsonl = str(jsonl_path)
    if prompt_path:
        profile_cfg.metadata.prompt_path = str(prompt_path)
    profile_cfg.metadata.repo_url = ctx.repo_url or target_url

    overrides = None
    if extractors:
        overrides = PipelineOverrides(file_extractor=_RuntimeExtractor(extractors))

    stats = convert(profile_cfg, overrides=overrides)

    print("\n=== Done ===")
    print("JSONL :", jsonl_path)
    print("Prompt:", prompt_path)
    print("Stats :", stats)
    if qc_enabled:
        print("QC Summary:", stats.get("qc", {}))


if __name__ == "__main__":
    main()
