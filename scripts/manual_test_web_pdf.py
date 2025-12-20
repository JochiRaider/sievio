from __future__ import annotations

import pathlib

# Allow running from a source checkout without installing the package.
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from pathlib import Path

from sievio import SievioConfig, convert, load_config_from_path
from sievio.cli.runner import default_paths_for_pdf
from sievio.core.builder import build_pipeline_plan
from sievio.core.chunk import ChunkPolicy
from sievio.core.config import QCMode, SinkSpec, SourceSpec
from sievio.core.interfaces import RepoContext
from sievio.core.log import configure_logging
from sievio.core.pipeline import PipelineEngine
from sievio.core.registries import quality_scorer_registry

try:
    from sievio.core.extras.qc import JSONLQualityScorer
except Exception:  # optional extras
    JSONLQualityScorer = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# User-editable knobs
# ---------------------------------------------------------------------------

# Direct PDF URLs to fetch (used when PAGE_URL is not set).
URLS = [
    "https://lamport.azurewebsites.net/pubs/time-clocks.pdf",
]

# Alternatively, set PAGE_URL to scrape a single page for PDF links.
PAGE_URL: str | None = None  # e.g., "https://example.org/resources"

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "out"
# Path to the TOML config used as a base for this manual test.
CONFIG_PATH = REPO_ROOT / "manual_test_github.toml"

# Chunking policy (adjust token targets/overlap as desired)
POLICY = ChunkPolicy(mode="doc")  # , target_tokens=1700, overlap_tokens=40, min_tokens=400

# QC knobs
ENABLE_QC = True
QC_MODE = QCMode.INLINE  # or QCMode.POST / QCMode.ADVISORY
QC_MIN_SCORE = 40.0
QC_DROP_NEAR_DUPS = False
QC_WRITE_CSV = True
QC_CSV_SUFFIX = "_quality.csv"

# Concurrency
EXECUTOR_KIND = "thread"     # "thread" or "process"
MAX_WORKERS: int | None = None
SUBMIT_WINDOW: int | None = None

# Prompt output?
ALSO_PROMPT_TEXT = True

# PDF fetch/streaming limits
MAX_PDF_MB = 100
MAX_DECODE_MB: int | None = None  # cap decode stage per file (MiB)

# QC extras probe (to avoid hard crashes when optional deps are missing)
try:
    import importlib

    importlib.import_module("sievio.core.extras.qc")
    QC_EXTRAS_AVAILABLE = True
except Exception:
    QC_EXTRAS_AVAILABLE = False

# Optional QC scorer override (to enable perplexity/LM-backed scoring)
USE_CUSTOM_QC_SCORER = True
QC_MODEL_ID = "Qwen/Qwen2.5-1.5B"
QC_DEVICE = "cuda"       # or "cpu"
QC_DTYPE = "bfloat16"    # or "float32" / "float16"
QC_LOCAL_ONLY = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_config(base_cfg: SievioConfig) -> SievioConfig:
    cfg = base_cfg
    cfg.chunk.policy = POLICY

    if MAX_DECODE_MB is not None:
        cfg.decode.max_bytes_per_file = int(MAX_DECODE_MB) * 1024 * 1024

    cfg.pipeline.executor_kind = EXECUTOR_KIND
    if MAX_WORKERS is not None:
        cfg.pipeline.max_workers = MAX_WORKERS
    if SUBMIT_WINDOW is not None:
        cfg.pipeline.submit_window = SUBMIT_WINDOW

    cfg.qc.enabled = bool(ENABLE_QC)
    if cfg.qc.enabled:
        cfg.qc.mode = QC_MODE
        cfg.qc.min_score = QC_MIN_SCORE
        cfg.qc.drop_near_dups = QC_DROP_NEAR_DUPS
        cfg.qc.write_csv = QC_WRITE_CSV
        cfg.qc.csv_suffix = QC_CSV_SUFFIX
    else:
        cfg.qc.scorer = None
    return cfg


def _maybe_disable_qc(cfg: SievioConfig, log) -> None:
    if getattr(cfg.qc, "enabled", False) and not QC_EXTRAS_AVAILABLE:
        log.warning("QC enabled but QC extras are not installed; disabling QC for this run.")
        cfg.qc.enabled = False
        cfg.qc.mode = QCMode.OFF
        cfg.qc.scorer = None


def _register_qc_factory(cfg: SievioConfig, log) -> None:
    if not (getattr(cfg.qc, "enabled", False) and USE_CUSTOM_QC_SCORER):
        return
    if not QC_EXTRAS_AVAILABLE:
        log.warning("QC extras are not installed; skipping custom QC scorer registration.")
        return

    class ManualQualityScorerFactory:
        id = "jsonl_default"  # override the default factory

        def build(self, qc_cfg):
            from sievio.core.extras.qc import JSONLQualityScorer

            return JSONLQualityScorer(
                lm_model_id=QC_MODEL_ID,
                device=QC_DEVICE,
                dtype=QC_DTYPE,
                local_files_only=QC_LOCAL_ONLY,
                heuristics=getattr(qc_cfg, "heuristics", None),
            )

    quality_scorer_registry.register(ManualQualityScorerFactory())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log = configure_logging(level="INFO")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    corpus_url = PAGE_URL or (URLS[0] if URLS else "webpdfs")
    jsonl_str, prompt_str = default_paths_for_pdf(
        url=corpus_url,
        title=None,
        license_spdx=None,
        out_dir=OUT_DIR,
        include_prompt=ALSO_PROMPT_TEXT,
    )
    jsonl_path = Path(jsonl_str)
    prompt_path = Path(prompt_str) if prompt_str else None

    log.info(
        "Converting web PDFs → %s (prompt=%s, qc=%s[%s])",
        jsonl_path,
        bool(prompt_path),
        "on" if ENABLE_QC else "off",
        QC_MODE,
    )

    base_cfg = load_config_from_path(CONFIG_PATH)
    if not isinstance(base_cfg, SievioConfig):
        raise TypeError(f"Expected SievioConfig from {CONFIG_PATH}, got {type(base_cfg)!r}")

    cfg = _build_config(base_cfg)
    _maybe_disable_qc(cfg, log)
    _register_qc_factory(cfg, log)
    ctx = RepoContext(repo_url=corpus_url, extra={"source": "webpdf"})

    # Configure sources/sinks via specs so the builder wires them up.
    add_prefix = "webpdfs"
    if PAGE_URL:
        cfg.sources.specs = [
            SourceSpec(kind="web_page_pdf", options={"page_url": PAGE_URL, "add_prefix": add_prefix}),
        ]
    else:
        if not URLS:
            raise ValueError("URLS must contain at least one PDF when PAGE_URL is not set.")
        cfg.sources.specs = [
            SourceSpec(kind="web_pdf_list", options={"urls": URLS, "add_prefix": add_prefix}),
        ]
    cfg.sinks.specs = [
        SinkSpec(
            kind="default_jsonl_prompt",
            options={"jsonl_path": str(jsonl_path), "prompt_path": str(prompt_path) if prompt_path else None},
        )
    ]
    cfg.sinks.context = ctx

    plan = build_pipeline_plan(cfg, scorer_registry=quality_scorer_registry)
    engine = PipelineEngine(plan)
    stats = convert(engine)

    print("\n=== Done ===")
    print("JSONL :", jsonl_path)
    print("Prompt:", prompt_path)
    print("Stats :", stats)
    qc_summary = stats.get("qc")
    if qc_summary:
        print("QC Summary:", qc_summary)


if __name__ == "__main__":
    main()
