from __future__ import annotations

# Allow running from a source checkout without installing the package.
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from pathlib import Path
from typing import Optional, Sequence

from repocapsule.log import configure_logging
from repocapsule import RepocapsuleConfig
from repocapsule.chunk import ChunkPolicy
from repocapsule.config import QCMode
from repocapsule.sources_webpdf import WebPdfListSource, WebPagePdfSource
from repocapsule.interfaces import RepoContext
from repocapsule.factories import build_default_sinks
from repocapsule.runner import convert, default_paths_for_pdf, _finalize_profile

try:
    from repocapsule.qc import JSONLQualityScorer
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
PAGE_URL: Optional[str] = None  # e.g., "https://example.org/resources"

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "out"

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
MAX_WORKERS: Optional[int] = None
SUBMIT_WINDOW: Optional[int] = None

# Prompt output?
ALSO_PROMPT_TEXT = True

# PDF fetch/streaming limits
MAX_PDF_MB = 100
MAX_DECODE_MB: Optional[int] = None  # cap decode stage per file (MiB)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_source() -> WebPdfListSource | WebPagePdfSource:
    max_pdf_bytes = int(MAX_PDF_MB) * 1024 * 1024
    if PAGE_URL:
        return WebPagePdfSource(
            PAGE_URL,
            same_domain=True,
            max_links=50,
            include_ambiguous=False,
            add_prefix="webpdfs",
            max_pdf_bytes=max_pdf_bytes,
            require_pdf=True,
        )
    if not URLS:
        raise ValueError("URLS must contain at least one PDF when PAGE_URL is not set.")
    return WebPdfListSource(
        URLS,
        max_pdf_bytes=max_pdf_bytes,
        require_pdf=True,
        add_prefix="webpdfs",
    )


def _build_config() -> RepocapsuleConfig:
    cfg = RepocapsuleConfig()
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
        if JSONLQualityScorer is None:
            raise RuntimeError("QC requested but optional QC extras are not installed.")
        cfg.qc.scorer = JSONLQualityScorer(
            lm_model_id="Qwen/Qwen2.5-1.5B",
            device="cuda",        # or "cpu"
            dtype="bfloat16",     # or "float32" / "float16"
            local_files_only=False,
        )
        cfg.qc.mode = QC_MODE
        cfg.qc.min_score = QC_MIN_SCORE
        cfg.qc.drop_near_dups = QC_DROP_NEAR_DUPS
        cfg.qc.write_csv = QC_WRITE_CSV
        cfg.qc.csv_suffix = QC_CSV_SUFFIX
    else:
        cfg.qc.scorer = None
    return cfg


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

    cfg = _build_config()

    sources: Sequence[object] = (_build_source(),)
    ctx = RepoContext(
        repo_full_name=None,
        repo_url=corpus_url,
        license_id=None,
        extra={"source": "webpdf"},
    )
    sink_result = build_default_sinks(
        cfg.sinks,
        jsonl_path=jsonl_path,
        prompt_path=prompt_path,
        context=ctx,
    )
    cfg = _finalize_profile(cfg, sources, sink_result, extra_metadata={"source": "webpdf"})

    stats = convert(cfg)

    print("\n=== Done ===")
    print("JSONL :", jsonl_path)
    print("Prompt:", prompt_path)
    print("Stats :", stats)
    qc_summary = stats.get("qc")
    if qc_summary:
        print("QC Summary:", qc_summary)


if __name__ == "__main__":
    main()
