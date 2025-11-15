# manual_test_github.py
# SPDX-License-Identifier: MIT
"""
RepoCapsule - manual smoke test

Run this from your workspace root (no install required). The script:
- Validates the GitHub URL up front
- Autonames outputs with SPDX license + optional ref + timestamp
- Builds a `RepocapsuleConfig` and passes it into `convert_github(...)`
- Treats KQL-from-Markdown as an optional Extractor
- Lets the pipeline own sink open/close
- Optionally runs QC if extras are available

Environment: set GITHUB_TOKEN or GH_TOKEN to avoid GitHub API rate limits.
"""

from __future__ import annotations

# Allow running from a source checkout without installing the package.
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from pathlib import Path
from typing import Optional, Sequence

from repocapsule.log import configure_logging
from repocapsule import RepocapsuleConfig, convert_github
from repocapsule.chunk import ChunkPolicy
from repocapsule.config import QCMode
from repocapsule.runner import default_paths_for_github

try:
    from repocapsule.qc import JSONLQualityScorer, score_jsonl_to_csv
except Exception:
    JSONLQualityScorer = None  # type: ignore[assignment]
    score_jsonl_to_csv = None  # type: ignore[assignment]

# Optional extractor for KQL blocks inside Markdown
try:
    from repocapsule.md_kql import KqlFromMarkdownExtractor
except Exception:
    KqlFromMarkdownExtractor = None  # type: ignore

# ──────────────────────────────────────────────────────────────────────────────
# User-editable knobs for this manual test
# ──────────────────────────────────────────────────────────────────────────────

# Example GitHub repo to test:
# URL = "https://github.com/pallets/flask/tree/main/docs"
# URL = "https://github.com/JochiRaider/URL_Research_Tool"
# URL = "https://github.com/chinapandaman/PyPDFForm"
URL = "https://github.com/SystemsApproach/book"
REF: Optional[str] = None  # e.g. "main", "v1.0.0", or a commit SHA (only used for naming if spec.ref is None)

# Output directory for artifacts (portable path under the repo root):
REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "out" 

# Markdown → KQL extraction (now via Extractor; this just toggles whether we add it)
ENABLE_KQL_MD_EXTRACTOR = False

# Inline/post QC (requires optional torch/transformers/tiktoken extras)
ENABLE_QC = True
QC_MODE = QCMode.INLINE  # or QCMode.POST / QCMode.ADVISORY
QC_MIN_SCORE = 40.0
QC_DROP_NEAR_DUPS = False
QC_WRITE_CSV = True
QC_CSV_SUFFIX = "_quality.csv"

# Chunking policy: tweak as needed
POLICY = ChunkPolicy(mode="doc")  # , target_tokens=1700, overlap_tokens=40, min_tokens=400

# Write prompt text too?
ALSO_PROMPT_TEXT = True

# Per-file byte cap for entries inside the GitHub zipball (MiB)
MAX_FILE_MB = 50

# Decode prefix cap for very large files (MiB); None => no extra cap.
MAX_DECODE_MB: int | None = None  # e.g. 50 to match MAX_FILE_MB

# Concurrency knobs
EXECUTOR_KIND = "thread"     # "thread" or "process"
MAX_WORKERS: int | None = None  # None/0 => let library pick
SUBMIT_WINDOW: int | None = None  # or an int, e.g., 100


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _plan_output_paths(url: str, out_dir: Path, *, ref_hint: str | None, with_prompt: bool) -> tuple[Path, Path | None]:
    jsonl_str, prompt_str, _ctx = default_paths_for_github(
        url,
        out_dir=out_dir,
        include_prompt=with_prompt,
    )
    jsonl_path = Path(jsonl_str)
    prompt_path = Path(prompt_str) if prompt_str is not None else None
    return jsonl_path, prompt_path


def _build_config(extractors: Sequence[object]) -> RepocapsuleConfig:
    cfg = RepocapsuleConfig()
    cfg.chunk.policy = POLICY
    cfg.sources.github.include_exts = {".rst"}
    cfg.pipeline.extractors = tuple(extractors)
    cfg.sources.github.per_file_cap = int(MAX_FILE_MB) * 1024 * 1024
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
        if QC_MODE == QCMode.POST:
            cfg.qc.parallel_post = True
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

    log.info(
        "Converting repo → %s (autoname=%s, prompt=%s, qc=%s[%s])",
        URL,
        True,
        ALSO_PROMPT_TEXT,
        "on" if ENABLE_QC else "off",
        QC_MODE,
    )

    jsonl_path, prompt_path = _plan_output_paths(URL, OUT_DIR, ref_hint=REF, with_prompt=ALSO_PROMPT_TEXT)

    extractors = []
    if ENABLE_KQL_MD_EXTRACTOR and KqlFromMarkdownExtractor is not None:
        extractors.append(KqlFromMarkdownExtractor())

    base_config = _build_config(extractors)

    stats = convert_github(
        URL,
        str(jsonl_path),
        out_prompt=(str(prompt_path) if prompt_path else None),
        base_config=base_config,
    )

    print("\n=== Done ===")
    print("JSONL :", jsonl_path)
    print("Prompt:", prompt_path)
    print("Stats :", stats)
    if ENABLE_QC:
        print("QC Summary:", stats.get("qc", {}))


if __name__ == "__main__":
    main()
