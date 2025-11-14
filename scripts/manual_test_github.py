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
from datetime import datetime

from repocapsule.log import configure_logging
from repocapsule import RepocapsuleConfig, parse_github_url, get_repo_license_spdx, convert_github
from repocapsule.chunk import ChunkPolicy
from repocapsule.factories import make_output_paths_for_github

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

# Inline QC (requires optional torch/transformers/tiktoken extras)
ENABLE_INLINE_QC = True
QC_MIN_SCORE = 40.0
QC_DROP_NEAR_DUPS = False
QC_WRITE_CSV = True
QC_CSV_SUFFIX = "_quality.csv"

# Post-QC (summaries/CSV after JSONL write)
ENABLE_POST_QC = False

# Chunking policy: tweak as needed
POLICY = ChunkPolicy(mode="doc")  # , target_tokens=1700, overlap_tokens=40, min_tokens=400

# Write prompt text too?
ALSO_PROMPT_TEXT = True

# Per-file byte cap for entries inside the GitHub zipball (MiB)
MAX_FILE_MB = 50


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _plan_output_paths(url: str, out_dir: Path, *, ref_hint: str | None, with_prompt: bool) -> tuple[Path, Path | None]:
    spec = parse_github_url(url)
    if not spec:
        raise ValueError(f"Not a valid GitHub URL: {url!r}")
    # Best-effort license lookup for filename
    spdx = None
    try:
        spdx = get_repo_license_spdx(spec)  # may be None
    except Exception:
        pass

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outputs = make_output_paths_for_github(
        owner=spec.owner,
        repo=spec.repo,
        ref=(spec.ref or ref_hint or "main"),
        license_spdx=spdx,
        out_dir=out_dir,
        include_prompt=with_prompt,
        timestamp=timestamp,
    )
    return outputs.jsonl, outputs.prompt


def _build_config(extractors: Sequence[object]) -> RepocapsuleConfig:
    cfg = RepocapsuleConfig()
    cfg.chunk.policy = POLICY
    cfg.sources.github.include_exts = {".rst"}
    cfg.pipeline.extractors = tuple(extractors)
    cfg.sources.github.per_file_cap = int(MAX_FILE_MB) * 1024 * 1024
    if ENABLE_INLINE_QC:
        if JSONLQualityScorer is None:
            raise RuntimeError("Inline QC requested but optional QC extras are not installed.")
        cfg.qc.scorer = JSONLQualityScorer(
            lm_model_id="Qwen/Qwen2.5-1.5B",
            device="cuda",        # or "cpu"
            dtype="bfloat16",     # or "float32" / "float16"
            local_files_only=False,
        )
        cfg.qc.enabled = True
        cfg.qc.mode = "inline"
        cfg.qc.min_score = QC_MIN_SCORE
        cfg.qc.drop_near_dups = QC_DROP_NEAR_DUPS
        cfg.qc.write_csv = QC_WRITE_CSV
        cfg.qc.csv_suffix = QC_CSV_SUFFIX
    else:
        cfg.qc.enabled = False
    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # Optional: set a token to avoid low GitHub rate limits
    # os.environ.setdefault("GITHUB_TOKEN", "<your token>")

    log = configure_logging(level="INFO")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Converting repo → %s (autoname=%s, prompt=%s)", URL, True, ALSO_PROMPT_TEXT)

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

    if ENABLE_POST_QC:
        _run_post_qc(jsonl_path)


def _run_post_qc(jsonl_path: Path) -> None:
    if score_jsonl_to_csv is None:
        print("Post-QC skipped: optional QC extras are not installed.")
        return

    try:
        qc_csv = score_jsonl_to_csv(
            str(jsonl_path),
            lm_model_id="Qwen/Qwen2.5-1.5B",
            device="cuda",            # or "cpu"
            dtype="bfloat16",         # or "float32" / "float16"
            local_files_only=False,
            simhash_hamm_thresh=2,
        )
        print("Post-QC CSV:", qc_csv)
    except Exception as e:
        print("Post-QC skipped:", e)


if __name__ == "__main__":
    main()
