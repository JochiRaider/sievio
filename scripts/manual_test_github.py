# manual_test_github.py
# SPDX-License-Identifier: MIT
"""
RepoCapsule - manual smoke test (updated for current API).

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
import os

from repocapsule.log import configure_logging
# Current public API surface (re-exported at package level)
from repocapsule import (
    RepocapsuleConfig,
    parse_github_url,
    get_repo_license_spdx,
    build_output_basename_github,
    convert_github,
)
from repocapsule.chunk import ChunkPolicy

# Optional extractor for KQL blocks inside Markdown
try:
    from repocapsule.md_kql import KqlFromMarkdownExtractor
except Exception:
    KqlFromMarkdownExtractor = None  # type: ignore

# ──────────────────────────────────────────────────────────────────────────────
# User-editable knobs for this manual test
# ──────────────────────────────────────────────────────────────────────────────

# Example GitHub repo to test:
URL = "https://github.com/microsoft/Microsoft-365-Defender-Hunting-Queries"
# URL = "https://github.com/chinapandaman/PyPDFForm"
# URL = "https://github.com/SystemsApproach/book"
REF: Optional[str] = None  # e.g. "main", "v1.0.0", or a commit SHA (only used for naming if spec.ref is None)

# Output directory for artifacts:
OUT_DIR = Path(r"C:\Users\wetou\OneDrive\Documents\projects\out\test")

# Markdown → KQL extraction (now via Extractor; this just toggles whether we add it)
ENABLE_KQL_MD_EXTRACTOR = True

# Chunking policy: tweak as needed
POLICY = ChunkPolicy(mode="auto")  # , target_tokens=1700, overlap_tokens=40, min_tokens=400

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

    base = build_output_basename_github(
        owner=spec.owner,
        repo=spec.repo,
        ref=(spec.ref or ref_hint or "main"),
        license_spdx=spdx,
    )
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl = out_dir / f"{base}__{ts}.jsonl"
    prompt = (out_dir / f"{base}__{ts}.prompt.txt") if with_prompt else None
    return jsonl, prompt


def _build_config(extractors: Sequence[object]) -> RepocapsuleConfig:
    cfg = RepocapsuleConfig()
    cfg.chunk.policy = POLICY
    cfg.pipeline.extractors = tuple(extractors)
    cfg.sources.github.per_file_cap = int(MAX_FILE_MB) * 1024 * 1024
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

    # Optional QC pass (best-effort; will skip if deps are missing)
    try:
        from repocapsule.qc import score_jsonl_to_csv
        qc_csv = score_jsonl_to_csv(
            str(jsonl_path),
            lm_model_id="Qwen/Qwen2.5-1.5B",
            device="cuda",            # or "cpu"
            dtype="bfloat16",         # or "float32" / "float16"
            local_files_only=False,
            simhash_hamm_thresh=2,
        )
        print("QC CSV:", qc_csv)
    except Exception as e:
        print("QC skipped:", e)


if __name__ == "__main__":
    main()
