"""Minimal driver to exercise repocapsule on sysmon-modular.

Run from VS Code (or `python scripts/test_sysmon_modular.py`). No CLI needed.
- Streams the GitHub zipball (RAM-safe)
- Writes both JSONL and prompt text in a single pass
- Optionally runs light QC scoring if available
"""
from __future__ import annotations

# put these 3 lines at the very top
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))


from pathlib import Path
from typing import Optional, Set
import os

from repocapsule.log import configure_logging

# Prefer streaming implementations if present
try:  # pragma: no cover
    from repocapsule.convert_streaming import (
        convert_github_url_to_jsonl_autoname,
    )
except Exception:  # fallback to non-streaming convert (still works)
    from repocapsule.convert import (
        convert_github_url_to_jsonl_autoname,
    )

from repocapsule.chunk import ChunkPolicy


# ──────────────────────────────────────────────────────────────────────────────
    # convert_github_url_to_jsonl_autoname(
    # url: str,
    # out_dir: str | os.PathLike,
    # *,
    # ref: str | None = None,
    # policy: ChunkPolicy | None = None,
    # include_exts: set[str] | None = None,
    # exclude_exts: set[str] | None = None,
    # kql_from_markdown: bool = False,
    # also_prompt_text: bool = False,
    # ) -> tuple[str, str | None]
    # ──────────────────────────────────────────────────────────────────────────────
    # Purpose
    #   Download a GitHub repo (optionally at a specific ref), chunk its files, and
    #   write a JSONL corpus to out_dir with an auto-generated filename. Optionally
    #   also writes a “prompt text” sidecar in the same pass.

    # Required
    #   url
    #     GitHub URL (https/ssh). Examples:
    #       "https://github.com/owner/repo"
    #       "https://github.com/owner/repo.git"
    #       "https://github.com/owner/repo/tree/<ref>[/subpath]"
    #       "git@github.com:owner/repo.git"

    #   out_dir
    #     Directory where outputs will be written. File names are auto-derived from
    #     owner/repo, ref, and license (if detected).

    # Optional (keyword-only)
    #   ref
    #     Branch / tag / commit SHA to fetch. If None, uses the repo’s default branch.

    #   policy
    #     ChunkPolicy controlling how text is chunked.
    #       - mode: "auto" | "doc" | "code"          (default "auto")
    #       - target_tokens: int                      (default 800)
    #       - overlap_tokens: int (docs)              (default 100)
    #       - min_tokens: int (docs)                  (default 200)
    #     Example:
    #       ChunkPolicy(mode="auto", target_tokens=1200, overlap_tokens=150, min_tokens=250)

    #   include_exts
    #     If provided, ONLY files whose suffix is in this set are processed.
    #     Values must be dot-prefixed lowercase extensions, e.g. {".py", ".md"}.
    #     If None, the default is CODE_EXTS ∪ DOC_EXTS.
    #       CODE_EXTS (code-ish & configs):
    #         .py .pyw .py3 .ipynb .ps1 .psm1 .psd1 .bat .cmd .sh .bash .zsh
    #         .c .h .cpp .hpp .cc .hh .cxx .hxx .cs .java .kt .kts .scala .go .rs .swift
    #         .ts .tsx .js .jsx .mjs .cjs .rb .php .pl .pm .lua .r .jl .sql .sparql
    #         .json .jsonc .yaml .yml .toml .ini .cfg .xml .xslt
    #         .yara .yar .sigma .ndjson .log
    #       DOC_EXTS (docs):
    #         .md .mdx .rst .adoc .txt

    #   exclude_exts
    #     Files with these suffixes are skipped even if include_exts would include
    #     them. Same format as include_exts.

    #   kql_from_markdown
    #     If True, for Markdown files only, extract KQL code blocks and emit those
    #     blocks as records instead of the whole document.

    #   also_prompt_text
    #     If True, stream-write a prompt-friendly text file alongside the JSONL
    #     (filename auto-matches the JSONL). The function then returns paths for both.
    # ──────────────────────────────────────────────────────────────────────────────

URL = "https://github.com/EONRaider/Simple-Async-Port-Scanner"
OUT_DIR = Path("out")  # change if you want an absolute path

INCLUDE_EXTS: Optional[Set[str]] = {".py",".txt",".md",".toml"}

# "https://github.com/SystemsApproach/book"

def main() -> None:
    # Optional: set a token to avoid low-rate limits
    # os.environ.setdefault("GITHUB_TOKEN", "<your token>")

    log = configure_logging(level="INFO")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Converting repo → JSONL + prompt: %s", URL)

    


    jsonl_path, prompt_path = convert_github_url_to_jsonl_autoname(
        URL,
        OUT_DIR,
        include_exts=INCLUDE_EXTS,
        also_prompt_text=True,      
        kql_from_markdown=False,
        # policy = ChunkPolicy(mode="doc", target_tokens=1700, overlap_tokens=40, min_tokens=400)    
    )

    print("\n=== Done ===")
    print("JSONL :", jsonl_path)
    print("Prompt:", prompt_path)

    # Optional: run QC (lightweight heuristics; heavy deps are optional)
    try:
        from repocapsule.qc import score_jsonl_to_csv
        qc_csv = score_jsonl_to_csv(
            jsonl_path, lm_model_id="Qwen/Qwen2.5-1.5B",
            device="cuda",
            dtype="bfloat16",
            local_files_only=False,
            simhash_hamm_thresh=2,)  # set a model id if you want perplexity
        print("QC CSV:", qc_csv)
    except Exception as e:  # pragma: no cover
        print("QC skipped:", e)


if __name__ == "__main__":
    main()
