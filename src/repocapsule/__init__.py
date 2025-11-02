"""repocapsule

Repository → JSONL converter with robust decoding, Markdown/KQL extraction,
structure-aware chunking, and GitHub helpers. Stdlib-only.

Public API re-exports live here for convenience.
"""
from __future__ import annotations

# Version (optional _version.py can override)
try:  # pragma: no cover
    from ._version import __version__  # type: ignore
except Exception:  # pragma: no cover
    __version__ = "0.1.0"

# Logging helpers
from .log import get_logger, configure_logging, temp_level

# Filesystem traversal / .gitignore
from .fs import (
    DEFAULT_SKIP_DIRS,
    DEFAULT_SKIP_FILES,
    GitignoreRule,
    GitignoreMatcher,
    iter_repo_files,
    collect_repo_files,
)

# Decoding
from .decode import read_text_robust, decode_bytes_robust

# Chunking
from .chunk import (
    Block,
    ChunkPolicy,
    count_tokens,
    split_doc_blocks,
    chunk_text,
    register_doc_splitter,
)

# Markdown ↔ KQL
from .md_kql import (
    KQLBlock,
    extract_kql_blocks_from_markdown,
    is_probable_kql,
    guess_kql_tables,
    derive_category_from_rel,
)

# GitHub I/O
from .githubio import (
    RepoSpec,
    parse_github_url,
    github_api_get,
    get_repo_info,
    get_repo_license_spdx,
    download_zipball_to_temp,
    iter_zip_members,
    build_output_basename,
)

# Records
from .records import (
    CODE_EXTS,
    DOC_EXTS,
    EXT_LANG,
    guess_lang_from_path,
    is_code_file,
    sha256_text,
    build_record,
)

# Converters
from .convert import (
    make_records_for_file,
    write_repo_prompt_text,
    convert_repo_to_jsonl,
    convert_repo_to_jsonl_autoname,
    convert_github_url_to_jsonl,
    convert_github_url_to_both,
    convert_github_url_to_jsonl_autoname,
)


__all__ = [
    "__version__",
    # log
    "get_logger", "configure_logging", "temp_level",
    # fs
    "DEFAULT_SKIP_DIRS", "DEFAULT_SKIP_FILES", "GitignoreRule", "GitignoreMatcher", "iter_repo_files", "collect_repo_files",
    # decode
    "read_text_robust", "decode_bytes_robust",
    # chunk
    "Block", "ChunkPolicy", "count_tokens", "split_doc_blocks", "chunk_text","register_doc_splitter",
    # md_kql
    "KQLBlock", "extract_kql_blocks_from_markdown", "is_probable_kql", "guess_kql_tables", "derive_category_from_rel",
    # github
    "RepoSpec", "parse_github_url", "github_api_get", "get_repo_info", "get_repo_license_spdx", "download_zipball_to_temp", "iter_zip_members", "build_output_basename",
    # records
    "CODE_EXTS", "DOC_EXTS", "EXT_LANG", "guess_lang_from_path", "is_code_file", "sha256_text", "build_record",
    # convert
    "make_records_for_file", "write_repo_prompt_text", "convert_repo_to_jsonl", "convert_repo_to_jsonl_autoname", "convert_github_url_to_jsonl", "convert_github_url_to_both", "convert_github_url_to_jsonl_autoname",
]

try:
    from .qc import JSONLQualityScorer, score_jsonl_to_csv
    __all__ += ["JSONLQualityScorer", "score_jsonl_to_csv"]
except Exception:
    pass  # keep optional