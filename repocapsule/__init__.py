# __init__.py
# SPDX-License-Identifier: MIT

from __future__ import annotations

# Version ---------------------------------------------------------------
# Prefer reading the installed distribution version; fall back for
# editable/dev contexts where distribution metadata may be absent.
try:
    from importlib.metadata import version as _pkg_version  # Py3.8+
    __version__ = _pkg_version("repocapsule")
except Exception:  # PackageNotFoundError or any runtime env oddities
    __version__ = "0.0.0+unknown"

# Logging helpers
from .log import get_logger, configure_logging, temp_level

# Filesystem traversal / .gitignore
from .fs import (
    DEFAULT_SKIP_DIRS,
    DEFAULT_SKIP_FILES,
    LocalDirSource,
    GitignoreRule,
    GitignoreMatcher,
    iter_repo_files,
    collect_repo_files,
)

# Decoding
from .decode import read_text, decode_bytes

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
    GitHubZipSource,
    parse_github_url,
    github_api_get,
    get_repo_info,
    get_repo_license_spdx,
    download_zipball_to_temp,
    iter_zip_members,
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
from .convert import make_records_for_file, make_records_from_bytes
from .export import annotate_exact_token_counts

# Web PDF sources
from .sources_webpdf import WebPdfListSource, WebPagePdfSource

# Output file naming 
from .naming import build_output_basename, build_output_basename_github, build_output_basename_pdf
# Runners
from .runner import (
    convert,
    convert_local_dir,
    convert_github,
)

from .sinks import JSONLSink, PromptTextSink, NoopSink
from .evtxio import handle_evtx,sniff_evtx
from .licenses import detect_license_in_tree, detect_license_in_zip
from .pipeline import run_pipeline, PipelineStats
from .interfaces import FileItem, RepoContext, Source, Sink, Extractor
from .config import RepocapsuleConfig

__all__ = [
    "__version__",
    # Config
    "RepocapsuleConfig",
    # log
    "get_logger", "configure_logging", "temp_level",
    # fs
    "DEFAULT_SKIP_DIRS", "DEFAULT_SKIP_FILES", "GitignoreRule", 
    "GitignoreMatcher", "iter_repo_files", "collect_repo_files",
    # decode
    "read_text", "decode_bytes",
    # chunk
    "Block", "ChunkPolicy", "count_tokens", "split_doc_blocks", "chunk_text","register_doc_splitter",
    # md_kql
    "KQLBlock", "extract_kql_blocks_from_markdown", "is_probable_kql", 
    "guess_kql_tables", "derive_category_from_rel",
    # github
    "RepoSpec", "parse_github_url", "github_api_get", "get_repo_info", 
    "get_repo_license_spdx", "download_zipball_to_temp", "iter_zip_members", 
    # records
    "CODE_EXTS", "DOC_EXTS", "EXT_LANG", "guess_lang_from_path", "is_code_file",
    "sha256_text", "build_record",
    # convert
    "make_records_for_file", "make_records_from_bytes", 
    "annotate_exact_token_counts",
    # web sources
    "WebPdfListSource", "WebPagePdfSource",  
    # runner / sinks
    "convert", "convert_local_dir", "convert_github",
    "JSONLSink", "PromptTextSink", "NoopSink",
    "LocalDirSource", "GitHubZipSource",
    # pipeline
    "run_pipeline", "PipelineStats",
    # interfaces
    "FileItem", "RepoContext", "Source", "Sink", "Extractor",
    # naming
    "build_output_basename", "build_output_basename_github", "build_output_basename_pdf",
    # Windows Event Logs
    "handle_evtx","sniff_evtx",
    # License helpers
    "detect_license_in_zip","detect_license_in_tree",
]
# Optional: QC Check (depends on torch, transformers, tiktoken, pyyaml); expose if available
try:
    from .qc import JSONLQualityScorer, score_jsonl_to_csv
    __all__ += ["JSONLQualityScorer", "score_jsonl_to_csv"]
except Exception:
    pass  

# Optional: PDF extractor (depends on pypdf); expose if available
try:
    from .pdfio import extract_pdf_records
    __all__ += ["extract_pdf_records"]
except Exception:
    pass
