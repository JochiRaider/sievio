# factories.py
# SPDX-License-Identifier: MIT
"""
Facade for factory helpers.

This module re-exports factories split across:
- factories_sinks (sinks and output paths)
- factories_sources (sources and bytes handlers)
- factories_qc (QC and safety scorer construction)
- factories_context (repo context and HTTP client)

Callers should continue to import from `sievio.core.factories`.
"""

from __future__ import annotations

from .factories_context import make_http_client, make_repo_context_from_git
from .factories_qc import make_qc_scorer, make_safety_scorer
from .factories_sinks import (
    DefaultJsonlPromptSinkFactory,
    OutputPaths,
    ParquetDatasetSinkFactory,
    SinkFactoryResult,
    build_default_sinks,
    make_output_paths_for_github,
    make_output_paths_for_pdf,
)
from .factories_sources import (
    BytesHandler,
    CsvTextSourceFactory,
    GitHubZipSourceFactory,
    LocalDirSourceFactory,
    Sniff,
    SQLiteSourceFactory,
    UnsupportedBinary,
    WebPagePdfSourceFactory,
    WebPdfListSourceFactory,
    make_bytes_handlers,
    make_csv_text_source,
    make_github_zip_source,
    make_jsonl_text_source,
    make_local_dir_source,
    make_web_pdf_source,
)

__all__ = [
    "BytesHandler",
    "OutputPaths",
    "SinkFactoryResult",
    "Sniff",
    "UnsupportedBinary",
    "build_default_sinks",
    "make_bytes_handlers",
    "make_github_zip_source",
    "make_http_client",
    "make_local_dir_source",
    "make_web_pdf_source",
    "make_csv_text_source",
    "make_output_paths_for_github",
    "make_output_paths_for_pdf",
    "make_jsonl_text_source",
    "make_qc_scorer",
    "make_safety_scorer",
    "make_repo_context_from_git",
    "LocalDirSourceFactory",
    "GitHubZipSourceFactory",
    "WebPdfListSourceFactory",
    "WebPagePdfSourceFactory",
    "SQLiteSourceFactory",
    "CsvTextSourceFactory",
    "DefaultJsonlPromptSinkFactory",
    "ParquetDatasetSinkFactory",
]
