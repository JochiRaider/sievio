# __init__.py
# SPDX-License-Identifier: MIT

"""Top-level package exports for :mod:`sievio`.

The names listed in :data:`PRIMARY_API` form the recommended, stable
surface for most callers: configure a run via :class:`SievioConfig`
or a TOML/JSON file, invoke a runner such as :func:`convert_local_dir`
/ :func:`convert_github` or :func:`convert`, and consume records via the
provided sinks.

High-level entry points such as :func:`convert`, :func:`convert_local_dir`,
and :func:`convert_github` are thin orchestration layers. They build
profiles, construct a :class:`PipelineEngine`, and run it; new
functionality should generally plug into the builder, registries, and
factories used by the engine rather than creating ad-hoc pipelines in
these helpers.

Quality control (inline or post-hoc) is configured via :class:`QCConfig`.
When ``mode="post"``, :func:`run_engine` (from
:mod:`sievio.cli.runner`) can rescore the primary JSONL after
extraction, optionally using process-based parallel scoring and emitting
QC CSV diagnostics.

Examples:
    Minimal local directory run::

        >>> from sievio import convert_local_dir
        >>> stats = convert_local_dir(
        ...     root_dir="path/to/repo",
        ...     out_jsonl="out/repo.jsonl",
        ... )

    Config-driven run::

        >>> from sievio import load_config_from_path, convert
        >>> cfg = load_config_from_path("example_config.toml")
        >>> stats = convert(cfg)

Advanced utilities are imported here for convenience, but anything not in
``PRIMARY_API`` is considered an expert surface and may change between
releases.

For HTTP access (GitHub zipballs, web PDFs), prefer configuring
``cfg.http``; the builder/factories will construct a :class:`SafeHttpClient`
and reuse it for remote sources. The module-level client in
``core.safe_http`` is intended primarily for simple one-shot scripts and
the CLI, not long-lived applications.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------
try:
    from importlib.metadata import version as _pkg_version  # Py3.8+

    __version__ = _pkg_version("sievio")
except Exception:  # PackageNotFoundError or any runtime env oddities
    __version__ = "0.0.0+unknown"


# ---------------------------------------------------------------------------
# Primary public API (stable)
# ---------------------------------------------------------------------------
from .core.config import SievioConfig, QCHeuristics, load_config_from_path
from .core.records import (
    build_record,
    RecordMeta,
    RunSummaryMeta,
    QCSummaryMeta,
    ensure_meta_dict,
    best_effort_record_path,
    is_summary_record,
)
from .cli.runner import (
    convert,
    convert_local_dir,
    convert_github,
    make_local_repo_config,
    make_github_repo_config,
)
from .core.builder import PipelineOverrides
from .sources.fs import LocalDirSource
from .sources.githubio import GitHubZipSource
from .sources.sources_webpdf import WebPdfListSource, WebPagePdfSource
from .sinks.sinks import JSONLSink, PromptTextSink, NoopSink
from .core.naming import (
    build_output_basename_github,
    build_output_basename_pdf,
)
from .core.licenses import detect_license_in_tree, detect_license_in_zip


# ---------------------------------------------------------------------------
# Advanced / expert API (subject to change)
# ---------------------------------------------------------------------------
from .core.builder import PipelinePlan, PipelineRuntime
from .core.log import get_logger, configure_logging, temp_level
from .sources.fs import (
    DEFAULT_SKIP_DIRS,
    DEFAULT_SKIP_FILES,
    GitignoreRule,
    GitignoreMatcher,
    iter_repo_files,
    collect_repo_files,
)
from .core.decode import read_text, decode_bytes
from .core.chunk import (
    Block,
    ChunkPolicy,
    count_tokens,
    split_doc_blocks,
    chunk_text,
    register_doc_splitter,
)
from .core.extras.md_kql import (
    KQLBlock,
    extract_kql_blocks_from_markdown,
    is_probable_kql,
    guess_kql_tables,
    derive_category_from_rel,
)
from .sources.githubio import (
    RepoSpec,
    parse_github_url,
    github_api_get,
    get_repo_info,
    get_repo_license_spdx,
    download_zipball_to_temp,
    iter_zip_members,
)
from .core.language_id import (
    LanguageConfig,
    CODE_EXTS,
    DOC_EXTS,
    EXT_LANG,
    guess_lang_from_path,
    is_code_file,
)
from .core.records import (
    sha256_text,
)
from .core.convert import (
    DefaultExtractor,
    iter_records_from_bytes,
    iter_records_from_bytes_with_plan,
    iter_records_from_file_item,
    list_records_for_file,
    list_records_from_bytes,
    make_records_for_file,
    make_records_from_bytes,
)
from .core.interfaces import FileExtractor, FileItem, RepoContext, Source, Sink, Extractor, RunLifecycleHook, RunContext
from .core.pipeline import run_pipeline, PipelineStats, PipelineEngine
from .cli.runner import run_engine
from .core.factories_sources import make_web_pdf_source


# Optional extras (not part of PRIMARY_API)
try:  # pragma: no cover - optional dependency
    from .core.extras.qc import JSONLQualityScorer, score_jsonl_to_csv
except Exception:  # pragma: no cover - keep import errors silent
    pass

try:  # pragma: no cover - optional dependency
    from .sources.evtxio import handle_evtx, sniff_evtx
except Exception:  # pragma: no cover - keep import errors silent
    pass

try:  # pragma: no cover - optional dependency
    from .sources.pdfio import extract_pdf_records
except Exception:  # pragma: no cover
    pass


PRIMARY_API = [
    "__version__",
    "SievioConfig",
    "load_config_from_path",
    "PipelineOverrides",
    "LanguageConfig",
    "convert",
    "best_effort_record_path",
    "run_engine",
    "convert_local_dir",
    "convert_github",
    "make_local_repo_config",
    "make_github_repo_config",
    "LocalDirSource",
    "GitHubZipSource",
    "WebPdfListSource",
    "WebPagePdfSource",
    "JSONLSink",
    "PromptTextSink",
    "NoopSink",
    "build_record",
    "RecordMeta",
    "RunSummaryMeta",
    "QCSummaryMeta",
    "ensure_meta_dict",
    "is_summary_record",
    "list_records_for_file",
    "list_records_from_bytes",
    "iter_records_from_bytes_with_plan",
    "make_records_for_file",
    "make_records_from_bytes",
    "build_output_basename_github",
    "build_output_basename_pdf",
    "detect_license_in_tree",
    "detect_license_in_zip",
]

__all__ = list(PRIMARY_API)
