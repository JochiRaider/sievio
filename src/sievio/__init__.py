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
from .cli.runner import (
    convert,
    convert_github,
    convert_local_dir,
    make_github_repo_config,
    make_local_repo_config,
    run_engine,
)

# ---------------------------------------------------------------------------
# Advanced / expert API (subject to change)
# ---------------------------------------------------------------------------
from .core.builder import PipelineOverrides, PipelinePlan, PipelineRuntime
from .core.chunk import (
    Block,
    ChunkPolicy,
    chunk_text,
    count_tokens,
    register_doc_splitter,
    split_doc_blocks,
)
from .core.config import QCHeuristics, SievioConfig, load_config_from_path
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
from .core.decode import decode_bytes, read_text
from .core.extras.md_kql import (
    KQLBlock,
    derive_category_from_rel,
    extract_kql_blocks_from_markdown,
    guess_kql_tables,
    is_probable_kql,
)
from .core.factories_sources import make_web_pdf_source
from .core.interfaces import (
    Extractor,
    FileExtractor,
    FileItem,
    RepoContext,
    RunContext,
    RunLifecycleHook,
    Sink,
    Source,
)
from .core.language_id import (
    CODE_EXTS,
    DOC_EXTS,
    EXT_LANG,
    LanguageConfig,
    guess_lang_from_path,
    is_code_file,
)
from .core.licenses import detect_license_in_tree, detect_license_in_zip
from .core.log import configure_logging, get_logger, temp_level
from .core.naming import (
    build_output_basename_github,
    build_output_basename_pdf,
)
from .core.pipeline import PipelineEngine, PipelineStats
from .core.runner import run_pipeline
from .core.records import (
    QCSummaryMeta,
    RecordMeta,
    RunSummaryMeta,
    best_effort_record_path,
    build_record,
    ensure_meta_dict,
    is_summary_record,
    sha256_text,
)
from .sinks.sinks import JSONLSink, NoopSink, PromptTextSink
from .sources.fs import (
    DEFAULT_SKIP_DIRS,
    DEFAULT_SKIP_FILES,
    GitignoreMatcher,
    GitignoreRule,
    LocalDirSource,
    collect_repo_files,
    iter_repo_files,
)
from .sources.githubio import (
    GitHubZipSource,
    RepoSpec,
    download_zipball_to_temp,
    get_repo_info,
    get_repo_license_spdx,
    github_api_get,
    iter_zip_members,
    parse_github_url,
)
from .sources.sources_webpdf import WebPagePdfSource, WebPdfListSource

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
