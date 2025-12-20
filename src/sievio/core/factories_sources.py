# factories_sources.py
# SPDX-License-Identifier: MIT
"""
Factory helpers for ingest sources and bytes-handler wiring.

Split out from core.factories to centralize source construction and binary
handler registration.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Optional,
)

from ..sources.fs import PatternFileSource
from ..sources.sources_webpdf import WebPagePdfSource, WebPdfListSource
from .interfaces import (
    Record,
    RepoContext,
    Source,
    SourceFactory,
    SourceFactoryContext,
)
from .registries import (
    BytesHandlerRegistry,
    bytes_handler_registry,
)

if TYPE_CHECKING:  # pragma: no cover - type-only imports
    from .chunk import ChunkPolicy
    from .config import (
        GitHubSourceConfig,
        LocalDirSourceConfig,
        PdfSourceConfig,
        SourceSpec,
    )
    from .interfaces import SourceFactory
    from .safe_http import SafeHttpClient

Sniff = Callable[[bytes, str], bool]
BytesHandler = Callable[
    [bytes, str, RepoContext | None, Optional["ChunkPolicy"]],
    Iterable[Record] | None,
]

__all__ = [
    "Sniff",
    "BytesHandler",
    "UnsupportedBinary",
    "make_bytes_handlers",
    "make_local_dir_source",
    "make_github_zip_source",
    "make_web_pdf_source",
    "make_csv_text_source",
    "make_jsonl_text_source",
    "LocalDirSourceFactory",
    "GitHubZipSourceFactory",
    "WebPdfListSourceFactory",
    "WebPagePdfSourceFactory",
    "SQLiteSourceFactory",
    "CsvTextSourceFactory",
]


class UnsupportedBinary(Exception):
    """Raised when a recognized binary handler is unavailable in this build."""


def make_jsonl_text_source(
    paths: Sequence[str | Path],
    *,
    context: RepoContext | None = None,
    text_key: str = "text",
    check_schema: bool = True,
):
    """
    Build a JSONLTextSource for a sequence of JSONL files.

    Args:
        paths (Sequence[str | Path]): JSONL file paths to read.
        context (RepoContext | None): Repository context to attach to records.
        text_key (str): Field containing text within each JSONL record.
        check_schema (bool): Whether to run schema checks on Sievio records.

    Returns:
        JSONLTextSource: Configured source for reading text rows.
    """

    from ..sources.jsonl_source import JSONLTextSource

    norm_paths = [Path(p) for p in paths]
    return JSONLTextSource(
        paths=tuple(norm_paths),
        context=context,
        text_key=text_key,
        check_schema=check_schema,
    )


def make_csv_text_source(
    paths: Sequence[str | Path],
    *,
    context: RepoContext | None = None,
    text_column: str = "text",
    delimiter: str | None = None,
    encoding: str = "utf-8",
    has_header: bool = True,
    text_column_index: int = 0,
):
    """
    Build a CSVTextSource for CSV or TSV inputs.

    Args:
        paths (Sequence[str | Path]): CSV paths to ingest.
        context (RepoContext | None): Repository context to attach to records.
        text_column (str): Column name containing text values.
        delimiter (str | None): Field delimiter override.
        encoding (str): File encoding for the CSV files.
        has_header (bool): Whether the CSV includes a header row.
        text_column_index (int): Fallback column index for text when headers
            are absent.

    Returns:
        CSVTextSource: Configured CSV text source.
    """

    from ..sources.csv_source import CSVTextSource

    norm_paths = [Path(p) for p in paths]
    return CSVTextSource(
        paths=tuple(norm_paths),
        context=context,
        text_column=text_column,
        delimiter=delimiter,
        encoding=encoding,
        has_header=has_header,
        text_column_index=text_column_index,
    )


def make_pattern_file_source(
    root: str | Path,
    patterns: Sequence[str],
    *,
    config: LocalDirSourceConfig,
    context: RepoContext | None = None,
) -> PatternFileSource:
    """
    Build a PatternFileSource rooted at a directory.

    Args:
        root (str | Path): Root directory to scan.
        patterns (Sequence[str]): Glob-style patterns to include.
        config (LocalDirSourceConfig): Source configuration controlling chunking
            and filtering.
        context (RepoContext | None): Repository context to attach to records.

    Returns:
        PatternFileSource: Configured file source.
    """

    return PatternFileSource(root, patterns, config=config, context=context)


@dataclass
class LocalDirSourceFactory(SourceFactory):
    """Build LocalDirSource instances from declarative specs."""

    id: str = "local_dir"

    def build(self, ctx: SourceFactoryContext, spec: SourceSpec) -> Sequence[Source]:
        """
        Construct a LocalDirSource from a source specification.

        Args:
            ctx (SourceFactoryContext): Factory context including defaults and
                repository metadata.
            spec (SourceSpec): Source specification containing options.

        Returns:
            Sequence[Source]: A single LocalDirSource wrapped in a sequence.

        Raises:
            ValueError: If ``root_dir`` is missing from the specification.
        """

        options = spec.options or {}
        root = options.get("root_dir")
        if root is None:
            raise ValueError("local_dir source spec requires root_dir")
        repo_ctx = ctx.repo_context
        from .config import (  # type: ignore
            LocalDirSourceConfig,
            build_config_from_defaults_and_options,
            validate_options_for_dataclass,
        )

        defaults = ctx.source_defaults.get(self.id, {}) or {}
        validate_options_for_dataclass(
            LocalDirSourceConfig,
            options=options,
            ignore_keys=("root_dir",),
            context="sources.specs.local_dir",
        )
        local_cfg = build_config_from_defaults_and_options(
            LocalDirSourceConfig,
            defaults=defaults,
            options=options,
            ignore_keys=("root_dir",),
        )
        src = make_local_dir_source(
            root=root,
            config=local_cfg,
            context=repo_ctx,
        )
        return [src]


@dataclass
class GitHubZipSourceFactory(SourceFactory):
    """Build GitHubZipSource instances from declarative specs."""

    id: str = "github_zip"

    def build(self, ctx: SourceFactoryContext, spec: SourceSpec) -> Sequence[Source]:
        """
        Construct a GitHubZipSource from a source specification.

        Args:
            ctx (SourceFactoryContext): Factory context with HTTP configuration
                and source defaults.
            spec (SourceSpec): Source specification containing options.

        Returns:
            Sequence[Source]: A single GitHubZipSource wrapped in a sequence.

        Raises:
            ValueError: If ``url`` is missing from the specification.
        """

        options = spec.options or {}
        url = options.get("url")
        if url is None:
            raise ValueError("github_zip source spec requires url")
        repo_ctx = ctx.repo_context
        http_client = ctx.http_client or ctx.http_config.build_client()
        from .config import (  # type: ignore
            GitHubSourceConfig,
            build_config_from_defaults_and_options,
            validate_options_for_dataclass,
        )

        defaults = ctx.source_defaults.get(self.id, {}) or {}
        validate_options_for_dataclass(
            GitHubSourceConfig,
            options=options,
            ignore_keys=("url",),
            context="sources.specs.github_zip",
        )
        gh_cfg = build_config_from_defaults_and_options(
            GitHubSourceConfig,
            defaults=defaults,
            options=options,
            ignore_keys=("url",),
        )
        src = make_github_zip_source(
            url,
            config=gh_cfg,
            context=repo_ctx,
            download_timeout=ctx.http_config.timeout,
            http_client=http_client,
        )
        return [src]


@dataclass
class WebPdfListSourceFactory(SourceFactory):
    """Build WebPdfListSource instances from declarative specs."""

    id: str = "web_pdf_list"

    def build(self, ctx: SourceFactoryContext, spec: SourceSpec) -> Sequence[Source]:
        """
        Construct a WebPdfListSource from a source specification.

        Args:
            ctx (SourceFactoryContext): Factory context carrying defaults and
                HTTP configuration.
            spec (SourceSpec): Source specification containing options.

        Returns:
            Sequence[Source]: A single WebPdfListSource wrapped in a sequence.

        Raises:
            ValueError: If ``urls`` is missing from the specification.
        """

        options = spec.options or {}
        urls = options.get("urls")
        if not urls:
            raise ValueError("web_pdf_list source spec requires urls")
        from .config import (  # type: ignore
            PdfSourceConfig,
            build_config_from_defaults_and_options,
            validate_options_for_dataclass,
        )

        defaults = ctx.source_defaults.get(self.id, {}) or ctx.source_defaults.get("pdf", {}) or {}
        validate_options_for_dataclass(
            PdfSourceConfig,
            options=options,
            ignore_keys=("urls", "add_prefix"),
            context="sources.specs.web_pdf_list",
        )
        pdf_cfg = build_config_from_defaults_and_options(
            PdfSourceConfig,
            defaults=defaults,
            options=options,
            ignore_keys=("urls", "add_prefix"),
        )
        src = WebPdfListSource(
            urls,
            timeout=pdf_cfg.timeout,
            max_pdf_bytes=pdf_cfg.max_pdf_bytes,
            require_pdf=pdf_cfg.require_pdf,
            add_prefix=options.get("add_prefix"),
            retries=pdf_cfg.retries,
            config=pdf_cfg,
            client=pdf_cfg.client or ctx.http_client or ctx.http_config.build_client(),
        )
        return [src]


@dataclass
class WebPagePdfSourceFactory(SourceFactory):
    """Build WebPagePdfSource instances from declarative specs."""

    id: str = "web_page_pdf"

    def build(self, ctx: SourceFactoryContext, spec: SourceSpec) -> Sequence[Source]:
        """
        Construct a WebPagePdfSource from a source specification.

        Args:
            ctx (SourceFactoryContext): Factory context carrying defaults and
                HTTP configuration.
            spec (SourceSpec): Source specification containing options.

        Returns:
            Sequence[Source]: A single WebPagePdfSource wrapped in a sequence.

        Raises:
            ValueError: If ``page_url`` is missing from the specification.
        """

        options = spec.options or {}
        page_url = options.get("page_url")
        if page_url is None:
            raise ValueError("web_page_pdf source spec requires page_url")
        from .config import (  # type: ignore
            PdfSourceConfig,
            build_config_from_defaults_and_options,
            validate_options_for_dataclass,
        )

        defaults = ctx.source_defaults.get(self.id, {}) or ctx.source_defaults.get("pdf", {}) or {}
        validate_options_for_dataclass(
            PdfSourceConfig,
            options=options,
            ignore_keys=("page_url", "add_prefix"),
            context="sources.specs.web_page_pdf",
        )
        pdf_cfg = build_config_from_defaults_and_options(
            PdfSourceConfig,
            defaults=defaults,
            options=options,
            ignore_keys=("page_url", "add_prefix"),
        )
        src = WebPagePdfSource(
            page_url,
            max_links=pdf_cfg.max_links,
            timeout=pdf_cfg.timeout,
            max_pdf_bytes=pdf_cfg.max_pdf_bytes,
            require_pdf=pdf_cfg.require_pdf,
            include_ambiguous=pdf_cfg.include_ambiguous,
            add_prefix=options.get("add_prefix"),
            retries=pdf_cfg.retries,
            config=pdf_cfg,
            client=pdf_cfg.client or ctx.http_client or ctx.http_config.build_client(),
        )
        return [src]


@dataclass
class SQLiteSourceFactory(SourceFactory):
    """Build SQLiteSource instances from declarative specs."""

    id: str = "sqlite"

    def build(self, ctx: SourceFactoryContext, spec: SourceSpec) -> Sequence[Source]:
        """
        Construct a SQLiteSource from a source specification.

        Args:
            ctx (SourceFactoryContext): Factory context with HTTP configuration
                and defaults.
            spec (SourceSpec): Source specification containing options.

        Returns:
            Sequence[Source]: A single SQLiteSource wrapped in a sequence.

        Raises:
            ValueError: If ``db_path`` is missing from the specification.
        """

        from ..sources.sqlite_source import SQLiteSource
        from .config import (  # type: ignore
            SQLiteSourceConfig,
            build_config_from_defaults_and_options,
            validate_options_for_dataclass,
        )

        options = spec.options or {}

        defaults = (
            ctx.source_defaults.get(self.id, {})
            or ctx.source_defaults.get("sqlite", {})
            or {}
        )
        validate_options_for_dataclass(
            SQLiteSourceConfig,
            options=options,
            ignore_keys=(
                "db_path",
                "db_url",
                "table",
                "sql",
                "text_columns",
                "id_column",
                "where",
                "download_timeout",
                "checksum",
                "sha256",
            ),
            context="sources.specs.sqlite",
        )
        sqlite_cfg = build_config_from_defaults_and_options(
            SQLiteSourceConfig,
            defaults=defaults,
            options=options,
            ignore_keys=(
                "db_path",
                "db_url",
                "table",
                "sql",
                "text_columns",
                "id_column",
                "where",
                "download_timeout",
                "checksum",
                "sha256",
            ),
        )

        db_path_str = options.get("db_path")
        if not db_path_str:
            raise ValueError("sqlite source spec requires db_path")
        db_url = options.get("db_url")

        table = options.get("table")
        sql = options.get("sql")
        text_columns = options.get("text_columns", sqlite_cfg.default_text_columns)
        if isinstance(text_columns, str):
            text_columns = (text_columns,)
        id_column = options.get("id_column")
        where = options.get("where")
        checksum = options.get("checksum") or options.get("sha256")

        batch_size = sqlite_cfg.batch_size
        download_timeout = options.get("download_timeout", ctx.http_config.timeout)
        download_max_bytes = sqlite_cfg.download_max_bytes
        retries = sqlite_cfg.retries

        db_path = Path(db_path_str)
        repo_ctx = ctx.repo_context
        client = ctx.http_client or ctx.http_config.build_client()

        src = SQLiteSource(
            db_path=db_path,
            context=repo_ctx,
            table=table,
            sql=sql,
            text_columns=text_columns,
            id_column=id_column,
            where=where,
            batch_size=batch_size,
            db_url=db_url,
            download_timeout=download_timeout,
            download_max_bytes=download_max_bytes,
            retries=retries,
            client=client,
            checksum=checksum,
        )
        return [src]


@dataclass
class CsvTextSourceFactory(SourceFactory):
    """Build CSVTextSource instances from declarative specs."""

    id: str = "csv_text"

    def build(self, ctx: SourceFactoryContext, spec: SourceSpec) -> Sequence[Source]:
        """
        Construct a CSVTextSource from a source specification.

        Args:
            ctx (SourceFactoryContext): Factory context carrying defaults and
                HTTP configuration.
            spec (SourceSpec): Source specification containing options.

        Returns:
            Sequence[Source]: A single CSVTextSource wrapped in a sequence.

        Raises:
            ValueError: If no CSV paths are provided.
        """

        from ..sources.csv_source import CSVTextSource
        from .config import (  # type: ignore
            CsvSourceConfig,
            build_config_from_defaults_and_options,
            validate_options_for_dataclass,
        )

        options = spec.options or {}

        defaults = (
            ctx.source_defaults.get(self.id, {})
            or ctx.source_defaults.get("csv", {})
            or {}
        )
        validate_options_for_dataclass(
            CsvSourceConfig,
            options=options,
            ignore_keys=(
                "paths",
                "path",
                "text_column",
                "delimiter",
                "encoding",
                "has_header",
                "text_column_index",
            ),
            context="sources.specs.csv_text",
        )
        csv_cfg = build_config_from_defaults_and_options(
            CsvSourceConfig,
            defaults=defaults,
            options=options,
            ignore_keys=(
                "paths",
                "path",
                "text_column",
                "delimiter",
                "encoding",
                "has_header",
                "text_column_index",
            ),
        )
        raw_paths = options.get("paths") or options.get("path")
        if not raw_paths:
            raise ValueError("csv_text source spec requires 'paths' (list) or 'path'")

        if isinstance(raw_paths, (str, Path)):
            paths = [raw_paths]
        else:
            paths = list(raw_paths)

        text_column = options.get("text_column", csv_cfg.default_text_column)
        delimiter = options.get("delimiter", csv_cfg.default_delimiter)
        encoding = options.get("encoding", csv_cfg.encoding)
        has_header = options.get("has_header", True)
        text_column_index = options.get("text_column_index", 0)

        norm_paths = [Path(p) for p in paths]
        repo_ctx = ctx.repo_context

        src = CSVTextSource(
            paths=tuple(norm_paths),
            context=repo_ctx,
            text_column=text_column,
            delimiter=delimiter,
            encoding=encoding,
            has_header=has_header,
            text_column_index=text_column_index,
        )
        return [src]


def make_local_dir_source(
    root: Path | str,
    *,
    config: LocalDirSourceConfig,
    context: RepoContext | None = None,
):
    """
    Build a LocalDirSource for a filesystem root.

    Args:
        root (Path | str): Root directory to scan.
        config (LocalDirSourceConfig): Source configuration controlling
            filtering and chunking.
        context (RepoContext | None): Repository context to attach to records.

    Returns:
        LocalDirSource: Configured local directory source.

    Raises:
        ValueError: If ``config`` is missing.
    """
    if config is None:
        raise ValueError("LocalDirSourceConfig is required")
    from ..sources.fs import LocalDirSource  # local import to break cycles

    return LocalDirSource(root, config=config, context=context)


def make_github_zip_source(
    url: str,
    *,
    config: GitHubSourceConfig,
    context: RepoContext | None,
    download_timeout: float | None,
    http_client: SafeHttpClient | None = None,
):
    """
    Build a GitHubZipSource for a GitHub archive URL.

    Args:
        url (str): GitHub zip or tarball URL.
        config (GitHubSourceConfig): Source configuration controlling download
            and filtering behavior.
        context (RepoContext | None): Repository context to attach to records.
        download_timeout (float | None): Request timeout for downloading the
            archive.
        http_client (SafeHttpClient | None): Optional HTTP client override.

    Returns:
        GitHubZipSource: Configured GitHub zip source.

    Raises:
        ValueError: If ``url`` is missing.
    """
    if not url:
        raise ValueError("url is required for GitHubZipSource")
    from ..sources.githubio import GitHubZipSource  # local import to avoid cycles

    return GitHubZipSource(
        url,
        config=config,
        context=context,
        download_timeout=download_timeout,
        http_client=http_client,
    )


def make_web_pdf_source(
    urls: Sequence[str | Path],
    *,
    config: PdfSourceConfig,
    http_client: SafeHttpClient | None = None,
):
    """
    Build a WebPdfListSource from a sequence of URLs.

    Args:
        urls (Sequence[str | Path]): PDF URLs to download.
        config (PdfSourceConfig): Source configuration controlling download
            behavior and validation.
        http_client (SafeHttpClient | None): Optional HTTP client override;
            defaults to the global client if omitted.

    Returns:
        WebPdfListSource: Configured PDF list source.
    """
    from ..sources.sources_webpdf import WebPdfListSource  # local import to avoid cycles

    norm_urls = [str(u) for u in urls]
    return WebPdfListSource(
        norm_urls,
        config=config,
        client=http_client,
    )


def make_bytes_handlers(
    registry: BytesHandlerRegistry | None = None,
) -> Sequence[tuple[Sniff, BytesHandler]]:
    """
    Return the default sniff/handler pairs for binary formats.

    Args:
        registry (BytesHandlerRegistry | None): Registry override for handler
            resolution. Falls back to the global registry.

    Returns:
        Sequence[Tuple[Sniff, BytesHandler]]: Registered sniff/handler pairs for
        PDF, EVTX, and Parquet files.
    """
    reg = registry or bytes_handler_registry
    if not reg.handlers():
        try:
            from ..sources import pdfio  # noqa: F401
        except Exception:
            pass
        try:
            from ..sources import evtxio  # noqa: F401
        except Exception:
            pass
        try:
            from ..sources import parquetio  # noqa: F401
        except Exception:
            pass
    handlers = list(reg.handlers())
    if handlers:
        return handlers
    reg.register(_fallback_sniff_pdf, _fallback_handle_pdf)
    reg.register(_fallback_sniff_evtx, _fallback_handle_evtx)
    return reg.handlers()


def _fallback_sniff_pdf(data: bytes, rel: str) -> bool:
    """Detect PDFs by extension or file header bytes."""

    return rel.lower().endswith(".pdf") or data.startswith(b"%PDF-")


def _fallback_handle_pdf(
    data: bytes,
    rel: str,
    ctx: RepoContext | None,
    policy: ChunkPolicy | None,
) -> Iterable[Record] | None:
    """Raise an error indicating PDF support is unavailable."""

    raise UnsupportedBinary("pdf support is not installed")


def _fallback_sniff_evtx(data: bytes, rel: str) -> bool:
    """Detect EVTX blobs by extension or signature markers."""

    name = rel.lower()
    if name.endswith(".evtx"):
        return True
    if data.startswith(b"ElfFile"):
        return True
    if b"ElfChnk" in data[:1_048_576]:
        return True
    return False


def _fallback_handle_evtx(
    data: bytes,
    rel: str,
    ctx: RepoContext | None,
    policy: ChunkPolicy | None,
) -> Iterable[Record] | None:
    """Raise an error indicating EVTX support is unavailable."""

    raise UnsupportedBinary("evtx support is not installed")
