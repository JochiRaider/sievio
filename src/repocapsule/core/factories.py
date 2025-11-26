# factories.py
# SPDX-License-Identifier: MIT
"""
Factory helpers for building clients, sources, sinks, and derived paths.

All helpers are deterministic, side-effect free, and keep construction logic
centralized so that orchestrators (runner/config) stay thin.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Callable,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from .interfaces import (
    RepoContext,
    Record,
    Sink,
    Source,
    SourceFactory,
    SinkFactory,
    SourceFactoryContext,
    SinkFactoryContext,
)
from ..sinks.sinks import JSONLSink, GzipJSONLSink, PromptTextSink
from ..sources.fs import PatternFileSource
from ..sources.sources_webpdf import WebPdfListSource, WebPagePdfSource
from .registries import BytesHandlerRegistry, QualityScorerRegistry, bytes_handler_registry, quality_scorer_registry

if TYPE_CHECKING:  # pragma: no cover - type-only imports
    from .chunk import ChunkPolicy
    from .config import (
        GitHubSourceConfig,
        HttpConfig,
        LocalDirSourceConfig,
        QCConfig,
        PdfSourceConfig,
        SinkConfig,
        RepocapsuleConfig,
        SourceSpec,
        SinkSpec,
    )
    from .qc import JSONLQualityScorer
    from .safe_http import SafeHttpClient
    from .interfaces import SourceFactory, SinkFactory

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

Sniff = Callable[[bytes, str], bool]
BytesHandler = Callable[
    [bytes, str, Optional[RepoContext], Optional["ChunkPolicy"]],
    Optional[Iterable[Record]],
]


@dataclass(frozen=True)
class SinkFactoryResult:
    """
    Container for the output of sink construction.

    Attributes:
        jsonl_path (str): Path to the primary JSONL output.
        sinks (Sequence[Sink]): Materialized sink instances.
        sink_config (SinkConfig): Effective sink configuration used to build
            sinks.
        metadata (Mapping[str, object]): Auxiliary details exposed to
            orchestrators.
    """

    jsonl_path: str
    sinks: Sequence[Sink]
    sink_config: "SinkConfig"
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class OutputPaths:
    """
    Bundle derived output locations for downstream consumers.

    Attributes:
        jsonl (Path): Path to the JSONL dataset.
        prompt (Path | None): Optional prompt text path.
        artifacts (Path | None): Optional directory for ancillary artifacts.
    """

    jsonl: Path
    prompt: Optional[Path] = None
    artifacts: Optional[Path] = None

    def as_tuple(self) -> Tuple[str, Optional[str]]:
        """
        Return the JSONL and prompt paths as strings.

        Returns:
            Tuple[str, Optional[str]]: JSONL path and optional prompt path.
        """

        return str(self.jsonl), (str(self.prompt) if self.prompt else None)


class UnsupportedBinary(Exception):
    """Raised when a recognized binary handler is unavailable in this build."""


def build_default_sinks(
    cfg: "SinkConfig",
    basename: Optional[str] = None,
    *,
    jsonl_path: Optional[str | Path] = None,
    prompt_path: Optional[str | Path] = None,
    context: Optional[RepoContext] = None,
) -> SinkFactoryResult:
    """
    Build the canonical JSONL and prompt sinks for a sink configuration.

    Exactly one of ``basename`` or ``jsonl_path`` must be provided. When a path
    is supplied explicitly, it takes precedence over ``cfg.output_dir``.

    Args:
        cfg (SinkConfig): Sink configuration providing defaults and output
            directory.
        basename (str | None): Basename for derived output files when
            ``jsonl_path`` is not supplied.
        jsonl_path (str | Path | None): Explicit JSONL output path. Overrides
            ``basename`` and ``cfg.output_dir``.
        prompt_path (str | Path | None): Explicit prompt output path. When
            omitted, uses a derived path if prompt output is enabled.
        context (RepoContext | None): Repository context to associate with
            sinks.

    Returns:
        SinkFactoryResult: Container for the built sinks and metadata.

    Raises:
        ValueError: If required path inputs are missing or incompatible.
    """
    if basename and jsonl_path:
        raise ValueError("Provide either basename or jsonl_path, not both")

    if jsonl_path is None:
        base = basename or (cfg.jsonl_basename or None)
        if not base:
            raise ValueError("A basename or jsonl_path is required")
        suffix = ".jsonl.gz" if cfg.compress_jsonl else ".jsonl"
        jsonl_path = cfg.output_dir / f"{base}{suffix}"
    jsonl_path = Path(jsonl_path)
    jsonl_str = str(jsonl_path)

    use_gzip = cfg.compress_jsonl or jsonl_str.endswith(".gz")
    sink_class = GzipJSONLSink if use_gzip else JSONLSink
    sinks: list[Sink] = [sink_class(jsonl_str)]

    prompt_target: Optional[str]
    if prompt_path is not None:
        prompt_target = str(Path(prompt_path))
    elif cfg.prompt.include_prompt_file:
        prompt_target = str(_default_prompt_path(jsonl_path))
    else:
        prompt_target = None

    if prompt_target:
        sinks.append(PromptTextSink(prompt_target, heading_fmt=cfg.prompt.heading_fmt))

    effective_context = context if context is not None else cfg.context
    sink_cfg = replace(
        cfg,
        sinks=tuple(sinks),
        context=effective_context,
        primary_jsonl_name=jsonl_str,
    )
    metadata = {"primary_jsonl": jsonl_str}
    if prompt_target:
        metadata["prompt_path"] = prompt_target
    return SinkFactoryResult(
        jsonl_path=jsonl_str,
        sinks=sink_cfg.sinks,
        sink_config=sink_cfg,
        metadata=metadata,
    )


def _default_prompt_path(jsonl_path: Path) -> Path:
    """
    Derive a prompt file path from a JSONL output path.

    Args:
        jsonl_path (Path): Primary JSONL path.

    Returns:
        Path: Prompt text path in the same directory.
    """

    name = jsonl_path.name
    if name.endswith(".jsonl.gz"):
        base = name[:-len(".jsonl.gz")]
    else:
        base = jsonl_path.stem
    prompt_name = f"{base}.prompt.txt"
    return jsonl_path.parent / prompt_name


def make_jsonl_text_source(
    paths: Sequence[str | Path],
    *,
    context: Optional[RepoContext] = None,
    text_key: str = "text",
):
    """
    Build a JSONLTextSource for a sequence of JSONL files.

    Args:
        paths (Sequence[str | Path]): JSONL file paths to read.
        context (RepoContext | None): Repository context to attach to records.
        text_key (str): Field containing text within each JSONL record.

    Returns:
        JSONLTextSource: Configured source for reading text rows.
    """

    from ..sources.jsonl_source import JSONLTextSource

    norm_paths = [Path(p) for p in paths]
    return JSONLTextSource(paths=tuple(norm_paths), context=context, text_key=text_key)


def make_csv_text_source(
    paths: Sequence[str | Path],
    *,
    context: Optional[RepoContext] = None,
    text_column: str = "text",
    delimiter: Optional[str] = None,
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
    config: "LocalDirSourceConfig",
    context: Optional[RepoContext] = None,
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


# ---------- Source/Sink factories for declarative specs ----------


@dataclass
class LocalDirSourceFactory(SourceFactory):
    """Build LocalDirSource instances from declarative specs."""

    id: str = "local_dir"

    def build(self, ctx: SourceFactoryContext, spec: "SourceSpec") -> Sequence[Source]:
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

        root = spec.options.get("root_dir")
        if root is None:
            raise ValueError("local_dir source spec requires root_dir")
        repo_ctx = ctx.repo_context
        # For now, keep using the typed LocalDirSourceConfig; per-kind defaults
        # come from the registered id.
        from .config import LocalDirSourceConfig  # type: ignore

        defaults = ctx.source_defaults.get(self.id, {})
        local_cfg = LocalDirSourceConfig(**dict(defaults)) if defaults else LocalDirSourceConfig()
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

    def build(self, ctx: SourceFactoryContext, spec: "SourceSpec") -> Sequence[Source]:
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

        url = spec.options.get("url")
        if url is None:
            raise ValueError("github_zip source spec requires url")
        repo_ctx = ctx.repo_context
        http_client = ctx.http_client or ctx.http_config.build_client()
        from .config import GitHubSourceConfig  # type: ignore

        defaults = ctx.source_defaults.get(self.id, {})
        gh_cfg = GitHubSourceConfig(**dict(defaults)) if defaults else GitHubSourceConfig()
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

    def build(self, ctx: SourceFactoryContext, spec: "SourceSpec") -> Sequence[Source]:
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

        urls = spec.options.get("urls")
        if not urls:
            raise ValueError("web_pdf_list source spec requires urls")
        from .config import PdfSourceConfig  # type: ignore

        defaults = ctx.source_defaults.get(self.id, {}) or ctx.source_defaults.get("pdf", {})
        pdf_cfg = PdfSourceConfig(**dict(defaults)) if defaults else PdfSourceConfig()
        src = WebPdfListSource(
            urls,
            timeout=pdf_cfg.timeout,
            max_pdf_bytes=pdf_cfg.max_pdf_bytes,
            require_pdf=pdf_cfg.require_pdf,
            add_prefix=spec.options.get("add_prefix"),
            retries=pdf_cfg.retries,
            config=pdf_cfg,
            client=pdf_cfg.client or ctx.http_client or ctx.http_config.build_client(),
        )
        return [src]


@dataclass
class WebPagePdfSourceFactory(SourceFactory):
    """Build WebPagePdfSource instances from declarative specs."""

    id: str = "web_page_pdf"

    def build(self, ctx: SourceFactoryContext, spec: "SourceSpec") -> Sequence[Source]:
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

        page_url = spec.options.get("page_url")
        if page_url is None:
            raise ValueError("web_page_pdf source spec requires page_url")
        from .config import PdfSourceConfig  # type: ignore

        defaults = ctx.source_defaults.get(self.id, {}) or ctx.source_defaults.get("pdf", {})
        pdf_cfg = PdfSourceConfig(**dict(defaults)) if defaults else PdfSourceConfig()
        src = WebPagePdfSource(
            page_url,
            max_links=pdf_cfg.max_links,
            require_pdf=pdf_cfg.require_pdf,
            include_ambiguous=pdf_cfg.include_ambiguous,
            add_prefix=spec.options.get("add_prefix"),
            config=pdf_cfg,
            client=pdf_cfg.client or ctx.http_client or ctx.http_config.build_client(),
        )
        return [src]


@dataclass
class SQLiteSourceFactory(SourceFactory):
    """Build SQLiteSource instances from declarative specs."""

    id: str = "sqlite"

    def build(self, ctx: SourceFactoryContext, spec: "SourceSpec") -> Sequence[Source]:
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

        options = spec.options or {}
        from .config import SQLiteSourceConfig  # type: ignore

        defaults = ctx.source_defaults.get(self.id, {}) or ctx.source_defaults.get("sqlite", {})
        sqlite_cfg = SQLiteSourceConfig(**dict(defaults)) if defaults else SQLiteSourceConfig()

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

        batch_size = options.get("batch_size", sqlite_cfg.batch_size)
        download_timeout = options.get("download_timeout", ctx.http_config.timeout)
        download_max_bytes = options.get("download_max_bytes", sqlite_cfg.download_max_bytes)
        retries = options.get("retries", sqlite_cfg.retries)

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
        )
        return [src]


@dataclass
class CsvTextSourceFactory(SourceFactory):
    """Build CSVTextSource instances from declarative specs."""

    id: str = "csv_text"

    def build(self, ctx: SourceFactoryContext, spec: "SourceSpec") -> Sequence[Source]:
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

        options = spec.options or {}
        from .config import CsvSourceConfig  # type: ignore

        defaults = ctx.source_defaults.get(self.id, {}) or ctx.source_defaults.get("csv", {})
        csv_cfg = CsvSourceConfig(**dict(defaults)) if defaults else CsvSourceConfig()
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


@dataclass
class DefaultJsonlPromptSinkFactory(SinkFactory):
    """Build the canonical JSONL + prompt sink pair from declarative specs."""

    id: str = "default_jsonl_prompt"

    def build(self, ctx: SinkFactoryContext, spec: "SinkSpec") -> SinkFactoryResult:
        """
        Construct JSONL and prompt sinks from a sink specification.

        Args:
            ctx (SinkFactoryContext): Factory context with sink configuration
                and repository metadata.
            spec (SinkSpec): Sink specification containing options.

        Returns:
            SinkFactoryResult: Built sinks and effective configuration.

        Raises:
            ValueError: If ``jsonl_path`` is missing from the specification.
        """

        jsonl_path = spec.options.get("jsonl_path")
        if jsonl_path is None:
            raise ValueError("default_jsonl_prompt sink spec requires jsonl_path")
        prompt_path = spec.options.get("prompt_path")
        sink_cfg = ctx.sink_config
        repo_ctx = sink_cfg.context or ctx.repo_context
        return build_default_sinks(
            sink_cfg,
            jsonl_path=jsonl_path,
            prompt_path=prompt_path,
            context=repo_ctx,
        )


@dataclass
class ParquetDatasetSinkFactory(SinkFactory):
    """Build ParquetDatasetSink instances from declarative specs."""

    id: str = "parquet_dataset"

    def build(self, ctx: SinkFactoryContext, spec: "SinkSpec") -> SinkFactoryResult:
        """
        Construct a ParquetDatasetSink from a sink specification.

        Args:
            ctx (SinkFactoryContext): Factory context with sink configuration
                and repository metadata.
            spec (SinkSpec): Sink specification containing options.

        Returns:
            SinkFactoryResult: Built sink and effective configuration.

        Raises:
            ValueError: If required options are missing or invalid.
            RuntimeError: If the Parquet extra is unavailable or construction
                fails.
        """

        sink_cfg = ctx.sink_config
        options = spec.options or {}
        path = options.get("path")
        if path is None:
            raise ValueError("parquet_dataset sink spec requires path")
        text_field = options.get("text_field", "text")
        meta_field = options.get("meta_field", "meta")
        partition_opt = options.get("partition_by")
        partition_by = [str(p) for p in partition_opt] if partition_opt else []
        row_group_size = options.get("row_group_size")
        if row_group_size is not None:
            try:
                row_group_size = int(row_group_size)
            except Exception as exc:
                raise ValueError("row_group_size must be an int") from exc
            if row_group_size <= 0:
                row_group_size = None
        compression = options.get("compression", "snappy") or "snappy"
        overwrite = bool(options.get("overwrite", True))
        try:
            from ..sinks.parquet import ParquetDatasetSink  # noqa: F401
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Parquet sink requires the 'parquet' extra (install repocapsule[parquet])."
            ) from exc
        except Exception as exc:  # pragma: no cover - defensive guard
            raise RuntimeError(f"Parquet sink could not be constructed: {exc}") from exc

        sink = ParquetDatasetSink(
            path=path,
            text_field=text_field,
            meta_field=meta_field,
            partition_by=partition_by,
            row_group_size=row_group_size,
            compression=compression,
            overwrite=overwrite,
        )
        jsonl_path = sink_cfg.primary_jsonl_name or ""
        metadata = {"parquet_path": str(path)}
        return SinkFactoryResult(
            jsonl_path=jsonl_path,
            sinks=[sink],
            sink_config=sink_cfg,
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# HTTP / QC factories
# ---------------------------------------------------------------------------

def make_http_client(http_cfg: "HttpConfig") -> "SafeHttpClient":
    """
    Build (or reuse) the SafeHttpClient described by ``http_cfg``.

    Args:
        http_cfg (HttpConfig): HTTP configuration describing the client.

    Returns:
        SafeHttpClient: Client configured per ``http_cfg``.

    Raises:
        ValueError: If ``http_cfg`` is missing.
    """
    if http_cfg is None:
        raise ValueError("http_cfg is required")
    return http_cfg.build_client()


def make_qc_scorer(
    qc_cfg: Optional["QCConfig"],
    *,
    new_instance: bool = False,
    scorer_registry: Optional[QualityScorerRegistry] = None,
) -> Optional["JSONLQualityScorer"]:
    """
    Instantiate a JSONLQualityScorer when QC is enabled and extras are loaded.

    Args:
        qc_cfg (QCConfig | None): Quality-control configuration.
        new_instance (bool): Force creation of a fresh scorer even if one is
            cached on the config.
        scorer_registry (QualityScorerRegistry | None): Registry override for
            resolving scorer factories.

    Returns:
        JSONLQualityScorer | None: Configured scorer, or ``None`` when QC is
        disabled or no scorer is registered.
    """
    if qc_cfg is None or not getattr(qc_cfg, "enabled", False):
        return None
    existing = getattr(qc_cfg, "scorer", None)
    if existing is not None and not new_instance:
        return existing
    # Trigger registration of built-in scorer factory (and any extras).
    try:
        from .extras import qc as _qc_module  # noqa: F401
    except Exception:
        pass
    reg = scorer_registry or quality_scorer_registry
    options = dict(getattr(qc_cfg, "scorer_options", {}) or {})
    factory_id = getattr(qc_cfg, "scorer_id", None)
    scorer = reg.build(options, factory_id=factory_id)
    if scorer is None:
        return None
    if not new_instance:
        qc_cfg.scorer = scorer
    return scorer


# ---------------------------------------------------------------------------
# Bytes-handler factory (PDF/EVTX)
# ---------------------------------------------------------------------------

def make_bytes_handlers(registry: Optional[BytesHandlerRegistry] = None) -> Sequence[Tuple[Sniff, BytesHandler]]:
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
    ctx: Optional[RepoContext],
    policy: Optional["ChunkPolicy"],
) -> Optional[Iterable[Record]]:
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
    ctx: Optional[RepoContext],
    policy: Optional["ChunkPolicy"],
) -> Optional[Iterable[Record]]:
    """Raise an error indicating EVTX support is unavailable."""

    raise UnsupportedBinary("evtx support is not installed")


# ---------------------------------------------------------------------------
# Repo/source/context helpers
# ---------------------------------------------------------------------------

def make_repo_context_from_git(repo_root: Path | str) -> Optional[RepoContext]:
    """
    Infer a RepoContext from ``.git/config`` when the remote points at GitHub.

    Args:
        repo_root (Path | str): Repository root containing a ``.git`` folder.

    Returns:
        RepoContext | None: Populated context or ``None`` when metadata is
        unavailable.
    """
    cfg_path = Path(repo_root) / ".git" / "config"
    try:
        text = cfg_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    current_remote: Optional[str] = None
    origin_url: Optional[str] = None
    fallback_url: Optional[str] = None
    remote_header = re.compile(r'\s*\[remote\s+"([^"]+)"\]')
    url_line = re.compile(r"^\s*url\s*=\s*([^\r\n]+)$")

    for line in text.splitlines():
        header = remote_header.match(line)
        if header:
            current_remote = header.group(1)
            continue
        if current_remote is None:
            continue
        m = url_line.match(line)
        if not m:
            continue
        url_value = m.group(1).strip()
        if current_remote == "origin":
            origin_url = url_value
            break
        if fallback_url is None:
            fallback_url = url_value

    remote = origin_url or fallback_url
    if not remote:
        return None
    from ..sources.githubio import parse_github_url  # local import to avoid cycles

    spec = parse_github_url(remote)
    if not spec:
        return None
    return RepoContext(
        repo_full_name=f"{spec.owner}/{spec.repo}",
        repo_url=f"https://github.com/{spec.owner}/{spec.repo}",
        license_id=None,
        extra={"source": "local"},
    )


def make_local_dir_source(
    root: Path | str,
    *,
    config: "LocalDirSourceConfig",
    context: Optional[RepoContext] = None,
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
    config: "GitHubSourceConfig",
    context: Optional[RepoContext],
    download_timeout: Optional[float],
    http_client: Optional["SafeHttpClient"] = None,
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
    config: "PdfSourceConfig",
    http_client: Optional["SafeHttpClient"] = None,
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


# ---------------------------------------------------------------------------
# Output path helpers
# ---------------------------------------------------------------------------

def make_output_paths_for_github(
    *,
    owner: str,
    repo: str,
    ref: Optional[str],
    license_spdx: Optional[str],
    out_dir: Path | str,
    include_prompt: bool = True,
    timestamp: Optional[str] = None,
    include_commit: Optional[str] = None,
) -> OutputPaths:
    """
    Build output paths for a GitHub dataset.

    Args:
        owner (str): GitHub owner or organization.
        repo (str): Repository name.
        ref (str | None): Commit-ish or ref used to build the dataset.
        license_spdx (str | None): SPDX license identifier for metadata.
        out_dir (Path | str): Output directory root.
        include_prompt (bool): Whether to include a prompt output path.
        timestamp (str | None): Timestamp suffix appended to the basename when
            provided.
        include_commit (str | None): Commit hash appended to the basename when
            provided.

    Returns:
        OutputPaths: Derived JSONL and prompt paths.

    Raises:
        ValueError: If ``owner`` or ``repo`` are missing.
    """
    if not owner or not repo:
        raise ValueError("owner and repo are required for GitHub output paths")
    from .naming import build_output_basename_github

    base = build_output_basename_github(
        owner=owner,
        repo=repo,
        ref=ref or "main",
        license_spdx=license_spdx,
        include_commit=include_commit,
    )
    base = _append_timestamp(base, timestamp)
    out_dir = _normalize_out_dir(out_dir)
    jsonl = out_dir / f"{base}.jsonl"
    prompt = (out_dir / f"{base}.prompt.txt") if include_prompt else None
    return OutputPaths(jsonl=jsonl, prompt=prompt)


def make_output_paths_for_pdf(
    *,
    url: str,
    title: Optional[str],
    license_spdx: Optional[str],
    out_dir: Path | str,
    include_prompt: bool = True,
    timestamp: Optional[str] = None,
) -> OutputPaths:
    """
    Build output paths for a PDF corpus using URL, title, and license metadata.

    Args:
        url (str): Source URL of the PDF corpus.
        title (str | None): Optional title incorporated into the basename.
        license_spdx (str | None): SPDX license identifier for metadata.
        out_dir (Path | str): Output directory root.
        include_prompt (bool): Whether to include a prompt output path.
        timestamp (str | None): Timestamp suffix appended to the basename when
            provided.

    Returns:
        OutputPaths: Derived JSONL and prompt paths.

    Raises:
        ValueError: If ``url`` is missing.
    """
    if not url:
        raise ValueError("url is required for PDF output paths")
    from .naming import build_output_basename_pdf

    base = build_output_basename_pdf(url=url, title=title, license_spdx=license_spdx)
    base = _append_timestamp(base, timestamp)
    out_dir = _normalize_out_dir(out_dir)
    jsonl = out_dir / f"{base}.jsonl"
    prompt = (out_dir / f"{base}.prompt.txt") if include_prompt else None
    return OutputPaths(jsonl=jsonl, prompt=prompt)


def _normalize_out_dir(out_dir: Path | str) -> Path:
    return Path(out_dir).expanduser()


def _append_timestamp(base: str, timestamp: Optional[str]) -> str:
    if not timestamp:
        return base
    cleaned = re.sub(r"[^\w\-]+", "_", timestamp.strip())
    cleaned = re.sub(r"_{2,}", "_", cleaned).strip("_")
    if not cleaned:
        return base
    return f"{base}__{cleaned}"
