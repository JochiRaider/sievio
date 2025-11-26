# interfaces.py
# SPDX-License-Identifier: MIT
"""Interfaces and protocols shared across sources, sinks, and pipeline hooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    IO,
    Any,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Protocol,
    TYPE_CHECKING,
    runtime_checkable,
)

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from pathlib import Path
    from .config import (
        RepocapsuleConfig,
        FileProcessingConfig,
        SourceSpec,
        SinkSpec,
        HttpConfig,
    )
    from .factories import SinkFactoryResult
    from .safe_http import SafeHttpClient


# -----------------------------------------------------------------------------
# Shared data types
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class FileItem:
    """
    A single file emitted by a Source.

    Attributes:
        path (str): Repository-relative path using forward slashes, e.g.,
            ``src/main.py``.
        data (bytes): Raw file bytes as obtained from the source (zip entry,
            filesystem, etc.). Data may be a truncated prefix when only part of
            a large file was read.
        size (int | None): Original size in bytes on disk or source (may differ
            from ``len(data)``).
        origin_path (str | None): Absolute/local path or synthetic identifier
            for reopening when possible.
        stream_hint (str | None): Optional tag describing how to reopen
            (e.g., ``file``, ``zip-member``).
        streamable (bool): True when a streaming path exists (e.g., local
            filesystem files).
    """
    path: str
    data: bytes
    size: int | None = None
    origin_path: str | None = None
    stream_hint: str | None = None
    streamable: bool = False


@dataclass(frozen=True, slots=True)
class RepoContext:
    """
    Optional repository-level context that sources and sinks may care about.

    All fields are optional by design to keep the contract stable.

    Attributes:
        repo_full_name (str | None): Repository name in ``owner/name`` form.
        repo_url (str | None): Canonical repository URL.
        license_id (str | None): SPDX-ish license identifier, if known.
        commit_sha (str | None): Commit hash or ref resolved for the source.
        extra (Mapping[str, Any] | None): Free-form metadata for downstream
            consumers.
    """

    repo_full_name: Optional[str] = None     # e.g., "owner/name"
    repo_url: Optional[str] = None           # https://github.com/owner/name
    license_id: Optional[str] = None         # SPDX-ish id if known (e.g., "MIT")
    commit_sha: Optional[str] = None         # archive commit or ref resolved
    extra: Optional[Mapping[str, Any]] = None

    def as_meta_seed(self) -> Dict[str, Any]:
        """
        Return a metadata seed dictionary for initializing records.

        Returns:
            Dict[str, Any]: Mapping of metadata keys to values.
        """
        meta: Dict[str, Any] = {}
        if self.repo_url:
            meta.setdefault("source", self.repo_url)
            meta.setdefault("repo_url", self.repo_url)
        if self.repo_full_name:
            meta.setdefault("repo", self.repo_full_name)
            meta.setdefault("repo_full_name", self.repo_full_name)
        if self.license_id:
            meta.setdefault("license", self.license_id)
        if self.commit_sha:
            meta.setdefault("commit_sha", self.commit_sha)
        if self.extra:
            for key, value in self.extra.items():
                if value is None:
                    continue
                meta.setdefault(str(key), value)
        return meta


# A JSONL record shape is intentionally loose: dict-like with string keys.
Record = Mapping[str, Any]

# -----------------------------------------------------------------------------
# Record hook protocols
# -----------------------------------------------------------------------------


@runtime_checkable
class RecordFilter(Protocol):
    """Decide whether a record should be kept."""

    def accept(self, record: Record) -> bool:
        """
        Evaluate a record for inclusion.

        Args:
            record (Record): Record to inspect.

        Returns:
            bool: True to keep the record, False to drop it.
        """
        ...


@runtime_checkable
class RecordObserver(Protocol):
    """Receive callbacks for every record produced."""

    def on_record(self, record: Record) -> None:
        """
        Observe a record emitted by the pipeline.

        Args:
            record (Record): Record produced by a source or extractor.
        """
        ...


# -----------------------------------------------------------------------------
# Extension-point protocols
# -----------------------------------------------------------------------------

@runtime_checkable
class Source(Protocol):
    """
    Produces repository files (as bytes) for downstream decoding and processing.

    Implementations should be streaming-friendly and avoid buffering whole
    archives in memory where possible.
    """
    def iter_files(self) -> Iterable[FileItem]:
        """
        Yield FileItem objects from the source.

        Yields:
            FileItem: File payload and metadata to process.
        """


@runtime_checkable
class ClosableSource(Protocol):
    """
    Optional extension for sources that need cleanup but are not context managers.
    """

    def close(self) -> None:
        """Release any held resources."""


@runtime_checkable
class Extractor(Protocol):
    """
    Optional content extractor that emits additional records derived from
    decoded file text (e.g., KQL blocks extracted from Markdown).

    Attributes:
        name (str | None): Short identifier for logging or registry display.
    """

    name: Optional[str]  # type: ignore[assignment]

    def extract(
        self,
        *,
        text: str,
        path: str,
        context: Optional[RepoContext] = None,
    ) -> Optional[Iterable[Record]]:
        """
        Produce derived records for a decoded file.

        Args:
            text (str): UTF-8 (or normalized) decoded content of the file.
            path (str): Repository-relative path for context.
            context (RepoContext | None): Repository metadata when available.

        Returns:
            Iterable[Record] | None: Records to append to the output stream, or
            ``None`` / empty iterable when nothing is emitted.
        """
        ...


@runtime_checkable
class FileExtractor(Protocol):
    """
    File-level extractor that consumes FileItem objects and yields records.
    """

    def extract(
        self,
        item: FileItem,
        *,
        config: "RepocapsuleConfig | FileProcessingConfig",
        context: Optional[RepoContext] = None,
    ) -> Iterable[Record]:
        """
        Process a file item and emit records.

        Args:
            item (FileItem): File to extract data from.
            config (RepocapsuleConfig | FileProcessingConfig): Execution
                configuration controlling extraction behavior.
            context (RepoContext | None): Repository metadata when available.

        Returns:
            Iterable[Record]: Records derived from the file.
        """
        ...


@runtime_checkable
class StreamingExtractor(Protocol):
    """
    Optional extension for extractors that can consume byte streams.

    Streaming extractors can avoid materializing the full file payload when
    running inside a thread-friendly pipeline that can reopen the underlying
    source (e.g., local files). The default pipeline only attempts this path
    when executor_kind == "thread" and the FileItem is marked streamable with
    a valid origin_path.
    """

    name: Optional[str]  # type: ignore[assignment]

    def extract_stream(
        self,
        *,
        stream: IO[bytes],
        path: str,
        context: Optional[RepoContext] = None,
    ) -> Optional[Iterable[Record]]:
        """
        Process a readable byte stream and emit records.

        Args:
            stream (IO[bytes]): Open binary stream for the file payload.
            path (str): Repository-relative path for context.
            context (RepoContext | None): Repository metadata when available.

        Returns:
            Iterable[Record] | None: Records derived from the stream, or
            ``None`` when nothing is emitted.
        """
        ...


@runtime_checkable
class Sink(Protocol):
    """
    A destination for records (e.g., JSONL writer, prompt-text writer, Parquet).

    Sinks are opened once, receive many records, then closed. Implementations
    should be robust to being opened/closed even if no records are written.
    """

    def open(self, context: Optional[RepoContext] = None) -> None:
        """
        Prepare resources prior to writes.

        Args:
            context (RepoContext | None): Repository metadata for initialization.
        """

    def write(self, record: Record) -> None:
        """
        Consume a single record.

        Args:
            record (Record): Record to persist.
        """

    def close(self) -> None:
        """Flush and free resources. Must not raise on repeated calls."""

    # Optional extension point: sinks can consume finalizer records such as run summaries.
    def finalize(self, records: Iterable[Record]) -> None:  # pragma: no cover - optional
        """
        Consume a sequence of finalizer records such as run summaries.

        Args:
            records (Iterable[Record]): Finalizer records to process.
        """


@runtime_checkable
class QualityScorer(Protocol):
    """
    Minimal contract for quality scorers used in inline or post-processing modes.

    Implementers may optionally provide:
    - ``clone_for_parallel()`` → return a fresh scorer for worker processes in parallel post-QC.
    - ``reset_state()`` → clear incremental state before rescoring JSONL for CSV export.
    """

    def score_record(self, record: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Compute QC metrics for a single record.

        Args:
            record (Mapping[str, Any]): Record to evaluate.

        Returns:
            Dict[str, Any]: Per-record QC metrics.
        """

    def score_jsonl_path(self, path: str) -> Iterable[Dict[str, Any]]:
        """
        Iterate QC metrics for every record within a JSONL file.

        Args:
            path (str): Path to the JSONL file to score.

        Returns:
            Iterable[Dict[str, Any]]: QC rows corresponding to each record.
        """

    def clone_for_parallel(self) -> "QualityScorer":  # pragma: no cover - optional
        """Return a fresh scorer for use in parallel workers."""


@runtime_checkable
class SourceFactory(Protocol):
    """Factory protocol for building Source objects from specifications."""

    id: str

    def build(self, ctx: "SourceFactoryContext", spec: "SourceSpec") -> Iterable["Source"]:
        """
        Create one or more sources from a specification.

        Args:
            ctx (SourceFactoryContext): Narrowed factory context.
            spec (SourceSpec): Declarative source specification.

        Returns:
            Iterable[Source]: Materialized sources.
        """
        ...


@runtime_checkable
class SinkFactory(Protocol):
    """Factory protocol for building sinks from specifications."""

    id: str

    def build(self, ctx: "SinkFactoryContext", spec: "SinkSpec") -> "SinkFactoryResult":
        """
        Create one or more sinks from a specification.

        Args:
            ctx (SinkFactoryContext): Narrowed factory context.
            spec (SinkSpec): Declarative sink specification.

        Returns:
            SinkFactoryResult: Built sinks and related metadata.
        """
        ...


@runtime_checkable
class RecordMiddleware(Protocol):
    """
    Per-record middleware that can transform or drop a record.
    Returning None drops the record.
    """

    def process(self, record: Record) -> Optional[Record]:
        """
        Inspect or modify a record before downstream handling.

        Args:
            record (Record): Input record.

        Returns:
            Record | None: Transformed record, or ``None`` to drop it.
        """
        ...


@runtime_checkable
class FileMiddleware(Protocol):
    """
    Per-file middleware that can transform or drop record iterators for a FileItem.
    Returning None drops all records for that item.
    """

    def process(self, item: Any, records: Iterable[Record]) -> Optional[Iterable[Record]]:
        """
        Inspect or modify records associated with a file item.

        Args:
            item (Any): Original file object provided by the source.
            records (Iterable[Record]): Records produced for the file.

        Returns:
            Iterable[Record] | None: Updated records, or ``None`` to drop all.
        """
        ...


@runtime_checkable
class ConcurrencyProfile(Protocol):
    """
    Optional hint for executor selection.

    Attributes:
        preferred_executor (str | None): Preferred executor kind (``thread`` or
            ``process``).
        cpu_intensive (bool): True when the work is CPU-bound.
    """

    preferred_executor: Optional[str]  # "thread" or "process"
    cpu_intensive: bool


@dataclass(frozen=True, slots=True)
class SourceFactoryContext:
    """
    Narrowed view of cross-cutting source settings passed to SourceFactory.build.

    Keeps factories decoupled from the full RepocapsuleConfig so they can be
    reused in other environments and by plugins.

    Attributes:
        repo_context (RepoContext | None): Repository metadata shared with
            sources.
        http_client (SafeHttpClient | None): HTTP client override, if any.
        http_config (HttpConfig): HTTP configuration used to construct clients.
        source_defaults (Mapping[str, Mapping[str, Any]]): Per-source defaults
            keyed by source id.
    """

    repo_context: Optional[RepoContext]
    http_client: Optional["SafeHttpClient"]
    http_config: "HttpConfig"
    source_defaults: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class SinkFactoryContext:
    """
    Narrowed view of sink-related settings passed to SinkFactory.build.

    Attributes:
        repo_context (RepoContext | None): Repository metadata shared with
            sinks.
        sink_config (SinkConfig): Effective sink configuration.
    """

    repo_context: Optional[RepoContext]
    sink_config: "SinkConfig"


__all__ = [
    "FileItem",
    "RepoContext",
    "Record",
    "Source",
    "Extractor",
    "FileExtractor",
    "StreamingExtractor",
    "Sink",
    "QualityScorer",
    "SourceFactory",
    "SinkFactory",
    "SourceFactoryContext",
    "SinkFactoryContext",
    "RecordFilter",
    "RecordObserver",
    "RecordMiddleware",
    "FileMiddleware",
    "ConcurrencyProfile",
]
