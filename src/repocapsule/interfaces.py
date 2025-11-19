# interfaces.py
# SPDX-License-Identifier: MIT

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
    from .config import RepocapsuleConfig, FileProcessingConfig


# -----------------------------------------------------------------------------
# Shared data types
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class FileItem:
    """
    A single file emitted by a Source.

    Attributes
    ----------
    path:
        Repository-relative path using forward slashes, e.g. "src/main.py".
    data:
        Raw file bytes as obtained from the source (zip entry, filesystem, etc.).
        Decoding to text is performed later by the pipeline/decoder. Data may be
        a truncated prefix when only part of a large file was read.
    size:
        Original size in bytes on disk / source (may differ from len(data)).
    origin_path:
        Absolute/local path or synthetic identifier for reopening when possible.
    stream_hint:
        Optional tag describing how to reopen (e.g., "file", "zip-member").
    streamable:
        True when a streaming path exists (e.g., local filesystem files).
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
    """
    repo_full_name: Optional[str] = None     # e.g., "owner/name"
    repo_url: Optional[str] = None           # https://github.com/owner/name
    license_id: Optional[str] = None         # SPDX-ish id if known (e.g., "MIT")
    commit_sha: Optional[str] = None         # archive commit or ref resolved
    # Free-form bag for future metadata (timestamps, labels, etc.)
    extra: Optional[Mapping[str, Any]] = None

    def as_meta_seed(self) -> Dict[str, Any]:
        """Return a dict of metadata fields that should seed every record."""
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
# Extension-point protocols
# -----------------------------------------------------------------------------

@runtime_checkable
class Source(Protocol):
    """
    Produces repository files (as bytes) for downstream decoding and processing.

    Implementations SHOULD be streaming-friendly and avoid buffering whole
    archives in memory where possible.
    """
    def iter_files(self) -> Iterable[FileItem]:
        """Yield FileItem objects. Must not raise on benign unreadable entries."""


@runtime_checkable
class Extractor(Protocol):
    """
    Optional content extractor that can emit *additional* records derived
    from the raw text of a file (e.g., KQL blocks extracted from Markdown).

    Return an iterable of Record to add them; return None or an empty iterable
    if the extractor has nothing to contribute for this file.
    """
    # Optional: a short name for logging/registry display
    name: Optional[str]  # type: ignore[assignment]

    def extract(
        self,
        *,
        text: str,
        path: str,
        context: Optional[RepoContext] = None,
    ) -> Optional[Iterable[Record]]:
        """
        Parameters
        ----------
        text:
            UTF-8 (or normalized) decoded content of the file.
        path:
            Repository-relative path for context.
        context:
            Repository-level metadata if available.

        Returns
        -------
        Optional[Iterable[Record]]
            Records to append to the output stream, or None / empty iterable.
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
        ...


@runtime_checkable
class Sink(Protocol):
    """
    A destination for records (e.g., JSONL writer, prompt-text writer, Parquet).

    Sinks are opened once, receive many records, then closed. Implementations
    should be robust to being opened/closed even if no records are written.
    """

    def open(self, context: Optional[RepoContext] = None) -> None:
        """Prepare resources (files, DB connections, headers)."""

    def write(self, record: Record) -> None:
        """Consume a single record. Implementations should be fast and minimal."""

    def close(self) -> None:
        """Flush and free resources. Must not raise on repeated calls."""


@runtime_checkable
class QualityScorer(Protocol):
    """
    Minimal contract for quality scorers used in inline or post-processing modes.
    """

    def score_record(self, record: Mapping[str, Any]) -> Dict[str, Any]:
        """Return per-record QC metrics."""

    def score_jsonl_path(self, path: str) -> Iterable[Dict[str, Any]]:
        """Iterate QC rows for every record within a JSONL file."""


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
]
