# interfaces.py
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Iterable,
    Mapping,
    Optional,
    Protocol,
    runtime_checkable,
)


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
        Decoding to text is performed later by the pipeline/decoder.
    size:
        Size in bytes (redundant with len(data) but convenient for logging).
    """
    path: str
    data: bytes
    size: int | None = None


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


__all__ = [
    "FileItem",
    "RepoContext",
    "Record",
    "Source",
    "Extractor",
    "Sink",
]
