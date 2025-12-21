# sinks.py
# SPDX-License-Identifier: MIT
"""Sinks for writing repository records to various file formats."""
from __future__ import annotations

import gzip
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Self, TextIO

from ..core.interfaces import Record, RepoContext


class _BaseJSONLSink:
    """Shared JSONL sink logic with optional header support."""

    def __init__(
        self,
        out_path: str | os.PathLike[str],
        *,
        header_record: Mapping[str, Any] | None = None,
    ):
        """Configure a JSONL sink.

        Args:
            out_path (str | os.PathLike[str]): Destination file path.
            header_record (Mapping[str, Any] | None): Optional record
                written once at the start of the file.
        """
        self._path = Path(out_path)
        self._fp: TextIO | None = None
        self._tmp_path: Path | None = None
        self._header_record: Mapping[str, Any] | None = (
            dict(header_record) if header_record else None
        )
        self._header_written = False

    def open(self, context: RepoContext | None = None) -> None:
        """Create a temp file for writing and emit the header if set."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # If we've already written a header (e.g., during the main pipeline run)
        # and the file exists, append instead of clobbering the existing dataset.
        # This keeps finalize() calls from overwriting prior records.
        if self._header_written and self._path.exists():
            self._tmp_path = None
            self._fp = self._open_append_handle(self._path)
            return

        tmp_name = f"{self._path.name}.tmp"
        self._tmp_path = self._path.parent / tmp_name
        self._fp = self._open_handle(self._tmp_path)
        if self._header_record and not self._header_written:
            self.write(dict(self._header_record))
            self._header_written = True

    def write(self, record: Mapping[str, Any]) -> None:
        """Write a single JSON record as a compact line."""
        assert self._fp is not None
        self._fp.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    def close(self) -> None:
        """Close any open handle and move the temp file into place."""
        if not self._fp:
            return
        try:
            self._fp.close()
        finally:
            self._fp = None
        if self._tmp_path:
            os.replace(self._tmp_path, self._path)
            self._tmp_path = None

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Ensure resources are closed when used as a context manager."""
        self.close()

    def _open_handle(self, path: Path):
        """Return a write handle for a fresh file path."""
        raise NotImplementedError

    def _open_append_handle(self, path: Path):
        """Return an append handle for an existing file path."""
        raise NotImplementedError

    def set_header_record(self, record: Mapping[str, Any] | None) -> None:
        """Replace the header record to be written on the next open call."""
        self._header_record = dict(record) if record else None
        self._header_written = False

    def finalize(self, records: Iterable[Record]) -> None:
        """Write records directly, appending if needed.

        Args:
            records (Iterable[Record]): Records to write without buffering.
        """
        fp = self._fp
        temp_opened = False
        if fp is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fp = self._open_append_handle(self._path)
            temp_opened = True
        for rec in records:
            fp.write(json.dumps(dict(rec), ensure_ascii=False, separators=(",", ":")) + "\n")
        if temp_opened and fp is not None:
            fp.close()


class JSONLSink(_BaseJSONLSink):
    """Simple streaming JSONL sink (one record per line)."""

    def __init__(
        self,
        out_path: str | os.PathLike[str],
        *,
        header_record: Mapping[str, Any] | None = None,
    ):
        super().__init__(out_path, header_record=header_record)

    def _open_handle(self, path: Path):
        return open(path, "w", encoding="utf-8", newline="")

    def _open_append_handle(self, path: Path):
        return open(path, "a", encoding="utf-8", newline="")


class GzipJSONLSink(_BaseJSONLSink):
    """Streaming JSONL sink that gzip-compresses its output."""

    def __init__(
        self,
        out_path: str | os.PathLike[str],
        *,
        header_record: Mapping[str, Any] | None = None,
    ):
        super().__init__(out_path, header_record=header_record)

    def _open_handle(self, path: Path):
        return gzip.open(path, "wt", encoding="utf-8", newline="")

    def _open_append_handle(self, path: Path):
        return gzip.open(path, "at", encoding="utf-8", newline="")


class PromptTextSink:
    """Write human-readable prompt text for chunked documents."""

    def __init__(
        self,
        out_path: str | os.PathLike[str],
        *,
        heading_fmt: str = "### {path} [chunk {chunk}]",
    ):
        """Configure the destination and heading format.

        Args:
            out_path (str | os.PathLike[str]): Destination file path.
            heading_fmt (str): Template for headings using path and chunk.
        """
        self._path = Path(out_path)
        self._heading_fmt = heading_fmt
        self._fp: TextIO | None = None
        self._tmp_path: Path | None = None

    def open(self, context: RepoContext | None = None) -> None:
        """Create a temp file for writing prompt text output."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_name = f"{self._path.name}.tmp"
        self._tmp_path = self._path.parent / tmp_name
        self._fp = open(self._tmp_path, "w", encoding="utf-8", newline="")

    def write(self, record: Mapping[str, Any]) -> None:
        """Write a heading and its associated text block.

        Args:
            record (Dict[str, Any]): Record containing text and metadata.
        """
        assert self._fp is not None
        meta = record.get("meta", {})
        rel = meta.get("path", "unknown")
        cid = meta.get("chunk_id", "?")
        text = record.get("text") or ""
        heading = self._heading_fmt.format(path=rel, chunk=cid, meta=meta)
        self._fp.write(f"{heading}\n{text}\n\n")

    def finalize(self, records: Iterable[Mapping[str, Any]]) -> None:
        """Optional sink hook for finalizer records (no-op for prompt text)."""

    def close(self) -> None:
        """Close any open handle and move the temp file into place."""
        if not self._fp:
            return
        try:
            self._fp.close()
        finally:
            self._fp = None
        if self._tmp_path:
            os.replace(self._tmp_path, self._path)
            self._tmp_path = None

    def __enter__(self) -> PromptTextSink:
        """Open the sink for use as a context manager."""
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Ensure resources are closed when used as a context manager."""
        self.close()


class NoopSink:
    """Trivial sink with no-op lifecycle hooks for tests or mixins."""

    def open(self, context: RepoContext | None = None) -> None:  # noqa: D401
        """Perform no setup."""
        pass

    def write(self, record: Record) -> None:  # pragma: no cover - for completeness
        """Raise to indicate subclasses must implement writes."""
        raise NotImplementedError("NoopSink.write must be overridden")

    def close(self) -> None:  # noqa: D401
        """Perform no teardown."""
        pass


__all__ = ["JSONLSink", "GzipJSONLSink", "PromptTextSink", "NoopSink"]
