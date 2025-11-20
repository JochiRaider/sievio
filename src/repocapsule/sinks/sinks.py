# sinks.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, Mapping
import os, json, gzip

from ..core.interfaces import RepoContext, Record


class _BaseJSONLSink:
    """Shared logic for JSONL sinks."""

    def __init__(self, out_path: str | os.PathLike[str], *, header_record: Optional[Mapping[str, Any]] = None):
        self._path = Path(out_path)
        self._fp = None
        self._tmp_path: Optional[Path] = None
        self._header_record: Optional[Mapping[str, Any]] = dict(header_record) if header_record else None
        self._header_written = False

    def open(self, context: Optional[RepoContext] = None) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_name = f"{self._path.name}.tmp"
        self._tmp_path = self._path.parent / tmp_name
        self._fp = self._open_handle(self._tmp_path)
        if self._header_record and not self._header_written:
            self.write(dict(self._header_record))
            self._header_written = True

    def write(self, record: Dict[str, Any]) -> None:
        assert self._fp is not None
        self._fp.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    def close(self) -> None:
        if not self._fp:
            return
        try:
            self._fp.close()
        finally:
            self._fp = None
        if self._tmp_path:
            os.replace(self._tmp_path, self._path)
            self._tmp_path = None

    def __enter__(self) -> "JSONLSink":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _open_handle(self, path: Path):
        raise NotImplementedError

    def _open_append_handle(self, path: Path):
        raise NotImplementedError

    def set_header_record(self, record: Optional[Mapping[str, Any]]) -> None:
        self._header_record = dict(record) if record else None
        self._header_written = False

    def finalize(self, records: Iterable[Record]) -> None:
        """Default finalize writes records as normal JSONL entries."""
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

    def __init__(self, out_path: str | os.PathLike[str], *, header_record: Optional[Mapping[str, Any]] = None):
        super().__init__(out_path, header_record=header_record)

    def _open_handle(self, path: Path):
        return open(path, "w", encoding="utf-8", newline="")

    def _open_append_handle(self, path: Path):
        return open(path, "a", encoding="utf-8", newline="")


class GzipJSONLSink(_BaseJSONLSink):
    """Streaming JSONL sink that gzip-compresses its output."""

    def __init__(self, out_path: str | os.PathLike[str], *, header_record: Optional[Mapping[str, Any]] = None):
        super().__init__(out_path, header_record=header_record)

    def _open_handle(self, path: Path):
        return gzip.open(path, "wt", encoding="utf-8", newline="")

    def _open_append_handle(self, path: Path):
        return gzip.open(path, "at", encoding="utf-8", newline="")

class PromptTextSink:
    """Writes human-readable prompt text (format as you prefer)."""

    def __init__(self, out_path: str | os.PathLike[str], *, heading_fmt: str = "### {path} [chunk {chunk}]"):
        self._path = Path(out_path)
        self._heading_fmt = heading_fmt
        self._fp = None
        self._tmp_path: Optional[Path] = None

    def open(self, context: Optional[RepoContext] = None) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_name = f"{self._path.name}.tmp"
        self._tmp_path = self._path.parent / tmp_name
        self._fp = open(self._tmp_path, "w", encoding="utf-8", newline="")

    def write(self, record: Dict[str, Any]) -> None:
        assert self._fp is not None
        meta = record.get("meta", {})
        rel = meta.get("path", "unknown")
        cid = meta.get("chunk_id", "?")
        text = record.get("text") or ""
        heading = self._heading_fmt.format(path=rel, chunk=cid, meta=meta)
        self._fp.write(f"{heading}\n{text}\n\n")

    def close(self) -> None:
        if not self._fp:
            return
        try:
            self._fp.close()
        finally:
            self._fp = None
        if self._tmp_path:
            os.replace(self._tmp_path, self._path)
            self._tmp_path = None

    def __enter__(self) -> "PromptTextSink":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class NoopSink:
    """
    A trivial Sink you can subclass; provides no-op lifecycle methods.
    Useful in tests or as a mixin when only ``write`` needs custom behavior.
    """

    def open(self, context: Optional[RepoContext] = None) -> None:  # noqa: D401
        pass

    def write(self, record: Record) -> None:  # pragma: no cover - for completeness
        raise NotImplementedError("NoopSink.write must be overridden")

    def close(self) -> None:  # noqa: D401
        pass


__all__ = ["JSONLSink", "GzipJSONLSink", "PromptTextSink", "NoopSink"]
