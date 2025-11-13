# sinks.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any
import os, json

from .interfaces import RepoContext, Record


class JSONLSink:
    """Simple streaming JSONL sink (one record per line)."""

    def __init__(self, out_path: str | os.PathLike[str]):
        self._path = Path(out_path)
        self._fp = None
        self._tmp_path: Optional[Path] = None

    def open(self, context: Optional[RepoContext] = None) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_name = f"{self._path.name}.tmp"
        self._tmp_path = self._path.parent / tmp_name
        self._fp = open(self._tmp_path, "w", encoding="utf-8", newline="")

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


__all__ = ["JSONLSink", "PromptTextSink", "NoopSink"]
