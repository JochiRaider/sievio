# csv_source.py
# SPDX-License-Identifier: MIT

"""CSV/TSV source that emits text content as repository file items."""

from __future__ import annotations

import csv
import gzip
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.interfaces import FileItem, RepoContext, Source
from ..core.log import get_logger

__all__ = ["CSVTextSource"]

log = get_logger(__name__)


@dataclass
class CSVTextSource(Source):
    """Stream text content from CSV/TSV files into file items.

    Attributes:
        paths (Sequence[Path]): Files to read from disk.
        context (RepoContext | None): Context for repository-aware
            operations.
        text_column (str): Name of the column containing text data when
            headers are present.
        delimiter (str | None): Custom delimiter, otherwise inferred
            from file suffix.
        encoding (str): Encoding used to read files.
        has_header (bool): Whether files include a header row.
        text_column_index (int): Column index used when no header is
            present.
    """
    paths: Sequence[Path]
    context: RepoContext | None = None
    text_column: str = "text"
    delimiter: str | None = None
    encoding: str = "utf-8"
    has_header: bool = True
    text_column_index: int = 0

    def iter_files(self) -> Iterable[FileItem]:
        """Iterate over CSV-like files and yield text file items.

        Reads each configured path (including .gz variants), selecting
        text from the configured column. Missing files or read errors are
        logged and skipped.

        Yields:
            FileItem: An item containing encoded text and its relative
                path reference.
        """
        for path in self.paths:
            is_gz = "".join(path.suffixes[-2:]).lower() in {".csv.gz", ".tsv.gz"}
            opener = _open_csv_gz if is_gz else _open_csv
            try:
                with opener(path, encoding=self.encoding) as fp:
                    dialect_delim = self._resolve_delimiter(path)
                    if self.has_header:
                        reader = csv.DictReader(fp, delimiter=dialect_delim)
                        for lineno, row in enumerate(reader, start=2):
                            yield from self._row_to_fileitems(row=row, path=path, lineno=lineno)
                    else:
                        reader = csv.reader(fp, delimiter=dialect_delim)
                        for lineno, row in enumerate(reader, start=1):
                            yield from self._row_to_fileitems_no_header(row=row, path=path, lineno=lineno)
            except FileNotFoundError:
                log.warning("CSV file not found: %s", path)
            except Exception as exc:
                log.warning("Failed to read CSV file %s: %s", path, exc)

    def _resolve_delimiter(self, path: Path) -> str:
        """Return the delimiter for a file, falling back by extension."""
        if self.delimiter is not None:
            return self.delimiter
        suffixes = "".join(path.suffixes).lower()
        if ".tsv" in suffixes:
            return "\t"
        return ","

    def _row_to_fileitems(self, *, row: dict[str, Any], path: Path, lineno: int) -> Iterable[FileItem]:
        """Create file items from a dict row using the configured column.

        Args:
            row (Dict[str, Any]): Row keyed by header values.
            path (Path): Source file path.
            lineno (int): Line number used for relative references.

        Yields:
            FileItem: An item per non-empty text value.
        """
        text = _extract_text_from_row_with_header(row, self.text_column)
        if text is None:
            return
        rel = _derive_rel_path(path, lineno, row)
        data = text.encode("utf-8")
        yield FileItem(path=rel, data=data, size=len(data))

    def _row_to_fileitems_no_header(self, *, row: Sequence[str], path: Path, lineno: int) -> Iterable[FileItem]:
        """Create file items from a row when no header is present.

        Args:
            row (Sequence[str]): Row values by index.
            path (Path): Source file path.
            lineno (int): Line number used for relative references.

        Yields:
            FileItem: An item per non-empty text value.
        """
        idx = self.text_column_index
        if not (0 <= idx < len(row)):
            return
        text = (row[idx] or "").strip()
        if not text:
            return
        rel = f"{path.name}:#{lineno}"
        data = text.encode("utf-8")
        yield FileItem(path=rel, data=data, size=len(data))


def _extract_text_from_row_with_header(row: dict[str, Any], text_column: str) -> str | None:
    """Return stripped text from a header row or None if missing."""
    val = row.get(text_column)
    if isinstance(val, str):
        text = val.strip()
        return text or None
    return None


def _derive_rel_path(path: Path, lineno: int, row: dict[str, Any]) -> str:
    """Derive a relative reference path from row metadata or fallback."""
    for key in ("path", "filepath", "file_path", "id"):
        val = row.get(key)
        if isinstance(val, str) and val:
            return val
    return f"{path.name}:#{lineno}"


def _open_csv(path: Path, *, encoding: str):
    """Open a plain CSV file with newline handling for the csv module."""
    # newline="" ensures correct handling of embedded newlines/quoting for csv module.
    return open(path, encoding=encoding, newline="")


def _open_csv_gz(path: Path, *, encoding: str):
    """Open a gzip-compressed CSV file in text mode."""
    return gzip.open(path, "rt", encoding=encoding)
