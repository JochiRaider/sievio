# parquet.py
# SPDX-License-Identifier: MIT
"""Parquet sink for writing records to files or partitioned datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Mapping, Any, Iterable, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from ..core.interfaces import Sink, RepoContext, Record
from ..core.log import get_logger

log = get_logger(__name__)


class ParquetDatasetSink(Sink):
    """Write records to a Parquet file or dataset.

    Records are buffered into Arrow tables with two primary columns:
    - text_field (default: "text")
    - meta_field (default: "meta") stored as a struct mirroring the
      meta dict

    Optional partition_by keys are lifted out of the meta dict into
    top-level columns so pyarrow.parquet.write_to_dataset can partition
    on them.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        text_field: str = "text",
        meta_field: str = "meta",
        partition_by: Optional[Iterable[str]] = None,
        compression: str = "snappy",
        row_group_size: Optional[int] = None,
        overwrite: bool = True,
    ) -> None:
        """Initialize the sink configuration.

        Args:
            path (str | Path): Target file path or dataset directory.
            text_field (str): Column name for text content.
            meta_field (str): Column name for metadata struct values.
            partition_by (Iterable[str] | None): Meta keys to elevate to
                top-level columns for dataset partitioning.
            compression (str): Parquet compression codec name.
            row_group_size (int | None): Maximum records per row group.
            overwrite (bool): Whether to replace existing output.
        """
        self._target = Path(path)
        self._is_dataset = False
        self._text_field = text_field or "text"
        self._meta_field = meta_field or "meta"
        self._partition_by: list[str] = [str(k) for k in (partition_by or [])]
        self._compression = compression or "snappy"
        self._row_group_size = row_group_size
        self._overwrite = bool(overwrite)
        self._buffer: list[Record] = []
        self._writer: Optional[pq.ParquetWriter] = None
        self._context: Optional[RepoContext] = None
        self._closed = False
        self._append_existing = False

    def open(self, context: Optional[RepoContext] = None) -> None:
        """Prepare the sink for writing.

        Args:
            context (RepoContext | None): Repository context for this
                write session.
        """
        self._context = context
        self._buffer.clear()
        self._writer = None
        self._closed = False
        self._append_existing = False

        # Decide whether the target is a dataset directory or a single file.
        path_is_file = self._target.suffix.lower() == ".parquet" and not self._partition_by
        self._is_dataset = not path_is_file

        if self._is_dataset:
            # Treat the provided path as a dataset directory.
            if self._target.exists() and self._target.is_file():
                if self._overwrite:
                    self._target.unlink()
                else:
                    raise FileExistsError(f"Parquet dataset path {self._target} already exists as a file")
            self._target.mkdir(parents=True, exist_ok=True)
        else:
            self._target.parent.mkdir(parents=True, exist_ok=True)
            if self._target.exists():
                if self._overwrite:
                    self._target.unlink()
                else:
                    self._append_existing = True

    def write(self, record: Record) -> None:
        """Buffer a record and flush when the row group is full.

        Calls are ignored after the sink is closed.

        Args:
            record (Record): Record to add to the buffer.
        """
        if self._closed:
            log.warning("Write called on closed ParquetDatasetSink; ignoring record.")
            return
        self._buffer.append(record)
        if self._row_group_size and len(self._buffer) >= self._row_group_size:
            self._flush_buffer()

    def finalize(self, records: Iterable[Record]) -> None:
        """Write a sequence of records using the same buffering logic.

        Args:
            records (Iterable[Record]): Records to enqueue for writing.
        """
        for rec in records:
            self.write(rec)

    def close(self) -> None:
        """Flush any buffered data and release resources."""
        if self._closed:
            return
        try:
            if self._buffer:
                self._flush_buffer()
        finally:
            if self._writer is not None:
                try:
                    self._writer.close()
                finally:
                    self._writer = None
            self._closed = True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _flush_buffer(self) -> None:
        """Write buffered records to disk and clear the buffer."""
        if not self._buffer:
            return
        table = self._build_table(self._buffer)
        if self._is_dataset:
            pq.write_to_dataset(
                table,
                root_path=self._target,
                partition_cols=self._partition_by or None,
                compression=self._compression,
                row_group_size=self._row_group_size,
            )
        else:
            if self._append_existing:
                pq.write_table(
                    table,
                    self._target,
                    compression=self._compression,
                    append=True,
                    row_group_size=self._row_group_size,
                )
            else:
                if self._writer is None:
                    self._writer = pq.ParquetWriter(
                        self._target,
                        table.schema,
                        compression=self._compression,
                    )
                self._writer.write_table(table, row_group_size=self._row_group_size)
        self._buffer.clear()

    def _build_table(self, rows: Sequence[Record]) -> pa.Table:
        """Convert buffered records into an Arrow table.

        Args:
            rows (Sequence[Record]): Records to convert.

        Returns:
            pa.Table: Table with text, metadata, and partition columns.
        """
        out_rows: list[dict[str, Any]] = []
        for rec in rows:
            text_val: Any = ""
            meta_val: Mapping[str, Any] | None = None
            if isinstance(rec, Mapping):
                text_val = rec.get("text", "")
                meta_val = rec.get("meta") if isinstance(rec.get("meta"), Mapping) else None
            if meta_val is None:
                meta_val = {}
            row: dict[str, Any] = {
                self._text_field: text_val if text_val is not None else "",
                self._meta_field: dict(meta_val),
            }
            for key in self._partition_by:
                row[key] = meta_val.get(key)
            out_rows.append(row)
        return pa.Table.from_pylist(out_rows)


__all__ = ["ParquetDatasetSink"]
