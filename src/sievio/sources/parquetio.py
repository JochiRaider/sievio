# parquetio.py
# SPDX-License-Identifier: MIT
"""Parquet helpers for detecting payloads and yielding text/meta records."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from ..core.chunk import ChunkPolicy
from ..core.interfaces import Record, RepoContext
from ..core.log import get_logger

log = get_logger(__name__)

DEFAULT_TEXT_COLUMN = "text"
DEFAULT_META_COLUMN = "meta"
_PARQUET_MAGIC = b"PAR1"


def sniff_parquet(data: bytes, rel: str) -> bool:
    """Heuristically detects Parquet content from a filename or magic bytes.

    Args:
        data (bytes): File bytes to inspect.
        rel (str): Relative path or filename for extension checking.

    Returns:
        bool: True if the payload appears to be Parquet.
    """
    name = (rel or "").lower()
    if name.endswith(".parquet"):
        return True
    return data.startswith(_PARQUET_MAGIC)


def _iter_rows_from_table(
    table: pa.Table,
    rel: str,
    ctx: RepoContext | None,
    *,
    text_column: str = DEFAULT_TEXT_COLUMN,
    meta_column: str = DEFAULT_META_COLUMN,
) -> Iterable[Record]:
    """Yields records from a PyArrow table using text/meta columns.

    Args:
        table (pa.Table): Table containing Parquet rows.
        rel (str): Path hint used when constructing metadata.
        ctx (RepoContext | None): Optional context for default metadata.
        text_column (str): Column name containing textual content.
        meta_column (str): Column name containing metadata dictionaries.

    Returns:
        Iterable[Record]: Records with text and merged metadata, or an
            empty tuple when required columns are missing.
    """
    if text_column not in table.column_names or meta_column not in table.column_names:
        log.info(
            "Parquet handler skipping %s: missing columns %s/%s",
            rel,
            text_column,
            meta_column,
        )
        return ()

    ctx_defaults = ctx.as_meta_seed() if ctx else None
    text_arr = table[text_column]
    meta_arr = table[meta_column]
    nrows = len(table)

    def _iter() -> Iterable[Record]:
        for idx in range(nrows):
            try:
                text_val = text_arr[idx].as_py()
            except Exception:
                text_val = None
            try:
                raw_meta = meta_arr[idx].as_py()
            except Exception:
                raw_meta = None
            meta: dict[str, Any] = dict(raw_meta) if isinstance(raw_meta, dict) else {}
            if ctx_defaults:
                for key, value in ctx_defaults.items():
                    meta.setdefault(key, value)
            path_hint = meta.get("path")
            rel_hint = (
                path_hint
                if isinstance(path_hint, str) and path_hint
                else f"{rel}#row={idx}"
            )
            meta.setdefault("path", rel_hint)
            meta.setdefault("chunk_id", 1)
            meta.setdefault("n_chunks", 1)
            text_str = (
                text_val
                if isinstance(text_val, str)
                else ("" if text_val is None else str(text_val))
            )
            yield {"text": text_str, "meta": meta}

    return _iter()


def handle_parquet(
    data: bytes,
    rel: str,
    ctx: RepoContext | None,
    chunk_policy: ChunkPolicy | None,
) -> Iterable[Record] | None:
    """Reads Parquet bytes and emits records, ignoring chunk policy.

    Args:
        data (bytes): Raw Parquet payload.
        rel (str): Path hint for logging and metadata.
        ctx (RepoContext | None): Optional context for metadata defaults.
        chunk_policy (ChunkPolicy | None): Unused for Parquet rows.

    Returns:
        Iterable[Record] | None: Iterator over parsed records, or None if
            the payload cannot be read.
    """
    # ChunkPolicy is intentionally ignored; Parquet rows are treated as final chunks.
    try:
        table = pq.read_table(pa.BufferReader(data))
    except Exception as exc:
        log.info("Parquet handler could not read %s: %s", rel, exc)
        return None

    return _iter_rows_from_table(table, rel, ctx)


def iter_parquet_records(
    path_or_paths: str | Path | Sequence[str | Path],
    *,
    text_column: str = DEFAULT_TEXT_COLUMN,
    meta_column: str = DEFAULT_META_COLUMN,
) -> Iterable[Record]:
    """Iterates records from Parquet files or directories of Parquet data.

    Args:
        path_or_paths (str | Path | Sequence[str | Path]): File or
            directory paths to read.
        text_column (str): Column name containing textual content.
        meta_column (str): Column name containing metadata dictionaries.

    Returns:
        Iterable[Record]: Generator of records with text and metadata.
    """
    def _coerce_paths(p: str | Path | Sequence[str | Path]) -> list[Path]:
        if isinstance(p, (str, Path)):
            return [Path(p)]
        return [Path(x) for x in p]

    paths = _coerce_paths(path_or_paths)

    def _iter() -> Iterable[Record]:
        for p in paths:
            try:
                if p.is_dir():
                    dataset = ds.dataset(p, format="parquet")
                    batches = dataset.to_batches()
                    for batch in batches:
                        table = pa.Table.from_batches([batch])
                        yield from _iter_rows_from_table(
                            table,
                            str(p),
                            None,
                            text_column=text_column,
                            meta_column=meta_column,
                        )
                    continue
                pf = pq.ParquetFile(p)
            except Exception as exc:
                log.info("Skipping Parquet path %s: %s", p, exc)
                continue
            for batch in pf.iter_batches():
                table = pa.Table.from_batches([batch])
                yield from _iter_rows_from_table(
                    table, str(p), None, text_column=text_column, meta_column=meta_column
                )

    return _iter()


__all__ = ["sniff_parquet", "handle_parquet", "iter_parquet_records"]

# Register default bytes handler for Parquet payloads.
try:
    from ..core.registries import bytes_handler_registry

    bytes_handler_registry.register(sniff_parquet, handle_parquet)
except Exception:
    pass
