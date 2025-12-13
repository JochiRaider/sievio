# sqlite_source.py
# SPDX-License-Identifier: MIT
"""Source for emitting text records from SQLite queries or tables."""

from __future__ import annotations

import hashlib
import sqlite3
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Any

from ..core import safe_http
from ..core.interfaces import Source, FileItem, RepoContext
from ..core.log import get_logger

__all__ = ["SQLiteSource"]

log = get_logger(__name__)

_USER_AGENT = "sievio/0.1 (+https://github.com/jochiraider/sievio)"
_READ_CHUNK = 64 * 1024


@dataclass
class SQLiteSource(Source):
    """Extracts text data from SQLite databases as file-like items.

    Attributes:
        db_path (Path): Local path to the SQLite database file.
        context (RepoContext | None): Optional repository context.
        table (str | None): Table name to select from when no SQL is provided.
        sql (str | None): Custom SQL query to execute.
        text_columns (Sequence[str]): Columns to concatenate into text.
        id_column (str | None): Column used to form stable item paths.
        where (str | None): Optional WHERE clause for table mode.
        batch_size (int): Number of rows to fetch per batch.
        db_url (str | None): URL to download the database if missing.
        download_timeout (float | None): Timeout for database download.
        download_max_bytes (int | None): Cap for downloaded database size.
        retries (int): Retry count for downloads.
        client (safe_http.SafeHttpClient | None): HTTP client to use.
    """

    db_path: Path
    context: Optional[RepoContext] = None
    table: Optional[str] = None
    sql: Optional[str] = None
    text_columns: Sequence[str] = ("text",)
    id_column: Optional[str] = None
    where: Optional[str] = None
    batch_size: int = 1000
    db_url: Optional[str] = None
    download_timeout: Optional[float] = None
    download_max_bytes: Optional[int] = None
    retries: int = 2
    client: Optional[safe_http.SafeHttpClient] = None
    checksum: Optional[str] = None

    def __post_init__(self) -> None:
        """Normalizes configuration after initialization."""
        if isinstance(self.text_columns, str):
            self.text_columns = (self.text_columns,)
        else:
            self.text_columns = tuple(self.text_columns)
        try:
            self.batch_size = max(1, int(self.batch_size))
        except Exception:
            self.batch_size = 1000

    def iter_files(self) -> Iterable[FileItem]:
        """Yields FileItems for each row returned by the configured query."""
        try:
            db_path = self._ensure_db_local()
        except FileNotFoundError:
            log.warning("SQLite DB not found and no db_url provided: %s", self.db_path)
            return
        except Exception as exc:
            log.warning("Failed to prepare SQLite DB %s: %s", self.db_path, exc)
            return

        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            try:
                sql, label = self._build_query()
                label = label or db_path.name
                cur = conn.cursor()
                cur.execute(sql)
                idx = 0
                while True:
                    rows = cur.fetchmany(self.batch_size)
                    if not rows:
                        break
                    for row in rows:
                        idx += 1
                        result = self._row_to_text_and_path(row, idx, label)
                        if result is None:
                            continue
                        data, rel_path = result
                        yield FileItem(path=rel_path, data=data, size=len(data))
            finally:
                conn.close()
        except sqlite3.Error as exc:
            log.warning("Failed to read SQLite DB %s: %s", db_path, exc)

    def _verify_checksum(self, path: Path) -> None:
        """Verify database content against the configured checksum when provided."""
        if not self.checksum:
            return
        expected = self.checksum.strip().lower()
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        actual = h.hexdigest()
        if actual != expected:
            raise ValueError(f"Checksum mismatch for {path}: expected {expected}, got {actual}")

    def _ensure_db_local(self) -> Path:
        """Ensures the database file is available locally, downloading if needed."""
        if self.db_path.exists():
            self._verify_checksum(self.db_path)
            return self.db_path
        if not self.db_url:
            raise FileNotFoundError(self.db_path)
        self._download_db()
        self._verify_checksum(self.db_path)
        if self.db_path.exists():
            return self.db_path
        raise FileNotFoundError(self.db_path)

    def _quote_identifier(self, name: Optional[str]) -> Optional[str]:
        """Quotes SQLite identifiers to avoid conflicts with reserved keywords."""
        if not name:
            return name
        escaped = name.replace('"', '""')
        return f'"{escaped}"'

    def _build_query(self) -> Tuple[str, Optional[str]]:
        """Constructs the SQL query and a label for result paths.

        Note: The WHERE clause is used verbatim and must come from trusted input.
        """
        if self.sql:
            return self.sql, "query"
        if self.table:
            cols = list(self.text_columns)
            if self.id_column and self.id_column not in cols:
                cols = [self.id_column, *cols]
            if not cols:
                raise ValueError("text_columns must be provided for SQLite table mode")
            safe_cols = [self._quote_identifier(col) or col for col in cols]
            safe_table = self._quote_identifier(self.table) or self.table
            select_cols = ", ".join(safe_cols)
            sql = f"SELECT {select_cols} FROM {safe_table}"
            if self.where:
                # WHERE remains raw SQL to preserve caller flexibility; callers must sanitize.
                sql = f"{sql} WHERE {self.where}"
            return sql, self.table
        raise ValueError("Either sql or table must be provided for SQLiteSource")

    def _row_to_text_and_path(
        self, row: sqlite3.Row, idx: int, table_or_label: str
    ) -> Optional[Tuple[bytes, str]]:
        """Converts a database row into encoded text and a relative path."""
        parts: list[str] = []
        for col in self.text_columns:
            try:
                val = row[col]
            except Exception:
                continue
            if val is None:
                continue
            text_part = str(val).strip()
            if text_part:
                parts.append(text_part)
        if not parts:
            return None
        text = "\n\n".join(parts)
        rel_path = f"{table_or_label}:#{idx}"
        if self.id_column:
            try:
                id_val = row[self.id_column]
            except Exception:
                id_val = None
            if id_val is not None and str(id_val):
                rel_path = f"{table_or_label}:{id_val}"
        data = text.encode("utf-8")
        return data, rel_path

    def _download_db(self) -> None:
        """Downloads the SQLite database from the configured URL."""
        if not self.db_url:
            raise FileNotFoundError(self.db_path)
        client = self.client or safe_http.get_global_http_client()
        req = urllib.request.Request(self.db_url, headers={"User-Agent": _USER_AGENT})
        timeout = self.download_timeout
        try:
            with client.open_with_retries(req, timeout=timeout, retries=self.retries) as resp:
                if resp.status >= 400:
                    raise urllib.error.HTTPError(
                        self.db_url, resp.status, resp.reason, resp.headers, None
                    )
                length_header = resp.headers.get("Content-Length")
                if self.download_max_bytes is not None and length_header:
                    try:
                        content_len = int(length_header)
                        if content_len > self.download_max_bytes:
                            raise ValueError(
                                f"Content length {content_len} exceeds limit {self.download_max_bytes}"
                            )
                    except ValueError:
                        raise
                    except Exception:
                        pass
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
                total = 0
                try:
                    with open(self.db_path, "wb") as fp:
                        while True:
                            chunk = resp.read(_READ_CHUNK)
                            if not chunk:
                                break
                            fp.write(chunk)
                            total += len(chunk)
                            if self.download_max_bytes is not None and total > self.download_max_bytes:
                                raise ValueError(
                                    f"Download exceeded max bytes ({self.download_max_bytes})"
                                )
                except Exception:
                    try:
                        self.db_path.unlink()
                    except Exception:
                        pass
                    raise
        except Exception as exc:
            raise RuntimeError(f"Failed to download SQLite DB from {self.db_url}: {exc}") from exc
