# dedup_store.py
# SPDX-License-Identifier: MIT
"""
SQLite-backed global deduplication store for MinHash signatures.

The store persists signatures and band keys while enforcing MinHash LSH
parameters via metadata. It supports both persistent and context-managed
connections plus batched inserts for seeding.
"""

from __future__ import annotations

import math
import sqlite3
import struct
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

from .log import get_logger
from .qc_utils import MinHashLSH

log = get_logger(__name__)
_BULK_INSERT_BATCH_SIZE = 10_000


class DedupCheckResult(NamedTuple):
    is_duplicate: bool
    score: float  # estimated Jaccard similarity from MinHash
    match_id: str | None


class GlobalDedupStore:
    """Persist MinHash signatures and LSH band mappings in SQLite."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        read_only: bool = False,
        persistent_connection: bool = True,
        n_perm: int = 128,
        bands: int = 32,
        jaccard_threshold: float = 0.82,
    ) -> None:
        # Expand ~ and normalize path
        self.db_path = Path(db_path).expanduser()
        self.read_only = read_only
        self._persistent = persistent_connection

        # Local LSH logic definition
        self.lsh_logic = MinHashLSH(
            n_perm=n_perm,
            bands=bands,
            jaccard_threshold=jaccard_threshold,
        )

        self._conn: sqlite3.Connection | None = None

        if self._persistent:
            self._conn = self._connect()
            if not self.read_only:
                self._init_schema_and_metadata(self._conn)
            else:
                self._verify_metadata(self._conn)
        elif not self.read_only:
            with self._connect() as conn:
                self._init_schema_and_metadata(conn)
        else:
            # Stateless + read_only: just verify metadata on demand
            with self._connect() as conn:
                self._verify_metadata(conn)

    def _connect(self) -> sqlite3.Connection:
        if self.read_only and not self.db_path.exists():
            raise FileNotFoundError(
                f"Global dedup DB {self.db_path} not found. "
                "Run the seeding script or disable read_only to initialize it."
            )

        if not self.read_only:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        uri = f"file:{self.db_path}?mode={'ro' if self.read_only else 'rwc'}"
        conn = sqlite3.connect(uri, uri=True, timeout=30.0)

        if not self.read_only:
            try:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
            except sqlite3.Error:
                log.debug("Failed to set WAL pragmas on %s", self.db_path, exc_info=True)

        return conn

    def _init_schema_and_metadata(self, conn: sqlite3.Connection) -> None:
        # Core tables
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS signatures (
                doc_id TEXT PRIMARY KEY,
                signature BLOB,
                content_hash TEXT
            );
            CREATE TABLE IF NOT EXISTS lsh_index (
                band_key TEXT,
                doc_id TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_band_key ON lsh_index(band_key);
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            """
        )
        self._ensure_content_hash_column(conn)
        self._ensure_unique_content_hash_index(conn)
        self._ensure_metadata(conn)
        conn.commit()
        self._ensure_content_hash_column_present(conn)

    @staticmethod
    def _ensure_content_hash_column(conn: sqlite3.Connection) -> None:
        cur = conn.execute("PRAGMA table_info(signatures)")
        cols = {row[1] for row in cur.fetchall()}
        if "content_hash" not in cols:
            conn.execute("ALTER TABLE signatures ADD COLUMN content_hash TEXT")

    @staticmethod
    def _ensure_content_hash_column_present(conn: sqlite3.Connection) -> None:
        cur = conn.execute("PRAGMA table_info(signatures)")
        cols = {row[1] for row in cur.fetchall()}
        if "content_hash" not in cols:
            raise ValueError(
                "Global dedup DB is missing content_hash column; "
                "run with write access to migrate."
            )

    def _ensure_unique_content_hash_index(self, conn: sqlite3.Connection) -> None:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_content_hash_unique'"
        )
        has_index = cur.fetchone() is not None
        if has_index:
            return

        if self.read_only:
            if conn.in_transaction:
                conn.rollback()
            raise ValueError(
                f"Global dedup DB {self.db_path} is missing the unique content_hash index; "
                "re-open with write access to migrate."
            )

        if self._has_duplicate_content_hashes(conn):
            duplicate = self._find_duplicate_content_hash(conn)
            hint = f" (example: {duplicate})" if duplicate else ""
            if conn.in_transaction:
                conn.rollback()
            raise ValueError(
                f"Global dedup DB {self.db_path} contains duplicate content_hash values{hint}; "
                "deduplicate the table before enabling the unique index."
            )

        try:
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_content_hash_unique "
                "ON signatures(content_hash) WHERE content_hash IS NOT NULL;"
            )
        except sqlite3.IntegrityError as exc:
            if conn.in_transaction:
                conn.rollback()
            raise ValueError(
                f"Failed to create unique content_hash index for {self.db_path}; "
                "existing duplicate hashes must be removed."
            ) from exc

    @staticmethod
    def _has_duplicate_content_hashes(conn: sqlite3.Connection) -> bool:
        cur = conn.execute(
            """
            SELECT 1
            FROM signatures
            WHERE content_hash IS NOT NULL
            GROUP BY content_hash
            HAVING COUNT(*) > 1
            LIMIT 1
            """
        )
        return cur.fetchone() is not None

    @staticmethod
    def _find_duplicate_content_hash(conn: sqlite3.Connection) -> str | None:
        cur = conn.execute(
            """
            SELECT content_hash
            FROM signatures
            WHERE content_hash IS NOT NULL
            GROUP BY content_hash
            HAVING COUNT(*) > 1
            LIMIT 1
            """
        )
        row = cur.fetchone()
        return row[0] if row else None

    def _ensure_metadata(self, conn: sqlite3.Connection) -> None:
        cur = conn.execute("SELECT key, value FROM metadata")
        rows = dict(cur.fetchall())

        expected = {
            "n_perm": str(self.lsh_logic.n_perm),
            "bands": str(self.lsh_logic.bands),
            "jaccard_threshold": repr(self.lsh_logic.jaccard_threshold),
        }

        if not rows:
            # Fresh DB: store our parameters
            conn.executemany(
                "INSERT INTO metadata (key, value) VALUES (?, ?)",
                expected.items(),
            )
            return

        # Existing DB: verify parameters
        try:
            stored_n_perm = int(rows["n_perm"])
            stored_bands = int(rows["bands"])
            stored_thresh = float(rows["jaccard_threshold"])
        except KeyError as e:
            raise ValueError(f"Global dedup DB {self.db_path} is missing metadata key {e}") from e

        if (
            stored_n_perm != self.lsh_logic.n_perm
            or stored_bands != self.lsh_logic.bands
            or not math.isclose(
                stored_thresh,
                self.lsh_logic.jaccard_threshold,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        ):
            raise ValueError(
                "Global dedup DB parameter mismatch: "
                f"stored (n_perm={stored_n_perm}, bands={stored_bands}, "
                f"jaccard_threshold={stored_thresh}) != "
                f"requested (n_perm={self.lsh_logic.n_perm}, "
                f"bands={self.lsh_logic.bands}, "
                f"jaccard_threshold={self.lsh_logic.jaccard_threshold})"
            )

    def _verify_metadata(self, conn: sqlite3.Connection) -> None:
        # Reuse the same logic, but do not create tables
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='metadata'")
        if not cur.fetchone():
            raise ValueError(
                f"Global dedup DB {self.db_path} has no metadata table; "
                "it may predate parameter tracking."
            )
        self._ensure_metadata(conn)
        self._ensure_content_hash_column_present(conn)
        self._ensure_unique_content_hash_index(conn)

    def _get_conn(self) -> sqlite3.Connection:
        if self._persistent:
            if self._conn is None:
                self._conn = self._connect()
            return self._conn
        return self._connect()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> GlobalDedupStore:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @staticmethod
    def _pack_sig(sig: tuple[int, ...]) -> bytes:
        return struct.pack(f"<{len(sig)}I", *sig)

    def _unpack_sig(self, blob: bytes) -> tuple[int, ...]:
        expected_len = self.lsh_logic.n_perm
        expected_size = expected_len * 4
        if len(blob) != expected_size:
            raise ValueError(
                f"Stored signature in {self.db_path} has length {len(blob)} bytes; "
                f"expected {expected_size} for {expected_len} permutations."
            )
        return struct.unpack(f"<{expected_len}I", blob)

    @staticmethod
    def _begin_immediate(conn: sqlite3.Connection) -> None:
        conn.execute("BEGIN IMMEDIATE;")

    def check_and_add(
        self,
        doc_id: str,
        sig: tuple[int, ...],
        *,
        content_hash: str | None = None,
        add_if_missing: bool = True,
    ) -> DedupCheckResult:
        """Check for near-duplicates and optionally add the signature.

        When ``add_if_missing`` is True, the lookup and insert run inside a
        single ``BEGIN IMMEDIATE`` transaction so concurrent writers cannot
        race past one another. The underlying database enforces uniqueness for
        non-null ``content_hash`` values.
        """
        assert len(sig) == self.lsh_logic.n_perm, (
            "Signature length mismatch for configured n_perm"
        )
        if add_if_missing and self.read_only:
            raise ValueError("Cannot add to GlobalDedupStore in read-only mode.")

        conn = self._get_conn()
        close_conn = not self._persistent
        try:
            if add_if_missing:
                self._begin_immediate(conn)

            result, band_keys = self._check_and_add_impl(
                conn, doc_id, sig, content_hash=content_hash
            )

            if add_if_missing and band_keys is not None:
                try:
                    self._insert_signature_and_index(conn, doc_id, sig, content_hash, band_keys)
                except sqlite3.IntegrityError as exc:
                    result = self._handle_integrity_error(
                        conn, doc_id, content_hash, fallback_result=result, exc=exc
                    )
                    if conn.in_transaction:
                        conn.rollback()
                    return result

            if add_if_missing and conn.in_transaction:
                conn.commit()

            return result
        except Exception:
            if add_if_missing and conn.in_transaction:
                conn.rollback()
            raise
        finally:
            if close_conn:
                conn.close()

    def _check_and_add_impl(
        self,
        conn: sqlite3.Connection,
        doc_id: str,
        sig: tuple[int, ...],
        *,
        content_hash: str | None,
    ) -> tuple[DedupCheckResult, list[str] | None]:
        if content_hash:
            cur = conn.execute(
                "SELECT doc_id FROM signatures WHERE content_hash = ?",
                (content_hash,),
            )
            row = cur.fetchone()
            if row is not None:
                return DedupCheckResult(True, 1.0, row[0]), None

        band_keys = [
            f"{b}:{self.lsh_logic.band_key(sig, b)[1]}" for b in range(self.lsh_logic.bands)
        ]

        candidates: set[str] = set()
        if band_keys:
            placeholders = ",".join("?" * len(band_keys))
            cur = conn.execute(
                f"SELECT DISTINCT doc_id FROM lsh_index WHERE band_key IN ({placeholders})",
                band_keys,
            )
            candidates = {row[0] for row in cur.fetchall()}

        best_eq = 0
        best_id: str | None = None
        if candidates:
            placeholders = ",".join("?" * len(candidates))
            cur = conn.execute(
                f"SELECT doc_id, signature FROM signatures WHERE doc_id IN ({placeholders})",
                tuple(candidates),
            )
            for cand_id, blob in cur.fetchall():
                cand_sig = self._unpack_sig(blob)
                eq = sum(1 for a, b in zip(sig, cand_sig, strict=False) if a == b)
                if eq > best_eq:
                    best_eq = eq
                    best_id = cand_id

        score = best_eq / self.lsh_logic.n_perm
        is_dup = score >= self.lsh_logic.jaccard_threshold
        return DedupCheckResult(is_dup, score, best_id), band_keys

    def _insert_signature_and_index(
        self,
        conn: sqlite3.Connection,
        doc_id: str,
        sig: tuple[int, ...],
        content_hash: str | None,
        band_keys: list[str],
    ) -> None:
        conn.execute(
            "INSERT INTO signatures (doc_id, signature, content_hash) VALUES (?, ?, ?)",
            (doc_id, self._pack_sig(sig), content_hash),
        )
        conn.executemany(
            "INSERT INTO lsh_index (band_key, doc_id) VALUES (?, ?)",
            [(band_key, doc_id) for band_key in band_keys],
        )

    def _handle_integrity_error(
        self,
        conn: sqlite3.Connection,
        doc_id: str,
        content_hash: str | None,
        *,
        fallback_result: DedupCheckResult,
        exc: sqlite3.IntegrityError,
    ) -> DedupCheckResult:
        if content_hash:
            cur = conn.execute(
                "SELECT doc_id FROM signatures WHERE content_hash = ?",
                (content_hash,),
            )
            row = cur.fetchone()
            if row is not None:
                return DedupCheckResult(True, 1.0, row[0])
        log.debug("Doc %s already present in dedup store %s", doc_id, self.db_path, exc_info=exc)
        return fallback_result

    def bulk_add(self, items: Iterable[tuple[str, tuple[int, ...], str | None]]) -> None:
        """Insert many signatures efficiently."""
        if self.read_only:
            raise ValueError("Cannot bulk add to GlobalDedupStore in read-only mode.")

        batch: list[tuple[str, tuple[int, ...], str | None]] = []
        for doc_id, sig, content_hash in items:
            assert len(sig) == self.lsh_logic.n_perm, (
                "Signature length mismatch for configured n_perm"
            )
            batch.append((doc_id, sig, content_hash))
            if len(batch) >= _BULK_INSERT_BATCH_SIZE:
                self._bulk_insert(batch)
                batch.clear()
        if batch:
            self._bulk_insert(batch)

    def _bulk_insert(self, batch: list[tuple[str, tuple[int, ...]]]) -> None:
        if not batch:
            return
        conn = self._get_conn()
        if self._persistent:
            self._bulk_insert_with_conn(conn, batch, commit=True)
        else:
            with conn:
                self._bulk_insert_with_conn(conn, batch, commit=False)

    def _bulk_insert_with_conn(
        self,
        conn: sqlite3.Connection,
        batch: list[tuple[str, tuple[int, ...], str | None]],
        *,
        commit: bool,
    ) -> None:
        # Deduplicate doc_ids within the batch
        unique_batch: dict[str, tuple[tuple[int, ...], str | None]] = {}
        for doc_id, sig, content_hash in batch:
            if doc_id not in unique_batch:
                unique_batch[doc_id] = (sig, content_hash)

        doc_ids = list(unique_batch.keys())
        if not doc_ids:
            return

        placeholders = ",".join("?" * len(doc_ids))
        cur = conn.execute(
            f"SELECT doc_id FROM signatures WHERE doc_id IN ({placeholders})",
            doc_ids,
        )
        existing = {row[0] for row in cur.fetchall()}

        to_insert = [
            (doc_id, sig, content_hash)
            for doc_id, (sig, content_hash) in unique_batch.items()
            if doc_id not in existing
        ]
        if not to_insert:
            return

        sig_rows = [
            (doc_id, self._pack_sig(sig), content_hash) for doc_id, sig, content_hash in to_insert
        ]
        conn.executemany(
            "INSERT INTO signatures (doc_id, signature, content_hash) VALUES (?, ?, ?)",
            sig_rows,
        )

        index_rows = []
        for doc_id, sig, _content_hash in to_insert:
            for b in range(self.lsh_logic.bands):
                band_key = f"{b}:{self.lsh_logic.band_key(sig, b)[1]}"
                index_rows.append((band_key, doc_id))
        if index_rows:
            conn.executemany("INSERT INTO lsh_index (band_key, doc_id) VALUES (?, ?)", index_rows)

        if commit:
            conn.commit()
