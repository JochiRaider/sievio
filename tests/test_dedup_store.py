import multiprocessing as mp
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.connection import Connection

import pytest

from sievio.core.dedup_store import GlobalDedupStore


def _run_check_and_add(
    start_conn: Connection,
    result_conn: Connection,
    db_path: str,
    n_perm: int,
    bands: int,
    doc_id: str,
    sig: tuple[int, ...],
    content_hash: str | None,
) -> None:
    result_conn.send("ready")
    start_conn.recv()
    with GlobalDedupStore(db_path, n_perm=n_perm, bands=bands) as store:
        res = store.check_and_add(doc_id, sig, content_hash=content_hash, add_if_missing=True)
    result_conn.send((res.is_duplicate, res.match_id))


def test_check_and_add_flags_duplicates(tmp_path) -> None:
    db_path = tmp_path / "dedup.db"
    with GlobalDedupStore(db_path, n_perm=64, bands=16, jaccard_threshold=0.8) as store:
        sig = tuple(range(64))
        res_first = store.check_and_add("a", sig, content_hash="h1", add_if_missing=True)
        assert res_first.is_duplicate is False
        res_second = store.check_and_add("b", sig, content_hash="h1", add_if_missing=True)
        assert res_second.is_duplicate is True
        assert res_second.match_id == "a"
        assert res_second.score == pytest.approx(1.0)


def test_signature_length_mismatch_raises(tmp_path) -> None:
    db_path = tmp_path / "dedup.db"
    with GlobalDedupStore(db_path, n_perm=64, bands=16) as store:
        bad_sig = tuple(range(10))
        with pytest.raises(AssertionError):
            store.check_and_add("a", bad_sig)
        with pytest.raises(AssertionError):
            store.bulk_add([("b", bad_sig, None)])


def test_read_only_missing_db(tmp_path) -> None:
    missing = tmp_path / "missing.db"
    with pytest.raises(FileNotFoundError):
        GlobalDedupStore(missing, read_only=True)


def test_metadata_parameter_mismatch_raises(tmp_path) -> None:
    db_path = tmp_path / "dedup.db"
    with GlobalDedupStore(db_path, n_perm=128, bands=32, jaccard_threshold=0.82):
        pass
    with pytest.raises(ValueError, match="parameter mismatch"):
        GlobalDedupStore(db_path, n_perm=256, bands=32, jaccard_threshold=0.82)


def test_exact_dedup_short_circuit(tmp_path) -> None:
    db_path = tmp_path / "dedup.db"
    with GlobalDedupStore(db_path, n_perm=64, bands=16, jaccard_threshold=0.8) as store:
        sig_a = tuple(range(64))
        sig_b = tuple(reversed(sig_a))
        store.check_and_add("a", sig_a, content_hash="samehash", add_if_missing=True)
        res = store.check_and_add("b", sig_b, content_hash="samehash", add_if_missing=True)
        assert res.is_duplicate is True
        assert res.match_id == "a"
        assert res.score == pytest.approx(1.0)


def test_schema_migration_adds_content_hash(tmp_path) -> None:
    db_path = tmp_path / "old.db"

    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE signatures (
                doc_id TEXT PRIMARY KEY,
                signature BLOB
            );
            CREATE TABLE lsh_index (
                band_key TEXT,
                doc_id TEXT
            );
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            """
        )
    with GlobalDedupStore(db_path, n_perm=64, bands=16):
        pass
    with sqlite3.connect(db_path) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(signatures)")}
        assert "content_hash" in cols


def test_global_dedup_store_handles_concurrent_writes(tmp_path) -> None:
    db_path = tmp_path / "dedup.db"
    n_perm = 8
    bands = 2

    def make_sig(seed: int) -> tuple[int, ...]:
        return tuple((seed + i) % 97 for i in range(n_perm))

    def add_one(i: int) -> bool:
        with GlobalDedupStore(db_path, n_perm=n_perm, bands=bands, persistent_connection=False) as store:
            res = store.check_and_add(f"id-{i}", make_sig(i), add_if_missing=True)
            return res.is_duplicate

    with ThreadPoolExecutor(max_workers=4) as pool:
        dup_flags = list(pool.map(add_one, range(20)))

    assert all(flag is False for flag in dup_flags)
    with sqlite3.connect(db_path) as conn:
        (count,) = conn.execute("SELECT COUNT(*) FROM signatures").fetchone()
        assert count == 20


def test_content_hash_enforced_uniqueness_under_concurrency(tmp_path) -> None:
    db_path = tmp_path / "dedup.db"
    n_perm = 8
    bands = 2
    sig = tuple(range(n_perm))

    ctx = mp.get_context("spawn")
    start_conns = []
    result_conns = []
    procs = []
    for doc_id in ("doc-a", "doc-b"):
        start_child, start_parent = ctx.Pipe(duplex=False)
        result_parent, result_child = ctx.Pipe(duplex=False)
        proc = ctx.Process(
            target=_run_check_and_add,
            args=(start_child, result_child, str(db_path), n_perm, bands, doc_id, sig, "samehash"),
        )
        procs.append(proc)
        start_conns.append(start_parent)
        result_conns.append(result_parent)

    for proc in procs:
        proc.start()
    for result_conn in result_conns:
        assert result_conn.recv() == "ready"
    for start_conn in start_conns:
        start_conn.send("go")
    for proc in procs:
        proc.join(timeout=15)
    for proc in procs:
        assert proc.exitcode == 0

    results = [conn.recv() for conn in result_conns]
    dup_flags = [dup for dup, _match in results]
    assert sum(dup_flags) == 1
    match_ids = [match for dup, match in results if dup]
    assert len(match_ids) == 1
    assert match_ids[0] in {"doc-a", "doc-b"}

    with sqlite3.connect(db_path) as conn:
        (count,) = conn.execute("SELECT COUNT(*) FROM signatures").fetchone()
        assert count == 1


def test_atomic_check_and_add_prevents_near_dup_race(tmp_path) -> None:
    db_path = tmp_path / "dedup.db"
    n_perm = 8
    bands = 2
    sig = tuple(range(n_perm))

    ctx = mp.get_context("spawn")
    start_conns = []
    result_conns = []
    procs = []
    for doc_id in ("doc-a", "doc-b"):
        start_child, start_parent = ctx.Pipe(duplex=False)
        result_parent, result_child = ctx.Pipe(duplex=False)
        proc = ctx.Process(
            target=_run_check_and_add,
            args=(start_child, result_child, str(db_path), n_perm, bands, doc_id, sig, None),
        )
        procs.append(proc)
        start_conns.append(start_parent)
        result_conns.append(result_parent)

    for proc in procs:
        proc.start()
    for result_conn in result_conns:
        assert result_conn.recv() == "ready"
    for start_conn in start_conns:
        start_conn.send("go")
    for proc in procs:
        proc.join(timeout=15)
    for proc in procs:
        assert proc.exitcode == 0

    results = [conn.recv() for conn in result_conns]
    dup_flags = [dup for dup, _match in results]
    assert sum(dup_flags) == 1
    match_ids = [match for dup, match in results if dup]
    assert len(match_ids) == 1
    assert match_ids[0] in {"doc-a", "doc-b"}

    with sqlite3.connect(db_path) as conn:
        (count,) = conn.execute("SELECT COUNT(*) FROM signatures").fetchone()
        assert count == 2
