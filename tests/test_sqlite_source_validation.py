import hashlib
from pathlib import Path

import pytest

from sievio.sources.sqlite_source import SQLiteSource


def test_sqlite_checksum_match(tmp_path):
    db_path = tmp_path / "good.db"
    payload = b"sqlite content"
    db_path.write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()

    source = SQLiteSource(db_path=db_path, checksum=sha)
    # Should not raise
    path = source._ensure_db_local()
    assert path == db_path


def test_sqlite_checksum_mismatch(tmp_path):
    db_path = tmp_path / "bad.db"
    db_path.write_bytes(b"malicious content")
    expected_sha = hashlib.sha256(b"valid content").hexdigest()

    source = SQLiteSource(db_path=db_path, checksum=expected_sha)

    with pytest.raises(ValueError, match="Checksum mismatch"):
        source._ensure_db_local()


def test_sqlite_checksum_verified_after_download(tmp_path, monkeypatch):
    db_path = tmp_path / "fetched.db"
    expected_payload = b"downloaded db"
    sha = hashlib.sha256(expected_payload).hexdigest()

    def fake_download():
        db_path.write_bytes(expected_payload)

    source = SQLiteSource(db_path=db_path, db_url="http://example.invalid/db", checksum=sha)
    monkeypatch.setattr(source, "_download_db", fake_download)

    assert source._ensure_db_local() == db_path
