import pytest

from repocapsule.core.dedup_store import GlobalDedupStore


def test_check_and_add_flags_duplicates(tmp_path) -> None:
    db_path = tmp_path / "dedup.db"
    with GlobalDedupStore(db_path, n_perm=64, bands=16, jaccard_threshold=0.8) as store:
        sig = tuple(range(64))
        res_first = store.check_and_add("a", sig, add_if_missing=True)
        assert res_first.is_duplicate is False
        res_second = store.check_and_add("b", sig, add_if_missing=True)
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
            store.bulk_add([("b", bad_sig)])


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
