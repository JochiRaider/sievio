import logging

from sievio.core.records import RECORD_META_SCHEMA_VERSION, check_record_schema


def test_check_record_schema_matches_version_no_warning(caplog):
    record = {"meta": {"schema_version": RECORD_META_SCHEMA_VERSION}}

    with caplog.at_level(logging.WARNING):
        check_record_schema(record)

    assert not caplog.records


def test_check_record_schema_future_version_warns(caplog):
    record = {"meta": {"schema_version": "999"}}

    with caplog.at_level(logging.WARNING):
        check_record_schema(record)

    assert any("schema_version" in rec.message for rec in caplog.records)


def test_check_record_schema_missing_version_no_warning(caplog):
    record = {"meta": {}}

    with caplog.at_level(logging.WARNING):
        check_record_schema(record)

    assert not caplog.records
