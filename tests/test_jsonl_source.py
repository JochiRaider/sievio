from __future__ import annotations

import gzip
import logging
from contextlib import contextmanager
from pathlib import Path

import pytest

from sievio.sources import jsonl_source
from sievio.sources.jsonl_source import JSONLTextSource


class _ListHandler(logging.Handler):
    def __init__(self, level: int) -> None:
        super().__init__(level)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@contextmanager
def _capture_jsonl_logs(level: int = logging.DEBUG):
    logger = logging.getLogger("sievio.sources.jsonl_source")
    handler = _ListHandler(level)
    logger.addHandler(handler)
    old_level = logger.level
    logger.setLevel(min(old_level, level) if old_level else level)
    try:
        yield handler.records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)


def test_jsonl_source_skips_non_dict_and_logs_summary(tmp_path: Path) -> None:
    path = tmp_path / "mix.jsonl"
    path.write_text(
        "\n".join(
            [
                "[]",
                '"text"',
                "123",
                '{"text": "ok", "meta": {"path": "meta.txt"}}',
                "{bad",
                '{"meta": {}}',
                '{"text": "later"}',
            ]
        )
    )
    src = JSONLTextSource(paths=(path,), check_schema=False)
    with _capture_jsonl_logs(level=logging.INFO) as records:
        items = list(src.iter_files())

    assert [item.path for item in items] == ["meta.txt", "mix/line_7.txt"]
    warnings = [rec for rec in records if rec.levelno == logging.WARNING]
    assert len(warnings) == 1  # invalid JSON line only
    summary = [rec for rec in records if rec.levelno == logging.INFO]
    assert summary
    assert any(
        "skipped_invalid=1" in rec.getMessage()
        and "skipped_nondict=3" in rec.getMessage()
        and "missing_text=1" in rec.getMessage()
        for rec in summary
    )


def test_jsonl_source_sanitizes_paths_and_fallback_suffix(tmp_path: Path) -> None:
    path = tmp_path / "sanitize.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"text": "ok", "meta": {"path": "/..\\\\unsafe\\\\segment\\\\file.md"}}',
                '{"text": "later"}',
                '{"text": "third", "meta": {"path": "a/../b.txt"}}',
                '{"text": "fourth", "meta": {"path": "\\\\a\\\\b\\\\c.txt"}}',
                '{"text": "fifth", "meta": {"path": "/abs/like/path.txt"}}',
                '{"text": "sixth", "meta": {"path": "/\\n..//dir/\\rfile.txt"}}',
            ]
        )
    )

    src = JSONLTextSource(paths=(path,), check_schema=False)
    items = list(src.iter_files())

    assert [item.path for item in items] == [
        "unsafe/segment/file.md",
        "sanitize/line_2.txt",
        "a/b.txt",
        "a/b/c.txt",
        "abs/like/path.txt",
        "dir/file.txt",
    ]


def test_jsonl_source_bounds_invalid_json_warnings(tmp_path: Path) -> None:
    path = tmp_path / "many_invalid.jsonl"
    lines = ["{bad"] * (jsonl_source._MAX_INVALID_JSON_WARNINGS + 2)
    lines.append('{"text": "ok"}')
    path.write_text("\n".join(lines))
    src = JSONLTextSource(paths=(path,), check_schema=False)
    with _capture_jsonl_logs(level=logging.WARNING) as records:
        items = list(src.iter_files())

    warning_msgs = [rec.getMessage() for rec in records if rec.levelno == logging.WARNING]
    assert len(warning_msgs) == jsonl_source._MAX_INVALID_JSON_WARNINGS
    assert all("Skipping invalid JSON" in msg for msg in warning_msgs)
    assert len(items) == 1
    assert items[0].path == "many_invalid/line_8.txt"


def test_jsonl_source_schema_check_toggle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "schema.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"text": "one", "meta": {"schema_version": "2"}}',
                '{"text": "two", "meta": {"custom": "x"}}',
            ]
        )
    )

    calls: list[dict] = []

    def fake_check(record, logger=None):
        calls.append(record)

    monkeypatch.setattr(jsonl_source, "check_record_schema", fake_check)

    src = jsonl_source.JSONLTextSource(paths=(path,), check_schema=True)
    list(src.iter_files())
    assert len(calls) == 1

    calls.clear()
    src_disabled = jsonl_source.JSONLTextSource(paths=(path,), check_schema=False)
    list(src_disabled.iter_files())
    assert not calls


def test_jsonl_source_meta_path_all_filtered_falls_back(tmp_path: Path) -> None:
    path = tmp_path / "dots.jsonl"
    path.write_text('{"text": "x", "meta": {"path": ".."}}\n{"text": "y", "meta": {"path": "."}}')

    src = JSONLTextSource(paths=(path,), check_schema=False)
    items = list(src.iter_files())

    assert [item.path for item in items] == ["dots/line_1.txt", "dots/line_2.txt"]


def test_jsonl_source_skips_oversized_line_and_continues(tmp_path: Path) -> None:
    path = tmp_path / "oversize.jsonl"
    oversize_line = '{"text": "' + ("x" * 40) + '"}'
    path.write_text("\n".join([oversize_line, '{"text": "ok"}']))
    policy = jsonl_source.JSONLReadPolicy(max_line_chars=20, max_text_chars=50)

    src = JSONLTextSource(paths=(path,), check_schema=False, read_policy=policy)
    with _capture_jsonl_logs(level=logging.WARNING) as records:
        items = list(src.iter_files())

    assert [item.path for item in items] == ["oversize/line_2.txt"]
    assert any("Skipping line" in rec.getMessage() for rec in records)


def test_jsonl_source_truncates_after_file_budget_for_gzip(tmp_path: Path) -> None:
    path = tmp_path / "budget.jsonl.gz"
    lines = ['{"text": "one"}', '{"text": "two"}', '{"text": "three"}']
    with gzip.open(path, "wt", encoding="utf-8") as fp:
        fp.write("\n".join(lines))
    first_line_with_newline = len(lines[0]) + 1
    policy = jsonl_source.JSONLReadPolicy(
        max_line_chars=50,
        max_text_chars=50,
        max_file_chars=first_line_with_newline,
    )

    src = JSONLTextSource(paths=(path,), check_schema=False, read_policy=policy)
    with _capture_jsonl_logs(level=logging.WARNING) as records:
        items = list(src.iter_files())

    assert [item.path for item in items] == ["budget/line_1.txt"]
    assert any("Truncated JSONL file" in rec.getMessage() for rec in records)
