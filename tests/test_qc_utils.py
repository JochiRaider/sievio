import pytest

from sievio.core.qc_utils import parse_ok


def test_parse_ok_rejects_partial_json():
    assert parse_ok('{"a": 1', "json") == 0.0
    assert parse_ok('{"a": 1}\n{invalid', "jsonl") == 0.0


def test_parse_ok_handles_invalid_yaml():
    try:
        import yaml  # noqa: F401
    except Exception:
        pytest.skip("yaml not available")

    assert parse_ok("key: : value\n -", "yaml") == 0.0


def test_parse_ok_handles_malformed_restructuredtext_or_markdown():
    # Very short or heading-free text should fall back to a reduced confidence score.
    assert parse_ok("nonsense content", "markdown") < 1.0
    assert parse_ok("title\n---\nbody", "restructuredtext") == pytest.approx(1.0)
