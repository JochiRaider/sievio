import logging

from sievio.core.log import configure_logging, temp_level
from sievio.core.naming import (
    build_output_basename_github,
    build_output_basename_pdf,
    normalize_extensions,
)


def test_configure_logging_sets_logger_level():
    logger = logging.getLogger("sievio")
    logger.setLevel(logging.WARNING)

    configure_logging(level="DEBUG")

    assert logger.level == logging.DEBUG


def test_temp_level_changes_and_restores():
    logger = logging.getLogger("sievio.test.temp")
    logger.setLevel(logging.WARNING)
    original_level = logger.level

    with temp_level(logging.DEBUG, name=logger.name):
        assert logger.level == logging.DEBUG

    assert logger.level == original_level


def test_build_output_basename_github_basic():
    name = build_output_basename_github(
        owner="owner",
        repo="repo-name",
        ref="main",
        license_spdx="MIT",
    )
    assert "repo-name" in name
    assert "MIT" in name
    assert "/" not in name
    assert "\\" not in name


def test_build_output_basename_github_sanitizes_and_limits_length():
    repo_full_name = "owner"
    long_repo = "this-is-a-very-long-repo-name-with-$%^&-chars"
    name = build_output_basename_github(
        owner=repo_full_name,
        repo=long_repo,
        ref="main",
        license_spdx="MIT",
    )
    for ch in "$%^& ":
        assert ch not in name
    assert len(name) <= 120


def test_build_output_basename_github_missing_license_uses_unknown():
    name = build_output_basename_github(
        owner="owner",
        repo="repo",
        ref="main",
        license_spdx=None,
    )
    assert "UNKNOWN" in name.upper()


def test_build_output_basename_pdf_basic():
    name = build_output_basename_pdf(
        url="https://example.com/My Document.pdf",
        title="My Document.pdf",
        license_spdx="CC-BY",
    )
    assert "my_document" in name or "my-document" in name
    assert "cc-by" in name or "CC-BY" in name
    assert "." not in name


def test_build_output_basename_pdf_missing_license_uses_unknown():
    name = build_output_basename_pdf(
        url="https://example.com/report.pdf",
        title="report.pdf",
        license_spdx=None,
    )
    assert "UNKNOWN" in name.upper()


def test_build_output_basename_pdf_sanitizes_and_limits_length():
    long_name = "a" * 200 + ".pdf"
    name = build_output_basename_pdf(
        url="https://example.com/very/long/path.pdf",
        title=long_name,
        license_spdx="MIT",
    )
    assert len(name) <= 120
    assert "/" not in name


def test_normalize_extensions_basic():
    normalized = normalize_extensions(["py", ".md", "  txt  "])
    assert normalized == {".py", ".md", ".txt"}


def test_normalize_extensions_handles_empty_and_none():
    assert normalize_extensions([]) is None
    assert normalize_extensions(None) is None
    assert normalize_extensions([" ", "", None]) is None
