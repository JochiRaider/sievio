import pytest

from sievio.core.config import SourceSpec
from sievio.core.factories_sources import (
    CsvTextSourceFactory,
    GitHubZipSourceFactory,
    LocalDirSourceFactory,
    SQLiteSourceFactory,
    WebPagePdfSourceFactory,
    WebPdfListSourceFactory,
)
from sievio.core.interfaces import SourceFactoryContext


class _DummyHttpClient:
    pass


class _DummyHttpConfig:
    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = timeout

    def build_client(self) -> _DummyHttpClient:
        return _DummyHttpClient()


def _make_ctx(defaults: dict[str, dict[str, object]]) -> SourceFactoryContext:
    return SourceFactoryContext(
        repo_context=None,
        http_client=_DummyHttpClient(),
        http_config=_DummyHttpConfig(),
        source_defaults=defaults,
    )


def test_local_dir_spec_overrides_defaults():
    ctx = _make_ctx({"local_dir": {"max_file_bytes": 100}})
    spec = SourceSpec(kind="local_dir", options={"root_dir": ".", "max_file_bytes": 200})

    source = LocalDirSourceFactory().build(ctx, spec)[0]

    assert source._cfg.max_file_bytes == 200


def test_github_zip_spec_overrides_defaults():
    ctx = _make_ctx({"github_zip": {"max_members": 100}})
    spec = SourceSpec(
        kind="github_zip",
        options={"url": "https://github.com/owner/repo", "max_members": 10},
    )

    source = GitHubZipSourceFactory().build(ctx, spec)[0]

    assert source._cfg.max_members == 10


def test_web_pdf_list_overlays_pdf_config_from_options():
    ctx = _make_ctx({"web_pdf_list": {"timeout": 10, "max_pdf_bytes": 1_024}})
    spec = SourceSpec(
        kind="web_pdf_list",
        options={
            "urls": ["https://example.com/doc.pdf"],
            "timeout": 5,
            "max_pdf_bytes": 2_048,
            "retries": 3,
        },
    )

    source = WebPdfListSourceFactory().build(ctx, spec)[0]

    assert source.timeout == 5
    assert source.max_pdf_bytes == 2_048
    assert source.retries == 3


def test_web_page_pdf_overlays_pdf_config_from_options():
    ctx = _make_ctx(
        {"web_page_pdf": {"timeout": 9, "max_pdf_bytes": 5_000, "include_ambiguous": True, "max_links": 5}}
    )
    spec = SourceSpec(
        kind="web_page_pdf",
        options={
            "page_url": "https://example.com/page",
            "timeout": 1,
            "max_pdf_bytes": 321,
            "include_ambiguous": False,
            "max_links": 2,
            "retries": 6,
        },
    )

    source = WebPagePdfSourceFactory().build(ctx, spec)[0]

    assert source.timeout == 1
    assert source.max_pdf_bytes == 321
    assert source.max_links == 2
    assert source.include_ambiguous is False
    assert source.retries == 6


def test_unknown_options_rejected():
    ctx = _make_ctx({"local_dir": {"skip_hidden": False}})
    spec = SourceSpec(kind="local_dir", options={"root_dir": ".", "unknown": "value"})

    with pytest.raises(ValueError):
        LocalDirSourceFactory().build(ctx, spec)


def test_sqlite_factory_applies_option_overrides():
    ctx = _make_ctx({"sqlite": {"batch_size": 10, "download_max_bytes": 100, "retries": 2}})
    spec = SourceSpec(
        kind="sqlite",
        options={
            "db_path": "dummy.db",
            "batch_size": 5,
            "download_max_bytes": 50,
            "retries": 7,
            "text_columns": ("body",),
        },
    )

    source = SQLiteSourceFactory().build(ctx, spec)[0]

    assert source.batch_size == 5
    assert source.download_max_bytes == 50
    assert source.retries == 7
    assert source.text_columns == ("body",)


def test_csv_factory_applies_option_overrides():
    ctx = _make_ctx({"csv_text": {"encoding": "utf-8"}})
    spec = SourceSpec(
        kind="csv_text",
        options={
            "paths": ["data.csv"],
            "encoding": "utf-16",
            "text_column": "body",
            "delimiter": "|",
            "has_header": False,
            "text_column_index": 3,
        },
    )

    source = CsvTextSourceFactory().build(ctx, spec)[0]

    assert source.encoding == "utf-16"
    assert source.text_column == "body"
    assert source.delimiter == "|"
    assert source.has_header is False
    assert source.text_column_index == 3
