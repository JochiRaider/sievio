from pathlib import Path

from sievio.core.chunk import ChunkPolicy
from sievio.core.config import SievioConfig
from sievio.core.convert import (
    _build_record_context,
    build_records_from_bytes,
    iter_records_from_bytes,
    iter_records_from_file_item,
    list_records_for_file,
)
from sievio.core.factories_sources import UnsupportedBinary
from sievio.core.interfaces import FileItem, RepoContext


def test_list_records_for_file_basic() -> None:
    cfg = SievioConfig()
    ctx = RepoContext(repo_full_name="owner/repo", repo_url="https://github.com/owner/repo", license_id="MIT")
    text = "Line1\nLine2\nLine3"
    file_bytes = len(text.encode("utf-8"))

    records = list(
        list_records_for_file(
            text=text,
            rel_path="src/file.py",
            config=cfg,
            context=ctx,
            encoding="utf-8",
            had_replacement=False,
            file_bytes=file_bytes,
        )
    )

    assert records
    meta = records[0]["meta"]
    assert meta["chunk_id"] == 1
    assert meta["n_chunks"] == len(records)
    assert meta["file_bytes"] == file_bytes
    assert meta["file_nlines"] == 3
    assert meta["repo"] == "owner/repo"
    assert meta["repo_url"] == "https://github.com/owner/repo"
    assert meta["repo_full_name"] == "owner/repo"


def test_iter_records_from_bytes_respects_max_bytes_and_source() -> None:
    cfg = SievioConfig()
    cfg.decode.max_bytes_per_file = 50
    ctx = RepoContext(repo_full_name="owner/repo", repo_url="https://github.com/owner/repo", license_id="MIT")
    text = "A" * 1_000
    data = text.encode("utf-8")
    source_url = "https://example.com/file.txt"
    source_domain = "example.com"

    records = list(
        iter_records_from_bytes(
            data,
            "file.txt",
            config=cfg,
            context=ctx,
            file_size=len(data),
            source_url=source_url,
            source_domain=source_domain,
        )
    )

    assert records
    for rec in records:
        meta = rec["meta"]
        assert len(rec["text"].encode("utf-8")) <= cfg.decode.max_bytes_per_file
        assert meta["file_bytes"] == len(data)
        assert meta.get("truncated_bytes") is None or meta["truncated_bytes"] <= len(data)
        assert meta.get("url") == source_url
        assert meta.get("source_domain") == source_domain


def test_build_records_from_bytes_allows_custom_handler() -> None:
    cfg = SievioConfig()
    ctx = RepoContext()
    record_ctx = _build_record_context(cfg, ctx)
    sentinel = {"text": "SENTINEL", "meta": {"custom": True}}

    def sniff_always_true(data: bytes, rel_path: str) -> bool:
        return True

    def handler_return_sentinel(data: bytes, rel_path: str, context: RepoContext | None, chunk_policy: ChunkPolicy):
        return [sentinel]

    records = list(
        build_records_from_bytes(
            b"ignored",
            "ignored.txt",
            record_ctx=record_ctx,
            bytes_handlers=[(sniff_always_true, handler_return_sentinel)],
            extractors=[],
            context=ctx,
            chunk_policy=cfg.chunk.policy,
        )
    )

    assert records == [sentinel]


def test_build_records_from_bytes_handler_raises_falls_back() -> None:
    cfg = SievioConfig()
    ctx = RepoContext()
    record_ctx = _build_record_context(cfg, ctx)
    text = "Hello from bytes"
    data = text.encode("utf-8")

    def sniff_true(_: bytes, __: str) -> bool:
        return True

    def handler_raises(_: bytes, __: str, ___: RepoContext | None, ____: ChunkPolicy):
        raise UnsupportedBinary("nope")

    records = list(
        build_records_from_bytes(
            data,
            "notes.txt",
            record_ctx=record_ctx,
            bytes_handlers=[(sniff_true, handler_raises)],
            extractors=[],
            context=ctx,
            chunk_policy=cfg.chunk.policy,
        )
    )

    assert records
    assert any(rec["text"] == text for rec in records)


def test_iter_records_from_file_item_uses_streaming_extractor_when_available(tmp_path: Path) -> None:
    class DummyStreamingExtractor:
        def __init__(self) -> None:
            self.called = False

        def extract_stream(self, *, stream, path: str, context):
            self.called = True
            data = stream.read()
            yield {"text": data.decode("utf-8"), "meta": {"from_stream": True}}

    file_path = tmp_path / "stream.txt"
    file_path.write_text("streamed content", encoding="utf-8")
    item = FileItem(
        path="stream.txt",
        data=None,  # type: ignore[arg-type]
        size=file_path.stat().st_size,
        origin_path=str(file_path),
        streamable=True,
    )
    cfg = SievioConfig()
    streaming = DummyStreamingExtractor()

    records = list(
        iter_records_from_file_item(
            item,
            config=cfg,
            context=None,
            streaming_extractor=streaming,  # type: ignore[arg-type]
        )
    )

    assert streaming.called is True
    assert len(records) == 1
    assert records[0]["meta"]["from_stream"] is True
    assert records[0]["text"] == "streamed content"
