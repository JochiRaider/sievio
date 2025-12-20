import io
import os
from pathlib import Path

import pytest

from sievio.core.chunk import ChunkPolicy
from sievio.core.config import SievioConfig
from sievio.core.convert import (
    _OPEN_STREAM_DEFAULT_MAX_BYTES,
    _build_record_context,
    build_records_from_bytes,
    iter_records_from_bytes,
    iter_records_from_file_item,
    list_records_for_file,
    make_limited_stream,
    maybe_reopenable_local_path,
    resolve_bytes_from_file_item,
)
from sievio.core.factories_sources import UnsupportedBinary
from sievio.core.interfaces import FileItem, RepoContext
from sievio.core.pipeline import _ProcessFileCallable, _WorkItem


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


def test_build_records_from_bytes_handler_raises_skips() -> None:
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

    assert records == []


def test_build_records_from_bytes_empty_iterable_short_circuits() -> None:
    cfg = SievioConfig()
    ctx = RepoContext()
    record_ctx = _build_record_context(cfg, ctx)

    def sniff_true(_: bytes, __: str) -> bool:
        return True

    def handler_returns_empty(_: bytes, __: str, ___: RepoContext | None, ____: ChunkPolicy):
        return []

    records = list(
        build_records_from_bytes(
            b"fallback?",
            "empty.bin",
            record_ctx=record_ctx,
            bytes_handlers=[(sniff_true, handler_returns_empty)],
            extractors=[],
            context=ctx,
            chunk_policy=cfg.chunk.policy,
        )
    )

    assert records == []


def test_build_records_from_bytes_none_handler_falls_back() -> None:
    cfg = SievioConfig()
    ctx = RepoContext()
    record_ctx = _build_record_context(cfg, ctx)
    text = "Decode me"
    data = text.encode("utf-8")

    def sniff_true(_: bytes, __: str) -> bool:
        return True

    def handler_returns_none(_: bytes, __: str, ___: RepoContext | None, ____: ChunkPolicy):
        return None

    records = list(
        build_records_from_bytes(
            data,
            "notes.txt",
            record_ctx=record_ctx,
            bytes_handlers=[(sniff_true, handler_returns_none)],
            extractors=[],
            context=ctx,
            chunk_policy=cfg.chunk.policy,
        )
    )

    assert any(rec["text"] == text for rec in records)


def test_iter_records_from_file_item_uses_streaming_extractor_when_available(tmp_path: Path) -> None:
    class DummyStreamingExtractor:
        def __init__(self) -> None:
            self.called = False
            self.stream = None

        def extract_stream(self, *, stream, path: str, context):
            self.called = True
            self.stream = stream
            data = stream.read()
            yield {"text": data.decode("utf-8"), "meta": {"from_stream": True}}

    file_path = tmp_path / "stream.txt"
    file_path.write_text("streamed content", encoding="utf-8")
    item = FileItem(
        path="stream.txt",
        data=None,
        size=file_path.stat().st_size,
        origin_path=str(file_path),
        stream_hint="file",
        streamable=True,
        open_stream=lambda: file_path.open("rb"),
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
    assert streaming.stream is not None and streaming.stream.closed is True


def test_iter_records_from_file_item_skips_streaming_when_hint_not_file() -> None:
    class DummyStreamingExtractor:
        def __init__(self) -> None:
            self.called = False

        def extract_stream(self, *, stream, path: str, context):
            self.called = True
            raise AssertionError("should not be called")

    item = FileItem(
        path="remote.pdf",
        data=b"payload",
        size=len(b"payload"),
        origin_path="https://example.com/remote.pdf",
        stream_hint="http",
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

    assert streaming.called is False
    assert records  # falls back to buffered decode


def test_iter_records_from_file_item_handles_missing_origin_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing.txt"
    item = FileItem(
        path="missing.txt",
        data=None,
        size=0,
        origin_path=str(missing),
        stream_hint="file",
        streamable=True,
    )
    cfg = SievioConfig()

    records = list(
        iter_records_from_file_item(
            item,
            config=cfg,
            context=None,
            streaming_extractor=None,
        )
    )

    assert records == []


def test_list_records_for_file_propagates_source_metadata_to_extractors() -> None:
    class DummyExtractor:
        def extract(self, *, text: str, path: str, context):
            return [{"text": "from extractor", "meta": {}}]

    cfg = SievioConfig()
    cfg.pipeline.extractors = (DummyExtractor(),)
    ctx = RepoContext()
    source_url = "https://example.com/resource"

    records = list(
        list_records_for_file(
            text="chunk me",
            rel_path="doc.txt",
            config=cfg,
            context=ctx,
            encoding="utf-8",
            had_replacement=False,
            source_url=source_url,
        )
    )

    extractor_meta = next(rec["meta"] for rec in records if rec["text"] == "from extractor")
    assert extractor_meta["url"] == source_url
    assert extractor_meta["source_domain"] == "example.com"


def test_make_limited_stream_enforces_cap_across_reads() -> None:
    stream = make_limited_stream(io.BytesIO(b"abcdefgh"), 4)
    assert stream.read(2) == b"ab"
    buf = bytearray(4)
    n = stream.readinto(buf)
    assert n == 2
    assert bytes(buf[:n]) == b"cd"
    assert stream.readline() == b""
    assert stream.read() == b""
    assert stream.raw.read(10) == b""


def test_make_limited_stream_limits_iteration_and_readline() -> None:
    stream = make_limited_stream(io.BytesIO(b"line1\nline2\n"), 7)
    assert stream.readline() == b"line1\n"
    assert next(stream) == b"l"
    with pytest.raises(StopIteration):
        next(stream)


def test_make_limited_stream_blocks_readinto1_when_available() -> None:
    class BytesIOWithReadinto1(io.BytesIO):
        def readinto1(self, b):  # type: ignore[override]
            return super().readinto(b)

    src = BytesIOWithReadinto1(b"123456")
    stream = make_limited_stream(src, 3)
    buf = bytearray(10)
    n1 = stream.readinto1(buf)
    assert n1 == 3
    assert bytes(buf[:n1]) == b"123"
    assert stream.read() == b""


def test_maybe_reopenable_local_path_rejects_remote_and_missing(tmp_path: Path) -> None:
    http_item = FileItem(
        path="remote.pdf",
        data=b"",
        origin_path="https://example.com/remote.pdf",
        stream_hint="http",
        streamable=True,
    )
    assert maybe_reopenable_local_path(http_item) is None

    missing = tmp_path / "absent.txt"
    missing_item = FileItem(
        path="absent.txt",
        data=b"",
        origin_path=str(missing),
        stream_hint="file",
        streamable=True,
    )
    assert maybe_reopenable_local_path(missing_item) is None


def test_maybe_reopenable_local_path_accepts_file_url(tmp_path: Path) -> None:
    p = tmp_path / "file.txt"
    p.write_text("hello", encoding="utf-8")
    item = FileItem(
        path="file.txt",
        data=b"",
        origin_path=f"file://{p}",
        stream_hint="file",
        streamable=True,
    )
    resolved = maybe_reopenable_local_path(item)
    assert resolved == p.resolve()


def test_pipeline_streaming_enforces_cap(tmp_path: Path) -> None:
    data = b"0123456789"
    path = tmp_path / "big.bin"
    path.write_bytes(data)
    cap = 5

    class ProbeExtractor:
        def __init__(self) -> None:
            self.chunks: list[bytes] = []
            self.stream = None

        def extract_stream(self, *, stream, path: str, context):
            self.stream = stream
            buf = bytearray(8)
            n1 = stream.readinto(buf)
            self.chunks.append(bytes(buf[: n1 or 0]))
            rest = stream.read()
            self.chunks.append(rest)
            yield {"text": "ok", "meta": {"first": n1, "rest": len(rest)}}

        def extract(self, item, *, config, context=None):
            return []

    extractor = ProbeExtractor()
    cfg = SievioConfig()
    cfg.decode.max_bytes_per_file = cap
    proc = _ProcessFileCallable(config=cfg, file_extractor=extractor, executor_kind="thread")
    item = FileItem(
        path="big.bin",
        data=None,
        size=len(data),
        origin_path=str(path),
        stream_hint="file",
        streamable=True,
        open_stream=lambda: path.open("rb"),
    )
    _, recs_iter = proc(_WorkItem(item=item, ctx=None))
    records = list(recs_iter)

    assert records
    meta = records[0]["meta"]
    assert (meta["first"] or 0) + meta["rest"] <= cap
    assert sum(len(chunk) for chunk in extractor.chunks) <= cap
    assert extractor.stream is not None and extractor.stream.closed is True


def test_source_url_sanitization_strips_credentials_and_query() -> None:
    cfg = SievioConfig()
    url = "https://user:pass@example.com:8443/path/to/file.txt?token=secret#frag"
    records = list(
        list_records_for_file(
            text="hi",
            rel_path="a.txt",
            config=cfg,
            context=RepoContext(),
            encoding="utf-8",
            had_replacement=False,
            source_url=url,
        )
    )
    meta = records[0]["meta"]
    assert meta["url"] == "https://example.com:8443/path/to/file.txt"
    assert meta["source_domain"] == "example.com"


def test_maybe_reopenable_local_path_rejects_directories(tmp_path: Path) -> None:
    dir_path = tmp_path / "dir"
    dir_path.mkdir()
    item = FileItem(
        path="dir",
        data=None,
        origin_path=str(dir_path),
        stream_hint="file",
        streamable=True,
    )
    assert maybe_reopenable_local_path(item) is None

    file_path = dir_path / "file.txt"
    file_path.write_text("data", encoding="utf-8")
    file_item = FileItem(
        path="dir/file.txt",
        data=None,
        origin_path=str(file_path),
        stream_hint="file",
        streamable=True,
    )
    assert maybe_reopenable_local_path(file_item) == file_path.resolve()


@pytest.mark.skipif(os.name != "nt", reason="Windows path semantics only")
def test_windows_style_path_not_treated_as_url(tmp_path: Path) -> None:
    win_style = tmp_path / "C" / "path"
    win_style.mkdir(parents=True)
    file_path = win_style / "file.txt"
    file_path.write_text("hello", encoding="utf-8")
    item = FileItem(
        path="C:\\path\\file.txt",
        data=None,
        origin_path=str(file_path),
        stream_hint="file",
        streamable=True,
        open_stream=lambda: file_path.open("rb"),
    )
    assert maybe_reopenable_local_path(item) == file_path.resolve()


def test_streaming_not_used_when_buffered_bytes_present(tmp_path: Path) -> None:
    class StreamingExtractor:
        def extract_stream(self, *, stream, path: str, context):
            raise AssertionError("should not be called")

    cfg = SievioConfig()
    item = FileItem(
        path="buf.txt",
        data=b"hello",
        origin_path=str(tmp_path / "buf.txt"),
        stream_hint="file",
        streamable=True,
        open_stream=lambda: (_ for _ in ()).throw(AssertionError("should not open stream")),  # type: ignore[arg-type]
    )
    records = list(
        iter_records_from_file_item(
            item,
            config=cfg,
            context=None,
            streaming_extractor=StreamingExtractor(),  # type: ignore[arg-type]
        )
    )
    assert records and records[0]["text"] == "hello"


def test_streaming_uses_byte_limit(tmp_path: Path) -> None:
    data = b"0123456789"
    path = tmp_path / "big.bin"
    path.write_bytes(data)
    cfg = SievioConfig()
    cfg.decode.max_bytes_per_file = 4

    class StreamingExtractor:
        def __init__(self) -> None:
            self.read = None

        def extract_stream(self, *, stream, path: str, context):
            self.read = stream.read()
            yield {"text": "ok", "meta": {}}

    extractor = StreamingExtractor()
    item = FileItem(
        path="big.bin",
        data=None,
        size=len(data),
        origin_path=str(path),
        stream_hint="file",
        streamable=True,
        open_stream=lambda: path.open("rb"),
    )
    list(
        iter_records_from_file_item(
            item,
            config=cfg,
            context=None,
            streaming_extractor=extractor,  # type: ignore[arg-type]
        )
    )
    assert extractor.read == data[: cfg.decode.max_bytes_per_file]


def test_open_stream_buffering_uses_default_limit_when_none(tmp_path: Path) -> None:
    big = b"A" * (_OPEN_STREAM_DEFAULT_MAX_BYTES + 1024)
    cfg = SievioConfig()
    cfg.decode.max_bytes_per_file = None

    item = FileItem(
        path="big.bin",
        data=None,
        size=len(big),
        origin_path="ignored",
        stream_hint="file",
        streamable=True,
        open_stream=lambda: io.BytesIO(big),
    )

    resolved = resolve_bytes_from_file_item(item, cfg.decode)
    assert resolved.data is not None
    assert len(resolved.data) == _OPEN_STREAM_DEFAULT_MAX_BYTES


def test_open_stream_buffering_respects_policy() -> None:
    called = {"open": False}

    def _bad_stream():
        called["open"] = True
        raise AssertionError("should not be called")

    cfg = SievioConfig()
    item = FileItem(
        path="a.bin",
        data=b"123",
        size=3,
        origin_path="ignored",
        stream_hint="zip",
        streamable=True,
        open_stream=_bad_stream,
    )

    resolved = resolve_bytes_from_file_item(item, cfg.decode)
    assert resolved.data == b"123"
    assert called["open"] is False


def test_open_stream_buffering_uses_max_bytes_when_set() -> None:
    data = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    cfg = SievioConfig()
    cfg.decode.max_bytes_per_file = 4

    item = FileItem(
        path="letters.bin",
        data=None,
        size=len(data),
        origin_path="ignored",
        stream_hint="file",
        streamable=True,
        open_stream=lambda: io.BytesIO(data),
    )

    resolved = resolve_bytes_from_file_item(item, cfg.decode)
    assert resolved.data == data[: cfg.decode.max_bytes_per_file]


def test_sniffer_and_handler_errors_do_not_crash() -> None:
    cfg = SievioConfig()
    ctx = RepoContext()
    record_ctx = _build_record_context(cfg, ctx)
    called = {"decode": False}

    def sniff_raises(data, rel):
        raise RuntimeError("sniff boom")

    def handler_raises(data, rel, context, policy):
        raise RuntimeError("handler boom")

    def sniff_false(data, rel):
        called["decode"] = True
        return False

    records = list(
        build_records_from_bytes(
            b"text",
            "f.txt",
            record_ctx=record_ctx,
            bytes_handlers=[(sniff_raises, handler_raises), (sniff_false, handler_raises)],
            extractors=[],
            context=ctx,
            chunk_policy=cfg.chunk.policy,
        )
    )
    assert called["decode"] is True
    assert any(rec["text"] == "text" for rec in records)


def test_extractor_invalid_record_is_skipped() -> None:
    class BadExtractor:
        def extract(self, *, text: str, path: str, context):
            return [{"text": "good", "meta": {}}, "bad"]

    cfg = SievioConfig()
    cfg.pipeline.extractors = (BadExtractor(),)
    records = list(
        list_records_for_file(
            text="hi",
            rel_path="a.txt",
            config=cfg,
            context=RepoContext(),
            encoding="utf-8",
            had_replacement=False,
        )
    )
    assert any(rec["text"] == "good" for rec in records)
    assert all(isinstance(rec["meta"], dict) for rec in records)
