from pathlib import Path

import pytest

from sievio.core.config import FileProcessingConfig, SievioConfig
from sievio.core.interfaces import FileItem
from sievio.core.pipeline import _ProcessFileCallable, _WorkItem


class RecordingExtractor:
    def __init__(self) -> None:
        self.extract_calls: list[dict] = []
        self.extract_stream_calls: list[dict] = []

    def extract(self, item, *, config, context=None):
        self.extract_calls.append({"item": item, "config": config, "context": context})
        return [{"kind": "extract"}]

    def extract_stream(self, *, stream, path, context=None):
        data = stream.read()
        self.extract_stream_calls.append({"path": path, "context": context, "data": data})
        return [{"kind": "stream"}]


def _make_file_cfg() -> FileProcessingConfig:
    cfg = SievioConfig()
    return FileProcessingConfig(decode=cfg.decode, chunk=cfg.chunk, pipeline=cfg.pipeline)


def _run_processor(extractor, item: FileItem, *, executor_kind: str = "thread"):
    processor = _ProcessFileCallable(
        config=_make_file_cfg(),
        file_extractor=extractor,
        executor_kind=executor_kind,
    )
    _, recs_iter = processor(_WorkItem(item=item, ctx=None))
    return list(recs_iter)


def test_streaming_path_used_for_thread_executor(tmp_path: Path):
    path = tmp_path / "stream.txt"
    path.write_bytes(b"hello")
    extractor = RecordingExtractor()
    item = FileItem(path="stream.txt", data=b"", origin_path=str(path), streamable=True)

    recs = _run_processor(extractor, item, executor_kind="thread")

    assert extractor.extract_stream_calls and not extractor.extract_calls
    assert extractor.extract_stream_calls[0]["data"] == b"hello"
    assert extractor.extract_stream_calls[0]["path"] == "stream.txt"
    assert recs == [{"kind": "stream"}]


def test_extract_used_when_not_thread_executor(tmp_path: Path):
    path = tmp_path / "nostream.txt"
    path.write_bytes(b"hello")
    extractor = RecordingExtractor()
    item = FileItem(path="nostream.txt", data=b"", origin_path=str(path), streamable=True)

    recs = _run_processor(extractor, item, executor_kind="process")

    assert extractor.extract_calls and not extractor.extract_stream_calls
    assert recs == [{"kind": "extract"}]


@pytest.mark.parametrize(
    "item",
    [
        lambda base: FileItem(path="not_streamable.txt", data=b"", origin_path=str(base / "file.txt"), streamable=False),
        lambda base: FileItem(path="no_origin.txt", data=b"", origin_path=None, streamable=True),
    ],
)
def test_extract_used_when_streaming_preconditions_missing(tmp_path: Path, item):
    path = tmp_path / "file.txt"
    path.write_bytes(b"content")
    extractor = RecordingExtractor()
    file_item = item(tmp_path)

    recs = _run_processor(extractor, file_item, executor_kind="thread")

    assert extractor.extract_calls and not extractor.extract_stream_calls
    assert recs == [{"kind": "extract"}]


def test_extract_used_when_stream_reopen_fails(tmp_path: Path):
    missing = tmp_path / "missing.txt"
    extractor = RecordingExtractor()
    item = FileItem(path="missing.txt", data=b"", origin_path=str(missing), streamable=True)

    recs = _run_processor(extractor, item, executor_kind="thread")

    assert extractor.extract_calls and not extractor.extract_stream_calls
    assert recs == [{"kind": "extract"}]
