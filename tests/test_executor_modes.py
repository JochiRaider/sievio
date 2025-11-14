from __future__ import annotations

import sys
from dataclasses import replace
from typing import Iterable, Iterator, List, Dict, Any

import pytest

from repocapsule.config import RepocapsuleConfig
from repocapsule.interfaces import FileItem, RepoContext, Source, Sink
from repocapsule.pipeline import run_pipeline


class StaticSource(Source):
    def __init__(self, items: Iterable[FileItem]):
        self._items = list(items)
        self.context: RepoContext | None = None

    def iter_files(self) -> Iterator[FileItem]:
        yield from self._items


class RecordingSink(Sink):
    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []
        self._opened = False

    def open(self, context: RepoContext | None = None) -> None:
        self._opened = True

    def write(self, record: Dict[str, Any]) -> None:
        if not self._opened:
            raise RuntimeError("Sink not opened")
        self.records.append(record)

    def close(self) -> None:
        self._opened = False


def _make_file(path: str, text: str) -> FileItem:
    data = text.encode("utf-8")
    return FileItem(path=path, data=data, size=len(data))


def _build_config(executor_kind: str) -> tuple[RepocapsuleConfig, RecordingSink]:
    source = StaticSource([
        _make_file("a.txt", "alpha"),
        _make_file("b.txt", "beta"),
        _make_file("c.txt", "gamma"),
    ])
    sink = RecordingSink()
    cfg = RepocapsuleConfig()
    cfg.sources = replace(cfg.sources, sources=(source,))
    cfg.sinks = replace(cfg.sinks, sinks=(sink,), context=None)
    cfg.pipeline.max_workers = 2
    cfg.pipeline.executor_kind = executor_kind
    cfg.prepare()
    return cfg, sink


@pytest.mark.skipif(sys.platform == "win32", reason="process executor test requires fork-friendly platform")
def test_pipeline_process_executor_matches_thread():
    cfg_thread, sink_thread = _build_config("thread")
    stats_thread = run_pipeline(config=cfg_thread)

    cfg_process, sink_process = _build_config("process")
    try:
        stats_process = run_pipeline(config=cfg_process)
    except PermissionError as exc:
        pytest.skip(f"Process executor unavailable in this environment: {exc}")

    assert stats_thread["records"] == stats_process["records"]
    assert [rec["meta"]["path"] for rec in sink_thread.records] == [
        rec["meta"]["path"] for rec in sink_process.records
    ]
