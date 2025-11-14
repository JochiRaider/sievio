from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Iterator

from repocapsule.config import RepocapsuleConfig
from repocapsule.factories import build_default_sinks
from repocapsule.interfaces import FileItem, RepoContext, Source
from repocapsule.runner import convert


class StaticSource(Source):
    def __init__(self, items: Iterable[FileItem], context: RepoContext | None = None) -> None:
        self._items = list(items)
        self.context = context

    def iter_files(self) -> Iterator[FileItem]:
        yield from self._items


def _make_file(path: str, text: str) -> FileItem:
    data = text.encode("utf-8")
    return FileItem(path=path, data=data, size=len(data))


def test_run_summary_footer_appended(tmp_path: Path) -> None:
    source = StaticSource([_make_file("doc.txt", "hello world")])

    cfg = RepocapsuleConfig()
    cfg.sources = replace(cfg.sources, sources=(source,))

    jsonl_path = tmp_path / "out.jsonl"
    sink_result = build_default_sinks(
        cfg.sinks,
        jsonl_path=jsonl_path,
        context=None,
    )
    cfg.sinks = sink_result.sink_config
    cfg.metadata = cfg.metadata.merged(sink_result.metadata)

    convert(cfg)

    assert jsonl_path.exists(), "JSONL output should be created"
    lines = [line for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    footer = json.loads(lines[-1])

    meta = footer.get("meta") or {}
    assert meta.get("kind") == "run_summary"
    assert isinstance(meta.get("config"), dict)
    assert isinstance(meta.get("stats"), dict)
    assert isinstance(meta.get("metadata"), dict)
    qc_summary = meta.get("qc_summary")
    assert isinstance(qc_summary, dict)
    assert qc_summary.get("mode") == cfg.qc.mode
    assert meta["metadata"].get("primary_jsonl", "").endswith("out.jsonl")
