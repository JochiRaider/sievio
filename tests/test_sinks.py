from __future__ import annotations

import gzip
import json
from pathlib import Path

from repocapsule.sinks import GzipJSONLSink
from repocapsule.config import SinkConfig
from repocapsule.factories import build_default_sinks
from repocapsule.interfaces import RepoContext


def test_gzip_jsonl_sink_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl.gz"
    sink = GzipJSONLSink(path)
    sink.open()
    sink.write({"hello": "world"})
    sink.close()

    with gzip.open(path, "rt", encoding="utf-8") as fp:
        lines = fp.read().splitlines()
    assert json.loads(lines[0]) == {"hello": "world"}


def test_build_default_sinks_compressed(tmp_path: Path) -> None:
    cfg = SinkConfig(output_dir=tmp_path, compress_jsonl=True, jsonl_basename="sample")
    result = build_default_sinks(cfg, basename="sample")
    assert result.jsonl_path.endswith(".jsonl.gz")
    # Ensure metadata and sink config point to gzip path
    assert result.metadata["primary_jsonl"].endswith(".jsonl.gz")
    sink = result.sinks[0]
    assert isinstance(sink, GzipJSONLSink)
