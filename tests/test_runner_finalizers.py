import json

from sievio.core.hooks import _dispatch_finalizers
from sievio.core.config import SievioConfig
from sievio.core.records import build_run_header_record
from sievio.sinks.sinks import JSONLSink


def test_dispatch_finalizers_does_not_clobber_jsonl(tmp_path) -> None:
    jsonl_path = tmp_path / "data.jsonl"
    sink = JSONLSink(jsonl_path)
    sink.set_header_record(build_run_header_record(SievioConfig()))

    # Simulate the main pipeline writing records.
    sink.open()
    sink.write({"text": "body", "meta": {"kind": "chunk", "path": "file.txt"}})
    sink.close()

    summary = {"text": "", "meta": {"kind": "run_summary"}}
    _dispatch_finalizers([sink], summary, str(jsonl_path), None)

    kinds = [json.loads(line)["meta"]["kind"] for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    assert kinds == ["run_header", "chunk", "run_summary"]
