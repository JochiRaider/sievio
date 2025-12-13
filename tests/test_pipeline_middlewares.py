from pathlib import Path
import json

from sievio.core.builder import build_pipeline_plan, build_engine, PipelineOverrides
from sievio.core.interfaces import RepoContext
from sievio.core.pipeline import PipelineEngine, _FuncRecordMiddleware
from sievio.core.config import SievioConfig, SourceSpec, SinkSpec


def _make_basic_plan(tmp_path: Path) -> PipelineEngine:
    cfg = SievioConfig()
    ctx = RepoContext(repo_full_name="local/test", repo_url="https://example.com/local", license_id="UNKNOWN")
    cfg.sinks.context = ctx

    src_root = tmp_path / "input"
    src_root.mkdir()
    (src_root / "file.py").write_text("print('hello')\n", encoding="utf-8")

    cfg.sources.specs = (SourceSpec(kind="local_dir", options={"root_dir": str(src_root)}),)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    jsonl_path = out_dir / "data.jsonl"
    prompt_path = out_dir / "data.prompt.txt"

    cfg.sinks.specs = (
        SinkSpec(
            kind="default_jsonl_prompt",
            options={"jsonl_path": str(jsonl_path), "prompt_path": str(prompt_path)},
        ),
    )

    plan = build_pipeline_plan(cfg, mutate=False, overrides=PipelineOverrides())
    return PipelineEngine(plan)


def _read_payloads(engine: PipelineEngine) -> list[dict]:
    cfg = engine.config
    jsonl_path = cfg.sinks.primary_jsonl_name or cfg.metadata.primary_jsonl
    lines = Path(jsonl_path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_record_middleware_adapter_sets_tag(tmp_path: Path):
    engine = _make_basic_plan(tmp_path)

    def tagger(record):
        meta = record.setdefault("meta", {})
        meta["tag"] = "x"
        return record

    engine.add_record_middleware(tagger)
    stats = engine.run()
    payloads = _read_payloads(engine)

    assert stats.records == 1
    assert any(rec.get("meta", {}).get("tag") == "x" for rec in payloads)


def test_file_middleware_adapter_can_filter(tmp_path: Path):
    engine = _make_basic_plan(tmp_path)

    def drop_all(item, records):
        return []

    engine.add_file_middleware(drop_all)
    stats = engine.run()
    payloads = _read_payloads(engine)
    has_file_records = any("path" in rec.get("meta", {}) for rec in payloads)

    assert stats.records == 0
    assert not has_file_records


def test_normalize_middlewares_wraps_bare_functions(tmp_path: Path):
    engine = _make_basic_plan(tmp_path)

    def tagger(record):
        meta = record.setdefault("meta", {})
        meta["normalized"] = True
        return record

    engine.record_middlewares = [tagger]
    engine._middlewares_normalized = False
    engine._normalize_middlewares()

    assert any(isinstance(mw, _FuncRecordMiddleware) for mw in engine.record_middlewares)

    sample = {"text": "hi", "meta": {}}
    processed = engine._apply_middlewares(sample)
    assert processed is not None
    assert processed["meta"]["normalized"] is True


def test_qc_hook_attaches_record_middleware(tmp_path: Path):
    engine = _make_basic_plan(tmp_path)

    class TagHook:
        def __init__(self) -> None:
            self.started = False
            self.finished = False

        def on_run_start(self, ctx):
            self.started = True

        def on_record(self, record):
            meta = record.setdefault("meta", {})
            meta["qc_tag"] = True
            return record

        def on_run_end(self, ctx):
            self.finished = True

    hook = TagHook()
    engine.plan.runtime.lifecycle_hooks = (hook,)
    # Keep engine's cached hooks in sync with the mutated runtime.
    engine._hooks = (hook,)

    stats = engine.run()
    payloads = _read_payloads(engine)

    assert stats.records == 1
    # Lifecycle hooks wired via PipelineRuntime are invoked by the engine.
    assert hook.started and hook.finished
    assert any(rec.get("meta", {}).get("qc_tag") is True for rec in payloads)


def test_sink_open_failure_increments_stats(tmp_path: Path):
    engine = _make_basic_plan(tmp_path)

    class FailingSink:
        def open(self, context=None):
            raise RuntimeError("boom")

        def write(self, record):
            raise AssertionError("should not be called")

        def close(self):
            return None

    engine.plan.runtime.sinks = (FailingSink(),)

    stats = engine.run()

    assert stats.sink_errors >= 1
    assert stats.records == 0


def test_pipeline_overrides_record_middlewares_wired_by_build_engine(tmp_path: Path):
    cfg = SievioConfig()
    ctx = RepoContext(repo_full_name="local/test", repo_url="https://example.com/local", license_id="UNKNOWN")
    cfg.sinks.context = ctx

    src_root = tmp_path / "input_overrides"
    src_root.mkdir()
    (src_root / "file.py").write_text("print('hello')\n", encoding="utf-8")

    cfg.sources.specs = (SourceSpec(kind="local_dir", options={"root_dir": str(src_root)}),)

    out_dir = tmp_path / "out_overrides"
    out_dir.mkdir()
    jsonl_path = out_dir / "data.jsonl"
    prompt_path = out_dir / "data.prompt.txt"

    cfg.sinks.specs = (
        SinkSpec(
            kind="default_jsonl_prompt",
            options={"jsonl_path": str(jsonl_path), "prompt_path": str(prompt_path)},
        ),
    )

    def tag(record):
        meta = record.setdefault("meta", {})
        meta["override_tag"] = True
        return record

    overrides = PipelineOverrides(record_middlewares=[tag])
    engine = build_engine(cfg, overrides=overrides)
    stats = engine.run()
    payloads = _read_payloads(engine)

    assert stats.records == 1
    assert any(rec.get("meta", {}).get("override_tag") is True for rec in payloads)


def test_pipeline_overrides_file_middlewares_wired_by_build_engine(tmp_path: Path):
    cfg = SievioConfig()
    ctx = RepoContext(repo_full_name="local/test", repo_url="https://example.com/local", license_id="UNKNOWN")
    cfg.sinks.context = ctx

    src_root = tmp_path / "input_overrides_file"
    src_root.mkdir()
    (src_root / "file.py").write_text("print('hello')\n", encoding="utf-8")

    cfg.sources.specs = (SourceSpec(kind="local_dir", options={"root_dir": str(src_root)}),)

    out_dir = tmp_path / "out_overrides_file"
    out_dir.mkdir()
    jsonl_path = out_dir / "data.jsonl"
    prompt_path = out_dir / "data.prompt.txt"

    cfg.sinks.specs = (
        SinkSpec(
            kind="default_jsonl_prompt",
            options={"jsonl_path": str(jsonl_path), "prompt_path": str(prompt_path)},
        ),
    )

    def tag_from_file(item, records):
        tagged = []
        for rec in records:
            meta = rec.setdefault("meta", {})
            meta["from_file_mw"] = getattr(item, "path", None)
            tagged.append(rec)
        return tagged

    overrides = PipelineOverrides(file_middlewares=[tag_from_file])
    engine = build_engine(cfg, overrides=overrides)
    stats = engine.run()
    payloads = _read_payloads(engine)

    assert stats.records == 1
    assert any("from_file_mw" in rec.get("meta", {}) for rec in payloads)
