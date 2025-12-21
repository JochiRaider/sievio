import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import sievio.core.pipeline as pipeline
from sievio.core.builder import (
    PipelineOverrides,
    PipelinePlan,
    PipelineRuntime,
    _assert_runtime_free_spec,
    build_pipeline_plan,
)
from sievio.core.concurrency import ExecutorConfig, resolve_pipeline_executor_config
from sievio.core.config import (
    PipelineConfig,
    QCConfig,
    QCHeuristics,
    SievioConfig,
    SinkSpec,
    SourceSpec,
    load_config_from_path,
)
from sievio.core.factories import SinkFactoryResult
from sievio.core.interfaces import RepoContext
from sievio.core.pipeline import PipelineEngine
from sievio.core.qc_controller import InlineQCHook
from sievio.core.registries import SinkRegistry, SourceRegistry
from sievio.core.runner import run_pipeline
from sievio.core.safe_http import SafeHttpClient


class DummyPdfSource:
    """Lightweight stand-in whose name triggers heavy-source detection."""


def handle_pdf(data, rel_path, ctx, policy):
    return None


def handle_evtx(data, rel_path, ctx, policy):
    return None


def _make_basic_spec(tmp_path: Path) -> SievioConfig:
    cfg = SievioConfig()

    ctx = RepoContext(
        repo_full_name="local/test",
        repo_url="https://example.com/local",
        license_id="UNKNOWN",
    )
    cfg.sinks.context = ctx

    src_root = tmp_path / "input"
    src_root.mkdir()
    (src_root / "file.py").write_text("print('hello')\n", encoding="utf-8")

    cfg.sources.specs = (
        SourceSpec(kind="local_dir", options={"root_dir": str(src_root)}),
    )

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    jsonl_path = out_dir / "data.jsonl"
    prompt_path = out_dir / "data.prompt.txt"

    cfg.sinks.specs = (
        SinkSpec(
            kind="default_jsonl_prompt",
            options={
                "jsonl_path": str(jsonl_path),
                "prompt_path": str(prompt_path),
            },
        ),
    )
    return cfg


# --------------------
# 2.1 config validate
# --------------------


def test_qcconfig_enabled_off_raises():
    qc = QCConfig(enabled=True, mode="off")
    with pytest.raises(ValueError):
        qc.validate()


def test_qcconfig_inline_without_scorer_raises():
    qc = QCConfig(enabled=True, mode="inline", scorer=None, scorer_id=None)
    with pytest.raises(ValueError) as excinfo:
        qc.validate()
    assert "Inline/advisory QC requires a scorer" in str(excinfo.value)

    qc = QCConfig(enabled=True, mode="advisory", scorer=None, scorer_id=None)
    with pytest.raises(ValueError) as excinfo:
        qc.validate()
    assert "Inline/advisory QC requires a scorer" in str(excinfo.value)


def test_qcconfig_inline_with_scorer_passes():
    qc = QCConfig(enabled=True, mode="inline", scorer=object(), scorer_id=None)
    qc.validate()

    qc = QCConfig(enabled=True, mode="advisory", scorer=object(), scorer_id=None)
    qc.validate()


def test_qcconfig_inline_with_scorer_id_passes():
    qc = QCConfig(enabled=True, mode="inline", scorer=None, scorer_id="jsonl_default")
    qc.validate()


def test_qcconfig_disabled_ignores_missing_scorer():
    qc = QCConfig(enabled=False, mode="inline", scorer=None, scorer_id=None)
    qc.validate()


def test_qcconfig_scorer_options_type_enforced():
    qc = QCConfig(enabled=True, mode="inline", scorer=None, scorer_id="jsonl_default", scorer_options=None)
    with pytest.raises(TypeError):
        qc.validate()


def test_qcheuristics_negative_values_raise():
    h = QCHeuristics()
    h.target_code_min = 0

    with pytest.raises(ValueError) as excinfo:
        h.validate()
    assert "QCHeuristics.target_code_min" in str(excinfo.value)


def test_qcheuristics_weight_out_of_range_raises():
    h = QCHeuristics()
    h.code_punct_weight = 1.5

    with pytest.raises(ValueError) as excinfo:
        h.validate()
    assert "QCHeuristics.code_punct_weight" in str(excinfo.value)


def test_pipeline_executor_kind_normalization_valid_values():
    cfg = SievioConfig()
    cfg.pipeline.executor_kind = " Thread "
    cfg.validate()
    assert cfg.pipeline.executor_kind == "thread"

    cfg = SievioConfig()
    cfg.pipeline.executor_kind = "AUTO"
    cfg.validate()
    assert cfg.pipeline.executor_kind == "auto"


def test_pipeline_executor_kind_invalid_raises():
    cfg = SievioConfig()
    cfg.pipeline.executor_kind = "invalid-kind"
    with pytest.raises(ValueError) as excinfo:
        cfg.validate()
    assert "pipeline.executor_kind" in str(excinfo.value)


def test_validate_paths_same_string_raises():
    cfg = SievioConfig()
    cfg.metadata.primary_jsonl = "out/data.jsonl"
    cfg.metadata.prompt_path = "out/data.jsonl"

    with pytest.raises(ValueError) as excinfo:
        cfg.validate()
    assert "primary_jsonl and prompt_path refer to the same file path" in str(excinfo.value)


def test_validate_paths_same_real_path_raises(tmp_path: Path):
    jsonl = tmp_path / "data.jsonl"

    cfg = SievioConfig()
    cfg.metadata.primary_jsonl = str(jsonl)
    cfg.metadata.prompt_path = str(tmp_path / "./data.jsonl")

    with pytest.raises(ValueError):
        cfg.validate()


def test_config_roundtrip_to_from_dict():
    cfg = SievioConfig()
    cfg.metadata.primary_jsonl = "out/data.jsonl"
    cfg.metadata.repo_url = "https://example.com/repo"

    data = cfg.to_dict()
    cfg2 = SievioConfig.from_dict(data)

    assert cfg2.metadata.primary_jsonl == cfg.metadata.primary_jsonl
    assert cfg2.metadata.repo_url == cfg.metadata.repo_url
    assert "sources" not in data.get("sources", {})
    assert "sinks" not in data.get("sinks", {})


def test_config_roundtrip_json(tmp_path: Path):
    cfg = SievioConfig()
    cfg.metadata.repo_url = "https://example.com/repo"
    path = tmp_path / "config.json"

    cfg.to_json(path)
    loaded = SievioConfig.from_json(path)

    assert loaded.metadata.repo_url == cfg.metadata.repo_url


config_mod = importlib.import_module("sievio.core.config")
HAS_TOML = getattr(config_mod, "tomllib", None) is not None


@pytest.mark.skipif(not HAS_TOML, reason="tomllib/tomli not available")
def test_config_roundtrip_toml(tmp_path: Path):
    cfg = SievioConfig()
    cfg.metadata.repo_url = "https://example.com/repo"

    path = tmp_path / "config.toml"
    content = f'[metadata]\nrepo_url = "{cfg.metadata.repo_url}"\n'
    path.write_text(content, encoding="utf-8")

    loaded = load_config_from_path(path)
    assert loaded.metadata.repo_url == cfg.metadata.repo_url


# ----------------------------
# 2.2 builder + concurrency
# ----------------------------


def test_assert_runtime_free_spec_with_http_client_raises():
    cfg = SievioConfig()
    cfg.http.client = SafeHttpClient(timeout=1.0)
    with pytest.raises(ValueError) as excinfo:
        _assert_runtime_free_spec(cfg)
    assert "http.client must be unset" in str(excinfo.value)


def test_assert_runtime_free_spec_with_file_extractor_raises():
    cfg = SievioConfig()
    cfg.pipeline.file_extractor = object()  # type: ignore[assignment]
    with pytest.raises(ValueError) as excinfo:
        _assert_runtime_free_spec(cfg)
    assert "pipeline.file_extractor must be unset" in str(excinfo.value)


def test_assert_runtime_free_spec_with_bytes_handlers_raises():
    def sniff(data: bytes, rel_path: str) -> bool:
        return False

    def handler(data: bytes, rel_path: str, ctx, policy):
        return None

    cfg = SievioConfig()
    cfg.pipeline.bytes_handlers = ((sniff, handler),)
    with pytest.raises(ValueError) as excinfo:
        _assert_runtime_free_spec(cfg)
    assert "pipeline.bytes_handlers must be empty" in str(excinfo.value)


def test_pipeline_overrides_http_client(tmp_path: Path):
    cfg = _make_basic_spec(tmp_path)
    override_client = SafeHttpClient(timeout=1.23)
    overrides = PipelineOverrides(http_client=override_client)

    plan = build_pipeline_plan(cfg, mutate=False, overrides=overrides)

    assert plan.runtime.http_client is override_client
    assert plan.spec.http.client is None


def test_pipeline_overrides_file_extractor(tmp_path: Path):
    class DummyExtractor:
        def extract(self, item, *, config, context=None):
            return []

    cfg = _make_basic_spec(tmp_path)
    extractor = DummyExtractor()
    overrides = PipelineOverrides(file_extractor=extractor)

    plan = build_pipeline_plan(cfg, mutate=False, overrides=overrides)

    assert plan.runtime.file_extractor is extractor
    assert plan.spec.pipeline.file_extractor is None


def test_pipeline_overrides_bytes_handlers(tmp_path: Path):
    def sniff(data: bytes, rel_path: str) -> bool:
        return True

    def handler(data: bytes, rel_path: str, ctx, policy):
        return []

    cfg = _make_basic_spec(tmp_path)
    overrides = PipelineOverrides(bytes_handlers=((sniff, handler),))

    plan = build_pipeline_plan(cfg, mutate=False, overrides=overrides)

    assert plan.runtime.bytes_handlers == ((sniff, handler),)
    assert plan.spec.pipeline.bytes_handlers == ()


def test_pipeline_overrides_qc_scorer(tmp_path: Path):
    cfg = _make_basic_spec(tmp_path)
    cfg.qc.enabled = True
    cfg.qc.mode = "inline"
    override_scorer = object()
    overrides = PipelineOverrides(qc_scorer=override_scorer)

    plan = build_pipeline_plan(cfg, mutate=False, overrides=overrides)

    assert any(isinstance(h, InlineQCHook) for h in plan.runtime.lifecycle_hooks)
    assert plan.spec.qc.scorer is None
    assert plan.runtime.qc_scorer_for_csv is None


def test_overrides_end_to_end_custom_extractor_and_qc(tmp_path: Path):
    class RecordingExtractor:
        def __init__(self):
            self.calls = 0

        def extract(self, item, *, config, context=None):
            self.calls += 1
            return [{"text": "override-extractor", "meta": {"path": getattr(item, "path", None)}}]

    class RecordingScorer:
        def __init__(self):
            self.calls = 0

        def score_record(self, record):
            self.calls += 1
            meta = record.get("meta", {}) if isinstance(record, dict) else {}
            return {"score": 99, "path": meta.get("path"), "tokens": 1}

    cfg = _make_basic_spec(tmp_path)
    cfg.qc.enabled = True
    cfg.qc.mode = "inline"
    cfg.qc.min_score = None

    extractor = RecordingExtractor()
    scorer = RecordingScorer()
    overrides = PipelineOverrides(file_extractor=extractor, qc_scorer=scorer)

    plan = build_pipeline_plan(cfg, mutate=False, overrides=overrides)
    engine = PipelineEngine(plan)
    stats = engine.run()

    jsonl_path = plan.spec.sinks.primary_jsonl_name or plan.spec.metadata.primary_jsonl
    lines = Path(jsonl_path).read_text(encoding="utf-8").splitlines()
    payloads = [json.loads(line) for line in lines if line.strip()]
    has_override_record = any(rec.get("text") == "override-extractor" for rec in payloads)

    assert extractor.calls == 1
    assert scorer.calls == 1
    assert stats.records == 1
    quality_stats = stats.qc.get_screener("quality", create=False)
    assert quality_stats is not None
    assert quality_stats.scored == 1
    assert has_override_record


def test_overrides_end_to_end_bytes_handler(tmp_path: Path):
    calls = {"handler": 0}

    def sniff(data: bytes, rel_path: str) -> bool:
        return True

    def handler(data: bytes, rel_path: str, ctx, policy):
        calls["handler"] += 1
        return [{"text": "override-bytes-handler", "meta": {"path": rel_path}}]

    cfg = _make_basic_spec(tmp_path)
    overrides = PipelineOverrides(bytes_handlers=((sniff, handler),))

    plan = build_pipeline_plan(cfg, mutate=False, overrides=overrides)
    engine = PipelineEngine(plan)
    stats = engine.run()

    jsonl_path = plan.spec.sinks.primary_jsonl_name or plan.spec.metadata.primary_jsonl
    lines = Path(jsonl_path).read_text(encoding="utf-8").splitlines()
    payloads = [json.loads(line) for line in lines if line.strip()]
    has_override_record = any(rec.get("text") == "override-bytes-handler" for rec in payloads)

    assert calls["handler"] == 1
    assert stats.records == 1
    assert has_override_record


def test_build_pipeline_plan_registry_wiring(tmp_path: Path):
    cfg = _make_basic_spec(tmp_path)
    plan = build_pipeline_plan(cfg, mutate=False)

    assert plan.runtime.sources
    assert plan.runtime.sinks
    assert plan.runtime.http_client is not None
    assert plan.spec.metadata.primary_jsonl == plan.spec.sinks.primary_jsonl_name


def test_resolve_executor_config_auto_no_heavy():
    cfg = SievioConfig()
    cfg.pipeline.executor_kind = "auto"

    exec_cfg, fail_fast = resolve_pipeline_executor_config(cfg, runtime=None)

    assert exec_cfg.kind == "thread"
    assert exec_cfg.max_workers >= 1
    assert exec_cfg.window >= exec_cfg.max_workers
    assert fail_fast is bool(cfg.pipeline.fail_fast)


def test_resolve_executor_config_auto_heavy_handlers_and_sources():
    cfg = SievioConfig()
    cfg.pipeline.executor_kind = "auto"
    runtime = SimpleNamespace(
        bytes_handlers=(
            (lambda b, p: False, handle_pdf),
            (lambda b, p: False, handle_evtx),
        ),
        sources=(DummyPdfSource(),),
    )

    exec_cfg, _ = resolve_pipeline_executor_config(cfg, runtime=runtime)

    assert exec_cfg.kind == "process"


def test_resolve_executor_config_auto_only_heavy_handlers():
    cfg = SievioConfig()
    cfg.pipeline.executor_kind = "auto"
    runtime = SimpleNamespace(
        bytes_handlers=((lambda b, p: False, handle_pdf),),
        sources=(),
    )

    exec_cfg, _ = resolve_pipeline_executor_config(cfg, runtime=runtime)
    assert exec_cfg.kind == "thread"


def test_resolve_executor_config_auto_only_heavy_sources():
    cfg = SievioConfig()
    cfg.pipeline.executor_kind = "auto"
    runtime = SimpleNamespace(
        bytes_handlers=(),
        sources=(DummyPdfSource(),),
    )

    exec_cfg, _ = resolve_pipeline_executor_config(cfg, runtime=runtime)
    assert exec_cfg.kind == "thread"


def test_process_parallel_submit_error_increments_errors(caplog):
    cfg = SievioConfig()
    runtime = PipelineRuntime(
        http_client=None,
        sources=(),
        sinks=(),
        file_extractor=cfg.pipeline.file_extractor or PipelineConfig().file_extractor or pipeline.DefaultExtractor(),
        bytes_handlers=(),
        lifecycle_hooks=(),
        executor_config=None,
        fail_fast=False,
        qc_scorer_for_csv=None,
        post_qc_scorer=None,
    )
    plan = PipelinePlan(spec=cfg, runtime=runtime)
    engine = PipelineEngine(plan)

    work = pipeline._WorkItem(  # type: ignore[attr-defined]
        item=SimpleNamespace(path="bad.txt", size=10, data=b""),
        ctx=None,
    )

    def processor(w):
        return w.item, []

    class DummyExecutor:
        def __init__(self):
            self.cfg = ExecutorConfig(max_workers=2, window=2, kind="process")

        def map_unordered(
            self,
            items,
            fn,
            on_result,
            *,
            fail_fast,
            on_error=None,
            on_submit_error=None,
            ):
                for it in items:
                    if on_submit_error:
                        on_submit_error(it, RuntimeError("submit fail"))

    with caplog.at_level("WARNING", logger="sievio.core.pipeline"):
        engine._process_parallel(  # type: ignore[attr-defined]
            [work],
            processor=processor,
            sinks=[],
            executor=DummyExecutor(),
            fail_fast=False,
        )

    assert engine.stats.files == 1
    assert engine.stats.source_errors == 1


def test_build_pipeline_plan_mutate_false_preserves_original_config(tmp_path: Path):
    config_path = Path(__file__).resolve().parents[1] / "example_config.toml"
    cfg = SievioConfig.from_toml(config_path)

    class StubSource:
        def iter_files(self):
            return []

    class StubSink:
        def open(self, context=None):
            return None

        def write(self, record):
            return None

        def close(self):
            return None

        def finalize(self, records):
            return None

    class StubSourceFactory:
        def __init__(self, factory_id: str):
            self.id = factory_id

        def build(self, ctx, spec):
            return [StubSource()]

    class StubSinkFactory:
        def __init__(self, factory_id: str):
            self.id = factory_id

        def build(self, ctx, spec):
            jsonl_path = str(tmp_path / f"{self.id}.jsonl")
            return SinkFactoryResult(
                jsonl_path=jsonl_path,
                sinks=[StubSink()],
                sink_config=ctx.sink_config,
                metadata={"primary_jsonl": jsonl_path},
            )

    source_registry = SourceRegistry()
    for kind in ("local_dir", "github_zip", "web_pdf_list", "web_page_pdf", "csv_text", "sqlite"):
        source_registry.register(StubSourceFactory(kind))

    sink_registry = SinkRegistry()
    sink_registry.register(StubSinkFactory("default_jsonl_prompt"))
    sink_registry.register(StubSinkFactory("parquet_dataset"))

    plan1 = build_pipeline_plan(
        cfg,
        mutate=False,
        source_registry=source_registry,
        sink_registry=sink_registry,
        load_plugins=False,
    )
    plan2 = build_pipeline_plan(
        cfg,
        mutate=False,
        source_registry=source_registry,
        sink_registry=sink_registry,
        load_plugins=False,
    )

    assert not cfg.sources.sources
    assert not cfg.sinks.sinks
    assert cfg.pipeline.file_extractor is None
    assert cfg.pipeline.bytes_handlers == ()
    assert cfg.http.client is None
    assert cfg.qc.scorer is None

    assert plan1 is not plan2
    assert plan1.runtime is not plan2.runtime

    stats1 = PipelineEngine(plan1).run()
    stats2 = PipelineEngine(plan2).run()

    assert stats1.files == stats2.files == 0
    assert stats1.records == stats2.records == 0


# ------------------------
# 2.3 pipeline smoke test
# ------------------------


def test_run_pipeline_smoke(tmp_path: Path):
    src_root = tmp_path / "input"
    src_root.mkdir()
    (src_root / "example.py").write_text("print('hello')\n", encoding="utf-8")
    (src_root / "README.md").write_text("# Title\n\nSome docs\n", encoding="utf-8")

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    jsonl_path = out_dir / "data.jsonl"
    prompt_path = out_dir / "data.prompt.txt"

    cfg = SievioConfig()
    ctx = RepoContext(
        repo_full_name="local/test",
        repo_url="https://example.com/local",
        license_id="UNKNOWN",
    )
    cfg.sinks.context = ctx

    cfg.sources.specs = (
        SourceSpec(kind="local_dir", options={"root_dir": str(src_root)}),
    )
    cfg.sinks.specs = (
        SinkSpec(
            kind="default_jsonl_prompt",
            options={
                "jsonl_path": str(jsonl_path),
                "prompt_path": str(prompt_path),
            },
        ),
    )

    stats = run_pipeline(config=cfg)

    assert stats["files"] >= 1
    assert stats["records"] >= 1

    assert jsonl_path.exists()
    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    assert lines
    first = json.loads(lines[0])
    assert "text" in first
    assert "meta" in first
