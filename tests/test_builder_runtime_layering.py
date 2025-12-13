from pathlib import Path

from sievio.core.builder import build_pipeline_plan
from sievio.core.config import SievioConfig, SinkSpec, SourceSpec
from sievio.core.interfaces import RepoContext


def _basic_cfg(tmp_path: Path) -> SievioConfig:
    cfg = SievioConfig()
    cfg.sinks.context = RepoContext(repo_full_name="local/test", repo_url="https://example.com/local", license_id="UNKNOWN")

    src_root = tmp_path / "input"
    src_root.mkdir()
    (src_root / "file.txt").write_text("hello", encoding="utf-8")

    cfg.sources.specs = (SourceSpec(kind="local_dir", options={"root_dir": str(src_root)}),)

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


def test_plan_spec_is_runtime_free(tmp_path: Path):
    cfg = _basic_cfg(tmp_path)

    plan = build_pipeline_plan(cfg, mutate=False, load_plugins=False)

    assert plan.spec.http.client is None
    assert not plan.spec.sources.sources
    assert not plan.spec.sinks.sinks
    assert plan.spec.pipeline.file_extractor is None
    assert not plan.spec.pipeline.bytes_handlers
    assert plan.spec.qc.scorer is None

    assert plan.runtime.http_client is not None
    assert plan.runtime.sources
    assert plan.runtime.sinks
    assert plan.runtime.file_extractor is not None
    assert plan.runtime.bytes_handlers


def test_plan_reuse_same_spec_produces_clean_specs(tmp_path: Path):
    cfg = _basic_cfg(tmp_path)

    plan1 = build_pipeline_plan(cfg, mutate=False, load_plugins=False)
    plan2 = build_pipeline_plan(cfg, mutate=False, load_plugins=False)

    assert plan1.spec is not cfg
    assert plan2.spec is not cfg
    assert not cfg.sources.sources
    assert not cfg.sinks.sinks
    assert cfg.pipeline.file_extractor is None
    assert not cfg.pipeline.bytes_handlers
    assert cfg.qc.scorer is None

    assert plan1.runtime is not plan2.runtime
    assert plan1.runtime.sources and plan2.runtime.sources
