import logging
from pathlib import Path

import pytest

from sievio.core.builder import PipelineOverrides, build_pipeline_plan
from sievio.core.config import SievioConfig, SinkSpec, SourceSpec
from sievio.core.interfaces import RepoContext
from sievio.core.pipeline import ErrorRateExceeded, PipelineEngine


def _make_engine(
    tmp_path: Path,
    *,
    extractor,
    file_names: list[str],
    max_error_rate: float | None = None,
) -> PipelineEngine:
    cfg = SievioConfig()
    cfg.pipeline.max_workers = 1
    cfg.pipeline.fail_fast = False
    cfg.pipeline.max_error_rate = max_error_rate

    ctx = RepoContext(repo_full_name="local/test", repo_url="https://example.com/local", license_id="UNKNOWN")
    cfg.sinks.context = ctx

    src_root = tmp_path / "input"
    src_root.mkdir()
    for name in file_names:
        (src_root / name).write_text("content", encoding="utf-8")

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

    overrides = PipelineOverrides(file_extractor=extractor)
    plan = build_pipeline_plan(cfg, mutate=False, overrides=overrides)
    return PipelineEngine(plan)


class AlwaysFailExtractor:
    def extract(self, item, config, context):  # pragma: no cover - exercised via engine
        raise RuntimeError("boom")


class SucceedThenFailExtractor:
    def __init__(self) -> None:
        self.calls = 0

    def extract(self, item, config, context):
        self.calls += 1
        if self.calls == 1:
            return [{"text": "ok", "meta": {}}]
        raise RuntimeError("expected failure")


def test_error_rate_abort_serial(tmp_path: Path):
    engine = _make_engine(
        tmp_path,
        extractor=AlwaysFailExtractor(),
        file_names=["bad.txt"],
        max_error_rate=0.5,
    )

    with pytest.raises(ErrorRateExceeded):
        engine.run()

    assert engine.stats.attempted_files == 1
    assert engine.stats.source_errors == 1
    assert engine.stats.records == 0


def test_error_rate_allows_progress_under_threshold(tmp_path: Path):
    engine = _make_engine(
        tmp_path,
        extractor=SucceedThenFailExtractor(),
        file_names=["good.txt", "bad.txt"],
        max_error_rate=0.6,
    )

    stats = engine.run()

    assert stats.attempted_files == 2
    assert stats.source_errors == 1
    assert stats.records == 1
    assert stats.files == 1


def test_error_summary_logged_on_completion(tmp_path: Path, caplog):
    engine = _make_engine(
        tmp_path,
        extractor=SucceedThenFailExtractor(),
        file_names=["good.txt", "bad.txt"],
        max_error_rate=None,
    )

    logger = logging.getLogger("sievio.core.pipeline")
    previous_level = logger.level
    logger.addHandler(caplog.handler)
    logger.setLevel(logging.WARNING)
    try:
        stats = engine.run()
    finally:
        logger.setLevel(previous_level)
        logger.removeHandler(caplog.handler)

    log_text = caplog.text
    assert "Ingestion summary:" in log_text
    assert "attempted=2" in log_text
    assert "source_errors=1" in log_text
    assert stats.records == 1
