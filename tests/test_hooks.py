import types
from collections.abc import Iterable, Mapping
from typing import Any

import pytest

from sievio.core.config import QCConfig, SievioConfig
from sievio.core.dataset_card import DatasetCardHook
from sievio.core.hooks import RunSummaryHook
from sievio.core.interfaces import RunArtifacts, RunContext
from sievio.core.pipeline import PipelineStats
from sievio.core.qc_controller import InlineQCHook
from sievio.core.qc_post import PostQCHook
from sievio.core.records import RunSummaryMeta


class DummySink:
    def __init__(self) -> None:
        self.finalize_calls: list[list[Mapping[str, Any]]] = []

    def finalize(self, records: Iterable[Mapping[str, Any]]) -> None:
        self.finalize_calls.append(list(records))


class InlineStub(InlineQCHook):
    def __init__(self) -> None:
        super().__init__(qc_cfg=QCConfig(enabled=False), scorer=None)
        self.artifacts_seen: list[RunArtifacts] = []

    def on_artifacts(self, artifacts: RunArtifacts, ctx: RunContext) -> None:
        self.artifacts_seen.append(artifacts)


class DatasetCardHookStub(DatasetCardHook):
    def __init__(self) -> None:
        super().__init__(enabled=True)
        self.artifacts_seen: list[RunArtifacts] = []

    def on_artifacts(self, artifacts: RunArtifacts, ctx: RunContext) -> None:
        self.artifacts_seen.append(artifacts)
        return None


def _make_runtime(**kwargs: Any) -> Any:
    defaults = {"sinks": (), "lifecycle_hooks": ()}
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)


def test_run_summary_hook_dispatches_artifacts() -> None:
    cfg = SievioConfig()
    stats = PipelineStats()
    sink = DummySink()
    inline_hook = InlineStub()
    dataset_hook = DatasetCardHookStub()
    summary_hook = RunSummaryHook()

    runtime = _make_runtime(
        sinks=[sink],
        lifecycle_hooks=[inline_hook, summary_hook, dataset_hook],
    )
    ctx = RunContext(cfg=cfg, stats=stats, runtime=runtime)

    summary_hook.on_run_end(ctx)

    assert len(sink.finalize_calls) == 1
    assert len(sink.finalize_calls[0]) == 1
    summary_record = sink.finalize_calls[0][0]
    assert summary_record.get("meta", {}).get("kind") == "run_summary"
    assert len(dataset_hook.artifacts_seen) == 1


def test_dataset_card_hook_uses_artifacts(monkeypatch: pytest.MonkeyPatch) -> None:
    hook = DatasetCardHook(enabled=True)
    cfg = SievioConfig()
    stats = PipelineStats(records=3, primary_jsonl_path="data.jsonl")
    summary_view = stats.to_summary_view()
    artifacts = RunArtifacts(summary_record={"meta": RunSummaryMeta().to_dict()}, summary_view=summary_view)
    runtime = _make_runtime()
    ctx = RunContext(cfg=cfg, stats=stats, runtime=runtime)

    called: dict[str, Any] = {}

    def _capture(cfg_arg: SievioConfig, view_arg: Any) -> None:
        called["cfg"] = cfg_arg
        called["view"] = view_arg

    monkeypatch.setattr("sievio.core.dataset_card.write_card_fragment_for_run", _capture)

    hook.on_artifacts(artifacts, ctx)

    assert called["cfg"] is cfg
    assert called["view"] is artifacts.summary_view


def test_summary_view_primary_jsonl_path_matches_config() -> None:
    cfg = SievioConfig()
    cfg.metadata.primary_jsonl = "meta.jsonl"
    stats = PipelineStats(primary_jsonl_path=cfg.metadata.primary_jsonl)
    view_from_metadata = stats.to_summary_view()
    assert view_from_metadata.primary_jsonl_path == cfg.metadata.primary_jsonl

    cfg.sinks.primary_jsonl_name = "sinks.jsonl"
    stats.primary_jsonl_path = cfg.sinks.primary_jsonl_name
    view_from_sinks = stats.to_summary_view()
    assert view_from_sinks.primary_jsonl_path == cfg.sinks.primary_jsonl_name


def test_all_hooks_expose_lifecycle_methods() -> None:
    cfg = SievioConfig()
    stats = PipelineStats()
    dummy_scorer = object()
    hooks = [
        RunSummaryHook(),
        DatasetCardHook(enabled=True),
        PostQCHook(cfg.qc, dummy_scorer),
        InlineStub(),
    ]
    ctx = RunContext(cfg=cfg, stats=stats, runtime=_make_runtime())

    for hook in hooks:
        hook.on_run_start(ctx)
        hook.on_record({})
        hook.on_run_end(ctx)
        on_artifacts = getattr(hook, "on_artifacts", None)
        if callable(on_artifacts):
            on_artifacts(
                RunArtifacts(summary_record={"meta": RunSummaryMeta().to_dict()}, summary_view=stats.to_summary_view()),
                ctx,
            )
