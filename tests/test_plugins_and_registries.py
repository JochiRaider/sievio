import importlib
from types import SimpleNamespace

import pytest

from sievio.core.builder import build_pipeline_plan
from sievio.core.config import SievioConfig, SinkSpec, SourceSpec
from sievio.core.factories import SinkFactoryResult
from sievio.core.interfaces import SinkFactoryContext, SourceFactoryContext
from sievio.core.plugins import load_entrypoint_plugins
from sievio.core.registries import (
    BytesHandlerRegistry,
    QualityScorerRegistry,
    RegistryBundle,
    SafetyScorerRegistry,
    SinkRegistry,
    SourceRegistry,
    bytes_handler_registry,
    default_registries,
    default_sink_registry,
    default_source_registry,
    quality_scorer_registry,
    safety_scorer_registry,
)


class FakeEntryPoint:
    def __init__(self, name, value=None, error=None):
        self.name = name
        self._value = value
        self._error = error

    def load(self):
        if self._error is not None:
            raise self._error
        return self._value


def test_load_entrypoint_plugins_logs_failures(monkeypatch, caplog):
    good_called = SimpleNamespace(value=False)

    def good_plugin(source_reg, sink_reg, bytes_reg, scorer_reg):
        good_called.value = True

    eps = [
        FakeEntryPoint("good_plugin", value=good_plugin),
        FakeEntryPoint("bad_plugin", error=ImportError("boom")),
    ]

    class DummyEntryPoints(list):
        def select(self, group=None):
            return self if group == "sievio.plugins" else []

    def fake_entry_points():
        return DummyEntryPoints(eps)

    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)
    messages: list[str] = []

    def fake_warning(msg, *args, **kwargs):
        text = msg % args if args else str(msg)
        messages.append(text)

    monkeypatch.setattr("sievio.core.plugins.log.warning", fake_warning)

    with caplog.at_level("WARNING", logger="sievio.core.plugins"):
        load_entrypoint_plugins(
            source_registry=SourceRegistry(),
            sink_registry=SinkRegistry(),
            bytes_registry=BytesHandlerRegistry(),
            scorer_registry=QualityScorerRegistry(),
            safety_scorer_registry=SafetyScorerRegistry(),
        )

    assert good_called.value is True
    assert any("bad_plugin" in msg for msg in messages)


def test_source_registry_unknown_kind_raises():
    registry = default_source_registry()
    cfg = SievioConfig()
    cfg.sources.specs = (SourceSpec(kind="does_not_exist", options={}),)
    ctx = SourceFactoryContext(
        repo_context=None,
        http_client=None,
        http_config=cfg.http,
        source_defaults=cfg.sources.defaults,
    )
    with pytest.raises(ValueError):
        registry.build_all(ctx, cfg.sources.specs)


def test_sink_registry_unknown_kind_raises():
    registry = default_sink_registry()
    cfg = SievioConfig()
    cfg.sinks.specs = (SinkSpec(kind="does_not_exist", options={}),)
    ctx = SinkFactoryContext(repo_context=None, sink_config=cfg.sinks, sink_defaults=cfg.sinks.defaults)
    with pytest.raises(ValueError):
        registry.build_all(ctx, cfg.sinks.specs)


def test_default_source_registry_has_expected_ids():
    registry = default_source_registry()
    kinds = set(registry._factories.keys())
    for expected in {"local_dir", "github_zip", "web_pdf_list", "web_page_pdf", "csv_text", "sqlite"}:
        assert expected in kinds


def test_default_sink_registry_has_expected_ids():
    registry = default_sink_registry()
    kinds = set(registry._factories.keys())
    assert "default_jsonl_prompt" in kinds
    assert "parquet_dataset" in kinds


def test_default_registries_populates_defaults():
    bundle = default_registries(load_plugins=False)

    assert {"local_dir", "github_zip", "web_pdf_list", "web_page_pdf", "csv_text", "sqlite"}.issubset(
        bundle.sources._factories
    )
    assert {"default_jsonl_prompt", "parquet_dataset"}.issubset(bundle.sinks._factories)
    assert bundle.bytes is bytes_handler_registry
    assert bundle.scorers is quality_scorer_registry
    assert bundle.safety_scorers is safety_scorer_registry


def test_default_registries_load_plugins_calls_loader(monkeypatch):
    calls: list[dict] = []

    def fake_loader(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("sievio.core.plugins.load_entrypoint_plugins", fake_loader)

    default_registries(load_plugins=False)
    assert calls == []

    bundle = default_registries(load_plugins=True)
    assert len(calls) == 1
    assert calls[0]["source_registry"] is bundle.sources
    assert calls[0]["sink_registry"] is bundle.sinks
    assert calls[0]["bytes_registry"] is bundle.bytes
    assert calls[0]["scorer_registry"] is bundle.scorers
    assert calls[0]["safety_scorer_registry"] is bundle.safety_scorers


class DummySource:
    def __init__(self, label: str):
        self.label = label

    def iter_files(self):
        return ()


class DummySourceFactory:
    id = "dummy_source"

    def __init__(self, label: str):
        self.label = label

    def build(self, ctx, spec):
        return [DummySource(spec.options.get("label", self.label))]


class DummySink:
    def __init__(self, name: str):
        self.name = name

    def write(self, record):
        return None

    def close(self):
        return None


class DummySinkFactory:
    id = "dummy_sink"

    def __init__(self, name: str):
        self.name = name

    def build(self, ctx, spec):
        jsonl_path = spec.options.get("jsonl_path", "dummy.jsonl")
        return SinkFactoryResult(
            jsonl_path=jsonl_path,
            sinks=[DummySink(self.name)],
            sink_config=ctx.sink_config,
            metadata={"primary_jsonl": jsonl_path},
        )


def test_build_pipeline_plan_uses_custom_bundle(monkeypatch, tmp_path):
    source_reg = SourceRegistry()
    source_reg.register(DummySourceFactory(label="bundle"))
    sink_reg = SinkRegistry()
    sink_reg.register(DummySinkFactory(name="bundle"))
    bundle = RegistryBundle(
        sources=source_reg,
        sinks=sink_reg,
        bytes=BytesHandlerRegistry(),
        scorers=QualityScorerRegistry(),
        safety_scorers=SafetyScorerRegistry(),
    )

    def fail_default_registries(*args, **kwargs):
        raise AssertionError("default_registries should not be called when registries are provided")

    monkeypatch.setattr("sievio.core.builder.default_registries", fail_default_registries)

    cfg = SievioConfig()
    cfg.sources.specs = (SourceSpec(kind="dummy_source", options={"label": "from_bundle"}),)
    cfg.sinks.specs = (SinkSpec(kind="dummy_sink", options={"jsonl_path": str(tmp_path / "data.jsonl")}),)

    plan = build_pipeline_plan(cfg, registries=bundle, load_plugins=False)
    assert isinstance(plan.runtime.sources[0], DummySource)
    assert plan.runtime.sources[0].label == "from_bundle"
    assert isinstance(plan.runtime.sinks[0], DummySink)
    assert plan.runtime.sinks[0].name == "bundle"


def test_build_pipeline_plan_per_registry_override_wins_over_bundle(monkeypatch, tmp_path):
    bundle_source_reg = SourceRegistry()
    bundle_source_reg.register(DummySourceFactory(label="bundle"))
    override_source_reg = SourceRegistry()
    override_source_reg.register(DummySourceFactory(label="override"))
    sink_reg = SinkRegistry()
    sink_reg.register(DummySinkFactory(name="bundle"))
    bundle = RegistryBundle(
        sources=bundle_source_reg,
        sinks=sink_reg,
        bytes=BytesHandlerRegistry(),
        scorers=QualityScorerRegistry(),
        safety_scorers=SafetyScorerRegistry(),
    )

    def fail_default_registries(*args, **kwargs):
        raise AssertionError("default_registries should not be called when registries are provided")

    monkeypatch.setattr("sievio.core.builder.default_registries", fail_default_registries)

    cfg = SievioConfig()
    cfg.sources.specs = (SourceSpec(kind="dummy_source", options={}),)
    cfg.sinks.specs = (SinkSpec(kind="dummy_sink", options={"jsonl_path": str(tmp_path / "custom.jsonl")}),)

    plan = build_pipeline_plan(cfg, registries=bundle, source_registry=override_source_reg, load_plugins=False)
    assert isinstance(plan.runtime.sources[0], DummySource)
    assert plan.runtime.sources[0].label == "override"
    assert isinstance(plan.runtime.sinks[0], DummySink)
    assert plan.runtime.sinks[0].name == "bundle"
