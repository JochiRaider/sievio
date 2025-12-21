from dataclasses import dataclass

import pytest

from sievio.core.builder import build_pipeline_plan
from sievio.core.config import SievioConfig, SinkSpec, SourceSpec
from sievio.core.factories import SinkFactoryResult
from sievio.core.interfaces import FileItem, SinkFactoryContext, SourceFactoryContext
from sievio.core.pipeline import PipelineEngine
from sievio.core.registries import SinkRegistry, SourceRegistry


class DummySink:
    def __init__(self, name: str = "dummy") -> None:
        self.name = name

    def open(self, context=None) -> None:
        return None

    def write(self, record) -> None:
        return None

    def close(self) -> None:
        return None


@dataclass(slots=True)
class DummySourceOptions:
    path: str = "data.txt"
    prefix: str = "hi"
    payload: str = "world"


def test_source_registry_register_callable_merges_defaults():
    source_reg = SourceRegistry()

    def make_source(options, *, ctx, spec):
        data = f"{options['prefix']}:{options['payload']}".encode()
        return [FileItem(path=options["path"], data=data)]

    source_reg.register_callable("callable_source", make_source, options_model=DummySourceOptions)

    cfg = SievioConfig()
    ctx = SourceFactoryContext(
        repo_context=None,
        http_client=None,
        http_config=cfg.http,
        source_defaults={"callable_source": {"prefix": "hello", "payload": "sievio"}},
    )
    spec = SourceSpec(kind="callable_source", options={"path": "custom.txt"})

    sources = source_reg.build_all(ctx, [spec])
    items = list(sources[0].iter_files())

    assert items[0].path == "custom.txt"
    assert items[0].data == b"hello:sievio"


def test_source_registry_register_callable_unknown_option_raises():
    source_reg = SourceRegistry()

    def make_source(options, *, ctx, spec):
        return []

    source_reg.register_callable("callable_source", make_source, options_model=DummySourceOptions)

    cfg = SievioConfig()
    ctx = SourceFactoryContext(
        repo_context=None,
        http_client=None,
        http_config=cfg.http,
        source_defaults={},
    )
    spec = SourceSpec(kind="callable_source", options={"unknown": "nope"})

    with pytest.raises(ValueError):
        source_reg.build_all(ctx, [spec])


@dataclass(slots=True)
class DummySinkOptions:
    name: str = "default"


def test_sink_registry_register_callable_builds_sinks():
    sink_reg = SinkRegistry()

    def make_sink(options, *, ctx, spec):
        return DummySink(options["name"])

    sink_reg.register_callable("dummy_sink", make_sink, options_model=DummySinkOptions)

    cfg = SievioConfig()
    cfg.sinks.defaults["dummy_sink"] = {"name": "from_defaults"}
    ctx = SinkFactoryContext(repo_context=None, sink_config=cfg.sinks, sink_defaults=cfg.sinks.defaults)
    spec = SinkSpec(kind="dummy_sink", options={})

    sinks, metadata, final_ctx = sink_reg.build_all(ctx, [spec])

    assert len(sinks) == 1
    assert sinks[0].name == "from_defaults"
    assert metadata == {}
    assert final_ctx.sink_config is cfg.sinks


def test_callable_source_cleanup_runs_via_pipeline(tmp_path):
    closed = {"value": False}

    class CloseAwareIter:
        def __init__(self, closed_flag):
            self._closed_flag = closed_flag
            self._done = False

        def __iter__(self):
            return self

        def __next__(self):
            if self._done:
                raise StopIteration
            self._done = True
            return FileItem(path="data.txt", data=b"hello")

        def close(self) -> None:
            self._closed_flag["value"] = True

    def make_source(options, *, ctx, spec):
        return CloseAwareIter(closed)

    def make_sink(options, *, ctx, spec):
        jsonl_path = tmp_path / "out.jsonl"
        return SinkFactoryResult(
            jsonl_path=str(jsonl_path),
            sinks=[DummySink()],
            sink_config=ctx.sink_config,
            metadata={"primary_jsonl": str(jsonl_path)},
        )

    source_reg = SourceRegistry()
    source_reg.register_callable("callable_source", make_source)
    sink_reg = SinkRegistry()
    sink_reg.register_callable("dummy_sink", make_sink)

    cfg = SievioConfig()
    cfg.sources.specs = (SourceSpec(kind="callable_source", options={}),)
    cfg.sinks.specs = (SinkSpec(kind="dummy_sink", options={}),)

    plan = build_pipeline_plan(cfg, source_registry=source_reg, sink_registry=sink_reg, load_plugins=False)
    stats = PipelineEngine(plan).run()

    assert stats.records == 1
    assert closed["value"] is True
