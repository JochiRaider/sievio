# builder.py
# SPDX-License-Identifier: MIT
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Sequence, Tuple, Iterable, Any, Callable

from .config import RepocapsuleConfig, QCMode
from .interfaces import Source, Sink, RepoContext, FileExtractor, Record
from .safe_http import SafeHttpClient
from .log import get_logger
from .factories import make_bytes_handlers, make_qc_scorer
from .convert import DefaultExtractor
from .chunk import ChunkPolicy
from .records import build_run_header_record
from .registries import (
    SourceRegistry,
    SinkRegistry,
    BytesHandlerRegistry,
    QualityScorerRegistry,
    bytes_handler_registry,
    quality_scorer_registry,
    default_source_registry,
    default_sink_registry,
)
from .qc_controller import InlineQCController
from .plugins import load_entrypoint_plugins
from .concurrency import resolve_pipeline_executor_config

log = get_logger(__name__)

# Local copies of the bytes-handler type aliases to avoid circular imports at runtime.
Sniff = Callable[[bytes, str], bool]
BytesHandler = Callable[[bytes, str, Optional[RepoContext], Optional[ChunkPolicy]], Optional[Iterable[Record]]]


@dataclass(slots=True)
class PipelineRuntime:
    """
    Resolved runtime wiring (sources/sinks/clients/hooks) derived from the declarative spec.

    Kept separate from the user-provided spec to avoid cross-run mutation and globals.
    """

    http_client: Optional[SafeHttpClient]
    sources: Sequence[Source]
    sinks: Sequence[Sink]
    file_extractor: FileExtractor
    bytes_handlers: Sequence[Tuple[Sniff, BytesHandler]]
    qc_hook_factory: Callable[[Any], tuple[Callable[[Record], bool], Callable[[Record], Record]]] | None = None
    executor_config: Any | None = None
    fail_fast: bool = False


@dataclass(slots=True)
class PipelinePlan:
    """
    End-to-end plan: immutable spec + resolved runtime dependencies.
    """

    spec: RepocapsuleConfig
    runtime: PipelineRuntime

    # Backwards-compatible accessors for existing callers.
    @property
    def config(self) -> RepocapsuleConfig:
        return self.spec

    @property
    def sources(self) -> Sequence[Source]:
        return self.runtime.sources

    @property
    def sinks(self) -> Sequence[Sink]:
        return self.runtime.sinks

    @property
    def file_extractor(self) -> FileExtractor:
        return self.runtime.file_extractor

    @property
    def bytes_handlers(self) -> Sequence[Tuple[Sniff, BytesHandler]]:
        return self.runtime.bytes_handlers

    @property
    def qc_hook_factory(self) -> Callable[[Any], tuple[Callable[[Record], bool], Callable[[Record], Record]]] | None:
        return self.runtime.qc_hook_factory


def build_pipeline_plan(
    config: RepocapsuleConfig,
    *,
    mutate: bool = False,
    source_registry: Optional[SourceRegistry] = None,
    sink_registry: Optional[SinkRegistry] = None,
    bytes_registry: Optional[BytesHandlerRegistry] = None,
    scorer_registry: Optional[QualityScorerRegistry] = None,
    load_plugins: bool = True,
) -> PipelinePlan:
    """
    Prepare a RepocapsuleConfig for runtime use and return a PipelinePlan.

    `mutate=False` (default) shallow-copies the config before applying wiring so the original remains reusable.
    """

    cfg = config if mutate else replace(config)
    _assert_runtime_free_spec(cfg)
    cfg.logging.apply()
    source_registry = source_registry or default_source_registry()
    sink_registry = sink_registry or default_sink_registry()
    bytes_registry = bytes_registry or bytes_handler_registry
    scorer_registry = scorer_registry or QualityScorerRegistry()
    if load_plugins:
        load_entrypoint_plugins(
            source_registry=source_registry,
            sink_registry=sink_registry,
            bytes_registry=bytes_registry,
            scorer_registry=scorer_registry,
        )
    http_client = _prepare_http(cfg)
    _prepare_sources(cfg, source_registry)
    _prepare_sinks(cfg, sink_registry)
    _prepare_pipeline(cfg, bytes_registry=bytes_registry)
    qc_hook_factory = _prepare_qc(cfg, scorer_registry=scorer_registry)
    _attach_run_header_record(cfg)
    cfg.validate()

    sources = tuple(cfg.sources.sources)
    sinks = tuple(cfg.sinks.sinks)
    bytes_handlers = cfg.pipeline.bytes_handlers
    file_extractor = cfg.pipeline.file_extractor
    temp_runtime = PipelineRuntime(
        http_client=http_client,
        sources=sources,
        sinks=sinks,
        file_extractor=file_extractor,
        bytes_handlers=bytes_handlers,
        qc_hook_factory=qc_hook_factory,
    )
    exec_cfg, fail_fast = resolve_pipeline_executor_config(cfg, runtime=temp_runtime)
    runtime = replace(temp_runtime, executor_config=exec_cfg, fail_fast=fail_fast)
    return PipelinePlan(spec=cfg, runtime=runtime)


def _prepare_http(cfg: RepocapsuleConfig) -> Optional[SafeHttpClient]:
    """
    Normalize HTTP client wiring (mirror of prior RepocapsuleConfig._prepare_http).
    """
    client = cfg.http.build_client()
    cfg.http.client = client
    # Avoid mutating global state; caller wires the client explicitly.
    return client


def _assert_runtime_free_spec(cfg: RepocapsuleConfig) -> None:
    """
    Enforce that declarative specs do not carry runtime objects.
    """
    if getattr(cfg.http, "client", None) is not None:
        raise ValueError("http.client must be unset in declarative specs; provide HTTP client via runtime wiring.")
    if getattr(cfg.qc, "scorer", None) is not None:
        raise ValueError("qc.scorer must be unset in declarative specs; use QC registry/plugins instead.")
    if getattr(cfg.pipeline, "file_extractor", None) is not None:
        raise ValueError("pipeline.file_extractor must be unset in declarative specs; register extractors instead.")
    if getattr(cfg.pipeline, "bytes_handlers", None):
        raise ValueError("pipeline.bytes_handlers must be empty in declarative specs; use registries/plugins instead.")


def _prepare_sources(cfg: RepocapsuleConfig, registry: SourceRegistry) -> None:
    """Populate cfg.sources.sources from declarative specs when not already provided."""
    if cfg.sources.sources:
        raise ValueError("sources.sources must be empty in specs; provide declarative specs instead.")
    if not cfg.sources.specs:
        return
    cfg.sources.sources = tuple(registry.build_all(cfg))


def _prepare_sinks(cfg: RepocapsuleConfig, registry: SinkRegistry) -> None:
    if cfg.sinks.sinks:
        raise ValueError("sinks.sinks must be empty in specs; provide declarative specs instead.")
    if cfg.sinks.specs:
        sinks, extra_meta = registry.build_all(cfg)
        cfg.sinks.sinks = tuple(sinks)
        cfg.metadata = cfg.metadata.merged(extra_meta)
    sinks = cfg.sinks
    meta = cfg.metadata
    primary = sinks.primary_jsonl_name or meta.primary_jsonl
    if primary:
        primary_str = str(primary)
        sinks.primary_jsonl_name = primary_str
        meta.primary_jsonl = primary_str

        output_dir = sinks.output_dir
        needs_output_dir = output_dir is None or str(output_dir) in {"", "."}
        if needs_output_dir:
            try:
                parent = Path(primary_str).parent
            except Exception:
                parent = None
            if parent:
                sinks.output_dir = parent


def _prepare_pipeline(cfg: RepocapsuleConfig, *, bytes_registry: BytesHandlerRegistry) -> None:
    if not cfg.pipeline.bytes_handlers:
        cfg.pipeline.bytes_handlers = tuple(make_bytes_handlers(bytes_registry))
    if cfg.pipeline.file_extractor is None:
        cfg.pipeline.file_extractor = DefaultExtractor()


def _prepare_qc(
    cfg: RepocapsuleConfig,
    *,
    scorer_registry: QualityScorerRegistry,
) -> Callable[[Any], tuple[Callable[[Record], bool], Callable[[Record], Record]]] | None:
    qc_cfg = cfg.qc
    mode = qc_cfg.normalize_mode()
    qc_hook_factory: Callable[[Any], tuple[Callable[[Record], bool], Callable[[Record], Record]]] | None = None
    if not qc_cfg.enabled or mode == QCMode.OFF:
        qc_cfg.enabled = False
        qc_cfg.mode = QCMode.OFF
        return None

    if mode in {QCMode.INLINE, QCMode.ADVISORY}:
        if qc_cfg.scorer is None:
            scorer = make_qc_scorer(qc_cfg, scorer_registry=scorer_registry)
            if scorer is None:
                raise RuntimeError(
                    "Inline/advisory QC requested but QC extras are not installed; disable qc.enabled or install QC dependencies."
                )
            qc_cfg.scorer = scorer
        enforce_drops = mode == QCMode.INLINE

        def qc_hook_factory(stats: Any) -> tuple[Callable[[Record], bool], Callable[[Record], Record]]:
            controller = InlineQCController(
                config=qc_cfg,
                stats=stats,
                scorer=qc_cfg.scorer,  # type: ignore[arg-type]
                logger=log,
                enforce_drops=enforce_drops,
            )
            return controller.accept, controller.on_record

        return qc_hook_factory

    if mode == QCMode.POST and qc_cfg.scorer is None:
        scorer = make_qc_scorer(qc_cfg, new_instance=False, scorer_registry=scorer_registry)
        if scorer is not None:
            qc_cfg.scorer = scorer
        else:
            log.warning("QC POST mode enabled but QC extras are not installed; skipping QC for this run.")
            qc_cfg.enabled = False
            qc_cfg.mode = QCMode.OFF
            qc_cfg.scorer = None
    return qc_hook_factory


def _attach_run_header_record(cfg: RepocapsuleConfig) -> None:
    header = build_run_header_record(cfg)
    for sink in getattr(cfg.sinks, "sinks", ()):
        setter = getattr(sink, "set_header_record", None)
        if callable(setter):
            setter(header)


def build_engine(config: RepocapsuleConfig, *, mutate: bool = False):
    from .pipeline import PipelineEngine

    plan = build_pipeline_plan(config, mutate=mutate)
    return PipelineEngine(plan)
