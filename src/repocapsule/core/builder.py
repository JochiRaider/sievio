# builder.py
# SPDX-License-Identifier: MIT
"""Builder helpers for constructing PipelinePlan instances.

This module turns declarative RepocapsuleConfig objects into immutable
PipelinePlan instances that separate pure configuration from runtime
wiring such as sources, sinks, HTTP clients, bytes handlers, and QC
hooks. The primary entry point is build_pipeline_plan().
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Sequence, Tuple, Iterable, Any, Callable

from .config import RepocapsuleConfig, QCMode, SinkConfig, RunMetadata, QCConfig
from .interfaces import (
    Source,
    Sink,
    RepoContext,
    FileExtractor,
    Record,
    SourceFactoryContext,
    SinkFactoryContext,
    QualityScorer,
)
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
class PipelineOverrides:
    """Define runtime-only overrides for pipeline wiring.


    Advanced callers can pass an instance to build_pipeline_plan() to
    override objects that would normally be resolved from registries or
    defaults.

    Attributes:
        http_client (SafeHttpClient | None): Shared HTTP client to use
            for remote-capable sources instead of building one from
            ``cfg.http``.
        qc_scorer (QualityScorer | None): Quality scorer to use for
            inline or post-hoc QC instead of resolving via the scorer
            registry.
        file_extractor (FileExtractor | None): File extractor to use in
            place of DefaultExtractor.
        bytes_handlers (Sequence[tuple[Sniff, BytesHandler]] | None):
            Bytes handlers to use instead of handlers built from the
            registry. When provided, these completely replace
            registry-built handlers.
    """

    http_client: SafeHttpClient | None = None
    qc_scorer: QualityScorer | None = None
    file_extractor: FileExtractor | None = None
    bytes_handlers: Sequence[Tuple[Sniff, BytesHandler]] | None = None


@dataclass(slots=True)
class PipelineRuntime:
    """Hold resolved runtime wiring and state for a pipeline run.

    This dataclass stores live objects (sources, sinks, clients, hooks)
    derived from a declarative RepocapsuleConfig. It is kept separate
    from the spec to avoid cross-run mutation and configuration drift.

    Attributes:
        http_client (SafeHttpClient | None): Shared HTTP client used by
            remote-capable sources, or None when no HTTP access is
            needed.
        sources (Sequence[Source]): Concrete sources that yield FileItem
            objects during the run.
        sinks (Sequence[Sink]): Concrete sinks that consume records.
        file_extractor (FileExtractor): Extractor used to decode and
            chunk files that are not handled by bytes handlers.
        bytes_handlers (Sequence[tuple[Sniff, BytesHandler]]): Ordered
            (sniff, handler) pairs used for binary formats.
        qc_hook_factory (Callable[[Any], tuple[Callable[[Record], bool],
            Callable[[Record], Record]]] | None): Factory that builds
            inline QC hooks given a stats object, or None when QC is
            disabled or uses post-hoc mode.
        executor_config (Any | None): Executor configuration determined
            by resolve_pipeline_executor_config().
        fail_fast (bool): Whether worker failures should abort the run.
        qc_scorer_for_csv (QualityScorer | None): Scorer used when
            emitting QC CSV reports.
        post_qc_scorer (QualityScorer | None): Scorer used for post-hoc
            QC runs when mode is QCMode.POST.
    """

    http_client: Optional[SafeHttpClient]
    sources: Sequence[Source]
    sinks: Sequence[Sink]
    file_extractor: FileExtractor
    bytes_handlers: Sequence[Tuple[Sniff, BytesHandler]]
    qc_hook_factory: Callable[[Any], tuple[Callable[[Record], bool], Callable[[Record], Record]]] | None = None
    executor_config: Any | None = None
    fail_fast: bool = False
    qc_scorer_for_csv: QualityScorer | None = None
    post_qc_scorer: QualityScorer | None = None


@dataclass(slots=True)
class SinksPreparationResult:
    """Bundle sink instances with normalized sink config and metadata.

    Attributes:
        sinks (tuple[Sink, ...]): Concrete sink instances built from the
            declarative specs.
        sinks_cfg (SinkConfig): Normalized sink configuration with
            runtime-only fields stripped out.
        metadata (RunMetadata): Updated run metadata derived from sink
            options, such as the primary JSONL path.
    """
    sinks: tuple[Sink, ...]
    sinks_cfg: SinkConfig
    metadata: RunMetadata


@dataclass(slots=True)
class PipelinePreparationResult:
    """Bundle bytes handlers and file extractor for the pipeline.

    Attributes:
        bytes_handlers (tuple[tuple[Sniff, BytesHandler], ...]): Ordered
            (sniff, handler) pairs used to handle binary formats.
        file_extractor (FileExtractor): Extractor used to process files
            that are not handled by bytes handlers.
    """
    bytes_handlers: tuple[Tuple[Sniff, BytesHandler], ...]
    file_extractor: FileExtractor


@dataclass(slots=True)
class QCPreparationResult:
    """Bundle normalized QC configuration and resolved scorers.

    Attributes:
        qc_cfg (QCConfig): Normalized QC configuration with runtime-only
            fields cleared.
        qc_hook_factory (Callable[[Any], tuple[Callable[[Record], bool],
            Callable[[Record], Record]]] | None): Factory that builds
            inline QC hooks given a stats object, or None when QC is
            disabled or running in post mode.
        scorer_for_csv (QualityScorer | None): Scorer to use when
            emitting QC CSV reports.
        post_qc_scorer (QualityScorer | None): Scorer to use for
            post-hoc QC runs when mode is QCMode.POST.
    """
    qc_cfg: QCConfig
    qc_hook_factory: Callable[[Any], tuple[Callable[[Record], bool], Callable[[Record], Record]]] | None
    scorer_for_csv: QualityScorer | None
    post_qc_scorer: QualityScorer | None


@dataclass(slots=True)
class PipelinePlan:
    """Represent an immutable plan derived from a RepocapsuleConfig.

    A PipelinePlan splits the declarative configuration (:attr:`spec`)
    from live runtime wiring (:attr:`runtime`). The spec is validated
    and stripped of runtime objects so that the plan can be reused
    across runs that share the same configuration.

    Attributes:
        spec (RepocapsuleConfig): Effective, validated configuration
            with runtime-only fields cleared.
        runtime (PipelineRuntime): Resolved runtime wiring, including
            sources, sinks, HTTP client, bytes handlers, QC hooks, and
            executor settings.
    """

    spec: RepocapsuleConfig
    runtime: PipelineRuntime


def build_pipeline_plan(
    config: RepocapsuleConfig,
    *,
    mutate: bool = False,
    overrides: PipelineOverrides | None = None,
    source_registry: Optional[SourceRegistry] = None,
    sink_registry: Optional[SinkRegistry] = None,
    bytes_registry: Optional[BytesHandlerRegistry] = None,
    scorer_registry: Optional[QualityScorerRegistry] = None,
    load_plugins: bool = True,
) -> PipelinePlan:
    """Build a PipelinePlan from a declarative RepocapsuleConfig.

    The builder validates the configuration, loads plugins, constructs
    sources and sinks, attaches run-header records, resolves QC and
    executor wiring, and separates runtime objects into a
    PipelineRuntime.

    Args:
        config (RepocapsuleConfig): Declarative configuration for the
            run.
        mutate (bool): Whether to mutate ``config`` in place. When False
            (default), the config is deep-copied so the original remains
            reusable.
        overrides (PipelineOverrides | None): Runtime-only overrides for
            HTTP client, QC scorer, file extractor, or bytes handlers.
        source_registry (SourceRegistry | None): Registry used to build
            sources. Defaults to :func:`default_source_registry`.
        sink_registry (SinkRegistry | None): Registry used to build
            sinks.
        bytes_registry (BytesHandlerRegistry | None): Registry used to
            build bytes handlers.
        scorer_registry (QualityScorerRegistry | None): Registry used to
            build QC scorers. When None, a new registry instance is
            used.
        load_plugins (bool): Whether to load entry-point plugins before
            building registries.

    Returns:
        PipelinePlan: Immutable plan containing the validated spec and
            resolved runtime wiring.

    Raises:
        ValueError: If the configuration contains baked-in runtime
            objects such as sources, sinks, HTTP clients, scorers, or
            extractors.
        RuntimeError: If inline or advisory QC is requested but QC
            extras are not installed.
    """

    cfg = config if mutate else deepcopy(config)
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
    http_client = _prepare_http(cfg, overrides=overrides)
    source_ctx = SourceFactoryContext(
        repo_context=cfg.sinks.context,
        http_client=http_client,
        http_config=cfg.http,
        source_defaults=cfg.sources.defaults,
    )
    sink_ctx = SinkFactoryContext(repo_context=cfg.sinks.context, sink_config=cfg.sinks)
    sources = _prepare_sources(cfg, source_registry, ctx=source_ctx)
    sinks_res = _prepare_sinks(cfg, sink_registry, ctx=sink_ctx)
    pipe_res = _prepare_pipeline(cfg, bytes_registry=bytes_registry, overrides=overrides)
    qc_res = _prepare_qc(cfg, scorer_registry=scorer_registry, overrides=overrides)
    cfg.sinks = sinks_res.sinks_cfg
    cfg.metadata = sinks_res.metadata
    cfg.qc = qc_res.qc_cfg
    _attach_run_header_record(cfg, sinks_res.sinks)
    cfg.validate()

    bytes_handlers = pipe_res.bytes_handlers
    file_extractor = pipe_res.file_extractor
    temp_runtime = PipelineRuntime(
        http_client=http_client,
        sources=sources,
        sinks=sinks_res.sinks,
        file_extractor=file_extractor,
        bytes_handlers=bytes_handlers,
        qc_hook_factory=qc_res.qc_hook_factory,
        qc_scorer_for_csv=qc_res.scorer_for_csv,
        post_qc_scorer=qc_res.post_qc_scorer,
    )
    exec_cfg, fail_fast = resolve_pipeline_executor_config(cfg, runtime=temp_runtime)
    runtime = replace(temp_runtime, executor_config=exec_cfg, fail_fast=fail_fast)
    _strip_runtime_from_spec(cfg)
    return PipelinePlan(spec=cfg, runtime=runtime)


def _prepare_http(cfg: RepocapsuleConfig, overrides: PipelineOverrides | None = None) -> Optional[SafeHttpClient]:
    """Resolve the SafeHttpClient to use for remote-capable sources.

    If an override is provided via PipelineOverrides, it is returned
    as-is. Otherwise a new client is built from ``cfg.http``.

    Args:
        cfg (RepocapsuleConfig): Effective configuration for the run.
        overrides (PipelineOverrides | None): Optional runtime
            overrides.

    Returns:
        SafeHttpClient | None: Resolved HTTP client, or None when HTTP
            access is not configured.
    """
    if overrides and overrides.http_client is not None:
        return overrides.http_client
    # Avoid mutating global state; caller wires the client explicitly.
    return cfg.http.build_client()


def _assert_runtime_free_spec(cfg: RepocapsuleConfig) -> None:
    """Validate that a RepocapsuleConfig does not embed runtime objects.

    This enforces the convention that declarative specs remain pure data
    by rejecting baked-in sources, sinks, HTTP clients, extractors,
    bytes handlers, or QC scorers.

    Args:
        cfg (RepocapsuleConfig): Configuration to validate.

    Raises:
        ValueError: If any runtime-only field is populated.
    """
    if getattr(cfg.sources, "sources", None):
        raise ValueError("sources.sources must be empty in declarative specs; provide declarative specs instead.")
    if getattr(cfg.sinks, "sinks", None):
        raise ValueError("sinks.sinks must be empty in declarative specs; provide declarative specs instead.")
    if getattr(cfg.http, "client", None) is not None:
        raise ValueError(
            "http.client must be unset in declarative specs; provide HTTP client via runtime wiring or PipelineOverrides.http_client."
        )
    if getattr(cfg.qc, "scorer", None) is not None:
        raise ValueError(
            "qc.scorer must be unset in declarative specs; use QC registry/plugins or PipelineOverrides.qc_scorer instead."
        )
    if getattr(cfg.pipeline, "file_extractor", None) is not None:
        raise ValueError(
            "pipeline.file_extractor must be unset in declarative specs; register extractors or use PipelineOverrides.file_extractor instead."
        )
    if getattr(cfg.pipeline, "extractors", None):
        raise ValueError("pipeline.extractors must be empty in declarative specs; register extractors via runtime wiring instead.")
    if getattr(cfg.pipeline, "bytes_handlers", None):
        raise ValueError(
            "pipeline.bytes_handlers must be empty in declarative specs; use registries/plugins or PipelineOverrides.bytes_handlers instead."
        )


def _prepare_sources(cfg: RepocapsuleConfig, registry: SourceRegistry, *, ctx: SourceFactoryContext) -> tuple[Source, ...]:
    """Build concrete Source instances from declarative specs.

    Args:
        cfg (RepocapsuleConfig): Effective configuration for the run.
        registry (SourceRegistry): Registry used to construct sources.
        ctx (SourceFactoryContext): Context shared across factories.

    Returns:
        tuple[Source, ...]: Concrete sources to enumerate input items.

    Raises:
        ValueError: If ``cfg.sources.sources`` is already populated.
    """
    if cfg.sources.sources:
        raise ValueError("sources.sources must be empty in specs; provide declarative specs instead.")
    if not cfg.sources.specs:
        return ()
    return tuple(registry.build_all(ctx, cfg.sources.specs))


def _prepare_sinks(cfg: RepocapsuleConfig, registry: SinkRegistry, *, ctx: SinkFactoryContext) -> SinksPreparationResult:
    """Build sinks and derive normalized sink config and metadata.

    This function leaves ``cfg.sinks.sinks`` empty in the spec while
    returning concrete Sink instances and an updated SinkConfig and
    RunMetadata pair.

    Args:
        cfg (RepocapsuleConfig): Effective configuration for the run.
        registry (SinkRegistry): Registry used to construct sinks.
        ctx (SinkFactoryContext): Context shared across factories.

    Returns:
        SinksPreparationResult: Container with runtime sinks, normalized
            sink config, and merged run metadata.

    Raises:
        ValueError: If ``cfg.sinks.sinks`` is already populated.
    """
    if cfg.sinks.sinks:
        raise ValueError("sinks.sinks must be empty in specs; provide declarative specs instead.")
    sinks_cfg = replace(cfg.sinks, sinks=tuple())
    metadata = cfg.metadata
    runtime_sinks: tuple[Sink, ...] = ()

    if cfg.sinks.specs:
        sinks, extra_meta, final_ctx = registry.build_all(ctx, cfg.sinks.specs)
        runtime_sinks = tuple(sinks)
        sinks_cfg = replace(final_ctx.sink_config, sinks=tuple())
        metadata = metadata.merged(extra_meta)

    primary = sinks_cfg.primary_jsonl_name or metadata.primary_jsonl
    if primary:
        primary_str = str(primary)
        sinks_cfg = replace(sinks_cfg, primary_jsonl_name=primary_str)
        metadata = metadata.merged({"primary_jsonl": primary_str})

        output_dir = sinks_cfg.output_dir
        needs_output_dir = output_dir is None or str(output_dir) in {"", "."}
        if needs_output_dir:
            try:
                parent = Path(primary_str).parent
            except Exception:
                parent = None
            if parent:
                sinks_cfg = replace(sinks_cfg, output_dir=parent)

    return SinksPreparationResult(
        sinks=runtime_sinks,
        sinks_cfg=sinks_cfg,
        metadata=metadata,
    )


def _prepare_pipeline(
    cfg: RepocapsuleConfig,
    *,
    bytes_registry: BytesHandlerRegistry,
    overrides: PipelineOverrides | None = None,
) -> PipelinePreparationResult:
    """Resolve bytes handlers and file extractor for the pipeline.

    Bytes handlers and file extractor are taken from overrides when
    provided; otherwise they are built from the bytes handler registry
    and defaults.

    Args:
        cfg (RepocapsuleConfig): Effective configuration for the run.
        bytes_registry (BytesHandlerRegistry): Registry used to
            construct bytes handlers when no override is provided.
        overrides (PipelineOverrides | None): Optional runtime
            overrides.

    Returns:
        PipelinePreparationResult: Container with bytes handlers and
            file extractor.
    """
    if overrides and overrides.bytes_handlers is not None:
        bytes_handlers = tuple(overrides.bytes_handlers)
    else:
        bytes_handlers = tuple(make_bytes_handlers(bytes_registry))

    if overrides and overrides.file_extractor is not None:
        file_extractor = overrides.file_extractor
    else:
        file_extractor = DefaultExtractor()

    return PipelinePreparationResult(
        bytes_handlers=bytes_handlers,
        file_extractor=file_extractor,
    )


def _prepare_qc(
    cfg: RepocapsuleConfig,
    *,
    scorer_registry: QualityScorerRegistry,
    overrides: PipelineOverrides | None = None,
) -> QCPreparationResult:
    """Resolve quality-control configuration and scorers for a run.

    This helper normalizes QC mode, wires inline or post-hoc scorers,
    and returns an updated QCConfig plus any factories needed by the
    pipeline.

    Args:
        cfg (RepocapsuleConfig): Effective configuration for the run.
        scorer_registry (QualityScorerRegistry): Registry used to
            construct QC scorers.
        overrides (PipelineOverrides | None): Optional runtime
            overrides.

    Returns:
        QCPreparationResult: Container with normalized QCConfig, inline
            QC hook factory, CSV scorer, and post-QC scorer.

    Raises:
        RuntimeError: If inline or advisory QC is requested but QC
            extras are not installed and no scorer override is provided.
    """
    qc_cfg = cfg.qc
    mode = qc_cfg.normalize_mode()
    qc_hook_factory: Callable[[Any], tuple[Callable[[Record], bool], Callable[[Record], Record]]] | None = None
    scorer_for_csv: QualityScorer | None = None
    post_qc_scorer: QualityScorer | None = None

    if not qc_cfg.enabled or mode == QCMode.OFF:
        qc_cfg.enabled = False
        qc_cfg.mode = QCMode.OFF
        qc_cfg.scorer = None
        return QCPreparationResult(qc_cfg=qc_cfg, qc_hook_factory=None, scorer_for_csv=None, post_qc_scorer=None)

    if overrides and overrides.qc_scorer is not None:
        qc_scorer = overrides.qc_scorer
    else:
        qc_scorer = None

    if mode in {QCMode.INLINE, QCMode.ADVISORY}:
        if qc_scorer is None:
            qc_scorer = make_qc_scorer(qc_cfg, scorer_registry=scorer_registry)
            if qc_scorer is None:
                raise RuntimeError(
                    "Inline/advisory QC requested but QC extras are not installed; disable qc.enabled or install QC dependencies."
                )
        enforce_drops = mode == QCMode.INLINE

        def qc_hook_factory(stats: Any) -> tuple[Callable[[Record], bool], Callable[[Record], Record]]:
            controller = InlineQCController(
                config=qc_cfg,
                stats=stats,
                scorer=qc_scorer,  # type: ignore[arg-type]
                logger=log,
                enforce_drops=enforce_drops,
            )
            return controller.accept, controller.on_record

        if qc_cfg.write_csv:
            scorer_for_csv = qc_scorer
        qc_cfg.scorer = None
        return QCPreparationResult(
            qc_cfg=qc_cfg,
            qc_hook_factory=qc_hook_factory,
            scorer_for_csv=scorer_for_csv,
            post_qc_scorer=None,
        )

    if mode == QCMode.POST:
        post_qc_scorer = qc_scorer or make_qc_scorer(qc_cfg, new_instance=False, scorer_registry=scorer_registry)
        if post_qc_scorer is None:
            log.warning("QC POST mode enabled but QC extras are not installed; skipping QC for this run.")
            qc_cfg.enabled = False
            qc_cfg.mode = QCMode.OFF
        qc_cfg.scorer = None
    return QCPreparationResult(
        qc_cfg=qc_cfg,
        qc_hook_factory=qc_hook_factory,
        scorer_for_csv=scorer_for_csv,
        post_qc_scorer=post_qc_scorer,
    )


def _attach_run_header_record(cfg: RepocapsuleConfig, sinks: Sequence[Sink]) -> None:
    """Attach the run-header record to sinks that support it.

    Any sink implementing ``set_header_record`` will receive a header
    built from the effective configuration.

    Args:
        cfg (RepocapsuleConfig): Effective configuration for the run.
        sinks (Sequence[Sink]): Runtime sinks that may accept header
            records.
    """
    header = build_run_header_record(cfg)
    for sink in sinks:
        setter = getattr(sink, "set_header_record", None)
        if callable(setter):
            setter(header)


def _strip_runtime_from_spec(cfg: RepocapsuleConfig) -> None:
    """Remove runtime-only objects from a RepocapsuleConfig.

    This is called after PipelineRuntime has been constructed so that
    the plan's spec remains a pure, reusable configuration object.

    Args:
        cfg (RepocapsuleConfig): Configuration to sanitize.
    """
    cfg.http.client = None
    cfg.sources.sources = ()
    cfg.sinks.sinks = ()
    cfg.pipeline.bytes_handlers = ()
    cfg.pipeline.file_extractor = None
    cfg.pipeline.extractors = ()
    cfg.qc.scorer = None


def build_engine(config: RepocapsuleConfig, *, mutate: bool = False, overrides: PipelineOverrides | None = None):
    """Build a PipelineEngine from a declarative configuration.

    This is a convenience wrapper around build_pipeline_plan() that
    immediately constructs a PipelineEngine from the resulting plan.

    Args:
        config (RepocapsuleConfig): Declarative configuration for the
            run.
        mutate (bool): Whether to mutate ``config`` in place when
            building the plan.
        overrides (PipelineOverrides | None): Optional runtime-only
            overrides for HTTP client, QC scorer, file extractor, or
            bytes handlers.

    Returns:
        PipelineEngine: Engine ready to be executed via run_pipeline()
            or PipelineEngine.run().
    """
    from .pipeline import PipelineEngine

    plan = build_pipeline_plan(config, mutate=mutate, overrides=overrides)
    return PipelineEngine(plan)
