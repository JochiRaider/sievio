# registries.py
# SPDX-License-Identifier: MIT
"""Registries for sources, sinks, byte handlers, and quality scorers."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
)

from .config import SievioConfig, SinkSpec, SourceSpec
from .interfaces import (
    QualityScorer,
    RunLifecycleHook,
    SafetyScorer,
    SafetyScorerFactory,
    Sink,
    SinkFactory,
    SinkFactoryContext,
    Source,
    SourceFactory,
    SourceFactoryContext,
)
from .log import get_logger
from .registry_callables import (
    CallableQualityScorerFactory,
    CallableSafetyScorerFactory,
    CallableSinkFactory,
    CallableSourceFactory,
)

if TYPE_CHECKING:  # pragma: no cover - type-only deps
    from .builder import PipelineRuntime
    from .factories import SinkFactoryResult


@dataclass
class SourceRegistry:
    """Registry for source factories keyed by their ids."""

    _factories: dict[str, SourceFactory] = field(default_factory=dict)

    def register(self, factory: SourceFactory) -> None:
        """Register a source factory."""
        self._factories[factory.id] = factory

    def register_callable(
        self,
        kind: str,
        fn: Callable[..., Iterable[Any] | None],
        *,
        options_model: type[Any] | None = None,
        replace: bool = False,
        name: str | None = None,
    ) -> None:
        """Register a callable as a SourceFactory-backed source."""
        if not replace and kind in self._factories:
            raise ValueError(f"Source factory {kind!r} is already registered")
        self._factories[kind] = CallableSourceFactory(
            id=kind,
            fn=fn,
            options_model=options_model,
            name=name,
        )

    def source(
        self,
        kind: str,
        *,
        options_model: type[Any] | None = None,
        replace: bool = False,
        name: str | None = None,
    ) -> Callable[
        [Callable[..., Iterable[Any] | None]],
        Callable[..., Iterable[Any] | None],
    ]:
        """Decorator to register a callable source for ``kind``."""
        def decorator(
            fn: Callable[..., Iterable[Any] | None],
        ) -> Callable[..., Iterable[Any] | None]:
            self.register_callable(
                kind,
                fn,
                options_model=options_model,
                replace=replace,
                name=name,
            )
            return fn

        return decorator

    def build_all(self, ctx: SourceFactoryContext, specs: Sequence[SourceSpec]) -> list[Source]:
        """Instantiate sources for each spec using the registered factories.

        Args:
            ctx (SourceFactoryContext): Shared context for factory builds.
            specs (Sequence[SourceSpec]): Source specifications to realize.

        Returns:
            List[Source]: Concrete sources produced by the factories.

        Raises:
            ValueError: If a spec references an unknown source kind.
        """
        out: list[Source] = []
        for spec in specs:
            factory = self._factories.get(spec.kind)
            if factory is None:
                raise ValueError(f"Unknown source kind {spec.kind!r}")
            out.extend(factory.build(ctx, spec))
        return out


@dataclass
class SinkRegistry:
    """Registry for sink factories and related metadata merges."""

    _factories: dict[str, SinkFactory] = field(default_factory=dict)

    def register(self, factory: SinkFactory) -> None:
        """Register a sink factory."""
        self._factories[factory.id] = factory

    def register_callable(
        self,
        kind: str,
        fn: Callable[..., Any],
        *,
        options_model: type[Any] | None = None,
        replace: bool = False,
        name: str | None = None,
    ) -> None:
        """Register a callable as a SinkFactory-backed sink."""
        if not replace and kind in self._factories:
            raise ValueError(f"Sink factory {kind!r} is already registered")
        self._factories[kind] = CallableSinkFactory(
            id=kind,
            fn=fn,
            options_model=options_model,
            name=name,
        )

    def sink(
        self,
        kind: str,
        *,
        options_model: type[Any] | None = None,
        replace: bool = False,
        name: str | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to register a callable sink for ``kind``."""
        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            self.register_callable(
                kind,
                fn,
                options_model=options_model,
                replace=replace,
                name=name,
            )
            return fn

        return decorator

    def build_all(
        self,
        ctx: SinkFactoryContext,
        specs: Sequence[SinkSpec],
    ) -> tuple[list[Sink], Mapping[str, Any], SinkFactoryContext]:
        """Instantiate sinks for each spec and merge factory metadata.

        The context may be updated by successive factories; the final context
        is returned for downstream consumers.

        Args:
            ctx (SinkFactoryContext): Initial sink factory context.
            specs (Sequence[SinkSpec]): Sink specifications to realize.

        Returns:
            Tuple[List[Sink], Mapping[str, Any], SinkFactoryContext]: Tuple
                containing sinks, merged metadata, and the final factory
                context.

        Raises:
            ValueError: If a spec references an unknown sink kind.
        """
        sinks: list[Sink] = []
        merged_meta: dict[str, Any] = {}
        current_ctx = ctx
        for spec in specs:
            factory = self._factories.get(spec.kind)
            if factory is None:
                raise ValueError(f"Unknown sink kind {spec.kind!r}")
            result: SinkFactoryResult = factory.build(current_ctx, spec)
            sinks.extend(result.sinks)
            for k, v in result.metadata.items():
                if k not in merged_meta or merged_meta[k] is None:
                    merged_meta[k] = v
            # Propagate any sink_config updates back into the context so that
            # subsequent factories see the latest settings.
            current_ctx = SinkFactoryContext(
                repo_context=result.sink_config.context or current_ctx.repo_context,
                sink_config=result.sink_config,
                sink_defaults=current_ctx.sink_defaults,
            )
        return sinks, merged_meta, current_ctx


class BytesHandlerRegistry:
    """Registry for handlers operating on raw bytes inputs."""

    def __init__(self) -> None:
        self._handlers: list[
            tuple[
                Callable[[bytes, str], bool],
                Callable[..., Iterable[Any] | None],
            ]
        ] = []

    def register(
        self,
        sniff: Callable[[bytes, str], bool],
        handler: Callable[..., Iterable[Any] | None],
    ) -> None:
        """Register a handler with a sniff predicate."""
        self._handlers.append((sniff, handler))

    def handlers(
        self,
    ) -> tuple[
        tuple[Callable[[bytes, str], bool], Callable[..., Iterable[Any] | None]],
        ...,
    ]:
        """Return registered (sniff, handler) pairs."""
        return tuple(self._handlers)


class QualityScorerFactory(Protocol):
    """Protocol describing quality scorer factories."""

    id: str

    def build(self, options: Mapping[str, Any]) -> QualityScorer:
        ...


class QualityScorerRegistry:
    """Registry for quality scorer factories with safe construction."""

    def __init__(self) -> None:
        self._factories: dict[str, QualityScorerFactory] = {}
        self.log = get_logger(__name__)
        # DEFAULT_QC_SCORER_ID is used when qc.scorer_id is None and a default scorer is registered.

    def register(self, factory: QualityScorerFactory) -> None:
        """Register a quality scorer factory."""
        self._factories[factory.id] = factory

    def register_callable(
        self,
        factory_id: str,
        fn: Callable[..., QualityScorer],
        *,
        options_model: type[Any] | None = None,
        replace: bool = False,
    ) -> None:
        """Register a callable as a quality scorer factory."""
        if not replace and factory_id in self._factories:
            raise ValueError(f"Quality scorer factory {factory_id!r} is already registered")
        self._factories[factory_id] = CallableQualityScorerFactory(
            id=factory_id,
            fn=fn,
            options_model=options_model,
        )

    def get(self, factory_id: str | None = None) -> QualityScorerFactory | None:
        """Return a scorer factory by id or the first registered one."""
        if factory_id is not None:
            return self._factories.get(factory_id)
        if not self._factories:
            return None
        first_key = next(iter(self._factories))
        return self._factories[first_key]

    def build(
        self,
        options: Mapping[str, Any],
        *,
        factory_id: str | None = None,
    ) -> QualityScorer | None:
        """Safely build a quality scorer instance.

        Args:
            options (Mapping[str, Any]): Configuration passed to the factory.
            factory_id (Optional[str]): Identifier for the desired factory.
                Defaults to the first registered factory when omitted.

        Returns:
            Optional[QualityScorer]: Constructed scorer or ``None`` on error.
        """
        factory = self.get(factory_id)
        if factory is None:
            return None
        try:
            return factory.build(options)
        except Exception as exc:
            self.log.warning(
                "Quality scorer factory %s failed: %s",
                getattr(factory, "id", None),
                exc,
            )
            return None

    def ids(self) -> tuple[str, ...]:
        """Return ids of registered quality scorer factories."""
        return tuple(self._factories.keys())


class SafetyScorerRegistry:
    """Registry for safety scorer factories with safe construction."""

    def __init__(self) -> None:
        self._factories: dict[str, SafetyScorerFactory] = {}
        self.log = get_logger(__name__)

    def register(self, factory: SafetyScorerFactory) -> None:
        """Register a safety scorer factory."""
        self._factories[factory.id] = factory

    def register_callable(
        self,
        factory_id: str,
        fn: Callable[..., SafetyScorer],
        *,
        options_model: type[Any] | None = None,
        replace: bool = False,
    ) -> None:
        """Register a callable as a safety scorer factory."""
        if not replace and factory_id in self._factories:
            raise ValueError(f"Safety scorer factory {factory_id!r} is already registered")
        self._factories[factory_id] = CallableSafetyScorerFactory(
            id=factory_id,
            fn=fn,
            options_model=options_model,
        )

    def get(self, factory_id: str | None = None) -> SafetyScorerFactory | None:
        """Return a scorer factory by id or the first registered one."""
        if factory_id is not None:
            return self._factories.get(factory_id)
        if not self._factories:
            return None
        first_key = next(iter(self._factories))
        return self._factories[first_key]

    def build(
        self,
        options: Mapping[str, Any],
        *,
        factory_id: str | None = None,
    ) -> SafetyScorer | None:
        """Safely build a safety scorer instance."""
        factory = self.get(factory_id)
        if factory is None:
            return None
        try:
            return factory.build(options)
        except Exception as exc:
            self.log.warning("Safety scorer factory %s failed: %s", factory_id or "<default>", exc)
            return None

    def ids(self) -> tuple[str, ...]:
        """Return ids of registered safety scorer factories."""
        return tuple(self._factories.keys())


class LifecycleHookFactory(Protocol):
    """Protocol describing lifecycle hook factories."""

    id: str

    def build(self, cfg: SievioConfig, runtime: PipelineRuntime) -> RunLifecycleHook:
        ...


class LifecycleHookRegistry:
    """Registry for lifecycle hook factories keyed by id."""

    def __init__(self) -> None:
        self._factories: dict[str, LifecycleHookFactory] = {}

    def register(self, factory: LifecycleHookFactory) -> None:
        self._factories[factory.id] = factory

    def build_all(
        self,
        cfg: SievioConfig,
        runtime: PipelineRuntime,
        ids: Sequence[str],
    ) -> list[RunLifecycleHook]:
        hooks: list[RunLifecycleHook] = []
        for hook_id in ids:
            factory = self._factories.get(hook_id)
            if factory is None:
                raise ValueError(f"Unknown lifecycle hook {hook_id!r}")
            hooks.append(factory.build(cfg, runtime))
        return hooks


def default_source_registry() -> SourceRegistry:
    """Build a SourceRegistry populated with the default factories."""
    from .factories import (
        CsvTextSourceFactory,
        GitHubZipSourceFactory,
        LocalDirSourceFactory,
        SQLiteSourceFactory,
        WebPagePdfSourceFactory,
        WebPdfListSourceFactory,
    )

    reg = SourceRegistry()
    reg.register(LocalDirSourceFactory())
    reg.register(GitHubZipSourceFactory())
    reg.register(WebPdfListSourceFactory())
    reg.register(WebPagePdfSourceFactory())
    reg.register(CsvTextSourceFactory())
    reg.register(SQLiteSourceFactory())
    return reg


def default_sink_registry() -> SinkRegistry:
    """Build a SinkRegistry populated with the default factories."""
    from .factories import DefaultJsonlPromptSinkFactory, ParquetDatasetSinkFactory

    reg = SinkRegistry()
    reg.register(DefaultJsonlPromptSinkFactory())
    reg.register(ParquetDatasetSinkFactory())
    return reg


# Shared registries that can be reused across runs.
bytes_handler_registry = BytesHandlerRegistry()
quality_scorer_registry = QualityScorerRegistry()
safety_scorer_registry = SafetyScorerRegistry()


@dataclass(slots=True)
class RegistryBundle:
    """Bundle of registries used to build a pipeline.

    This makes plugin and registry configuration explicit for
    advanced callers and tests.
    """

    sources: SourceRegistry
    sinks: SinkRegistry
    bytes: BytesHandlerRegistry
    scorers: QualityScorerRegistry
    safety_scorers: SafetyScorerRegistry


def default_registries(*, load_plugins: bool = True) -> RegistryBundle:
    """Return a bundle of default registries, optionally with plugins loaded."""
    from .plugins import load_entrypoint_plugins  # local import to avoid cycles

    source_reg = default_source_registry()
    sink_reg = default_sink_registry()
    bytes_reg = bytes_handler_registry
    scorer_reg = quality_scorer_registry
    safety_reg = safety_scorer_registry

    if load_plugins:
        load_entrypoint_plugins(
            source_registry=source_reg,
            sink_registry=sink_reg,
            bytes_registry=bytes_reg,
            scorer_registry=scorer_reg,
            safety_scorer_registry=safety_reg,
        )

    return RegistryBundle(
        sources=source_reg,
        sinks=sink_reg,
        bytes=bytes_reg,
        scorers=scorer_reg,
        safety_scorers=safety_reg,
    )


__all__ = [
    "SourceRegistry",
    "SinkRegistry",
    "BytesHandlerRegistry",
    "QualityScorerRegistry",
    "LifecycleHookFactory",
    "LifecycleHookRegistry",
    "RegistryBundle",
    "default_source_registry",
    "default_sink_registry",
    "default_registries",
    "bytes_handler_registry",
    "quality_scorer_registry",
    "safety_scorer_registry",
    "SafetyScorerRegistry",
]
