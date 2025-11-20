# registries.py
# SPDX-License-Identifier: MIT
from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

from .config import RepocapsuleConfig, SourceSpec, SinkSpec, QCConfig
from .interfaces import Source, Sink, SourceFactory, SinkFactory, QualityScorer
from .log import get_logger

if TYPE_CHECKING:  # pragma: no cover - type-only deps
    from .factories import SinkFactoryResult


@dataclass
class SourceRegistry:
    _factories: Dict[str, SourceFactory] = field(default_factory=dict)

    def register(self, factory: SourceFactory) -> None:
        self._factories[factory.id] = factory

    def build_all(self, cfg: RepocapsuleConfig) -> List[Source]:
        out: List[Source] = []
        for spec in cfg.sources.specs:
            factory = self._factories.get(spec.kind)
            if factory is None:
                raise ValueError(f"Unknown source kind {spec.kind!r}")
            out.extend(factory.build(cfg, spec))
        return out


@dataclass
class SinkRegistry:
    _factories: Dict[str, SinkFactory] = field(default_factory=dict)

    def register(self, factory: SinkFactory) -> None:
        self._factories[factory.id] = factory

    def build_all(self, cfg: RepocapsuleConfig) -> Tuple[List[Sink], Mapping[str, Any]]:
        sinks: List[Sink] = []
        merged_meta: Dict[str, Any] = {}
        for spec in cfg.sinks.specs:
            factory = self._factories.get(spec.kind)
            if factory is None:
                raise ValueError(f"Unknown sink kind {spec.kind!r}")
            result: "SinkFactoryResult" = factory.build(cfg, spec)
            sinks.extend(result.sinks)
            for k, v in result.metadata.items():
                if k not in merged_meta or merged_meta[k] is None:
                    merged_meta[k] = v
            cfg.sinks = result.sink_config
        return sinks, merged_meta


class BytesHandlerRegistry:
    def __init__(self) -> None:
        self._handlers: List[Tuple[Callable[[bytes, str], bool], Callable[..., Optional[Iterable[Any]]]]] = []

    def register(self, sniff: Callable[[bytes, str], bool], handler: Callable[..., Optional[Iterable[Any]]]) -> None:
        self._handlers.append((sniff, handler))

    def handlers(self) -> Tuple[Tuple[Callable[[bytes, str], bool], Callable[..., Optional[Iterable[Any]]]], ...]:
        return tuple(self._handlers)


class QualityScorerFactory(Protocol):
    id: str

    def build(self, cfg: QCConfig) -> QualityScorer:
        ...


class QualityScorerRegistry:
    def __init__(self) -> None:
        self._factories: Dict[str, QualityScorerFactory] = {}
        self.log = get_logger(__name__)

    def register(self, factory: QualityScorerFactory) -> None:
        self._factories[factory.id] = factory

    def get(self, factory_id: Optional[str] = None) -> Optional[QualityScorerFactory]:
        if factory_id is not None:
            return self._factories.get(factory_id)
        if not self._factories:
            return None
        first_key = next(iter(self._factories))
        return self._factories[first_key]

    def build(self, cfg: QCConfig, *, factory_id: Optional[str] = None) -> Optional[QualityScorer]:
        factory = self.get(factory_id)
        if factory is None:
            return None
        try:
            return factory.build(cfg)
        except Exception as exc:
            self.log.warning("Quality scorer factory %s failed: %s", getattr(factory, "id", None), exc)
            return None

    def ids(self) -> Tuple[str, ...]:
        return tuple(self._factories.keys())


def default_source_registry() -> SourceRegistry:
    from .factories import (
        LocalDirSourceFactory,
        GitHubZipSourceFactory,
        WebPdfListSourceFactory,
        WebPagePdfSourceFactory,
    )

    reg = SourceRegistry()
    reg.register(LocalDirSourceFactory())
    reg.register(GitHubZipSourceFactory())
    reg.register(WebPdfListSourceFactory())
    reg.register(WebPagePdfSourceFactory())
    return reg


def default_sink_registry() -> SinkRegistry:
    from .factories import DefaultJsonlPromptSinkFactory

    reg = SinkRegistry()
    reg.register(DefaultJsonlPromptSinkFactory())
    return reg


bytes_handler_registry = BytesHandlerRegistry()
quality_scorer_registry = QualityScorerRegistry()
