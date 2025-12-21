# registry_callables.py
# SPDX-License-Identifier: MIT
"""Callable adapters for registry-based extension points."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from typing import TYPE_CHECKING, Any, cast

from .config import build_config_from_defaults_and_options, validate_options_for_dataclass
from .factories_sinks import SinkFactoryResult
from .interfaces import (
    Sink,
    SinkFactory,
    SinkFactoryContext,
    Source,
    SourceFactory,
    SourceFactoryContext,
)

if TYPE_CHECKING:  # pragma: no cover - typing-only import guard for forward references
    from .config import SinkSpec, SourceSpec
    from .interfaces import FileItem, QualityScorer, RepoContext, SafetyScorer


def _options_from_dataclass(cfg: Any) -> dict[str, Any]:
    return {field.name: getattr(cfg, field.name) for field in fields(cfg)}


def _merge_callable_options(
    *,
    kind: str,
    defaults: Mapping[str, Any] | None,
    options: Mapping[str, Any] | None,
    options_model: type[Any] | None,
    context_label: str,
) -> dict[str, Any]:
    if options_model is None:
        merged: dict[str, Any] = dict(defaults or {})
        if options:
            merged.update(options)
        return merged
    if not is_dataclass(options_model):
        raise TypeError(f"{kind} options_model must be a dataclass type")
    validate_options_for_dataclass(options_model, options=options, context=context_label)
    cfg = build_config_from_defaults_and_options(
        options_model,
        defaults=defaults,
        options=options,
    )
    return _options_from_dataclass(cfg)


def _is_sinkish(value: Any) -> bool:
    return callable(getattr(value, "write", None)) and callable(getattr(value, "close", None))


@dataclass(slots=True)
class CallableSource(Source):
    """Source adapter that forwards iteration to a user callable."""

    fn: Callable[..., Iterable[FileItem] | None]
    options: Mapping[str, Any]
    ctx: SourceFactoryContext
    spec: SourceSpec
    name: str | None = None
    context: RepoContext | None = None
    _last_iter: Iterable[FileItem] | None = None

    def iter_files(self) -> Iterable[FileItem]:
        result = self.fn(self.options, ctx=self.ctx, spec=self.spec)
        if result is None:
            result = ()
        self._last_iter = result
        return result

    def close(self) -> None:
        last_iter = self._last_iter
        self._last_iter = None
        if last_iter is None:
            return
        close_fn = getattr(last_iter, "close", None)
        if callable(close_fn):
            close_fn()


@dataclass(slots=True)
class CallableSourceFactory(SourceFactory):
    """Factory adapter wrapping a user callable into a Source."""

    id: str
    fn: Callable[..., Iterable[FileItem] | None]
    options_model: type[Any] | None = None
    name: str | None = None

    def build(self, ctx: SourceFactoryContext, spec: SourceSpec) -> Iterable[Source]:
        options = _merge_callable_options(
            kind="source",
            defaults=ctx.source_defaults.get(self.id, {}) or {},
            options=spec.options,
            options_model=self.options_model,
            context_label=f"sources.specs.{self.id}",
        )
        source = CallableSource(
            fn=self.fn,
            options=options,
            ctx=ctx,
            spec=spec,
            name=self.name or getattr(self.fn, "__name__", None),
            context=ctx.repo_context,
        )
        return [source]


@dataclass(slots=True)
class CallableSinkFactory(SinkFactory):
    """Factory adapter wrapping a callable into a Sink factory."""

    id: str
    fn: Callable[..., SinkFactoryResult | Any | Iterable[Any]]
    options_model: type[Any] | None = None
    name: str | None = None

    def build(self, ctx: SinkFactoryContext, spec: SinkSpec) -> SinkFactoryResult:
        options = _merge_callable_options(
            kind="sink",
            defaults=ctx.sink_defaults.get(self.id, {}) or {},
            options=spec.options,
            options_model=self.options_model,
            context_label=f"sinks.specs.{self.id}",
        )
        result = self.fn(options, ctx=ctx, spec=spec)
        if isinstance(result, SinkFactoryResult):
            return result
        if _is_sinkish(result):
            sinks = [cast(Sink, result)]
        else:
            try:
                sinks = list(cast(Iterable[Sink], result))
            except TypeError as exc:
                raise TypeError(
                    "Callable sink factory must return a Sink or iterable of Sink"
                ) from exc
        if not all(_is_sinkish(sink) for sink in sinks):
            raise TypeError("Callable sink factory returned non-Sink entries")
        jsonl_path = ctx.sink_config.primary_jsonl_name or ""
        return SinkFactoryResult(
            jsonl_path=str(jsonl_path),
            sinks=cast(Sequence[Sink], sinks),
            sink_config=ctx.sink_config,
            metadata={},
        )


@dataclass(slots=True)
class CallableQualityScorerFactory:
    """Factory adapter wrapping a callable into a quality scorer factory."""

    id: str
    fn: Callable[..., QualityScorer]
    options_model: type[Any] | None = None

    def build(self, options: Mapping[str, Any]) -> QualityScorer:
        opts = _merge_callable_options(
            kind="quality_scorer",
            defaults=None,
            options=options,
            options_model=self.options_model,
            context_label=f"qc.scorer_options.{self.id}",
        )
        return self.fn(opts)


@dataclass(slots=True)
class CallableSafetyScorerFactory:
    """Factory adapter wrapping a callable into a safety scorer factory."""

    id: str
    fn: Callable[..., SafetyScorer]
    options_model: type[Any] | None = None

    def build(self, options: Mapping[str, Any]) -> SafetyScorer:
        opts = _merge_callable_options(
            kind="safety_scorer",
            defaults=None,
            options=options,
            options_model=self.options_model,
            context_label=f"safety.scorer_options.{self.id}",
        )
        return self.fn(opts)


__all__ = [
    "CallableSource",
    "CallableSourceFactory",
    "CallableSinkFactory",
    "CallableQualityScorerFactory",
    "CallableSafetyScorerFactory",
]
