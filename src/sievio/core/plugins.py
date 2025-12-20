# plugins.py
# SPDX-License-Identifier: MIT
"""Plugin discovery helpers for registering sources, sinks, and scorers."""

from __future__ import annotations

from collections.abc import Callable
from importlib import metadata

from .log import get_logger
from .registries import (
    BytesHandlerRegistry,
    QualityScorerRegistry,
    SafetyScorerRegistry,
    SinkRegistry,
    SourceRegistry,
)

log = get_logger(__name__)

PluginRegistrar = Callable[..., None]


def load_entrypoint_plugins(
    *,
    source_registry: SourceRegistry,
    sink_registry: SinkRegistry,
    bytes_registry: BytesHandlerRegistry,
    scorer_registry: QualityScorerRegistry,
    safety_scorer_registry: SafetyScorerRegistry,
    group: str = "sievio.plugins",
) -> None:
    """Discover and load plugin entry points.

    Plugins must expose a callable that accepts the registries and performs
    registrations. Failures are logged and ignored to keep the pipeline
    resilient.

    Args:
        source_registry (SourceRegistry): Registry to register sources.
        sink_registry (SinkRegistry): Registry to register sinks.
        bytes_registry (BytesHandlerRegistry): Registry to register bytes
            handlers.
        scorer_registry (QualityScorerRegistry): Registry to register quality
            scorers.
        safety_scorer_registry (SafetyScorerRegistry): Registry to register
            safety scorers.
        group (str): Entry-point group name to search for plugins.
    """
    try:
        entry_points = metadata.entry_points()
    except Exception as exc:  # pragma: no cover - importlib.metadata safety
        log.debug("Plugin discovery skipped: %s", exc)
        return

    eps = entry_points.select(group=group) if hasattr(entry_points, "select") else entry_points.get(group, [])
    for ep in eps:
        try:
            func = ep.load()
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to import plugin %s: %s", ep.name, exc)
            continue
        try:
            try:
                func(source_registry, sink_registry, bytes_registry, scorer_registry, safety_scorer_registry)
            except TypeError:
                func(source_registry, sink_registry, bytes_registry, scorer_registry)
        except Exception as exc:  # noqa: BLE001
            log.warning("Plugin %s execution failed: %s", ep.name, exc)
