# plugins.py
# SPDX-License-Identifier: MIT
from __future__ import annotations

from importlib import metadata
from typing import Callable, Optional

from .registries import (
    SourceRegistry,
    SinkRegistry,
    BytesHandlerRegistry,
    QualityScorerRegistry,
)
from .log import get_logger

log = get_logger(__name__)

PluginRegistrar = Callable[[SourceRegistry, SinkRegistry, BytesHandlerRegistry, QualityScorerRegistry], None]


def load_entrypoint_plugins(
    *,
    source_registry: SourceRegistry,
    sink_registry: SinkRegistry,
    bytes_registry: BytesHandlerRegistry,
    scorer_registry: QualityScorerRegistry,
    group: str = "repocapsule.plugins",
) -> None:
    """
    Discover and load plugins from Python entry points.

    Plugins should expose a callable that accepts the registries and performs registrations.
    Failures are logged and skipped to keep the pipeline resilient.
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
            func(source_registry, sink_registry, bytes_registry, scorer_registry)
        except Exception as exc:  # noqa: BLE001
            log.warning("Plugin %s execution failed: %s", ep.name, exc)
