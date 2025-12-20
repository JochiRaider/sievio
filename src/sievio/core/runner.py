# SPDX-License-Identifier: MIT
"""Orchestration helpers that bridge configuration, builder, and engine."""
from __future__ import annotations

from .builder import PipelineOverrides, build_engine
from .config import SievioConfig

__all__ = ["run_pipeline"]


def run_pipeline(
    *,
    config: SievioConfig,
    overrides: PipelineOverrides | None = None,
) -> dict[str, int]:
    """Run the end-to-end pipeline described by config.

    Args:
        config (SievioConfig): Pipeline configuration object.
        overrides (PipelineOverrides | None): Optional runtime overrides
            merged into the plan and engine.

    Returns:
        dict[str, int]: Statistics from execution as primitive values.
    """
    engine = build_engine(config, overrides=overrides)
    stats_obj = engine.run()
    return stats_obj.as_dict()
