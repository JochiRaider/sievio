# factories_qc.py
# SPDX-License-Identifier: MIT
"""
QC and safety scorer construction helpers.

Split out from core.factories to isolate quality and safety scorer wiring.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from .interfaces import SafetyScorer
from .registries import (
    QualityScorerRegistry,
    SafetyScorerRegistry,
    quality_scorer_registry,
    safety_scorer_registry,
)

if TYPE_CHECKING:  # pragma: no cover - type-only imports
    from .config import QCConfig, SafetyConfig
    from .qc import JSONLQualityScorer

__all__ = [
    "make_qc_scorer",
    "make_safety_scorer",
]


def make_qc_scorer(
    qc_cfg: Optional["QCConfig"],
    *,
    new_instance: bool = False,
    scorer_registry: Optional[QualityScorerRegistry] = None,
) -> Optional["JSONLQualityScorer"]:
    """
    Instantiate a JSONLQualityScorer when QC is enabled and extras are loaded.

    Args:
        qc_cfg (QCConfig | None): Quality-control configuration.
        new_instance (bool): Force creation of a fresh scorer even if one is
            cached on the config.
        scorer_registry (QualityScorerRegistry | None): Registry override for
            resolving scorer factories.

    Returns:
        JSONLQualityScorer | None: Configured scorer, or ``None`` when QC is
        disabled or no scorer is registered.
    """
    if qc_cfg is None or not getattr(qc_cfg, "enabled", False):
        return None
    existing = getattr(qc_cfg, "scorer", None)
    if existing is not None and not new_instance:
        return existing
    # Trigger registration of built-in scorer factory (and any extras).
    try:
        from .extras import qc as _qc_module  # noqa: F401
    except Exception:
        pass
    reg = scorer_registry or quality_scorer_registry
    options = dict(getattr(qc_cfg, "scorer_options", {}) or {})
    factory_id = getattr(qc_cfg, "scorer_id", None)
    scorer = reg.build(options, factory_id=factory_id)
    if scorer is None:
        return None
    if not new_instance:
        qc_cfg.scorer = scorer
    return scorer


def make_safety_scorer(
    safety_cfg: Optional["SafetyConfig"],
    *,
    new_instance: bool = False,
    registry: Optional[SafetyScorerRegistry] = None,
) -> Optional[SafetyScorer]:
    """
    Instantiate a SafetyScorer when safety is enabled and extras are loaded.

    Args:
        safety_cfg (SafetyConfig | None): Safety configuration.
        new_instance (bool): Force creation of a fresh scorer even if one is
            cached on the config.
        registry (SafetyScorerRegistry | None): Registry override for resolving
            scorer factories.

    Returns:
        SafetyScorer | None: Configured scorer, or ``None`` when safety is
        disabled or no scorer is registered.
    """
    if safety_cfg is None or not getattr(safety_cfg, "enabled", False):
        return None
    existing = getattr(safety_cfg, "scorer", None)
    if existing is not None and not new_instance:
        return existing
    try:
        from .extras import safety as _safety_module  # noqa: F401
    except Exception:
        pass
    reg = registry or safety_scorer_registry
    options = dict(getattr(safety_cfg, "scorer_options", {}) or {})
    factory_id = getattr(safety_cfg, "scorer_id", None)
    scorer = reg.build(options, factory_id=factory_id)
    if scorer is None:
        return None
    if not new_instance:
        safety_cfg.scorer = scorer
    return scorer
