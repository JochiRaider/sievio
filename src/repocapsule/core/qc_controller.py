# qc_controller.py
# SPDX-License-Identifier: MIT
"""Helpers for inline QC execution and summary building."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .config import QCConfig, QCMode, SafetyConfig
from .interfaces import InlineScreener, QualityScorer, Record, RunLifecycleHook, RunContext, RunArtifacts, SafetyScorer
from .log import get_logger
from .records import (
    ensure_meta_dict,
    merge_meta_defaults,
    best_effort_record_path,
    filter_qc_meta,
    filter_safety_meta,
)
from .qc_utils import update_dup_family_counts, top_dup_families

log = get_logger(__name__)


@dataclass(slots=True)
class ScalarSignalStats:
    count: int = 0
    sum: float = 0.0
    sum_sq: float = 0.0
    min: float | None = None
    max: float | None = None

    def observe(self, v: float) -> None:
        self.count += 1
        self.sum += v
        self.sum_sq += v * v
        self.min = v if self.min is None or v < self.min else self.min
        self.max = v if self.max is None or v > self.max else self.max

    def as_dict(self) -> Dict[str, Any]:
        if self.count == 0:
            return {"count": 0}
        mean = self.sum / self.count
        var = max(self.sum_sq / self.count - mean * mean, 0.0)
        return {
            "count": self.count,
            "mean": mean,
            "min": self.min,
            "max": self.max,
            "stdev": var ** 0.5,
        }


@dataclass(slots=True)
class ScreenerStats:
    """Per-screener summary used by QCSummaryTracker."""

    id: str
    enabled: bool = False
    mode: str = QCMode.INLINE
    scored: int = 0
    kept: int = 0
    dropped: int = 0
    errors: int = 0
    signal_stats: Dict[str, ScalarSignalStats] = field(default_factory=dict)
    flags: Dict[str, int] = field(default_factory=dict)
    candidates: Dict[str, int] = field(default_factory=dict)
    drops: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "enabled": bool(self.enabled),
            "mode": self.mode,
            "scored": int(self.scored),
            "kept": int(self.kept),
            "dropped": int(self.dropped),
            "errors": int(self.errors),
            "signal_stats": {k: s.as_dict() for k, s in self.signal_stats.items()},
            "flags": dict(self.flags),
            "candidates": dict(self.candidates),
            "drops": dict(self.drops),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, default_id: str | None = None) -> "ScreenerStats":
        sid = str(data.get("id") or default_id or "")
        stats = cls(id=sid)
        stats.enabled = bool(data.get("enabled"))
        mode = data.get("mode")
        if isinstance(mode, str) and mode:
            stats.mode = mode
        stats.scored = int(data.get("scored") or 0)
        stats.kept = int(data.get("kept") or 0)
        stats.dropped = int(data.get("dropped") or 0)
        stats.errors = int(data.get("errors") or 0)
        signals = data.get("signal_stats")
        if isinstance(signals, Mapping):
            parsed: Dict[str, ScalarSignalStats] = {}
            for name, payload in signals.items():
                if not isinstance(payload, Mapping):
                    continue
                parsed[str(name)] = _parse_scalar_signal_stats(payload)
            stats.signal_stats = parsed
        flags = data.get("flags")
        if isinstance(flags, Mapping):
            stats.flags = {str(k): int(v) for k, v in flags.items() if v is not None}
        candidates = data.get("candidates")
        if isinstance(candidates, Mapping):
            stats.candidates = {str(k): int(v) for k, v in candidates.items() if v is not None}
        drops = data.get("drops")
        if isinstance(drops, Mapping):
            stats.drops = {str(k): int(v) for k, v in drops.items() if v is not None}
        return stats


def _parse_scalar_signal_stats(payload: Mapping[str, Any]) -> ScalarSignalStats:
    stats = ScalarSignalStats()
    try:
        stats.count = int(payload.get("count") or 0)
    except Exception:
        stats.count = 0
    try:
        stats.min = float(payload["min"]) if payload.get("min") is not None else None
    except Exception:
        stats.min = None
    try:
        stats.max = float(payload["max"]) if payload.get("max") is not None else None
    except Exception:
        stats.max = None
    mean_val = payload.get("mean")
    stdev_val = payload.get("stdev")
    try:
        mean = float(mean_val) if mean_val is not None else None
    except Exception:
        mean = None
    try:
        stdev = float(stdev_val) if stdev_val is not None else None
    except Exception:
        stdev = None
    if stats.count and mean is not None:
        stats.sum = mean * stats.count
        variance = (stdev * stdev) if stdev is not None else 0.0
        stats.sum_sq = (variance + mean * mean) * stats.count
    return stats


@dataclass(slots=True, init=False)
class QCSummaryTracker:
    """Track screening outcomes (quality + safety) and duplicate families.

    near_dup is treated as a combined flag (Simhash OR MinHash). With
    drop_near_dups=True, any record flagged near-duplicate by either mechanism
    will be dropped. Duplicate families are keyed by dup_family_id with
    counts/examples for post-QC reporting.
    """
    enabled: bool = False
    mode: str = QCMode.INLINE
    min_score: Optional[float] = None
    drop_near_dups: bool = False
    dup_families: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    top_dup_snapshot: List[Dict[str, Any]] = field(default_factory=list)
    screeners: Dict[str, ScreenerStats] = field(default_factory=dict)

    def __init__(
        self,
        *,
        enabled: bool = False,
        mode: str = QCMode.INLINE,
        min_score: Optional[float] = None,
        drop_near_dups: bool = False,
        screeners: Mapping[str, ScreenerStats | Mapping[str, Any]] | None = None,
        dup_families: Mapping[str, Dict[str, Any]] | None = None,
        top_dup_snapshot: List[Dict[str, Any]] | None = None,
        **legacy: Any,
    ) -> None:
        self.enabled = bool(enabled)
        self.mode = mode
        self.min_score = min_score
        self.drop_near_dups = bool(drop_near_dups)
        self.dup_families = dict(dup_families) if dup_families else {}
        self.top_dup_snapshot = list(top_dup_snapshot) if top_dup_snapshot else []
        self.screeners = {}
        if screeners:
            for sid, payload in screeners.items():
                if isinstance(payload, ScreenerStats):
                    self.screeners[str(sid)] = payload
                elif isinstance(payload, Mapping):
                    self.screeners[str(sid)] = ScreenerStats.from_dict(payload, default_id=str(sid))
        self._apply_legacy_kwargs(legacy)

    def reset_for_run(
        self,
        *,
        enabled: bool = False,
        mode: str = QCMode.INLINE,
        min_score: Optional[float] = None,
        drop_near_dups: bool = False,
    ) -> None:
        """Reconfigure tracker fields and clear per-run state in place."""
        self.enabled = bool(enabled)
        self.mode = mode
        self.min_score = min_score
        self.drop_near_dups = bool(drop_near_dups)
        self.dup_families.clear()
        self.top_dup_snapshot.clear()
        self.screeners.clear()

    # ------------------------------------------------------------------
    # Legacy QC-centric views routed through per-screener stats.
    # ------------------------------------------------------------------
    @property
    def scored(self) -> int:
        return self._quality_stats().scored

    @scored.setter
    def scored(self, value: int) -> None:
        self._quality_stats().scored = int(value or 0)

    @property
    def kept(self) -> int:
        return self._quality_stats().kept

    @kept.setter
    def kept(self, value: int) -> None:
        self._quality_stats().kept = int(value or 0)

    @property
    def dropped_low_score(self) -> int:
        return self._quality_stats().drops.get("low_score", 0)

    @dropped_low_score.setter
    def dropped_low_score(self, value: int) -> None:
        self._quality_stats().drops["low_score"] = int(value or 0)

    @property
    def dropped_near_dup(self) -> int:
        return self._quality_stats().drops.get("near_dup", 0)

    @dropped_near_dup.setter
    def dropped_near_dup(self, value: int) -> None:
        self._quality_stats().drops["near_dup"] = int(value or 0)

    @property
    def errors(self) -> int:
        return self._quality_stats().errors

    @errors.setter
    def errors(self, value: int) -> None:
        self._quality_stats().errors = int(value or 0)

    @property
    def candidates_low_score(self) -> int:
        return self._quality_stats().candidates.get("low_score", 0)

    @candidates_low_score.setter
    def candidates_low_score(self, value: int) -> None:
        self._quality_stats().candidates["low_score"] = int(value or 0)

    @property
    def candidates_near_dup(self) -> int:
        return self._quality_stats().candidates.get("near_dup", 0)

    @candidates_near_dup.setter
    def candidates_near_dup(self, value: int) -> None:
        self._quality_stats().candidates["near_dup"] = int(value or 0)

    @property
    def signal_stats(self) -> Dict[str, ScalarSignalStats]:
        return self._quality_stats().signal_stats

    @signal_stats.setter
    def signal_stats(self, value: Mapping[str, ScalarSignalStats]) -> None:
        self._quality_stats().signal_stats = dict(value)

    @property
    def safety_enabled(self) -> bool:
        stats = self._safety_stats(create=False)
        return stats.enabled if stats else False

    @safety_enabled.setter
    def safety_enabled(self, value: bool) -> None:
        self._safety_stats().enabled = bool(value)

    @property
    def safety_scored(self) -> int:
        stats = self._safety_stats(create=False)
        return stats.scored if stats else 0

    @safety_scored.setter
    def safety_scored(self, value: int) -> None:
        self._safety_stats().scored = int(value or 0)

    @property
    def safety_dropped(self) -> int:
        stats = self._safety_stats(create=False)
        return stats.dropped if stats else 0

    @safety_dropped.setter
    def safety_dropped(self, value: int) -> None:
        self._safety_stats().dropped = int(value or 0)

    @property
    def safety_errors(self) -> int:
        stats = self._safety_stats(create=False)
        return stats.errors if stats else 0

    @safety_errors.setter
    def safety_errors(self, value: int) -> None:
        self._safety_stats().errors = int(value or 0)

    @property
    def safety_flags(self) -> Dict[str, int]:
        stats = self._safety_stats(create=False)
        return stats.flags if stats else {}

    @safety_flags.setter
    def safety_flags(self, value: Mapping[str, int]) -> None:
        self._safety_stats().flags = {str(k): int(v) for k, v in value.items()}

    # ------------------------------------------------------------------
    # Observation + serialization
    # ------------------------------------------------------------------
    def observe(self, qc_result: Dict[str, Any], *, apply_gates: bool = True, screener_id: str = "quality") -> bool:
        """Update counters based on a QC row and return whether to keep it.

        Args:
            qc_result (dict[str, Any]): QC metrics for a single record.
            apply_gates (bool): Whether to apply scoring and near-duplicate
                drop rules.
            screener_id (str): Screener identifier to route stats.

        Returns:
            bool: True when the record should be kept.
        """
        screener = self._get_screener(screener_id, mode=self.mode if screener_id == "quality" else None)
        screener.enabled = True
        screener.scored += 1
        family_id = qc_result.get("dup_family_id") or qc_result.get("doc_id")
        path = qc_result.get("path")
        if family_id:
            update_dup_family_counts(self.dup_families, family_id, path)
            if self.top_dup_snapshot:
                self.top_dup_snapshot.clear()

        low_score = self._is_low_score(qc_result) if screener_id == "quality" else False
        # near_dup aggregates simhash + MinHash signals from the scorer
        near_dup = bool(qc_result.get("near_dup"))

        if low_score:
            self._increment_candidate(screener, "low_score")
        if near_dup:
            self._increment_candidate(screener, "near_dup")

        keep = True
        if apply_gates and low_score:
            self._increment_drop(screener, "low_score")
            keep = False
        elif apply_gates and self.drop_near_dups and near_dup:
            self._increment_drop(screener, "near_dup")
            keep = False

        if keep:
            screener.kept += 1
        elif apply_gates:
            screener.dropped += 1
        self._observe_signals(qc_result, screener_id=screener_id)
        return keep

    def record_error(self) -> None:
        """Increment error count for a failed QC attempt."""
        stats = self._quality_stats()
        stats.enabled = True
        stats.errors += 1

    def record_screener_error(self, screener_id: str) -> None:
        """Increment error count for an arbitrary screener."""
        stats = self._get_screener(screener_id)
        if stats is None:
            return
        stats.enabled = True
        stats.errors += 1

    def as_dict(self) -> Dict[str, Any]:
        """Return a summary dictionary suitable for serialization."""
        return {
            "enabled": bool(self.enabled),
            "mode": self.mode,
            "min_score": self.min_score,
            "drop_near_dups": bool(self.drop_near_dups),
            "scored": int(self.scored),
            "kept": int(self.kept),
            "dropped_low_score": int(self.dropped_low_score),
            "dropped_near_dup": int(self.dropped_near_dup),
            "errors": int(self.errors),
            "candidates_low_score": int(self.candidates_low_score),
            "candidates_near_dup": int(self.candidates_near_dup),
            "top_dup_families": self.top_dup_families(),
            "signal_stats": {k: s.as_dict() for k, s in self.signal_stats.items()},
            "safety": {
                "enabled": bool(self.safety_enabled),
                "scored": int(self.safety_scored),
                "dropped": int(self.safety_dropped),
                "errors": int(self.safety_errors),
                "flags": dict(self.safety_flags),
            },
            "screeners": {sid: stats.as_dict() for sid, stats in self.screeners.items()},
        }

    def merge_from_summary_dict(self, summary: Mapping[str, Any], *, replace_screeners: set[str]) -> None:
        """Merge screener stats from another summary, replacing selected ids."""
        if not summary or not replace_screeners:
            return
        other = QCSummaryTracker.from_summary_dict(summary)
        for screener_id in replace_screeners:
            incoming = other.screeners.get(screener_id)
            if incoming is None:
                continue
            self.screeners[screener_id] = incoming
            if screener_id == "quality":
                self.dup_families = dict(other.dup_families)
                self.top_dup_snapshot = list(other.top_dup_snapshot)
        self._apply_screeners_to_legacy_fields()

    def _is_low_score(self, qc_result: Dict[str, Any]) -> bool:
        """Return True when qc_result score falls below the configured min."""
        if self.min_score is None:
            return False
        score_value = qc_result.get("score")
        if score_value is None:
            return False
        try:
            return float(score_value) < float(self.min_score)
        except Exception:
            return False

    def top_dup_families(self) -> List[Dict[str, Any]]:
        """Return the largest duplicate families with cached snapshot reuse."""
        if self.top_dup_snapshot:
            return [dict(entry) for entry in self.top_dup_snapshot]
        return top_dup_families(self.dup_families)

    @classmethod
    def from_summary_dict(cls, data: Mapping[str, Any]) -> "QCSummaryTracker":
        """Rehydrate a tracker from a serialized summary dictionary.

        Args:
            data (Mapping[str, Any]): Summary produced by as_dict().

        Returns:
            QCSummaryTracker: Tracker populated with summary values.
        """
        tracker = cls(
            enabled=bool(data.get("enabled")),
            mode=data.get("mode") or QCMode.INLINE,
            min_score=data.get("min_score"),
            drop_near_dups=bool(data.get("drop_near_dups", False)),
            top_dup_snapshot=[dict(entry) for entry in data.get("top_dup_families") or [] if isinstance(entry, dict)],
        )
        screeners_payload = data.get("screeners")
        if isinstance(screeners_payload, Mapping):
            for sid, payload in screeners_payload.items():
                if not isinstance(payload, Mapping):
                    continue
                tracker.screeners[str(sid)] = ScreenerStats.from_dict(payload, default_id=str(sid))
        tracker._apply_screeners_to_legacy_fields()
        top = data.get("top_dup_families") or []
        if isinstance(top, list) and not tracker.top_dup_snapshot:
            tracker.top_dup_snapshot = [dict(entry) for entry in top if isinstance(entry, dict)]
        signal_stats = data.get("signal_stats")
        if isinstance(signal_stats, Mapping) and (not tracker.signal_stats):
            tracker._merge_signal_stats_from_dict(signal_stats, screener_id="quality")
        safety_payload = data.get("safety")
        if isinstance(safety_payload, Mapping) and "safety" not in tracker.screeners:
            tracker.safety_enabled = bool(safety_payload.get("enabled"))
            tracker.safety_scored = int(safety_payload.get("scored") or 0)
            tracker.safety_dropped = int(safety_payload.get("dropped") or 0)
            tracker.safety_errors = int(safety_payload.get("errors") or 0)
            flags = safety_payload.get("flags")
            if isinstance(flags, Mapping):
                tracker.safety_flags = {str(k): int(v) for k, v in flags.items() if v is not None}
        tracker._apply_legacy_kwargs(
            {
                "scored": data.get("scored"),
                "kept": data.get("kept"),
                "dropped_low_score": data.get("dropped_low_score"),
                "dropped_near_dup": data.get("dropped_near_dup"),
                "errors": data.get("errors"),
                "candidates_low_score": data.get("candidates_low_score"),
                "candidates_near_dup": data.get("candidates_near_dup"),
            }
        )
        return tracker

    def _observe_signals(self, qc_result: Dict[str, Any], *, screener_id: str = "quality") -> None:
        """Update scalar stats for numeric/boolean QC signals."""
        if screener_id != "quality":
            return
        try:
            _, qc_signals = filter_qc_meta(qc_result)
        except Exception:
            return
        signal_bucket = self._quality_stats().signal_stats
        for key, value in qc_signals.items():
            if value is None:
                continue
            if isinstance(value, bool):
                v = 1.0 if value else 0.0
            elif isinstance(value, (int, float)):
                v = float(value)
            else:
                continue
            stats = signal_bucket.get(key)
            if stats is None:
                stats = signal_bucket[key] = ScalarSignalStats()
            stats.observe(v)

    def observe_safety(
        self,
        result: dict[str, Any],
        *,
        apply_gates: bool,
        screener_id: str = "safety",
        mode: str = QCMode.INLINE,
    ) -> bool:
        """Update safety counters and return whether to keep the record.

        Gating is controlled by ``apply_gates``; configuration such as
        annotate_only is handled by the controller.
        """

        stats = self._get_screener(screener_id, mode=mode)
        if stats is None:
            return True
        stats.enabled = True
        stats.scored += 1

        flags = result.get("safety_flags") or {}
        if isinstance(flags, Mapping):
            for name, value in flags.items():
                if value:
                    stats.flags[name] = stats.flags.get(name, 0) + 1

        decision = (result.get("safety_decision") or "").lower()
        drop = decision == "drop" and apply_gates
        if drop:
            stats.dropped += 1
            return False
        return True

    def record_safety_error(self, *, screener_id: str = "safety", mode: str = QCMode.INLINE) -> None:
        """Increment error count for a failed safety scoring attempt."""
        stats = self._get_screener(screener_id, mode=mode)
        if stats is None:
            return
        stats.enabled = True
        stats.errors += 1

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_screener(
        self, screener_id: str, *, mode: str | None = None, create: bool = True
    ) -> ScreenerStats | None:
        stats = self.screeners.get(screener_id)
        if stats is None and create:
            stats = ScreenerStats(id=screener_id, mode=mode or QCMode.INLINE)
            self.screeners[screener_id] = stats
        return stats

    def _quality_stats(self) -> ScreenerStats:
        return self._get_screener("quality", mode=self.mode)

    def _safety_stats(self, create: bool = True) -> ScreenerStats | None:
        return self._get_screener("safety", mode=QCMode.INLINE, create=create)

    def _increment_candidate(self, screener: ScreenerStats, key: str) -> None:
        screener.candidates[key] = screener.candidates.get(key, 0) + 1

    def _increment_drop(self, screener: ScreenerStats, key: str) -> None:
        screener.drops[key] = screener.drops.get(key, 0) + 1

    def _merge_signal_stats_from_dict(self, payload: Mapping[str, Any], *, screener_id: str) -> None:
        bucket = self._get_screener(screener_id).signal_stats
        for name, stats_payload in payload.items():
            if not isinstance(stats_payload, Mapping):
                continue
            bucket[str(name)] = _parse_scalar_signal_stats(stats_payload)

    def _apply_legacy_kwargs(self, payload: Mapping[str, Any]) -> None:
        if not payload:
            return
        if "scored" in payload and payload.get("scored") is not None:
            self.scored = payload.get("scored") or 0
        if "kept" in payload and payload.get("kept") is not None:
            self.kept = payload.get("kept") or 0
        if "dropped_low_score" in payload and payload.get("dropped_low_score") is not None:
            self.dropped_low_score = payload.get("dropped_low_score") or 0
        if "dropped_near_dup" in payload and payload.get("dropped_near_dup") is not None:
            self.dropped_near_dup = payload.get("dropped_near_dup") or 0
        if "errors" in payload and payload.get("errors") is not None:
            self.errors = payload.get("errors") or 0
        if "candidates_low_score" in payload and payload.get("candidates_low_score") is not None:
            self.candidates_low_score = payload.get("candidates_low_score") or 0
        if "candidates_near_dup" in payload and payload.get("candidates_near_dup") is not None:
            self.candidates_near_dup = payload.get("candidates_near_dup") or 0

    def _apply_screeners_to_legacy_fields(self) -> None:
        quality = self.screeners.get("quality")
        if quality is not None:
            self.scored = quality.scored
            self.kept = quality.kept
            self.errors = quality.errors
            self.dropped_low_score = quality.drops.get("low_score", 0)
            self.dropped_near_dup = quality.drops.get("near_dup", 0)
            self.candidates_low_score = quality.candidates.get("low_score", 0)
            self.candidates_near_dup = quality.candidates.get("near_dup", 0)
            if quality.signal_stats and not self.signal_stats:
                self.signal_stats = quality.signal_stats
        safety = self.screeners.get("safety")
        if safety is not None:
            self.safety_enabled = safety.enabled
            self.safety_scored = safety.scored
            self.safety_dropped = safety.dropped
            self.safety_errors = safety.errors
            if safety.flags:
                self.safety_flags = safety.flags


class InlineScreeningController:
    """Coordinate one or more InlineScreener instances for a run."""

    def __init__(
        self,
        *,
        summary: QCSummaryTracker | None,
        screeners: list[InlineScreener] | None,
        logger,
        qc_cfg: QCConfig | None = None,
        safety_cfg: SafetyConfig | None = None,
    ) -> None:
        self.summary = summary or QCSummaryTracker()
        self.screeners = screeners or []
        self.logger = logger
        self.apply_qc_gates: bool | None = None
        self.apply_safety_gates: bool | None = None
        self._qc_cfg = qc_cfg
        self._safety_cfg = safety_cfg
        self._sync_screeners()

    @property
    def tracker(self) -> QCSummaryTracker:
        return self.summary

    @property
    def enforce_drops(self) -> bool:
        return any(getattr(s, "enforce_drops", False) for s in self.screeners)

    def reset(
        self,
        stats: Any | None = None,
        *,
        qc_cfg: QCConfig | None = None,
        safety_cfg: SafetyConfig | None = None,
    ) -> None:
        """Reset tracker and reattach to PipelineStats if provided."""
        effective_qc = qc_cfg if qc_cfg is not None else self._qc_cfg
        effective_safety = safety_cfg if safety_cfg is not None else self._safety_cfg
        self._qc_cfg = effective_qc
        self._safety_cfg = effective_safety
        tracker: QCSummaryTracker | None = None
        if stats is not None:
            existing = getattr(stats, "qc", None)
            if isinstance(existing, QCSummaryTracker):
                tracker = existing
        if tracker is None:
            tracker = self.summary if isinstance(self.summary, QCSummaryTracker) else QCSummaryTracker()
            if stats is not None and not isinstance(getattr(stats, "qc", None), QCSummaryTracker):
                try:
                    stats.qc = tracker  # attach summary on PipelineStats
                except Exception:
                    pass
        self.summary = tracker

        enabled = bool(effective_qc and effective_qc.enabled) or bool(effective_safety and effective_safety.enabled)
        tracker.reset_for_run(
            enabled=enabled,
            mode=effective_qc.mode if effective_qc is not None else QCMode.INLINE,
            min_score=effective_qc.min_score if effective_qc is not None else None,
            drop_near_dups=bool(effective_qc.drop_near_dups) if effective_qc is not None else False,
        )

        if effective_qc is not None:
            self.apply_qc_gates = effective_qc.enabled and effective_qc.mode == QCMode.INLINE
        else:
            self.apply_qc_gates = None
        if effective_safety is not None:
            self.apply_safety_gates = (
                effective_safety.enabled and effective_safety.mode == QCMode.INLINE and not effective_safety.annotate_only
            )
        else:
            self.apply_safety_gates = None

        self._sync_screeners()

    def process_record(self, record: Record) -> Record | None:
        """Run all screeners; drop if any screener drops."""
        current = record
        self._sync_screeners()
        kept_screeners: list[str] = []
        for screener in self.screeners:
            self._apply_gate_override(screener)
            try:
                result = screener.process_record(current)
            except Exception:
                self.logger.warning("screener %s failed", getattr(screener, "id", screener), exc_info=True)
                screener_id = getattr(screener, "id", "")
                if screener_id == "quality":
                    self.summary.record_error()
                elif screener_id == "safety":
                    self.summary.record_safety_error()
                else:
                    self.summary.record_screener_error(screener_id or "unknown")
                self._rollback_kept(kept_screeners)
                return None
            if result is None:
                self._rollback_kept(kept_screeners)
                return None
            current = result
            kept_screeners.append(getattr(screener, "id", ""))
        return current

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _apply_gate_override(self, screener: InlineScreener) -> None:
        sid = getattr(screener, "id", None)
        enforce = None
        if sid == "quality" and self.apply_qc_gates is not None:
            enforce = self.apply_qc_gates
        elif sid == "safety" and self.apply_safety_gates is not None:
            enforce = self.apply_safety_gates
        if enforce is None:
            return
        try:
            setattr(screener, "enforce_drops", enforce)
        except Exception:
            return

    def _sync_screeners(self) -> None:
        for screener in self.screeners:
            try:
                setattr(screener, "summary", self.summary)
            except Exception:
                continue

    def _rollback_kept(self, screener_ids: list[str]) -> None:
        if not screener_ids:
            return
        for sid in screener_ids:
            stats = self.summary.screeners.get(sid)
            if stats and stats.kept > 0:
                stats.kept -= 1


@dataclass(slots=True)
class QualityInlineScreener:
    id: str = "quality"
    cfg: QCConfig = field(default_factory=QCConfig)
    scorer: QualityScorer | None = None
    summary: QCSummaryTracker = field(default_factory=QCSummaryTracker)
    logger: Any | None = None
    enforce_drops: bool = True  # inline vs advisory

    def process_record(self, record: Record) -> Record | None:
        if not (self.cfg.enabled and self.scorer):
            return record  # QC disabled; no-op

        try:
            qc_result = self.scorer.score_record(record)
        except Exception as exc:
            self.summary.record_error()
            if getattr(self.cfg, "fail_on_error", False):
                raise
            if self.logger:
                path = best_effort_record_path(record)
                self.logger.warning(
                    "QC scoring failed for %s (mode=%s): %s",
                    path,
                    getattr(self.cfg, "mode", None),
                    exc,
                )
            return None if self.enforce_drops else self._mark_qc_error(record)

        keep = self.summary.observe(qc_result, apply_gates=self.enforce_drops, screener_id=self.id)
        if not keep and self.enforce_drops:
            return None

        try:
            return self._merge_qc_meta(record, qc_result)
        except Exception as exc:
            self.summary.record_error()
            if getattr(self.cfg, "fail_on_error", False):
                raise
            if self.logger:
                path = best_effort_record_path(record)
                self.logger.warning("QC post-processing failed for %s: %s", path, exc)
            return None if self.enforce_drops else self._mark_qc_error(record)

    def _merge_qc_meta(self, record: Record, qc_result: Dict[str, Any]) -> Record:
        """Attach QC-derived metadata to the record meta dictionary."""

        if not isinstance(record, dict):
            return record
        meta = ensure_meta_dict(record)
        tokens_est = qc_result.get("tokens")
        if tokens_est is not None:
            meta["approx_tokens"] = tokens_est
            meta.setdefault("tokens", tokens_est)
        canonical_qc, qc_signals = filter_qc_meta(qc_result)
        merge_meta_defaults(record, canonical_qc)
        extra = meta.get("extra")
        if not isinstance(extra, dict):
            extra = {}
            meta["extra"] = extra
        qc_extra = extra.get("qc_signals")
        if not isinstance(qc_extra, dict):
            qc_extra = {}
            extra["qc_signals"] = qc_extra
        for key, value in qc_signals.items():
            if key in qc_extra:
                continue
            qc_extra[key] = value
        return record

    def _mark_qc_error(self, record: Record) -> Record:
        """Annotate a record when QC processing fails but we keep the record."""
        if isinstance(record, dict):
            meta = ensure_meta_dict(record)
            meta["qc_error"] = True
        return record


@dataclass(slots=True)
class SafetyInlineScreener:
    id: str = "safety"
    cfg: SafetyConfig = field(default_factory=SafetyConfig)
    scorer: SafetyScorer | None = None
    summary: QCSummaryTracker = field(default_factory=QCSummaryTracker)
    logger: Any | None = None
    enforce_drops: bool = True  # whether safety is gating or annotate-only

    def process_record(self, record: Record) -> Record | None:
        if not (self.cfg.enabled and self.scorer):
            return record

        apply_gates = self.enforce_drops and not self.cfg.annotate_only

        try:
            result = self.scorer.score_record(record)
        except Exception as exc:
            self.summary.record_safety_error()
            if self.cfg.fail_on_error:
                raise
            if self.logger:
                path = best_effort_record_path(record)
                self.logger.warning("Safety scoring failed for %s: %s", path, exc)
            return record

        keep = self.summary.observe_safety(result, apply_gates=apply_gates, screener_id=self.id)
        if not keep and apply_gates:
            return None

        meta = ensure_meta_dict(record)
        _, safety_meta = filter_safety_meta(result)
        extra = meta.get("extra")
        if not isinstance(extra, dict):
            extra = {}
            meta["extra"] = extra
        safety_bucket = extra.get("safety")
        if not isinstance(safety_bucket, dict):
            safety_bucket = {}
            extra["safety"] = safety_bucket
        for key, value in safety_meta.items():
            safety_bucket.setdefault(key, value)
        return record


class InlineQCController:
    """
    Backwards-compatible wrapper exposing the generic screening layer.
    """

    def __init__(
        self,
        *,
        config: QCConfig,
        stats: Any | None,
        scorer: QualityScorer | None,
        logger,
        enforce_drops: bool = True,
        safety_scorer: SafetyScorer | None = None,
        safety_cfg: SafetyConfig | None = None,
    ) -> None:
        summary = QCSummaryTracker()
        screeners: list[InlineScreener] = []

        if config.enabled and scorer is not None and config.mode in {QCMode.INLINE, QCMode.ADVISORY}:
            qc_screener = QualityInlineScreener(
                cfg=config,
                scorer=scorer,
                summary=summary,
                logger=logger,
                enforce_drops=(config.mode == QCMode.INLINE),
            )
            screeners.append(qc_screener)

        if safety_cfg and safety_cfg.enabled and safety_scorer is not None and safety_cfg.mode in {QCMode.INLINE, QCMode.ADVISORY}:
            safety_screener = SafetyInlineScreener(
                cfg=safety_cfg,
                scorer=safety_scorer,
                summary=summary,
                logger=logger,
                enforce_drops=(safety_cfg.mode == QCMode.INLINE),
            )
            screeners.append(safety_screener)

        self._controller = InlineScreeningController(
            summary=summary,
            screeners=screeners,
            logger=logger,
            qc_cfg=config,
            safety_cfg=safety_cfg,
        )
        self.cfg = config  # used for CSV / post-QC wiring
        self.safety_cfg = safety_cfg
        self.summary = summary
        self.scorer = scorer
        self.safety_scorer = safety_scorer
        self.enforce_drops = any(getattr(s, "enforce_drops", False) for s in screeners)
        self.reset(stats)

    @property
    def tracker(self) -> QCSummaryTracker:
        return self.summary

    def reset(
        self,
        stats: Any | None = None,
        *,
        qc_cfg: QCConfig | None = None,
        safety_cfg: SafetyConfig | None = None,
    ) -> None:
        if qc_cfg is not None:
            self.cfg = qc_cfg
        if safety_cfg is not None:
            self.safety_cfg = safety_cfg
        self._controller.reset(stats, qc_cfg=self.cfg, safety_cfg=self.safety_cfg)
        self.summary = self._controller.summary

    def process_record(self, record: Record) -> Record | None:
        return self._controller.process_record(record)

    def accept(self, record: Record) -> bool:
        """Compatibility helper for record filters."""
        return self.process_record(record) is not None

    def _quality_screener(self) -> QualityInlineScreener | None:
        for screener in getattr(self._controller, "screeners", []):
            if isinstance(screener, QualityInlineScreener):
                return screener
        return None

    def _mark_qc_error(self, record: Record) -> Record:
        screener = self._quality_screener()
        if screener is not None:
            return screener._mark_qc_error(record)
        if isinstance(record, dict):
            meta = ensure_meta_dict(record)
            meta["qc_error"] = True
        return record


class InlineQCHook(RunLifecycleHook):
    """Lifecycle hook that applies inline screening and summaries."""

    def __init__(
        self,
        *,
        qc_cfg: QCConfig,
        scorer: QualityScorer | None,
        safety_cfg: SafetyConfig | None = None,
        safety_scorer: SafetyScorer | None = None,
        controller: InlineScreeningController | None = None,
    ) -> None:
        self._qc_cfg = qc_cfg
        self._safety_cfg = safety_cfg
        self._scorer = scorer
        self._safety_scorer = safety_scorer
        if controller is not None:
            self._controller = controller
        else:
            self._controller = InlineQCController(
                config=qc_cfg,
                stats=None,
                scorer=scorer,
                logger=log,
                enforce_drops=(qc_cfg.mode == QCMode.INLINE),
                safety_scorer=safety_scorer,
                safety_cfg=safety_cfg,
            )
        self._write_csv = bool(qc_cfg.write_csv)
        self._csv_suffix = qc_cfg.csv_suffix

    def on_run_start(self, ctx: RunContext) -> None:
        self._reset_controller(ctx.stats)

    def on_record(self, record: Record) -> Record | None:
        try:
            return self._controller.process_record(record)
        except Exception:
            self._controller.tracker.record_error()
            log.warning("Inline QC hook failed", exc_info=True)
            if getattr(self._controller, "enforce_drops", False):
                return None
            return self._mark_qc_error(record)

    def on_run_end(self, ctx: RunContext) -> None:
        ctx.stats.qc = self._controller.tracker
        if not self._write_csv:
            return
        try:
            self._write_csv_report(ctx)
        except Exception:  # pragma: no cover - best-effort logging
            log.warning("Failed to write inline QC CSV", exc_info=True)

    def on_artifacts(self, artifacts: RunArtifacts, ctx: RunContext) -> None:
        # Inline QC doesn't need run-level artifacts; all work is handled via
        # per-record processing and run-end summary propagation.
        return None

    def _write_csv_report(self, ctx: RunContext) -> None:
        jsonl_path = ctx.cfg.sinks.primary_jsonl_name or ctx.cfg.metadata.primary_jsonl
        if not jsonl_path:
            return
        out_csv = _derive_csv_path(jsonl_path, self._csv_suffix)
        if not out_csv:
            return
        scorer = self._resolve_quality_scorer()
        if scorer is None:
            return

        reset = getattr(scorer, "reset_state", None)
        if callable(reset):
            try:
                reset()
            except Exception:
                log.debug("QC scorer reset_state failed; continuing", exc_info=True)

        cfg = self._qc_cfg or getattr(self._controller, "cfg", None)
        if cfg is None:
            return
        tracker = QCSummaryTracker(
            enabled=True,
            mode=cfg.mode,
            min_score=cfg.min_score,
            drop_near_dups=bool(cfg.drop_near_dups),
        )
        from .qc_post import collect_qc_rows_from_jsonl, emit_qc_csv

        rows = collect_qc_rows_from_jsonl(
            str(jsonl_path),
            qc_cfg=self._controller.cfg,
            config=ctx.cfg,
            scorer=scorer,
            runtime=getattr(ctx, "runtime", None),
            executor_hint=None,
            tracker=tracker,
        )
        emit_qc_csv(rows, str(jsonl_path), out_csv)

        err_count = tracker.errors
        if err_count and hasattr(ctx, "stats"):
            try:
                ctx.stats.qc.errors += err_count  # type: ignore[attr-defined]
            except Exception:
                pass
            log.warning("Inline QC CSV scoring for %s skipped %s lines", jsonl_path, err_count)

    def _reset_controller(self, stats: Any | None) -> None:
        reset_fn = getattr(self._controller, "reset", None)
        if not callable(reset_fn):
            return
        try:
            reset_fn(stats, qc_cfg=self._qc_cfg, safety_cfg=self._safety_cfg)
        except TypeError:
            reset_fn(stats)
        self._qc_cfg = getattr(self._controller, "cfg", self._qc_cfg)
        try:
            self._safety_cfg = getattr(self._controller, "safety_cfg")
        except Exception:
            pass

    def _resolve_quality_scorer(self) -> QualityScorer | None:
        scorer = getattr(self._controller, "scorer", None)
        if scorer is not None:
            return scorer
        if self._scorer is not None:
            return self._scorer
        for screener in getattr(self._controller, "screeners", []):
            if getattr(screener, "id", None) == "quality":
                candidate = getattr(screener, "scorer", None)
                if candidate is not None:
                    return candidate
        return None

    def _mark_qc_error(self, record: Record) -> Record:
        mark = getattr(self._controller, "_mark_qc_error", None)
        if callable(mark):
            return mark(record)
        for screener in getattr(self._controller, "screeners", []):
            helper = getattr(screener, "_mark_qc_error", None)
            if callable(helper):
                return helper(record)
        if isinstance(record, dict):
            meta = ensure_meta_dict(record)
            meta["qc_error"] = True
        return record


def _derive_csv_path(jsonl_path: Optional[str], suffix: Optional[str]) -> Optional[str]:
    """Derive a QC CSV path from a primary JSONL path and suffix."""

    if not jsonl_path:
        return None
    if suffix and ((os.sep and os.sep in suffix) or (os.altsep and os.altsep in suffix)):
        return suffix
    suffix = suffix or "_quality.csv"
    base = str(jsonl_path)
    if base.endswith(".jsonl"):
        base = base[:-6]
    return base + suffix


def summarize_qc_rows(
    rows: Iterable[Dict[str, Any]],
    *,
    mode: str,
    min_score: Optional[float],
    drop_near_dups: bool,
    apply_gates: bool = False,
    enabled: bool = True,
) -> Dict[str, Any]:
    """Build a summary dictionary from QC rows for post-processing mode.

    Args:
        rows (Iterable[dict[str, Any]]): QC rows to aggregate.
        mode (str): QC mode label to store.
        min_score (float | None): Minimum score threshold.
        drop_near_dups (bool): Whether to drop near duplicates.
        apply_gates (bool): Whether to apply gating during aggregation.
        enabled (bool): Whether QC was enabled for the run.

    Returns:
        dict[str, Any]: Summary produced by QCSummaryTracker.as_dict().
    """
    tracker = QCSummaryTracker(
        enabled=enabled,
        mode=mode,
        min_score=min_score,
        drop_near_dups=drop_near_dups,
    )
    for row in rows:
        tracker.observe(row, apply_gates=apply_gates)
    return tracker.as_dict()
