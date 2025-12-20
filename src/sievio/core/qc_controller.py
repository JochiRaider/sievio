# qc_controller.py
# SPDX-License-Identifier: MIT
"""Helpers for inline QC execution and summary building."""
from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .config import QCConfig, QCMode, SafetyConfig
from .interfaces import (
    InlineScreener,
    QualityScorer,
    Record,
    RunArtifacts,
    RunContext,
    RunLifecycleHook,
    SafetyScorer,
)
from .log import get_logger
from .qc_utils import top_dup_families, update_dup_family_counts
from .records import (
    best_effort_record_path,
    ensure_meta_dict,
    filter_qc_meta,
    filter_safety_meta,
    merge_meta_defaults,
)

log = get_logger(__name__)
_SUMMARY_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ScreenDecision:
    """Decision output from a screening policy."""

    candidates: tuple[str, ...] = ()
    # Reasons that would trigger a drop if gates are applied.
    would_drop: tuple[str, ...] = ()


@dataclass(slots=True)
class QualityDecisionPolicy:
    """Policy for translating QC results into gating-agnostic decisions."""

    def decide(self, qc_result: Mapping[str, Any], *, cfg: QCConfig) -> ScreenDecision:
        low_score = False
        if cfg.min_score is not None:
            score_val = qc_result.get("score")
            if score_val is not None:
                try:
                    low_score = float(score_val) < float(cfg.min_score)
                except (TypeError, ValueError):
                    low_score = False
        near_dup = bool(qc_result.get("near_dup"))

        candidates: list[str] = []
        if low_score:
            candidates.append("low_score")
        if near_dup:
            candidates.append("near_dup")

        would_drop: list[str] = []
        if low_score:
            would_drop.append("low_score")
        if cfg.drop_near_dups and near_dup:
            would_drop.append("near_dup")

        return ScreenDecision(candidates=tuple(candidates), would_drop=tuple(would_drop))


@dataclass(slots=True)
class SafetyDecisionPolicy:
    """Policy for translating safety results into gating-agnostic decisions."""

    def decide(self, result: Mapping[str, Any], *, cfg: SafetyConfig | None) -> ScreenDecision:
        decision = (result.get("safety_decision") or "").lower()
        candidates: tuple[str, ...] = ()
        would_drop: tuple[str, ...] = ()
        if decision == "drop":
            candidates = ("drop",)
            would_drop = ("drop",)
        return ScreenDecision(candidates=candidates, would_drop=would_drop)


def _coerce_bool(value: Any, *, default: bool = False, strict: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        val = value.strip().lower()
        if val in {"true", "1", "yes"}:
            return True
        if val in {"false", "0", "no"}:
            return False
    if strict:
        raise TypeError(f"Expected bool-like value, got {type(value).__name__}")
    return default


def _coerce_int(value: Any, *, default: int = 0, strict: bool = False) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        if strict:
            raise
        return default


def _coerce_float(
    value: Any,
    *,
    default: float | None = None,
    strict: bool = False,
) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        if strict:
            raise
        return default


def _coerce_str(value: Any, *, default: str = "", strict: bool = False) -> str:
    if value is None:
        return default
    try:
        return str(value)
    except Exception:
        if strict:
            raise
        return default


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

    def merge_from(self, other: ScalarSignalStats) -> None:
        if other.count <= 0:
            return
        if self.count == 0:
            self.count = other.count
            self.sum = other.sum
            self.sum_sq = other.sum_sq
            self.min = other.min
            self.max = other.max
            return
        self.count += other.count
        self.sum += other.sum
        self.sum_sq += other.sum_sq
        if other.min is not None:
            self.min = other.min if self.min is None or other.min < self.min else self.min
        if other.max is not None:
            self.max = other.max if self.max is None or other.max > self.max else self.max

    def clone(self) -> ScalarSignalStats:
        stats = ScalarSignalStats()
        stats.count = self.count
        stats.sum = self.sum
        stats.sum_sq = self.sum_sq
        stats.min = self.min
        stats.max = self.max
        return stats

    def as_dict(self) -> dict[str, Any]:
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
    """Per-screener summary used by QCSummaryTracker.

    drops counts track would-drop reasons (independent of whether gates were applied).
    """

    id: str
    enabled: bool = False
    mode: str = QCMode.INLINE
    scored: int = 0
    kept: int = 0
    dropped: int = 0
    errors: int = 0
    signal_stats: dict[str, ScalarSignalStats] = field(default_factory=dict)
    flags: dict[str, int] = field(default_factory=dict)
    candidates: dict[str, int] = field(default_factory=dict)
    drops: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
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
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        default_id: str | None = None,
        strict: bool = False,
    ) -> ScreenerStats:
        sid = _coerce_str(data.get("id") or default_id or "", strict=strict)
        stats = cls(id=sid)
        stats.enabled = _coerce_bool(data.get("enabled"), strict=strict)
        mode = data.get("mode")
        if isinstance(mode, str) and mode:
            stats.mode = mode
        stats.scored = _coerce_int(data.get("scored"), strict=strict)
        stats.kept = _coerce_int(data.get("kept"), strict=strict)
        stats.dropped = _coerce_int(data.get("dropped"), strict=strict)
        stats.errors = _coerce_int(data.get("errors"), strict=strict)
        signals = data.get("signal_stats")
        if isinstance(signals, Mapping):
            parsed: dict[str, ScalarSignalStats] = {}
            for name, payload in signals.items():
                if not isinstance(payload, Mapping):
                    continue
                parsed[str(name)] = _parse_scalar_signal_stats(payload, strict=strict)
            stats.signal_stats = parsed
        flags = data.get("flags")
        if isinstance(flags, Mapping):
            stats.flags = {
                str(k): _coerce_int(v, strict=strict)
                for k, v in flags.items()
                if v is not None
            }
        candidates = data.get("candidates")
        if isinstance(candidates, Mapping):
            stats.candidates = {
                str(k): _coerce_int(v, strict=strict)
                for k, v in candidates.items()
                if v is not None
            }
        drops = data.get("drops")
        if isinstance(drops, Mapping):
            stats.drops = {
                str(k): _coerce_int(v, strict=strict)
                for k, v in drops.items()
                if v is not None
            }
        return stats

    def clone(self) -> ScreenerStats:
        stats = ScreenerStats(id=self.id)
        stats.enabled = self.enabled
        stats.mode = self.mode
        stats.scored = self.scored
        stats.kept = self.kept
        stats.dropped = self.dropped
        stats.errors = self.errors
        stats.signal_stats = {k: v.clone() for k, v in self.signal_stats.items()}
        stats.flags = dict(self.flags)
        stats.candidates = dict(self.candidates)
        stats.drops = dict(self.drops)
        return stats

    def merge_from(self, other: ScreenerStats) -> None:
        self.enabled = self.enabled or other.enabled
        if other.mode:
            self.mode = other.mode
        self.scored += other.scored
        self.kept += other.kept
        self.dropped += other.dropped
        self.errors += other.errors
        for name, stats in other.signal_stats.items():
            existing = self.signal_stats.get(name)
            if existing is None:
                self.signal_stats[name] = stats.clone()
            else:
                existing.merge_from(stats)
        for bucket, incoming in (
            ("flags", other.flags),
            ("candidates", other.candidates),
            ("drops", other.drops),
        ):
            target = getattr(self, bucket)
            for key, value in incoming.items():
                target[key] = target.get(key, 0) + int(value)


def _parse_scalar_signal_stats(
    payload: Mapping[str, Any],
    *,
    strict: bool = False,
) -> ScalarSignalStats:
    stats = ScalarSignalStats()
    stats.count = _coerce_int(payload.get("count"), strict=strict)
    stats.min = _coerce_float(payload.get("min"), strict=strict)
    stats.max = _coerce_float(payload.get("max"), strict=strict)
    mean_val = payload.get("mean")
    stdev_val = payload.get("stdev")
    mean = _coerce_float(mean_val, strict=strict)
    stdev = _coerce_float(stdev_val, strict=strict)
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
    counts/examples for post-QC reporting. Decisions are computed by screeners
    (gating-agnostic policies + explicit gate application), leaving this tracker
    to aggregate results; for concurrency, use one tracker per worker and merge
    after processing (not thread-safe for concurrent mutation).
    """
    enabled: bool = False
    mode: str = QCMode.INLINE
    min_score: float | None = None
    drop_near_dups: bool = False
    dup_families: dict[str, dict[str, Any]] = field(default_factory=dict)
    top_dup_snapshot: list[dict[str, Any]] = field(default_factory=list)
    screeners: dict[str, ScreenerStats] = field(default_factory=dict)

    def __init__(
        self,
        *,
        enabled: bool = False,
        mode: str = QCMode.INLINE,
        min_score: float | None = None,
        drop_near_dups: bool = False,
        screeners: Mapping[str, ScreenerStats | Mapping[str, Any]] | None = None,
        dup_families: Mapping[str, dict[str, Any]] | None = None,
        top_dup_snapshot: list[dict[str, Any]] | None = None,
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

    def reset_for_run(
        self,
        *,
        enabled: bool = False,
        mode: str = QCMode.INLINE,
        min_score: float | None = None,
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
    # Observation + serialization
    # ------------------------------------------------------------------
    def observe_quality(
        self,
        qc_result: Mapping[str, Any],
        decision: ScreenDecision,
        *,
        did_drop: bool,
        screener_id: str = "quality",
    ) -> None:
        """Update counters based on a QC row and explicit gate outcome."""
        screener = self.get_screener(
            screener_id,
            mode=self.mode if screener_id == "quality" else None,
        )
        screener.enabled = True
        screener.scored += 1
        family_id = qc_result.get("dup_family_id") or qc_result.get("doc_id")
        path = qc_result.get("path")
        if family_id:
            update_dup_family_counts(self.dup_families, family_id, path)
            if self.top_dup_snapshot:
                self.top_dup_snapshot.clear()

        for reason in decision.candidates:
            self._increment_candidate(screener, reason)
        for reason in decision.would_drop:
            self._increment_drop(screener, reason)

        if did_drop:
            screener.dropped += 1
        else:
            screener.kept += 1
        self._observe_signals(qc_result, screener_id=screener_id)

    def record_error(self) -> None:
        """Increment error count for a failed QC attempt."""
        stats = self._quality_stats()
        stats.enabled = True
        stats.errors += 1

    def record_screener_error(self, screener_id: str) -> None:
        """Increment error count for an arbitrary screener."""
        stats = self.get_screener(screener_id)
        if stats is None:
            return
        stats.enabled = True
        stats.errors += 1

    def as_dict(self) -> dict[str, Any]:
        """Return a summary dictionary suitable for serialization."""
        return {
            "schema_version": _SUMMARY_SCHEMA_VERSION,
            "enabled": bool(self.enabled),
            "mode": self.mode,
            "min_score": self.min_score,
            "drop_near_dups": bool(self.drop_near_dups),
            "top_dup_families": self.top_dup_families(),
            "screeners": {sid: stats.as_dict() for sid, stats in self.screeners.items()},
        }

    def merge_from_summary_dict(
        self,
        summary: Mapping[str, Any],
        *,
        replace_screeners: set[str],
        strict: bool = False,
    ) -> None:
        """Merge screener stats from another summary, replacing selected ids."""
        if not summary or not replace_screeners:
            return
        other = QCSummaryTracker.from_summary_dict(summary, strict=strict)
        self.merge(other, replace_screeners=replace_screeners)

    def _is_low_score(self, qc_result: dict[str, Any]) -> bool:
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

    def top_dup_families(self) -> list[dict[str, Any]]:
        """Return the largest duplicate families with cached snapshot reuse."""
        if self.top_dup_snapshot:
            return [dict(entry) for entry in self.top_dup_snapshot]
        return top_dup_families(self.dup_families)

    @classmethod
    def from_summary_dict(
        cls,
        data: Mapping[str, Any],
        *,
        strict: bool = False,
    ) -> QCSummaryTracker:
        """Rehydrate a tracker from a serialized summary dictionary.

        Args:
            data (Mapping[str, Any]): Summary produced by as_dict().

        Returns:
            QCSummaryTracker: Tracker populated with summary values.
        """
        schema_version = data.get("schema_version")
        if schema_version is not None:
            parsed_version = _coerce_int(schema_version, strict=strict)
            if strict and parsed_version > _SUMMARY_SCHEMA_VERSION:
                raise ValueError(f"Unsupported QC summary schema version {parsed_version}.")

        tracker = cls(
            enabled=_coerce_bool(data.get("enabled"), strict=strict),
            mode=data.get("mode") or QCMode.INLINE,
            min_score=_coerce_float(data.get("min_score"), default=None, strict=strict),
            drop_near_dups=_coerce_bool(data.get("drop_near_dups", False), strict=strict),
            top_dup_snapshot=[
                dict(entry)
                for entry in data.get("top_dup_families") or []
                if isinstance(entry, dict)
            ],
        )
        screeners_payload = data.get("screeners")
        if isinstance(screeners_payload, Mapping):
            for sid, payload in screeners_payload.items():
                if not isinstance(payload, Mapping):
                    continue
                tracker.screeners[str(sid)] = ScreenerStats.from_dict(
                    payload,
                    default_id=str(sid),
                    strict=strict,
                )
        return tracker

    def _observe_signals(
        self,
        qc_result: Mapping[str, Any],
        *,
        screener_id: str = "quality",
    ) -> None:
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
        result: Mapping[str, Any],
        decision: ScreenDecision,
        *,
        did_drop: bool,
        screener_id: str = "safety",
        mode: str = QCMode.INLINE,
    ) -> None:
        """Update safety counters with an explicit decision and drop result."""
        stats = self.get_screener(screener_id, mode=mode)
        if stats is None:
            return
        stats.enabled = True
        stats.scored += 1

        flags = result.get("safety_flags") or {}
        if isinstance(flags, Mapping):
            for name, value in flags.items():
                if value:
                    stats.flags[name] = stats.flags.get(name, 0) + 1

        for reason in decision.candidates:
            self._increment_candidate(stats, reason)
        for reason in decision.would_drop:
            self._increment_drop(stats, reason)

        if did_drop:
            stats.dropped += 1
        else:
            stats.kept += 1

    def record_safety_error(
        self,
        *,
        screener_id: str = "safety",
        mode: str = QCMode.INLINE,
    ) -> None:
        """Increment error count for a failed safety scoring attempt."""
        stats = self.get_screener(screener_id, mode=mode)
        if stats is None:
            return
        stats.enabled = True
        stats.errors += 1

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def get_screener(
        self, screener_id: str, *, mode: str | None = None, create: bool = True
    ) -> ScreenerStats | None:
        """Return stats for a screener, optionally creating a new bucket."""
        stats = self.screeners.get(screener_id)
        if stats is None and create:
            stats = ScreenerStats(id=screener_id, mode=mode or QCMode.INLINE)
            self.screeners[screener_id] = stats
        return stats

    def _quality_stats(self) -> ScreenerStats:
        stats = self.get_screener("quality", mode=self.mode)
        assert stats is not None
        return stats

    def _safety_stats(self, create: bool = True) -> ScreenerStats | None:
        return self.get_screener("safety", mode=QCMode.INLINE, create=create)

    def _increment_candidate(self, screener: ScreenerStats, key: str) -> None:
        screener.candidates[key] = screener.candidates.get(key, 0) + 1

    def _increment_drop(self, screener: ScreenerStats, key: str) -> None:
        screener.drops[key] = screener.drops.get(key, 0) + 1

    def merge(self, other: QCSummaryTracker, *, replace_screeners: set[str] | None = None) -> None:
        """Merge another tracker into this instance."""
        if other is self:
            return
        self.enabled = self.enabled or other.enabled
        if other.mode:
            self.mode = other.mode
        if other.min_score is not None:
            self.min_score = other.min_score
        self.drop_near_dups = self.drop_near_dups or other.drop_near_dups
        replace_screeners = replace_screeners or set()
        for screener_id, incoming in other.screeners.items():
            if screener_id in replace_screeners:
                self.screeners[screener_id] = incoming.clone()
                if screener_id == "quality":
                    self.dup_families = dict(other.dup_families)
                    self.top_dup_snapshot = list(other.top_dup_snapshot)
                continue
            existing = self.screeners.get(screener_id)
            if existing is None:
                self.screeners[screener_id] = incoming.clone()
            else:
                existing.merge_from(incoming)

        if "quality" not in replace_screeners and other.dup_families:
            self._merge_dup_families(other.dup_families)

    def _merge_dup_families(self, other: Mapping[str, Mapping[str, Any]]) -> None:
        changed = False
        for family_id, payload in other.items():
            if not family_id:
                continue
            entry = self.dup_families.setdefault(family_id, {"count": 0, "examples": []})
            entry["count"] = _coerce_int(entry.get("count")) + _coerce_int(payload.get("count"))
            examples = entry.get("examples")
            if not isinstance(examples, list):
                examples = []
                entry["examples"] = examples
            incoming_examples = payload.get("examples") or []
            if isinstance(incoming_examples, list):
                for example in incoming_examples:
                    if len(examples) >= 3:
                        break
                    if example not in examples:
                        examples.append(example)
            changed = True
        if changed and self.top_dup_snapshot:
            self.top_dup_snapshot.clear()

    @property
    def safety_scored(self) -> int:
        stats = self._safety_stats(create=False)
        return stats.scored if stats else 0

    @property
    def safety_dropped(self) -> int:
        stats = self._safety_stats(create=False)
        return stats.dropped if stats else 0


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
    def cfg(self) -> QCConfig | None:
        """Expose the current QCConfig for downstream helpers (CSV/post-QC)."""
        return self._qc_cfg

    @cfg.setter
    def cfg(self, value: QCConfig | None) -> None:
        self._qc_cfg = value

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
            tracker = (
                self.summary
                if isinstance(self.summary, QCSummaryTracker)
                else QCSummaryTracker()
            )
            if stats is not None and not isinstance(getattr(stats, "qc", None), QCSummaryTracker):
                try:
                    stats.qc = tracker  # attach summary on PipelineStats
                except Exception:
                    pass
        self.summary = tracker

        enabled = (
            bool(effective_qc and effective_qc.enabled)
            or bool(effective_safety and effective_safety.enabled)
        )
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
                effective_safety.enabled
                and effective_safety.mode == QCMode.INLINE
                and not effective_safety.annotate_only
            )
        else:
            self.apply_safety_gates = None

        self._sync_screeners()

    def process_record(self, record: Record) -> Record | None:
        """Run all screeners; drop if any screener drops.

        Per-screener kept reflects local pass; downstream drops do not roll back.
        """
        current = record
        self._sync_screeners()
        for screener in self.screeners:
            self._apply_gate_override(screener)
            try:
                result = screener.process_record(current)
            except Exception:
                self.logger.warning(
                    "screener %s failed",
                    getattr(screener, "id", screener),
                    exc_info=True,
                )
                screener_id = getattr(screener, "id", "")
                enforce = getattr(screener, "enforce_drops", True)
                if enforce is None:
                    enforce = True
                enforce = bool(enforce)
                if screener_id == "quality":
                    self.summary.record_error()
                elif screener_id == "safety":
                    self.summary.record_safety_error()
                else:
                    self.summary.record_screener_error(screener_id or "unknown")
                if enforce:
                    return None
                # advisory screeners should not drop; continue with current record
                continue
            if result is None:
                return None
            current = result
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
            screener.enforce_drops = enforce
        except Exception:
            return

    def _sync_screeners(self) -> None:
        for screener in self.screeners:
            try:
                screener.summary = self.summary
            except Exception:
                continue

@dataclass(slots=True)
class QualityInlineScreener:
    id: str = "quality"
    cfg: QCConfig = field(default_factory=QCConfig)
    scorer: QualityScorer | None = None
    summary: QCSummaryTracker = field(default_factory=QCSummaryTracker)
    decision_policy: QualityDecisionPolicy = field(default_factory=QualityDecisionPolicy)
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

        decision = self.decision_policy.decide(qc_result, cfg=self.cfg)
        did_drop = self.enforce_drops and bool(decision.would_drop)
        self.summary.observe_quality(qc_result, decision, did_drop=did_drop, screener_id=self.id)
        if did_drop:
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

    def _merge_qc_meta(self, record: Record, qc_result: dict[str, Any]) -> Record:
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
    decision_policy: SafetyDecisionPolicy = field(default_factory=SafetyDecisionPolicy)
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

        decision = self.decision_policy.decide(result, cfg=self.cfg)
        did_drop = apply_gates and bool(decision.would_drop)
        self.summary.observe_safety(
            result,
            decision,
            did_drop=did_drop,
            screener_id=self.id,
            mode=self.cfg.mode,
        )
        if did_drop:
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

        if (
            config.enabled
            and scorer is not None
            and config.mode in {QCMode.INLINE, QCMode.ADVISORY}
        ):
            qc_screener = QualityInlineScreener(
                cfg=config,
                scorer=scorer,
                summary=summary,
                logger=logger,
                enforce_drops=(config.mode == QCMode.INLINE),
            )
            screeners.append(qc_screener)

        if (
            safety_cfg
            and safety_cfg.enabled
            and safety_scorer is not None
            and safety_cfg.mode in {QCMode.INLINE, QCMode.ADVISORY}
        ):
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

        quality_stats = tracker.get_screener("quality", create=False)
        err_count = quality_stats.errors if quality_stats else 0
        if err_count and hasattr(ctx, "stats"):
            try:
                target_stats = getattr(ctx.stats, "qc", None)
                target_quality = target_stats.get_screener("quality") if target_stats else None  # type: ignore[attr-defined]
                if target_quality is not None:
                    target_quality.errors += err_count
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
            self._safety_cfg = self._controller.safety_cfg
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


def _derive_csv_path(jsonl_path: str | None, suffix: str | None) -> str | None:
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
    rows: Iterable[dict[str, Any]],
    *,
    mode: str,
    min_score: float | None,
    drop_near_dups: bool,
    apply_gates: bool = False,
    enabled: bool = True,
) -> dict[str, Any]:
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
    policy = QualityDecisionPolicy()
    cfg = QCConfig(min_score=min_score, drop_near_dups=bool(drop_near_dups))
    for row in rows:
        decision = policy.decide(row, cfg=cfg)
        did_drop = apply_gates and bool(decision.would_drop)
        tracker.observe_quality(row, decision, did_drop=did_drop)
    return tracker.as_dict()
