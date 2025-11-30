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
    scored: int = 0
    kept: int = 0
    dropped_low_score: int = 0
    dropped_near_dup: int = 0
    errors: int = 0
    candidates_low_score: int = 0
    candidates_near_dup: int = 0
    dup_families: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    top_dup_snapshot: List[Dict[str, Any]] = field(default_factory=list)
    signal_stats: Dict[str, ScalarSignalStats] = field(default_factory=dict)
    safety_enabled: bool = False
    safety_scored: int = 0
    safety_dropped: int = 0
    safety_errors: int = 0
    safety_flags: Dict[str, int] = field(default_factory=dict)

    def observe(self, qc_result: Dict[str, Any], *, apply_gates: bool = True) -> bool:
        """Update counters based on a QC row and return whether to keep it.

        Args:
            qc_result (dict[str, Any]): QC metrics for a single record.
            apply_gates (bool): Whether to apply scoring and near-duplicate
                drop rules.

        Returns:
            bool: True when the record should be kept.
        """
        self.scored += 1
        family_id = qc_result.get("dup_family_id") or qc_result.get("doc_id")
        path = qc_result.get("path")
        if family_id:
            update_dup_family_counts(self.dup_families, family_id, path)
            if self.top_dup_snapshot:
                self.top_dup_snapshot.clear()

        low_score = self._is_low_score(qc_result)
        # near_dup aggregates simhash + MinHash signals from the scorer
        near_dup = bool(qc_result.get("near_dup"))

        if low_score:
            self.candidates_low_score += 1
        if near_dup:
            self.candidates_near_dup += 1

        keep = True
        if apply_gates and low_score:
            self.dropped_low_score += 1
            keep = False
        elif apply_gates and self.drop_near_dups and near_dup:
            self.dropped_near_dup += 1
            keep = False

        if keep:
            self.kept += 1
        self._observe_signals(qc_result)
        return keep

    def record_error(self) -> None:
        """Increment error count for a failed QC attempt."""
        self.errors += 1

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
        }

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
        tracker = cls()
        tracker.enabled = bool(data.get("enabled"))
        mode = data.get("mode")
        if isinstance(mode, str) and mode:
            tracker.mode = mode
        tracker.scored = int(data.get("scored") or 0)
        tracker.kept = int(data.get("kept") or 0)
        tracker.dropped_low_score = int(data.get("dropped_low_score") or 0)
        tracker.dropped_near_dup = int(data.get("dropped_near_dup") or 0)
        tracker.errors = int(data.get("errors") or 0)
        tracker.candidates_low_score = int(data.get("candidates_low_score") or 0)
        tracker.candidates_near_dup = int(data.get("candidates_near_dup") or 0)
        tracker.drop_near_dups = bool(data.get("drop_near_dups", tracker.drop_near_dups))
        tracker.min_score = data.get("min_score", tracker.min_score)
        top = data.get("top_dup_families") or []
        if isinstance(top, list):
            tracker.top_dup_snapshot = [dict(entry) for entry in top if isinstance(entry, dict)]
        signal_stats = data.get("signal_stats")
        if isinstance(signal_stats, Mapping):
            parsed_stats: Dict[str, ScalarSignalStats] = {}
            for name, payload in signal_stats.items():
                if not isinstance(payload, Mapping):
                    continue
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
                parsed_stats[name] = stats
            tracker.signal_stats = parsed_stats
        safety_payload = data.get("safety")
        if isinstance(safety_payload, Mapping):
            tracker.safety_enabled = bool(safety_payload.get("enabled"))
            tracker.safety_scored = int(safety_payload.get("scored") or 0)
            tracker.safety_dropped = int(safety_payload.get("dropped") or 0)
            tracker.safety_errors = int(safety_payload.get("errors") or 0)
            flags = safety_payload.get("flags")
            if isinstance(flags, Mapping):
                tracker.safety_flags = {str(k): int(v) for k, v in flags.items() if v is not None}
        return tracker

    def _observe_signals(self, qc_result: Dict[str, Any]) -> None:
        """Update scalar stats for numeric/boolean QC signals."""
        try:
            _, qc_signals = filter_qc_meta(qc_result)
        except Exception:
            return
        for key, value in qc_signals.items():
            if value is None:
                continue
            if isinstance(value, bool):
                v = 1.0 if value else 0.0
            elif isinstance(value, (int, float)):
                v = float(value)
            else:
                continue
            stats = self.signal_stats.get(key)
            if stats is None:
                stats = self.signal_stats[key] = ScalarSignalStats()
            stats.observe(v)

    def observe_safety(self, result: dict[str, Any], *, apply_gates: bool) -> bool:
        """Update safety counters and return whether to keep the record.

        Gating is controlled by ``apply_gates``; configuration such as
        annotate_only is handled by the controller.
        """

        self.safety_enabled = True
        self.safety_scored += 1

        flags = result.get("safety_flags") or {}
        if isinstance(flags, Mapping):
            for name, value in flags.items():
                if value:
                    self.safety_flags[name] = self.safety_flags.get(name, 0) + 1

        decision = (result.get("safety_decision") or "").lower()
        drop = decision == "drop" and apply_gates
        if drop:
            self.safety_dropped += 1
            return False
        return True

    def record_safety_error(self) -> None:
        """Increment error count for a failed safety scoring attempt."""
        self.safety_enabled = True
        self.safety_errors += 1


class InlineScreeningController:
    """Coordinate one or more InlineScreener instances for a run."""

    def __init__(self, *, summary: QCSummaryTracker, screeners: list[InlineScreener], logger) -> None:
        self.summary = summary
        self.screeners = screeners
        self.logger = logger

    def reset(self, stats: Any | None = None, *, qc_cfg: QCConfig | None = None) -> None:
        """Reset tracker and reattach to PipelineStats if provided."""
        if qc_cfg is not None:
            tracker = QCSummaryTracker(
                enabled=qc_cfg.enabled,
                mode=qc_cfg.mode,
                min_score=qc_cfg.min_score,
                drop_near_dups=bool(qc_cfg.drop_near_dups),
            )
        else:
            tracker = QCSummaryTracker()
        self.summary = tracker
        for screener in self.screeners:
            try:
                setattr(screener, "summary", tracker)
            except Exception:
                pass
        if stats is not None:
            try:
                stats.qc = tracker  # screening summary on PipelineStats
            except Exception:
                pass

    def process_record(self, record: Record) -> Record | None:
        """Run all screeners; drop if any screener drops."""
        current = record
        for screener in self.screeners:
            try:
                result = screener.process_record(current)
            except Exception:
                self.logger.warning("screener %s failed", getattr(screener, "id", screener), exc_info=True)
                return None
            if result is None:
                return None
            current = result
        return current


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

        keep = self.summary.observe(qc_result, apply_gates=self.enforce_drops)
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

        keep = self.summary.observe_safety(result, apply_gates=apply_gates)
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

        self._controller = InlineScreeningController(summary=summary, screeners=screeners, logger=logger)
        self.cfg = config  # used for CSV / post-QC wiring
        self.summary = summary
        self.scorer = scorer
        self.safety_scorer = safety_scorer
        self.enforce_drops = any(getattr(s, "enforce_drops", False) for s in screeners)
        self.reset(stats)

    @property
    def tracker(self) -> QCSummaryTracker:
        return self.summary

    def reset(self, stats: Any | None = None) -> None:
        self._controller.reset(stats, qc_cfg=self.cfg)
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
    ) -> None:
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
        self._controller.reset(ctx.stats)

    def on_record(self, record: Record) -> Record | None:
        try:
            return self._controller.process_record(record)
        except Exception:
            self._controller.tracker.record_error()
            log.warning("Inline QC hook failed", exc_info=True)
            if self._controller.enforce_drops:
                return None
            return self._controller._mark_qc_error(record)

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
        scorer = getattr(self._controller, "scorer", None)
        if scorer is None:
            return

        reset = getattr(scorer, "reset_state", None)
        if callable(reset):
            try:
                reset()
            except Exception:
                log.debug("QC scorer reset_state failed; continuing", exc_info=True)

        tracker = QCSummaryTracker(
            enabled=True,
            mode=self._controller.cfg.mode,
            min_score=self._controller.cfg.min_score,
            drop_near_dups=bool(self._controller.cfg.drop_near_dups),
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
