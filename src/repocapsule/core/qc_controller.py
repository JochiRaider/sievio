# qc_controller.py
# SPDX-License-Identifier: MIT
"""Helpers for inline QC execution and summary building."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .config import QCMode
from .interfaces import QualityScorer, Record
from .records import ensure_meta_dict, merge_meta_defaults, best_effort_record_path, filter_qc_meta
from .qc_utils import update_dup_family_counts, top_dup_families


@dataclass(slots=True)
class QCSummaryTracker:
    """Track QC scoring outcomes and duplicate families.

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
        return tracker


class InlineQCController:
    """Wrap scorer and gating logic used by inline QC."""

    def __init__(
        self,
        *,
        config,
        stats,
        scorer: QualityScorer,
        logger,
        enforce_drops: bool = True,
    ) -> None:
        """Initialize the controller with scorer and configuration.

        Args:
            config: QC configuration object.
            stats: Pipeline stats object containing a QC tracker.
            scorer (QualityScorer): Scorer used to evaluate records.
            logger: Logger for warning and error messages.
            enforce_drops (bool): Whether to drop records based on QC results.
        """
        self.cfg = config
        self.stats = stats
        self.scorer = scorer
        self.logger = logger
        self.enforce_drops = enforce_drops
        tracker = getattr(stats, "qc", None)
        if not isinstance(tracker, QCSummaryTracker):
            tracker = QCSummaryTracker()
            stats.qc = tracker  # type: ignore[attr-defined]
        tracker.enabled = True
        tracker.mode = config.mode
        tracker.min_score = config.min_score
        tracker.drop_near_dups = bool(config.drop_near_dups)
        self.summary = tracker

    def accept(self, record: Record) -> bool:
        """Score a record and apply QC gating rules.

        Args:
            record (Record): Record to score.

        Returns:
            bool: True if the record passes QC and should be kept.
        """
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
            return True

        keep = self.summary.observe(qc_result, apply_gates=self.enforce_drops)
        self._merge_qc_meta(record, qc_result)
        return keep

    def on_record(self, record: Record) -> Record:
        """No-op observer hook; returns the record unchanged."""
        # Inline QC performs all work inside accept(); observer hook is a pass-through.
        return record

    def summary_dict(self) -> Dict[str, Any]:
        """Return the current QC summary as a dictionary."""
        return self.summary.as_dict()

    def _merge_qc_meta(self, record: Record, qc_result: Dict[str, Any]) -> None:
        """Attach QC-derived metadata to the record meta dictionary.

        Populates approximate token counts, merges canonical QC fields, and
        stores additional QC signals under meta["extra"]["qc_signals"] without
        clobbering existing entries.

        Args:
            record (Record): Record whose meta will be updated.
            qc_result (dict[str, Any]): QC result data for the record.
        """
        if not isinstance(record, dict):
            return
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
