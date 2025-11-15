"""
Helpers for inline QC execution and summary building.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .config import QCMode
from .interfaces import QualityScorer, Record
from .records import ensure_meta_dict, merge_meta_defaults
from .qc_utils import update_dup_family_counts, top_dup_families


@dataclass(slots=True)
class QCSummaryTracker:
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
        """Update counters based on a QC row, returning True if it should be kept."""
        self.scored += 1
        family_id = qc_result.get("dup_family_id") or qc_result.get("doc_id")
        path = qc_result.get("path")
        if family_id:
            update_dup_family_counts(self.dup_families, family_id, path)
            if self.top_dup_snapshot:
                self.top_dup_snapshot.clear()

        low_score = self._is_low_score(qc_result)
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
        self.errors += 1

    def as_dict(self) -> Dict[str, Any]:
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
        if self.top_dup_snapshot:
            return [dict(entry) for entry in self.top_dup_snapshot]
        return top_dup_families(self.dup_families)

    @classmethod
    def from_summary_dict(cls, data: Mapping[str, Any]) -> "QCSummaryTracker":
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
    """
    Wraps scorer + gating logic so pipeline.run_pipeline stays lean.
    """

    def __init__(
        self,
        *,
        config,
        stats,
        scorer: QualityScorer,
        logger,
        enforce_drops: bool = True,
    ) -> None:
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

    def should_keep(self, record: Record) -> bool:
        try:
            qc_result = self.scorer.score_record(record)
        except Exception as exc:
            self.summary.record_error()
            if getattr(self.cfg, "fail_on_error", False):
                raise
            if self.logger:
                path = _record_path(record)
                self.logger.warning("QC scoring failed for %s: %s", path, exc)
            return True

        keep = self.summary.observe(qc_result, apply_gates=self.enforce_drops)
        self._merge_qc_meta(record, qc_result)
        return keep

    def summary_dict(self) -> Dict[str, Any]:
        return self.summary.as_dict()

    def _merge_qc_meta(self, record: Record, qc_result: Dict[str, Any]) -> None:
        if not isinstance(record, dict):
            return
        meta = ensure_meta_dict(record)
        tokens_est = qc_result.get("tokens")
        if tokens_est is not None:
            meta["approx_tokens"] = tokens_est
            meta.setdefault("tokens", tokens_est)
        merge_meta_defaults(record, qc_result)


def summarize_qc_rows(
    rows: Iterable[Dict[str, Any]],
    *,
    mode: str,
    min_score: Optional[float],
    drop_near_dups: bool,
    apply_gates: bool = False,
    enabled: bool = True,
) -> Dict[str, Any]:
    """
    Build a summary dict from QC rows (used by post-processing mode).
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


def _record_path(record: Record) -> str:
    if isinstance(record, dict):
        meta = record.get("meta")
        if isinstance(meta, dict) and meta.get("path"):
            return str(meta.get("path"))
        if record.get("path"):
            return str(record.get("path"))
    return "<unknown>"
