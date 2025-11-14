"""
Helpers for inline QC execution and summary building.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional

from .interfaces import QualityScorer, Record
from .qc_utils import update_dup_family_counts, top_dup_families


@dataclass
class QCSummaryTracker:
    enabled: bool = False
    mode: str = "inline"
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

    def observe(self, qc_result: Dict[str, Any], *, apply_gates: bool = True) -> bool:
        """Update counters based on a QC row, returning True if it should be kept."""
        self.scored += 1
        family_id = qc_result.get("dup_family_id") or qc_result.get("doc_id")
        path = qc_result.get("path")
        if family_id:
            update_dup_family_counts(self.dup_families, family_id, path)

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
            "scored": int(self.scored),
            "kept": int(self.kept),
            "dropped_low_score": int(self.dropped_low_score),
            "dropped_near_dup": int(self.dropped_near_dup),
            "errors": int(self.errors),
            "candidates_low_score": int(self.candidates_low_score),
            "candidates_near_dup": int(self.candidates_near_dup),
            "top_dup_families": top_dup_families(self.dup_families),
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
        dup_families = getattr(stats, "qc_dup_families", None)
        self.summary = QCSummaryTracker(
            enabled=True,
            mode=config.mode,
            min_score=config.min_score,
            drop_near_dups=bool(config.drop_near_dups),
            dup_families=dup_families if isinstance(dup_families, dict) else {},
        )

    def should_keep(self, record: Record) -> bool:
        try:
            qc_result = self.scorer.score_record(record)
        except Exception as exc:
            self.summary.record_error()
            self._sync_stats()
            if getattr(self.cfg, "fail_on_error", False):
                raise
            if self.logger:
                path = _record_path(record)
                self.logger.warning("QC scoring failed for %s: %s", path, exc)
            return True

        keep = self.summary.observe(qc_result, apply_gates=self.enforce_drops)
        self._merge_qc_meta(record, qc_result)
        self._sync_stats()
        return keep

    def summary_dict(self) -> Dict[str, Any]:
        return self.summary.as_dict()

    def _sync_stats(self) -> None:
        self.stats.qc_scored = self.summary.scored
        self.stats.qc_kept = self.summary.kept
        self.stats.qc_dropped_low_score = self.summary.dropped_low_score
        self.stats.qc_dropped_near_dup = self.summary.dropped_near_dup
        self.stats.qc_errors = self.summary.errors
        self.stats.qc_candidates_low_score = self.summary.candidates_low_score
        self.stats.qc_candidates_near_dup = self.summary.candidates_near_dup

    def _merge_qc_meta(self, record: Record, qc_result: Dict[str, Any]) -> None:
        if not isinstance(record, dict):
            return
        meta = record.get("meta")
        if not isinstance(meta, dict):
            meta = {}
            record["meta"] = meta
        tokens_est = qc_result.get("tokens")
        if tokens_est is not None:
            meta["approx_tokens"] = tokens_est
            meta.setdefault("tokens", tokens_est)
        for key, value in qc_result.items():
            if key in meta or value is None:
                continue
            meta[key] = value


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
