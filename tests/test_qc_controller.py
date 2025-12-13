from dataclasses import dataclass

import pytest

from sievio.core.config import QCConfig, QCMode, SafetyConfig
from sievio.core.pipeline import PipelineStats
from sievio.core.qc_controller import (
    InlineQCController,
    InlineScreeningController,
    QCSummaryTracker,
    QualityInlineScreener,
    SafetyInlineScreener,
)
from sievio.core.records import ensure_meta_dict


@dataclass
class DummyStats:
    qc: QCSummaryTracker | None = None


class DummyScorer:
    def __init__(self, rows):
        self._rows = iter(rows)

    def score_record(self, record):
        return next(self._rows)

    def score_jsonl_path(self, path):
        raise NotImplementedError


class DummyLogger:
    def warning(self, *args, **kwargs):
        return None


@pytest.fixture
def tracker_basic():
    return QCSummaryTracker(min_score=80.0, drop_near_dups=True)


def test_tracker_observe_keep_vs_drop(tracker_basic):
    row_good = {"score": 90.0, "near_dup": False, "dup_family_id": "fam1", "path": "a.py"}
    row_low = {"score": 50.0, "near_dup": False, "dup_family_id": "fam2", "path": "b.py"}
    row_mid = {"score": 79.9, "near_dup": False, "dup_family_id": "fam3", "path": "c.py"}

    keep_good = tracker_basic.observe(row_good, apply_gates=True)
    keep_low = tracker_basic.observe(row_low, apply_gates=True)
    keep_mid = tracker_basic.observe(row_mid, apply_gates=True)

    assert keep_good is True
    assert keep_low is False
    assert keep_mid is False

    assert tracker_basic.scored == 3
    assert tracker_basic.kept == 1
    assert tracker_basic.candidates_low_score == 2
    assert tracker_basic.dropped_low_score == 2
    assert tracker_basic.dropped_near_dup == 0


def test_tracker_observe_near_dups_and_dup_families():
    tracker = QCSummaryTracker(min_score=None, drop_near_dups=True)
    row_base = {"score": 90.0, "near_dup": False, "dup_family_id": "famA", "path": "a.py"}
    row_dup1 = {"score": 92.0, "near_dup": True, "dup_family_id": "famA", "path": "a_2.py"}
    row_dup2 = {"score": 88.0, "near_dup": True, "dup_family_id": "famA", "path": "a_3.py"}

    tracker.observe(row_base, apply_gates=True)
    tracker.observe(row_dup1, apply_gates=True)
    tracker.observe(row_dup2, apply_gates=True)

    assert tracker.kept == 1
    assert tracker.dropped_near_dup == 2
    assert "famA" in tracker.dup_families
    assert tracker.dup_families["famA"]["count"] == 3
    top = tracker.top_dup_families()
    assert top and top[0]["dup_family_id"] == "famA"


def test_tracker_summary_roundtrip(tracker_basic):
    tracker_basic.observe({"score": 85.0, "near_dup": False}, apply_gates=True)
    tracker_basic.observe({"score": 70.0, "near_dup": True, "dup_family_id": "famB"}, apply_gates=True)

    summary = tracker_basic.as_dict()
    tracker2 = QCSummaryTracker.from_summary_dict(summary)

    assert tracker2.scored == tracker_basic.scored
    assert tracker2.kept == tracker_basic.kept
    assert tracker2.dropped_low_score == tracker_basic.dropped_low_score
    assert tracker2.dropped_near_dup == tracker_basic.dropped_near_dup
    assert tracker2.top_dup_families() == tracker_basic.top_dup_families()


def test_merge_from_summary_replaces_only_quality():
    tracker = QCSummaryTracker()
    tracker.observe_safety({"safety_decision": "keep", "safety_flags": {"flagged": True}}, apply_gates=False)

    incoming = QCSummaryTracker()
    incoming.observe({"score": 50.0, "near_dup": True, "dup_family_id": "famC"}, apply_gates=True)

    tracker.merge_from_summary_dict(incoming.as_dict(), replace_screeners={"quality"})

    assert tracker.screeners["quality"].scored == 1
    assert tracker.screeners["quality"].dropped == 0
    assert tracker.screeners["safety"].scored == 1
    assert tracker.screeners["safety"].flags.get("flagged") == 1


def test_merge_from_summary_replaces_only_safety():
    tracker = QCSummaryTracker()
    tracker.observe({"score": 90.0, "near_dup": False}, apply_gates=False)

    incoming = QCSummaryTracker()
    incoming.observe_safety(
        {"safety_decision": "drop", "safety_flags": {"pii": True}}, apply_gates=True, mode=QCMode.POST
    )

    tracker.merge_from_summary_dict(incoming.as_dict(), replace_screeners={"safety"})

    assert tracker.screeners["quality"].scored == 1
    assert tracker.screeners["safety"].mode == QCMode.POST
    assert tracker.safety_dropped == 1
    assert tracker.safety_flags.get("pii") == 1


def test_observe_safety_tracks_mode():
    tracker = QCSummaryTracker()
    tracker.observe_safety({"safety_decision": "keep"}, apply_gates=False, mode=QCMode.POST)

    assert tracker.screeners["safety"].mode == QCMode.POST


def test_tracker_signal_stats_numeric_and_bool():
    tracker = QCSummaryTracker(min_score=None)

    tracker.observe({"score": 90.0, "ascii_ratio": 0.4, "parse_ok": True}, apply_gates=False)
    tracker.observe({"score": 91.0, "ascii_ratio": 0.6, "parse_ok": False}, apply_gates=False)

    ascii_stats = tracker.signal_stats["ascii_ratio"].as_dict()
    parse_stats = tracker.signal_stats["parse_ok"].as_dict()

    assert ascii_stats["count"] == 2
    assert ascii_stats["mean"] == pytest.approx(0.5)
    assert ascii_stats["min"] == pytest.approx(0.4)
    assert ascii_stats["max"] == pytest.approx(0.6)

    assert parse_stats["count"] == 2
    assert parse_stats["mean"] == pytest.approx(0.5)
    assert parse_stats["min"] == 0
    assert parse_stats["max"] == 1


def test_tracker_reset_for_run_clears_state_in_place():
    tracker = QCSummaryTracker(enabled=True, mode=QCMode.ADVISORY, min_score=10.0, drop_near_dups=True)
    tracker.observe({"score": 5.0, "near_dup": True, "dup_family_id": "fam1", "path": "a.py"}, apply_gates=True)
    tracker.observe_safety({"safety_decision": "drop", "safety_flags": {"blocked": True}}, apply_gates=True)
    tracker.record_error()
    tracker.top_dup_snapshot = [{"dup_family_id": "fam1", "count": 1}]

    assert tracker.screeners
    assert tracker.dup_families

    original_id = id(tracker)
    tracker.reset_for_run(enabled=False, mode=QCMode.INLINE, min_score=None, drop_near_dups=False)

    assert id(tracker) == original_id
    assert tracker.enabled is False
    assert tracker.mode == QCMode.INLINE
    assert tracker.min_score is None
    assert tracker.drop_near_dups is False
    assert tracker.dup_families == {}
    assert tracker.top_dup_snapshot == []
    assert tracker.screeners == {}


def test_inline_controller_reset_reuses_stats_tracker():
    stats = PipelineStats()
    tracker = stats.qc
    tracker.record_error()
    tracker.observe({"score": 1.0, "near_dup": False}, apply_gates=True)

    controller = InlineScreeningController(
        summary=None,
        screeners=[],
        logger=DummyLogger(),
        qc_cfg=QCConfig(enabled=True, mode=QCMode.INLINE, min_score=0.1),
    )
    controller.reset(
        stats,
        qc_cfg=QCConfig(enabled=True, mode=QCMode.ADVISORY, min_score=0.5, drop_near_dups=True),
    )

    assert controller.tracker is tracker is stats.qc
    assert tracker.enabled is True
    assert tracker.mode == QCMode.ADVISORY
    assert tracker.min_score == 0.5
    assert tracker.drop_near_dups is True
    assert tracker.screeners == {}
    assert tracker.errors == 0


def make_controller(*, min_score=60.0, drop_near_dups=False, enforce_drops=True, qc_rows=()):
    cfg = QCConfig(
        enabled=True,
        min_score=min_score,
        drop_near_dups=drop_near_dups,
        mode=QCMode.INLINE if enforce_drops else QCMode.ADVISORY,
    )
    stats = DummyStats()
    scorer = DummyScorer(qc_rows)
    controller = InlineQCController(
        config=cfg,
        stats=stats,
        scorer=scorer,
        logger=None,
        enforce_drops=enforce_drops,
    )
    return controller, stats


def test_inline_qc_accept_attachs_meta_and_keeps_when_passing():
    qc_row = {
        "score": 90.0,
        "tokens": 123,
        "len": 456,
        "near_dup": False,
        "dup_family_id": "famX",
        "ascii_ratio": 0.9,
    }
    controller, stats = make_controller(qc_rows=[qc_row])
    record = {"text": "hello", "meta": {"path": "foo.py", "chunk_id": 1, "n_chunks": 1}}

    accepted = controller.accept(record)

    assert accepted is True
    assert stats.qc is not None
    assert stats.qc.kept == 1
    assert stats.qc.scored == 1
    meta = record["meta"]
    assert meta["qc_score"] == qc_row["score"]
    assert meta["approx_tokens"] == qc_row["tokens"]
    assert meta["tokens"] == qc_row["tokens"]
    assert "extra" in meta and "qc_signals" in meta["extra"]
    assert meta["extra"]["qc_signals"].get("ascii_ratio") == qc_row["ascii_ratio"]
    assert meta["extra"]["qc_signals"].get("len_tok") == qc_row["tokens"]
    assert meta["extra"]["qc_signals"].get("len_char") == qc_row["len"]
    assert meta["near_dup"] is False
    assert meta["dup_family_id"] == "famX"


def test_inline_vs_advisory_low_score_drops_or_keeps():
    qc_low = {"score": 50.0, "tokens": 10, "near_dup": False, "dup_family_id": "famY"}
    controller_inline, stats_inline = make_controller(min_score=60.0, enforce_drops=True, qc_rows=[qc_low])
    controller_adv, stats_adv = make_controller(min_score=60.0, enforce_drops=False, qc_rows=[qc_low])

    record_inline = {"text": "x", "meta": {"path": "a.py"}}
    record_adv = {"text": "x", "meta": {"path": "b.py"}}

    acc_inline = controller_inline.accept(record_inline)
    acc_adv = controller_adv.accept(record_adv)

    assert acc_inline is False
    assert stats_inline.qc.dropped_low_score == 1
    assert stats_inline.qc.kept == 0

    assert acc_adv is True
    assert stats_adv.qc.candidates_low_score == 1
    assert stats_adv.qc.dropped_low_score == 0
    assert stats_adv.qc.kept == 1


def test_near_dup_drop_vs_advisory_keep():
    qc_near_dup = {"score": 95.0, "tokens": 50, "near_dup": True, "dup_family_id": "famZ", "path": "dup.py"}
    controller_inline, stats_inline = make_controller(drop_near_dups=True, enforce_drops=True, qc_rows=[qc_near_dup])
    controller_adv, stats_adv = make_controller(drop_near_dups=True, enforce_drops=False, qc_rows=[qc_near_dup])

    record_inline = {"text": "x", "meta": {"path": "dup.py"}}
    record_adv = {"text": "x", "meta": {"path": "dup.py"}}

    acc_inline = controller_inline.accept(record_inline)
    acc_adv = controller_adv.accept(record_adv)

    assert acc_inline is False
    assert stats_inline.qc.dropped_near_dup == 1
    assert stats_inline.qc.candidates_near_dup >= 1

    assert acc_adv is True
    assert stats_adv.qc.dropped_near_dup == 0
    assert stats_adv.qc.candidates_near_dup >= 1


def test_merge_qc_meta_preserves_existing_extra():
    qc_row = {"score": 80.0, "tokens": 10, "near_dup": False, "dup_family_id": "famQ"}
    controller, stats = make_controller(enforce_drops=False, qc_rows=[qc_row])
    record = {"text": "x", "meta": {"path": "foo.py", "extra": {"foo": "bar"}}}

    accepted = controller.accept(record)

    assert accepted is True
    meta = ensure_meta_dict(record)
    assert meta["extra"]["foo"] == "bar"
    qc_signals = meta["extra"]["qc_signals"]
    assert isinstance(qc_signals, dict)
    assert meta.get("near_dup") is False
    assert meta.get("dup_family_id") == "famQ"
    assert "qc_score" not in qc_signals


def test_qc_summary_tracker_legacy_roundtrip_preserves_top_fields():
    summary = {
        "enabled": True,
        "mode": QCMode.INLINE,
        "min_score": 0.25,
        "drop_near_dups": True,
        "scored": 3,
        "kept": 2,
        "dropped_low_score": 1,
        "dropped_near_dup": 0,
        "errors": 1,
        "candidates_low_score": 1,
        "candidates_near_dup": 0,
        "signal_stats": {"foo": {"count": 1, "mean": 0.5, "min": 0.5, "max": 0.5, "stdev": 0.0}},
        "safety": {
            "enabled": True,
            "scored": 2,
            "dropped": 1,
            "errors": 0,
            "flags": {"p1": 1},
        },
    }
    tracker = QCSummaryTracker.from_summary_dict(summary)
    roundtrip = tracker.as_dict()

    assert roundtrip["min_score"] == summary["min_score"]
    assert roundtrip["drop_near_dups"] is True
    assert roundtrip["scored"] == summary["scored"]
    assert roundtrip["kept"] == summary["kept"]
    assert roundtrip["dropped_low_score"] == summary["dropped_low_score"]
    assert roundtrip["safety"]["scored"] == summary["safety"]["scored"]
    assert roundtrip["safety"]["dropped"] == summary["safety"]["dropped"]
    # legacy input should have screeners synthesized
    assert "screeners" in roundtrip
    assert "quality" in roundtrip["screeners"]
    assert "safety" in roundtrip["screeners"]


def test_qc_summary_tracker_reads_new_screener_payload():
    summary = {
        "enabled": True,
        "mode": QCMode.INLINE,
        "screeners": {
            "quality": {
                "id": "quality",
                "mode": QCMode.INLINE,
                "scored": 2,
                "kept": 1,
                "drops": {"low_score": 1},
                "candidates": {"low_score": 1},
                "signal_stats": {"len_tok": {"count": 1, "mean": 10.0, "min": 10.0, "max": 10.0, "stdev": 0.0}},
            },
            "safety": {
                "id": "safety",
                "mode": QCMode.INLINE,
                "scored": 1,
                "dropped": 1,
                "flags": {"blocked": 1},
            },
        },
    }
    tracker = QCSummaryTracker.from_summary_dict(summary)
    result = tracker.as_dict()

    assert result["scored"] == 2  # quality view
    assert result["kept"] == 1
    assert result["dropped_low_score"] == 1
    assert result["safety"]["scored"] == 1
    assert result["safety"]["dropped"] == 1
    assert result["screeners"]["quality"]["signal_stats"]["len_tok"]["count"] == 1


class DropScorer:
    def __init__(self, score: float, near_dup: bool = False):
        self.score = score
        self.near_dup = near_dup

    def score_record(self, record):
        return {"score": self.score, "near_dup": self.near_dup}


def test_inline_screening_controller_single_screener_drops_and_counts():
    qc_cfg = QCConfig(enabled=True, min_score=0.5, mode=QCMode.INLINE)
    scorer = DropScorer(score=0.1)
    tracker = QCSummaryTracker()
    controller = InlineScreeningController(
        summary=tracker,
        screeners=[
            QualityInlineScreener(cfg=qc_cfg, scorer=scorer, summary=tracker, logger=None, enforce_drops=True)
        ],
        logger=None,
        qc_cfg=qc_cfg,
    )
    stats = DummyStats()
    controller.reset(stats, qc_cfg=qc_cfg)
    kept = controller.process_record({"text": "x", "meta": {"path": "foo.py"}})

    assert kept is None
    quality_stats = controller.tracker.screeners["quality"]
    assert quality_stats.scored == 1
    assert quality_stats.kept == 0
    assert quality_stats.drops.get("low_score") == 1


class SafetyDropper:
    def score_record(self, record):
        return {"safety_decision": "drop", "safety_flags": {"blocked": True}}


class DummyScreener:
    def __init__(self, sid: str):
        self.id = sid
        self.enforce_drops = None

    def process_record(self, record):
        return record


class ExplodingScreener:
    def __init__(self, sid: str):
        self.id = sid
        self.enforce_drops = None

    def process_record(self, record):
        raise RuntimeError("boom")


def test_inline_screening_controller_two_screeners_drop_by_safety():
    qc_cfg = QCConfig(enabled=True, min_score=0.0, mode=QCMode.INLINE)
    safety_cfg = SafetyConfig(enabled=True, mode=QCMode.INLINE, annotate_only=False)
    tracker = QCSummaryTracker()
    controller = InlineScreeningController(
        summary=tracker,
        screeners=[
            QualityInlineScreener(
                cfg=qc_cfg, scorer=DropScorer(score=1.0), summary=tracker, logger=None, enforce_drops=True
            ),
            SafetyInlineScreener(
                cfg=safety_cfg, scorer=SafetyDropper(), summary=tracker, logger=None, enforce_drops=True
            ),
        ],
        logger=None,
        qc_cfg=qc_cfg,
        safety_cfg=safety_cfg,
    )
    controller.reset(DummyStats(), qc_cfg=qc_cfg, safety_cfg=safety_cfg)

    kept = controller.process_record({"text": "hello", "meta": {"path": "bar.py"}})

    assert kept is None
    quality = controller.tracker.screeners["quality"]
    safety = controller.tracker.screeners["safety"]
    assert quality.scored == 1
    assert quality.kept == 0  # dropped downstream by safety
    assert safety.scored == 1
    assert safety.dropped == 1


def test_controller_reset_derives_gate_flags_from_configs():
    qc_cfg = QCConfig(enabled=True, min_score=0.0, mode=QCMode.INLINE)
    safety_cfg = SafetyConfig(enabled=True, mode=QCMode.INLINE, annotate_only=True)
    quality = DummyScreener("quality")
    safety = DummyScreener("safety")
    controller = InlineScreeningController(
        summary=None,
        screeners=[quality, safety],
        logger=DummyLogger(),
        qc_cfg=qc_cfg,
        safety_cfg=safety_cfg,
    )
    stats = PipelineStats()
    controller.reset(stats, qc_cfg=qc_cfg, safety_cfg=safety_cfg)
    controller.process_record({"text": "x", "meta": {"path": "foo.py"}})

    assert controller.apply_qc_gates is True
    assert quality.enforce_drops is True
    assert controller.apply_safety_gates is False
    assert safety.enforce_drops is False

    qc_cfg.mode = QCMode.ADVISORY
    safety_cfg.annotate_only = False
    quality.enforce_drops = None
    safety.enforce_drops = None

    controller.reset(stats, qc_cfg=qc_cfg, safety_cfg=safety_cfg)
    controller.process_record({"text": "y", "meta": {"path": "bar.py"}})

    assert controller.apply_qc_gates is False
    assert quality.enforce_drops is False
    assert controller.apply_safety_gates is True
    assert safety.enforce_drops is True


def test_process_record_records_quality_error_on_exception():
    qc_cfg = QCConfig(enabled=True, min_score=0.0, mode=QCMode.INLINE)
    controller = InlineScreeningController(
        summary=None,
        screeners=[ExplodingScreener("quality")],
        logger=DummyLogger(),
        qc_cfg=qc_cfg,
    )
    stats = PipelineStats()
    controller.reset(stats, qc_cfg=qc_cfg)

    kept = controller.process_record({"text": "boom", "meta": {"path": "err.py"}})

    assert kept is None
    assert stats.qc.errors == 1
    assert stats.qc.screeners["quality"].errors == 1


def test_process_record_records_safety_error_on_exception():
    qc_cfg = QCConfig(enabled=False, mode=QCMode.OFF)
    safety_cfg = SafetyConfig(enabled=True, mode=QCMode.INLINE, annotate_only=False)
    controller = InlineScreeningController(
        summary=None,
        screeners=[ExplodingScreener("safety")],
        logger=DummyLogger(),
        qc_cfg=qc_cfg,
        safety_cfg=safety_cfg,
    )
    stats = PipelineStats()
    controller.reset(stats, qc_cfg=qc_cfg, safety_cfg=safety_cfg)

    kept = controller.process_record({"text": "boom", "meta": {"path": "safety_err.py"}})

    assert kept is None
    assert stats.qc.safety_errors == 1
    assert stats.qc.screeners["safety"].errors == 1


def test_process_record_records_custom_error_and_rolls_back_kept():
    qc_cfg = QCConfig(enabled=True, min_score=0.0, mode=QCMode.INLINE)
    quality = QualityInlineScreener(
        cfg=qc_cfg,
        scorer=DropScorer(score=1.0),
        summary=QCSummaryTracker(),
        logger=None,
        enforce_drops=True,
    )
    exploding = ExplodingScreener("custom_x")
    controller = InlineScreeningController(
        summary=None,
        screeners=[quality, exploding],
        logger=DummyLogger(),
        qc_cfg=qc_cfg,
    )
    stats = PipelineStats()
    controller.reset(stats, qc_cfg=qc_cfg)

    kept = controller.process_record({"text": "ok", "meta": {"path": "keep_then_fail.py"}})
    quality_stats = stats.qc.screeners["quality"]

    assert kept is None
    assert stats.qc.screeners["custom_x"].errors == 1
    assert quality_stats.kept == 0
    assert quality_stats.scored == 1
