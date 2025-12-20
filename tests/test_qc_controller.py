from dataclasses import dataclass

import pytest

from sievio.core.config import QCConfig, QCMode, SafetyConfig
from sievio.core.pipeline import PipelineStats
from sievio.core.qc_controller import (
    InlineQCController,
    InlineScreeningController,
    QCSummaryTracker,
    QualityDecisionPolicy,
    QualityInlineScreener,
    SafetyDecisionPolicy,
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
    policy = QualityDecisionPolicy()
    cfg = QCConfig(min_score=tracker_basic.min_score, drop_near_dups=tracker_basic.drop_near_dups)
    row_good = {"score": 90.0, "near_dup": False, "dup_family_id": "fam1", "path": "a.py"}
    row_low = {"score": 50.0, "near_dup": False, "dup_family_id": "fam2", "path": "b.py"}
    row_mid = {"score": 79.9, "near_dup": False, "dup_family_id": "fam3", "path": "c.py"}

    decision_good = policy.decide(row_good, cfg=cfg)
    decision_low = policy.decide(row_low, cfg=cfg)
    decision_mid = policy.decide(row_mid, cfg=cfg)

    tracker_basic.observe_quality(row_good, decision_good, did_drop=bool(decision_good.would_drop))
    tracker_basic.observe_quality(row_low, decision_low, did_drop=bool(decision_low.would_drop))
    tracker_basic.observe_quality(row_mid, decision_mid, did_drop=bool(decision_mid.would_drop))

    assert decision_good.would_drop == ()
    assert decision_low.would_drop == ("low_score",)
    assert decision_mid.would_drop == ("low_score",)

    quality = tracker_basic.get_screener("quality", create=False)
    assert quality is not None
    assert quality.scored == 3
    assert quality.kept == 1
    assert quality.candidates.get("low_score") == 2
    assert quality.drops.get("low_score") == 2
    assert quality.drops.get("near_dup", 0) == 0


def test_tracker_observe_near_dups_and_dup_families():
    tracker = QCSummaryTracker(min_score=None, drop_near_dups=True)
    policy = QualityDecisionPolicy()
    cfg = QCConfig(min_score=None, drop_near_dups=True)
    row_base = {"score": 90.0, "near_dup": False, "dup_family_id": "famA", "path": "a.py"}
    row_dup1 = {"score": 92.0, "near_dup": True, "dup_family_id": "famA", "path": "a_2.py"}
    row_dup2 = {"score": 88.0, "near_dup": True, "dup_family_id": "famA", "path": "a_3.py"}

    decision_base = policy.decide(row_base, cfg=cfg)
    decision_dup1 = policy.decide(row_dup1, cfg=cfg)
    decision_dup2 = policy.decide(row_dup2, cfg=cfg)
    tracker.observe_quality(row_base, decision_base, did_drop=bool(decision_base.would_drop))
    tracker.observe_quality(row_dup1, decision_dup1, did_drop=bool(decision_dup1.would_drop))
    tracker.observe_quality(row_dup2, decision_dup2, did_drop=bool(decision_dup2.would_drop))

    quality = tracker.get_screener("quality", create=False)
    assert quality is not None
    assert quality.kept == 1
    assert quality.drops.get("near_dup") == 2
    assert "famA" in tracker.dup_families
    assert tracker.dup_families["famA"]["count"] == 3
    top = tracker.top_dup_families()
    assert top and top[0]["dup_family_id"] == "famA"


def test_tracker_summary_roundtrip(tracker_basic):
    policy = QualityDecisionPolicy()
    cfg = QCConfig(min_score=tracker_basic.min_score, drop_near_dups=tracker_basic.drop_near_dups)
    row_good = {"score": 85.0, "near_dup": False}
    row_dup = {"score": 70.0, "near_dup": True, "dup_family_id": "famB"}
    decision_good = policy.decide(row_good, cfg=cfg)
    decision_dup = policy.decide(row_dup, cfg=cfg)
    tracker_basic.observe_quality(row_good, decision_good, did_drop=bool(decision_good.would_drop))
    tracker_basic.observe_quality(row_dup, decision_dup, did_drop=bool(decision_dup.would_drop))

    summary = tracker_basic.as_dict()
    tracker2 = QCSummaryTracker.from_summary_dict(summary)

    quality1 = tracker_basic.get_screener("quality", create=False)
    quality2 = tracker2.get_screener("quality", create=False)

    assert quality1 is not None and quality2 is not None
    assert quality2.scored == quality1.scored
    assert quality2.kept == quality1.kept
    assert quality2.drops.get("low_score") == quality1.drops.get("low_score")
    assert quality2.drops.get("near_dup") == quality1.drops.get("near_dup")
    assert tracker2.top_dup_families() == tracker_basic.top_dup_families()
    assert summary["schema_version"] == 1


def test_merge_from_summary_replaces_only_quality():
    tracker = QCSummaryTracker()
    safety_policy = SafetyDecisionPolicy()
    safety_row = {"safety_decision": "keep", "safety_flags": {"flagged": True}}
    decision = safety_policy.decide(safety_row, cfg=None)
    tracker.observe_safety(safety_row, decision, did_drop=False)

    incoming = QCSummaryTracker()
    qc_policy = QualityDecisionPolicy()
    qc_cfg = QCConfig(min_score=None, drop_near_dups=True)
    qc_row = {"score": 50.0, "near_dup": True, "dup_family_id": "famC"}
    decision = qc_policy.decide(qc_row, cfg=qc_cfg)
    incoming.observe_quality(qc_row, decision, did_drop=False)

    tracker.merge_from_summary_dict(incoming.as_dict(), replace_screeners={"quality"})

    assert tracker.screeners["quality"].scored == 1
    assert tracker.screeners["quality"].dropped == 0
    assert tracker.screeners["safety"].scored == 1
    assert tracker.screeners["safety"].flags.get("flagged") == 1


def test_merge_from_summary_replaces_only_safety():
    tracker = QCSummaryTracker()
    qc_policy = QualityDecisionPolicy()
    qc_cfg = QCConfig(min_score=None, drop_near_dups=False)
    qc_row = {"score": 90.0, "near_dup": False}
    decision = qc_policy.decide(qc_row, cfg=qc_cfg)
    tracker.observe_quality(qc_row, decision, did_drop=False)

    incoming = QCSummaryTracker()
    safety_policy = SafetyDecisionPolicy()
    safety_row = {"safety_decision": "drop", "safety_flags": {"pii": True}}
    decision = safety_policy.decide(safety_row, cfg=None)
    incoming.observe_safety(safety_row, decision, did_drop=True, mode=QCMode.POST)

    tracker.merge_from_summary_dict(incoming.as_dict(), replace_screeners={"safety"})

    assert tracker.screeners["quality"].scored == 1
    assert tracker.screeners["safety"].mode == QCMode.POST
    assert tracker.screeners["safety"].dropped == 1
    assert tracker.screeners["safety"].flags.get("pii") == 1


def test_observe_safety_tracks_mode():
    tracker = QCSummaryTracker()
    safety_policy = SafetyDecisionPolicy()
    safety_row = {"safety_decision": "keep"}
    decision = safety_policy.decide(safety_row, cfg=None)
    tracker.observe_safety(safety_row, decision, did_drop=False, mode=QCMode.POST)

    assert tracker.screeners["safety"].mode == QCMode.POST


def test_safety_kept_accounting():
    tracker = QCSummaryTracker()
    policy = SafetyDecisionPolicy()

    keep_row = {"safety_decision": "keep", "safety_flags": {"flagged": True}}
    keep_decision = policy.decide(keep_row, cfg=None)
    tracker.observe_safety(keep_row, keep_decision, did_drop=False)

    drop_row = {"safety_decision": "drop", "safety_flags": {"flagged": True}}
    annotate_decision = policy.decide(drop_row, cfg=None)
    tracker.observe_safety(drop_row, annotate_decision, did_drop=False)

    drop_decision = policy.decide(drop_row, cfg=None)
    tracker.observe_safety(drop_row, drop_decision, did_drop=True)

    safety = tracker.screeners["safety"]
    assert safety.scored == 3
    assert safety.kept == 2
    assert safety.dropped == 1
    assert safety.flags.get("flagged") == 3


def test_quality_policy_parity():
    policy = QualityDecisionPolicy()
    cfg = QCConfig(min_score=60.0, drop_near_dups=True)

    low_score = {"score": 50.0, "near_dup": False}
    low_decision = policy.decide(low_score, cfg=cfg)
    assert low_decision.candidates == ("low_score",)
    assert low_decision.would_drop == ("low_score",)

    near_dup = {"score": 90.0, "near_dup": True}
    near_decision = policy.decide(near_dup, cfg=cfg)
    assert "near_dup" in near_decision.candidates
    assert near_decision.would_drop == ("near_dup",)

    both = {"score": 50.0, "near_dup": True}
    both_decision = policy.decide(both, cfg=cfg)
    assert set(both_decision.candidates) == {"low_score", "near_dup"}
    assert set(both_decision.would_drop) == {"low_score", "near_dup"}

    neither = {"score": 90.0, "near_dup": False}
    neither_decision = policy.decide(neither, cfg=cfg)
    assert neither_decision.candidates == ()
    assert neither_decision.would_drop == ()


def test_safety_policy_would_drop_on_drop_decision():
    policy = SafetyDecisionPolicy()
    keep_row = {"safety_decision": "keep"}
    drop_row = {"safety_decision": "drop"}

    keep_decision = policy.decide(keep_row, cfg=None)
    drop_decision = policy.decide(drop_row, cfg=None)

    assert keep_decision.candidates == ()
    assert keep_decision.would_drop == ()
    assert drop_decision.candidates == ("drop",)
    assert drop_decision.would_drop == ("drop",)


def test_tracker_signal_stats_numeric_and_bool():
    tracker = QCSummaryTracker(min_score=None)
    policy = QualityDecisionPolicy()
    cfg = QCConfig(min_score=None, drop_near_dups=False)

    row_a = {"score": 90.0, "ascii_ratio": 0.4, "parse_ok": True}
    row_b = {"score": 91.0, "ascii_ratio": 0.6, "parse_ok": False}
    decision_a = policy.decide(row_a, cfg=cfg)
    decision_b = policy.decide(row_b, cfg=cfg)
    tracker.observe_quality(row_a, decision_a, did_drop=False)
    tracker.observe_quality(row_b, decision_b, did_drop=False)

    quality = tracker.get_screener("quality", create=False)
    assert quality is not None
    ascii_stats = quality.signal_stats["ascii_ratio"].as_dict()
    parse_stats = quality.signal_stats["parse_ok"].as_dict()

    assert ascii_stats["count"] == 2
    assert ascii_stats["mean"] == pytest.approx(0.5)
    assert ascii_stats["min"] == pytest.approx(0.4)
    assert ascii_stats["max"] == pytest.approx(0.6)

    assert parse_stats["count"] == 2
    assert parse_stats["mean"] == pytest.approx(0.5)
    assert parse_stats["min"] == 0
    assert parse_stats["max"] == 1


def test_tracker_scored_equals_kept_plus_dropped():
    tracker = QCSummaryTracker()
    policy = QualityDecisionPolicy()
    cfg = QCConfig(min_score=60.0, drop_near_dups=False)

    rows = [
        {"score": 90.0, "near_dup": False},
        {"score": 50.0, "near_dup": False},
    ]
    for row in rows:
        decision = policy.decide(row, cfg=cfg)
        tracker.observe_quality(row, decision, did_drop=bool(decision.would_drop))

    safety_policy = SafetyDecisionPolicy()
    keep_decision = safety_policy.decide({"safety_decision": "keep"}, cfg=None)
    drop_decision = safety_policy.decide({"safety_decision": "drop"}, cfg=None)
    tracker.observe_safety({"safety_decision": "keep"}, keep_decision, did_drop=False)
    tracker.observe_safety({"safety_decision": "drop"}, drop_decision, did_drop=True)

    quality = tracker.get_screener("quality", create=False)
    safety = tracker.get_screener("safety", create=False)
    assert quality is not None
    assert safety is not None
    assert quality.scored == quality.kept + quality.dropped
    assert safety.scored == safety.kept + safety.dropped


def test_tracker_reset_for_run_clears_state_in_place():
    tracker = QCSummaryTracker(enabled=True, mode=QCMode.ADVISORY, min_score=10.0, drop_near_dups=True)
    qc_policy = QualityDecisionPolicy()
    qc_cfg = QCConfig(min_score=10.0, drop_near_dups=True)
    qc_row = {"score": 5.0, "near_dup": True, "dup_family_id": "fam1", "path": "a.py"}
    decision = qc_policy.decide(qc_row, cfg=qc_cfg)
    tracker.observe_quality(qc_row, decision, did_drop=True)
    safety_policy = SafetyDecisionPolicy()
    safety_row = {"safety_decision": "drop", "safety_flags": {"blocked": True}}
    decision = safety_policy.decide(safety_row, cfg=None)
    tracker.observe_safety(safety_row, decision, did_drop=True)
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
    policy = QualityDecisionPolicy()
    cfg = QCConfig(min_score=0.1, drop_near_dups=False)
    row = {"score": 1.0, "near_dup": False}
    decision = policy.decide(row, cfg=cfg)
    tracker.observe_quality(row, decision, did_drop=bool(decision.would_drop))

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
    quality = tracker.get_screener("quality", create=False)
    assert quality is None or quality.errors == 0


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
    quality_stats = stats.qc.get_screener("quality", create=False)
    assert quality_stats is not None
    assert quality_stats.kept == 1
    assert quality_stats.scored == 1
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
    quality_inline = stats_inline.qc.get_screener("quality", create=False)
    assert quality_inline is not None
    assert quality_inline.drops.get("low_score") == 1
    assert quality_inline.kept == 0

    assert acc_adv is True
    quality_adv = stats_adv.qc.get_screener("quality", create=False)
    assert quality_adv is not None
    assert quality_adv.candidates.get("low_score") == 1
    assert quality_adv.drops.get("low_score", 0) == 1
    assert quality_adv.kept == 1


def test_near_dup_drop_vs_advisory_keep():
    qc_near_dup = {"score": 95.0, "tokens": 50, "near_dup": True, "dup_family_id": "famZ", "path": "dup.py"}
    controller_inline, stats_inline = make_controller(drop_near_dups=True, enforce_drops=True, qc_rows=[qc_near_dup])
    controller_adv, stats_adv = make_controller(drop_near_dups=True, enforce_drops=False, qc_rows=[qc_near_dup])

    record_inline = {"text": "x", "meta": {"path": "dup.py"}}
    record_adv = {"text": "x", "meta": {"path": "dup.py"}}

    acc_inline = controller_inline.accept(record_inline)
    acc_adv = controller_adv.accept(record_adv)

    assert acc_inline is False
    quality_inline = stats_inline.qc.get_screener("quality", create=False)
    assert quality_inline is not None
    assert quality_inline.drops.get("near_dup") == 1
    assert quality_inline.candidates.get("near_dup", 0) >= 1

    assert acc_adv is True
    quality_adv = stats_adv.qc.get_screener("quality", create=False)
    assert quality_adv is not None
    assert quality_adv.drops.get("near_dup", 0) == 1
    assert quality_adv.candidates.get("near_dup", 0) >= 1


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


def test_qc_summary_tracker_roundtrip_preserves_screeners_only():
    summary = {
        "schema_version": 1,
        "enabled": True,
        "mode": QCMode.INLINE,
        "min_score": 0.25,
        "drop_near_dups": True,
        "top_dup_families": [{"dup_family_id": "fam1", "count": 2}],
        "screeners": {
            "quality": {
                "id": "quality",
                "mode": QCMode.INLINE,
                "scored": 3,
                "kept": 2,
                "dropped": 1,
                "errors": 1,
                "candidates": {"low_score": 1},
                "drops": {"low_score": 1},
                "signal_stats": {"foo": {"count": 1, "mean": 0.5, "min": 0.5, "max": 0.5, "stdev": 0.0}},
            },
            "safety": {"id": "safety", "mode": QCMode.INLINE, "scored": 2, "dropped": 1, "errors": 0, "flags": {"p1": 1}},
        },
    }
    tracker = QCSummaryTracker.from_summary_dict(summary)
    roundtrip = tracker.as_dict()

    assert set(roundtrip.keys()) == {
        "schema_version",
        "enabled",
        "mode",
        "min_score",
        "drop_near_dups",
        "top_dup_families",
        "screeners",
    }
    quality = roundtrip["screeners"]["quality"]
    assert quality["scored"] == 3
    assert quality["kept"] == 2
    assert quality["drops"]["low_score"] == 1
    safety = roundtrip["screeners"]["safety"]
    assert safety["scored"] == 2
    assert safety["dropped"] == 1


def test_qc_summary_tracker_reads_new_screener_payload():
    summary = {
        "schema_version": 1,
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

    quality_summary = result["screeners"]["quality"]
    assert quality_summary["scored"] == 2
    assert quality_summary["kept"] == 1
    assert quality_summary["drops"]["low_score"] == 1
    safety_summary = result["screeners"]["safety"]
    assert safety_summary["scored"] == 1
    assert safety_summary["dropped"] == 1
    assert quality_summary["signal_stats"]["len_tok"]["count"] == 1


def test_qc_summary_strict_parsing_rejects_invalid():
    summary = {
        "schema_version": 1,
        "enabled": True,
        "mode": QCMode.INLINE,
        "screeners": {"quality": {"scored": "nope"}},
    }
    tracker = QCSummaryTracker.from_summary_dict(summary, strict=False)
    quality = tracker.screeners["quality"]
    assert quality.scored == 0

    with pytest.raises((TypeError, ValueError)):
        QCSummaryTracker.from_summary_dict(summary, strict=True)


def test_tracker_merge_accumulates_counts():
    policy = QualityDecisionPolicy()
    cfg = QCConfig(min_score=0.0, drop_near_dups=True)

    tracker_a = QCSummaryTracker(min_score=0.0, drop_near_dups=True)
    row_a = {"score": 10.0, "near_dup": False, "dup_family_id": "fam1", "path": "a.py", "ascii_ratio": 0.5}
    decision_a = policy.decide(row_a, cfg=cfg)
    tracker_a.observe_quality(row_a, decision_a, did_drop=bool(decision_a.would_drop))

    tracker_b = QCSummaryTracker(min_score=0.0, drop_near_dups=True)
    row_b = {"score": 10.0, "near_dup": True, "dup_family_id": "fam1", "path": "b.py", "ascii_ratio": 0.7}
    decision_b = policy.decide(row_b, cfg=cfg)
    tracker_b.observe_quality(row_b, decision_b, did_drop=bool(decision_b.would_drop))

    tracker_a.merge(tracker_b)

    quality = tracker_a.screeners["quality"]
    assert quality.scored == 2
    assert quality.dropped == 1
    assert tracker_a.dup_families["fam1"]["count"] == 2
    assert quality.signal_stats["ascii_ratio"].count == 2


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
    assert quality.kept == 1
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
    quality_stats = stats.qc.get_screener("quality", create=False)
    assert quality_stats is not None
    assert quality_stats.errors == 1


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
    safety_stats = stats.qc.get_screener("safety", create=False)
    assert safety_stats is not None
    assert safety_stats.errors == 1


def test_process_record_records_custom_error_without_rollback():
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
    assert quality_stats.kept == 1
    assert quality_stats.scored == 1


class AdvisoryExplodingScreener:
    def __init__(self, sid: str):
        self.id = sid
        self.enforce_drops = False

    def process_record(self, record):
        raise RuntimeError("oops")


def test_process_record_advisory_exception_keeps_record_and_counts_error():
    qc_cfg = QCConfig(enabled=True, min_score=0.0, mode=QCMode.ADVISORY)
    advisory = AdvisoryExplodingScreener("advisory_x")
    controller = InlineScreeningController(
        summary=None,
        screeners=[advisory],
        logger=DummyLogger(),
        qc_cfg=qc_cfg,
    )
    stats = PipelineStats()
    controller.reset(stats, qc_cfg=qc_cfg)

    kept = controller.process_record({"text": "keep me", "meta": {"path": "ok.py"}})

    assert kept is not None
    advisory_stats = stats.qc.screeners["advisory_x"]
    assert advisory_stats.errors == 1
