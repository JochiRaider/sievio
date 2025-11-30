from dataclasses import dataclass

import pytest

from repocapsule.core.config import QCConfig, QCMode
from repocapsule.core.qc_controller import InlineQCController, QCSummaryTracker
from repocapsule.core.records import ensure_meta_dict


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
