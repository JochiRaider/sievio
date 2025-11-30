import pytest

from repocapsule.core.builder import _prepare_qc
from repocapsule.core.config import QCConfig, QCMode, RepocapsuleConfig, SafetyConfig
from repocapsule.core.pipeline import PipelineStats
from repocapsule.core.qc_controller import InlineQCController, InlineQCHook
from repocapsule.core.qc_post import PostQCHook
from repocapsule.core.registries import QualityScorerRegistry, SafetyScorerRegistry


class ConstantQCScorer:
    def __init__(self, result):
        self.result = result

    def score_record(self, record):
        return self.result


class ConstantSafetyScorer:
    def __init__(self, result):
        self.result = result

    def score_record(self, record):
        return self.result


class ErrorSafetyScorer:
    def __init__(self, exc: Exception):
        self.exc = exc

    def score_record(self, record):
        raise self.exc


class ConstantQCFactory:
    id = "qc"

    def __init__(self, result):
        self.result = result

    def build(self, options):
        return ConstantQCScorer(self.result)


class ConstantSafetyFactory:
    id = "safety"

    def __init__(self, result):
        self.result = result

    def build(self, options):
        return ConstantSafetyScorer(self.result)


def test_post_qc_mode_skips_inline():
    qc_reg = QualityScorerRegistry()
    qc_reg.register(ConstantQCFactory({"score": 10.0, "near_dup": False}))
    safety_reg = SafetyScorerRegistry()

    cfg = RepocapsuleConfig()
    cfg.qc.enabled = True
    cfg.qc.mode = QCMode.POST
    cfg.qc.min_score = 0.0
    cfg.qc.safety.enabled = False

    res = _prepare_qc(cfg, scorer_registry=qc_reg, safety_scorer_registry=safety_reg)

    assert all(isinstance(h, PostQCHook) for h in res.hooks)
    assert res.post_qc_scorer is not None
    assert not any(isinstance(h, InlineQCHook) for h in res.hooks)


def test_inline_safety_without_qc_drops():
    qc_reg = QualityScorerRegistry()
    safety_reg = SafetyScorerRegistry()
    safety_reg.register(ConstantSafetyFactory({"safety_decision": "drop", "safety_flags": {"flagged": True}}))

    cfg = RepocapsuleConfig()
    cfg.qc.enabled = False
    cfg.qc.mode = QCMode.OFF
    cfg.qc.safety.enabled = True
    cfg.qc.safety.mode = QCMode.INLINE
    cfg.qc.safety.annotate_only = False

    res = _prepare_qc(cfg, scorer_registry=qc_reg, safety_scorer_registry=safety_reg)

    assert len(res.hooks) == 1
    inline_hook = res.hooks[0]
    assert isinstance(inline_hook, InlineQCHook)

    controller = inline_hook._controller  # type: ignore[attr-defined]
    stats = PipelineStats()
    controller.reset(stats)
    kept = controller.process_record({"text": "x", "meta": {"path": "safety_only.py"}})

    assert kept is None
    assert stats.qc.safety_enabled is True
    assert stats.qc.safety_dropped == 1
    assert stats.qc.safety_flags.get("flagged") == 1


def test_qc_inline_safety_advisory_annotates_only():
    qc_reg = QualityScorerRegistry()
    qc_reg.register(ConstantQCFactory({"score": 10.0, "near_dup": False}))
    safety_reg = SafetyScorerRegistry()
    safety_reg.register(ConstantSafetyFactory({"safety_decision": "keep", "safety_flags": {"note": True}}))

    cfg = RepocapsuleConfig()
    cfg.qc.enabled = True
    cfg.qc.mode = QCMode.INLINE
    cfg.qc.min_score = 0.0
    cfg.qc.safety.enabled = True
    cfg.qc.safety.mode = QCMode.ADVISORY
    cfg.qc.safety.annotate_only = True

    res = _prepare_qc(cfg, scorer_registry=qc_reg, safety_scorer_registry=safety_reg)
    inline_hook = next(h for h in res.hooks if isinstance(h, InlineQCHook))
    controller = inline_hook._controller  # type: ignore[attr-defined]

    stats = PipelineStats()
    controller.reset(stats)
    record = {"text": "hello", "meta": {"path": "foo.py"}}
    kept = controller.process_record(record)

    assert kept is record
    assert stats.qc.safety_dropped == 0
    safety_meta = record["meta"]["extra"]["safety"]
    assert safety_meta["safety_flags"]["note"] is True


def test_safety_post_mode_rejected():
    cfg = RepocapsuleConfig()
    cfg.qc.enabled = False
    cfg.qc.safety.enabled = True
    cfg.qc.safety.mode = QCMode.POST

    with pytest.raises(ValueError):
        cfg.validate()


def test_screening_summary_includes_safety_stats():
    qc_cfg = QCConfig(enabled=True, mode=QCMode.INLINE, min_score=0.0)
    safety_cfg = SafetyConfig(enabled=True, mode=QCMode.INLINE, annotate_only=False)
    qc_cfg.safety = safety_cfg
    controller = InlineQCController(
        config=qc_cfg,
        stats=None,
        scorer=ConstantQCScorer({"score": 99.0, "near_dup": False}),
        logger=None,
        enforce_drops=True,
        safety_scorer=ConstantSafetyScorer({"safety_decision": "drop", "safety_flags": {"p1": True}}),
        safety_cfg=safety_cfg,
    )

    controller.process_record({"text": "hi", "meta": {"path": "bar.py"}})
    summary = controller.tracker.as_dict()

    assert summary["safety"]["enabled"] is True
    assert summary["safety"]["scored"] == 1
    assert summary["safety"]["dropped"] == 1
    assert summary["safety"]["flags"]["p1"] == 1
