import json
import types

import pytest

from sievio.core.builder import _prepare_qc
from sievio.core.config import QCConfig, QCMode, SievioConfig, SafetyConfig
from sievio.core.interfaces import RunContext
from sievio.core.pipeline import PipelineStats
from sievio.core.qc_controller import (
    InlineQCController,
    InlineQCHook,
    QualityInlineScreener,
    SafetyInlineScreener,
)
from sievio.core.qc_post import PostQCHook, PostSafetyHook, run_safety_over_jsonl
from sievio.core.registries import QualityScorerRegistry, SafetyScorerRegistry


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


def _make_runtime(**kwargs):
    defaults = {"sinks": (), "lifecycle_hooks": ()}
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)


def test_post_qc_mode_skips_inline():
    qc_reg = QualityScorerRegistry()
    qc_reg.register(ConstantQCFactory({"score": 10.0, "near_dup": False}))
    safety_reg = SafetyScorerRegistry()

    cfg = SievioConfig()
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

    cfg = SievioConfig()
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

    cfg = SievioConfig()
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


def test_safety_post_mode_allowed():
    cfg = SievioConfig()
    cfg.qc.enabled = False
    cfg.qc.safety.enabled = True
    cfg.qc.safety.mode = QCMode.POST

    cfg.validate()
    assert cfg.qc.safety.mode == QCMode.POST


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


def test_prepare_qc_post_mode_attaches_safety_only_inline():
    qc_reg = QualityScorerRegistry()
    qc_reg.register(ConstantQCFactory({"score": 10.0, "near_dup": False}))
    safety_reg = SafetyScorerRegistry()
    safety_reg.register(ConstantSafetyFactory({"safety_decision": "drop"}))

    cfg = SievioConfig()
    cfg.qc.enabled = True
    cfg.qc.mode = QCMode.POST
    cfg.qc.min_score = 0.0
    cfg.qc.safety.enabled = True
    cfg.qc.safety.mode = QCMode.INLINE
    cfg.qc.safety.annotate_only = False

    res = _prepare_qc(cfg, scorer_registry=qc_reg, safety_scorer_registry=safety_reg)

    inline_hook = next(h for h in res.hooks if isinstance(h, InlineQCHook))
    controller = inline_hook._controller  # type: ignore[attr-defined]
    screeners = getattr(controller, "screeners", None)
    if screeners is None:  # fallback for InlineQCController wrapper
        inner = getattr(controller, "_controller", None)
        screeners = getattr(inner, "screeners", ())
    assert any(isinstance(s, SafetyInlineScreener) for s in screeners)
    assert not any(isinstance(s, QualityInlineScreener) for s in screeners)
    assert any(isinstance(h, PostQCHook) for h in res.hooks)


def test_prepare_qc_inline_mode_attaches_quality_and_safety():
    qc_reg = QualityScorerRegistry()
    qc_reg.register(ConstantQCFactory({"score": 10.0, "near_dup": False}))
    safety_reg = SafetyScorerRegistry()
    safety_reg.register(ConstantSafetyFactory({"safety_decision": "keep"}))

    cfg = SievioConfig()
    cfg.qc.enabled = True
    cfg.qc.mode = QCMode.INLINE
    cfg.qc.min_score = 0.0
    cfg.qc.safety.enabled = True
    cfg.qc.safety.mode = QCMode.INLINE
    cfg.qc.safety.annotate_only = False

    res = _prepare_qc(cfg, scorer_registry=qc_reg, safety_scorer_registry=safety_reg)
    inline_hook = next(h for h in res.hooks if isinstance(h, InlineQCHook))
    controller = inline_hook._controller  # type: ignore[attr-defined]
    screeners = getattr(controller, "screeners", None)
    if screeners is None:
        inner = getattr(controller, "_controller", None)
        screeners = getattr(inner, "screeners", ())
    assert any(isinstance(s, QualityInlineScreener) for s in screeners)
    assert any(isinstance(s, SafetyInlineScreener) for s in screeners)


def test_prepare_qc_safety_post_attaches_hook():
    qc_reg = QualityScorerRegistry()
    safety_reg = SafetyScorerRegistry()
    safety_reg.register(ConstantSafetyFactory({"safety_decision": "keep"}))

    cfg = SievioConfig()
    cfg.qc.enabled = False
    cfg.qc.mode = QCMode.OFF
    cfg.qc.safety.enabled = True
    cfg.qc.safety.mode = QCMode.POST

    res = _prepare_qc(cfg, scorer_registry=qc_reg, safety_scorer_registry=safety_reg)

    assert any(isinstance(h, PostSafetyHook) for h in res.hooks)
    assert not any(isinstance(h, InlineQCHook) for h in res.hooks)


def test_post_qc_merge_preserves_safety(tmp_path):
    qc_scorer = ConstantQCScorer({"score": 10.0, "near_dup": False})

    cfg = SievioConfig()
    cfg.qc.enabled = True
    cfg.qc.mode = QCMode.POST
    cfg.qc.min_score = 0.0
    cfg.qc.safety.enabled = True
    cfg.qc.safety.mode = QCMode.INLINE

    jsonl_path = tmp_path / "post_qc.jsonl"
    jsonl_path.write_text(json.dumps({"text": "hi", "meta": {"path": "foo.py"}}) + "\n", encoding="utf-8")
    cfg.sinks.primary_jsonl_name = str(jsonl_path)

    stats = PipelineStats()
    stats.qc.observe_safety({"safety_decision": "drop", "safety_flags": {"flagged": True}}, apply_gates=True)

    hook = PostQCHook(cfg.qc, qc_scorer)
    ctx = RunContext(cfg=cfg, stats=stats, runtime=_make_runtime(post_qc_scorer=qc_scorer))

    hook.on_run_end(ctx)

    assert ctx.stats.qc.safety_scored == 1
    assert ctx.stats.qc.screeners["quality"].scored == 1


def test_post_safety_hook_preserves_quality(tmp_path):
    safety_scorer = ConstantSafetyScorer({"safety_decision": "drop", "safety_flags": {"flagged": True}})

    cfg = SievioConfig()
    cfg.qc.enabled = False
    cfg.qc.mode = QCMode.OFF
    cfg.qc.safety.enabled = True
    cfg.qc.safety.mode = QCMode.POST

    jsonl_path = tmp_path / "post_safety.jsonl"
    jsonl_path.write_text(json.dumps({"text": "hi", "meta": {"path": "bar.py"}}) + "\n", encoding="utf-8")
    cfg.sinks.primary_jsonl_name = str(jsonl_path)

    stats = PipelineStats()
    stats.qc.observe({"score": 90.0, "near_dup": False}, apply_gates=False)

    hook = PostSafetyHook(cfg.qc.safety, safety_scorer)
    ctx = RunContext(cfg=cfg, stats=stats, runtime=_make_runtime(post_safety_scorer=safety_scorer))

    hook.on_run_end(ctx)

    assert ctx.stats.qc.screeners["quality"].scored == 1
    assert ctx.stats.qc.safety_dropped == 1


def test_run_safety_over_jsonl_emits_csv(tmp_path):
    cfg = SievioConfig()
    cfg.qc.enabled = False
    cfg.qc.safety.enabled = True
    cfg.qc.safety.mode = QCMode.POST
    cfg.qc.safety.write_csv = True

    jsonl_path = tmp_path / "safety.jsonl"
    jsonl_path.write_text(json.dumps({"text": "hello", "meta": {"path": "x.py"}}) + "\n", encoding="utf-8")

    summary, _rows = run_safety_over_jsonl(
        str(jsonl_path),
        cfg.qc.safety,
        config=cfg,
        scorer=ConstantSafetyScorer({"safety_decision": "drop", "safety_flags": {"flagged": True}}),
    )

    assert summary["safety"]["scored"] == 1
    assert summary["safety"]["dropped"] == 1

    csv_path = tmp_path / "safety_safety.csv"
    assert csv_path.exists()
    content = csv_path.read_text(encoding="utf-8")
    assert "safety_decision" in content.splitlines()[0]
