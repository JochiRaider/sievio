import json
from pathlib import Path

import pytest

from sievio.core.config import DEFAULT_QC_SCORER_ID, QCHeuristics
from sievio.core.extras.qc import DefaultQualityScorerFactory, JSONLQualityScorer
from sievio.cli import runner
from sievio.core import builder
from sievio.core.builder import build_pipeline_plan
from sievio.core.config import SievioConfig, SinkSpec, SourceSpec
from sievio.core.interfaces import RepoContext
from sievio.core.pipeline import PipelineEngine


def _make_basic_config(tmp_path: Path) -> SievioConfig:
    cfg = SievioConfig()
    ctx = RepoContext(
        repo_full_name="local/test",
        repo_url="https://example.com/local",
        license_id="UNKNOWN",
    )
    cfg.sinks.context = ctx

    src_root = tmp_path / "input"
    src_root.mkdir()
    (src_root / "file.txt").write_text("hello", encoding="utf-8")
    cfg.sources.specs = (SourceSpec(kind="local_dir", options={"root_dir": str(src_root)}),)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    jsonl_path = out_dir / "data.jsonl"
    prompt_path = out_dir / "data.prompt.txt"

    cfg.sinks.specs = (
        SinkSpec(
            kind="default_jsonl_prompt",
            options={
                "jsonl_path": str(jsonl_path),
                "prompt_path": str(prompt_path),
            },
        ),
    )
    return cfg


def test_default_scorer_id_constant_and_factory():
    assert DEFAULT_QC_SCORER_ID == "jsonl_default"
    assert DefaultQualityScorerFactory.id == DEFAULT_QC_SCORER_ID


def test_default_factory_validates_heuristics():
    h = QCHeuristics()
    h.target_code_min = 0
    factory = DefaultQualityScorerFactory()

    with pytest.raises(ValueError):
        factory.build({"heuristics": h})

    with pytest.raises(ValueError):
        factory.build({"heuristics": {"code_punct_weight": 1.5}})


def test_jsonl_quality_scorer_clone_preserves_config_and_state_isolated():
    heur = QCHeuristics(simhash_window=8, simhash_hamm_thresh=6, enable_minhash=True, minhash_shingle_k=7)
    scorer = JSONLQualityScorer(heuristics=heur, enable_gopher=False, gopher_weight=0.2)
    clone = scorer.clone_for_parallel()

    assert type(clone) is type(scorer)
    assert clone is not scorer
    assert clone.sim_thresh == scorer.sim_thresh
    assert clone.enable_minhash == scorer.enable_minhash
    assert clone.minhash_k == scorer.minhash_k
    assert clone.gopher_weight == pytest.approx(scorer.gopher_weight)
    assert clone.last_stats is None

    rec = {"text": "hello world", "meta": {"lang": "text", "tokens": 2}}
    scorer.score_record(rec)
    assert len(scorer.sim_seen) == 1
    assert len(clone.sim_seen) == 0  # clone has independent state


def test_parallel_qc_without_clone_uses_factory(monkeypatch, tmp_path: Path):
    cfg = _make_basic_config(tmp_path)
    cfg.qc.enabled = True
    cfg.qc.mode = "post"
    cfg.qc.parallel_post = True
    cfg.qc.post_executor_kind = "thread"
    cfg.qc.scorer = None

    class DummyScorer:
        def score_record(self, record):
            meta = record.setdefault("meta", {})
            meta["qc_checked"] = True
            return {"score": 1, "path": meta.get("path"), "tokens": meta.get("tokens", 0)}

        def score_jsonl_path(self, path):
            return []

    def fake_make_qc_scorer(qc_cfg, new_instance=False, scorer_registry=None):
        return DummyScorer()

    monkeypatch.setattr(builder, "make_qc_scorer", fake_make_qc_scorer)

    plan = build_pipeline_plan(cfg, mutate=False)
    engine = PipelineEngine(plan)
    stats_dict = runner.run_engine(engine)

    assert stats_dict["records"] == 1
    assert stats_dict["qc"]["enabled"] is True
