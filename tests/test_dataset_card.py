from pathlib import Path
import json

import pytest

from sievio.core.config import SievioConfig
from sievio.core.builder import build_pipeline_plan
from sievio.core.pipeline import PipelineStats
from sievio.core.dataset_card import (
    CardFragment,
    DatasetCardHook,
    build_dataset_card_from_fragments,
    build_card_fragment_for_run,
    load_card_fragment,
    merge_fragments,
    write_card_fragment_for_run,
)


def test_card_fragment_round_trip() -> None:
    frag = CardFragment(
        file="data.jsonl",
        split="train",
        num_examples=5,
        num_bytes=10,
        language=["en"],
        multilinguality="monolingual",
        license="MIT",
        size_categories="n<1K",
        task_categories=["text-classification"],
        task_ids="tc",
        tags=["modality:text"],
        source_repos=["https://example.com"],
        extra={"run": 1},
    )

    payload = frag.to_dict()
    reloaded = CardFragment.from_dict(payload)
    assert reloaded == frag


def test_merge_fragments_union_splits_and_langs() -> None:
    frag_train = CardFragment(
        file="a.jsonl",
        split="train",
        num_examples=800,
        num_bytes=100,
        language=["en"],
        multilinguality="monolingual",
        license="MIT",
    )
    frag_val = CardFragment(
        file="b.jsonl",
        split="validation",
        num_examples=400,
        num_bytes=50,
        language=["fr"],
        multilinguality="monolingual",
        license="MIT",
    )

    fields = merge_fragments([frag_train, frag_val])
    assert fields.language == ["en", "fr"]
    assert fields.multilinguality == "multilingual"
    assert fields.size_categories == ["1K<n<10K"]

    splits = fields.dataset_info["splits"]  # type: ignore[index]
    split_names = {s["name"] for s in splits}
    assert {"train", "validation"} == split_names


def test_render_dataset_card_contains_yaml_and_sections(tmp_path: Path) -> None:
    frag = CardFragment(
        file="c.jsonl",
        split="train",
        num_examples=1500,
        num_bytes=200,
        language=["en"],
        multilinguality="monolingual",
        license="Apache-2.0",
    )
    frag_path = tmp_path / "c.jsonl.card.json"
    frag_path.write_text(json.dumps(frag.to_dict()), encoding="utf-8")

    card_md = build_dataset_card_from_fragments([frag_path], overrides={"pretty_name": "Demo"})
    assert "# Dataset Card for Demo" in card_md
    assert "language:" in card_md
    assert "size_categories:" in card_md
    assert "## Dataset Description" in card_md
    assert "### Dataset Summary" in card_md
    assert "### Quality Signals" in card_md


def test_dataset_card_renders_quality_signals_section(tmp_path: Path) -> None:
    frag = CardFragment(
        file="signals.jsonl",
        split="train",
        num_examples=2,
        num_bytes=10,
        language=["en"],
        multilinguality="monolingual",
        license="MIT",
        extra={
            "stats": {
                "qc": {
                    "signal_stats": {
                        "len_tok": {"count": 2, "mean": 100, "min": 90, "max": 110, "stdev": 10},
                        "ascii_ratio": {"count": 2, "mean": 0.9, "min": 0.8, "max": 1.0, "stdev": 0.1},
                    }
                }
            }
        },
    )
    frag_path = tmp_path / "signals.jsonl.card.json"
    frag_path.write_text(json.dumps(frag.to_dict()), encoding="utf-8")

    card_md = build_dataset_card_from_fragments([frag_path], overrides={"pretty_name": "Signals Demo"})

    assert "Tokens per record: mean=100.00" in card_md
    assert "ASCII ratio: mean=0.90" in card_md


def test_dataset_card_renders_quality_signals_from_qc_summary(tmp_path: Path) -> None:
    frag = CardFragment(
        file="signals.jsonl",
        split="train",
        num_examples=2,
        num_bytes=10,
        language=["en"],
        multilinguality="monolingual",
        license="MIT",
        extra={
            "stats": {
                "qc_summary": {
                    "signal_stats": {
                        "len_tok": {"count": 2, "mean": 50, "min": 40, "max": 60, "stdev": 10},
                        "perplexity": {"count": 2, "mean": 20, "min": 10, "max": 30, "stdev": 5},
                    }
                }
            }
        },
    )
    frag_path = tmp_path / "signals.jsonl.card.json"
    frag_path.write_text(json.dumps(frag.to_dict()), encoding="utf-8")

    card_md = build_dataset_card_from_fragments([frag_path], overrides={"pretty_name": "QC Summary Demo"})

    assert "Tokens per record: mean=50.00" in card_md
    assert "Perplexity: mean=20.00" in card_md


def test_dataset_card_includes_screening_summary_and_safety_note(tmp_path: Path) -> None:
    frag = CardFragment(
        file="signals.jsonl",
        split="train",
        num_examples=2,
        num_bytes=10,
        language=["en"],
        multilinguality="monolingual",
        license="MIT",
        extra={
            "stats": {
                "qc_summary": {
                    "signal_stats": {
                        "len_tok": {"count": 2, "mean": 75, "min": 50, "max": 100, "stdev": 15},
                    },
                    "screeners": {
                        "quality": {
                            "scored": 3,
                            "kept": 2,
                            "dropped": 1,
                            "errors": 0,
                            "candidates": {"low_score": 1},
                            "drops": {"low_score": 1},
                        },
                        "safety": {
                            "scored": 2,
                            "kept": 1,
                            "dropped": 1,
                            "errors": 1,
                            "flags": {"pii": 2, "toxicity": 1},
                        },
                    },
                }
            }
        },
    )
    frag_path = tmp_path / "signals.jsonl.card.json"
    frag_path.write_text(json.dumps(frag.to_dict()), encoding="utf-8")

    card_md = build_dataset_card_from_fragments([frag_path], overrides={"pretty_name": "Screeners Demo"})

    assert "#### Screening Summary" in card_md
    assert "#### Scalar Quality Signals" in card_md
    assert "- quality: scored=3, kept=2, dropped=1, errors=0 (candidates: low_score=1; drops: low_score=1)" in card_md
    assert "- safety: scored=2, kept=1, dropped=1, errors=1 (top flags: pii (2), toxicity (1))" in card_md
    assert "Automated safety screening was applied; see Screening Summary for aggregate counts and top flag categories." in card_md


def test_write_card_fragment_for_run(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "data.jsonl"
    sample_line = {"text": "hello", "meta": {"language": "en"}}
    jsonl_path.write_text(json.dumps(sample_line) + "\n", encoding="utf-8")

    cfg = SievioConfig()
    cfg.metadata.primary_jsonl = str(jsonl_path)

    stats = PipelineStats(records=1)
    sidecar_path = write_card_fragment_for_run(cfg, stats)

    assert sidecar_path.exists()
    data = load_card_fragment(sidecar_path)
    assert data.file == "data.jsonl"
    assert data.num_examples == 1


def test_dataset_card_enabled_by_default() -> None:
    cfg = SievioConfig()

    plan = build_pipeline_plan(cfg, load_plugins=False)

    hooks = plan.runtime.lifecycle_hooks
    assert any(isinstance(h, DatasetCardHook) for h in hooks)


def test_dataset_card_disabled_skips_hook() -> None:
    cfg = SievioConfig()
    cfg.dataset_card.enabled = False

    plan = build_pipeline_plan(cfg, load_plugins=False)

    hooks = plan.runtime.lifecycle_hooks
    assert not any(isinstance(h, DatasetCardHook) for h in hooks)
