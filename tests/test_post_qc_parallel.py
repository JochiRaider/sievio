from __future__ import annotations

import json
from pathlib import Path

from repocapsule.config import RepocapsuleConfig
from repocapsule.runner import _run_post_qc  # type: ignore[attr-defined]


class FakeScorer:
    def __init__(self):
        self._count = 0

    def score_record(self, record):
        self._count += 1
        meta = record.get("meta") or {}
        doc_id = meta.get("path", f"doc-{self._count}")
        return {
            "doc_id": doc_id,
            "score": 99.0,
            "tokens": 10,
            "len": len(record.get("text", "")),
            "parse_ok": True,
            "repetition": 0.0,
            "ascii_ratio": 1.0,
            "code_complexity": 0.0,
            "gopher_quality": 1.0,
            "gopher_flags": {},
            "near_dup": False,
            "near_dup_minhash": False,
            "near_dup_simhash": False,
            "minhash_jaccard": 0.0,
            "minhash_dup_of": None,
            "simhash_dup_of": None,
            "dup_family_id": doc_id,
            "perplexity": None,
        }

    def clone_for_parallel(self):
        return FakeScorer()


def _write_jsonl(path: Path, records):
    with open(path, "w", encoding="utf-8") as fp:
        for rec in records:
            fp.write(json.dumps(rec))
            fp.write("\n")


def test_post_qc_parallel_scoring(tmp_path: Path):
    jsonl_path = tmp_path / "sample.jsonl"
    records = [
        {"text": "one", "meta": {"path": "one.txt"}},
        {"text": "two", "meta": {"path": "two.txt"}},
        {"text": "three", "meta": {"path": "three.txt"}},
        {"text": "", "meta": {"kind": "run_summary"}},
    ]
    _write_jsonl(jsonl_path, records)

    cfg = RepocapsuleConfig()
    cfg.qc.enabled = True
    cfg.qc.mode = "post"
    cfg.qc.parallel_post = True
    cfg.qc.write_csv = False
    cfg.qc.scorer = FakeScorer()
    cfg.pipeline.max_workers = 2

    summary = _run_post_qc(str(jsonl_path), cfg)

    assert summary["scored"] == 3
    assert summary["kept"] == 3
