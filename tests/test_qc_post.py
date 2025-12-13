import csv
import json
from pathlib import Path

import pytest

from sievio.core.config import QCConfig, QCMode, SievioConfig
from sievio.core.qc_post import run_qc_over_jsonl


class DummyQCScorer:
    def score_record(self, record):
        meta = record.get("meta", {})
        doc_id = meta.get("doc_id")
        text = record.get("text", "")
        tokens = meta.get("tokens") or len(text.split())
        return {
            "score": 99.0,
            "doc_id": doc_id,
            "tokens": tokens,
            "len": len(text),
            "ascii_ratio": 0.8,
        }


def test_run_qc_over_jsonl_writes_signals_csv(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "data.jsonl"
    rows = [
        {"text": "hello world", "meta": {"doc_id": "a", "tokens": 5}},
        {"text": "another line", "meta": {"doc_id": "b", "tokens": 2}},
    ]
    jsonl_path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    qc_cfg = QCConfig(
        enabled=True,
        mode=QCMode.POST,
        write_signals_sidecar=True,
        signals_suffix="_signals.csv",
    )
    summary, _rows = run_qc_over_jsonl(
        str(jsonl_path),
        qc_cfg,
        config=SievioConfig(),
        scorer=DummyQCScorer(),
        write_signals_sidecar=True,
        signals_suffix="_signals.csv",
    )

    assert summary["signal_stats"].get("len_tok", {}).get("count") == 2

    signals_path = Path(str(jsonl_path)[:-6] + "_signals.csv")
    assert signals_path.exists()

    with signals_path.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        rows_out = list(reader)

    assert rows_out[0]["doc_id"] == "a"
    assert rows_out[0]["len_tok"] == "5"
    assert "ascii_ratio" in rows_out[0]


def test_run_qc_over_jsonl_writes_signals_parquet(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    jsonl_path = tmp_path / "data.jsonl"
    rows = [
        {"text": "hello world", "meta": {"doc_id": "a", "tokens": 5}},
        {"text": "another line", "meta": {"doc_id": "b", "tokens": 2}},
    ]
    jsonl_path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    qc_cfg = QCConfig(
        enabled=True,
        mode=QCMode.POST,
        write_signals_sidecar=True,
        signals_format="parquet",
    )
    summary, _rows = run_qc_over_jsonl(
        str(jsonl_path),
        qc_cfg,
        config=SievioConfig(),
        scorer=DummyQCScorer(),
        write_signals_sidecar=True,
        signals_format="parquet",
    )

    assert summary["signal_stats"].get("len_tok", {}).get("count") == 2

    signals_path = Path(str(jsonl_path)[:-6] + "_signals.parquet")
    assert signals_path.exists()

    table = pa.parquet.read_table(signals_path)
    assert "len_tok" in table.column_names
    data = table.to_pydict()
    assert data["doc_id"][0] == "a"
    assert data["len_tok"][1] == 2
