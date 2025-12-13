import pytest

from sievio.core.stats_aggregate import merge_pipeline_stats


def test_merge_pipeline_stats_sums_top_level_counts():
    stats = [
        {"files": 2, "bytes": 100, "records": 5, "sink_errors": 1, "source_errors": 0, "by_ext": {".py": 2}},
        {"files": 3, "bytes": 50, "records": 7, "sink_errors": 0, "source_errors": 2, "by_ext": {".md": 1}},
    ]

    merged = merge_pipeline_stats(stats)

    assert merged["files"] == 5
    assert merged["bytes"] == 150
    assert merged["records"] == 12
    assert merged["sink_errors"] == 1
    assert merged["source_errors"] == 2
    assert merged["by_ext"] == {".py": 2, ".md": 1}


def test_merge_pipeline_stats_qc_counts_only():
    stats = [
        {
            "qc": {
                "enabled": True,
                "mode": "inline",
                "min_score": 0.1,
                "drop_near_dups": True,
                "scored": 10,
                "kept": 8,
                "dropped_low_score": 2,
                "dropped_near_dup": 1,
                "errors": 0,
                "candidates_low_score": 4,
                "candidates_near_dup": 2,
                "safety": {"enabled": False, "scored": 0, "dropped": 0, "errors": 0, "flags": {}},
            }
        },
        {
            "qc": {
                "enabled": True,
                "mode": "inline",
                "min_score": 0.1,
                "drop_near_dups": True,
                "scored": 5,
                "kept": 3,
                "dropped_low_score": 1,
                "dropped_near_dup": 0,
                "errors": 1,
                "candidates_low_score": 1,
                "candidates_near_dup": 0,
                "safety": {"enabled": False, "scored": 0, "dropped": 0, "errors": 0, "flags": {}},
            }
        },
    ]

    merged = merge_pipeline_stats(stats)
    qc = merged["qc"]

    assert qc["scored"] == 15
    assert qc["kept"] == 11
    assert qc["dropped_low_score"] == 3
    assert qc["dropped_near_dup"] == 1
    assert qc["errors"] == 1
    assert qc["candidates_low_score"] == 5
    assert qc["candidates_near_dup"] == 2


def test_merge_pipeline_stats_safety_flags():
    stats = [
        {
            "qc": {
                "enabled": True,
                "mode": "inline",
                "min_score": 0.0,
                "drop_near_dups": False,
                "scored": 1,
                "kept": 1,
                "dropped_low_score": 0,
                "dropped_near_dup": 0,
                "errors": 0,
                "candidates_low_score": 0,
                "candidates_near_dup": 0,
                "safety": {"enabled": True, "scored": 1, "dropped": 0, "errors": 0, "flags": {"pii": 1}},
            }
        },
        {
            "qc": {
                "enabled": True,
                "mode": "inline",
                "min_score": 0.0,
                "drop_near_dups": False,
                "scored": 2,
                "kept": 1,
                "dropped_low_score": 1,
                "dropped_near_dup": 0,
                "errors": 0,
                "candidates_low_score": 0,
                "candidates_near_dup": 0,
                "safety": {"enabled": True, "scored": 2, "dropped": 1, "errors": 0, "flags": {"pii": 2, "toxicity": 1}},
            }
        },
    ]

    merged = merge_pipeline_stats(stats)
    flags = merged["qc"]["safety"]["flags"]

    assert flags == {"pii": 3, "toxicity": 1}


def test_merge_pipeline_stats_resets_signal_stats_and_top_dup_families():
    stats = [
        {
            "qc": {
                "enabled": True,
                "mode": "inline",
                "min_score": 0.0,
                "drop_near_dups": False,
                "scored": 1,
                "kept": 1,
                "dropped_low_score": 0,
                "dropped_near_dup": 0,
                "errors": 0,
                "candidates_low_score": 0,
                "candidates_near_dup": 0,
                "signal_stats": {"foo": 1},
                "top_dup_families": ["abc"],
                "safety": {"enabled": False, "scored": 0, "dropped": 0, "errors": 0, "flags": {}},
            }
        }
    ]

    merged = merge_pipeline_stats(stats)
    qc = merged["qc"]

    assert qc["signal_stats"] == {}
    assert qc["top_dup_families"] == []


def test_merge_pipeline_stats_handles_missing_qc():
    stats = [
        {"files": 1, "bytes": 10, "records": 2},
        {
            "files": 0,
            "bytes": 0,
            "records": 0,
            "qc": {
                "enabled": True,
                "mode": "inline",
                "min_score": 0.0,
                "drop_near_dups": False,
                "scored": 3,
                "kept": 2,
                "dropped_low_score": 1,
                "dropped_near_dup": 0,
                "errors": 0,
                "candidates_low_score": 0,
                "candidates_near_dup": 0,
                "safety": {"enabled": False, "scored": 0, "dropped": 0, "errors": 0, "flags": {}},
            },
        },
    ]

    merged = merge_pipeline_stats(stats)

    assert merged["files"] == 1
    assert merged["qc"]["scored"] == 3
    assert merged["qc"]["kept"] == 2
