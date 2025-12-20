
from sievio.core.stats_aggregate import merge_pipeline_stats


def test_merge_pipeline_stats_sums_top_level_counts():
    stats = [
        {
            "files": 2,
            "attempted_files": 3,
            "bytes": 100,
            "records": 5,
            "sink_errors": 1,
            "source_errors": 0,
            "middleware_errors": 1,
            "by_ext": {".py": 2},
        },
        {
            "files": 3,
            "attempted_files": 4,
            "bytes": 50,
            "records": 7,
            "sink_errors": 0,
            "source_errors": 2,
            "middleware_errors": 2,
            "by_ext": {".md": 1},
        },
    ]

    merged = merge_pipeline_stats(stats)

    assert merged["files"] == 5
    assert merged["attempted_files"] == 7
    assert merged["bytes"] == 150
    assert merged["records"] == 12
    assert merged["sink_errors"] == 1
    assert merged["source_errors"] == 2
    assert merged["middleware_errors"] == 3
    assert merged["by_ext"] == {".py": 2, ".md": 1}


def test_merge_pipeline_stats_qc_counts_only():
    stats = [
        {
            "qc": {
                "enabled": True,
                "mode": "inline",
                "min_score": 0.1,
                "drop_near_dups": True,
                "screeners": {
                    "quality": {
                        "id": "quality",
                        "mode": "inline",
                        "scored": 10,
                        "kept": 8,
                        "dropped": 3,
                        "drops": {"low_score": 2, "near_dup": 1},
                        "errors": 0,
                        "candidates": {"low_score": 4, "near_dup": 2},
                        "flags": {},
                    }
                },
            }
        },
        {
            "qc": {
                "enabled": True,
                "mode": "inline",
                "min_score": 0.1,
                "drop_near_dups": True,
                "screeners": {
                    "quality": {
                        "id": "quality",
                        "mode": "inline",
                        "scored": 5,
                        "kept": 3,
                        "dropped": 1,
                        "drops": {"low_score": 1},
                        "errors": 1,
                        "candidates": {"low_score": 1, "near_dup": 0},
                        "flags": {},
                    }
                },
            }
        },
    ]

    merged = merge_pipeline_stats(stats)
    quality = merged["qc"]["screeners"]["quality"]

    assert quality["scored"] == 15
    assert quality["kept"] == 11
    assert quality["drops"]["low_score"] == 3
    assert quality["drops"]["near_dup"] == 1
    assert quality["errors"] == 1
    assert quality["candidates"]["low_score"] == 5
    assert quality["candidates"]["near_dup"] == 2


def test_merge_pipeline_stats_safety_flags():
    stats = [
        {
            "qc": {
                "enabled": True,
                "mode": "inline",
                "min_score": 0.0,
                "drop_near_dups": False,
                "screeners": {
                    "quality": {"id": "quality", "mode": "inline", "scored": 1, "kept": 1, "dropped": 0, "errors": 0},
                    "safety": {
                        "id": "safety",
                        "mode": "inline",
                        "scored": 1,
                        "dropped": 0,
                        "errors": 0,
                        "flags": {"pii": 1},
                    },
                },
            }
        },
        {
            "qc": {
                "enabled": True,
                "mode": "inline",
                "min_score": 0.0,
                "drop_near_dups": False,
                "screeners": {
                    "quality": {"id": "quality", "mode": "inline", "scored": 2, "kept": 1, "dropped": 1, "errors": 0},
                    "safety": {
                        "id": "safety",
                        "mode": "inline",
                        "scored": 2,
                        "dropped": 1,
                        "errors": 0,
                        "flags": {"pii": 2, "toxicity": 1},
                    },
                },
            }
        },
    ]

    merged = merge_pipeline_stats(stats)
    flags = merged["qc"]["screeners"]["safety"]["flags"]

    assert flags == {"pii": 3, "toxicity": 1}


def test_merge_pipeline_stats_resets_signal_stats_and_top_dup_families():
    stats = [
        {
            "qc": {
                "enabled": True,
                "mode": "inline",
                "min_score": 0.0,
                "drop_near_dups": False,
                "top_dup_families": ["abc"],
                "screeners": {
                    "quality": {
                        "id": "quality",
                        "mode": "inline",
                        "scored": 1,
                        "kept": 1,
                        "dropped": 0,
                        "errors": 0,
                        "signal_stats": {"foo": {"count": 1, "mean": 1.0, "min": 1.0, "max": 1.0, "stdev": 0.0}},
                    }
                },
            }
        }
    ]

    merged = merge_pipeline_stats(stats)
    qc = merged["qc"]
    quality = qc["screeners"]["quality"]

    assert quality["signal_stats"] == {}
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
                "screeners": {
                    "quality": {
                        "id": "quality",
                        "mode": "inline",
                        "scored": 3,
                        "kept": 2,
                        "dropped": 1,
                        "drops": {"low_score": 1},
                        "errors": 0,
                        "candidates": {},
                    }
                },
            },
        },
    ]

    merged = merge_pipeline_stats(stats)

    assert merged["files"] == 1
    quality = merged["qc"]["screeners"]["quality"]
    assert quality["scored"] == 3
    assert quality["kept"] == 2
