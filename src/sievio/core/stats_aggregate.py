# stats_aggregate.py
# SPDX-License-Identifier: MIT
"""
Aggregation helpers for PipelineStats.as_dict() outputs.

Given multiple stats dictionaries, produce a merged view that sums
counts, merges by-extension tallies, and aggregates QC/safety metrics
while clearing non-additive QC fields.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence


def _init_qc_template(sample_qc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build an aggregation template using config fields from sample_qc and zeroed counts.
    """
    tmpl: Dict[str, Any] = {
        "enabled": sample_qc.get("enabled"),
        "mode": sample_qc.get("mode"),
        "min_score": sample_qc.get("min_score"),
        "drop_near_dups": sample_qc.get("drop_near_dups"),
        "scored": 0,
        "kept": 0,
        "dropped_low_score": 0,
        "dropped_near_dup": 0,
        "errors": 0,
        "candidates_low_score": 0,
        "candidates_near_dup": 0,
    }
    safety = sample_qc.get("safety") or {}
    tmpl["safety"] = {
        "enabled": bool(safety.get("enabled", False)),
        "scored": 0,
        "dropped": 0,
        "errors": 0,
        "flags": {},
    }
    tmpl["signal_stats"] = {}
    tmpl["top_dup_families"] = []
    return tmpl


def _accumulate_qc_counts(out_qc: Dict[str, Any], qc: Dict[str, Any]) -> None:
    """
    Add numeric QC counters and merge safety flag counts into out_qc.
    """
    for field in (
        "scored",
        "kept",
        "dropped_low_score",
        "dropped_near_dup",
        "errors",
        "candidates_low_score",
        "candidates_near_dup",
    ):
        out_qc[field] += int(qc.get(field, 0))

    safety_out = out_qc["safety"]
    safety_in = qc.get("safety") or {}
    for key in ("scored", "dropped", "errors"):
        safety_out[key] += int(safety_in.get(key, 0))

    flags_out = safety_out["flags"]
    for flag, count in (safety_in.get("flags") or {}).items():
        flags_out[flag] = flags_out.get(flag, 0) + int(count)


def merge_pipeline_stats(stats_dicts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge a sequence of PipelineStats.as_dict()-style dictionaries.
    """
    merged: Dict[str, Any] = {
        "files": 0,
        "bytes": 0,
        "records": 0,
        "sink_errors": 0,
        "source_errors": 0,
        "middleware_errors": 0,
        "by_ext": {},
        "qc": {},
    }
    qc_agg: Dict[str, Any] | None = None

    for data in stats_dicts:
        merged["files"] += int(data.get("files", 0))
        merged["bytes"] += int(data.get("bytes", 0))
        merged["records"] += int(data.get("records", 0))
        merged["sink_errors"] += int(data.get("sink_errors", 0))
        merged["source_errors"] += int(data.get("source_errors", 0))
        merged["middleware_errors"] += int(data.get("middleware_errors", 0))

        ext_counts = data.get("by_ext") or {}
        for ext, count in ext_counts.items():
            merged["by_ext"][ext] = merged["by_ext"].get(ext, 0) + int(count)

        qc = data.get("qc")
        if not qc:
            continue
        if qc_agg is None:
            qc_agg = _init_qc_template(qc)
        else:
            if qc_agg.get("mode") != qc.get("mode") or qc_agg.get("min_score") != qc.get("min_score"):
                raise ValueError("Inconsistent QC config across stats (mode/min_score).")
            if qc_agg.get("drop_near_dups") != qc.get("drop_near_dups"):
                raise ValueError("Inconsistent QC config across stats (drop_near_dups).")
            if (qc_agg.get("safety") or {}).get("enabled") != (qc.get("safety") or {}).get("enabled"):
                raise ValueError("Inconsistent QC safety.enabled across stats.")
        _accumulate_qc_counts(qc_agg, qc)

    merged["qc"] = qc_agg or {}
    return merged
