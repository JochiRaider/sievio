# stats_aggregate.py
# SPDX-License-Identifier: MIT
"""
Aggregation helpers for PipelineStats.as_dict() outputs.

Given multiple stats dictionaries, produce a merged view that sums
counts, merges by-extension tallies, and aggregates QC/safety metrics
while clearing non-additive QC fields.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _init_qc_template(sample_qc: dict[str, Any]) -> dict[str, Any]:
    """
    Build an aggregation template using config fields from sample_qc and zeroed counts.
    """
    tmpl: dict[str, Any] = {
        "schema_version": sample_qc.get("schema_version"),
        "enabled": sample_qc.get("enabled"),
        "mode": sample_qc.get("mode"),
        "min_score": sample_qc.get("min_score"),
        "drop_near_dups": sample_qc.get("drop_near_dups"),
        "top_dup_families": [],
        "screeners": {},
    }
    screeners = sample_qc.get("screeners") or {}
    if isinstance(screeners, Mapping):
        for sid, payload in screeners.items():
            if not isinstance(payload, Mapping):
                continue
            tmpl["screeners"][sid] = _init_screener_template(payload, default_id=str(sid))
    return tmpl


def _init_screener_template(sample: Mapping[str, Any], *, default_id: str) -> dict[str, Any]:
    """Build a zeroed-out screener payload preserving config fields."""
    return {
        "id": sample.get("id") or default_id,
        "enabled": bool(sample.get("enabled")),
        "mode": sample.get("mode"),
        "scored": 0,
        "kept": 0,
        "dropped": 0,
        "errors": 0,
        "signal_stats": {},
        "flags": {},
        "candidates": {},
        "drops": {},
    }


def _accumulate_qc_counts(out_qc: dict[str, Any], qc: dict[str, Any]) -> None:
    """
    Add numeric QC counters and merge safety flag counts into out_qc.
    """
    out_qc["enabled"] = out_qc.get("enabled") or qc.get("enabled")
    screeners = qc.get("screeners") or {}
    if not isinstance(screeners, Mapping):
        return
    out_screeners = out_qc.setdefault("screeners", {})
    for sid, payload in screeners.items():
        if not isinstance(payload, Mapping):
            continue
        bucket = out_screeners.get(sid)
        if bucket is None:
            bucket = _init_screener_template(payload, default_id=str(sid))
            out_screeners[sid] = bucket
        else:
            if bucket.get("mode") != payload.get("mode"):
                raise ValueError(f"Inconsistent screener mode for {sid}.")
        bucket["enabled"] = bucket.get("enabled") or bool(payload.get("enabled"))
        for key in ("scored", "kept", "dropped", "errors"):
            bucket[key] = int(bucket.get(key, 0)) + int(payload.get(key, 0))
        for key in ("flags", "candidates", "drops"):
            out_map = bucket.get(key) or {}
            for name, count in (payload.get(key) or {}).items():
                out_map[name] = out_map.get(name, 0) + int(count)
            bucket[key] = out_map
        bucket["signal_stats"] = {}  # non-additive; cleared during aggregation


def merge_pipeline_stats(stats_dicts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """
    Merge a sequence of PipelineStats.as_dict()-style dictionaries.
    """
    merged: dict[str, Any] = {
        "files": 0,
        "attempted_files": 0,
        "bytes": 0,
        "records": 0,
        "sink_errors": 0,
        "source_errors": 0,
        "middleware_errors": 0,
        "by_ext": {},
        "qc": {},
    }
    qc_agg: dict[str, Any] | None = None

    for data in stats_dicts:
        merged["files"] += int(data.get("files", 0))
        merged["attempted_files"] += int(data.get("attempted_files", 0))
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
            if (
                qc_agg.get("mode") != qc.get("mode")
                or qc_agg.get("min_score") != qc.get("min_score")
            ):
                raise ValueError(
                    "Inconsistent QC config across stats "
                    "(mode/min_score)."
                )
            if qc_agg.get("drop_near_dups") != qc.get("drop_near_dups"):
                raise ValueError(
                    "Inconsistent QC config across stats "
                    "(drop_near_dups)."
                )
        _accumulate_qc_counts(qc_agg, qc)

    merged["qc"] = qc_agg or {}
    return merged
