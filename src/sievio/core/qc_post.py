# qc_post.py
# SPDX-License-Identifier: MIT
"""Lifecycle hook for post-hoc quality-control scoring."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal, cast

from .concurrency import Executor, ExecutorConfig, infer_executor_kind, resolve_qc_executor_config
from .config import QCConfig, QCMode, SafetyConfig, SievioConfig
from .factories_qc import make_qc_scorer, make_safety_scorer
from .interfaces import RunArtifacts, RunContext, RunLifecycleHook, SafetyScorer
from .log import get_logger
from .qc_controller import (
    QCSummaryTracker,
    QualityDecisionPolicy,
    SafetyDecisionPolicy,
    _derive_csv_path,
    summarize_qc_rows,
)
from .qc_utils import open_jsonl_maybe_gz
from .records import (
    best_effort_record_path,
    check_record_schema,
    filter_qc_meta,
    filter_safety_meta,
)

log = get_logger(__name__)


class PostQCHook(RunLifecycleHook):
    """Run post-hoc quality screening/QC scoring after the pipeline completes."""

    def __init__(self, qc_cfg: QCConfig, scorer, *, executor_hint: Any | None = None) -> None:
        self._qc_cfg = qc_cfg
        self._scorer = scorer
        self._executor_hint = executor_hint

    def on_run_start(self, ctx: RunContext) -> None:
        return None

    def on_record(self, record: Mapping[str, Any]) -> Mapping[str, Any] | None:
        return record

    def on_run_end(self, ctx: RunContext) -> None:
        qc_cfg = self._qc_cfg
        if not qc_cfg.enabled or qc_cfg.mode != QCMode.POST:
            return
        jsonl_path = ctx.cfg.sinks.primary_jsonl_name or ctx.cfg.metadata.primary_jsonl
        if not jsonl_path:
            log.warning("Post-QC requested but no primary JSONL path was found.")
            return
        summary = _run_post_qc(
            jsonl_path,
            ctx.cfg,
            scorer=self._scorer,
            runtime=ctx.runtime,
            executor_hint=self._executor_hint,
        )
        existing = getattr(ctx.stats, "qc", None)
        tracker = existing if isinstance(existing, QCSummaryTracker) else QCSummaryTracker()
        tracker.merge_from_summary_dict(summary, replace_screeners={"quality"})
        ctx.stats.qc = tracker

    def on_artifacts(self, artifacts: RunArtifacts, ctx: RunContext) -> None:
        # Post-hoc QC operates directly on JSONL paths pulled from cfg.
        return None


class PostSafetyHook(RunLifecycleHook):
    """Run post-hoc safety screening after the pipeline completes."""

    def __init__(
        self,
        safety_cfg: SafetyConfig,
        scorer: SafetyScorer,
        *,
        executor_hint: Any | None = None,
    ) -> None:
        self._safety_cfg = safety_cfg
        self._scorer = scorer
        self._executor_hint = executor_hint

    def on_run_start(self, ctx: RunContext) -> None:
        return None

    def on_record(self, record: Mapping[str, Any]) -> Mapping[str, Any] | None:
        return record

    def on_run_end(self, ctx: RunContext) -> None:
        safety_cfg = self._safety_cfg
        if not safety_cfg.enabled or safety_cfg.mode != QCMode.POST:
            return
        jsonl_path = ctx.cfg.sinks.primary_jsonl_name or ctx.cfg.metadata.primary_jsonl
        if not jsonl_path:
            log.warning("Post-safety requested but no primary JSONL path was found.")
            return
        summary = _run_post_safety(
            jsonl_path,
            ctx.cfg,
            scorer=self._scorer,
            runtime=ctx.runtime,
            executor_hint=self._executor_hint,
        )
        existing = getattr(ctx.stats, "qc", None)
        tracker = existing if isinstance(existing, QCSummaryTracker) else QCSummaryTracker()
        tracker.merge_from_summary_dict(summary, replace_screeners={"safety"})
        ctx.stats.qc = tracker

    def on_artifacts(self, artifacts: RunArtifacts, ctx: RunContext) -> None:
        return None


def run_qc_over_jsonl(
    jsonl_path: str,
    qc_cfg: QCConfig,
    *,
    config: SievioConfig,
    scorer: Any,
    runtime: Any | None = None,
    executor_hint: Any | None = None,
    write_csv: bool | None = None,
    csv_suffix: str | None = None,
    write_signals_sidecar: bool | None = None,
    signals_suffix: str | None = None,
    signals_format: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]] | None]:
    """
    Score a JSONL file with the provided scorer and return a summary plus rows.

    The helper streams rows when CSV output is not requested. When CSV is
    requested, rows are collected to preserve ordering for the written file.
    """

    if scorer is None:
        raise RuntimeError(
            "QC scorer unavailable for post-QC; configuration should have disabled QC "
            "when extras were missing."
        )

    tracker = QCSummaryTracker(
        enabled=bool(qc_cfg.enabled),
        mode=qc_cfg.mode,
        min_score=qc_cfg.min_score,
        drop_near_dups=bool(qc_cfg.drop_near_dups),
    )
    policy = QualityDecisionPolicy()
    should_write_csv = qc_cfg.write_csv if write_csv is None else bool(write_csv)
    should_write_signals = (
        qc_cfg.write_signals_sidecar
        if write_signals_sidecar is None
        else bool(write_signals_sidecar)
    )
    signals_format_val = (signals_format or qc_cfg.signals_format or "csv").lower()
    if signals_format_val not in {"csv", "parquet"}:
        raise ValueError("signals_format must be 'csv' or 'parquet'.")
    rows_for_csv: list[dict[str, Any]] | None = (
        [] if (should_write_csv or should_write_signals) else None
    )

    def _consume_rows(rows: Iterable[dict[str, Any]]) -> None:
        for row in rows:
            decision = policy.decide(row, cfg=qc_cfg)
            tracker.observe_quality(row, decision, did_drop=False)
            if rows_for_csv is not None:
                rows_for_csv.append(row)

    jsonl_path_str = str(jsonl_path)
    rows_result: list[dict[str, Any]] | None = None
    summary: dict[str, Any]

    if rows_for_csv is None:
        if qc_cfg.parallel_post:
            ok = _score_jsonl_parallel_streaming(
                _iter_jsonl_shards(jsonl_path_str),
                qc_cfg,
                config,
                runtime,
                _consume_rows,
                executor_hint=executor_hint,
            )
            if not ok:
                _score_jsonl_sequential_streaming(
                    jsonl_path_str,
                    scorer,
                    _consume_rows,
                    fail_on_error=bool(qc_cfg.fail_on_error),
                    tracker=tracker,
                )
        else:
            _score_jsonl_sequential_streaming(
                jsonl_path_str,
                scorer,
                _consume_rows,
                fail_on_error=bool(qc_cfg.fail_on_error),
                tracker=tracker,
            )
        summary = tracker.as_dict()
    else:
        shards = list(_iter_jsonl_shards(jsonl_path_str))
        rows: list[dict[str, Any]] = []
        if shards:
            if qc_cfg.parallel_post:
                parallel_rows = _score_jsonl_parallel_collecting(
                    shards,
                    qc_cfg,
                    config,
                    runtime,
                    executor_hint=executor_hint,
                )
                if parallel_rows is None:
                    rows = _score_jsonl_sequential(
                        shards,
                        scorer,
                        fail_on_error=bool(qc_cfg.fail_on_error),
                        tracker=tracker,
                    )
                else:
                    rows = parallel_rows
            else:
                rows = _score_jsonl_sequential(
                    shards,
                    scorer,
                    fail_on_error=bool(qc_cfg.fail_on_error),
                    tracker=tracker,
                )

        summary = summarize_qc_rows(
            rows,
            mode=qc_cfg.mode,
            min_score=qc_cfg.min_score,
            drop_near_dups=bool(qc_cfg.drop_near_dups),
            apply_gates=False,
            enabled=bool(qc_cfg.enabled),
        )
        tracker_quality = tracker.get_screener("quality", create=False)
        if tracker_quality is not None:
            screeners_payload = summary.setdefault("screeners", {})
            quality_payload = screeners_payload.get("quality") or {}
            if isinstance(quality_payload, Mapping):
                quality_payload = dict(quality_payload)
            quality_payload["errors"] = tracker_quality.errors
            screeners_payload["quality"] = quality_payload
        rows_result = rows
        out_csv = _derive_csv_path(
            jsonl_path,
            csv_suffix if csv_suffix is not None else qc_cfg.csv_suffix,
        )
        if should_write_csv and out_csv:
            emit_qc_csv(rows, jsonl_path_str, out_csv)
        if should_write_signals:
            derived_suffix = (
                signals_suffix if signals_suffix is not None else qc_cfg.signals_suffix
            )
            if not derived_suffix:
                derived_suffix = (
                    "_signals.parquet"
                    if signals_format_val == "parquet"
                    else "_signals.csv"
                )
            out_signals = _derive_csv_path(jsonl_path, derived_suffix)
            if out_signals:
                if signals_format_val == "parquet":
                    emit_qc_signals_parquet(rows, jsonl_path_str, out_signals)
                else:
                    emit_qc_signals_csv(rows, jsonl_path_str, out_signals)

    _log_post_qc_summary(summary)
    return summary, rows_result


def _run_post_qc(
    jsonl_path: str,
    config: SievioConfig,
    *,
    scorer: Any | None,
    runtime: Any | None = None,
    executor_hint: Any | None = None,
) -> dict[str, Any]:
    """Run post-hoc QC over the primary JSONL file."""

    qc_cfg = config.qc
    runtime_scorer = getattr(runtime, "post_qc_scorer", None) if runtime is not None else None
    scorer_obj = scorer or runtime_scorer or make_qc_scorer(qc_cfg)
    summary, _rows = run_qc_over_jsonl(
        jsonl_path,
        qc_cfg,
        config=config,
        scorer=scorer_obj,
        runtime=runtime,
        executor_hint=executor_hint,
        write_csv=qc_cfg.write_csv,
        csv_suffix=qc_cfg.csv_suffix,
        write_signals_sidecar=qc_cfg.write_signals_sidecar,
        signals_suffix=qc_cfg.signals_suffix,
        signals_format=qc_cfg.signals_format,
    )
    return summary


def run_safety_over_jsonl(
    jsonl_path: str,
    safety_cfg: SafetyConfig,
    *,
    config: SievioConfig,
    scorer: SafetyScorer,
    runtime: Any | None = None,
    executor_hint: Any | None = None,
    write_csv: bool | None = None,
    csv_suffix: str | None = None,
    write_signals_sidecar: bool | None = None,
    signals_suffix: str | None = None,
    signals_format: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]] | None]:
    """
    Score a JSONL file with the provided safety scorer and return a summary plus rows.

    Safety POST runs never mutate the source JSONL; gating only affects the summary
    counters and optional reports.
    """

    if scorer is None:
        raise RuntimeError(
            "Safety scorer unavailable for post-safety; "
            "configuration should have disabled safety when extras were missing."
        )

    tracker = QCSummaryTracker(enabled=bool(safety_cfg.enabled), mode=safety_cfg.mode)
    policy = SafetyDecisionPolicy()
    should_write_csv = safety_cfg.write_csv if write_csv is None else bool(write_csv)
    should_write_signals = (
        safety_cfg.write_signals_sidecar
        if write_signals_sidecar is None
        else bool(write_signals_sidecar)
    )
    signals_format_val = (signals_format or safety_cfg.signals_format or "csv").lower()
    if signals_format_val not in {"csv", "parquet"}:
        raise ValueError("signals_format must be 'csv' or 'parquet'.")
    rows_for_output: list[dict[str, Any]] | None = (
        []
        if (should_write_csv or should_write_signals)
        else None
    )

    def _consume_rows(rows: Iterable[dict[str, Any]]) -> None:
        for row in rows:
            decision = policy.decide(row, cfg=safety_cfg)
            tracker.observe_safety(
                row,
                decision,
                did_drop=False,
                screener_id="safety",
                mode=QCMode.POST,
            )
            if rows_for_output is not None:
                rows_for_output.append(row)

    jsonl_path_str = str(jsonl_path)
    rows_result: list[dict[str, Any]] | None = None

    if rows_for_output is None:
        if safety_cfg.parallel_post:
            ok = _score_jsonl_safety_parallel_streaming(
                _iter_jsonl_shards(jsonl_path_str),
                safety_cfg,
                config,
                runtime,
                _consume_rows,
                executor_hint=executor_hint,
            )
            if not ok:
                _score_jsonl_safety_streaming(
                    jsonl_path_str,
                    scorer,
                    _consume_rows,
                    fail_on_error=bool(safety_cfg.fail_on_error),
                    tracker=tracker,
                )
        else:
            _score_jsonl_safety_streaming(
                jsonl_path_str,
                scorer,
                _consume_rows,
                fail_on_error=bool(safety_cfg.fail_on_error),
                tracker=tracker,
            )
    else:
        shards = list(_iter_jsonl_shards(jsonl_path_str))
        rows: list[dict[str, Any]] = []
        if shards:
            if safety_cfg.parallel_post:
                parallel_rows = _score_jsonl_safety_parallel_collecting(
                    shards,
                    safety_cfg,
                    config,
                    runtime,
                    executor_hint=executor_hint,
                )
                if parallel_rows is None:
                    rows = _score_jsonl_safety_collecting(
                        shards,
                        scorer,
                        fail_on_error=bool(safety_cfg.fail_on_error),
                        tracker=tracker,
                    )
                else:
                    rows = parallel_rows
            else:
                rows = _score_jsonl_safety_collecting(
                    shards,
                    scorer,
                    fail_on_error=bool(safety_cfg.fail_on_error),
                    tracker=tracker,
                )
        for row in rows:
            decision = policy.decide(row, cfg=safety_cfg)
            tracker.observe_safety(
                row,
                decision,
                did_drop=False,
                screener_id="safety",
                mode=QCMode.POST,
            )

        rows_result = rows
        out_csv = _derive_csv_path(
            jsonl_path,
            csv_suffix if csv_suffix is not None else safety_cfg.csv_suffix,
        )
        if should_write_csv and out_csv:
            emit_safety_csv(rows, jsonl_path_str, out_csv)
        if should_write_signals:
            derived_suffix = (
                signals_suffix
                if signals_suffix is not None
                else safety_cfg.signals_suffix
            )
            if not derived_suffix:
                derived_suffix = (
                    "_safety_signals.parquet"
                    if signals_format_val == "parquet"
                    else "_safety_signals.csv"
                )
            out_signals = _derive_csv_path(jsonl_path, derived_suffix)
            if out_signals:
                if signals_format_val == "parquet":
                    emit_safety_signals_parquet(rows, jsonl_path_str, out_signals)
                else:
                    emit_safety_signals_csv(rows, jsonl_path_str, out_signals)

    summary = tracker.as_dict()
    _log_post_safety_summary(summary)
    return summary, rows_result


def _run_post_safety(
    jsonl_path: str,
    config: SievioConfig,
    *,
    scorer: SafetyScorer | None,
    runtime: Any | None = None,
    executor_hint: Any | None = None,
) -> dict[str, Any]:
    """Run post-hoc safety screening over the primary JSONL file."""

    safety_cfg = getattr(config.qc, "safety", None)
    if safety_cfg is None:
        return QCSummaryTracker().as_dict()
    runtime_scorer = getattr(runtime, "post_safety_scorer", None) if runtime is not None else None
    scorer_obj = scorer or runtime_scorer or make_safety_scorer(safety_cfg, new_instance=True)
    if scorer_obj is None:
        raise RuntimeError(
            "Safety scorer unavailable for post-safety; "
            "configuration should have disabled safety when extras were missing."
        )
    summary, _rows = run_safety_over_jsonl(
        jsonl_path,
        safety_cfg,
        config=config,
        scorer=scorer_obj,
        runtime=runtime,
        executor_hint=executor_hint,
        write_csv=safety_cfg.write_csv,
        csv_suffix=safety_cfg.csv_suffix,
        write_signals_sidecar=safety_cfg.write_signals_sidecar,
        signals_suffix=safety_cfg.signals_suffix,
        signals_format=safety_cfg.signals_format,
    )
    return summary


def iter_qc_rows_from_jsonl(
    jsonl_path: str,
    *,
    qc_cfg: QCConfig,
    config: SievioConfig,
    scorer: Any,
    runtime: Any | None = None,
    executor_hint: Any | None = None,
    tracker: QCSummaryTracker | None = None,
) -> Iterable[dict[str, Any]]:
    """Iterate QC rows from a JSONL file, updating an optional tracker."""

    def _generator() -> Iterator[dict[str, Any]]:
        buffer: list[dict[str, Any]] = []

        policy = QualityDecisionPolicy()

        def _consume_rows(rows: Iterable[dict[str, Any]]) -> None:
            for row in rows:
                if tracker is not None:
                    decision = policy.decide(row, cfg=qc_cfg)
                    tracker.observe_quality(row, decision, did_drop=False)
                buffer.append(row)

        jsonl_path_str = str(jsonl_path)
        if qc_cfg.parallel_post:
            ok = _score_jsonl_parallel_streaming(
                _iter_jsonl_shards(jsonl_path_str),
                qc_cfg,
                config,
                runtime,
                _consume_rows,
                executor_hint=executor_hint,
            )
            if not ok:
                _score_jsonl_sequential_streaming(
                    jsonl_path_str,
                    scorer,
                    _consume_rows,
                    fail_on_error=bool(qc_cfg.fail_on_error),
                    tracker=tracker,
                )
        else:
            _score_jsonl_sequential_streaming(
                jsonl_path_str,
                scorer,
                _consume_rows,
                fail_on_error=bool(qc_cfg.fail_on_error),
                tracker=tracker,
            )

        yield from buffer

    return _generator()


def collect_qc_rows_from_jsonl(
    jsonl_path: str,
    *,
    qc_cfg: QCConfig,
    config: SievioConfig,
    scorer: Any,
    runtime: Any | None = None,
    executor_hint: Any | None = None,
    tracker: QCSummaryTracker | None = None,
) -> list[dict[str, Any]]:
    """Collect QC rows from a JSONL file into a list."""
    rows: list[dict[str, Any]] = []
    for row in iter_qc_rows_from_jsonl(
        jsonl_path,
        qc_cfg=qc_cfg,
        config=config,
        scorer=scorer,
        runtime=runtime,
        executor_hint=executor_hint,
        tracker=tracker,
    ):
        rows.append(row)
    return rows


def emit_qc_csv(rows: Sequence[dict[str, Any]], jsonl_path: str, out_csv: str) -> None:
    """Write QC rows to CSV using available helpers."""
    _write_csv: Callable[..., Any] | None = None
    _score_jsonl_to_csv: Callable[..., Any] | None = None
    try:  # pragma: no cover - optional QC extras
        from .extras.qc import score_jsonl_to_csv as _score_jsonl_to_csv
        from .extras.qc import write_csv as _write_csv
    except Exception:  # pragma: no cover - optional QC extras
        pass

    if _write_csv is not None:
        _write_csv(rows, out_csv)
    elif _score_jsonl_to_csv is not None:
        _score_jsonl_to_csv(jsonl_path, out_csv)
    else:
        raise RuntimeError("QC CSV helpers unavailable; reinstall optional dependencies.")


def _collect_signal_rows(rows: Sequence[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    """Return fieldnames and per-row signal dicts with doc_id."""
    signal_keys: set[str] = set()
    raw_signals: list[tuple[Any, dict[str, Any]]] = []
    for row in rows:
        _, signals = filter_qc_meta(row)
        signals_dict = dict(signals)
        raw_signals.append((row.get("doc_id"), signals_dict))
        signal_keys.update(signals_dict.keys())
    fieldnames = ["doc_id", *sorted(signal_keys)]
    out_rows: list[dict[str, Any]] = []
    for doc_id, signal_row in raw_signals:
        payload: dict[str, Any] = {key: None for key in fieldnames}
        payload["doc_id"] = doc_id
        for key, value in signal_row.items():
            payload[key] = value
        out_rows.append(payload)
    return fieldnames, out_rows


def emit_qc_signals_csv(
    rows: Sequence[dict[str, Any]],
    jsonl_path: str,
    out_csv: str,
) -> None:
    """Write per-record QC signals to a CSV sidecar keyed by doc_id."""
    import csv

    if not rows:
        return

    fieldnames, signal_rows = _collect_signal_rows(rows)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in signal_rows:
            writer.writerow(row)


def emit_qc_signals_parquet(
    rows: Sequence[dict[str, Any]],
    jsonl_path: str,
    out_parquet: str,
) -> None:
    """Write per-record QC signals to a Parquet sidecar."""
    if not rows:
        return
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError(
            "PyArrow is required for Parquet signal sidecars; "
            "install sievio[parquet]."
        ) from exc

    fieldnames, signal_rows = _collect_signal_rows(rows)
    table = pa.Table.from_pylist(signal_rows).select(fieldnames)
    pq.write_table(table, out_parquet)


def _score_jsonl_sequential(
    shards: list[_JsonlShard],
    scorer,
    *,
    fail_on_error: bool = False,
    tracker: QCSummaryTracker | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shard in shards:
        rows.extend(
            _score_lines(
                shard.lines,
                scorer,
                source_path=shard.path,
                shard_label=f"shard-{shard.index}:{shard.path}",
                fail_on_error=fail_on_error,
                tracker=tracker,
            )
        )
    return rows


def _score_jsonl_sequential_streaming(
    jsonl_path: str,
    scorer,
    consume_rows: Callable[[Iterable[dict[str, Any]]], None],
    *,
    fail_on_error: bool = False,
    tracker: QCSummaryTracker | None = None,
) -> None:
    for shard in _iter_jsonl_shards(jsonl_path):
        rows = _score_lines(
            shard.lines,
            scorer,
            source_path=shard.path,
            shard_label=f"shard-{shard.index}:{shard.path}",
            fail_on_error=fail_on_error,
            tracker=tracker,
        )
        if rows:
            consume_rows(rows)


def _run_parallel_qc(
    shards: Iterable[_JsonlShard],
    qc_cfg: QCConfig,
    config: SievioConfig,
    runtime: Any | None,
    handle_rows: Callable[[int, list[dict[str, Any]]], None],
    *,
    executor_hint: Any | None = None,
) -> bool:
    try:
        runtime_scorer = (
            getattr(runtime, "post_qc_scorer", None)
            if runtime is not None
            else None
        )
        scorer_proto = runtime_scorer or make_qc_scorer(qc_cfg, new_instance=True)
    except Exception:
        scorer_proto = None
    if scorer_proto is None:
        log.warning(
            "QC parallel_post requested but scorer could not be instantiated; "
            "falling back to sequential mode."
        )
        return False

    def _scorer_factory():
        cloned = getattr(scorer_proto, "clone_for_parallel", None)
        if callable(cloned):
            return cloned()
        additional = make_qc_scorer(qc_cfg, new_instance=True)
        if additional is None:
            raise RuntimeError("Unable to create scorer for parallel QC processing.")
        return additional

    exec_cfg = executor_hint or resolve_qc_executor_config(config, runtime=runtime)
    init = None
    initargs: tuple[Any, ...] = ()
    if exec_cfg.kind == "process":
        payload_cfg = replace(qc_cfg, scorer=None)
        qc_payload = asdict(payload_cfg)
        init = _qc_worker_initializer
        initargs = (qc_payload,)

    executor = Executor(
        exec_cfg,
        initializer=init,
        initargs=initargs,
    )

    def _worker(shard: _JsonlShard) -> tuple[int, list[dict[str, Any]]]:
        if exec_cfg.kind == "process":
            return _qc_parallel_worker(shard)
        scorer = _scorer_factory()
        label = f"shard-{shard.index}:{shard.path}"
        return shard.index, _score_lines(
            shard.lines,
            scorer,
            source_path=shard.path,
            shard_label=label,
            fail_on_error=bool(qc_cfg.fail_on_error),
            tracker=None,
        )

    def _on_result(result: tuple[int, list[dict[str, Any]]]) -> None:
        shard_idx, shard_rows = result
        rows_list = list(shard_rows)
        if rows_list:
            handle_rows(shard_idx, rows_list)

    def _on_error(exc: BaseException) -> None:
        log.error("Parallel QC worker failed: %s", exc)

    try:
        executor.map_unordered(
            shards,
            _worker,
            _on_result,
            fail_fast=True,
            on_error=_on_error,
        )
    except Exception as exc:
        log.warning("Parallel QC scoring failed (%s); falling back to sequential mode.", exc)
        return False

    return True


def _score_jsonl_parallel_collecting(
    shards: list[_JsonlShard],
    qc_cfg: QCConfig,
    config: SievioConfig,
    runtime: Any | None,
    *,
    executor_hint: Any | None = None,
) -> list[dict[str, Any]] | None:
    shard_results: dict[int, list[dict[str, Any]]] = {}

    def _handle_rows(idx: int, rows: list[dict[str, Any]]) -> None:
        shard_results[idx] = rows

    ok = _run_parallel_qc(
        shards,
        qc_cfg,
        config,
        runtime,
        _handle_rows,
        executor_hint=executor_hint,
    )
    if not ok:
        return None

    rows: list[dict[str, Any]] = []
    for idx in sorted(shard_results):
        rows.extend(shard_results[idx])
    return rows


def _score_jsonl_parallel_streaming(
    shards: Iterable[_JsonlShard],
    qc_cfg: QCConfig,
    config: SievioConfig,
    runtime: Any | None,
    consume_rows: Callable[[Iterable[dict[str, Any]]], None],
    *,
    executor_hint: Any | None = None,
) -> bool:
    def _handle_rows(_: int, rows: list[dict[str, Any]]) -> None:
        consume_rows(rows)

    return _run_parallel_qc(
        shards,
        qc_cfg,
        config,
        runtime,
        _handle_rows,
        executor_hint=executor_hint,
    )


@dataclass(slots=True)
class _JsonlShard:
    index: int
    lines: list[str]
    path: str


_QC_WORKER_SCORER: Any | None = None


def _qc_worker_initializer(qc_cfg_payload: dict[str, Any]) -> None:
    global _QC_WORKER_SCORER
    qc_cfg = QCConfig(**qc_cfg_payload)
    scorer = make_qc_scorer(qc_cfg, new_instance=True)
    if scorer is None:
        raise RuntimeError(
            "QC worker failed to initialize scorer "
            "(missing optional dependencies?)."
        )
    _QC_WORKER_SCORER = scorer


def _qc_parallel_worker(shard: _JsonlShard) -> tuple[int, list[dict[str, Any]]]:
    if _QC_WORKER_SCORER is None:
        raise RuntimeError("QC worker scorer not initialized")
    label = f"shard-{shard.index}:{shard.path}"
    return shard.index, _score_lines(
        shard.lines,
        _QC_WORKER_SCORER,
        source_path=shard.path,
        shard_label=label,
        fail_on_error=False,
        tracker=None,
    )


def _iter_jsonl_shards(path: str, shard_size: int = 500) -> Iterator[_JsonlShard]:
    chunk: list[str] = []
    idx = 0
    with open_jsonl_maybe_gz(path) as fp:
        for line in fp:
            if not line.strip():
                continue
            chunk.append(line)
            if len(chunk) >= shard_size:
                yield _JsonlShard(idx, chunk, path)
                idx += 1
                chunk = []
    if chunk:
        yield _JsonlShard(idx, chunk, path)


def _score_lines(
    lines: list[str],
    scorer,
    *,
    source_path: str,
    shard_label: str | None = None,
    fail_on_error: bool = False,
    tracker: QCSummaryTracker | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    label = shard_label or source_path
    checked_schema = False
    for idx, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except Exception as exc:
            if tracker is not None:
                tracker.record_error()
            if fail_on_error:
                msg = (
                    "Failed to parse JSONL line "
                    f"{idx} in {label}: {exc} "
                    f"(source_path={source_path})"
                )
                raise RuntimeError(msg) from exc
            log.warning(
                "Failed to parse JSONL line %d in %s: %s "
                "(this may indicate compressed JSONL or a corrupted line)",
                idx,
                label,
                exc,
            )
            continue
        if not isinstance(record, dict):
            continue
        meta = record.get("meta") if isinstance(record, dict) else None
        if isinstance(meta, dict):
            kind = meta.get("kind")
            if kind in {"run_header", "run_summary", "qc_summary"}:
                continue
        if not checked_schema:
            check_record_schema(record, log)
            checked_schema = True
        try:
            rows.append(scorer.score_record(record))
        except StopIteration:
            break
        except Exception as exc:
            if tracker is not None:
                tracker.record_error()
            if fail_on_error:
                raise RuntimeError(
                    f"QC scorer failed for line {idx} in {label}: {exc} (source_path={source_path})"
                ) from exc
            log.warning("QC scorer failed for line %d in %s: %s", idx, label, exc)
            continue
    return rows


def _score_safety_lines(
    lines: list[str],
    scorer: SafetyScorer,
    *,
    source_path: str,
    shard_label: str | None = None,
    fail_on_error: bool = False,
    tracker: QCSummaryTracker | None = None,
    screener_mode: str = QCMode.POST,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    label = shard_label or source_path
    checked_schema = False
    for idx, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except Exception as exc:
            if tracker is not None:
                tracker.record_safety_error(mode=screener_mode)
            if fail_on_error:
                msg = (
                    "Failed to parse JSONL line "
                    f"{idx} in {label}: {exc} "
                    f"(source_path={source_path})"
                )
                raise RuntimeError(msg) from exc
            log.warning(
                "Failed to parse JSONL line %d in %s: %s "
                "(this may indicate compressed JSONL or a corrupted line)",
                idx,
                label,
                exc,
            )
            continue
        if not isinstance(record, dict):
            continue
        meta = record.get("meta") if isinstance(record, dict) else None
        if isinstance(meta, Mapping):
            kind = meta.get("kind")
            if kind in {"run_header", "run_summary", "qc_summary"}:
                continue
        if not checked_schema:
            check_record_schema(record, log)
            checked_schema = True
        try:
            raw_result = scorer.score_record(record)
        except StopIteration:
            break
        except Exception as exc:
            if tracker is not None:
                tracker.record_safety_error(mode=screener_mode)
            if fail_on_error:
                msg = (
                    "Safety scorer failed for line "
                    f"{idx} in {label}: {exc} "
                    f"(source_path={source_path})"
                )
                raise RuntimeError(msg) from exc
            log.warning("Safety scorer failed for line %d in %s: %s", idx, label, exc)
            continue
        if not isinstance(raw_result, Mapping):
            continue
        result: dict[str, Any] = dict(raw_result)
        path_hint = best_effort_record_path(record)
        if path_hint and "record_path" not in result:
            result["record_path"] = path_hint
        rows.append(result)
    return rows


def _log_post_qc_summary(summary: dict[str, Any]) -> None:
    screeners = summary.get("screeners") if isinstance(summary, Mapping) else None
    quality = screeners.get("quality") if isinstance(screeners, Mapping) else {}
    quality_payload = quality if isinstance(quality, Mapping) else {}
    candidates = quality_payload.get("candidates") if isinstance(quality_payload, Mapping) else {}
    scored_val = int(quality_payload.get("scored", 0) or 0)
    cand_low = int((candidates or {}).get("low_score", 0) or 0)
    cand_near_dup = int((candidates or {}).get("near_dup", 0) or 0)
    log.info(
        "Post-QC summary (mode=%s, scored=%d, candidates_low_score=%d, candidates_near_dup=%d)",
        summary.get("mode"),
        scored_val,
        cand_low,
        cand_near_dup,
    )
    top = summary.get("top_dup_families") or []
    if top:
        lines = [
            (
                f"    - {entry['dup_family_id']}: count={entry['count']} "
                f"examples={entry.get('examples', [])}"
            )
            for entry in top
            if isinstance(entry, Mapping)
        ]
        log.info("Largest duplicate families (post-QC):\n%s", "\n".join(lines))
    else:
        log.info("Largest duplicate families (post-QC): none")


def _score_jsonl_safety_collecting(
    shards: list[_JsonlShard],
    scorer: SafetyScorer,
    *,
    fail_on_error: bool = False,
    tracker: QCSummaryTracker | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shard in shards:
        rows.extend(
            _score_safety_lines(
                shard.lines,
                scorer,
                source_path=shard.path,
                shard_label=f"shard-{shard.index}:{shard.path}",
                fail_on_error=fail_on_error,
                tracker=tracker,
            )
        )
    return rows


def _score_jsonl_safety_streaming(
    jsonl_path: str,
    scorer: SafetyScorer,
    consume_rows: Callable[[Iterable[dict[str, Any]]], None],
    *,
    fail_on_error: bool = False,
    tracker: QCSummaryTracker | None = None,
) -> None:
    for shard in _iter_jsonl_shards(jsonl_path):
        rows = _score_safety_lines(
            shard.lines,
            scorer,
            source_path=shard.path,
            shard_label=f"shard-{shard.index}:{shard.path}",
            fail_on_error=fail_on_error,
            tracker=tracker,
        )
        if rows:
            consume_rows(rows)


def _resolve_safety_executor_config(
    cfg: SievioConfig,
    safety_cfg: SafetyConfig,
    runtime: Any | None = None,
) -> ExecutorConfig:
    pc = cfg.pipeline
    pipeline_max_workers = pc.max_workers or (os.cpu_count() or 1)
    pipeline_max_workers = max(1, pipeline_max_workers)
    max_workers = (
        pipeline_max_workers
        if safety_cfg.post_max_workers is None
        else safety_cfg.post_max_workers
    )
    max_workers = max(1, max_workers)

    if safety_cfg.post_submit_window is not None:
        window = safety_cfg.post_submit_window
    elif pc.submit_window is not None:
        window = pc.submit_window
    else:
        window = max_workers * 4

    raw_kind = (
        safety_cfg.post_executor_kind or pc.executor_kind or "thread"
    ).strip().lower()
    kind: str
    if raw_kind == "auto":
        kind = infer_executor_kind(cfg, runtime=runtime)
    else:
        kind = raw_kind
    if kind not in {"thread", "process"}:
        kind = "thread"
    kind = cast(Literal["thread", "process"], kind)
    return ExecutorConfig(
        max_workers=max_workers,
        window=max(window, max_workers),
        kind=kind,
    )


def _run_parallel_safety(
    shards: Iterable[_JsonlShard],
    safety_cfg: SafetyConfig,
    config: SievioConfig,
    runtime: Any | None,
    handle_rows: Callable[[int, list[dict[str, Any]]], None],
    *,
    executor_hint: Any | None = None,
) -> bool:
    try:
        runtime_scorer = (
            getattr(runtime, "post_safety_scorer", None)
            if runtime is not None
            else None
        )
        scorer_proto = runtime_scorer or make_safety_scorer(safety_cfg, new_instance=True)
    except Exception:
        scorer_proto = None
    if scorer_proto is None:
        log.warning(
            "Safety parallel_post requested but scorer could not be instantiated; "
            "falling back to sequential mode."
        )
        return False

    def _scorer_factory():
        cloned = getattr(scorer_proto, "clone_for_parallel", None)
        if callable(cloned):
            return cloned()
        additional = make_safety_scorer(safety_cfg, new_instance=True)
        if additional is None:
            raise RuntimeError("Unable to create safety scorer for parallel processing.")
        return additional

    exec_cfg = executor_hint or _resolve_safety_executor_config(config, safety_cfg, runtime=runtime)
    init = None
    initargs: tuple[Any, ...] = ()
    if exec_cfg.kind == "process":
        payload_cfg = replace(safety_cfg, scorer=None)
        safety_payload = asdict(payload_cfg)
        init = _safety_worker_initializer
        initargs = (safety_payload,)

    executor = Executor(
        exec_cfg,
        initializer=init,
        initargs=initargs,
    )

    def _worker(shard: _JsonlShard) -> tuple[int, list[dict[str, Any]]]:
        if exec_cfg.kind == "process":
            return _safety_parallel_worker(shard)
        scorer_obj = _scorer_factory()
        label = f"shard-{shard.index}:{shard.path}"
        return shard.index, _score_safety_lines(
            shard.lines,
            scorer_obj,
            source_path=shard.path,
            shard_label=label,
            fail_on_error=bool(safety_cfg.fail_on_error),
            tracker=None,
        )

    def _on_result(result: tuple[int, list[dict[str, Any]]]) -> None:
        shard_idx, shard_rows = result
        rows_list = list(shard_rows)
        if rows_list:
            handle_rows(shard_idx, rows_list)

    def _on_error(exc: BaseException) -> None:
        log.error("Parallel safety worker failed: %s", exc)

    try:
        executor.map_unordered(
            shards,
            _worker,
            _on_result,
            fail_fast=True,
            on_error=_on_error,
        )
    except Exception as exc:
        log.warning("Parallel safety scoring failed (%s); falling back to sequential mode.", exc)
        return False

    return True


def _score_jsonl_safety_parallel_collecting(
    shards: list[_JsonlShard],
    safety_cfg: SafetyConfig,
    config: SievioConfig,
    runtime: Any | None,
    *,
    executor_hint: Any | None = None,
) -> list[dict[str, Any]] | None:
    shard_results: dict[int, list[dict[str, Any]]] = {}

    def _handle_rows(idx: int, rows: list[dict[str, Any]]) -> None:
        shard_results[idx] = rows

    ok = _run_parallel_safety(
        shards,
        safety_cfg,
        config,
        runtime,
        _handle_rows,
        executor_hint=executor_hint,
    )
    if not ok:
        return None

    rows: list[dict[str, Any]] = []
    for idx in sorted(shard_results):
        rows.extend(shard_results[idx])
    return rows


def _score_jsonl_safety_parallel_streaming(
    shards: Iterable[_JsonlShard],
    safety_cfg: SafetyConfig,
    config: SievioConfig,
    runtime: Any | None,
    consume_rows: Callable[[Iterable[dict[str, Any]]], None],
    *,
    executor_hint: Any | None = None,
) -> bool:
    def _handle_rows(_: int, rows: list[dict[str, Any]]) -> None:
        consume_rows(rows)

    return _run_parallel_safety(
        shards,
        safety_cfg,
        config,
        runtime,
        _handle_rows,
        executor_hint=executor_hint,
    )


_SAFETY_WORKER_SCORER: SafetyScorer | None = None


def _safety_worker_initializer(safety_cfg_payload: dict[str, Any]) -> None:
    global _SAFETY_WORKER_SCORER
    cfg = SafetyConfig(**safety_cfg_payload)
    scorer = make_safety_scorer(cfg, new_instance=True)
    if scorer is None:
        raise RuntimeError(
            "Safety worker failed to initialize scorer "
            "(missing optional dependencies?)."
        )
    _SAFETY_WORKER_SCORER = scorer


def _safety_parallel_worker(shard: _JsonlShard) -> tuple[int, list[dict[str, Any]]]:
    if _SAFETY_WORKER_SCORER is None:
        raise RuntimeError("Safety worker scorer not initialized")
    label = f"shard-{shard.index}:{shard.path}"
    return shard.index, _score_safety_lines(
        shard.lines,
        _SAFETY_WORKER_SCORER,
        source_path=shard.path,
        shard_label=label,
        fail_on_error=False,
        tracker=None,
    )


def emit_safety_csv(rows: Sequence[dict[str, Any]], jsonl_path: str, out_csv: str) -> None:
    """Write safety rows to CSV."""
    import csv

    fieldnames = [
        "record_path",
        "safety_decision",
        "safety_reason",
        "safety_drop_reason",
        "pii_detected",
        "toxicity",
        "flags",
        "signals",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            canonical, signals = filter_safety_meta(row)
            flags_payload = row.get("safety_flags")
            if not isinstance(flags_payload, Mapping):
                saved_flags = signals.get("safety_flags")
                if isinstance(saved_flags, Mapping):
                    flags_payload = saved_flags
                else:
                    flags_payload = {}
            signals_payload = {
                k: v for k, v in signals.items() if k != "safety_flags"
            }
            writer.writerow(
                {
                    "record_path": row.get("record_path") or "",
                    "safety_decision": canonical.get("safety_decision"),
                    "safety_reason": canonical.get("safety_reason"),
                    "safety_drop_reason": canonical.get("safety_drop_reason"),
                    "pii_detected": canonical.get("pii_detected"),
                    "toxicity": canonical.get("toxicity"),
                    "flags": json.dumps(flags_payload or {}, sort_keys=True),
                    "signals": json.dumps(signals_payload, sort_keys=True),
                }
            )


def _collect_safety_signal_rows(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    signal_keys: set[str] = set()
    raw_signals: list[tuple[Any, dict[str, Any]]] = []
    for row in rows:
        _, signals = filter_safety_meta(row)
        signals_payload = {k: v for k, v in signals.items() if k != "safety_flags"}
        raw_signals.append((row.get("doc_id"), signals_payload))
        signal_keys.update(signals_payload.keys())
    fieldnames = ["doc_id", *sorted(signal_keys)]
    out_rows: list[dict[str, Any]] = []
    for doc_id, signals in raw_signals:
        payload = {key: None for key in fieldnames}
        payload["doc_id"] = doc_id
        for key, value in signals.items():
            payload[key] = value
        out_rows.append(payload)
    return fieldnames, out_rows


def emit_safety_signals_csv(
    rows: Sequence[dict[str, Any]],
    jsonl_path: str,
    out_csv: str,
) -> None:
    """Write per-record safety signals to a CSV sidecar keyed by doc_id."""
    import csv

    if not rows:
        return

    fieldnames, signal_rows = _collect_safety_signal_rows(rows)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in signal_rows:
            writer.writerow(row)


def emit_safety_signals_parquet(
    rows: Sequence[dict[str, Any]],
    jsonl_path: str,
    out_parquet: str,
) -> None:
    """Write per-record safety signals to a Parquet sidecar."""
    if not rows:
        return
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError(
            "PyArrow is required for Parquet safety sidecars; "
            "install sievio[parquet]."
        ) from exc

    fieldnames, signal_rows = _collect_safety_signal_rows(rows)
    table = pa.Table.from_pylist(signal_rows).select(fieldnames)
    pq.write_table(table, out_parquet)


def _log_post_safety_summary(summary: dict[str, Any]) -> None:
    screeners = summary.get("screeners") if isinstance(summary, Mapping) else None
    safety = screeners.get("safety") if isinstance(screeners, Mapping) else {}
    safety_payload = safety if isinstance(safety, Mapping) else {}
    log.info(
        "Post-safety summary (mode=%s, scored=%s, would_drop_records=%s, errors=%s)",
        safety_payload.get("mode") or summary.get("mode"),
        int(safety_payload.get("scored", 0) or 0),
        int(safety_payload.get("would_drop_records", 0) or 0),
        int(safety_payload.get("errors", 0) or 0),
    )
    flags = safety_payload.get("flags") or {}
    if flags:
        lines = [f"    - {name}: {count}" for name, count in flags.items()]
        log.info("Safety flags:\n%s", "\n".join(lines))
    else:
        log.info("Safety flags: none")


__all__ = [
    "PostQCHook",
    "PostSafetyHook",
    "run_qc_over_jsonl",
    "run_safety_over_jsonl",
    "iter_qc_rows_from_jsonl",
    "collect_qc_rows_from_jsonl",
    "emit_qc_csv",
    "emit_safety_csv",
]
