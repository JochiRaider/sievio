# qc_post.py
# SPDX-License-Identifier: MIT
"""Lifecycle hook for post-hoc quality-control scoring."""

from __future__ import annotations

from dataclasses import dataclass, replace, asdict
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple
import json

from .config import RepocapsuleConfig, QCConfig, QCMode
from .concurrency import Executor, resolve_qc_executor_config
from .factories import make_qc_scorer
from .interfaces import RunLifecycleHook, RunContext, RunArtifacts
from .log import get_logger
from .qc_controller import QCSummaryTracker, summarize_qc_rows, _derive_csv_path
from .records import filter_qc_meta
from .qc_utils import open_jsonl_maybe_gz

log = get_logger(__name__)


class PostQCHook(RunLifecycleHook):
    """Run post-hoc quality screening/QC scoring after the pipeline completes."""

    def __init__(self, qc_cfg: QCConfig, scorer, *, executor_hint: Any | None = None) -> None:
        self._qc_cfg = qc_cfg
        self._scorer = scorer
        self._executor_hint = executor_hint

    def on_run_start(self, ctx: RunContext) -> None:
        return None

    def on_record(self, record: Mapping[str, Any]) -> Mapping[str, Any] | None:  # type: ignore[override]
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
        ctx.stats.qc = QCSummaryTracker.from_summary_dict(summary)

    def on_artifacts(self, artifacts: RunArtifacts, ctx: RunContext) -> None:
        # Post-hoc QC operates directly on JSONL paths pulled from cfg.
        return None


def run_qc_over_jsonl(
    jsonl_path: str,
    qc_cfg: QCConfig,
    *,
    config: RepocapsuleConfig,
    scorer: Any,
    runtime: Any | None = None,
    executor_hint: Any | None = None,
    write_csv: bool | None = None,
    csv_suffix: str | None = None,
    write_signals_sidecar: bool | None = None,
    signals_suffix: str | None = None,
    signals_format: str | None = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]] | None]:
    """
    Score a JSONL file with the provided scorer and return a summary plus rows.

    The helper streams rows when CSV output is not requested. When CSV is
    requested, rows are collected to preserve ordering for the written file.
    """

    if scorer is None:
        raise RuntimeError("QC scorer unavailable for post-QC; configuration should have disabled QC when extras were missing.")

    tracker = QCSummaryTracker(
        enabled=bool(qc_cfg.enabled),
        mode=qc_cfg.mode,
        min_score=qc_cfg.min_score,
        drop_near_dups=bool(qc_cfg.drop_near_dups),
    )
    should_write_csv = qc_cfg.write_csv if write_csv is None else bool(write_csv)
    should_write_signals = qc_cfg.write_signals_sidecar if write_signals_sidecar is None else bool(write_signals_sidecar)
    signals_format_val = (signals_format or qc_cfg.signals_format or "csv").lower()
    if signals_format_val not in {"csv", "parquet"}:
        raise ValueError("signals_format must be 'csv' or 'parquet'.")
    rows_for_csv: Optional[List[Dict[str, Any]]] = [] if (should_write_csv or should_write_signals) else None

    def _consume_rows(rows: Iterable[Dict[str, Any]]) -> None:
        for row in rows:
            tracker.observe(row, apply_gates=False)
            if rows_for_csv is not None:
                rows_for_csv.append(row)

    jsonl_path_str = str(jsonl_path)
    rows_result: List[Dict[str, Any]] | None = None
    summary: Dict[str, Any]

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
        rows: List[Dict[str, Any]] = []
        if shards:
            if qc_cfg.parallel_post:
                parallel_rows = _score_jsonl_parallel_collecting(shards, qc_cfg, config, runtime, executor_hint=executor_hint)
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
        summary["errors"] = tracker.errors
        rows_result = rows
        out_csv = _derive_csv_path(jsonl_path, csv_suffix if csv_suffix is not None else qc_cfg.csv_suffix)
        if should_write_csv and out_csv:
            emit_qc_csv(rows, jsonl_path_str, out_csv)
        if should_write_signals:
            derived_suffix = signals_suffix if signals_suffix is not None else qc_cfg.signals_suffix
            if not derived_suffix:
                derived_suffix = "_signals.parquet" if signals_format_val == "parquet" else "_signals.csv"
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
    config: RepocapsuleConfig,
    *,
    scorer: Any | None,
    runtime: Any | None = None,
    executor_hint: Any | None = None,
) -> Dict[str, Any]:
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


def iter_qc_rows_from_jsonl(
    jsonl_path: str,
    *,
    qc_cfg: QCConfig,
    config: RepocapsuleConfig,
    scorer: Any,
    runtime: Any | None = None,
    executor_hint: Any | None = None,
    tracker: QCSummaryTracker | None = None,
) -> Iterable[Dict[str, Any]]:
    """Iterate QC rows from a JSONL file, updating an optional tracker."""

    def _generator() -> Iterator[Dict[str, Any]]:
        buffer: List[Dict[str, Any]] = []

        def _consume_rows(rows: Iterable[Dict[str, Any]]) -> None:
            for row in rows:
                if tracker is not None:
                    tracker.observe(row, apply_gates=False)
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
    config: RepocapsuleConfig,
    scorer: Any,
    runtime: Any | None = None,
    executor_hint: Any | None = None,
    tracker: QCSummaryTracker | None = None,
) -> List[Dict[str, Any]]:
    """Collect QC rows from a JSONL file into a list."""
    rows: List[Dict[str, Any]] = []
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


def emit_qc_csv(rows: Sequence[Dict[str, Any]], jsonl_path: str, out_csv: str) -> None:
    """Write QC rows to CSV using available helpers."""
    try:  # pragma: no cover - optional QC extras
        from .extras.qc import write_csv as _write_csv, score_jsonl_to_csv as _score_jsonl_to_csv
    except Exception:  # pragma: no cover - optional QC extras
        _write_csv = None
        _score_jsonl_to_csv = None

    if _write_csv is not None:
        _write_csv(rows, out_csv)
    elif _score_jsonl_to_csv is not None:
        _score_jsonl_to_csv(jsonl_path, out_csv)
    else:
        raise RuntimeError("QC CSV helpers unavailable; reinstall optional dependencies.")


def _collect_signal_rows(rows: Sequence[Dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    """Return fieldnames and per-row signal dicts with doc_id."""
    signal_keys: set[str] = set()
    raw_signals: list[tuple[Any, Dict[str, Any]]] = []
    for row in rows:
        _, signals = filter_qc_meta(row)
        raw_signals.append((row.get("doc_id"), signals))
        signal_keys.update(signals.keys())
    fieldnames = ["doc_id", *sorted(signal_keys)]
    out_rows: list[dict[str, Any]] = []
    for doc_id, signals in raw_signals:
        payload = {key: None for key in fieldnames}
        payload["doc_id"] = doc_id
        for key, value in signals.items():
            payload[key] = value
        out_rows.append(payload)
    return fieldnames, out_rows


def emit_qc_signals_csv(
    rows: Sequence[Dict[str, Any]],
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
    rows: Sequence[Dict[str, Any]],
    jsonl_path: str,
    out_parquet: str,
) -> None:
    """Write per-record QC signals to a Parquet sidecar."""
    if not rows:
        return
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyArrow is required for Parquet signal sidecars; install repocapsule[parquet].") from exc

    fieldnames, signal_rows = _collect_signal_rows(rows)
    table = pa.Table.from_pylist(signal_rows).select(fieldnames)
    pq.write_table(table, out_parquet)


def _score_jsonl_sequential(
    shards: List["_JsonlShard"],
    scorer,
    *,
    fail_on_error: bool = False,
    tracker: Optional[QCSummaryTracker] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
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
    consume_rows: Callable[[Iterable[Dict[str, Any]]], None],
    *,
    fail_on_error: bool = False,
    tracker: Optional[QCSummaryTracker] = None,
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
    shards: Iterable["_JsonlShard"],
    qc_cfg: QCConfig,
    config: RepocapsuleConfig,
    runtime: Any | None,
    handle_rows: Callable[[int, List[Dict[str, Any]]], None],
    *,
    executor_hint: Any | None = None,
) -> bool:
    try:
        runtime_scorer = getattr(runtime, "post_qc_scorer", None) if runtime is not None else None
        scorer_proto = runtime_scorer or make_qc_scorer(qc_cfg, new_instance=True)
    except Exception:
        scorer_proto = None
    if scorer_proto is None:
        log.warning("QC parallel_post requested but scorer could not be instantiated; falling back to sequential mode.")
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

    def _worker(shard: _JsonlShard) -> Tuple[int, List[Dict[str, Any]]]:
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

    def _on_result(result: Tuple[int, List[Dict[str, Any]]]) -> None:
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
    shards: List["_JsonlShard"],
    qc_cfg: QCConfig,
    config: RepocapsuleConfig,
    runtime: Any | None,
    *,
    executor_hint: Any | None = None,
) -> Optional[List[Dict[str, Any]]]:
    shard_results: Dict[int, List[Dict[str, Any]]] = {}

    def _handle_rows(idx: int, rows: List[Dict[str, Any]]) -> None:
        shard_results[idx] = rows

    ok = _run_parallel_qc(shards, qc_cfg, config, runtime, _handle_rows, executor_hint=executor_hint)
    if not ok:
        return None

    rows: List[Dict[str, Any]] = []
    for idx in sorted(shard_results):
        rows.extend(shard_results[idx])
    return rows


def _score_jsonl_parallel_streaming(
    shards: Iterable["_JsonlShard"],
    qc_cfg: QCConfig,
    config: RepocapsuleConfig,
    runtime: Any | None,
    consume_rows: Callable[[Iterable[Dict[str, Any]]], None],
    *,
    executor_hint: Any | None = None,
) -> bool:
    def _handle_rows(_: int, rows: List[Dict[str, Any]]) -> None:
        consume_rows(rows)

    return _run_parallel_qc(shards, qc_cfg, config, runtime, _handle_rows, executor_hint=executor_hint)


@dataclass(slots=True)
class _JsonlShard:
    index: int
    lines: List[str]
    path: str


_QC_WORKER_SCORER: Optional[Any] = None


def _qc_worker_initializer(qc_cfg_payload: Dict[str, Any]) -> None:
    global _QC_WORKER_SCORER
    qc_cfg = QCConfig(**qc_cfg_payload)
    scorer = make_qc_scorer(qc_cfg, new_instance=True)
    if scorer is None:
        raise RuntimeError("QC worker failed to initialize scorer (missing optional dependencies?).")
    _QC_WORKER_SCORER = scorer


def _qc_parallel_worker(shard: _JsonlShard) -> Tuple[int, List[Dict[str, Any]]]:
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
    chunk: List[str] = []
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
    lines: List[str],
    scorer,
    *,
    source_path: str,
    shard_label: Optional[str] = None,
    fail_on_error: bool = False,
    tracker: Optional[QCSummaryTracker] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    label = shard_label or source_path
    for idx, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except Exception as exc:
            if tracker is not None:
                tracker.record_error()
            if fail_on_error:
                raise RuntimeError(
                    f"Failed to parse JSONL line {idx} in {label}: {exc} (source_path={source_path})"
                ) from exc
            else:
                log.warning(
                    "Failed to parse JSONL line %d in %s: %s (this may indicate compressed JSONL or a corrupted line)",
                    idx,
                    label,
                    exc,
                )
                continue
        if not isinstance(record, dict):
            continue
        meta = record.get("meta") if isinstance(record, dict) else None
        if isinstance(meta, dict) and meta.get("kind") in {"run_header", "run_summary", "qc_summary"}:
            continue
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


def _log_post_qc_summary(summary: Dict[str, Any]) -> None:
    log.info(
        "Post-QC summary (mode=%s, scored=%d, candidates_low_score=%d, candidates_near_dup=%d)",
        summary.get("mode"),
        summary.get("scored"),
        summary.get("candidates_low_score"),
        summary.get("candidates_near_dup"),
    )
    top = summary.get("top_dup_families") or []
    if top:
        lines = [
            f"    - {entry['dup_family_id']}: count={entry['count']} examples={entry.get('examples', [])}"
            for entry in top
            if isinstance(entry, Mapping)
        ]
        log.info("Largest duplicate families (post-QC):\n%s", "\n".join(lines))
    else:
        log.info("Largest duplicate families (post-QC): none")


__all__ = [
    "PostQCHook",
    "run_qc_over_jsonl",
    "iter_qc_rows_from_jsonl",
    "collect_qc_rows_from_jsonl",
    "emit_qc_csv",
]
