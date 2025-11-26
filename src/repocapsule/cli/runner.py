# runner.py
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass, replace, asdict
from pathlib import Path
from typing import Optional, Dict, Any, Sequence, Mapping, List, Iterator, Iterable, Tuple, Callable
import json
import os

from ..core.config import QCConfig, QCMode, RepocapsuleConfig, SourceSpec, SinkSpec
from ..core.factories import (
    SinkFactoryResult,
    make_output_paths_for_github,
    make_output_paths_for_pdf,
    make_qc_scorer,
    make_repo_context_from_git,
)
from ..core.interfaces import RepoContext
from ..core.licenses import detect_license_in_tree, apply_license_to_context
from ..core.builder import build_pipeline_plan, PipelineOverrides, PipelineRuntime
from ..core.pipeline import PipelineEngine
from ..core.concurrency import Executor, resolve_qc_executor_config
from ..sources.githubio import get_repo_info, parse_github_url, RepoSpec, detect_license_for_github_repo
from ..core.log import get_logger
from ..core.qc_controller import QCSummaryTracker, summarize_qc_rows
from ..core.records import RunSummaryMeta, is_summary_record
from ..core.qc_utils import open_jsonl_maybe_gz, open_jsonl_output_maybe_gz
from ..core.dataset_card import write_card_fragment_for_run

try:
    from ..core.extras.qc import JSONLQualityScorer, score_jsonl_to_csv, write_csv
except Exception:
    JSONLQualityScorer = None
    score_jsonl_to_csv = None
    write_csv = None

log = get_logger(__name__)


@dataclass(slots=True)
class GitHubRepoProfile:
    """Profile metadata for a GitHub repository conversion.

    Captures the parsed GitHub repo spec, the ref that will be used for
    conversion, the detected SPDX license identifier (if any), and the
    base ``RepoContext`` that should be propagated through the pipeline.

    Attributes:
        spec (RepoSpec): Parsed GitHub repository specification including
            owner, repo, and optional ref.
        ref (str): Git reference (branch, tag, or commit) resolved for
            this run.
        license_spdx (str | None): Detected SPDX license identifier or
            expression, if available.
        ctx (RepoContext): Repository context carrying repo-level
            metadata, including license, URL, and optional commit SHA.
    """
    spec: RepoSpec
    ref: str
    license_spdx: str | None
    ctx: RepoContext


@dataclass(slots=True)
class RunSummary:
    """Run-level summary that can be appended to primary outputs.

    The summary captures a snapshot of the configuration, aggregate
    pipeline statistics, QC summary information, and any extra metadata
    that should be embedded as a footer record in JSONL sinks.

    Attributes:
        config (Dict[str, Any]): Serialized configuration snapshot for
            the run.
        stats (Dict[str, Any]): Aggregate pipeline statistics as
            returned by the engine.
        qc_summary (dict | None): Aggregated QC summary for the run, or
            None if QC was disabled.
        metadata (Dict[str, Any]): Run-level metadata describing the
            corpus and primary outputs.
    """
    config: Dict[str, Any]
    stats: Dict[str, Any]
    qc_summary: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]

    def to_record(self) -> Dict[str, Any]:
        """Convert this run summary into a JSONL-compatible record.

        The record follows the standard Repocapsule schema with an empty
        ``text`` field and all run metadata placed under ``meta``.

        Returns:
            Dict[str, Any]: A JSON-serializable record ready to be
            written to JSONL sinks.
        """
        meta = RunSummaryMeta(
            config=self.config,
            stats=self.stats,
            qc_summary=self.qc_summary,
            metadata=self.metadata,
        )
        return {"text": "", "meta": meta.to_dict()}


def run_engine(engine: PipelineEngine) -> Dict[str, int]:
    """Run a prepared pipeline engine and handle QC/post-processing.

    Executes the pipeline, optionally runs inline or post-hoc QC,
    appends a run summary record to the primary JSONL output (if any),
    and emits a dataset card fragment for the run.

    For post-hoc QC (``QCConfig.mode == "post"``), the primary JSONL is
    rescored using a ``QualityScorer`` obtained either from the runtime
    (``plan.runtime.post_qc_scorer``) or from the registry via
    ``make_qc_scorer``. When ``qc.parallel_post`` is enabled, scoring
    may use a process-based executor for better throughput. QC summaries
    are merged back into the run statistics.

    Args:
        engine (PipelineEngine): A pipeline engine whose plan and config
            were produced by ``build_pipeline_plan``.

    Returns:
        Dict[str, int]: Aggregate pipeline statistics keyed by counter
        name.
    """
    cfg = engine.config
    stats_obj = engine.run()
    stats_dict = stats_obj.as_dict()

    log.info("convert complete: %s", stats_dict)

    jsonl_path = cfg.sinks.primary_jsonl_name or cfg.metadata.primary_jsonl
    qc_summary: Optional[Dict[str, Any]] = None

    qc_cfg = cfg.qc
    mode = qc_cfg.mode
    qc_enabled = bool(qc_cfg.enabled) and mode != QCMode.OFF
    if qc_enabled:
        if mode == QCMode.POST:
            runtime_post_scorer = getattr(engine.plan.runtime, "post_qc_scorer", None)
            if JSONLQualityScorer is None and runtime_post_scorer is None:
                log.warning("Post-QC requested but QC extras are not installed; skipping QC.")
                qc_enabled = False
            else:
                qc_summary = _run_post_qc(jsonl_path, cfg, runtime=engine.plan.runtime)
                stats_obj.qc = QCSummaryTracker.from_summary_dict(qc_summary)
        else:
            qc_summary = stats_obj.qc.as_dict()

        if cfg.qc.write_csv and mode == QCMode.INLINE:
            out_csv = _derive_csv_path(jsonl_path, cfg.qc.csv_suffix)
            if out_csv:
                scorer_for_csv = getattr(engine.plan.runtime, "qc_scorer_for_csv", None)
                if scorer_for_csv is not None and write_csv is not None:
                    reset = getattr(scorer_for_csv, "reset_state", None)
                    if callable(reset):
                        reset()
                    rows = scorer_for_csv.score_jsonl_path(
                        str(jsonl_path),
                        fail_on_error=bool(cfg.qc.fail_on_error),
                    )
                    stats = getattr(scorer_for_csv, "last_stats", None)
                    write_csv(rows, out_csv)
                    if stats is not None:
                        err_count = getattr(stats, "parse_errors", 0) + getattr(stats, "score_errors", 0)
                        if err_count:
                            if hasattr(stats_obj, "qc"):
                                try:
                                    stats_obj.qc.errors += int(err_count)  # type: ignore[attr-defined]
                                except Exception:
                                    pass
                            log.warning(
                                "QC CSV scoring for %s skipped lines (total=%s, scored=%s, parse_errors=%s, score_errors=%s)",
                                jsonl_path,
                                getattr(stats, "total_lines", None),
                                getattr(stats, "scored_ok", None),
                                getattr(stats, "parse_errors", None),
                                getattr(stats, "score_errors", None),
                            )
                            if hasattr(stats, "error_examples"):
                                for example in getattr(stats, "error_examples", [])[:3]:
                                    log.debug("QC CSV example skip: %s", example)
                else:
                    if score_jsonl_to_csv is None:
                        raise RuntimeError("QC CSV helpers unavailable; reinstall optional dependencies.")
                    score_jsonl_to_csv(str(jsonl_path), out_csv)

    qc_summary = qc_summary or stats_obj.qc.as_dict()
    stats_dict = stats_obj.as_dict()
    summary_record = RunSummary(
        config=cfg.to_dict(),
        stats=stats_dict,
        qc_summary=qc_summary if isinstance(qc_summary, dict) else None,
        metadata=cfg.metadata.to_dict(),
    ).to_record()
    _dispatch_finalizers(engine.plan.runtime.sinks, summary_record, jsonl_path, cfg.sinks.context)
    try:
        dc_cfg = getattr(cfg, "dataset_card", None)
        if dc_cfg is None or getattr(dc_cfg, "enabled", True):
            write_card_fragment_for_run(cfg, stats_obj)
    except Exception:
        log.exception("Failed to write dataset card fragment")

    return stats_obj.as_dict()


# ---------- One generic entry point ----------
def convert(config: RepocapsuleConfig | PipelineEngine, *, overrides: PipelineOverrides | None = None) -> Dict[str, int]:
    """Convert sources to datasets using a config or prepared engine.

    This is the main programmatic entry point for Repocapsule. When
    given a ``RepocapsuleConfig``, it builds a ``PipelinePlan`` via the
    builder, constructs a ``PipelineEngine``, and runs it. When given an
    existing ``PipelineEngine``, it simply runs that engine.

    Callers adding new functionality should plug into the builder,
    registries, and factories used here, rather than constructing ad-hoc
    pipelines in this function.

    Args:
        config (RepocapsuleConfig | PipelineEngine): Either a declarative
            configuration for building a plan/engine or an already
            prepared engine instance.
        overrides (PipelineOverrides | None): Optional pipeline overrides
            applied during plan construction when a config is provided.

    Returns:
        Dict[str, int]: Aggregate statistics for the completed run.
    """
    if isinstance(config, PipelineEngine):
        return run_engine(config)

    plan = build_pipeline_plan(config, overrides=overrides)
    cfg = plan.spec
    engine = PipelineEngine(plan)
    return run_engine(engine)


def _run_post_qc(jsonl_path: Optional[str], config: RepocapsuleConfig, runtime: PipelineRuntime | None = None) -> Dict[str, Any]:
    """Run post-hoc QC over the primary JSONL file.

    Rescores the primary JSONL using a quality scorer resolved from the
    runtime (``post_qc_scorer``) or from the QC registry via
    ``make_qc_scorer``. Supports sequential and parallel scoring
    (controlled by ``qc.parallel_post``) and can optionally write a QC
    CSV file when ``qc.write_csv`` is enabled. QC summaries are
    aggregated and returned as a dictionary.

    Args:
        jsonl_path (str | None): Path to the primary JSONL file. Must
            not be None.
        config (RepocapsuleConfig): Configuration object whose QC
            section controls scorer construction and scoring options.
        runtime (PipelineRuntime | None): Optional runtime carrying
            scorer instances and executor hints.

    Returns:
        Dict[str, Any]: Summary of QC results, including scored counts,
        candidate counts, and error statistics.

    Raises:
        RuntimeError: If the JSONL path is missing or a scorer cannot be
            constructed given the current configuration.
    """
    if not jsonl_path:
        raise RuntimeError("Cannot run post-QC summary without a primary JSONL path")
    qc_cfg = config.qc
    runtime_scorer = getattr(runtime, "post_qc_scorer", None) if runtime is not None else None
    scorer = runtime_scorer or make_qc_scorer(qc_cfg)
    if scorer is None:
        raise RuntimeError("QC scorer unavailable for post-QC; configuration should have disabled QC when extras were missing.")

    tracker = QCSummaryTracker(
        enabled=bool(qc_cfg.enabled),
        mode=qc_cfg.mode,
        min_score=qc_cfg.min_score,
        drop_near_dups=bool(qc_cfg.drop_near_dups),
    )
    rows_for_csv: Optional[List[Dict[str, Any]]] = [] if qc_cfg.write_csv else None

    def _consume_rows(rows: Iterable[Dict[str, Any]]) -> None:
        for row in rows:
            tracker.observe(row, apply_gates=False)
            if rows_for_csv is not None:
                rows_for_csv.append(row)

    jsonl_path_str = str(jsonl_path)

    if not qc_cfg.write_csv:
        if qc_cfg.parallel_post:
            ok = _score_jsonl_parallel_streaming(
                _iter_jsonl_shards(jsonl_path_str),
                qc_cfg,
                config,
                runtime,
                _consume_rows,
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
                parallel_rows = _score_jsonl_parallel_collecting(shards, qc_cfg, config, runtime)
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
        out_csv = _derive_csv_path(jsonl_path, qc_cfg.csv_suffix)
        if qc_cfg.write_csv and out_csv:
            if write_csv is not None:
                write_csv(rows, out_csv)
            elif score_jsonl_to_csv is not None:
                score_jsonl_to_csv(jsonl_path_str, out_csv)
            else:
                raise RuntimeError("QC CSV helpers unavailable; reinstall optional dependencies.")

    _log_post_qc_summary(summary)
    return summary


def _derive_csv_path(jsonl_path: Optional[str], suffix: Optional[str]) -> Optional[str]:
    """Derive a QC CSV path from a primary JSONL path and suffix.

    Args:
        jsonl_path (str | None): Path to the primary JSONL file, or
            None if no QC CSV should be produced.
        suffix (str | None): Suffix to append to the base name. If not
            provided, ``"_quality.csv"`` is used.

    Returns:
        str | None: Derived CSV path, or None if ``jsonl_path`` is
        missing.
    """
    if not jsonl_path:
        return None
    suffix = suffix or "_quality.csv"
    base = str(jsonl_path)
    if base.endswith(".jsonl"):
        base = base[:-6]  # remove '.jsonl'
    return base + suffix


def _append_run_summary(jsonl_path: str, summary: RunSummary) -> None:
    """Append a run summary record to a JSONL file.

    Uses the JSONL output helper so that gzipped primary files are also
    supported.

    Args:
        jsonl_path (str): Path to the JSONL file to append to.
        summary (RunSummary | dict): Summary object or pre-built record
            to write as a single line.
    """
    record = summary if isinstance(summary, dict) else summary.to_record()
    with open_jsonl_output_maybe_gz(jsonl_path, "a") as fp:
        fp.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        fp.write("\n")


def _log_post_qc_summary(summary: Dict[str, Any]) -> None:
    """Log a concise human-readable post-QC summary.

    Logs aggregate counts and, when available, the largest duplicate
    families discovered during QC.

    Args:
        summary (dict): QC summary dictionary produced by post-QC
            scoring.
    """
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
        ]
        log.info("Largest duplicate families (post-QC):\n%s", "\n".join(lines))
    else:
        log.info("Largest duplicate families (post-QC): none")


def _dispatch_finalizers(
    sinks: Sequence[Sink],
    summary_record: Dict[str, Any],
    primary_jsonl: Optional[str],
    context: Optional[RepoContext] = None,
) -> None:
    """Dispatch finalize hooks to sinks and ensure JSONL footer behavior.

    Invokes ``finalize()`` on all sinks that provide it, passing the
    run-summary record. If no JSONL sink wrote the summary, a legacy
    footer line is appended directly to the primary JSONL file.

    Args:
        sinks (Sequence[Sink]): All sinks associated with the pipeline
            plan.
        summary_record (dict): Run summary record to pass to sinks.
        primary_jsonl (str | None): Path to the primary JSONL file, used
            when a direct footer append is required.
        context (RepoContext | None): Optional repository context
            (currently unused but reserved for future extensions).
    """
    from ..sinks.sinks import JSONLSink, GzipJSONLSink  # local import to avoid cycles

    wrote_jsonl = False
    for sink in sinks:
        finalize = getattr(sink, "finalize", None)
        if callable(finalize):
            try:
                finalize([summary_record])
                if isinstance(sink, (JSONLSink, GzipJSONLSink)):
                    wrote_jsonl = True
            except Exception as exc:
                log.warning("Sink %s failed to finalize: %s", type(sink).__name__, exc)
    # Preserve legacy JSONL footer write when we haven't already written via a JSONL sink
    if primary_jsonl and not wrote_jsonl:
        _append_run_summary(primary_jsonl, summary_record)


def _score_jsonl_sequential(
    shards: List[_JsonlShard],
    scorer,
    *,
    fail_on_error: bool = False,
    tracker: Optional[QCSummaryTracker] = None,
) -> List[Dict[str, Any]]:
    """Score JSONL shards sequentially using a single scorer instance.

    Args:
        shards (list[_JsonlShard]): Shards containing raw JSONL lines.
        scorer: Quality scorer implementing ``score_record``.
        fail_on_error (bool): Whether to raise on parse/score errors
            instead of logging and continuing.
        tracker (QCSummaryTracker | None): Optional tracker for recording
            QC errors.

    Returns:
        list[dict]: QC rows produced for all non-summary records.
    """
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
    """Stream JSONL shards sequentially and feed QC rows to a consumer.

    Args:
        jsonl_path (str): Path to the primary JSONL file.
        scorer: Quality scorer implementing ``score_record``.
        consume_rows (Callable): Callback that receives batches of QC
            rows per shard.
        fail_on_error (bool): Whether to raise on parse/score errors.
        tracker (QCSummaryTracker | None): Optional tracker for error
            accounting.
    """
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
    qc_cfg,
    config: RepocapsuleConfig,
    runtime: PipelineRuntime | None,
    handle_rows: Callable[[int, List[Dict[str, Any]]], None],
) -> bool:
    """Run QC scoring in parallel over a collection of shards.

    Uses the QC executor configuration to dispatch work to threads or
    processes. In process mode, scorers are constructed once per
    process via ``_qc_worker_initializer``.

    Args:
        shards (Iterable[_JsonlShard]): Shards to score.
        qc_cfg: QC configuration object used to construct scorers.
        config (RepocapsuleConfig): Global configuration for executor
            inference.
        runtime (PipelineRuntime | None): Optional runtime carrying
            scorer instances and executor hints.
        handle_rows (Callable): Callback accepting ``(shard_index,
            rows)`` results for each completed shard.

    Returns:
        bool: True if parallel scoring completed successfully, False if a
        fallback to sequential is required.
    """
    try:
        runtime_scorer = getattr(runtime, "post_qc_scorer", None) if runtime is not None else None
        scorer_proto = runtime_scorer or make_qc_scorer(qc_cfg, new_instance=True)
    except Exception:
        scorer_proto = None
    if scorer_proto is None:
        log.warning("QC parallel_post requested but scorer could not be instantiated; falling back to sequential mode.")
        return False

    # Prefer scorer.clone_for_parallel() when available so workers get fresh, isolated scorer instances.
    def _scorer_factory():
        cloned = getattr(scorer_proto, "clone_for_parallel", None)
        if callable(cloned):
            return cloned()
        additional = make_qc_scorer(qc_cfg, new_instance=True)
        if additional is None:
            raise RuntimeError("Unable to create scorer for parallel QC processing.")
        return additional

    exec_cfg = resolve_qc_executor_config(config, runtime=runtime)
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
    shards: List[_JsonlShard],
    qc_cfg,
    config: RepocapsuleConfig,
    runtime: PipelineRuntime | None,
) -> Optional[List[Dict[str, Any]]]:
    """Score JSONL shards in parallel and return a merged row list.

    Args:
        shards (list[_JsonlShard]): Shards to score.
        qc_cfg: QC configuration governing scorer construction.
        config (RepocapsuleConfig): Global configuration object.
        runtime (PipelineRuntime | None): Optional runtime with scorer
            instances.

    Returns:
        list[dict] | None: Collected QC rows ordered by shard index, or
        None if parallel QC could not be run and a sequential fallback is
        required.
    """
    shard_results: Dict[int, List[Dict[str, Any]]] = {}

    def _handle_rows(idx: int, rows: List[Dict[str, Any]]) -> None:
        shard_results[idx] = rows

    ok = _run_parallel_qc(shards, qc_cfg, config, runtime, _handle_rows)
    if not ok:
        return None

    rows: List[Dict[str, Any]] = []
    for idx in sorted(shard_results):
        rows.extend(shard_results[idx])
    return rows


def _score_jsonl_parallel_streaming(
    shards: Iterable[_JsonlShard],
    qc_cfg,
    config: RepocapsuleConfig,
    runtime: PipelineRuntime | None,
    consume_rows: Callable[[Iterable[Dict[str, Any]]], None],
) -> bool:
    """Score JSONL shards in parallel and stream rows to a consumer.

    Args:
        shards (Iterable[_JsonlShard]): Shards to score.
        qc_cfg: QC configuration object.
        config (RepocapsuleConfig): Global configuration for executor
            selection.
        runtime (PipelineRuntime | None): Optional runtime carrying
            scorer instances and hints.
        consume_rows (Callable): Callback that receives each batch of QC
            rows.

    Returns:
        bool: True if parallel scoring completed successfully, False if a
        sequential fallback is required.
    """
    def _handle_rows(_: int, rows: List[Dict[str, Any]]) -> None:
        consume_rows(rows)

    return _run_parallel_qc(shards, qc_cfg, config, runtime, _handle_rows)


def _clone_base_config(base_config: Optional[RepocapsuleConfig]) -> RepocapsuleConfig:
    """Clone a base configuration or build a fresh default one.

    Uses a shallow ``dataclasses.replace`` so that the returned config
    can be mutated without affecting the original top-level object.

    Args:
        base_config (RepocapsuleConfig | None): Optional config to clone.

    Returns:
        RepocapsuleConfig: Cloned or freshly constructed configuration.
    """
    return replace(base_config) if base_config is not None else RepocapsuleConfig()


def _build_github_repo_profile(
    url: str,
    *,
    base_context: RepoContext | None = None,
) -> GitHubRepoProfile:
    """Build a GitHub repository profile from a URL.

    Parses the URL, queries the GitHub API for metadata, attempts to
    detect the license, and constructs a ``GitHubRepoProfile`` with a
    suitable ``RepoContext`` for downstream use.

    Args:
        url (str): GitHub repository URL (optionally including a ref).
        base_context (RepoContext | None): Optional starting context to
            extend with GitHub metadata.

    Returns:
        GitHubRepoProfile: Profile containing spec, ref, license, and
        context for the repository.

    Raises:
        ValueError: If the URL cannot be parsed as a GitHub repository.
    """
    spec = parse_github_url(url)
    if not spec:
        raise ValueError(f"Invalid GitHub URL: {url!r}")
    info: Dict[str, Any] | None = None
    try:
        info = get_repo_info(spec)
    except Exception:
        info = None
    ref = spec.ref or ((info or {}).get("default_branch")) or "main"
    detected_license: str | None
    try:
        detected_license = detect_license_for_github_repo(spec, ref=ref)
    except Exception:
        detected_license = None
    api_license = (info or {}).get("license_spdx")
    license_spdx = detected_license or api_license
    ctx = base_context
    if ctx is None:
        ctx = RepoContext(
            repo_full_name=f"{spec.owner}/{spec.repo}",
            repo_url=f"https://github.com/{spec.owner}/{spec.repo}",
            license_id=None,
            commit_sha=None,
            extra={"source": "github", "ref": ref},
        )
    if not ctx.license_id and license_spdx:
        ctx = RepoContext(
            repo_full_name=ctx.repo_full_name,
            repo_url=ctx.repo_url,
            license_id=license_spdx,
            commit_sha=ctx.commit_sha,
            extra=ctx.extra,
        )
    return GitHubRepoProfile(spec=spec, ref=ref, license_spdx=license_spdx, ctx=ctx)


# ---------- Path helpers ----------
def default_paths_for_github(
    url: str,
    out_dir: str | os.PathLike[str] = ".",
    *,
    include_prompt: bool = True,
) -> tuple[str, str | None, RepoContext]:
    """Compute default output paths and context for a GitHub repository.

    Uses best-effort GitHub API calls to determine the default branch
    and SPDX license, then delegates to ``make_output_paths_for_github``
    to construct output paths.

    Args:
        url (str): GitHub repository URL.
        out_dir (str | os.PathLike): Base directory for outputs.
        include_prompt (bool): Whether to allocate a prompt file path in
            addition to the primary JSONL path.

    Returns:
        tuple[str, str | None, RepoContext]: Primary JSONL path, prompt
        path (or None), and constructed ``RepoContext``.
    """
    profile = _build_github_repo_profile(url)
    outputs = make_output_paths_for_github(
        owner=profile.spec.owner,
        repo=profile.spec.repo,
        ref=profile.ref,
        license_spdx=profile.license_spdx,
        out_dir=Path(out_dir),
        include_prompt=include_prompt,
    )
    jsonl = str(outputs.jsonl)
    prompt = str(outputs.prompt) if outputs.prompt else None
    return jsonl, prompt, profile.ctx


def default_paths_for_pdf(
    *,
    url: str,
    title: str | None = None,
    license_spdx: str | None = None,
    out_dir: str | os.PathLike[str] = ".",
    include_prompt: bool = True,
) -> tuple[str, str | None]:
    """Compute default output paths for a PDF corpus.

    Delegates to ``make_output_paths_for_pdf`` using the provided URL,
    title, and license to derive deterministic JSONL and optional prompt
    paths.

    Args:
        url (str): Canonical URL for the PDF or PDF corpus.
        title (str | None): Optional human-readable title.
        license_spdx (str | None): Optional SPDX license identifier.
        out_dir (str | os.PathLike): Base directory for outputs.
        include_prompt (bool): Whether to allocate a prompt file path.

    Returns:
        tuple[str, str | None]: Primary JSONL path and optional prompt
        path.
    """
    outputs = make_output_paths_for_pdf(
        url=url,
        title=title,
        license_spdx=license_spdx,
        out_dir=Path(out_dir),
        include_prompt=include_prompt,
    )
    return str(outputs.jsonl), (str(outputs.prompt) if outputs.prompt else None)


# ---------- Convenience wrappers ----------
def convert_local_dir(
    root_dir: str | Path,
    out_jsonl: str | Path,
    *,
    out_prompt: str | Path | None = None,
    base_config: Optional[RepocapsuleConfig] = None,
) -> Dict[str, int]:
    """Convert a local directory into a dataset.

    Builds a ``RepocapsuleConfig`` for a local directory via
    ``make_local_profile`` and runs the engine via ``convert``.

    Args:
        root_dir (str | Path): Root directory of the repository or
            corpus to ingest.
        out_jsonl (str | Path): Path where the primary JSONL output will
            be written.
        out_prompt (str | Path | None): Optional path for a prompt file.
        base_config (RepocapsuleConfig | None): Optional base config to
            clone and extend.

    Returns:
        Dict[str, int]: Aggregate statistics for the completed run.
    """
    cfg = make_local_profile(
        root_dir,
        out_jsonl,
        out_prompt=out_prompt,
        base_config=base_config,
    )
    return convert(cfg)


def convert_github(
    url: str,
    out_jsonl: str | Path,
    *,
    out_prompt: str | Path | None = None,
    base_config: Optional[RepocapsuleConfig] = None,
) -> Dict[str, int]:
    """Convert a GitHub repository into a dataset.

    Builds a ``RepocapsuleConfig`` for a GitHub repository via
    ``make_github_profile`` and runs the engine via ``convert``.

    Args:
        url (str): GitHub repository URL.
        out_jsonl (str | Path): Path where the primary JSONL output will
            be written.
        out_prompt (str | Path | None): Optional path for a prompt file.
        base_config (RepocapsuleConfig | None): Optional base config to
            clone and extend.

    Returns:
        Dict[str, int]: Aggregate statistics for the completed run.
    """
    cfg = make_github_profile(
        url,
        out_jsonl,
        out_prompt=out_prompt,
        base_config=base_config,
    )
    return convert(cfg)


def make_local_repo_config(
    *,
    root_dir: str | Path,
    out_jsonl: str | Path,
    out_prompt: str | Path | None = None,
    base_config: Optional[RepocapsuleConfig] = None,
) -> RepocapsuleConfig:
    """Build a configuration for a local repository without running it.

    Convenience helper for callers that want to inspect or tweak the
    generated ``RepocapsuleConfig`` for a local directory before running
    the pipeline.

    Args:
        root_dir (str | Path): Root directory of the repository or
            corpus to ingest.
        out_jsonl (str | Path): Path to the primary JSONL output.
        out_prompt (str | Path | None): Optional prompt file path.
        base_config (RepocapsuleConfig | None): Optional base config to
            clone.

    Returns:
        RepocapsuleConfig: Prepared configuration for the local
        repository.
    """

    return make_local_profile(
        root_dir=root_dir,
        out_jsonl=out_jsonl,
        out_prompt=out_prompt,
        base_config=base_config,
    )


def make_github_repo_config(
    *,
    url: str,
    out_jsonl: str | Path,
    out_prompt: str | Path | None = None,
    base_config: Optional[RepocapsuleConfig] = None,
) -> RepocapsuleConfig:
    """Build a configuration for a GitHub repository without running it.

    Convenience helper for callers that want to inspect or customize the
    generated ``RepocapsuleConfig`` for a GitHub repo.

    Args:
        url (str): GitHub repository URL.
        out_jsonl (str | Path): Path to the primary JSONL output.
        out_prompt (str | Path | None): Optional prompt file path.
        base_config (RepocapsuleConfig | None): Optional base config to
            clone.

    Returns:
        RepocapsuleConfig: Prepared configuration for the GitHub
        repository.
    """

    return make_github_profile(
        url=url,
        out_jsonl=out_jsonl,
        out_prompt=out_prompt,
        base_config=base_config,
    )


def make_local_profile(
    root_dir: str | Path,
    out_jsonl: str | Path,
    *,
    out_prompt: str | Path | None = None,
    base_config: Optional[RepocapsuleConfig] = None,
) -> RepocapsuleConfig:
    """Construct a local-directory profile config.

    Clones or builds a base ``RepocapsuleConfig``, infers a
    ``RepoContext`` (including Git metadata when available), attempts to
    detect a license from the filesystem, and wires source/sink specs
    and metadata for a local directory run.

    Args:
        root_dir (str | Path): Root directory of the repository or
            corpus to ingest.
        out_jsonl (str | Path): Path to the primary JSONL output.
        out_prompt (str | Path | None): Optional prompt file path.
        base_config (RepocapsuleConfig | None): Optional config to
            clone.

    Returns:
        RepocapsuleConfig: Fully wired configuration for the local run.
    """
    cfg = _clone_base_config(base_config)
    ctx = (
        cfg.sinks.context
        or make_repo_context_from_git(Path(root_dir))
        or RepoContext(extra={"source": "local"})
    )
    if not ctx.license_id:
        license_id, meta = detect_license_in_tree(str(root_dir), None)
        if license_id:
            ctx = apply_license_to_context(ctx, license_id, meta)
    cfg.sinks.context = ctx
    cfg.sources.specs = [
        SourceSpec(kind="local_dir", options={"root_dir": str(root_dir)}),
    ]
    cfg.sinks.specs = [
        SinkSpec(
            kind="default_jsonl_prompt",
            options={
                "jsonl_path": str(out_jsonl),
                "prompt_path": str(out_prompt) if out_prompt is not None else None,
            },
        )
    ]
    extra_meta = {"repo_url": ctx.repo_url} if ctx.repo_url else {}
    meta_update = {"primary_jsonl": str(out_jsonl)}
    if out_prompt is not None:
        meta_update["prompt_path"] = str(out_prompt)
    meta_update.update(extra_meta)
    cfg.metadata = cfg.metadata.merged(meta_update)
    return cfg


def make_github_profile(
    url: str,
    out_jsonl: str | Path,
    *,
    out_prompt: str | Path | None = None,
    base_config: Optional[RepocapsuleConfig] = None,
    repo_context: Optional[RepoContext] = None,
) -> RepocapsuleConfig:
    """Construct a GitHub repository profile config.

    Clones or builds a base ``RepocapsuleConfig``, constructs or
    reuses a ``RepoContext`` for the GitHub repository (including
    detected SPDX license where possible), and wires source/sink specs
    and metadata for a GitHub zipball run.

    Args:
        url (str): GitHub repository URL.
        out_jsonl (str | Path): Path to the primary JSONL output.
        out_prompt (str | Path | None): Optional prompt file path.
        base_config (RepocapsuleConfig | None): Optional base config to
            clone.
        repo_context (RepoContext | None): Optional pre-built context to
            seed the profile.

    Returns:
        RepocapsuleConfig: Fully wired configuration for the GitHub run.
    """
    cfg = _clone_base_config(base_config)
    base_ctx = repo_context or cfg.sinks.context
    profile = _build_github_repo_profile(url, base_context=base_ctx)
    ctx = profile.ctx
    cfg.sinks.context = ctx
    cfg.sources.specs = [
        SourceSpec(kind="github_zip", options={"url": url}),
    ]
    cfg.sinks.specs = [
        SinkSpec(
            kind="default_jsonl_prompt",
            options={
                "jsonl_path": str(out_jsonl),
                "prompt_path": str(out_prompt) if out_prompt is not None else None,
            },
        )
    ]
    extra_meta = {"repo_url": ctx.repo_url} if ctx.repo_url else {}
    meta_update = {"primary_jsonl": str(out_jsonl)}
    if out_prompt is not None:
        meta_update["prompt_path"] = str(out_prompt)
    meta_update.update(extra_meta)
    cfg.metadata = cfg.metadata.merged(meta_update)
    return cfg

@dataclass(slots=True)
class _JsonlShard:
    """In-memory representation of a JSONL shard.

    Attributes:
        index (int): Shard index in the overall sequence.
        lines (list[str]): Raw JSONL lines contained in the shard.
        path (str): Path to the source JSONL file.
    """
    index: int
    lines: List[str]
    path: str


_QC_WORKER_SCORER: Optional[Any] = None


def _qc_worker_initializer(qc_cfg_payload: Dict[str, Any]) -> None:
    """Initialize a process-local QC scorer for parallel workers.

    Constructs a ``QCConfig`` from a serialized payload and builds a new
    scorer instance to be reused by the worker process.

    Args:
        qc_cfg_payload (dict): Serialized QC configuration suitable for
            constructing a ``QCConfig``.
    """
    global _QC_WORKER_SCORER
    qc_cfg = QCConfig(**qc_cfg_payload)
    scorer = make_qc_scorer(qc_cfg, new_instance=True)
    if scorer is None:
        raise RuntimeError("QC worker failed to initialize scorer (missing optional dependencies?).")
    _QC_WORKER_SCORER = scorer


def _qc_parallel_worker(shard: _JsonlShard) -> Tuple[int, List[Dict[str, Any]]]:
    """Score a single shard using the process-local scorer.

    Args:
        shard (_JsonlShard): Shard whose lines should be scored.

    Returns:
        tuple[int, list[dict]]: Shard index and QC rows produced for
        that shard.

    Raises:
        RuntimeError: If the worker scorer was not initialized.
    """
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
    """Iterate over a JSONL file in fixed-size shards.

    Uses the JSONL helper so that gzipped files are supported. Empty
    lines are skipped.

    Args:
        path (str): Path to the JSONL (or gzipped JSONL) file.
        shard_size (int): Maximum number of lines per shard.

    Yields:
        _JsonlShard: Shards containing up to ``shard_size`` lines each.
    """
    chunk: List[str] = []
    idx = 0
    # Use helper so post-QC works with gzipped primary JSONL files.
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
    """Score a batch of JSONL lines with a quality scorer.

    Parses raw JSONL lines into records, skips summary records, and
    passes each record to ``scorer.score_record``. Errors are either
    logged or raised depending on ``fail_on_error``. QC errors are
    recorded in an optional tracker.

    Args:
        lines (list[str]): Raw JSONL lines to score.
        scorer: Quality scorer implementing ``score_record``.
        source_path (str): Path of the originating JSONL file.
        shard_label (str | None): Human-readable label used in logs.
        fail_on_error (bool): Whether to raise on parse/score errors.
        tracker (QCSummaryTracker | None): Optional tracker for error
            accounting.

    Returns:
        list[dict]: QC rows produced for non-summary records.
    """
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
        if is_summary_record(record):
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
