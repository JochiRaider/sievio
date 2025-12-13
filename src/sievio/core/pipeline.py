# pipeline.py
# SPDX-License-Identifier: MIT
"""Pipeline execution engine coordinating sources, sinks, and middleware."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from typing import Optional, Iterable, Sequence, Dict, List, Tuple, Any, Callable, Mapping
from pathlib import Path
import pickle

from .config import SievioConfig, FileProcessingConfig
from .builder import PipelinePlan, PipelineOverrides, PipelineRuntime, build_pipeline_plan, build_engine
from .interfaces import (
    Source,
    Sink,
    RepoContext,
    Record,
    FileExtractor,
    ClosableSource,
    RecordMiddleware,
    FileMiddleware,
    RunContext,
    RunLifecycleHook,
    RunSummaryView,
)
from .concurrency import (
    Executor,
    process_items_parallel,
    resolve_pipeline_executor_config,
)
from .convert import DefaultExtractor
from .log import get_logger
from .qc_controller import QCSummaryTracker


log = get_logger(__name__)


@dataclass(frozen=True)
class _WorkItem:
    item: Any
    ctx: Optional[RepoContext]


class _FuncRecordMiddleware:
    """Internal adapter to treat bare functions as RecordMiddleware."""

    def __init__(self, fn: Callable[[Record], Optional[Record]]) -> None:
        self._fn = fn

    def process(self, record: Record) -> Optional[Record]:
        return self._fn(record)

    def __call__(self, record: Record) -> Optional[Record]:
        return self.process(record)


class _FuncFileMiddleware:
    """Internal adapter to treat bare functions as FileMiddleware."""

    def __init__(self, fn: Callable[[Any, Iterable[Record]], Optional[Iterable[Record]]]) -> None:
        self._fn = fn

    def process(self, item: Any, records: Iterable[Record]) -> Optional[Iterable[Record]]:
        return self._fn(item, records)


def apply_overrides_to_engine(
    engine: "PipelineEngine",
    overrides: "PipelineOverrides | None",
) -> "PipelineEngine":
    """
    Apply runtime-only overrides that target the PipelineEngine itself.

    Currently this wires record/file middlewares defined in
    PipelineOverrides onto the engine via add_record_middleware and
    add_file_middleware.
    """
    if overrides is None:
        return engine

    if getattr(overrides, "record_middlewares", None):
        for mw in overrides.record_middlewares:  # type: ignore[attr-defined]
            engine.add_record_middleware(mw)

    if getattr(overrides, "file_middlewares", None):
        for mw in overrides.file_middlewares:  # type: ignore[attr-defined]
            engine.add_file_middleware(mw)

    return engine


def _coerce_record_middleware(mw: RecordMiddleware | Callable[[Record], Any]) -> RecordMiddleware:
    """Wrap a bare callable as RecordMiddleware when needed."""
    return mw if hasattr(mw, "process") else _FuncRecordMiddleware(mw)  # type: ignore[arg-type]


def _coerce_file_middleware(mw: FileMiddleware | Callable[[Any, Iterable[Record]], Any]) -> FileMiddleware:
    """Wrap a bare callable as FileMiddleware when needed."""
    return mw if hasattr(mw, "process") else _FuncFileMiddleware(mw)  # type: ignore[arg-type]


@dataclass
class _ProcessFileCallable:
    config: FileProcessingConfig
    file_extractor: FileExtractor
    materialize: bool = False

    def __call__(self, work: _WorkItem) -> Tuple[Any, Iterable[Record]]:
        """Extract records from a work item using the configured extractor.

        Args:
            work (_WorkItem): Work item containing the source object and
                optional context.

        Returns:
            tuple[Any, Iterable[Record]]: The original item and an iterable of
                extracted records (materialized when configured).
        """
        item = work.item
        ctx = work.ctx
        rel = getattr(item, "path", None) or getattr(item, "rel_path", None)
        if rel is None:
            raise ValueError("FileItem missing 'path'")
        recs_iter = self.file_extractor.extract(
            item,
            config=self.config,
            context=ctx,
        )
        if self.materialize:
            recs_iter = list(recs_iter)
        return item, recs_iter

# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

# Convention: hot-path dataclasses use slots=True to reduce per-instance overhead.
@dataclass(slots=True)
class PipelineStats:
    """Mutable counters and aggregates collected during pipeline execution.

    files/bytes track items that completed extraction, middleware, and sink writes
    without unhandled errors (not just files attempted). ``qc`` holds screening
    (quality + safety) stats for the run.
    """

    files: int = 0
    bytes: int = 0
    records: int = 0
    sink_errors: int = 0
    source_errors: int = 0
    by_ext: Dict[str, int] = field(default_factory=dict)
    qc: QCSummaryTracker = field(default_factory=QCSummaryTracker)
    primary_jsonl_path: str | None = None

    def as_dict(self) -> Dict[str, object]:
        """Return a stable dict shape for reporting and JSONL footers."""
        data: Dict[str, object] = {
            "files": int(self.files),
            "bytes": int(self.bytes),
            "records": int(self.records),
            "sink_errors": int(self.sink_errors),
            "source_errors": int(self.source_errors),
            "by_ext": dict(self.by_ext),
        }
        data["qc"] = self.qc.as_dict()
        return data

    def qc_top_dup_families(self) -> List[Dict[str, Any]]:
        """Return duplicate-family summary for reporting."""
        return self.qc.top_dup_families()

    def to_summary_view(self, *, primary_jsonl_path: str | None = None) -> RunSummaryView:
        """Return a lightweight summary view for hooks and reporting."""
        if primary_jsonl_path is None:
            primary_jsonl_path = self.primary_jsonl_path

        qc_summary = self.qc.as_dict() if hasattr(self, "qc") and self.qc else None
        return RunSummaryView(
            num_records=int(self.records),
            ext_counts=dict(self.by_ext),
            qc_summary=qc_summary,
            primary_jsonl_path=primary_jsonl_path,
        )




# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _open_source_with_stack(stack: ExitStack, src: Source) -> Source:
    """Enter a source context if supported; otherwise register close().

    Args:
        stack (ExitStack): ExitStack to manage clean-up.
        src (Source): Source instance to enter or register.

    Returns:
        Source: Source object, potentially context-managed.
    """
    enter = getattr(src, "__enter__", None)
    if callable(enter):
        return stack.enter_context(src)  
    close = getattr(src, "close", None)
    if callable(close):
        stack.callback(close)  
    return src

def _prepare_sinks(stack: ExitStack, sinks: Sequence[Sink], ctx: Optional[RepoContext], stats: PipelineStats) -> List[Sink]:
    """Open sinks once, tracking failures in stats.

    Args:
        stack (ExitStack): ExitStack to register closers.
        sinks (Sequence[Sink]): Sinks to open.
        ctx (RepoContext | None): Optional repository context passed to open().
        stats (PipelineStats): Stats container for error tracking.

    Returns:
        list[Sink]: Sinks successfully opened.
    """
    open_sinks: List[Sink] = []
    for s in sinks:
        try:
            # Call explicit open(context) if available.
            open_fn = getattr(s, "open", None)
            if callable(open_fn):
                open_fn(ctx)  
            # Ensure we close at the end if close exists.
            close_fn = getattr(s, "close", None)
            if callable(close_fn):
                stack.callback(close_fn)  
            open_sinks.append(s)
        except Exception as e:
            log.warning("Sink %s failed to open: %s", getattr(s, "__class__", type(s)).__name__, e)
            stats.sink_errors += 1
    return open_sinks


def _get_context_from_source(source: Source) -> Optional[RepoContext]:
    """Extract RepoContext from a source when available."""
    return getattr(source, "context", None)  


def _ext_key(path: str) -> str:
    """Return lowercase file extension from a path-like string."""
    try:
        return Path(path).suffix.lower()
    except Exception:
        return ""


def _build_file_processing_config(cfg: SievioConfig, runtime: PipelineRuntime) -> FileProcessingConfig:
    """Create per-file configuration for worker processes.

    Args:
        cfg (SievioConfig): Base configuration.
        runtime (PipelineRuntime): Runtime-specific overrides.

    Returns:
        FileProcessingConfig: Sanitized configuration for workers.
    """
    pipeline_cfg = replace(cfg.pipeline, bytes_handlers=runtime.bytes_handlers)
    return FileProcessingConfig(
        decode=cfg.decode,
        chunk=cfg.chunk,
        pipeline=pipeline_cfg,
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class PipelineEngine:
    """Execute a prepared PipelinePlan against configured sources and sinks.

    Record middlewares implement process(record) -> Optional[Record]; file
    middlewares implement process(item, records) -> Optional[Iterable[Record]].
    Simple callables are wrapped automatically.
    """
    def __init__(self, plan: PipelinePlan) -> None:
        """Initialize an engine with the provided plan.

        Args:
            plan (PipelinePlan): Plan containing sources, sinks, and runtime
                knobs.
        """
        self.plan = plan
        self.config = plan.spec
        self.stats = PipelineStats()
        self.log = get_logger(__name__)
        self._hooks: Tuple[RunLifecycleHook, ...] = tuple(getattr(plan.runtime, "lifecycle_hooks", ()))
        self.before_record_hooks: List[Callable[[Record], Record]] = []
        self.after_record_hooks: List[Callable[[Record], Record]] = []
        self.record_filter_hooks: List[Callable[[Record], bool]] = []
        self.record_middlewares: List[RecordMiddleware] = []
        self.file_middlewares: List[FileMiddleware] = []
        self.before_source_hooks: List[Callable[[Source], None]] = []
        self.after_source_hooks: List[Callable[[Source], None]] = []
        self._middlewares_normalized = False
        rt = getattr(plan, "runtime", None)
        if rt and getattr(rt, "record_middlewares", None):
            for mw in rt.record_middlewares:
                self.add_record_middleware(mw)

    def add_record_middleware(self, middleware: RecordMiddleware | Callable[[Record], Any]) -> None:
        """Register a record middleware or bare callable."""
        self.record_middlewares.append(_coerce_record_middleware(middleware))
        self._middlewares_normalized = False

    def add_file_middleware(
        self,
        middleware: FileMiddleware | Callable[[Any, Iterable[Record]], Any],
    ) -> None:
        """Register a file middleware or bare callable."""
        self.file_middlewares.append(_coerce_file_middleware(middleware))
        self._middlewares_normalized = False

    def _normalize_middlewares(self) -> None:
        """Ensure middleware lists contain adapter-wrapped instances."""
        if self._middlewares_normalized:
            return

        self.record_middlewares = [_coerce_record_middleware(mw) for mw in self.record_middlewares]
        self.file_middlewares = [_coerce_file_middleware(mw) for mw in self.file_middlewares]

        self._middlewares_normalized = True

    def _apply_middlewares(self, record: Record) -> Optional[Record]:
        """Run a record through all registered record middlewares."""
        current = record
        for middleware in self.record_middlewares:
            try:
                process = getattr(middleware, "process", None)
                callable_mw = process if callable(process) else middleware  # type: ignore[assignment]
                current = callable_mw(current)
            except Exception as exc:  # noqa: BLE001
                self.log.warning(
                    "record middleware %s failed: %s",
                    getattr(middleware, "__name__", middleware.__class__.__name__),
                    exc,
                )
                return None
            if current is None:
                return None
        return current

    def _apply_file_middlewares(self, item: Any, records: Iterable[Record]) -> Optional[Iterable[Record]]:
        """Run a file's records through registered file middlewares."""
        current = records
        for middleware in self.file_middlewares:
            try:
                current = middleware.process(item, current)
            except Exception as exc:  # noqa: BLE001
                self.log.warning(
                    "file middleware %s failed for %s: %s",
                    getattr(middleware, "__name__", middleware.__class__.__name__),
                    getattr(item, "path", "<unknown>"),
                    exc,
                )
                return None
            if current is None:
                return None
        return current

    def _increment_file_stats(self, item: Any) -> None:
        """Update stats counters for a processed file item."""
        stats = self.stats
        size = getattr(item, "size", None)
        if size is None:
            data = getattr(item, "data", b"")
            if isinstance(data, (bytes, bytearray)):
                size = len(data)
            else:
                size = 0
        stats.files += 1
        stats.bytes += int(size or 0)
        ext = _ext_key(getattr(item, "path", ""))
        stats.by_ext[ext] = stats.by_ext.get(ext, 0) + 1

    def _make_processor(self, *, materialize: bool) -> _ProcessFileCallable:
        """Build a callable that extracts records for each work item.

        Args:
            materialize (bool): Whether to materialize iterators in-process
                workers for pickling.

        Returns:
            _ProcessFileCallable: Callable that yields extracted records.
        """
        cfg = self.config
        extractor = self.plan.runtime.file_extractor or cfg.pipeline.file_extractor
        if extractor is None:
            extractor = DefaultExtractor()
        file_cfg = _build_file_processing_config(cfg, self.plan.runtime)
        return _ProcessFileCallable(
            config=file_cfg,
            file_extractor=extractor,
            materialize=materialize,
        )

    def _build_executor(self) -> Tuple[Executor, bool]:
        """Resolve executor configuration and fail-fast behavior.

        Returns:
            tuple[Executor, bool]: Executor instance and fail-fast flag.
        """
        exec_cfg = getattr(self.plan.runtime, "executor_config", None)
        fail_fast = getattr(self.plan.runtime, "fail_fast", False)
        if exec_cfg is None:
            exec_cfg, fail_fast = resolve_pipeline_executor_config(self.config, runtime=self.plan.runtime)
        return Executor(exec_cfg), fail_fast

    def _write_records(
        self,
        item: Any,
        recs: Iterable[Record],
        *,
        sinks: Sequence[Sink],
    ) -> None:
        """Apply hooks, filter, and write records to sinks.

        Args:
            item (Any): Source item associated with the records.
            recs (Iterable[Record]): Records to process.
            sinks (Sequence[Sink]): Destinations to receive records.
        """
        stats = self.stats

        for record in recs:
            for hook in self.before_record_hooks:
                try:
                    record = hook(record)
                except Exception as exc:
                    self.log.warning(
                        "before_record hook %s failed: %s",
                        getattr(hook, "__name__", hook),
                        exc,
                    )

            keep = True
            for check in self.record_filter_hooks:
                try:
                    if not check(record):
                        keep = False
                        break
                except Exception as exc:
                    self.log.warning(
                        "record_filter hook %s failed: %s",
                        getattr(check, "__name__", check),
                        exc,
                    )
                    keep = False
                    break
            if not keep:
                continue

            record = self._apply_middlewares(record)
            if record is None:
                continue

            for hook in self.after_record_hooks:
                try:
                    record = hook(record)
                except Exception as exc:
                    self.log.warning(
                        "after_record hook %s failed: %s",
                        getattr(hook, "__name__", hook),
                        exc,
                    )

            rec = record
            for hook in self._hooks:
                if rec is None:
                    break
                try:
                    rec = hook.on_record(rec)
                except Exception as exc:
                    self.log.warning(
                        "lifecycle hook %s failed on record: %s",
                        getattr(hook, "__class__", type(hook)).__name__,
                        exc,
                    )
                    rec = None
                    break
            if rec is None:
                continue
            record = rec

            wrote_any = False
            for sink in sinks:
                try:
                    sink.write(record)  # type: ignore[attr-defined]
                    wrote_any = True
                except Exception as exc:
                    self.log.warning(
                        "Sink %s failed to write record for %s: %s",
                        getattr(sink, "__class__", type(sink)).__name__,
                        getattr(item, "path", "<unknown>"),
                        exc,
                    )
                    stats.sink_errors += 1

            if wrote_any:
                stats.records += 1

    def _iter_source_items(self, sources: Sequence[Source]) -> Iterable[_WorkItem]:
        """Iterate work items from sources while honoring hooks.

        Args:
            sources (Sequence[Source]): Sources to enumerate.

        Returns:
            Iterable[_WorkItem]: Generator yielding items with context.
        """
        cfg = self.config
        stats = self.stats
        default_ctx = cfg.sinks.context
        log = self.log

        def _gen() -> Iterable[_WorkItem]:
            for source in sources:
                for hook in self.before_source_hooks:
                    try:
                        hook(source)
                    except Exception as exc:
                        log.warning(
                            "before_source hook %s failed for %s: %s",
                            getattr(hook, "__name__", hook),
                            getattr(source, "__class__", type(source)).__name__,
                            exc,
                        )
                ctx = getattr(source, "context", default_ctx)
                try:
                    for item in source.iter_files():
                        yield _WorkItem(item=item, ctx=ctx)
                except Exception as exc:
                    log.warning(
                        "Source %s failed while iterating files: %s",
                        getattr(source, "__class__", type(source)).__name__,
                        exc,
                    )
                    stats.source_errors += 1
                    if cfg.pipeline.fail_fast:
                        raise
                finally:
                    for hook in self.after_source_hooks:
                        try:
                            hook(source)
                        except Exception as exc:
                            log.warning(
                                "after_source hook %s failed for %s: %s",
                                getattr(hook, "__name__", hook),
                                getattr(source, "__class__", type(source)).__name__,
                                exc,
                            )

        return _gen()

    def _process_serial(
        self,
        work: _WorkItem,
        *,
        processor: _ProcessFileCallable,
        sinks: Sequence[Sink],
        fail_fast: bool,
    ) -> None:
        """Process a single work item serially.

        Args:
            work (_WorkItem): Work item to handle.
            processor (_ProcessFileCallable): Extractor callable.
            sinks (Sequence[Sink]): Destinations for records.
            fail_fast (bool): Whether to propagate errors immediately.
        """
        try:
            _, recs = processor(work)
            recs = self._apply_file_middlewares(work.item, recs)
            if recs is not None:
                self._write_records(work.item, recs, sinks=sinks)
            # Count only after successful processing (even if recs is None)
            self._increment_file_stats(work.item)
        except Exception as exc:
            self.log.warning(
                "Processing failed for %s: %s",
                getattr(work.item, "path", "<unknown>"),
                exc,
            )
            self.stats.source_errors += 1
            if fail_fast:
                raise

    def _process_parallel(
        self,
        items: Iterable[_WorkItem],
        *,
        processor: _ProcessFileCallable,
        sinks: Sequence[Sink],
        executor: Executor,
        fail_fast: bool,
    ) -> None:
        """Process work items in parallel using the configured executor.

        Args:
            items (Iterable[_WorkItem]): Stream of work items.
            processor (_ProcessFileCallable): Extractor callable.
            sinks (Sequence[Sink]): Destinations for records.
            executor (Executor): Executor to schedule processing.
            fail_fast (bool): Whether worker errors halt processing.
        """
        stats = self.stats
        log = self.log
        exec_cfg = executor.cfg

        def _log_pickling_hint(exc: BaseException) -> None:
            if exec_cfg.kind != "process":
                return
            if isinstance(exc, (pickle.PicklingError, TypeError)):
                log.warning(
                    "Process executor failed to serialize worker arguments; "
                    "set pipeline.executor_kind='thread' or ensure extractors/config are picklable."
                )

        def _items_with_stats() -> Iterable[_WorkItem]:
            yield from items

        def _process_one(work: _WorkItem) -> Tuple[Any, Iterable[Record]]:
            return processor(work)

        def _on_worker_error(exc: BaseException) -> None:
            log.warning("Worker failed: %s", exc)
            _log_pickling_hint(exc)
            stats.source_errors += 1

        def _on_result(result: Tuple[Any, Iterable[Record]]) -> None:
            item, recs = result
            recs = self._apply_file_middlewares(item, recs)
            if recs is not None:
                self._write_records(item, recs, sinks=sinks)
            # Count only after successful processing/writes (even if recs is None)
            self._increment_file_stats(item)

        def _on_submit_error(work: _WorkItem, exc: BaseException) -> None:
            log.warning(
                "Failed to submit %s to executor: %s",
                getattr(work.item, "path", "<unknown>"),
                exc,
            )
            _log_pickling_hint(exc)
            stats.source_errors += 1
            self._increment_file_stats(work.item)

        try:
            executor.map_unordered(
                _items_with_stats(),
                _process_one,
                _on_result,
                fail_fast=fail_fast,
                on_error=_on_worker_error,
                on_submit_error=_on_submit_error,
            )
        except Exception as exc:
            _log_pickling_hint(exc)
            stats.source_errors += 1
            if fail_fast:
                raise
            log.warning("Parallel processing aborted: %s", exc)

    def _log_qc_summary(self) -> None:
        """Emit QC summary metrics after pipeline completion."""
        cfg = self.config
        tracker = self.stats.qc
        if not tracker.enabled:
            return
        min_score_str = (
            f"{tracker.min_score:.1f}" if tracker.min_score is not None else "off"
        )
        self.log.info(
            "QC summary (min_score=%s, drop_near_dups=%s)\n"
            "  scored: %d\n"
            "  kept: %d\n"
            "  dropped_low_score: %d\n"
            "  dropped_near_dup: %d\n"
            "  candidates_low_score: %d\n"
            "  candidates_near_dup: %d\n"
            "  errors: %d",
            min_score_str,
            "on" if tracker.drop_near_dups else "off",
            tracker.scored,
            tracker.kept,
            tracker.dropped_low_score,
            tracker.dropped_near_dup,
            tracker.candidates_low_score,
            tracker.candidates_near_dup,
            tracker.errors,
        )
        top = tracker.top_dup_families()
        if top:
            lines = [
                f"    - {entry['dup_family_id']}: count={entry['count']} examples={entry.get('examples', [])}"
                for entry in top
            ]
            self.log.info("Largest duplicate families:\n%s", "\n".join(lines))
        else:
            self.log.info("Largest duplicate families: none")

    def run(self) -> PipelineStats:
        """Execute the pipeline plan and return collected statistics."""
        cfg = self.config
        self.stats = PipelineStats()
        stats = self.stats

        jsonl_path = cfg.sinks.primary_jsonl_name or cfg.metadata.primary_jsonl
        stats.primary_jsonl_path = jsonl_path
        self._normalize_middlewares()

        start_ctx = RunContext(cfg=cfg, stats=stats, runtime=self.plan.runtime)
        for hook in self._hooks:
            try:
                hook.on_run_start(start_ctx)
            except Exception as exc:  # noqa: BLE001
                self.log.warning(
                    "lifecycle hook %s failed on run start: %s",
                    getattr(hook, "__class__", type(hook)).__name__,
                    exc,
                )

        try:
            with ExitStack() as stack:
                open_sources: List[Source] = [
                    _open_source_with_stack(stack, src) for src in self.plan.runtime.sources
                ]
                initial_ctx: Optional[RepoContext] = cfg.sinks.context
                open_sinks: List[Sink] = _prepare_sinks(stack, self.plan.runtime.sinks, initial_ctx, stats)
                if not open_sinks:
                    self.log.warning("No sinks are open; processed records will be dropped.")

                executor, fail_fast = self._build_executor()
                materialize_results = executor.cfg.kind == "process"
                processor = self._make_processor(materialize=materialize_results)
                source_items = self._iter_source_items(open_sources)

                requested_kind = (cfg.pipeline.executor_kind or "auto").strip().lower()
                resolved_kind = executor.cfg.kind
                self.log.debug(
                    "Executor kind requested=%s resolved=%s max_workers=%d window=%d",
                    requested_kind,
                    resolved_kind,
                    executor.cfg.max_workers,
                    executor.cfg.window,
                )

                if executor.cfg.max_workers <= 1:
                    for work in source_items:
                        self._process_serial(
                            work,
                            processor=processor,
                            sinks=open_sinks,
                            fail_fast=fail_fast,
                        )
                else:
                    self._process_parallel(
                        source_items,
                        processor=processor,
                        sinks=open_sinks,
                        executor=executor,
                        fail_fast=fail_fast,
                    )
        finally:
            end_ctx = RunContext(cfg=cfg, stats=stats, runtime=self.plan.runtime)
            for hook in self._hooks:
                try:
                    hook.on_run_end(end_ctx)
                except Exception as exc:  # noqa: BLE001
                    self.log.warning(
                        "lifecycle hook %s failed on run end: %s",
                        getattr(hook, "__class__", type(hook)).__name__,
                        exc,
                    )

        self._log_qc_summary()
        return stats

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_pipeline(*, config: SievioConfig, overrides: PipelineOverrides | None = None) -> Dict[str, int]:
    """Run the end-to-end pipeline described by config.

    Args:
        config (SievioConfig): Pipeline configuration object.
        overrides (PipelineOverrides | None): Optional runtime overrides
            merged into the plan and engine.

    Returns:
        dict[str, int]: Statistics from execution as primitive values.
    """
    engine = build_engine(config, overrides=overrides)
    stats_obj = engine.run()
    return stats_obj.as_dict()

__all__ = ["run_pipeline", "PipelineStats", "PipelineEngine", "process_items_parallel"]
