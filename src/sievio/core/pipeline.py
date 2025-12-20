# pipeline.py
# SPDX-License-Identifier: MIT
"""Pipeline execution engine coordinating sources, sinks, and middleware."""

from __future__ import annotations

import pickle
from collections.abc import Callable, Iterable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any
from .concurrency import (
    Executor,
    process_items_parallel,
    resolve_pipeline_executor_config,
)
from .config import FileProcessingConfig, SievioConfig
from .convert import (
    _OPEN_STREAM_DEFAULT_MAX_BYTES,
    DefaultExtractor,
    _can_open_stream,
    make_limited_stream,
    maybe_reopenable_local_path,
)
from .interfaces import (
    FileExtractor,
    FileMiddleware,
    Record,
    RecordMiddleware,
    RepoContext,
    RunContext,
    RunLifecycleHook,
    RunSummaryView,
    Sink,
    Source,
)
from .log import get_logger
from .qc_controller import QCSummaryTracker
from .records import best_effort_record_path

log = get_logger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from .builder import PipelineOverrides, PipelinePlan, PipelineRuntime


class MiddlewareError(RuntimeError):
    """Raised when a middleware failure should abort the pipeline."""


class ErrorRateExceeded(RuntimeError):
    """Raised when the pipeline aborts due to an error-rate threshold."""


@dataclass(frozen=True)
class _WorkItem:
    item: Any
    ctx: RepoContext | None


class _FuncRecordMiddleware:
    """Internal adapter to treat bare functions as RecordMiddleware."""

    def __init__(self, fn: Callable[[Record], Record | None]) -> None:
        self._fn = fn
        self.__name__ = getattr(fn, "__name__", fn.__class__.__name__)

    def process(self, record: Record) -> Record | None:
        return self._fn(record)

    def __call__(self, record: Record) -> Record | None:
        return self.process(record)


class _FuncFileMiddleware:
    """Internal adapter to treat bare functions as FileMiddleware."""

    def __init__(self, fn: Callable[[Any, Iterable[Record]], Iterable[Record] | None]) -> None:
        self._fn = fn
        self.__name__ = getattr(fn, "__name__", fn.__class__.__name__)

    def process(self, item: Any, records: Iterable[Record]) -> Iterable[Record] | None:
        return self._fn(item, records)


def apply_overrides_to_engine(
    engine: PipelineEngine,
    overrides: PipelineOverrides | None,
) -> PipelineEngine:
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


def _coerce_file_middleware(
    mw: FileMiddleware | Callable[[Any, Iterable[Record]], Any],
) -> FileMiddleware:
    """Wrap a bare callable as FileMiddleware when needed."""
    return mw if hasattr(mw, "process") else _FuncFileMiddleware(mw)  # type: ignore[arg-type]


@dataclass
class _ProcessFileCallable:
    config: FileProcessingConfig
    file_extractor: FileExtractor
    executor_kind: str = "thread"
    materialize: bool = False

    def __call__(self, work: _WorkItem) -> tuple[Any, Iterable[Record]]:
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
        recs_iter: Iterable[Record]
        extract_stream = getattr(self.file_extractor, "extract_stream", None)
        data = getattr(item, "data", None)
        data_len = len(data) if isinstance(data, (bytes, bytearray)) else None
        size = getattr(item, "size", None)
        should_stream = (
            data is None
            or data_len == 0
            or (size is not None and data_len is not None and data_len < size)
        )
        stream_opener = getattr(item, "open_stream", None) if _can_open_stream(item) else None
        reopenable = None
        if stream_opener is None:
            reopenable = maybe_reopenable_local_path(item)
            if reopenable is not None:
                stream_opener = lambda: reopenable.open("rb")
        can_stream = (
            self.executor_kind == "thread"
            and callable(extract_stream)
            and should_stream
            and callable(stream_opener)
        )
        if can_stream:
            stream = None
            try:
                raw_stream = stream_opener()  # type: ignore[misc]
                limit = self.config.decode.max_bytes_per_file
                if limit is None:
                    limit = _OPEN_STREAM_DEFAULT_MAX_BYTES
                stream = make_limited_stream(raw_stream, limit)
                stream_ref = stream

                def _iter_with_close(iterable: Iterable[Record]) -> Iterable[Record]:
                    try:
                        yield from iterable
                    finally:
                        try:
                            stream_ref.close()
                        except Exception:
                            pass

                out = extract_stream(  # type: ignore[misc]
                    stream=stream,
                    path=str(rel),
                    context=ctx,
                )
                recs_iter = _iter_with_close(out if out is not None else ())
                stream = None
            except Exception:
                if stream:
                    try:
                        stream.close()
                    except Exception:
                        pass
                log.debug(
                    "Streaming extraction failed for %s; falling back to extract()",
                    rel,
                    exc_info=True,
                )
                recs_iter = self.file_extractor.extract(
                    item,
                    config=self.config,
                    context=ctx,
                )
        else:
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
    without unhandled errors (not just files attempted); attempted_files counts each
    item the pipeline tried to process. ``qc`` holds screening (quality + safety)
    stats for the run.
    """

    files: int = 0
    attempted_files: int = 0
    bytes: int = 0
    records: int = 0
    sink_errors: int = 0
    source_errors: int = 0
    middleware_errors: int = 0
    by_ext: dict[str, int] = field(default_factory=dict)
    qc: QCSummaryTracker = field(default_factory=QCSummaryTracker)
    primary_jsonl_path: str | None = None

    def as_dict(self) -> dict[str, object]:
        """Return a stable dict shape for reporting and JSONL footers."""
        data: dict[str, object] = {
            "files": int(self.files),
            "attempted_files": int(self.attempted_files),
            "bytes": int(self.bytes),
            "records": int(self.records),
            "sink_errors": int(self.sink_errors),
            "source_errors": int(self.source_errors),
            "middleware_errors": int(self.middleware_errors),
            "by_ext": dict(self.by_ext),
        }
        data["qc"] = self.qc.as_dict()
        return data

    def qc_top_dup_families(self) -> list[dict[str, Any]]:
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

def _prepare_sinks(
    stack: ExitStack,
    sinks: Sequence[Sink],
    ctx: RepoContext | None,
    stats: PipelineStats,
) -> list[Sink]:
    """Open sinks once, tracking failures in stats.

    Args:
        stack (ExitStack): ExitStack to register closers.
        sinks (Sequence[Sink]): Sinks to open.
        ctx (RepoContext | None): Optional repository context passed to open().
        stats (PipelineStats): Stats container for error tracking.

    Returns:
        list[Sink]: Sinks successfully opened.
    """
    open_sinks: list[Sink] = []
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


def _get_context_from_source(source: Source) -> RepoContext | None:
    """Extract RepoContext from a source when available."""
    return getattr(source, "context", None)  


def _ext_key(path: str) -> str:
    """Return lowercase file extension from a path-like string."""
    try:
        return Path(path).suffix.lower()
    except Exception:
        return ""


def _build_file_processing_config(
    cfg: SievioConfig,
    runtime: PipelineRuntime,
) -> FileProcessingConfig:
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
        self._hooks: tuple[RunLifecycleHook, ...] = tuple(
            getattr(plan.runtime, "lifecycle_hooks", ())
        )
        self.before_record_hooks: list[Callable[[Record], Record]] = []
        self.after_record_hooks: list[Callable[[Record], Record]] = []
        self.record_filter_hooks: list[Callable[[Record], bool]] = []
        self.record_middlewares: list[RecordMiddleware] = []
        self.file_middlewares: list[FileMiddleware] = []
        self.before_source_hooks: list[Callable[[Source], None]] = []
        self.after_source_hooks: list[Callable[[Source], None]] = []
        self._record_chain: list[Callable[[Record], Record | None]] = []
        self._record_chain_dirty = True
        self._middlewares_normalized = False
        rt = getattr(plan, "runtime", None)
        if rt and getattr(rt, "record_middlewares", None):
            for mw in rt.record_middlewares:
                self.add_record_middleware(mw)
        self._fail_on_middleware_error = bool(
            getattr(plan.runtime, "fail_fast", False)
            or getattr(self.config.pipeline, "fail_fast", False)
        )

    def add_record_middleware(self, middleware: RecordMiddleware | Callable[[Record], Any]) -> None:
        """Register a record middleware or bare callable."""
        self.record_middlewares.append(_coerce_record_middleware(middleware))
        self._middlewares_normalized = False
        self._record_chain_dirty = True

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

    def _middleware_chain_step(self, record: Record) -> Record | None:
        """Chain adapter for record middlewares."""
        return self._apply_middlewares(record)

    def _apply_middlewares(self, record: Record) -> Record | None:
        """Run a record through all registered record middlewares."""
        current = record
        for middleware in self.record_middlewares:
            middleware_name = getattr(middleware, "__name__", middleware.__class__.__name__)
            record_label = best_effort_record_path(current)
            try:
                process = getattr(middleware, "process", None)
                callable_mw = process if callable(process) else middleware  # type: ignore[assignment]
                current = callable_mw(current)
            except Exception:  # noqa: BLE001
                self.log.exception(
                    "Record middleware %s failed for %s",
                    middleware_name,
                    record_label,
                )
                if self._fail_on_middleware_error:
                    raise
                return None
            if current is None:
                return None
        return current

    def _wrap_before_record_hook(
        self,
        hook: Callable[[Record], Record],
    ) -> Callable[[Record], Record | None]:
        def _step(record: Record) -> Record | None:
            try:
                return hook(record)
            except Exception as exc:  # noqa: BLE001
                self.log.warning(
                    "before_record hook %s failed: %s",
                    getattr(hook, "__name__", hook),
                    exc,
                )
                return record

        return _step

    def _wrap_after_record_hook(
        self,
        hook: Callable[[Record], Record],
    ) -> Callable[[Record], Record | None]:
        def _step(record: Record) -> Record | None:
            try:
                return hook(record)
            except Exception as exc:  # noqa: BLE001
                self.log.warning(
                    "after_record hook %s failed: %s",
                    getattr(hook, "__name__", hook),
                    exc,
                )
                return record

        return _step

    def _wrap_record_filter(
        self,
        check: Callable[[Record], bool],
    ) -> Callable[[Record], Record | None]:
        def _step(record: Record) -> Record | None:
            try:
                if check(record):
                    return record
            except Exception as exc:  # noqa: BLE001
                self.log.warning(
                    "record_filter hook %s failed: %s",
                    getattr(check, "__name__", check),
                    exc,
                )
                return None
            return None

        return _step

    def _wrap_lifecycle_hooks(self) -> Callable[[Record], Record | None]:
        def _step(record: Record) -> Record | None:
            rec: Record | None = record
            for hook in self._hooks:
                if rec is None:
                    break
                try:
                    rec = hook.on_record(rec)
                except Exception as exc:  # noqa: BLE001
                    self.log.warning(
                        "lifecycle hook %s failed on record: %s",
                        getattr(hook, "__class__", type(hook)).__name__,
                        exc,
                    )
                    rec = None
                    break
            return rec

        return _step

    def _build_record_chain(self) -> None:
        """Construct the unified per-record processing chain."""
        chain: list[Callable[[Record], Record | None]] = []
        for hook in self.before_record_hooks:
            chain.append(self._wrap_before_record_hook(hook))
        for check in self.record_filter_hooks:
            chain.append(self._wrap_record_filter(check))
        if self.record_middlewares:
            chain.append(self._middleware_chain_step)
        for hook in self.after_record_hooks:
            chain.append(self._wrap_after_record_hook(hook))
        if self._hooks:
            chain.append(self._wrap_lifecycle_hooks())
        self._record_chain = chain
        self._record_chain_dirty = False

    def _ensure_record_chain(self) -> None:
        if self._record_chain_dirty:
            self._build_record_chain()

    def _apply_file_middlewares(
        self,
        item: Any,
        records: Iterable[Record],
    ) -> Iterable[Record] | None:
        """Run a file's records through registered file middlewares."""
        current = records
        for middleware in self.file_middlewares:
            middleware_name = getattr(middleware, "__name__", middleware.__class__.__name__)
            item_label = (
                getattr(item, "path", None)
                or getattr(item, "rel_path", None)
                or "<unknown>"
            )
            try:
                item_label = str(item_label)
            except Exception:
                item_label = "<unknown>"
            try:
                current = middleware.process(item, current)
            except Exception as exc:  # noqa: BLE001
                self.stats.middleware_errors += 1
                self.log.exception(
                    "File middleware %s failed for %s",
                    middleware_name,
                    item_label,
                )
                if self._fail_on_middleware_error:
                    raise MiddlewareError(
                        f"File middleware {middleware_name} failed for {item_label}"
                    ) from exc
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

    def _make_processor(self, *, materialize: bool, executor_kind: str) -> _ProcessFileCallable:
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
            executor_kind=executor_kind,
            materialize=materialize,
        )

    def _build_executor(self) -> tuple[Executor, bool]:
        """Resolve executor configuration and fail-fast behavior.

        Returns:
            tuple[Executor, bool]: Executor instance and fail-fast flag.
        """
        exec_cfg = getattr(self.plan.runtime, "executor_config", None)
        fail_fast = getattr(self.plan.runtime, "fail_fast", False)
        if exec_cfg is None:
            exec_cfg, fail_fast = resolve_pipeline_executor_config(
                self.config,
                runtime=self.plan.runtime,
            )
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
        self._ensure_record_chain()
        chain = self._record_chain

        for record in recs:
            current: Record | None = record
            for step in chain:
                if current is None:
                    break
                current = step(current)
            if current is None:
                continue
            record = current
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
                    self._maybe_abort_for_error_rate(exc)
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
        self.stats.attempted_files += 1
        try:
            _, recs = processor(work)
            recs = self._apply_file_middlewares(work.item, recs)
            if recs is not None:
                self._write_records(work.item, recs, sinks=sinks)
            # Count only after successful processing (even if recs is None)
            self._increment_file_stats(work.item)
        except MiddlewareError:
            raise
        except Exception as exc:
            self.log.warning(
                "Processing failed for %s: %s",
                getattr(work.item, "path", "<unknown>"),
                exc,
            )
            self.stats.source_errors += 1
            self._maybe_abort_for_error_rate(exc)
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
            for work in items:
                stats.attempted_files += 1
                yield work

        def _process_one(work: _WorkItem) -> tuple[Any, Iterable[Record]]:
            return processor(work)

        def _on_worker_error(exc: BaseException) -> None:
            log.warning("Worker failed: %s", exc)
            _log_pickling_hint(exc)
            stats.source_errors += 1
            self._maybe_abort_for_error_rate(exc)

        def _on_result(result: tuple[Any, Iterable[Record]]) -> None:
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
            self._maybe_abort_for_error_rate(exc)
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
        except MiddlewareError:
            raise
        except Exception as exc:
            _log_pickling_hint(exc)
            stats.source_errors += 1
            self._maybe_abort_for_error_rate(exc)
            if fail_fast:
                raise
            log.warning("Parallel processing aborted: %s", exc)

    def _maybe_abort_for_error_rate(self, cause: BaseException | None = None) -> None:
        """Abort the pipeline when the source error-rate exceeds the configured max."""
        cfg_threshold = getattr(self.config.pipeline, "max_error_rate", None)
        if cfg_threshold is None:
            return

        attempted = self.stats.attempted_files
        if attempted <= 0:
            return

        rate = self.stats.source_errors / attempted
        if rate > cfg_threshold:
            msg = (
                f"Aborting pipeline: source error rate {rate:.3f} exceeded "
                f"limit {cfg_threshold:.3f} "
                f"(source_errors={self.stats.source_errors}, attempted_files={attempted})"
            )
            raise ErrorRateExceeded(msg) from cause

    def _log_error_summary(self) -> None:
        """Emit aggregated error counters at the end of a run."""
        stats = self.stats
        has_errors = any((stats.source_errors, stats.sink_errors, stats.middleware_errors))
        level = self.log.warning if has_errors else self.log.info
        level(
            "Ingestion summary: attempted=%d succeeded=%d records=%d "
            "source_errors=%d sink_errors=%d middleware_errors=%d",
            stats.attempted_files,
            stats.files,
            stats.records,
            stats.source_errors,
            stats.sink_errors,
            stats.middleware_errors,
        )

    def _log_qc_summary(self) -> None:
        """Emit QC summary metrics after pipeline completion."""
        tracker = self.stats.qc
        if not tracker.enabled:
            return
        quality = tracker.get_screener("quality", create=False)
        if quality is None:
            return
        min_score_str = (
            f"{tracker.min_score:.1f}" if tracker.min_score is not None else "off"
        )
        self.log.info(
            "QC summary (min_score=%s, drop_near_dups=%s)\n"
            "  scored: %d\n"
            "  kept: %d\n"
            "  would_drop_low_score: %d\n"
            "  would_drop_near_dup: %d\n"
            "  candidates_low_score: %d\n"
            "  candidates_near_dup: %d\n"
            "  errors: %d",
            min_score_str,
            "on" if tracker.drop_near_dups else "off",
            quality.scored,
            quality.kept,
            quality.drops.get("low_score", 0),
            quality.drops.get("near_dup", 0),
            quality.candidates.get("low_score", 0),
            quality.candidates.get("near_dup", 0),
            quality.errors,
        )
        top = tracker.top_dup_families()
        if top:
            lines = [
                (
                    f"    - {entry['dup_family_id']}: count={entry['count']} "
                    f"examples={entry.get('examples', [])}"
                )
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
        self._record_chain_dirty = True

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
                open_sources: list[Source] = [
                    _open_source_with_stack(stack, src) for src in self.plan.runtime.sources
                ]
                initial_ctx: RepoContext | None = cfg.sinks.context
                open_sinks: list[Sink] = _prepare_sinks(
                    stack,
                    self.plan.runtime.sinks,
                    initial_ctx,
                    stats,
                )
                if not open_sinks:
                    self.log.warning("No sinks are open; processed records will be dropped.")

                executor, fail_fast = self._build_executor()
                materialize_results = executor.cfg.kind == "process"
                processor = self._make_processor(
                    materialize=materialize_results,
                    executor_kind=executor.cfg.kind,
                )
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

        self._log_error_summary()
        self._log_qc_summary()
        return stats

__all__ = ["PipelineStats", "PipelineEngine", "process_items_parallel"]
