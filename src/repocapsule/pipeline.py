# pipeline.py
# SPDX-License-Identifier: MIT

from __future__ import annotations

from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, FIRST_COMPLETED, wait, Future
from dataclasses import dataclass, field
from typing import Optional, Iterable, Sequence, Dict, List, Tuple, Any, Callable, TypeVar
from pathlib import Path
import os

from .config import QCMode, RepocapsuleConfig
from .interfaces import Source, Sink, RepoContext, Record
from .convert import DefaultExtractor
from .log import get_logger
from .qc_controller import InlineQCController, QCSummaryTracker


log = get_logger(__name__)
T = TypeVar("T")


@dataclass(frozen=True)
class _WorkItem:
    item: Any
    ctx: Optional[RepoContext]


@dataclass
class _ProcessFileCallable:
    config: RepocapsuleConfig
    materialize: bool = False

    def __call__(self, work: _WorkItem) -> Tuple[Any, Iterable[Record]]:
        item = work.item
        ctx = work.ctx
        rel = getattr(item, "path", None) or getattr(item, "rel_path", None)
        if rel is None:
            raise ValueError("FileItem missing 'path'")
        extractor = getattr(self.config.pipeline, "file_extractor", None)
        if extractor is None:
            extractor = DefaultExtractor()
            self.config.pipeline.file_extractor = extractor
        recs_iter = extractor.extract(
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

@dataclass(slots=True)
class PipelineStats:
    files: int = 0
    bytes: int = 0
    records: int = 0
    sink_errors: int = 0
    source_errors: int = 0
    by_ext: Dict[str, int] = field(default_factory=dict)
    qc: QCSummaryTracker = field(default_factory=QCSummaryTracker)

    def as_dict(self) -> Dict[str, object]:
        # Keep a simple, stable shape for external reporting/JSONL footers.
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
        return self.qc.top_dup_families()



def process_items_parallel(
    items: Iterable[T],
    process_one: Callable[[T], Tuple[Any, Iterable[Record]]],
    write_records: Callable[[Any, Iterable[Record]], None],
    *,
    max_workers: int,
    window: int,
    fail_fast: bool,
    on_submit_error: Optional[Callable[[T, BaseException], None]] = None,
    on_worker_error: Optional[Callable[[BaseException], None]] = None,
    executor_kind: str = "thread",
    initializer: Optional[Callable[[], Any]] = None,
    initargs: tuple[Any, ...] = (),
) -> None:
    """
    Execute work items in a bounded ThreadPoolExecutor and stream results.

    Parameters
    ----------
    items:
        Iterable of work items. Items are consumed lazily, so callers can feed
        an unbounded generator (e.g., streaming file sources).
    process_one:
        Callable executed in worker threads. Must return (item, iterable).
    write_records:
        Callable invoked in the main thread with each completed result.
    max_workers:
        ThreadPoolExecutor worker count (must be >= 1).
    window:
        Bounded in-flight submission count. Acts as backpressure so sources do
        not outrun processing. Automatically clamped to >= max_workers.
    fail_fast:
        If True, re-raise errors from submission/worker execution immediately.
    on_submit_error / on_worker_error:
        Optional callbacks for bookkeeping/logging when failures occur.
    executor_kind:
        "thread" (default) or "process". Process executors require picklable work
        items and callables.
    initializer / initargs:
        Optional initializer invoked once per process when using a process executor.
        Ignored for thread executors. The initializer must be picklable and any
        required state should be stored in module-level globals.
    """
    if max_workers < 1:
        raise ValueError("process_items_parallel requires max_workers >= 1")
    window = max(window, max_workers)
    kind = (executor_kind or "thread").strip().lower()
    initargs = tuple(initargs or ())
    extra_kwargs: Dict[str, Any] = {}
    if kind == "process":
        executor_cls = ProcessPoolExecutor
        if initializer is not None:
            extra_kwargs["initializer"] = initializer
            extra_kwargs["initargs"] = initargs
    else:
        executor_cls = ThreadPoolExecutor
        kind = "thread"
        if initializer is not None:
            log.debug("Initializer ignored for thread executors in process_items_parallel.")
    log.debug("process_items_parallel using %s executor (max_workers=%d)", kind, max_workers)
    with executor_cls(max_workers=max_workers, **extra_kwargs) as pool:
        pending: List[Future[Tuple[Any, Iterable[Record]]]] = []

        def _drain(block: bool = False) -> None:
            nonlocal pending
            if not pending:
                return
            done, still = wait(
                pending,
                timeout=None if block else 0.0,
                return_when=FIRST_COMPLETED,
            )
            pending = list(still)
            for fut in done:
                try:
                    item, recs = fut.result()
                except Exception as exc:
                    if on_worker_error:
                        on_worker_error(exc)
                    if fail_fast:
                        raise
                    continue
                write_records(item, recs)

        for work in items:
            try:
                pending.append(pool.submit(process_one, work))
            except Exception as exc:
                if on_submit_error:
                    on_submit_error(work, exc)
                if fail_fast:
                    raise
                continue
            if len(pending) >= window:
                _drain(block=True)

        while pending:
            _drain(block=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _open_source_with_stack(stack: ExitStack, src: Source) -> Source:
    """Enter context manager if supported; else register close() if present."""
    enter = getattr(src, "__enter__", None)
    if callable(enter):
        return stack.enter_context(src)  
    close = getattr(src, "close", None)
    if callable(close):
        stack.callback(close)  
    return src

def _prepare_sinks(stack: ExitStack, sinks: Sequence[Sink], ctx: Optional[RepoContext]) -> List[Sink]:
    """
    Open sinks exactly once with RepoContext if supported.
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
    return open_sinks


def _get_context_from_source(source: Source) -> Optional[RepoContext]:
    return getattr(source, "context", None)  


def _ext_key(path: str) -> str:
    try:
        return Path(path).suffix.lower()
    except Exception:
        return ""


def _resolve_pipeline_concurrency(cfg: RepocapsuleConfig) -> Tuple[int, int, str, bool]:
    """
    Normalize pipeline concurrency knobs (workers/window/executor-kind/fail-fast).
    """
    pc = cfg.pipeline
    max_workers = pc.max_workers or (os.cpu_count() or 1)
    max_workers = max(1, max_workers)
    window = pc.submit_window or (max_workers * 4)
    kind = (pc.executor_kind or "thread").strip().lower()
    if kind not in {"thread", "process"}:
        kind = "thread"
    fail_fast = bool(pc.fail_fast)
    return max_workers, window, kind, fail_fast


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class PipelineEngine:
    def __init__(self, config: RepocapsuleConfig) -> None:
        self.config = config
        self.stats = PipelineStats()
        self.qc_controller: Optional[InlineQCController] = None
        self.log = get_logger(__name__)
        self.before_record_hooks: List[Callable[[Record], Record]] = []
        self.after_record_hooks: List[Callable[[Record], Record]] = []
        self.record_filter_hooks: List[Callable[[Record], bool]] = []
        self.before_source_hooks: List[Callable[[Source], None]] = []
        self.after_source_hooks: List[Callable[[Source], None]] = []

    def _prepare_qc(self) -> None:
        cfg = self.config
        stats = self.stats
        qc_cfg = cfg.qc

        tracker = stats.qc
        tracker.enabled = bool(qc_cfg.enabled)
        tracker.mode = qc_cfg.mode
        tracker.min_score = qc_cfg.min_score
        tracker.drop_near_dups = bool(qc_cfg.drop_near_dups)

        advisory_mode = qc_cfg.mode == QCMode.ADVISORY
        qc_active = bool(
            qc_cfg.enabled
            and qc_cfg.mode == QCMode.INLINE
            and getattr(qc_cfg, "scorer", None)
        )

        if qc_cfg.enabled and qc_cfg.mode == QCMode.INLINE and not qc_active:
            self.log.warning("QC enabled but no scorer is configured; skipping inline annotations.")

        self.qc_controller = None
        if qc_active:
            self.qc_controller = InlineQCController(
                config=qc_cfg,
                stats=stats,
                scorer=qc_cfg.scorer,  # type: ignore[arg-type]
                logger=self.log,
                enforce_drops=not advisory_mode,
            )

    def _increment_file_stats(self, item: Any) -> None:
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
        return _ProcessFileCallable(config=self.config, materialize=materialize)

    def _write_records(
        self,
        item: Any,
        recs: Iterable[Record],
        *,
        sinks: Sequence[Sink],
    ) -> None:
        qc_controller = self.qc_controller
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

            filtered_out = False
            for check in self.record_filter_hooks:
                try:
                    if not check(record):
                        filtered_out = True
                        break
                except Exception as exc:
                    self.log.warning(
                        "record_filter hook %s failed: %s",
                        getattr(check, "__name__", check),
                        exc,
                    )
                    filtered_out = True
                    break
            if filtered_out:
                continue

            if qc_controller and not qc_controller.should_keep(record):
                continue

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

            for hook in self.after_record_hooks:
                try:
                    record = hook(record)
                except Exception as exc:
                    self.log.warning(
                        "after_record hook %s failed: %s",
                        getattr(hook, "__name__", hook),
                        exc,
                    )

            if wrote_any:
                stats.records += 1

    def _iter_source_items(self, sources: Sequence[Source]) -> Iterable[_WorkItem]:
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
        try:
            self._increment_file_stats(work.item)
            _, recs = processor(work)
            self._write_records(work.item, recs, sinks=sinks)
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
        window: int,
        max_workers: int,
        executor_kind: str,
        fail_fast: bool,
    ) -> None:
        cfg = self.config
        stats = self.stats
        log = self.log

        def _items_with_stats() -> Iterable[_WorkItem]:
            for work in items:
                self._increment_file_stats(work.item)
                yield work

        def _write_records(item: Any, recs: Iterable[Record]) -> None:
            self._write_records(item, recs, sinks=sinks)

        def _on_submit_error(work: _WorkItem, exc: BaseException) -> None:
            log.warning("Scheduling failed for %s: %s", getattr(work.item, "path", "<unknown>"), exc)
            stats.source_errors += 1

        def _on_worker_error(exc: BaseException) -> None:
            log.warning("Worker failed: %s", exc)
            stats.source_errors += 1

        process_items_parallel(
            _items_with_stats(),
            processor,
            _write_records,
            max_workers=max_workers,
            window=window,
            fail_fast=fail_fast,
            on_submit_error=_on_submit_error,
            on_worker_error=_on_worker_error,
            executor_kind=executor_kind,
        )

    def _log_qc_summary(self) -> None:
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
        cfg = self.config
        self.stats = PipelineStats()
        stats = self.stats
        self._prepare_qc()

        with ExitStack() as stack:
            open_sources: List[Source] = [
                _open_source_with_stack(stack, src) for src in cfg.sources.sources
            ]
            initial_ctx: Optional[RepoContext] = cfg.sinks.context
            open_sinks: List[Sink] = _prepare_sinks(stack, cfg.sinks.sinks, initial_ctx)
            if not open_sinks:
                self.log.warning("No sinks are open; processed records will be dropped.")

            materialize_results = (cfg.pipeline.executor_kind or "thread").strip().lower() == "process"
            processor = self._make_processor(materialize=materialize_results)
            source_items = self._iter_source_items(open_sources)

            max_workers, window, executor_kind, fail_fast = _resolve_pipeline_concurrency(cfg)

            if max_workers <= 1:
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
                    window=window,
                    max_workers=max_workers,
                    executor_kind=executor_kind,
                    fail_fast=fail_fast,
                )

        self._log_qc_summary()
        return stats

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_pipeline(*, config: RepocapsuleConfig) -> Dict[str, int]:
    """Run the end-to-end pipeline described by ``config``."""
    engine = PipelineEngine(config=config)
    stats = engine.run()
    return stats.as_dict()

__all__ = ["run_pipeline", "PipelineStats", "PipelineEngine"]
