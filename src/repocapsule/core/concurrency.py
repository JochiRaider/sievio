# concurrency.py
# SPDX-License-Identifier: MIT
"""Concurrency helpers and executor configuration for RepoCapsule.

Wraps thread and process pool executors with a bounded submission
window and provides helpers to infer executor settings for the main
ingestion pipeline and quality-control scoring.
"""
from __future__ import annotations

from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Literal, Tuple, TypeVar, Optional
import os

from .config import RepocapsuleConfig
from .log import get_logger

log = get_logger(__name__)

T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class ExecutorConfig:
    """Immutable executor settings used to construct worker pools.


    Attributes:
        max_workers (int): Maximum number of worker threads or
            processes.
        window (int): Maximum number of in-flight tasks allowed
            before backpressure is applied.
        kind (Literal["thread", "process"]): Executor implementation
            to use.
    """
    max_workers: int
    window: int
    kind: Literal["thread", "process"]


class Executor:
    """Run tasks in a thread or process pool with bounded submission.


    This wrapper keeps at most ``cfg.window`` tasks in flight and
    delivers results to callbacks in completion order, not submission
    order.

    Attributes:
        cfg (ExecutorConfig): Executor configuration for this
            instance.
        initializer (Callable | None): Optional initializer called in
            each worker process when using a process pool.
        initargs (tuple[Any, ...]): Positional arguments passed to
            the initializer.
    """

    def __init__(
        self,
        cfg: ExecutorConfig,
        *,
        initializer: Callable[..., Any] | None = None,
        initargs: tuple[Any, ...] = (),
    ) -> None:
        self.cfg = cfg
        self.initializer = initializer
        self.initargs = tuple(initargs or ())

    def _make_executor(self):
        if self.cfg.max_workers < 1:
            raise ValueError("Executor requires max_workers >= 1")
        kind = self.cfg.kind
        init_kwargs: Dict[str, Any] = {}
        if kind == "process":
            if self.initializer is not None:
                init_kwargs["initializer"] = self.initializer
                init_kwargs["initargs"] = self.initargs
            executor_cls = ProcessPoolExecutor
        else:
            executor_cls = ThreadPoolExecutor
            if self.initializer is not None:
                log.debug("Executor initializer ignored for thread executor.")
        return executor_cls(max_workers=self.cfg.max_workers, **init_kwargs)

    def map_unordered(
        self,
        items: Iterable[T],
        fn: Callable[[T], R],
        on_result: Callable[[R], None],
        *,
        fail_fast: bool = False,
        on_error: Callable[[BaseException], None] | None = None,
        on_submit_error: Callable[[T, BaseException], None] | None = None,
    ) -> None:
        """Submit items to workers and consume results as they complete.

        Items are submitted up to the configured submission window.
        Completed results are passed to ``on_result`` in completion
        order. Errors can be reported via callbacks or cause early
        termination when ``fail_fast`` is True.

        Args:
            items (Iterable[T]): Items to process.
            fn (Callable[[T], R]): Worker function invoked for each
                item.
            on_result (Callable[[R], None]): Callback invoked for each
                successful result.
            fail_fast (bool): Whether to re-raise the first worker or
                submission error and abort further processing.
            on_error (Callable[[BaseException], None] | None): Optional
                callback invoked when a worker raises an exception.
            on_submit_error (Callable[[T, BaseException], None] | None):
                Optional callback invoked when submitting a task to the
                executor fails.

        Raises:
            Exception: Propagates the first worker or submission error
                when ``fail_fast`` is True.
        """

        window = max(self.cfg.window, self.cfg.max_workers)
        with self._make_executor() as pool:
            pending: list[Future[R]] = []

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
                        result = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        if on_error:
                            on_error(exc)
                        if fail_fast:
                            raise
                        continue
                    on_result(result)

            for item in items:
                try:
                    fut = pool.submit(fn, item)
                    pending.append(fut)
                except Exception as exc:  # noqa: BLE001
                    if on_submit_error:
                        on_submit_error(item, exc)
                    if fail_fast:
                        raise
                    continue
                if len(pending) >= window:
                    _drain(block=True)

            while pending:
                _drain(block=True)


def process_items_parallel(
    items: Iterable[T],
    process_one: Callable[[T], Tuple[Any, Iterable[Any]]],
    write_records: Callable[[Any, Iterable[Any]], None],
    *,
    max_workers: int,
    window: int,
    fail_fast: bool,
    on_submit_error: Callable[[T, BaseException], None] | None = None,
    on_worker_error: Callable[[BaseException], None] | None = None,
    executor_kind: str = "thread",
    initializer: Callable[..., Any] | None = None,
    initargs: tuple[Any, ...] = (),
) -> None:
    """Process items in parallel and stream resulting records.


    This is a convenience wrapper around :class:`Executor` that
    applies ``process_one`` to each input item and then passes the
    original item and its records to ``write_records``.

    Args:
        items (Iterable[T]): Items to process.
        process_one (Callable[[T], tuple[Any, Iterable[Any]]]): Function
            that transforms a single item into a result and an iterable
            of records.
        write_records (Callable[[Any, Iterable[Any]], None]): Callback
            responsible for writing or emitting records for each item.
        max_workers (int): Maximum number of worker threads or
            processes.
        window (int): Maximum number of in-flight tasks allowed before
            blocking submissions.
        fail_fast (bool): Whether to re-raise the first worker or
            submission error and abort further processing.
        on_submit_error (Callable[[T, BaseException], None] | None):
            Optional callback for errors raised while submitting tasks
            to the executor.
        on_worker_error (Callable[[BaseException], None] | None):
            Optional callback for exceptions raised by worker
            functions.
        executor_kind (str): Pool implementation to use,
            ``"thread"`` or ``"process"``. Any other value is treated
            as ``"thread"``.
        initializer (Callable[..., Any] | None): Optional initializer
            invoked in each worker process when using a process pool.
        initargs (tuple[Any, ...]): Positional arguments passed to the
            initializer.

    Raises:
        ValueError: If ``max_workers`` is less than 1.
        Exception: Propagates the first worker or submission error when
            ``fail_fast`` is True.
    """
    if max_workers < 1:
        raise ValueError("process_items_parallel requires max_workers >= 1")
    normalized_kind = (executor_kind or "thread").strip().lower()
    if normalized_kind not in {"thread", "process"}:
        normalized_kind = "thread"
    cfg = ExecutorConfig(
        max_workers=max_workers,
        window=max(window, max_workers),
        kind=normalized_kind,  # type: ignore[arg-type]
    )
    executor = Executor(
        cfg,
        initializer=initializer if normalized_kind == "process" else None,
        initargs=initargs if normalized_kind == "process" else (),
    )

    def _worker(item: T) -> Tuple[Any, Iterable[Any]]:
        return process_one(item)

    def _on_result(result: Tuple[Any, Iterable[Any]]) -> None:
        item, recs = result
        write_records(item, recs)

    def _on_error(exc: BaseException) -> None:
        if on_worker_error:
            on_worker_error(exc)

    executor.map_unordered(
        items,
        _worker,
        _on_result,
        fail_fast=fail_fast,
        on_error=_on_error,
        on_submit_error=on_submit_error,
    )


def _has_heavy_binary_handlers(cfg: RepocapsuleConfig, runtime: Any | None = None) -> bool:
    """Detect whether the pipeline uses heavy binary bytes handlers.


    The heuristic looks for PDF/EVTX-oriented handlers on the pipeline
    configuration or runtime, based on handler names and module
    prefixes.

    Args:
        cfg (RepocapsuleConfig): Top-level configuration object.
        runtime (Any | None): Optional runtime object overriding or
            augmenting the configured bytes handlers.

    Returns:
        bool: True if handlers strongly suggest heavy PDF/EVTX
            processing, False otherwise.
    """
    try:
        handlers = cfg.pipeline.bytes_handlers
    except Exception:
        handlers = ()
    if runtime is not None:
        handlers = getattr(runtime, "bytes_handlers", None) or handlers

    for _sniff, handler in handlers:
        name = getattr(handler, "__name__", "").lower()
        mod = getattr(handler, "__module__", "").lower()
        if name in {"handle_pdf", "handle_evtx"}:
            return True
        if "pdfio" in mod or "evtxio" in mod:
            return True
    return False


def _has_heavy_sources(cfg: RepocapsuleConfig, runtime: Any | None = None) -> bool:
    """Heuristically detect heavy binary sources in the configuration.


    Sources and source runtime objects are inspected for PDF/EVTX-like
    types or filename extensions that imply expensive binary parsing.

    Args:
        cfg (RepocapsuleConfig): Top-level configuration object.
        runtime (Any | None): Optional runtime object providing active
            source instances.

    Returns:
        bool: True if the configured or runtime sources suggest heavy
            PDF/EVTX ingestion, False otherwise.
    """
    try:
        src_cfg = cfg.sources
    except Exception:
        src_cfg = None

    runtime_sources = getattr(runtime, "sources", None)
    if runtime_sources:
        for src in runtime_sources:
            try:
                cls_name = type(src).__name__.lower()
            except Exception:
                continue
            if "pdf" in cls_name or "evtx" in cls_name:
                return True

    if src_cfg:
        try:
            for src in getattr(src_cfg, "sources", ()):
                cls_name = type(src).__name__.lower()
                if "pdf" in cls_name or "evtx" in cls_name:
                    return True
        except Exception:
            pass

    local_cfg = getattr(src_cfg, "local", None)
    if local_cfg and getattr(local_cfg, "include_exts", None):
        exts = {e.lower() for e in local_cfg.include_exts}
        if ".pdf" in exts or ".evtx" in exts:
            return True

    gh_cfg = getattr(src_cfg, "github", None)
    if gh_cfg and getattr(gh_cfg, "include_exts", None):
        exts = {e.lower() for e in gh_cfg.include_exts}
        if ".pdf" in exts or ".evtx" in exts:
            return True

    return False


def _extract_concurrency_hint(obj: Any) -> tuple[Optional[str], bool]:
    """Extract concurrency preferences from an object.


    The object is inspected directly and via an optional
    ``concurrency_profile`` attribute for a preferred executor kind
    and whether the work is CPU intensive.

    Args:
        obj (Any): Object that may expose concurrency hints.

    Returns:
        tuple[Optional[str], bool]: Normalized preferred executor
            (``"thread"`` or ``"process"``) and a flag indicating
            whether the workload is CPU intensive.
    """
    if obj is None:
        return None, False
    preferred = getattr(obj, "preferred_executor", None)
    cpu_intensive = bool(getattr(obj, "cpu_intensive", False))
    profile = getattr(obj, "concurrency_profile", None)
    if profile is not None:
        preferred = getattr(profile, "preferred_executor", preferred)
        cpu_intensive = bool(getattr(profile, "cpu_intensive", cpu_intensive))
    if isinstance(preferred, str):
        val = preferred.strip().lower()
        if val in {"thread", "process"}:
            preferred = val
        else:
            preferred = None
    else:
        preferred = None
    return preferred, cpu_intensive


def _preferred_executor_from_hints(runtime: Any | None) -> Optional[str]:
    """Aggregate explicit concurrency hints from runtime components.


    Sources, bytes handlers, and file extractors attached to the
    runtime are inspected for concurrency hints. CPU-heavy components
    bias the result toward a process executor.

    Args:
        runtime (Any | None): Runtime object holding sources, handlers,
            and extractors.

    Returns:
        Optional[str]: Preferred executor kind (``"thread"`` or
            ``"process"``) if one can be determined, otherwise None.
    """
    if runtime is None:
        return None
    preferred: Optional[str] = None
    cpu_heavy = False

    def _update(pref: Optional[str], cpu: bool) -> None:
        nonlocal preferred, cpu_heavy
        if cpu:
            cpu_heavy = True
        if pref == "process":
            preferred = "process"
        elif pref == "thread" and preferred is None:
            preferred = "thread"

    src_seq = getattr(runtime, "sources", None) or getattr(getattr(runtime, "runtime", None), "sources", None)
    if src_seq:
        for src in src_seq:
            pref, cpu = _extract_concurrency_hint(src)
            _update(pref, cpu)
    handlers = getattr(runtime, "bytes_handlers", None) or getattr(getattr(runtime, "runtime", None), "bytes_handlers", None)
    if handlers:
        for _sniff, handler in handlers:
            pref, cpu = _extract_concurrency_hint(handler)
            _update(pref, cpu)
    extractor = getattr(runtime, "file_extractor", None) or getattr(getattr(runtime, "runtime", None), "file_extractor", None)
    if extractor:
        pref, cpu = _extract_concurrency_hint(extractor)
        _update(pref, cpu)

    if cpu_heavy:
        return "process"
    return preferred


def _infer_executor_kind(cfg: RepocapsuleConfig, runtime: Any | None = None) -> str:
    """Infer an executor kind from configuration and runtime hints.

    Explicit concurrency hints on runtime components take precedence
    over heuristics based on configured sources and bytes handlers.

    Args:
        cfg (RepocapsuleConfig): Top-level configuration object.
        runtime (Any | None): Optional runtime object used to refine
            the decision.

    Returns:
        str: Inferred executor kind, either ``"thread"`` or
            ``"process"``.
    """    
    hinted = _preferred_executor_from_hints(runtime)
    if hinted in {"thread", "process"}:
        return hinted
    heavy_handlers = _has_heavy_binary_handlers(cfg, runtime)
    heavy_sources = _has_heavy_sources(cfg, runtime)
    if heavy_handlers and heavy_sources:
        return "process"
    return "thread"


def infer_executor_kind(cfg: RepocapsuleConfig, *, default: str = "thread", runtime: Any | None = None) -> Literal["thread", "process"]:
    """Return a validated executor kind for the pipeline.


    This wraps :func:`_infer_executor_kind` and falls back to the given
    default when the heuristics cannot determine a valid kind.

    Args:
        cfg (RepocapsuleConfig): Top-level configuration object.
        default (str): Fallback executor kind to use when inference
            fails or returns an unknown value.
        runtime (Any | None): Optional runtime object used to refine
            the decision.

    Returns:
        Literal["thread", "process"]: Normalized executor kind.
    """    
    normalized_default = default if default in {"thread", "process"} else "thread"
    kind = _infer_executor_kind(cfg, runtime=runtime)
    if kind not in {"thread", "process"}:
        return normalized_default  # type: ignore[return-value]
    return kind  # type: ignore[return-value]


def resolve_pipeline_executor_config(cfg: RepocapsuleConfig, runtime: Any | None = None) -> tuple[ExecutorConfig, bool]:
    """Build executor settings for the main ingestion pipeline.

    The pipeline section of the configuration controls the maximum
    worker count, submission window, executor kind, and fail-fast
    behavior. When ``executor_kind`` is ``"auto"``, a suitable kind
    is inferred from the configured sources and bytes handlers.

    Args:
        cfg (RepocapsuleConfig): Top-level configuration object.
        runtime (Any | None): Optional runtime object whose components
            may influence executor inference.

    Returns:
        tuple[ExecutorConfig, bool]: Executor configuration for the
            pipeline and the ``fail_fast`` flag.
    """
    pc = cfg.pipeline
    max_workers = pc.max_workers or (os.cpu_count() or 1)
    max_workers = max(1, max_workers)
    window = pc.submit_window or (max_workers * 4)
    raw_kind = (pc.executor_kind or "auto").strip().lower()
    if raw_kind == "auto":
        kind = _infer_executor_kind(cfg, runtime=runtime)
    else:
        kind = raw_kind
    if kind not in {"thread", "process"}:
        kind = "thread"
    fail_fast = bool(pc.fail_fast)
    exec_cfg = ExecutorConfig(max_workers=max_workers, window=max(window, max_workers), kind=kind)  # type: ignore[arg-type]
    return exec_cfg, fail_fast


def resolve_qc_executor_config(cfg: RepocapsuleConfig, runtime: Any | None = None) -> ExecutorConfig:
    """Build executor settings for post-extraction QC scoring.


    QC executor settings are derived from both the QC and pipeline
    sections of the configuration. When the QC-specific fields are
    unset, pipeline defaults are reused, and ``"auto"`` executor kinds
    are resolved using the same heuristics as the main pipeline.

    Args:
        cfg (RepocapsuleConfig): Top-level configuration object.
        runtime (Any | None): Optional runtime object whose components
            may influence executor inference.

    Returns:
        ExecutorConfig: Executor configuration to use for QC scoring.
    """
    qc = cfg.qc
    pc = cfg.pipeline
    pipeline_max_workers = pc.max_workers or (os.cpu_count() or 1)
    pipeline_max_workers = max(1, pipeline_max_workers)
    max_workers = pipeline_max_workers if qc.post_max_workers is None else qc.post_max_workers
    max_workers = max(1, max_workers)

    if qc.post_submit_window is not None:
        window = qc.post_submit_window
    elif pc.submit_window is not None:
        window = pc.submit_window
    else:
        window = max_workers * 4

    raw_kind = (qc.post_executor_kind or pc.executor_kind or "thread").strip().lower()
    if raw_kind == "auto":
        kind = _infer_executor_kind(cfg, runtime=runtime)
    else:
        kind = raw_kind
    if kind not in {"thread", "process"}:
        kind = "thread"
    return ExecutorConfig(
        max_workers=max_workers,
        window=max(window, max_workers),
        kind=kind,  # type: ignore[arg-type]
    )


__all__ = [
    "Executor",
    "ExecutorConfig",
    "process_items_parallel",
    "infer_executor_kind",
    "resolve_pipeline_executor_config",
    "resolve_qc_executor_config",
]
