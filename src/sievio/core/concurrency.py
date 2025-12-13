# concurrency.py
# SPDX-License-Identifier: MIT
"""Concurrency helpers and executor configuration for Sievio.

Wraps thread and process pool executors with a bounded submission
window and provides helpers to infer executor settings for the main
ingestion pipeline and quality-control scoring.
"""
from __future__ import annotations

import functools
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

from .config import SievioConfig
from .log import get_logger

log = get_logger(__name__)

T = TypeVar("T")
R = TypeVar("R")


def _call_process_one(
    process_one: Callable[[T], Tuple[Any, Iterable[Any]]],
    item: T,
) -> Tuple[Any, Iterable[Any]]:
    """Top-level helper so process pool workers can pickle the callable."""
    return process_one(item)


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

    worker_fn: Callable[[T], Tuple[Any, Iterable[Any]]]
    if normalized_kind == "process":
        worker_fn = functools.partial(_call_process_one, process_one)
    else:
        worker_fn = _worker

    def _on_result(result: Tuple[Any, Iterable[Any]]) -> None:
        item, recs = result
        write_records(item, recs)

    def _on_error(exc: BaseException) -> None:
        if on_worker_error:
            on_worker_error(exc)

    executor.map_unordered(
        items,
        worker_fn,
        _on_result,
        fail_fast=fail_fast,
        on_error=_on_error,
        on_submit_error=on_submit_error,
    )


def _extract_concurrency_hint(obj: Any) -> tuple[Optional[str], bool]:
    """
    Extract concurrency preferences from an object.

    Checks for direct attributes matching the ConcurrencyProfile
    protocol and an optional ``concurrency_profile`` attribute.
    """
    if obj is None:
        return None, False

    preferred = getattr(obj, "preferred_executor", None)
    cpu_intensive = getattr(obj, "cpu_intensive", False)

    profile = getattr(obj, "concurrency_profile", None)
    if profile:
        preferred = getattr(profile, "preferred_executor", preferred)
        cpu_intensive = getattr(profile, "cpu_intensive", cpu_intensive)

    if isinstance(preferred, str):
        val = preferred.strip().lower()
        if val in {"thread", "process"}:
            preferred = val
        else:
            preferred = None
    else:
        preferred = None

    return preferred, bool(cpu_intensive)


def _infer_executor_kind(cfg: SievioConfig, runtime: Any | None = None) -> str:
    """
    Infer executor kind by inspecting registered components for concurrency hints.
    """
    components: list[Any] = []

    if runtime and getattr(runtime, "sources", None):
        components.extend(runtime.sources)
    elif getattr(cfg.sources, "sources", None):
        components.extend(cfg.sources.sources)

    handler_pairs: list[tuple[Any, Any]] = list(getattr(cfg.pipeline, "bytes_handlers", ()))
    if runtime and getattr(runtime, "bytes_handlers", None):
        handler_pairs.extend(runtime.bytes_handlers)
    for _, handler in handler_pairs:
        components.append(handler)

    extractor = getattr(cfg.pipeline, "file_extractor", None)
    if extractor:
        components.append(extractor)
    if runtime and getattr(runtime, "file_extractor", None):
        components.append(runtime.file_extractor)

    cpu_bound_votes = 0
    forced_kind: Optional[str] = None

    for obj in components:
        pref, cpu = _extract_concurrency_hint(obj)
        if cpu:
            cpu_bound_votes += 1
        if pref == "process":
            forced_kind = "process"
        elif pref == "thread" and forced_kind is None:
            forced_kind = "thread"

    if forced_kind:
        return forced_kind

    if cpu_bound_votes > 0:
        return "process"

    return "thread"


def infer_executor_kind(cfg: SievioConfig, *, default: str = "thread", runtime: Any | None = None) -> Literal["thread", "process"]:
    """Return a validated executor kind for the pipeline.


    This wraps :func:`_infer_executor_kind` and falls back to the given
    default when the heuristics cannot determine a valid kind.

    Args:
        cfg (SievioConfig): Top-level configuration object.
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


def resolve_pipeline_executor_config(cfg: SievioConfig, runtime: Any | None = None) -> tuple[ExecutorConfig, bool]:
    """Build executor settings for the main ingestion pipeline.

    The pipeline section of the configuration controls the maximum
    worker count, submission window, executor kind, and fail-fast
    behavior. When ``executor_kind`` is ``"auto"``, a suitable kind
    is inferred from the configured sources and bytes handlers.

    Args:
        cfg (SievioConfig): Top-level configuration object.
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


def resolve_qc_executor_config(cfg: SievioConfig, runtime: Any | None = None) -> ExecutorConfig:
    """Build executor settings for post-extraction QC scoring.


    QC executor settings are derived from both the QC and pipeline
    sections of the configuration. When the QC-specific fields are
    unset, pipeline defaults are reused, and ``"auto"`` executor kinds
    are resolved using the same heuristics as the main pipeline.

    Args:
        cfg (SievioConfig): Top-level configuration object.
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
