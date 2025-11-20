# concurrency.py
# SPDX-License-Identifier: MIT
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
    max_workers: int
    window: int
    kind: Literal["thread", "process"]


class Executor:
    """
    Small wrapper around concurrent.futures executors that adds a bounded
    submission window and unordered consumption.
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
    ) -> None:
        """
        Submit items to workers and stream results as they complete.
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
                fut = pool.submit(fn, item)
                pending.append(fut)
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

    with executor._make_executor() as pool_exec:
        pending: list[Future[Tuple[Any, Iterable[Any]]]] = []

        def _submit_work(item: T) -> None:
            try:
                pending.append(pool_exec.submit(process_one, item))
            except Exception as exc:  # noqa: BLE001
                if on_submit_error:
                    on_submit_error(item, exc)
                if fail_fast:
                    raise

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
                except Exception as exc:  # noqa: BLE001
                    if on_worker_error:
                        on_worker_error(exc)
                    if fail_fast:
                        raise
                    continue
                write_records(item, recs)

        for work in items:
            _submit_work(work)
            if len(pending) >= cfg.window:
                _drain(block=True)
        while pending:
            _drain(block=True)


def _has_heavy_binary_handlers(cfg: RepocapsuleConfig) -> bool:
    """Return True if the pipeline is configured with PDF/EVTX handlers."""
    try:
        handlers = cfg.pipeline.bytes_handlers
    except Exception:
        return False

    for _sniff, handler in handlers:
        name = getattr(handler, "__name__", "").lower()
        mod = getattr(handler, "__module__", "").lower()
        if name in {"handle_pdf", "handle_evtx"}:
            return True
        if "pdfio" in mod or "evtxio" in mod:
            return True
    return False


def _has_heavy_sources(cfg: RepocapsuleConfig) -> bool:
    """
    Heuristic: consider the workload heavy when configured sources strongly suggest PDF/EVTX ingestion.
    """
    try:
        src_cfg = cfg.sources
    except Exception:
        return False

    try:
        for src in src_cfg.sources:
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
    """
    Return (preferred_executor, cpu_intensive) from an object or its ``concurrency_profile`` attribute.
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
    """
    Examine runtime objects (sources/handlers/extractors) for explicit concurrency hints.
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
    hinted = _preferred_executor_from_hints(runtime)
    if hinted in {"thread", "process"}:
        return hinted
    heavy_handlers = _has_heavy_binary_handlers(cfg)
    heavy_sources = _has_heavy_sources(cfg)
    if heavy_handlers and heavy_sources:
        return "process"
    return "thread"


def infer_executor_kind(cfg: RepocapsuleConfig, *, default: str = "thread", runtime: Any | None = None) -> Literal["thread", "process"]:
    normalized_default = default if default in {"thread", "process"} else "thread"
    kind = _infer_executor_kind(cfg, runtime=runtime)
    if kind not in {"thread", "process"}:
        return normalized_default  # type: ignore[return-value]
    return kind  # type: ignore[return-value]


def resolve_pipeline_executor_config(cfg: RepocapsuleConfig, runtime: Any | None = None) -> tuple[ExecutorConfig, bool]:
    """
    Return (executor_config, fail_fast) for the main pipeline.
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


def resolve_qc_executor_config(cfg: RepocapsuleConfig) -> ExecutorConfig:
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
        kind = _infer_executor_kind(cfg)
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
