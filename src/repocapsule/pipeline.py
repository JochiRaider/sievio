# pipeline.py
# SPDX-License-Identifier: MIT

from __future__ import annotations

from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait, Future
from dataclasses import dataclass, field
from typing import Optional, Iterable, Sequence, Dict, List, Tuple, Any
from pathlib import Path

from .config import RepocapsuleConfig
from .interfaces import Source, Sink, RepoContext, Record
from .convert import make_records_from_bytes
from .log import get_logger


log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class PipelineStats:
    files: int = 0
    bytes: int = 0
    records: int = 0
    sink_errors: int = 0
    source_errors: int = 0
    by_ext: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        # Keep a simple, stable shape for external reporting/JSONL footers.
        return {
            "files": int(self.files),
            "bytes": int(self.bytes),
            "records": int(self.records),
            "sink_errors": int(self.sink_errors),
            "source_errors": int(self.source_errors),
            "by_ext": dict(self.by_ext),
        }



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

def make_records_from_bytes_iter(
    data: bytes,
    rel_path: str,
    *,
    config,
    context=None,
):
    """Yield records one-by-one to reduce peak memory per file."""
    recs = make_records_from_bytes(data, rel_path, config=config, context=context)
    # Existing make_records... may return list; normalize to iterator
    for r in (recs if isinstance(recs, list) else recs):
        yield r

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_pipeline(*, config: RepocapsuleConfig) -> Dict[str, int]:
    """Run the end-to-end pipeline described by ``config``."""
    cfg = config
    stats = PipelineStats()

    with ExitStack() as stack:
        open_sources: List[Source] = [_open_source_with_stack(stack, src) for src in cfg.sources.sources]
        initial_ctx: Optional[RepoContext] = cfg.sinks.context
        open_sinks: List[Sink] = _prepare_sinks(stack, cfg.sinks.sinks, initial_ctx)
        if not open_sinks:
            log.warning("No sinks are open; processed records will be dropped.")

        def process_one(item: Any, ctx: Optional[RepoContext]) -> Tuple[Any, Iterable[Record]]:
            rel = getattr(item, "path", None) or getattr(item, "rel_path", None)
            data = getattr(item, "data", None)
            if rel is None or data is None:
                raise ValueError("FileItem missing 'path' or 'data'")
            recs = make_records_from_bytes(
                data,
                rel,
                config=cfg,
                context=ctx,
            )
            # do NOT materialize; allow generators to stream
            return item, (recs if isinstance(recs, list) else recs)

        def _increment_file_stats(item: Any) -> None:
            size = getattr(item, "size", None)
            if size is None:
                data = getattr(item, "data", b"")
                size = len(data) if isinstance(data, (bytes, bytearray)) else 0
            stats.files += 1
            stats.bytes += int(size or 0)
            ext = _ext_key(getattr(item, "path", ""))
            stats.by_ext[ext] = stats.by_ext.get(ext, 0) + 1

        def _write_records(item: Any, recs: Iterable[Record]) -> None:
            for record in recs:
                wrote_any = False
                for sink in open_sinks:
                    try:
                        sink.write(record)  # type: ignore[attr-defined]
                        wrote_any = True
                    except Exception as exc:  # sink failure should not stop pipeline
                        log.warning(
                            "Sink %s failed to write record for %s: %s",
                            getattr(sink, "__class__", type(sink)).__name__,
                            getattr(item, "path", "<unknown>"),
                            exc,
                        )
                        stats.sink_errors += 1
                if wrote_any:
                    stats.records += 1

        def _process_serial(item: Any, ctx: Optional[RepoContext]) -> None:
            try:
                _increment_file_stats(item)
                _, recs = process_one(item, ctx)
                _write_records(item, recs)
            except Exception as exc:
                log.warning(
                    "Processing failed for %s: %s",
                    getattr(item, "path", "<unknown>"),
                    exc,
                )
                stats.source_errors += 1
                if cfg.pipeline.fail_fast:
                    raise

        for source in open_sources:
            ctx = getattr(source, "context", cfg.sinks.context)
            items = source.iter_files()

            if cfg.pipeline.max_workers <= 1:
                for item in items:
                    _process_serial(item, ctx)
                continue

            window = cfg.pipeline.submit_window or (4 * cfg.pipeline.max_workers)
            window = max(window, cfg.pipeline.max_workers)
            with ThreadPoolExecutor(max_workers=cfg.pipeline.max_workers) as pool:
                pending: List[Future[Tuple[Any, Iterable[Record]]]] = []

                def _drain(block: bool = False) -> List[Tuple[Any, List[Record]]]:
                    nonlocal pending
                    if not pending:
                        return []
                    done, still = wait(
                        pending,
                        timeout=None if block else 0.0,
                        return_when=FIRST_COMPLETED,
                    )
                    pending = list(still)
                    results: List[Tuple[Any, List[Record]]] = []
                    for fut in done:
                        try:
                            results.append(fut.result())
                        except Exception as exc:
                            log.warning("Worker failed: %s", exc)
                            stats.source_errors += 1
                            if cfg.pipeline.fail_fast:
                                raise
                    return results

                for item in items:
                    _increment_file_stats(item)
                    try:
                        pending.append(pool.submit(process_one, item, ctx))
                        if len(pending) >= window:
                            for _item, recs in _drain(block=True):
                                _write_records(_item, recs)
                    except Exception as exc:
                        log.warning("Scheduling failed for %s: %s", getattr(item, "path", "<unknown>"), exc)
                        stats.source_errors += 1
                        if cfg.pipeline.fail_fast:
                            raise

                while pending:
                    for _item, recs in _drain(block=True):
                        _write_records(_item, recs)

    return stats.as_dict()

__all__ = ["run_pipeline", "PipelineStats"]    
