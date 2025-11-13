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
from .qc_utils import update_dup_family_counts, top_dup_families


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
    qc_scored: int = 0
    qc_kept: int = 0
    qc_dropped_low_score: int = 0
    qc_dropped_near_dup: int = 0
    qc_errors: int = 0
    qc_enabled: bool = False
    qc_mode: str = "inline"
    qc_dup_families: Dict[str, Dict[str, Any]] = field(default_factory=dict)

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
        data["qc"] = {
            "enabled": bool(self.qc_enabled),
            "mode": self.qc_mode,
            "scored": int(self.qc_scored),
            "kept": int(self.qc_kept),
            "dropped_low_score": int(self.qc_dropped_low_score),
            "dropped_near_dup": int(self.qc_dropped_near_dup),
            "errors": int(self.qc_errors),
            "top_dup_families": top_dup_families(self.qc_dup_families),
        }
        return data

    def record_dup_family(self, family_id: Optional[str], path: Optional[str]) -> None:
        update_dup_family_counts(self.qc_dup_families, family_id, path)

    def qc_top_dup_families(self) -> List[Dict[str, Any]]:
        return top_dup_families(self.qc_dup_families)



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
    qc_cfg = cfg.qc
    stats.qc_enabled = bool(qc_cfg.enabled)
    stats.qc_mode = qc_cfg.mode
    qc_active = bool(qc_cfg.enabled and qc_cfg.mode == "inline" and getattr(qc_cfg, "scorer", None))
    if qc_cfg.enabled and not qc_active:
        log.warning("QC enabled but no scorer is configured; skipping inline annotations.")

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

        def _merge_qc_meta(record: Dict[str, Any], qc_result: Dict[str, Any]) -> None:
            if not isinstance(record, dict):
                return
            meta = record.get("meta")
            if not isinstance(meta, dict):
                meta = {}
                record["meta"] = meta
            tokens_est = qc_result.get("tokens")
            if tokens_est is not None:
                meta["approx_tokens"] = tokens_est
                meta.setdefault("tokens", tokens_est)
            updates = {
                "quality_score": qc_result.get("score"),
                "parse_ok": qc_result.get("parse_ok"),
                "repetition": qc_result.get("repetition"),
                "ascii_ratio": qc_result.get("ascii_ratio"),
                "code_complexity": qc_result.get("code_complexity"),
                "gopher_quality": qc_result.get("gopher_quality"),
                "gopher_flags": qc_result.get("gopher_flags"),
                "near_dup": qc_result.get("near_dup"),
                "near_dup_minhash": qc_result.get("near_dup_minhash"),
                "near_dup_simhash": qc_result.get("near_dup_simhash"),
                "minhash_jaccard": qc_result.get("minhash_jaccard"),
                "minhash_dup_of": qc_result.get("minhash_dup_of"),
                "simhash_dup_of": qc_result.get("simhash_dup_of"),
                "dup_family_id": qc_result.get("dup_family_id"),
                "perplexity": qc_result.get("perplexity"),
            }
            for key, value in updates.items():
                if value is None:
                    continue
                meta[key] = value

        def _score_and_gate_record(record: Dict[str, Any]) -> bool:
            if not qc_active:
                return True
            scorer = qc_cfg.scorer
            if scorer is None:
                return True
            meta = record.get("meta") if isinstance(record, dict) else None
            path = (meta.get("path") if isinstance(meta, dict) else None) or record.get("path") or "<unknown>"
            try:
                qc_result = scorer.score_record(record)
            except Exception as exc:  # pragma: no cover - depends on scorer internals
                stats.qc_errors += 1
                if qc_cfg.fail_on_error:
                    raise
                log.warning("QC scoring failed for %s: %s", path, exc)
                return True
            stats.qc_scored += 1
            _merge_qc_meta(record, qc_result)
            meta_after = record.get("meta") if isinstance(record, dict) else None
            path_for_dup = (meta_after.get("path") if isinstance(meta_after, dict) else None) or path
            family_id = qc_result.get("dup_family_id")
            if isinstance(meta_after, dict):
                family_id = meta_after.get("dup_family_id", family_id)
            stats.record_dup_family(family_id, path_for_dup)
            drop_reason: Optional[str] = None
            score_value = qc_result.get("score")
            min_score = qc_cfg.min_score
            if min_score is not None and score_value is not None and float(score_value) < float(min_score):
                drop_reason = "low_score"
            if drop_reason is None and qc_cfg.drop_near_dups and qc_result.get("near_dup"):
                drop_reason = "near_dup"
            if drop_reason == "low_score":
                stats.qc_dropped_low_score += 1
                log.debug(
                    "Dropping %s: quality_score %.2f < %.2f",
                    path,
                    float(score_value or 0.0),
                    float(min_score or 0.0),
                )
                return False
            if drop_reason == "near_dup":
                stats.qc_dropped_near_dup += 1
                log.debug(
                    "Dropping %s: near duplicate (minhash_jaccard=%.4f)",
                    path,
                    float(qc_result.get("minhash_jaccard") or 0.0),
                )
                return False
            stats.qc_kept += 1
            return True

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
                if not _score_and_gate_record(record):
                    continue
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

                def _drain(block: bool = False) -> List[Tuple[Any, Iterable[Record]]]:
                    nonlocal pending
                    if not pending:
                        return []
                    done, still = wait(
                        pending,
                        timeout=None if block else 0.0,
                        return_when=FIRST_COMPLETED,
                    )
                    pending = list(still)
                    results: List[Tuple[Any, Iterable[Record]]] = []
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

    if qc_cfg.enabled:
        min_score_str = (
            f"{qc_cfg.min_score:.1f}" if qc_cfg.min_score is not None else "off"
        )
        log.info(
            "QC summary (min_score=%s, drop_near_dups=%s)\n"
            "  scored: %d\n"
            "  kept: %d\n"
            "  dropped_low_score: %d\n"
            "  dropped_near_dup: %d\n"
            "  errors: %d",
            min_score_str,
            "on" if qc_cfg.drop_near_dups else "off",
            stats.qc_scored,
            stats.qc_kept,
            stats.qc_dropped_low_score,
            stats.qc_dropped_near_dup,
            stats.qc_errors,
        )
        top = stats.qc_top_dup_families()
        if top:
            lines = [
                f"    - {entry['dup_family_id']}: count={entry['count']} examples={entry.get('examples', [])}"
                for entry in top
            ]
            log.info("Largest duplicate families:\n%s", "\n".join(lines))
        else:
            log.info("Largest duplicate families: none")

    return stats.as_dict()

__all__ = ["run_pipeline", "PipelineStats"]    
