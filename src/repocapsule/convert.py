# convert.py
# SPDX-License-Identifier: MIT
from __future__ import annotations

import io
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from .config import RepocapsuleConfig, FileProcessingConfig
from .chunk import ChunkPolicy, iter_chunk_dicts
from .decode import decode_bytes
from .factories import UnsupportedBinary
from .fs import read_file_prefix
from .interfaces import FileExtractor, FileItem, RepoContext, Record, StreamingExtractor
from .log import get_logger
from .records import build_record

ConfigForRecords = RepocapsuleConfig | FileProcessingConfig


def _get_streaming_extractor(config: ConfigForRecords) -> Optional[StreamingExtractor]:
    """
    Return the pipeline file_extractor if it also supports the StreamingExtractor protocol.
    """

    pipeline_cfg = getattr(config, "pipeline", None)
    if pipeline_cfg is None:
        return None
    extractor = getattr(pipeline_cfg, "file_extractor", None)
    if extractor is None:
        return None
    return extractor if isinstance(extractor, StreamingExtractor) else None


class _LimitedStream(io.BufferedReader):
    """
    Buffered reader that enforces a hard byte limit on read() calls.
    """

    def __init__(self, raw: io.BufferedIOBase, limit: int):
        super().__init__(raw)
        self._remaining = max(0, int(limit))

    def read(self, size: int = -1) -> bytes:  # type: ignore[override]
        if self._remaining <= 0:
            return b""
        if size < 0 or size > self._remaining:
            size = self._remaining
        chunk = super().read(size)
        self._remaining -= len(chunk)
        return chunk

__all__ = [
    "make_records_for_file",
    "make_records_from_bytes",
    "iter_records_for_file",
    "iter_records_from_bytes",
    "iter_records_from_file_item",
    "DefaultExtractor",
]

log = get_logger(__name__)

Sniff = Callable[[bytes, str], bool]
BytesHandler = Callable[
    [bytes, str, Optional[RepoContext], Optional[ChunkPolicy]],
    Optional[Iterable[Record]],
]

# ---------------------------------------------------------------------------
# Record creation for a single file
# ---------------------------------------------------------------------------

# Shape of the callable adapter the pipeline passes into this module
# (it wraps interfaces.Extractor.extract and returns an Iterable[Record])
ExtractorFn = Callable[[str, str], Optional[Iterable[Record]]]

_MD_EXTS = {".md", ".mdx", ".markdown"}
_RST_EXTS = {".rst"}
_DOC_EXTS = _MD_EXTS | _RST_EXTS | {".adoc", ".txt"}


def _infer_mode_and_fmt(rel_path: str) -> Tuple[str, Optional[str]]:
    """Return (mode, fmt) from filename."""
    ext = Path(rel_path).suffix.lower()
    if ext in _MD_EXTS:
        return "doc", "md"
    if ext in _RST_EXTS:
        return "doc", "rst"
    if ext in _DOC_EXTS:
        return "doc", None
    return "code", None


# --- Public entrypoint the pipeline can call ---

def make_records_for_file(
    *,
    text: str,
    rel_path: str,
    config: ConfigForRecords,
    context: Optional[RepoContext],
    encoding: str,
    had_replacement: bool,
    file_bytes: int | None = None,
    truncated_bytes: int | None = None,
) -> List[Dict[str, object]]:
    """Materialize the record iterator for a decoded file into a list."""
    return list(
        iter_records_for_file(
            text=text,
            rel_path=rel_path,
            config=config,
            context=context,
            encoding=encoding,
            had_replacement=had_replacement,
            file_bytes=file_bytes,
            truncated_bytes=truncated_bytes,
        )
    )


def iter_records_for_file(
    *,
    text: str,
    rel_path: str,
    config: ConfigForRecords,
    context: Optional[RepoContext],
    encoding: str,
    had_replacement: bool,
    file_bytes: int | None = None,
    truncated_bytes: int | None = None,
) -> Iterator[Dict[str, object]]:
    """Yield chunk records followed by extractor records for a decoded file."""
    cfg = config
    extractor_recs: List[Dict[str, object]] = []
    context_meta = (context.as_meta_seed() or None) if context else None
    if cfg.pipeline.extractors:
        for extractor in cfg.pipeline.extractors:
            try:
                out = extractor.extract(text=text, path=rel_path, context=context)
            except Exception as exc:
                log.warning(
                    "Extractor %s failed for %s: %s",
                    getattr(extractor, "name", extractor),
                    rel_path,
                    exc,
                )
                continue
            if not out:
                continue
            extractor_recs.extend(dict(rec) for rec in out)

    mode, fmt = _infer_mode_and_fmt(rel_path)
    chunk_dicts = list(
        iter_chunk_dicts(
            text,
            mode=mode,
            fmt=fmt,
            policy=cfg.chunk.policy,
            tokenizer_name=cfg.chunk.tokenizer_name,
        )
    )
    total_chunks = len(chunk_dicts)

    for idx, chunk in enumerate(chunk_dicts, start=1):
        yield build_record(
            text=chunk.get("text", ""),
            rel_path=rel_path,
            repo_full_name=(context.repo_full_name if context else None),
            repo_url=(context.repo_url if context else None),
            license_id=(context.license_id if context else None),
            encoding=encoding,
            had_replacement=had_replacement,
            chunk_id=idx,
            n_chunks=total_chunks,
            lang=chunk.get("lang") if cfg.chunk.attach_language_metadata else None,
            tokens=chunk.get("n_tokens"),
            extra_meta=context_meta,
            file_bytes=file_bytes,
            truncated_bytes=truncated_bytes,
        )

    if extractor_recs and context_meta:
        for rec in extractor_recs:
            meta = rec.get("meta")
            if isinstance(meta, dict):
                for key, value in context_meta.items():
                    meta.setdefault(key, value)

    for rec in extractor_recs:
        yield rec


def iter_records_from_bytes(
    data: bytes,
    rel_path: str,
    *,
    config: ConfigForRecords,
    context: Optional[RepoContext],
    file_size: int | None = None,
) -> Iterator[Dict[str, object]]:
    cfg = config
    handlers = list(cfg.pipeline.bytes_handlers)
    for sniff, handler in handlers:
        if sniff(data, rel_path):
            try:
                records = handler(data, rel_path, context, cfg.chunk.policy)
            except UnsupportedBinary as e:
                log.info("Skipping unsupported binary for %s: %s", rel_path, e)
                records = None
            if records:
                yield from records
                return

    max_bytes = cfg.decode.max_bytes_per_file
    if max_bytes is not None and len(data) > max_bytes:
        data = data[:max_bytes]
    processed_len = len(data)
    file_bytes = file_size if file_size is not None else processed_len
    truncated_bytes = None
    if file_bytes is not None and file_bytes > processed_len:
        truncated_bytes = file_bytes - processed_len
        log.debug(
            "Truncated %s: file_bytes=%s used_bytes=%s", rel_path, file_bytes, processed_len
        )

    dec = decode_bytes(
        data,
        normalize=cfg.decode.normalize,
        strip_controls=cfg.decode.strip_controls,
        fix_mojibake=cfg.decode.fix_mojibake,
    )
    yield from iter_records_for_file(
        text=dec.text,
        rel_path=rel_path,
        config=cfg,
        context=context,
        encoding=dec.encoding,
        had_replacement=dec.had_replacement,
        file_bytes=file_bytes,
        truncated_bytes=truncated_bytes,
    )


def make_records_from_bytes(
    data: bytes,
    rel_path: str,
    *,
    config: ConfigForRecords,
    context: Optional[RepoContext],
) -> List[Dict[str, object]]:
    return list(
        iter_records_from_bytes(
            data,
            rel_path,
            config=config,
            context=context,
        )
    )


def iter_records_from_file_item(
    item: FileItem,
    *,
    config: ConfigForRecords,
    context: Optional[RepoContext],
) -> Iterator[Dict[str, object]]:
    cfg = config
    data = getattr(item, "data", None)
    file_size = getattr(item, "size", None)

    pipeline_cfg = getattr(cfg, "pipeline", None)
    exec_kind = "thread"
    if pipeline_cfg is not None:
        exec_kind = (getattr(pipeline_cfg, "executor_kind", "thread") or "thread").strip().lower()

    streaming_extractor = _get_streaming_extractor(cfg)
    if (
        streaming_extractor is not None
        and exec_kind == "thread"
        and getattr(item, "streamable", False)
        and item.origin_path
    ):
        origin = Path(item.origin_path)
        max_bytes = cfg.decode.max_bytes_per_file
        try:
            with origin.open("rb") as raw:
                if max_bytes is not None:
                    stream: io.BufferedReader = _LimitedStream(raw, max_bytes)
                else:
                    stream = io.BufferedReader(raw)
                rows = streaming_extractor.extract_stream(
                    stream=stream,
                    path=item.path,
                    context=context,
                )
                if rows is not None:
                    for row in rows:
                        yield row
                    return
        except Exception as exc:
            log.warning(
                "Streaming extractor failed for %s; falling back to buffered decode (%s)",
                item.path,
                exc,
            )

    if data is not None:
        yield from iter_records_from_bytes(
            data,
            item.path,
            config=cfg,
            context=context,
            file_size=file_size,
        )
        return
    if getattr(item, "streamable", False) and item.origin_path:
        origin = Path(item.origin_path)
        max_bytes = cfg.decode.max_bytes_per_file
        try:
            reopened, size = read_file_prefix(origin, max_bytes, file_size=file_size)
        except Exception as exc:
            raise RuntimeError(f"Failed to reopen stream for {item.path}: {exc}") from exc
        yield from iter_records_from_bytes(
            reopened,
            item.path,
            config=cfg,
            context=context,
            file_size=size,
        )
        return
    raise ValueError(f"FileItem {item.path!r} missing data and not streamable")


class DefaultExtractor(FileExtractor):
    """Fallback file-level extractor that reuses the built-in decoding/record flow."""

    def extract(
        self,
        item: FileItem,
        *,
        config: ConfigForRecords,
        context: Optional[RepoContext] = None,
    ) -> Iterable[Dict[str, object]]:
        return iter_records_from_file_item(item, config=config, context=context)
