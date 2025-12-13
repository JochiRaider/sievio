# convert.py
# SPDX-License-Identifier: MIT
"""Helpers to decode file inputs and produce chunk and extractor records."""
from __future__ import annotations

import io
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse

from .config import SievioConfig, FileProcessingConfig, DecodeConfig, ChunkConfig
from .chunk import ChunkPolicy, iter_chunk_dicts
from .decode import decode_bytes
from .factories_sources import UnsupportedBinary
from ..sources.fs import read_file_prefix
from .interfaces import FileExtractor, FileItem, RepoContext, Record, StreamingExtractor
from .language_id import classify_path_kind
from .log import get_logger
from .records import build_record

ConfigForRecords = SievioConfig | FileProcessingConfig


class _LimitedStream(io.BufferedReader):
    """Buffered reader that enforces a hard byte limit on read() calls."""

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
    "list_records_for_file",
    "list_records_from_bytes",
    "iter_records_from_bytes_with_plan",
    "build_records_from_bytes",
    "ByteSource",
    "RecordBuilderContext",
    "iter_records_from_bytes",
    "iter_records_for_file",
    "build_records_from_bytes",
    "iter_records_from_file_item",
    "resolve_bytes_from_file_item",
    "DefaultExtractor",
    "make_records_for_file",
    "make_records_from_bytes",
]

log = get_logger(__name__)

Sniff = Callable[[bytes, str], bool]
BytesHandler = Callable[
    [bytes, str, Optional[RepoContext], Optional[ChunkPolicy]],
    Optional[Iterable[Record]],
]


@dataclass(slots=True)
class ByteSource:
    """Container for file bytes with origin metadata.

    Attributes:
        data (bytes | None): Raw file bytes when available.
        origin (Path | None): Path used to reopen streamable files.
        size (int | None): File size in bytes when known.
    """

    data: Optional[bytes]
    origin: Optional[Path]
    size: Optional[int]


@dataclass(slots=True)
class RecordBuilderContext:
    """Configuration needed to build records.

    Attributes:
        decode (DecodeConfig): Decode settings to apply to bytes.
        chunk (ChunkConfig): Chunking configuration for record generation.
        metadata_seed (Mapping[str, Any] | None): Metadata added to each
            record when provided.
    """

    decode: DecodeConfig
    chunk: ChunkConfig
    metadata_seed: Mapping[str, Any] | None = None

# ---------------------------------------------------------------------------
# Record creation for a single file
# ---------------------------------------------------------------------------

# Shape of the callable adapter the pipeline passes into this module
# (it wraps interfaces.Extractor.extract and returns an Iterable[Record])
ExtractorFn = Callable[[str, str], Optional[Iterable[Record]]]

def _build_record_context(config: ConfigForRecords, context: Optional[RepoContext]) -> RecordBuilderContext:
    """Build a RecordBuilderContext from config and repository context.

    Args:
        config (ConfigForRecords): Config holding decode and chunk settings.
        context (RepoContext | None): Repository context used for metadata.

    Returns:
        RecordBuilderContext: Context used during record creation.
    """
    meta_seed = (context.as_meta_seed() or None) if context else None
    return RecordBuilderContext(
        decode=config.decode,
        chunk=config.chunk,
        metadata_seed=meta_seed,
    )


def _augment_context_with_source(
    context: Optional[RepoContext],
    source_url: Optional[str],
    source_domain: Optional[str],
) -> Optional[RepoContext]:
    """Merge source URL metadata into an existing repository context.

    Args:
        context (RepoContext | None): Existing repository context.
        source_url (str | None): Source URL to attach when provided.
        source_domain (str | None): Domain derived from the source URL.

    Returns:
        RepoContext | None: Augmented context with added extra fields.
    """
    if not source_url and not source_domain:
        return context
    extra: Dict[str, Any] = {}
    if context and context.extra:
        extra.update(context.extra)
    if source_url:
        extra.setdefault("url", source_url)
    if source_domain:
        extra.setdefault("source_domain", source_domain)
    return replace(
        context or RepoContext(),
        extra=extra,
    )


def resolve_bytes_from_file_item(item: FileItem, decode_cfg: DecodeConfig) -> ByteSource:
    """Normalize a FileItem into a ByteSource, reopening the origin when needed.

    Args:
        item (FileItem): File descriptor that may carry bytes or a path.
        decode_cfg (DecodeConfig): Decode settings controlling byte limits.

    Returns:
        ByteSource: Loaded bytes plus origin and size metadata.
    """
    data = getattr(item, "data", None)
    file_size = getattr(item, "size", None)
    origin = getattr(item, "origin_path", None)

    if data is not None:
        return ByteSource(data=data, origin=None, size=file_size)

    if getattr(item, "streamable", False) and origin:
        origin_path = Path(origin)
        max_bytes = decode_cfg.max_bytes_per_file
        reopened, size = read_file_prefix(origin_path, max_bytes, file_size=file_size)
        return ByteSource(data=reopened, origin=origin_path, size=size)

    return ByteSource(data=None, origin=None, size=file_size)


# --- Public entrypoint the pipeline can call ---

def list_records_for_file(
    *,
    text: str,
    rel_path: str,
    config: ConfigForRecords,
    context: Optional[RepoContext],
    encoding: str,
    had_replacement: bool,
    file_bytes: int | None = None,
    truncated_bytes: int | None = None,
    source_url: Optional[str] = None,
    source_domain: Optional[str] = None,
) -> List[Dict[str, object]]:
    """Materialize records for a decoded file using configured extractors.

    Args:
        text (str): Text content already decoded.
        rel_path (str): Repository-relative path for the file.
        config (ConfigForRecords): File processing configuration.
        context (RepoContext | None): Repository context for metadata.
        encoding (str): Encoding detected during decode.
        had_replacement (bool): Whether decoding substituted characters.
        file_bytes (int | None): Total file size in bytes when known.
        truncated_bytes (int | None): Bytes omitted due to size limits.
        source_url (str | None): Original source URL if available.
        source_domain (str | None): Domain associated with the source URL.

    Returns:
        List[Dict[str, object]]: Materialized record dictionaries.
    """
    cfg = config
    record_ctx = _build_record_context(cfg, context)
    return list(
        iter_records_for_file(
            text=text,
            rel_path=rel_path,
            record_ctx=record_ctx,
            context=context,
            encoding=encoding,
            had_replacement=had_replacement,
            file_bytes=file_bytes,
            truncated_bytes=truncated_bytes,
            source_url=source_url,
            source_domain=source_domain,
            extractors=cfg.pipeline.extractors,
        )
    )


def iter_records_for_file(
    *,
    text: str,
    rel_path: str,
    record_ctx: RecordBuilderContext,
    context: Optional[RepoContext],
    encoding: str,
    had_replacement: bool,
    file_bytes: int | None = None,
    truncated_bytes: int | None = None,
    source_url: Optional[str] = None,
    source_domain: Optional[str] = None,
    extractors: Sequence[Any] = (),
) -> Iterator[Dict[str, object]]:
    """Yield chunk records followed by extractor records for decoded text.

    Args:
        text (str): Decoded file content.
        rel_path (str): Repository-relative path.
        record_ctx (RecordBuilderContext): Decode and chunk configuration.
        context (RepoContext | None): Repository context for metadata.
        encoding (str): Encoding detected during decode.
        had_replacement (bool): Whether decoding substituted characters.
        file_bytes (int | None): File size in bytes when known.
        truncated_bytes (int | None): Bytes omitted because of limits.
        source_url (str | None): Original source URL if available.
        source_domain (str | None): Domain associated with the source URL.
        extractors (Sequence[Any]): Additional extractors to run.

    Yields:
        Dict[str, object]: Chunk or extractor record dictionaries.
    """
    extractor_recs: List[Dict[str, object]] = []
    context_meta = record_ctx.metadata_seed
    file_nlines = 0 if text == "" else text.count("\n") + 1
    if extractors:
        for extractor in extractors:
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

    mode, fmt = classify_path_kind(rel_path)
    chunk_dicts = list(
        iter_chunk_dicts(
            text,
            mode=mode,
            fmt=fmt,
            policy=record_ctx.chunk.policy,
            tokenizer_name=record_ctx.chunk.tokenizer_name,
        )
    )
    total_chunks = len(chunk_dicts)
    attach_lang = record_ctx.chunk.attach_language_metadata

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
            lang=chunk.get("lang") if attach_lang else None,
            tokens=chunk.get("n_tokens"),
            extra_meta=context_meta,
            file_bytes=file_bytes,
            truncated_bytes=truncated_bytes,
            file_nlines=file_nlines,
            url=source_url,
            source_domain=source_domain,
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
    source_url: Optional[str] = None,
    source_domain: Optional[str] = None,
) -> Iterator[Dict[str, object]]:
    """Decode bytes for a file and yield chunk and extractor records.

    Args:
        data (bytes): Raw file bytes.
        rel_path (str): Repository-relative path.
        config (ConfigForRecords): File processing configuration.
        context (RepoContext | None): Repository context for metadata.
        file_size (int | None): File size in bytes when known.
        source_url (str | None): Source URL associated with the file.
        source_domain (str | None): Domain associated with the source URL.

    Yields:
        Dict[str, object]: Chunk or extractor record dictionaries.
    """
    cfg = config
    derived_domain = source_domain
    if derived_domain is None and source_url:
        try:
            derived_domain = urlparse(source_url).hostname
        except Exception:
            derived_domain = None
    ctx_with_source = _augment_context_with_source(context, source_url, derived_domain)
    record_ctx = _build_record_context(cfg, ctx_with_source)
    handlers = list(cfg.pipeline.bytes_handlers)
    yield from build_records_from_bytes(
        data,
        rel_path,
        record_ctx=record_ctx,
        bytes_handlers=handlers,
        extractors=cfg.pipeline.extractors,
        context=ctx_with_source,
        chunk_policy=cfg.chunk.policy,
        source_url=source_url,
        source_domain=derived_domain,
        file_size=file_size,
    )


def iter_records_from_bytes_with_plan(
    data: bytes,
    rel_path: str,
    *,
    plan: Any,
    context: Optional[RepoContext] = None,
    file_size: int | None = None,
    source_url: Optional[str] = None,
    source_domain: Optional[str] = None,
) -> Iterator[Dict[str, object]]:
    """Build records from bytes using handlers from a pipeline plan.

    Args:
        data (bytes): Raw file bytes.
        rel_path (str): Repository-relative path.
        plan (Any): Pipeline plan containing runtime handlers.
        context (RepoContext | None): Repository context for metadata.
        file_size (int | None): File size in bytes when known.
        source_url (str | None): Source URL associated with the file.
        source_domain (str | None): Domain associated with the source URL.

    Returns:
        Iterator[Dict[str, object]]: Records produced from the bytes.
    """
    cfg = plan.spec
    runtime = plan.runtime
    pipeline_cfg = replace(cfg.pipeline, bytes_handlers=runtime.bytes_handlers)
    fp_cfg = FileProcessingConfig(decode=cfg.decode, chunk=cfg.chunk, pipeline=pipeline_cfg)
    return iter_records_from_bytes(
        data,
        rel_path,
        config=fp_cfg,
        context=context,
        file_size=file_size,
        source_url=source_url,
        source_domain=source_domain,
    )


def list_records_from_bytes(
    data: bytes,
    rel_path: str,
    *,
    config: ConfigForRecords,
    context: Optional[RepoContext],
) -> List[Dict[str, object]]:
    """Materialize records from raw bytes for a single file.

    Args:
        data (bytes): Raw file bytes.
        rel_path (str): Repository-relative path.
        config (ConfigForRecords): File processing configuration.
        context (RepoContext | None): Repository context for metadata.

    Returns:
        List[Dict[str, object]]: Materialized record dictionaries.
    """
    return list(
        iter_records_from_bytes(
            data,
            rel_path,
            config=config,
            context=context,
        )
    )

# Compatibility aliases
make_records_for_file = list_records_for_file
make_records_from_bytes = list_records_from_bytes


def build_records_from_bytes(
    data: bytes,
    rel_path: str,
    *,
    record_ctx: RecordBuilderContext,
    bytes_handlers: Sequence[Tuple[Sniff, BytesHandler]],
    extractors: Sequence[Any],
    context: Optional[RepoContext],
    chunk_policy: ChunkPolicy,
    file_size: int | None = None,
    source_url: Optional[str] = None,
    source_domain: Optional[str] = None,
) -> Iterator[Dict[str, object]]:
    """Build records from bytes by applying sniffers, decoding, and chunking.

    The caller is responsible for deciding streaming versus buffered handling
    and constructing the RecordBuilderContext along with handler lists.

    Args:
        data (bytes): Raw file bytes.
        rel_path (str): Repository-relative path.
        record_ctx (RecordBuilderContext): Decode and chunk configuration.
        bytes_handlers (Sequence[Tuple[Sniff, BytesHandler]]): Ordered
            sniff handlers that may short-circuit processing.
        extractors (Sequence[Any]): Extractors to run after chunking.
        context (RepoContext | None): Repository context for metadata.
        chunk_policy (ChunkPolicy): Chunking policy passed to byte handlers.
        file_size (int | None): File size in bytes when known.
        source_url (str | None): Source URL associated with the file.
        source_domain (str | None): Domain associated with the source URL.

    Yields:
        Dict[str, object]: Chunk or extractor record dictionaries.
    """
    for sniff, handler in bytes_handlers:
        if sniff(data, rel_path):
            try:
                records = handler(data, rel_path, context, chunk_policy)
            except UnsupportedBinary as e:
                log.info("Skipping unsupported binary for %s: %s", rel_path, e)
                records = None
            if records:
                yield from records
                return

    max_bytes = record_ctx.decode.max_bytes_per_file
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
        normalize=record_ctx.decode.normalize,
        strip_controls=record_ctx.decode.strip_controls,
        fix_mojibake=record_ctx.decode.fix_mojibake,
    )
    yield from iter_records_for_file(
        text=dec.text,
        rel_path=rel_path,
        record_ctx=record_ctx,
        context=context,
        encoding=dec.encoding,
        had_replacement=dec.had_replacement,
        file_bytes=file_bytes,
        truncated_bytes=truncated_bytes,
        source_url=source_url,
        source_domain=source_domain,
        extractors=extractors,
    )


def iter_records_from_file_item(
    item: FileItem,
    *,
    config: ConfigForRecords,
    context: Optional[RepoContext],
    streaming_extractor: Optional[StreamingExtractor] = None,
) -> Iterator[Dict[str, object]]:
    """Yield records for a FileItem using streaming or buffered decoding.

    Args:
        item (FileItem): File descriptor that may be streamable.
        config (ConfigForRecords): File processing configuration.
        context (RepoContext | None): Repository context for metadata.
        streaming_extractor (StreamingExtractor | None): Optional extractor
            that consumes a limited stream before buffering.

    Yields:
        Dict[str, object]: Chunk or extractor record dictionaries.

    Raises:
        ValueError: If the file item lacks bytes and cannot be streamed.
    """
    cfg = config
    origin_url: Optional[str] = None
    origin_domain: Optional[str] = None
    if getattr(item, "origin_path", None):
        try:
            parsed = urlparse(str(item.origin_path))
            if parsed.scheme in {"http", "https"}:
                origin_url = str(item.origin_path)
                origin_domain = parsed.hostname
        except Exception:
            origin_url = None
            origin_domain = None

    if (
        streaming_extractor is not None
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

    resolved = resolve_bytes_from_file_item(item, cfg.decode)
    if resolved.data is None:
        raise ValueError(f"FileItem {item.path!r} missing data and not streamable")

    yield from iter_records_from_bytes(
        resolved.data,
        item.path,
        config=cfg,
        context=context,
        file_size=resolved.size,
        source_url=origin_url,
        source_domain=origin_domain,
    )


class DefaultExtractor(FileExtractor):
    """Adapter that reuses built-in decoding and record generation.

    Attributes:
        streaming_extractor (StreamingExtractor | None): Optional streaming
            extractor used before buffered decoding.
    """

    def __init__(self, streaming_extractor: Optional[StreamingExtractor] = None) -> None:
        """Initialize the extractor.

        Args:
            streaming_extractor (StreamingExtractor | None): Optional
                streaming extractor to run before buffered decoding.
        """
        self.streaming_extractor = streaming_extractor

    def extract(
        self,
        item: FileItem,
        *,
        config: ConfigForRecords,
        context: Optional[RepoContext] = None,
    ) -> Iterable[Dict[str, object]]:
        """Extract records for a file item using the configured pipeline.

        Args:
            item (FileItem): File descriptor to process.
            config (ConfigForRecords): File processing configuration.
            context (RepoContext | None): Repository context for metadata.

        Returns:
            Iterable[Dict[str, object]]: Iterator over record dictionaries.
        """
        return iter_records_from_file_item(
            item,
            config=config,
            context=context,
            streaming_extractor=self.streaming_extractor,
        )
