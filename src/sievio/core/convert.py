# convert.py
# SPDX-License-Identifier: MIT
"""Helpers to decode file inputs and produce chunk and extractor records."""
from __future__ import annotations

import io
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse, urlunparse

from ..sources.fs import read_file_prefix
from .chunk import ChunkPolicy, iter_chunk_dicts
from .config import ChunkConfig, DecodeConfig, FileProcessingConfig, SievioConfig
from .decode import decode_bytes
from .factories_sources import UnsupportedBinary
from .interfaces import FileExtractor, FileItem, Record, RepoContext, StreamingExtractor
from .language_id import classify_path_kind
from .log import get_logger
from .records import build_record

ConfigForRecords = SievioConfig | FileProcessingConfig

_OPEN_STREAM_DEFAULT_MAX_BYTES = 16 * 1024 * 1024  # 16 MiB safety cap for open_stream buffering


class _LimitedRaw(io.RawIOBase):
    """Raw IO wrapper enforcing a total byte budget across read APIs."""

    def __init__(self, raw: io.BufferedIOBase, limit: int) -> None:
        super().__init__()
        self._raw = raw
        self._remaining = max(0, int(limit))

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        if self._remaining <= 0:
            return b""
        if size is None or size < 0 or size > self._remaining:
            size = self._remaining
        data = self._raw.read(size) or b""
        self._remaining -= len(data)
        return data

    def readinto(self, b: Any) -> int | None:
        if self._remaining <= 0:
            return 0
        view = memoryview(b)
        n = min(len(view), self._remaining)
        readinto = getattr(self._raw, "readinto", None)
        if callable(readinto):
            wrote = readinto(view[:n])
        else:
            data = self._raw.read(n) or b""
            view[: len(data)] = data
            wrote = len(data)
        if wrote is None:
            return None
        self._remaining -= int(wrote)
        return int(wrote)

    def readinto1(self, b: Any) -> int | None:
        fn = getattr(self._raw, "readinto1", None)
        if callable(fn):
            if self._remaining <= 0:
                return 0
            view = memoryview(b)
            n = min(len(view), self._remaining)
            wrote = fn(view[:n])
            if wrote is None:
                return None
            self._remaining -= int(wrote)
            return int(wrote)
        return self.readinto(b)

    def read1(self, size: int = -1) -> bytes:
        fn = getattr(self._raw, "read1", None)
        if callable(fn):
            if self._remaining <= 0:
                return b""
            if size is None or size < 0 or size > self._remaining:
                size = self._remaining
            data = fn(size) or b""
            self._remaining -= len(data)
            return data
        return self.read(size)

    def readline(self, size: int | None = -1) -> bytes:
        if self._remaining <= 0:
            return b""
        limit = self._remaining if size is None or size < 0 else min(size, self._remaining)
        data = self._raw.readline(limit) or b""
        self._remaining -= len(data)
        return data

    def seekable(self) -> bool:
        return False

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        raise io.UnsupportedOperation("seek not supported on limited stream")

    def close(self) -> None:
        try:
            self._raw.close()
        finally:
            super().close()


def make_limited_stream(raw: io.BufferedIOBase, limit: int | None) -> io.BufferedReader:
    """Wrap a raw binary stream with an optional total-byte limiter."""

    if limit is None:
        return io.BufferedReader(raw)
    return io.BufferedReader(_LimitedRaw(raw, limit))

__all__ = [
    "list_records_for_file",
    "list_records_from_bytes",
    "iter_records_from_bytes_with_plan",
    "build_records_from_bytes",
    "make_limited_stream",
    "maybe_reopenable_local_path",
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
    [bytes, str, RepoContext | None, ChunkPolicy | None],
    Iterable[Record] | None,
]


@dataclass(slots=True)
class ByteSource:
    """Container for file bytes with origin metadata.

    Attributes:
        data (bytes | None): Raw file bytes when available.
        origin (Path | None): Path used to reopen streamable files.
        size (int | None): File size in bytes when known.
    """

    data: bytes | None
    origin: Path | None
    size: int | None


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


def _derive_source_domain(source_url: str | None, source_domain: str | None) -> str | None:
    """Return a hostname derived from the source URL when none is provided."""

    if source_domain is not None:
        return source_domain
    if source_url:
        try:
            return urlparse(source_url).hostname
        except Exception:
            return None
    return None


def _normalize_source_ref(
    source_url: str | None,
    source_domain: str | None,
) -> tuple[str | None, str | None]:
    """Sanitize source URL metadata to avoid leaking credentials/tokens.

    Only http/https URLs are normalized; other schemes are omitted from
    record metadata to avoid surprising propagation of opaque identifiers.
    Userinfo, query strings, and fragments are stripped.
    """

    if not source_url:
        return None, source_domain
    try:
        parsed = urlparse(source_url)
    except Exception:
        return None, source_domain
    if parsed.scheme not in {"http", "https"}:
        return None, source_domain
    host = parsed.hostname
    port = f":{parsed.port}" if parsed.port else ""
    netloc = f"{host or ''}{port}"
    sanitized = urlunparse((parsed.scheme, netloc, parsed.path or "", "", "", ""))
    return sanitized or None, _derive_source_domain(sanitized, source_domain)

# ---------------------------------------------------------------------------
# Record creation for a single file
# ---------------------------------------------------------------------------

# Shape of the callable adapter the pipeline passes into this module
# (it wraps interfaces.Extractor.extract and returns an Iterable[Record])
ExtractorFn = Callable[[str, str], Iterable[Record] | None]

def _build_record_context(
    config: ConfigForRecords,
    context: RepoContext | None,
) -> RecordBuilderContext:
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
    context: RepoContext | None,
    source_url: str | None,
    source_domain: str | None,
) -> RepoContext | None:
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
    extra: dict[str, Any] = {}
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


def _can_open_stream(item: FileItem) -> bool:
    """Return True when the FileItem explicitly allows safe local streaming."""

    return bool(
        getattr(item, "streamable", False)
        and getattr(item, "stream_hint", None) in (None, "file")
        and callable(getattr(item, "open_stream", None))
    )


def maybe_reopenable_local_path(item: FileItem) -> Path | None:
    """Return a safe local Path to reopen for streaming, or None when disallowed.

    A path is reopenable only when the FileItem is marked streamable, the
    stream_hint indicates a real local file, and the origin resembles a local
    path (or file:// URL). Remote-looking origins (http/https/etc.) are
    rejected to avoid unintended network or filesystem access.
    """

    origin = getattr(item, "origin_path", None)
    if not origin or not getattr(item, "streamable", False):
        return None
    if getattr(item, "stream_hint", None) not in (None, "file"):
        return None
    origin_str = str(origin)
    candidate: Path | None = None
    if "://" in origin_str:
        try:
            parsed = urlparse(origin_str)
        except Exception:
            return None
        if parsed.scheme not in ("file",):
            return None
        if parsed.netloc not in ("", "localhost"):
            return None
        candidate = Path(parsed.path)
    else:
        # Treat as filesystem path to avoid mis-parsing Windows drive letters.
        candidate = Path(origin_str)
    try:
        resolved = candidate.resolve()
    except OSError:
        return None
    if not resolved.exists() or not resolved.is_file():
        return None
    return resolved


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
    reopenable = maybe_reopenable_local_path(item)

    if data is not None:
        return ByteSource(data=data, origin=None, size=file_size)

    if _can_open_stream(item):
        try:
            with closing(item.open_stream()) as raw:  # type: ignore[misc]
                limit = decode_cfg.max_bytes_per_file
                if limit is None:
                    limit = _OPEN_STREAM_DEFAULT_MAX_BYTES
                limited = make_limited_stream(cast(io.BufferedIOBase, raw), limit)
                try:
                    read = limited.read()
                finally:
                    try:
                        limited.close()
                    except Exception:
                        pass
                return ByteSource(data=read, origin=None, size=file_size)
        except Exception as exc:
            log.warning("Failed to open stream for %s: %s", getattr(item, "path", None), exc)
            return ByteSource(data=None, origin=None, size=file_size)

    if reopenable:
        max_bytes = decode_cfg.max_bytes_per_file
        try:
            reopened, size = read_file_prefix(reopenable, max_bytes, file_size=file_size)
        except OSError as exc:
            log.warning("Failed to reopen %s: %s", reopenable, exc)
        else:
            return ByteSource(data=reopened, origin=reopenable, size=size)

    return ByteSource(data=None, origin=None, size=file_size)


# --- Public entrypoint the pipeline can call ---

def list_records_for_file(
    *,
    text: str,
    rel_path: str,
    config: ConfigForRecords,
    context: RepoContext | None,
    encoding: str,
    had_replacement: bool,
    file_bytes: int | None = None,
    truncated_bytes: int | None = None,
    source_url: str | None = None,
    source_domain: str | None = None,
) -> list[Record]:
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
    sanitized_url, derived_domain = _normalize_source_ref(source_url, source_domain)
    ctx_with_source = _augment_context_with_source(context, sanitized_url, derived_domain)
    record_ctx = _build_record_context(cfg, ctx_with_source)
    return list(
        iter_records_for_file(
            text=text,
            rel_path=rel_path,
            record_ctx=record_ctx,
            context=ctx_with_source,
            encoding=encoding,
            had_replacement=had_replacement,
            file_bytes=file_bytes,
            truncated_bytes=truncated_bytes,
            source_url=sanitized_url,
            source_domain=derived_domain,
            extractors=cfg.pipeline.extractors,
        )
    )


def iter_records_for_file(
    *,
    text: str,
    rel_path: str,
    record_ctx: RecordBuilderContext,
    context: RepoContext | None,
    encoding: str,
    had_replacement: bool,
    file_bytes: int | None = None,
    truncated_bytes: int | None = None,
    source_url: str | None = None,
    source_domain: str | None = None,
    extractors: Sequence[Any] = (),
) -> Iterator[Record]:
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
    extractor_recs: list[dict[str, object]] = []
    context_meta = record_ctx.metadata_seed
    extra_meta = dict(context_meta) if context_meta else None
    file_nlines = 0 if text == "" else text.count("\n") + 1
    if extractors:
        for extractor in extractors:
            try:
                out = extractor.extract(text=text, path=rel_path, context=context)
                if not out:
                    continue
                for rec in out:
                    if not isinstance(rec, Mapping):
                        log.warning(
                            "Extractor %s produced non-mapping for %s; skipping",
                            getattr(extractor, "name", extractor),
                            rel_path,
                        )
                        continue
                    try:
                        extractor_recs.append(dict(rec))
                    except Exception as exc:
                        log.warning(
                            "Extractor %s record coercion failed for %s: %s",
                            getattr(extractor, "name", extractor),
                            rel_path,
                            exc,
                        )
            except Exception as exc:
                log.warning(
                    "Extractor %s failed for %s: %s",
                    getattr(extractor, "name", extractor),
                    rel_path,
                    exc,
                )
                continue

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
            extra_meta=extra_meta,
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
    context: RepoContext | None,
    file_size: int | None = None,
    source_url: str | None = None,
    source_domain: str | None = None,
) -> Iterator[Record]:
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
    sanitized_url, derived_domain = _normalize_source_ref(source_url, source_domain)
    ctx_with_source = _augment_context_with_source(context, sanitized_url, derived_domain)
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
        source_url=sanitized_url,
        source_domain=derived_domain,
        file_size=file_size,
    )


def iter_records_from_bytes_with_plan(
    data: bytes,
    rel_path: str,
    *,
    plan: Any,
    context: RepoContext | None = None,
    file_size: int | None = None,
    source_url: str | None = None,
    source_domain: str | None = None,
) -> Iterator[Record]:
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
    context: RepoContext | None,
) -> list[Record]:
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
    bytes_handlers: Sequence[tuple[Sniff, BytesHandler]],
    extractors: Sequence[Any],
    context: RepoContext | None,
    chunk_policy: ChunkPolicy,
    file_size: int | None = None,
    source_url: str | None = None,
    source_domain: str | None = None,
) -> Iterator[Record]:
    """Build records from bytes by applying sniffers, decoding, and chunking.

    The caller is responsible for deciding streaming versus buffered handling
    and constructing the RecordBuilderContext along with handler lists.

    Bytes handlers follow explicit semantics:
    - Return None to fall back to decode/chunk processing.
    - Return any iterable (including empty) to short-circuit downstream decode.
    - Raise UnsupportedBinary to skip the file entirely.

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
        try:
            should_handle = sniff(data, rel_path)
        except Exception as exc:
            log.warning(
                "Sniffer %s failed for %s: %s",
                getattr(sniff, "__name__", sniff),
                rel_path,
                exc,
            )
            continue
        if should_handle:
            try:
                records = handler(data, rel_path, context, chunk_policy)
            except UnsupportedBinary as e:
                log.info("Skipping unsupported binary for %s: %s", rel_path, e)
                return
            except Exception as exc:
                log.warning(
                    "Bytes handler %s failed for %s: %s",
                    getattr(handler, "__name__", handler),
                    rel_path,
                    exc,
                )
                continue
            if records is not None:
                try:
                    yield from records
                except Exception as exc:
                    log.warning(
                        "Bytes handler %s iteration failed for %s: %s",
                        getattr(handler, "__name__", handler),
                        rel_path,
                        exc,
                    )
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
    context: RepoContext | None,
    streaming_extractor: StreamingExtractor | None = None,
) -> Iterator[Record]:
    """Yield records for a FileItem using streaming or buffered decoding.

    Args:
        item (FileItem): File descriptor that may be streamable.
        config (ConfigForRecords): File processing configuration.
        context (RepoContext | None): Repository context for metadata.
        streaming_extractor (StreamingExtractor | None): Optional extractor
            that consumes a limited stream before buffering.

    Yields:
        Dict[str, object]: Chunk or extractor record dictionaries.
    """
    cfg = config
    origin_url_raw: str | None = None
    origin_domain_raw: str | None = None
    if getattr(item, "origin_path", None):
        try:
            parsed = urlparse(str(item.origin_path))
            if parsed.scheme in {"http", "https"}:
                origin_url_raw = str(item.origin_path)
                origin_domain_raw = parsed.hostname
        except Exception:
            origin_url_raw = None
            origin_domain_raw = None

    source_url, source_domain = _normalize_source_ref(origin_url_raw, origin_domain_raw)
    ctx_with_source = _augment_context_with_source(context, source_url, source_domain)

    if streaming_extractor is not None and getattr(item, "data", None) is None:
        opener = getattr(item, "open_stream", None)
        if _can_open_stream(item) and callable(opener):
            max_bytes = cfg.decode.max_bytes_per_file
            if max_bytes is None:
                max_bytes = _OPEN_STREAM_DEFAULT_MAX_BYTES
            try:
                with closing(opener()) as raw:
                    stream = make_limited_stream(raw, max_bytes)
                    rows = streaming_extractor.extract_stream(
                        stream=stream,
                        path=item.path,
                        context=ctx_with_source,
                    )
                    if rows is not None:
                        try:
                            yield from rows
                            return
                        finally:
                            try:
                                stream.close()
                            except Exception:
                                pass
            except Exception as exc:
                log.warning(
                    "Streaming extractor failed for %s; falling back to buffered decode (%s)",
                    item.path,
                    exc,
                )

    resolved = resolve_bytes_from_file_item(item, cfg.decode)
    if resolved.data is None:
        log.warning("Skipping %s: missing data and cannot reopen origin", item.path)
        return

    yield from iter_records_from_bytes(
        resolved.data,
        item.path,
        config=cfg,
        context=ctx_with_source,
        file_size=resolved.size,
        source_url=source_url,
        source_domain=source_domain,
    )


class DefaultExtractor(FileExtractor):
    """Adapter that reuses built-in decoding and record generation.

    Attributes:
        streaming_extractor (StreamingExtractor | None): Optional streaming
            extractor used before buffered decoding.
    """

    def __init__(self, streaming_extractor: StreamingExtractor | None = None) -> None:
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
        context: RepoContext | None = None,
    ) -> Iterable[Record]:
        """Extract records for a file item using the configured pipeline.

        Args:
            item (FileItem): File descriptor to process.
            config (ConfigForRecords): File processing configuration.
            context (RepoContext | None): Repository context for metadata.

        Returns:
            Iterable[Record]: Iterator over record mappings.
        """
        return iter_records_from_file_item(
            item,
            config=config,
            context=context,
            streaming_extractor=self.streaming_extractor,
        )
