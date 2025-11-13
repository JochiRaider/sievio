# convert.py
# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Iterable

import logging

from .config import RepocapsuleConfig
from .chunk import ChunkPolicy, chunk_text
from .decode import decode_bytes
from .factories import UnsupportedBinary
from .interfaces import RepoContext, Record
from .records import build_record

__all__ = [
    "make_records_for_file",
    "make_records_from_bytes",
]

log = logging.getLogger(__name__)

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
    config: RepocapsuleConfig,
    context: Optional[RepoContext],
    encoding: str,
    had_replacement: bool,
) -> List[Dict[str, object]]:
    cfg = config
    extractor_recs: List[Dict[str, object]] = []
    if cfg.pipeline.extractors:
        for extractor in cfg.pipeline.extractors:
            try:
                out = extractor.extract(text=text, path=rel_path, context=context)
            except Exception as exc:
                log.warning("Extractor %s failed for %s: %s", getattr(extractor, "name", extractor), rel_path, exc)
                continue
            if not out:
                continue
            extractor_recs.extend(dict(rec) for rec in out)

    mode, fmt = _infer_mode_and_fmt(rel_path)
    chunks = chunk_text(
        text,
        mode=mode,
        fmt=fmt,
        policy=cfg.chunk.policy,
        tokenizer_name=cfg.chunk.tokenizer_name,
    )
    records: List[Dict[str, object]] = []
    n = len(chunks)
    for idx, chunk in enumerate(chunks, start=1):
        records.append(
            build_record(
                text=chunk.get("text", ""),
                rel_path=rel_path,
                repo_full_name=(context.repo_full_name if context else None),
                repo_url=(context.repo_url if context else None),
                license_id=(context.license_id if context else None),
                encoding=encoding,
                had_replacement=had_replacement,
                chunk_id=idx,
                n_chunks=n,
                lang=chunk.get("lang") if cfg.chunk.attach_language_metadata else None,
                tokens=chunk.get("n_tokens"),
            )
        )
    if extractor_recs:
        records.extend(extractor_recs)
    return records

def make_records_from_bytes(
    data: bytes,
    rel_path: str,
    *,
    config: RepocapsuleConfig,
    context: Optional[RepoContext],
) -> List[Dict[str, object]]:
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
                # preserve streaming if handler yields
                return list(records) if isinstance(records, list) else list(records)

    max_bytes = cfg.decode.max_bytes_per_file
    if max_bytes is not None and len(data) > max_bytes:
        data = data[:max_bytes]

    dec = decode_bytes(
        data,
        normalize=cfg.decode.normalize,
        strip_controls=cfg.decode.strip_controls,
        fix_mojibake=cfg.decode.fix_mojibake,
    )
    return make_records_for_file(
        text=dec.text,
        rel_path=rel_path,
        config=cfg,
        context=context,
        encoding=dec.encoding,
        had_replacement=dec.had_replacement,
    )
