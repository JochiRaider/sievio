# pdfio.py
# SPDX-License-Identifier: MIT

"""PDF helpers for sniffing, parsing, and emitting Sievio records."""

from __future__ import annotations
from io import BytesIO
from typing import Optional, List, Dict, Any, Iterable
from datetime import datetime
from pypdf import PdfReader

from ..core.chunk import ChunkPolicy, chunk_text
from ..core.records import build_record
from ..core.interfaces import RepoContext, Record

# Re-export for DI registration
__all__ = ["extract_pdf_records", "sniff_pdf", "handle_pdf"]

def sniff_pdf(data: bytes, rel: str) -> bool:
    """Detects whether the payload looks like a PDF file.

    Args:
        data (bytes): Raw file bytes.
        rel (str): Relative path used for extension checks.

    Returns:
        bool: True if the data appears to be a PDF.
    """
    return rel.lower().endswith(".pdf") or data.startswith(b"%PDF-")

def handle_pdf(
    data: bytes,
    rel: str,
    ctx: Optional[RepoContext],
    policy: Optional[ChunkPolicy],
) -> Optional[Iterable[Record]]:
    """BytesHandler adapter that processes PDF payloads.

    Args:
        data (bytes): Raw PDF bytes.
        rel (str): Relative path for metadata.
        ctx (RepoContext | None): Optional repository context.
        policy (ChunkPolicy | None): Chunking policy; passed through.

    Returns:
        Iterable[Record] | None: Extracted records or None on failure.
    """
    return extract_pdf_records(
        data,
        rel_path=rel,
        policy=policy,
        repo_full_name=(ctx.repo_full_name if ctx else None),
        repo_url=(ctx.repo_url if ctx else None),
        license_id=(ctx.license_id if ctx else None),
    )


handle_pdf.cpu_intensive = True
handle_pdf.preferred_executor = "process"



def _iso8601(v: Any) -> str | None:
    """Formats pypdf date-like fields into ISO-8601 when possible.

    Args:
        v (Any): Source value, typically a datetime or PDF date string.

    Returns:
        str | None: Normalized ISO-8601 string or None if no value.
    """
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.isoformat()
    s = str(v)
    # Keep as-is if we can't confidently parse; callers can post-process further if desired.
    if s.startswith("D:"):
        # Minimal normalization: strip leading 'D:'; return raw string for traceability
        return s.replace("D:", "", 1)
    return s

def _first_lang_value(xmp_lang_alt: Any) -> str | None:
    """Picks a human-friendly value from an XMP language alternative dict.

    Args:
        xmp_lang_alt (Any): XMP lang alternative mapping or value.

    Returns:
        str | None: Preferred localized string, if available.
    """
    if isinstance(xmp_lang_alt, dict):
        return xmp_lang_alt.get("x-default") or next(iter(xmp_lang_alt.values()), None)
    return str(xmp_lang_alt) if xmp_lang_alt else None

def _collect_pdf_metadata(reader: PdfReader) -> Dict[str, Any]:
    """Collects classic Info fields and a subset of XMP metadata.

    Args:
        reader (PdfReader): PDF reader instance to inspect.

    Returns:
        Dict[str, Any]: Metadata dictionary with empty values removed.
    """
    out: Dict[str, Any] = {}

    # Classic PDF metadata (DocumentInformation)
    info = getattr(reader, "metadata", None)
    if info:
        for k in ("title", "author", "subject", "creator", "producer"):
            val = getattr(info, k, None)
            if val:
                out[k] = val
        cd = _iso8601(getattr(info, "creation_date", None))
        md = _iso8601(getattr(info, "modification_date", None))
        if cd:
            out["creation_date"] = cd
        if md:
            out["modification_date"] = md

    # XMP subset (if present)
    xmp = getattr(reader, "xmp_metadata", None)
    if xmp:
        xmp_out: Dict[str, Any] = {}
        title = _first_lang_value(getattr(xmp, "dc_title", None))
        if title:
            xmp_out["dc_title"] = title
        creators = getattr(xmp, "dc_creator", None)
        if creators:
            xmp_out["dc_creator"] = list(creators)
        subjects = getattr(xmp, "dc_subject", None)
        if subjects:
            xmp_out["dc_subject"] = list(subjects)
        xmp_cd = _iso8601(getattr(xmp, "xmp_create_date", None))
        xmp_md = _iso8601(getattr(xmp, "xmp_modify_date", None))
        if xmp_cd:
            xmp_out["xmp_create_date"] = xmp_cd
        if xmp_md:
            xmp_out["xmp_modify_date"] = xmp_md
        producer = getattr(xmp, "pdf_producer", None)
        if producer:
            xmp_out["pdf_producer"] = producer

        if xmp_out:
            out["xmp"] = xmp_out

    return out

def extract_pdf_records(
    data: bytes,
    *,
    rel_path: str,
    policy: Optional[ChunkPolicy] = None,
    repo_full_name: Optional[str] = None,
    repo_url: Optional[str] = None,
    license_id: Optional[str] = None,
    password: Optional[str] = None,
    mode: str = "page",  # "page" => 1 record per page; "chunk" => join+chunk
) -> List[Dict[str, object]]:
    """Converts PDF bytes into Sievio JSONL records with metadata.

    This routine is CPU-bound (pypdf parsing, text extraction, and
    chunking). For large batches, configure the pipeline to use process
    execution so PDF-heavy workloads can be parallelized.

    Args:
        data (bytes): Raw PDF content.
        rel_path (str): Relative path used in record metadata.
        policy (ChunkPolicy | None): Chunking policy for non-page mode.
        repo_full_name (str | None): Repository identifier for metadata.
        repo_url (str | None): Repository URL for metadata.
        license_id (str | None): SPDX license identifier.
        password (str | None): Password for encrypted PDFs.
        mode (str): "page" for one record per page, "chunk" to join and
            chunk the entire document.

    Returns:
        List[Dict[str, object]]: Extracted records with text and metadata.
    """
    policy = policy or ChunkPolicy(mode="doc")
    reader = PdfReader(BytesIO(data))
    ctx = RepoContext(
        repo_full_name=repo_full_name,
        repo_url=repo_url,
        license_id=license_id,
    )
    ctx_seed = ctx.as_meta_seed() or None
    url_hint = ctx_seed.get("url") if ctx_seed else None
    domain_hint = ctx_seed.get("source_domain") if ctx_seed else None

    def _with_context_extra(extra: Dict[str, Any]) -> Dict[str, Any]:
        if ctx_seed:
            merged = dict(ctx_seed)
            merged.update(extra)
            return merged
        return extra

    if reader.is_encrypted:
        if not password:
            return []
        try:
            reader.decrypt(password)
        except Exception:
            return []

    # Try to collect metadata (robust to absence)
    try:
        pdf_meta = _collect_pdf_metadata(reader)
    except Exception:
        pdf_meta = {}

    # Extract text per page
    pages_text: List[str] = []
    for p in reader.pages:
        try:
            txt = p.extract_text() or ""
        except Exception:
            txt = ""
        pages_text.append(txt)
    file_nlines = sum((t.count("\n") + 1 if t else 0) for t in pages_text) if pages_text else 0

    records: List[Dict[str, object]] = []
    if mode == "page":
        n = len(pages_text)
        for i, text in enumerate(pages_text, start=1):
            records.append(
                build_record(
                    text=text,
                    rel_path=rel_path,
                    repo_full_name=repo_full_name,
                    repo_url=repo_url,
                    license_id=license_id,
                    chunk_id=i,
                    n_chunks=n,
                    url=url_hint,
                    source_domain=domain_hint,
                    extra_meta=_with_context_extra({"subkind": "pdf", "page": i, "n_pages": n, "pdf_meta": pdf_meta or None}),
                    file_nlines=file_nlines,
                )
            )
    else:
        all_text = "\n\n".join(pages_text)
        chunks = chunk_text(all_text, mode="doc", fmt="text", policy=policy)
        n = len(chunks)
        for i, ch in enumerate(chunks, start=1):
            records.append(
                build_record(
                    text=ch["text"],
                    rel_path=rel_path,
                    repo_full_name=repo_full_name,
                    repo_url=repo_url,
                    license_id=license_id,
                    chunk_id=i,
                    n_chunks=n,
                    url=url_hint,
                    source_domain=domain_hint,
                    extra_meta=_with_context_extra({"subkind": "pdf", "pdf_meta": pdf_meta or None}),
                    tokens=ch.get("n_tokens"),
                    file_nlines=file_nlines,
                )
            )

    return records

# Register default bytes handler
try:
    from ..core.registries import bytes_handler_registry

    bytes_handler_registry.register(sniff_pdf, handle_pdf)
except Exception:
    pass
