# pdfio.py
# SPDX-License-Identifier: MIT

from __future__ import annotations
from io import BytesIO
from typing import Optional, List, Dict, Any, Iterable
from datetime import datetime
from pypdf import PdfReader

from .chunk import ChunkPolicy, chunk_text
from .records import build_record
from .interfaces import RepoContext, Record

# Re-export for DI registration
__all__ = ["extract_pdf_records", "sniff_pdf", "handle_pdf"]

def sniff_pdf(data: bytes, rel: str) -> bool:
    """Sniff for PDF files by extension or magic bytes."""
    return rel.lower().endswith(".pdf") or data.startswith(b"%PDF-")

def handle_pdf(
    data: bytes,
    rel: str,
    ctx: Optional[RepoContext],
    policy: Optional[ChunkPolicy],
) -> Optional[Iterable[Record]]:
    """
    BytesHandler implementation for PDF files.
    Wraps extract_pdf_records to match the BytesHandler protocol.
    """
    return extract_pdf_records(
        data,
        rel_path=rel,
        policy=policy,
        repo_full_name=(ctx.repo_full_name if ctx else None),
        repo_url=(ctx.repo_url if ctx else None),
        license_id=(ctx.license_id if ctx else None),
    )



def _iso8601(v: Any) -> str | None:
    """Best-effort ISO-8601 for pypdf date-like fields (datetime or PDF 'D:YYYY...' strings)."""
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
    """Pick a human-friendly value from an XMP language alternative dict."""
    if isinstance(xmp_lang_alt, dict):
        return xmp_lang_alt.get("x-default") or next(iter(xmp_lang_alt.values()), None)
    return str(xmp_lang_alt) if xmp_lang_alt else None

def _collect_pdf_metadata(reader: PdfReader) -> Dict[str, Any]:
    """Collect classic Info dict + a small XMP subset; drop empty/None values."""
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
    """Turn a PDF (bytes) into RepoCapsule JSONL records with metadata."""
    policy = policy or ChunkPolicy(mode="doc")
    reader = PdfReader(BytesIO(data))

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
                    extra_meta={"kind": "pdf", "page": i, "n_pages": n, "pdf_meta": pdf_meta or None},
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
                    extra_meta={"kind": "pdf", "pdf_meta": pdf_meta or None},
                    tokens=ch.get("n_tokens"),
                )
            )

    return records
