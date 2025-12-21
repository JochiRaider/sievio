# evtxio.py
# SPDX-License-Identifier: MIT
"""EVTX parser utilities for producing normalized JSON event records."""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from io import BytesIO
from typing import Any, Protocol, cast

from Evtx.Evtx import Evtx  # python-evtx

from ..core.chunk import ChunkPolicy
from ..core.interfaces import Record, RepoContext
from ..core.records import build_record

__all__ = ["sniff_evtx", "handle_evtx"]


# --- Sniffing primitives -----------------------------------------------------

_EVTX_FILE_MAGIC = b"ElfFile"   # EVTX file header signature
# EVTX chunks also carry their own signature; useful when file headers are damaged.
_EVTX_CHUNK_MAGIC = b"ElfChnk"  # often found at 64 KiB boundaries

# Cap how much we scan looking for a chunk signature when sniffing:
_SNIFF_SCAN_LIMIT = 1_048_576  # 1 MiB


def sniff_evtx(data: bytes, rel: str) -> bool:
    """Detect whether a blob is likely an EVTX file.

    Checks common signatures and extensions, including chunk signatures
    that help with damaged headers. Keep this lightweight; deeper
    verification happens during QC.

    Args:
        data (bytes): Raw bytes to inspect.
        rel (str): Relative path or name for extension checks.

    Returns:
        bool: True if the data appears to be EVTX.
    """
    name = (rel or "").lower()
    if name.endswith(".evtx"):
        return True
    if data.startswith(_EVTX_FILE_MAGIC):
        return True
    if _EVTX_CHUNK_MAGIC in data[:_SNIFF_SCAN_LIMIT]:
        return True
    return False


# --- XML -> compact JSON normalizer ------------------------------------------

def _pick(elem: ET.Element | None, tag: str, attr: str | None = None) -> str | None:
    """Find first matching descendant ignoring namespaces; return text or attribute."""
    if elem is None:
        return None
    found = elem.find(f".//{{*}}{tag}")
    if found is None:
        return None
    if attr:
        return found.attrib.get(attr)
    return (found.text or "").strip() or None


def _put_coalescing(m: dict[str, Any], k: str, v: str) -> None:
    """Accumulate duplicate Data Name= keys as lists, preserving order."""
    if k in m:
        if isinstance(m[k], list):
            m[k].append(v)
        else:
            m[k] = [m[k], v]
    else:
        m[k] = v


def _parse_event_xml(xml_text: str) -> dict[str, Any]:
    """Parse Windows Event XML into a compact dict.

    Pulls core fields from System, maps EventData entries, and preserves
    unnamed Data values. The caller is responsible for attaching the raw
    XML when needed for audit.

    Args:
        xml_text (str): XML string for a single event.

    Returns:
        Dict[str, Any]: Normalized event details.
    """
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return {"raw_xml_unparsed": True}

    system = root.find(".//{*}System")
    eventdata = root.find(".//{*}EventData")

    provider = None
    if system is not None:
        prov = system.find(".//{*}Provider")
        if prov is not None:
            provider = prov.attrib.get("Name") or prov.attrib.get("Guid")

    out: dict[str, Any] = {
        "provider": provider,
        "event_id": _pick(system, "EventID"),
        "level": _pick(system, "Level"),
        "channel": _pick(system, "Channel"),
        "computer": _pick(system, "Computer"),
        "event_record_id": _pick(system, "EventRecordID"),
        "timestamp": _pick(system, "TimeCreated", "SystemTime"),
    }

    data_map: dict[str, Any] = {}
    unnamed: list[str] = []
    if eventdata is not None:
        for d in list(eventdata):
            # namespace-agnostic <Data>
            if d.tag.endswith("Data"):
                name = d.attrib.get("Name")
                val = (d.text or "").strip()
                if name:
                    _put_coalescing(data_map, name, val)
                else:
                    unnamed.append(val)

    if data_map:
        out["event_data"] = data_map
    if unnamed:
        out.setdefault("event_data_extra", {})["_unnamed"] = unnamed
    return out


def _event_json_with_raw(xml_text: str) -> str:
    """Return a JSON string containing parsed fields and the raw XML."""
    payload = _parse_event_xml(xml_text)
    payload["raw_xml"] = xml_text
    return json.dumps(payload, ensure_ascii=False)


# --- Optional recovery (EVTXtract) -------------------------------------------

def _scan_evtx_file_magic_offsets(data: bytes, max_hits: int = 16) -> list[int]:
    """Return offsets where an EVTX file header magic is found.

    We cap hits to avoid pathological scans on very large blobs.
    """
    offsets: list[int] = []
    start = 0
    magic = _EVTX_FILE_MAGIC
    while len(offsets) < max_hits:
        idx = data.find(magic, start)
        if idx == -1:
            break
        offsets.append(idx)
        start = idx + 1
    return offsets


def _recover_best_effort_with_python_evtx(data: bytes) -> Iterable[str]:
    """Best-effort recovery using python-evtx only.

    Strategy:
    - Scan for EVTX file header magic ("ElfFile") within the blob.
    - For each candidate offset, attempt to open an Evtx view on the
      slice starting at that offset.
    - Yield record.xml() for any successfully parsed records.

    This does not attempt heavy-duty carving of individual chunks; it is
    a lightweight, dependency-free salvage path for blobs that contain
    embedded or slightly damaged EVTX files.

    Args:
        data (bytes): Raw bytes that may contain EVTX content.

    Returns:
        Iterable[str]: Iterator over recovered XML strings.
    """
    # First, if the blob itself starts with an EVTX header, we've already
    # tried that in the main handler and failed. Focus on secondary headers.
    offsets = _scan_evtx_file_magic_offsets(data)

    # If there is exactly one hit at offset 0, there's likely nothing more
    # to salvage that the primary parse didn't already attempt.
    if offsets == [0]:
        return ()

    def _iter() -> Iterable[str]:
        for off in offsets:
            # Skip the primary header at offset 0; we already tried it.
            if off == 0:
                continue
            try:
                with Evtx(BytesIO(data[off:])) as log:
                    for rec in log.records():
                        yield rec.xml()
            except Exception:
                # Any failures here are expected; just move on to the next
                # candidate offset.
                continue

    return _iter()



# --- Public handler -----------------------------------------------------------

def _env_wants_recovery() -> bool:
    """Return whether recovery is enabled via environment variable."""
    return os.getenv("SIEVIO_EVTX_RECOVER", "").strip().lower() in {"1", "true", "yes"}


def handle_evtx(
    data: bytes,
    rel: str,
    ctx: RepoContext | None,
    policy: ChunkPolicy | None,
    *,
    allow_recovery: bool | None = None,
) -> Iterable[Record]:
    """Stream one normalized JSON record per EVTX event.

    Normal path: python-evtx parses records to XML, which is normalized
    to JSON while preserving raw_xml. Parsing is CPU-bound; large EVTX
    sets benefit from process executors.

    The JSON payload is intentionally lossy: nested Data elements are
    flattened into event_data / event_data_extra and ordering may not
    match the source XML. raw_xml is retained for fidelity.

    Optional fallback: enable recovery (allow_recovery=True or
    SIEVIO_EVTX_RECOVER=1) to scan for embedded EVTX headers and
    attempt parsing from those offsets when primary parsing yields
    nothing.

    Args:
        data (bytes): Raw EVTX blob.
        rel (str): Relative path used in emitted records.
        ctx (RepoContext | None): Context for repository metadata.
        policy (ChunkPolicy | None): Chunking policy (unused here).
        allow_recovery (bool | None): Override environment-driven
            recovery behavior.

    Yields:
        Record: One JSONL record per parsed event.
    """
    context_meta = (ctx.as_meta_seed() or None) if ctx else None

    def _with_context(extra: dict[str, Any]) -> dict[str, Any]:
        if context_meta:
            merged = dict(context_meta)
            merged.update(extra)
            return merged
        return extra

    use_recovery = allow_recovery
    if use_recovery is None:
        use_recovery = _env_wants_recovery()

    # Primary parse
    parsed_any = False
    try:
        with Evtx(BytesIO(data)) as log:
            for rec in log.records():
                xml_text = rec.xml()  # ASCII XML
                j = _parse_event_xml(xml_text)
                parsed_any = True

                yield build_record(
                    text=_event_json_with_raw(xml_text),
                    rel_path=rel,
                    repo_full_name=(ctx.repo_full_name if ctx else None),
                    repo_url=(ctx.repo_url if ctx else None),
                    license_id=(ctx.license_id if ctx else None),
                    lang="WindowsEventLog",
                    extra_meta=_with_context({
                        "subkind": "evtx",
                        "provider": j.get("provider"),
                        "event_id": j.get("event_id"),
                        "level": j.get("level"),
                        "channel": j.get("channel"),
                        "computer": j.get("computer"),
                        "event_record_id": j.get("event_record_id"),
                        "timestamp": j.get("timestamp"),
                        "recovered": False,
                    }),
                )
    except Exception:
        parsed_any = False

    # Recovery path
    if not parsed_any and use_recovery:
        for xml_text in _recover_best_effort_with_python_evtx(data):
            j = _parse_event_xml(xml_text)
            yield build_record(
                text=_event_json_with_raw(xml_text),
                rel_path=rel,
                repo_full_name=(ctx.repo_full_name if ctx else None),
                repo_url=(ctx.repo_url if ctx else None),
                license_id=(ctx.license_id if ctx else None),
                lang="WindowsEventLog",
                extra_meta=_with_context({
                    "subkind": "evtx",
                    "provider": j.get("provider"),
                    "event_id": j.get("event_id"),
                    "level": j.get("level"),
                    "channel": j.get("channel"),
                    "computer": j.get("computer"),
                    "event_record_id": j.get("event_record_id"),
                    "timestamp": j.get("timestamp"),
                    "recovered": True,
                    "recovery_strategy": "python-evtx-carve",
                }),
            )


class _BytesHandlerAttrs(Protocol):
    cpu_intensive: bool
    preferred_executor: str


_handler = cast(_BytesHandlerAttrs, handle_evtx)
_handler.cpu_intensive = True
_handler.preferred_executor = "process"

# Register default bytes handler
try:
    from ..core.registries import bytes_handler_registry

    bytes_handler_registry.register(sniff_evtx, handle_evtx)
except Exception:
    pass
