# evtxio.py
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from io import BytesIO
from typing import Dict, Iterable, Optional, Any

from Evtx.Evtx import Evtx  # python-evtx

from .chunk import ChunkPolicy
from .interfaces import RepoContext, Record
from .records import build_record

__all__ = ["sniff_evtx", "handle_evtx"]


# --- Sniffing primitives -----------------------------------------------------

_EVTX_FILE_MAGIC = b"ElfFile"   # EVTX file header signature
# EVTX chunks also carry their own signature; useful when file headers are damaged.
_EVTX_CHUNK_MAGIC = b"ElfChnk"  # often found at 64 KiB boundaries

# Cap how much we scan looking for a chunk signature when sniffing:
_SNIFF_SCAN_LIMIT = 1_048_576  # 1 MiB


def sniff_evtx(data: bytes, rel: str) -> bool:
    """
    Lightweight EVTX sniff that handles common and slightly-damaged cases:

    1) .evtx extension
    2) EVTX file header magic 'ElfFile' at offset 0
    3) Presence of a chunk signature 'ElfChnk' in the first ~1 MiB
       (helps with carved/truncated headers)

    Keep this fast; deeper verification belongs in the QC stage.
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


def _put_coalescing(m: Dict[str, Any], k: str, v: str) -> None:
    """Accumulate duplicate Data Name= keys as lists, preserving order."""
    if k in m:
        if isinstance(m[k], list):
            m[k].append(v)
        else:
            m[k] = [m[k], v]
    else:
        m[k] = v


def _parse_event_xml(xml_text: str) -> Dict[str, Any]:
    """
    Parse Windows Event XML into a compact dict.

    - Pull core fields from <System>
    - Convert <EventData><Data Name="...">...</Data> into a map,
      coalescing duplicate names into lists and preserving unnamed Data.
    - Leave room for audit by embedding the original XML at the call site.
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

    out: Dict[str, Any] = {
        "provider": provider,
        "event_id": _pick(system, "EventID"),
        "level": _pick(system, "Level"),
        "channel": _pick(system, "Channel"),
        "computer": _pick(system, "Computer"),
        "event_record_id": _pick(system, "EventRecordID"),
        "timestamp": _pick(system, "TimeCreated", "SystemTime"),
    }

    data_map: Dict[str, Any] = {}
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
    payload = _parse_event_xml(xml_text)
    payload["raw_xml"] = xml_text
    return json.dumps(payload, ensure_ascii=False)


# --- Optional recovery (EVTXtract) -------------------------------------------

def _recover_with_evtxtract(data: bytes) -> Iterable[str]:
    """
    Best-effort recovery that shells out to EVTXtract if available.

    Returns XML fragments (strings). If EVTXtract isn't installed or fails,
    yields nothing.
    """
    # Resolve binary (either "evtxtract" or "python -m evtxtract")
    candidates = [["evtxtract"], ["python", "-m", "evtxtract"], ["python3", "-m", "evtxtract"]]
    with tempfile.NamedTemporaryFile(prefix="rc_evtx_", suffix=".bin", delete=True) as tf:
        tf.write(data)
        tf.flush()
        for cmd in candidates:
            try:
                proc = subprocess.run(
                    cmd + [tf.name],
                    check=False,
                    capture_output=True,
                    text=True,
                )
            except Exception:
                continue
            if proc.returncode != 0 or not proc.stdout:
                continue
            # Heuristic: EVTXtract prints <Event ...>...</Event> fragments; extract them.
            # Keep it robust to whitespace and newlines.
            return re.findall(r"<Event[^>]*>.*?</Event>", proc.stdout, flags=re.DOTALL)
    return ()


# --- Public handler -----------------------------------------------------------

def _env_wants_recovery() -> bool:
    return os.getenv("REPOCAPSULE_EVTX_RECOVER", "").strip().lower() in {"1", "true", "yes"}


def handle_evtx(
    data: bytes,
    rel: str,
    ctx: Optional[RepoContext],
    policy: Optional[ChunkPolicy],
    *,
    allow_recovery: Optional[bool] = None,
) -> Iterable[Record]:
    """
    Stream one JSONL record per event.

    Normal path: python-evtx -> record.xml() -> normalize to JSON (plus raw_xml).
    Optional fallback: pass allow_recovery=True (or set REPOCAPSULE_EVTX_RECOVER=1)
    to try EVTXtract recovery when python-evtx fails or yields zero events.
    """
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
                    extra_meta={
                        "kind": "evtx",
                        "provider": j.get("provider"),
                        "event_id": j.get("event_id"),
                        "level": j.get("level"),
                        "channel": j.get("channel"),
                        "computer": j.get("computer"),
                        "event_record_id": j.get("event_record_id"),
                        "timestamp": j.get("timestamp"),
                        "recovered": False,
                    },
                )
    except Exception:
        parsed_any = False

    # Recovery path
    if not parsed_any and use_recovery:
        for xml_text in _recover_with_evtxtract(data):
            j = _parse_event_xml(xml_text)
            yield build_record(
                text=_event_json_with_raw(xml_text),
                rel_path=rel,
                repo_full_name=(ctx.repo_full_name if ctx else None),
                repo_url=(ctx.repo_url if ctx else None),
                license_id=(ctx.license_id if ctx else None),
                lang="WindowsEventLog",
                extra_meta={
                    "kind": "evtx",
                    "provider": j.get("provider"),
                    "event_id": j.get("event_id"),
                    "level": j.get("level"),
                    "channel": j.get("channel"),
                    "computer": j.get("computer"),
                    "event_record_id": j.get("event_record_id"),
                    "timestamp": j.get("timestamp"),
                    "recovered": True,
                    "recovery_tool": "EVTXtract",
                },
            )
