# jsonl_source.py
# SPDX-License-Identifier: MIT

"""JSONL source that re-chunks text records into file-like items."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Dict, Any, Mapping
import json
import gzip

from ..core.interfaces import Source, FileItem, RepoContext
from ..core.log import get_logger
from ..core.records import check_record_schema, STANDARD_META_FIELDS

log = get_logger(__name__)

__all__ = ["JSONLTextSource"]

_MAX_INVALID_JSON_WARNINGS = 5


@dataclass
class JSONLTextSource(Source):
    """Treats JSONL lines as pseudo-files for re-chunking.

    Each line should be a JSON object. Lines that are invalid JSON, not
    dictionaries, or missing the configured text field are skipped to
    keep processing forgiving for heterogeneous streams.

    Attributes:
        paths (Sequence[Path]): JSONL or JSONL.GZ files to read.
        context (RepoContext | None): Optional repository context.
        text_key (str): Field name containing text content to emit.
        check_schema (bool): Whether to run schema checks on Sievio records.
    """

    paths: Sequence[Path]
    context: Optional[RepoContext] = None
    text_key: str = "text"
    check_schema: bool = True

    def iter_files(self) -> Iterable[FileItem]:
        """Yields FileItems constructed from JSONL records."""

        for path in self.paths:
            opener = _open_jsonl_gz if "".join(path.suffixes[-2:]).lower() == ".jsonl.gz" else _open_jsonl
            try:
                with opener(path) as fp:
                    checked_schema = False
                    invalid_json_lines = 0
                    non_dict_lines = 0
                    missing_text_lines = 0
                    emitted_lines = 0
                    for lineno, line in enumerate(fp, start=1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except Exception as exc:
                            invalid_json_lines += 1
                            if invalid_json_lines <= _MAX_INVALID_JSON_WARNINGS:
                                log.warning("Skipping invalid JSON at %s:#%d: %s", path, lineno, exc)
                                if invalid_json_lines == _MAX_INVALID_JSON_WARNINGS:
                                    log.debug("Suppressing further invalid JSON warnings for %s", path)
                            continue
                        if not isinstance(record, dict):
                            non_dict_lines += 1
                            continue
                        if self.check_schema and not checked_schema and _should_check_schema(record):
                            check_record_schema(record, log)
                            checked_schema = True
                        text = _extract_text(record, self.text_key)
                        if text is None:
                            # Skip records missing the configured text field.
                            missing_text_lines += 1
                            continue
                        rel = _derive_path(record, path.name, lineno)
                        data = text.encode("utf-8")
                        emitted_lines += 1
                        yield FileItem(path=rel, data=data, size=len(data))
                    if any((invalid_json_lines, non_dict_lines, missing_text_lines)):
                        log.info(
                            "Finished %s: emitted=%d skipped_invalid=%d skipped_nondict=%d missing_text=%d",
                            path,
                            emitted_lines,
                            invalid_json_lines,
                            non_dict_lines,
                            missing_text_lines,
                        )
                    else:
                        log.debug("Finished %s: emitted=%d", path, emitted_lines)
            except FileNotFoundError:
                log.warning("JSONL file not found: %s", path)
            except Exception as exc:
                log.warning("Failed to read JSONL file %s: %s", path, exc)


def _extract_text(record: Dict[str, Any], key: str) -> Optional[str]:
    """Extracts a string value from the given record by key."""

    val = record.get(key)
    if isinstance(val, str):
        return val
    return None


def _derive_path(record: Dict[str, Any], fallback_name: str, lineno: int) -> str:
    """Derives a path-like label for a record, falling back to line number."""

    meta = record.get("meta")
    if isinstance(meta, dict):
        path_val = meta.get("path")
        if isinstance(path_val, str) and path_val:
            sanitized = _sanitize_path_label(path_val)
            if sanitized:
                return sanitized
    base_stem = Path(fallback_name).stem
    if base_stem.endswith(".jsonl"):
        base_stem = Path(base_stem).stem
    if not base_stem:
        base_stem = "jsonl"
    return f"{base_stem}/line_{lineno}.txt"


def _sanitize_path_label(path_val: str) -> str:
    """Normalize a meta-supplied path to a safe, path-like label."""

    cleaned = path_val.replace("\\", "/").lstrip("/")
    parts = []
    for part in cleaned.split("/"):
        if not part or part == "." or part == "..":
            continue
        parts.append(part)
    return "/".join(parts) if parts else ""


def _should_check_schema(record: Mapping[str, Any]) -> bool:
    """Heuristic to decide whether to run schema checks on a record."""

    meta = record.get("meta")
    if not isinstance(meta, Mapping):
        return False
    if "schema_version" in meta:
        return True
    return any(key in STANDARD_META_FIELDS for key in meta)


def _open_jsonl(path: Path):
    """Opens an uncompressed JSONL file for reading."""

    return open(path, "r", encoding="utf-8")


def _open_jsonl_gz(path: Path):
    """Opens a gzip-compressed JSONL file for reading."""

    return gzip.open(path, "rt", encoding="utf-8")
