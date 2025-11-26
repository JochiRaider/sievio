# jsonl_source.py
# SPDX-License-Identifier: MIT

"""JSONL source that re-chunks text records into file-like items."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Dict, Any
import json
import gzip

from ..core.interfaces import Source, FileItem, RepoContext
from ..core.log import get_logger

log = get_logger(__name__)

__all__ = ["JSONLTextSource"]


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
    """

    paths: Sequence[Path]
    context: Optional[RepoContext] = None
    text_key: str = "text"

    def iter_files(self) -> Iterable[FileItem]:
        """Yields FileItems constructed from JSONL records."""

        for path in self.paths:
            opener = _open_jsonl_gz if "".join(path.suffixes[-2:]).lower() == ".jsonl.gz" else _open_jsonl
            try:
                with opener(path) as fp:
                    for lineno, line in enumerate(fp, start=1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except Exception as exc:
                            log.warning("Skipping invalid JSON at %s:#%d: %s", path, lineno, exc)
                            continue
                        text = _extract_text(record, self.text_key)
                        if text is None:
                            # Skip records missing the configured text field.
                            continue
                        rel = _derive_path(record, path.name, lineno)
                        data = text.encode("utf-8")
                        yield FileItem(path=rel, data=data, size=len(data))
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
            return path_val
    return f"{fallback_name}:#{lineno}"


def _open_jsonl(path: Path):
    """Opens an uncompressed JSONL file for reading."""

    return open(path, "r", encoding="utf-8")


def _open_jsonl_gz(path: Path):
    """Opens a gzip-compressed JSONL file for reading."""

    return gzip.open(path, "rt", encoding="utf-8")
