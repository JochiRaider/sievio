# jsonl_source.py
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Dict, Any
import json
import gzip

from .interfaces import Source, FileItem, RepoContext
from .log import get_logger

log = get_logger(__name__)

__all__ = ["JSONLTextSource"]


@dataclass
class JSONLTextSource(Source):
    """
    Treat existing JSONL lines as pseudo-files for re-chunking.
    """

    paths: Sequence[Path]
    context: Optional[RepoContext] = None
    text_key: str = "text"

    def iter_files(self) -> Iterable[FileItem]:
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
                            continue
                        rel = _derive_path(record, path.name, lineno)
                        data = text.encode("utf-8")
                        yield FileItem(path=rel, data=data, size=len(data))
            except FileNotFoundError:
                log.warning("JSONL file not found: %s", path)
            except Exception as exc:
                log.warning("Failed to read JSONL file %s: %s", path, exc)


def _extract_text(record: Dict[str, Any], key: str) -> Optional[str]:
    val = record.get(key)
    if isinstance(val, str):
        return val
    return None


def _derive_path(record: Dict[str, Any], fallback_name: str, lineno: int) -> str:
    meta = record.get("meta")
    if isinstance(meta, dict):
        path_val = meta.get("path")
        if isinstance(path_val, str) and path_val:
            return path_val
    return f"{fallback_name}:#{lineno}"


def _open_jsonl(path: Path):
    return open(path, "r", encoding="utf-8")


def _open_jsonl_gz(path: Path):
    return gzip.open(path, "rt", encoding="utf-8")
