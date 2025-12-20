# jsonl_source.py
# SPDX-License-Identifier: MIT

"""JSONL source that re-chunks text records into file-like items."""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from ..core.interfaces import FileItem, RepoContext, Source
from ..core.log import get_logger
from ..core.records import STANDARD_META_FIELDS, check_record_schema

log = get_logger(__name__)

__all__ = ["JSONLReadPolicy", "JSONLTextSource"]

_MAX_INVALID_JSON_WARNINGS = 5
_DEFAULT_MAX_LINE_CHARS = 2_000_000
_DEFAULT_MAX_TEXT_CHARS = 2_000_000
_MAX_SANITIZED_PATH_LENGTH = 1024
_MAX_PATH_SEGMENT_LENGTH = 255


@dataclass
class JSONLReadPolicy:
    """Limits and decoding controls for JSONL ingestion."""

    max_invalid_json_warnings: int = _MAX_INVALID_JSON_WARNINGS
    max_line_chars: int | None = _DEFAULT_MAX_LINE_CHARS
    max_text_chars: int | None = _DEFAULT_MAX_TEXT_CHARS
    max_file_chars: int | None = None
    decode_errors: str = "strict"
    max_oversized_line_warnings: int = 3


@dataclass
class _LineReadStats:
    oversized_lines: int = 0
    truncated: bool = False


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
        read_policy (JSONLReadPolicy): Controls line/text limits and decoding.
    """

    paths: Sequence[Path]
    context: RepoContext | None = None
    text_key: str = "text"
    check_schema: bool = True
    read_policy: JSONLReadPolicy = field(default_factory=JSONLReadPolicy)

    def iter_files(self) -> Iterable[FileItem]:
        """Yields FileItems constructed from JSONL records."""

        policy = self.read_policy
        for path in self.paths:
            opener = _open_jsonl_gz if "".join(path.suffixes[-2:]).lower() == ".jsonl.gz" else _open_jsonl
            try:
                with opener(path, errors=policy.decode_errors) as fp:
                    checked_schema = False
                    invalid_json_lines = 0
                    non_dict_lines = 0
                    missing_text_lines = 0
                    oversized_text_lines = 0
                    emitted_lines = 0
                    read_stats = _LineReadStats()
                    for lineno, raw_line in _iter_lines_with_limits(
                        fp, policy=policy, path=path, stats=read_stats
                    ):
                        line = raw_line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError as exc:
                            invalid_json_lines += 1
                            if invalid_json_lines <= policy.max_invalid_json_warnings:
                                log.warning("Skipping invalid JSON at %s:#%d: %s", path, lineno, exc)
                                if invalid_json_lines == policy.max_invalid_json_warnings:
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
                        if policy.max_text_chars is not None and len(text) > policy.max_text_chars:
                            oversized_text_lines += 1
                            if oversized_text_lines == 1:
                                log.warning(
                                    "Skipping text at %s:#%d exceeding max_text_chars=%d",
                                    path,
                                    lineno,
                                    policy.max_text_chars,
                                )
                            continue
                        rel = _derive_path(record, path.name, lineno)
                        data = text.encode("utf-8")
                        emitted_lines += 1
                        yield FileItem(path=rel, data=data, size=len(data))
                    if any(
                        (
                            invalid_json_lines,
                            non_dict_lines,
                            missing_text_lines,
                            read_stats.oversized_lines,
                            oversized_text_lines,
                            read_stats.truncated,
                        )
                    ):
                        log.info(
                            (
                                "Finished %s: emitted=%d skipped_invalid=%d skipped_nondict=%d "
                                "missing_text=%d oversized_lines=%d oversized_text=%d truncated=%d"
                            ),
                            path,
                            emitted_lines,
                            invalid_json_lines,
                            non_dict_lines,
                            missing_text_lines,
                            read_stats.oversized_lines,
                            oversized_text_lines,
                            int(read_stats.truncated),
                        )
                    else:
                        log.debug("Finished %s: emitted=%d", path, emitted_lines)
            except FileNotFoundError:
                log.warning("JSONL file not found: %s", path)
            except (OSError, UnicodeDecodeError, gzip.BadGzipFile) as exc:
                log.warning("Failed to read JSONL file %s: %s", path, exc)


def _iter_lines_with_limits(
    fp: TextIO,
    *,
    policy: JSONLReadPolicy,
    path: Path,
    stats: _LineReadStats,
) -> Iterator[tuple[int, str]]:
    """Iterate over lines while enforcing per-line and total size limits."""

    max_line_chars = policy.max_line_chars
    max_file_chars = policy.max_file_chars
    read_limit = (max_line_chars + 1) if max_line_chars is not None else None
    total_chars = 0
    lineno = 0

    while True:
        line = fp.readline() if read_limit is None else fp.readline(read_limit)
        if line == "":
            break
        lineno += 1
        total_chars += len(line)
        if max_file_chars is not None and total_chars > max_file_chars:
            if not stats.truncated:
                stats.truncated = True
                log.warning("Truncated JSONL file %s after reaching read limit (%d chars)", path, max_file_chars)
            break
        if max_line_chars is not None and len(line) > max_line_chars and not line.endswith("\n"):
            stats.oversized_lines += 1
            if stats.oversized_lines <= policy.max_oversized_line_warnings:
                log.warning("Skipping line %s:#%d exceeding max_line_chars=%d", path, lineno, max_line_chars)
                if stats.oversized_lines == policy.max_oversized_line_warnings:
                    log.debug("Suppressing further oversized line warnings for %s", path)
            total_chars, halted = _drain_oversized_line(
                fp,
                read_limit=read_limit,
                max_file_chars=max_file_chars,
                total_chars=total_chars,
                path=path,
                stats=stats,
            )
            if halted:
                break
            continue
        yield lineno, line


def _drain_oversized_line(
    fp: TextIO,
    *,
    read_limit: int | None,
    max_file_chars: int | None,
    total_chars: int,
    path: Path,
    stats: _LineReadStats,
) -> tuple[int, bool]:
    """Consume the remainder of an oversized line to realign iteration."""

    while True:
        chunk = fp.readline() if read_limit is None else fp.readline(read_limit)
        if chunk == "":
            return total_chars, False
        total_chars += len(chunk)
        if max_file_chars is not None and total_chars > max_file_chars:
            if not stats.truncated:
                stats.truncated = True
                log.warning("Truncated JSONL file %s after reaching read limit (%d chars)", path, max_file_chars)
            return total_chars, True
        if chunk.endswith("\n"):
            return total_chars, False


def _extract_text(record: dict[str, Any], key: str) -> str | None:
    """Extracts a string value from the given record by key."""

    val = record.get(key)
    if isinstance(val, str):
        return val
    return None


def _derive_path(record: dict[str, Any], fallback_name: str, lineno: int) -> str:
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
    cleaned = "".join(ch for ch in cleaned if ch.isprintable() and ch not in {"\r", "\n", "\x00"})
    parts = []
    for part in cleaned.split("/"):
        if not part or part == "." or part == "..":
            continue
        parts.append(part[:_MAX_PATH_SEGMENT_LENGTH])
    rel_path = "/".join(parts) if parts else ""
    if len(rel_path) > _MAX_SANITIZED_PATH_LENGTH:
        rel_path = rel_path[:_MAX_SANITIZED_PATH_LENGTH]
    return rel_path


def _should_check_schema(record: Mapping[str, Any]) -> bool:
    """Heuristic to decide whether to run schema checks on a record."""

    meta = record.get("meta")
    if not isinstance(meta, Mapping):
        return False
    if "schema_version" in meta:
        return True
    return any(key in STANDARD_META_FIELDS for key in meta)


def _open_jsonl(path: Path, *, errors: str):
    """Opens an uncompressed JSONL file for reading."""

    return open(path, encoding="utf-8", errors=errors)


def _open_jsonl_gz(path: Path, *, errors: str):
    """Opens a gzip-compressed JSONL file for reading."""

    return gzip.open(path, "rt", encoding="utf-8", errors=errors)
