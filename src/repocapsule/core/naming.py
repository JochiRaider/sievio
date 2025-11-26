# naming.py
# SPDX-License-Identifier: MIT
"""Helpers for constructing safe output names and normalizing extensions."""

from __future__ import annotations
import re, unicodedata
from urllib.parse import urlparse
from typing import Iterable, Optional, Set

__all__ = [
    "build_output_basename_github",
    "build_output_basename_pdf",
    "normalize_extensions",
]

_WINDOWS_FORBIDDEN = r'<>:"/\\|?*'
_SEPS_RX = re.compile(r"[^\w\-]+", flags=re.ASCII)

def _sanitize_component(s: Optional[str], *, lower: bool = True, maxlen: int = 64) -> str:
    """Sanitize a text component into a safe ASCII token.

    Args:
        s (str | None): Raw component value.
        lower (bool): Whether to lowercase the output token.
        maxlen (int): Maximum length of the sanitized token.

    Returns:
        str: Sanitized token, or "unknown" when empty after cleaning.
    """
    if not s:
        return "unknown"
    # ASCII-ize and collapse to safe chars
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = "".join(ch for ch in s if ch not in _WINDOWS_FORBIDDEN)
    if lower:
        s = s.lower()
    s = _SEPS_RX.sub("_", s).strip("_")
    # collapse repeats of underscores
    s = re.sub(r"_{2,}", "_", s)
    return (s or "unknown")[:maxlen]

def _normalize_spdx(spdx: Optional[str]) -> str:
    """Normalize an SPDX identifier into uppercase, defaulting to UNKNOWN.

    Args:
        spdx (str | None): SPDX string from repository or document metadata.

    Returns:
        str: Uppercase SPDX value or "UNKNOWN" when missing.
    """
    if not spdx:
        return "UNKNOWN"
    spdx = spdx.strip().upper()
    return spdx or "UNKNOWN"

def build_output_basename_github(*,
    owner: Optional[str],
    repo: Optional[str],
    ref: Optional[str],            # branch/tag/short-commit; pass "master" if unknown
    license_spdx: Optional[str],   # e.g., "MIT", "Apache-2.0", or None
    include_commit: Optional[str] = None,  # short sha (optional suffix)
    maxlen: int = 120,
) -> str:
    """Build a deterministic basename for GitHub-sourced outputs.

    Args:
        owner (str | None): Repository owner component.
        repo (str | None): Repository name component.
        ref (str | None): Branch, tag, or short commit reference.
        license_spdx (str | None): SPDX identifier to embed.
        include_commit (str | None): Commit suffix appended as another
            component when provided.
        maxlen (int): Maximum length of the assembled basename.

    Returns:
        str: Safe basename like owner__repo__ref__SPDX, optionally suffixed
            with a sanitized commit component, truncated to maxlen.
    """
    owner_s = _sanitize_component(owner, lower=True)
    repo_s  = _sanitize_component(repo,  lower=True)
    ref_s   = _sanitize_component(ref,   lower=True)
    spdx    = _normalize_spdx(license_spdx)  # keep SPDX uppercase
    parts = [owner_s, repo_s, ref_s, spdx]
    if include_commit:
        parts.append(_sanitize_component(include_commit, lower=True, maxlen=12))
    base = "__".join(parts)
    # Hard cap to allow extension (.jsonl/.txt) without exceeding path limits
    return base[:maxlen]

def build_output_basename_pdf(*,
    url: Optional[str],
    title: Optional[str],
    license_spdx: Optional[str],
    maxlen: int = 120,
) -> str:
    """Build a deterministic basename for PDF-sourced outputs.

    Args:
        url (str | None): Source URL used to derive the host component.
        title (str | None): PDF title or filename stem component.
        license_spdx (str | None): SPDX identifier to embed.
        maxlen (int): Maximum length of the assembled basename.

    Returns:
        str: Safe basename like host__title__SPDX, truncated to maxlen.
    """
    host = "unknown"
    if url:
        try:
            host = urlparse(url).netloc or "unknown"
        except Exception:
            host = "unknown"
    host_s  = _sanitize_component(host, lower=True)
    title_s = _sanitize_component(title, lower=True)
    spdx    = _normalize_spdx(license_spdx)
    base = "__".join([host_s, title_s, spdx])
    return base[:maxlen]

def normalize_extensions(exts: Optional[Iterable[str]]) -> Optional[Set[str]]:
    """Normalize extension strings into dotted lowercase values.

    Args:
        exts (Iterable[str] | None): Iterable of extensions to normalize.

    Returns:
        set[str] | None: Lowercase extensions prefixed with ".", or None when
            no values remain after cleaning.
    """
    if not exts:
        return None
    out: Set[str] = set()
    for ext in exts:
        if not ext:
            continue
        cleaned = ext.strip().lower()
        if not cleaned:
            continue
        out.add(cleaned if cleaned.startswith(".") else f".{cleaned}")
    return out or None
