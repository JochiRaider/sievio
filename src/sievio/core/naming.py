# naming.py
# SPDX-License-Identifier: MIT
"""Helpers for constructing safe output names and normalizing extensions."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from urllib.parse import urlparse

__all__ = [
    "build_output_basename_github",
    "build_output_basename_pdf",
    "normalize_extensions",
]

_WINDOWS_FORBIDDEN = r'<>:"/\\|?*'
_WINDOWS_RESERVED = {
    "con",
    "prn",
    "aux",
    "nul",
    "com1",
    "com2",
    "com3",
    "com4",
    "com5",
    "com6",
    "com7",
    "com8",
    "com9",
    "lpt1",
    "lpt2",
    "lpt3",
    "lpt4",
    "lpt5",
    "lpt6",
    "lpt7",
    "lpt8",
    "lpt9",
}
_SEPS_RX = re.compile(r"[^\w\-]+", flags=re.ASCII)

def _sanitize_component(s: str | None, *, lower: bool = True, maxlen: int = 64) -> str:
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
    token = (s or "unknown")[:maxlen]
    if token.lower() in _WINDOWS_RESERVED:
        token = f"{token}_"
        if len(token) > maxlen:
            token = f"{token[:maxlen-1]}_"
    return token

def _normalize_spdx(spdx: str | None) -> str:
    """Normalize an SPDX identifier into a safe uppercase token.

    Args:
        spdx (str | None): SPDX string from repository or document metadata.

    Returns:
        str: Uppercase, sanitized SPDX value or "UNKNOWN" when missing.
    """
    if not spdx:
        return "UNKNOWN"
    spdx_clean = _sanitize_component(spdx, lower=False, maxlen=64)
    spdx_clean = spdx_clean.upper()
    return spdx_clean or "UNKNOWN"

def build_output_basename_github(*,
    owner: str | None,
    repo: str | None,
    ref: str | None,            # branch/tag/short-commit; pass "master" if unknown
    license_spdx: str | None,   # e.g., "MIT", "Apache-2.0", or None
    include_commit: str | None = None,  # short sha (optional suffix)
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
    url: str | None,
    title: str | None,
    license_spdx: str | None,
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

def normalize_extensions(exts: Iterable[str] | None) -> set[str] | None:
    """Normalize extension strings into dotted lowercase values.

    Args:
        exts (Iterable[str] | None): Iterable of extensions to normalize.

    Returns:
        set[str] | None: Lowercase extensions prefixed with ".", or None when
            no values remain after cleaning.
    """
    if not exts:
        return None
    out: set[str] = set()
    for ext in exts:
        if not ext:
            continue
        cleaned = ext.strip().lower()
        if not cleaned:
            continue
        out.add(cleaned if cleaned.startswith(".") else f".{cleaned}")
    return out or None
