# naming.py

from __future__ import annotations
import re, unicodedata
from urllib.parse import urlparse
from typing import Optional

_WINDOWS_FORBIDDEN = r'<>:"/\\|?*'
_SEPS_RX = re.compile(r"[^\w\-]+", flags=re.ASCII)

def _sanitize_component(s: Optional[str], *, lower: bool = True, maxlen: int = 64) -> str:
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
    """
    author_repo_branch_license -> "owner__repo__ref__SPDX" (optionally + "__sha7").
    Example: karpathy__nanochat__master__MIT
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
    """
    Equivalent metadata for PDFs -> "host__title__SPDX".
    - host comes from URL netloc (fallback: 'unknown')
    - title may be PDF title or filename stem (sanitized)
    Example: arxiv.org__attention_is_all_you_need__UNKNOWN
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

# Back-compat single entry-point mirroring older scripts:
def build_output_basename(*,
    kind: str,
    owner: Optional[str] = None,
    repo: Optional[str] = None,
    ref: Optional[str] = None,
    license_spdx: Optional[str] = None,
    include_commit: Optional[str] = None,
    url: Optional[str] = None,
    title: Optional[str] = None,
    maxlen: int = 120,
) -> str:
    """
    Unified builder:
      kind="github" -> use owner/repo/ref/license[+commit]
      kind="pdf"    -> use url/title/license
    """
    k = (kind or "").lower()
    if k == "github":
        return build_output_basename_github(
            owner=owner, repo=repo, ref=ref,
            license_spdx=license_spdx, include_commit=include_commit, maxlen=maxlen
        )
    elif k == "pdf":
        return build_output_basename_pdf(
            url=url, title=title, license_spdx=license_spdx, maxlen=maxlen
        )
    else:
        raise ValueError(f"unknown kind for build_output_basename: {kind!r}")
