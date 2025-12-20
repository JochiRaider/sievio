# factories_context.py
# SPDX-License-Identifier: MIT
"""
Helpers for HTTP client construction and repository context inference.

Split out from core.factories to keep contextual helpers localized.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from .interfaces import RepoContext
from .safe_http import SafeHttpClient

if TYPE_CHECKING:  # pragma: no cover - type-only imports
    from .config import HttpConfig

__all__ = [
    "make_http_client",
    "make_repo_context_from_git",
]


def make_http_client(http_cfg: HttpConfig) -> SafeHttpClient:
    """
    Build (or reuse) the SafeHttpClient described by ``http_cfg``.

    Args:
        http_cfg (HttpConfig): HTTP configuration describing the client.

    Returns:
        SafeHttpClient: Client configured per ``http_cfg``.

    Raises:
        ValueError: If ``http_cfg`` is missing.
    """
    if http_cfg is None:
        raise ValueError("http_cfg is required")
    return http_cfg.build_client()


def make_repo_context_from_git(repo_root: Path | str) -> RepoContext | None:
    """
    Infer a RepoContext from ``.git/config`` when the remote points at GitHub.

    Args:
        repo_root (Path | str): Repository root containing a ``.git`` folder.

    Returns:
        RepoContext | None: Populated context or ``None`` when metadata is
        unavailable.
    """
    cfg_path = Path(repo_root) / ".git" / "config"
    try:
        text = cfg_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    current_remote: str | None = None
    origin_url: str | None = None
    fallback_url: str | None = None
    remote_header = re.compile(r'\s*\[remote\s+"([^"]+)"\]')
    url_line = re.compile(r"^\s*url\s*=\s*([^\r\n]+)$")

    for line in text.splitlines():
        header = remote_header.match(line)
        if header:
            current_remote = header.group(1)
            continue
        if current_remote is None:
            continue
        m = url_line.match(line)
        if not m:
            continue
        url_value = m.group(1).strip()
        if current_remote == "origin":
            origin_url = url_value
            break
        if fallback_url is None:
            fallback_url = url_value

    remote = origin_url or fallback_url
    if not remote:
        return None
    from ..sources.githubio import parse_github_url  # local import to avoid cycles

    spec = parse_github_url(remote)
    if not spec:
        return None
    return RepoContext(
        repo_full_name=f"{spec.owner}/{spec.repo}",
        repo_url=f"https://github.com/{spec.owner}/{spec.repo}",
        license_id=None,
        extra={"source": "local"},
    )
