"""repocapsule.githubio (stream-safe)

GitHub I/O helpers with **streamed zipball download** and **zip-bomb defenses**.
Stdlib-only; drop-in replacement for `githubio.py`.

Public API (compatible with previous version):
- RepoSpec, parse_github_url
- github_api_get, get_repo_info, get_repo_license_spdx
- download_zipball_to_temp  (now streams in chunks; RAM-safe)
- iter_zip_members          (adds robust safeguards)
- build_output_basename
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple, Dict, Any
import contextlib
import json
import logging
import os
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

__all__ = [
    "RepoSpec",
    "parse_github_url",
    "github_api_get",
    "get_repo_info",
    "get_repo_license_spdx",
    "download_zipball_to_temp",
    "iter_zip_members",
    "build_output_basename",
]

log = logging.getLogger(__name__)

# -----------------
# Data structure
# -----------------

@dataclass(frozen=True)
class RepoSpec:
    owner: str
    repo: str
    ref: Optional[str] = None   # branch/tag/SHA; None → default branch
    subpath: Optional[str] = None  # path under repo (for partial conversions)

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.repo}"


# -----------------
# URL parsing
# -----------------

_GH_REPO = re.compile(r"^https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/#?]+?)(?:\.git)?(?:$|[?#/])")
_GH_TREE = re.compile(r"^https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/tree/(?P<ref>[^/]+)(?:/(?P<sub>.*))?$")
_GH_BLOB = re.compile(r"^https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/blob/(?P<ref>[^/]+)/(?:/(?P<sub>.*)|(?P<sub2>.*))$")
_SSH = re.compile(r"^(?:git@|ssh://git@)github\.com[:/](?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?$")


def parse_github_url(url: str) -> Optional[RepoSpec]:
    """Parse common GitHub URL forms into a `RepoSpec`.

    Supported forms:
      - https://github.com/owner/repo
      - https://github.com/owner/repo.git
      - https://github.com/owner/repo/tree/<ref>[/subpath]
      - https://github.com/owner/repo/blob/<ref>/<subpath>
      - git@github.com:owner/repo(.git)
    Returns None if it doesn't look like GitHub.
    """
    u = url.strip()
    m = _GH_TREE.match(u)
    if m:
        sub = m.group("sub") or None
        return RepoSpec(m.group("owner"), m.group("repo"), m.group("ref"), sub)
    m = _GH_BLOB.match(u)
    if m:
        sub = (m.group("sub") or m.group("sub2") or None)
        return RepoSpec(m.group("owner"), m.group("repo"), m.group("ref"), sub)
    m = _GH_REPO.match(u)
    if m:
        return RepoSpec(m.group("owner"), m.group("repo"), None, None)
    m = _SSH.match(u)
    if m:
        return RepoSpec(m.group("owner"), m.group("repo"), None, None)
    return None


# -----------------
# HTTP helpers
# -----------------

_API_BASE = "https://api.github.com"
_USER_AGENT = "repocapsule/0.1 (+https://github.com)"


def _auth_token() -> Optional[str]:
    return os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")


def _build_request(url: str, *, accept: Optional[str] = None) -> urllib.request.Request:
    headers = {
        "User-Agent": _USER_AGENT,
        "Accept": accept or "application/vnd.github+json",
    }
    tok = _auth_token()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    return urllib.request.Request(url, headers=headers)


def github_api_get(path: str, *, accept: Optional[str] = None, timeout: int = 30) -> Tuple[int, Dict[str, Any], bytes]:
    """GET `path` (starting with '/') from the GitHub API.

    Returns (status, headers_dict, body_bytes). Does not raise on HTTPError; it
    captures the code and body for diagnostics.
    """
    if not path.startswith("/"):
        path = "/" + path
    url = _API_BASE + path
    req = _build_request(url, accept=accept)
    try:
        with contextlib.closing(urllib.request.urlopen(req, timeout=timeout)) as resp:
            body = resp.read()
            headers = {k: v for k, v in resp.headers.items()}
            return resp.status, headers, body
    except urllib.error.HTTPError as e:
        body = e.read() if hasattr(e, "read") else b""
        headers = {k: v for k, v in getattr(e, "headers", {}).items()}
        return e.code, headers, body
    except urllib.error.URLError as e:
        log.error("GitHub API network error: %s", e)
        raise


def _rate_limit_note(headers: Dict[str, Any]) -> str:
    rem = headers.get("X-RateLimit-Remaining")
    rst = headers.get("X-RateLimit-Reset")
    if rem is None:
        return ""
    try:
        rem_i = int(rem)
        if rem_i <= 0 and rst:
            try:
                ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(int(rst)))
                return f" (rate limit resets at {ts})"
            except Exception:
                return " (rate limited)"
    except Exception:
        pass
    return ""


# -----------------------
# Repo info / license API
# -----------------------

def get_repo_info(spec: RepoSpec) -> Dict[str, Any]:
    """Return repo metadata dict: {default_branch, license_spdx?, license_name?}."""
    status, headers, body = github_api_get(f"/repos/{spec.owner}/{spec.repo}")
    if status != 200:
        note = _rate_limit_note(headers)
        raise RuntimeError(f"GitHub /repos request failed: HTTP {status}{note}: {body[:256]!r}")
    meta = json.loads(body.decode("utf-8", "replace"))
    out: Dict[str, Any] = {"default_branch": meta.get("default_branch") or "main"}
    lic = meta.get("license") or {}
    if isinstance(lic, dict):
        spdx = lic.get("spdx_id")
        if spdx and spdx != "NOASSERTION":
            out["license_spdx"] = spdx
            out["license_name"] = lic.get("name")

    # Try license endpoint for better fidelity
    status, headers, body = github_api_get(
        f"/repos/{spec.owner}/{spec.repo}/license",
        accept="application/vnd.github+json",
    )
    if status == 200:
        licobj = json.loads(body.decode("utf-8", "replace"))
        lic_meta = (licobj.get("license") or {}) if isinstance(licobj, dict) else {}
        spdx = lic_meta.get("spdx_id")
        if spdx and spdx != "NOASSERTION":
            out["license_spdx"] = spdx
            out["license_name"] = lic_meta.get("name")
        if isinstance(licobj, dict) and licobj.get("path"):
            out["license_path"] = licobj.get("path")
    elif status in (403, 429):
        log.warning("GitHub license API throttled for %s: HTTP %s%s", spec.full_name, status, _rate_limit_note(headers))
    else:
        log.info("No license info for %s (HTTP %s)", spec.full_name, status)

    return out


def get_repo_license_spdx(spec: RepoSpec) -> Optional[str]:
    try:
        info = get_repo_info(spec)
    except Exception:
        return None
    return info.get("license_spdx")


# -------------------
# Zipball download (STREAMED)
# -------------------

_DEF_ZIP_TIMEOUT = 60


def download_zipball_to_temp(
    spec: RepoSpec,
    *,
    ref: Optional[str] = None,
    timeout: int = _DEF_ZIP_TIMEOUT,
    chunk_size: int = 1024 * 1024,                  # 1 MiB
    max_zip_bytes: int = 3 * 1024 * 1024 * 1024     # 3 GiB hard cap
) -> str:
    """Download a repository zipball to a temp file and return its path.

    Streamed to disk in chunks to avoid loading the whole archive into RAM.
    """
    # Resolve ref
    ref_used = ref or spec.ref
    if not ref_used:
        try:
            info = get_repo_info(spec)
            ref_used = info.get("default_branch") or "main"
        except Exception:
            ref_used = "main"

    url = f"{_API_BASE}/repos/{spec.owner}/{spec.repo}/zipball/{urllib.parse.quote(ref_used)}"
    req = _build_request(url, accept="application/vnd.github+json")

    try:
        with contextlib.closing(urllib.request.urlopen(req, timeout=timeout)) as resp:
            # Preflight: enforce Content-Length if present
            cl = resp.headers.get("Content-Length")
            if cl:
                try:
                    if int(cl) > max_zip_bytes:
                        raise RuntimeError(
                            f"Zipball Content-Length {cl} exceeds max {max_zip_bytes} bytes"
                        )
                except Exception:
                    pass  # fall back to streamed cap

            fd, tmp_path = tempfile.mkstemp(prefix="repocapsule_", suffix=".zip")
            total = 0
            try:
                with os.fdopen(fd, "wb") as f:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > max_zip_bytes:
                            raise RuntimeError(
                                f"Zipball exceeded max size cap ({max_zip_bytes} bytes)"
                            )
                        f.write(chunk)
                return tmp_path
            except Exception:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
                raise
    except urllib.error.HTTPError as e:
        body = e.read() if hasattr(e, "read") else b""
        note = _rate_limit_note(getattr(e, "headers", {}) or {})
        raise RuntimeError(
            f"GitHub zipball failed for {spec.full_name}@{ref_used}: "
            f"HTTP {getattr(e,'code','?')}{note}: {body[:256]!r}"
        )
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error downloading zipball: {e}")


# -------------------
# Zip utilities (zip-bomb defenses)
# -------------------

def _safe_relpath(name: str) -> Optional[str]:
    # Normalize and drop the leading top-level folder GitHub adds
    s = name.replace("\\", "/")
    if s.startswith("/"):
        return None
    if ".." in s.split("/"):
        return None
    parts = s.split("/")
    if len(parts) <= 1:
        return None
    # Disallow any '.' or '..' traversal components
    for seg in parts:
        if seg in (".", ".."):
            return None
    rel = "/".join(parts[1:])  # strip "owner-repo-SHA/"
    return rel


def iter_zip_members(
    zip_path: str,
    *,
    max_bytes_per_file: Optional[int] = None,
    max_total_uncompressed: int = 2 * 1024 * 1024 * 1024,  # 2 GiB cap across all files
    max_members: int = 200_000,
    max_compression_ratio: float = 100.0,                   # file_size / compress_size
) -> Iterator[Tuple[str, bytes]]:
    """Yield (relative_path, data_bytes) for each regular file in the zipball.

    Zip-bomb defenses:
      - Reject if estimated uncompressed sum (ZipInfo.file_size) exceeds `max_total_uncompressed`
      - Reject if member count exceeds `max_members`
      - Skip entries with suspicious compression ratio
      - Enforce per-file cap via `max_bytes_per_file`
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Preflight: estimate uncompressed total + member count
        est_total = 0
        est_members = 0
        for zi in zf.infolist():
            if zi.is_dir():
                continue
            est_members += 1
            est_total += int(getattr(zi, "file_size", 0) or 0)
            if est_members > max_members:
                raise RuntimeError(f"Zip has too many members (> {max_members})")
        if est_total > max_total_uncompressed:
            raise RuntimeError(
                f"Zip claims {est_total} uncompressed bytes, exceeds limit {max_total_uncompressed}"
            )

        total = 0
        for zi in zf.infolist():
            if zi.is_dir():
                continue
            rel = _safe_relpath(zi.filename)
            if not rel:
                continue
            # Skip symlinks
            is_symlink = (zi.external_attr >> 16) & 0o170000 == 0o120000
            if is_symlink:
                continue

            comp_sz = int(getattr(zi, "compress_size", 0) or 0)
            file_sz = int(getattr(zi, "file_size", 0) or 0)
            if comp_sz > 0 and file_sz / max(1, comp_sz) > max_compression_ratio:
                log.warning("Skipping %s: suspicious compression ratio (%.1fx)",
                            rel, file_sz / max(1, comp_sz))
                continue

            if max_bytes_per_file is not None and file_sz > max_bytes_per_file:
                # Too big; skip without reading
                continue

            projected_total = total + (min(file_sz, max_bytes_per_file)
                                       if max_bytes_per_file is not None else file_sz)
            if projected_total > max_total_uncompressed:
                raise RuntimeError("Total uncompressed bytes would exceed safety limit; aborting")

            # Read entry in bounded chunks (respects per-file cap)
            with contextlib.closing(zf.open(zi, "r")) as fh:
                remaining = max_bytes_per_file if max_bytes_per_file is not None else None
                parts: list[bytes] = []
                while True:
                    to_read = 1024 * 1024 if remaining is None else min(1024 * 1024, max(0, remaining))
                    if to_read == 0:
                        break
                    chunk = fh.read(to_read)
                    if not chunk:
                        break
                    parts.append(chunk)
                    if remaining is not None:
                        remaining -= len(chunk)
                data = b"".join(parts)

            total += len(data)
            if total > max_total_uncompressed:
                raise RuntimeError("Total uncompressed bytes exceeded safety limit")

            yield rel, data


# -------------------
# Naming helpers
# -------------------

def build_output_basename(spec: RepoSpec, *, license_spdx: Optional[str] = None, ref: Optional[str] = None) -> str:
    """Return a safe base name like 'owner_repo__ref__SPDX' (no extension)."""
    owner = re.sub(r"[^A-Za-z0-9_.-]", "_", spec.owner)
    repo = re.sub(r"[^A-Za-z0-9_.-]", "_", spec.repo)
    parts = [owner, repo]
    ref_used = ref or spec.ref
    if ref_used:
        parts.append(re.sub(r"[^A-Za-z0-9_.-]", "_", ref_used))
    if license_spdx:
        parts.append(re.sub(r"[^A-Za-z0-9A-Za-z+.-]", "_", license_spdx.upper()))
    return "__".join(parts)
