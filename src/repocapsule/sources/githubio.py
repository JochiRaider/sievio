# githubio.py
# SPDX-License-Identifier: MIT

"""GitHub helpers for downloading repositories and scanning their contents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple, Dict, Any, Iterable
import contextlib, json, os, re, tempfile, time, urllib.error, urllib.parse, urllib.request, zipfile
from pathlib import Path

from ..core import safe_http
from ..core.interfaces import FileItem, RepoContext, Source
from ..core.licenses import detect_license_in_zip, apply_license_to_context
from ..core.log import get_logger
from ..core.naming import normalize_extensions

__all__ = [
    "RepoSpec",
    "parse_github_url",
    "github_api_get",
    "get_repo_info",
    "get_repo_license_spdx",
    "download_zipball_to_temp",
    "iter_zip_members",
    "detect_license_for_github_repo",
]

log = get_logger(__name__)


# -----------------
# Data structure
# -----------------

@dataclass(frozen=True)
class RepoSpec:
    """Immutable specification for a GitHub repository reference.

    Attributes:
        owner (str): Repository owner.
        repo (str): Repository name.
        ref (str | None): Optional branch, tag, or commit ref.
        subpath (str | None): Optional subdirectory within the repo.
    """

    owner: str
    repo: str
    ref: Optional[str] = None  
    subpath: Optional[str] = None  

    @property
    def full_name(self) -> str:
        """Returns the full owner/repo string."""

        return f"{self.owner}/{self.repo}"


# -----------------
# URL parsing
# -----------------

_GH_REPO = re.compile(r"^https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/#?]+?)(?:\.git)?(?:$|[?#/])")
_GH_TREE = re.compile(r"^https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/tree/(?P<ref>[^/]+)(?:/(?P<sub>.*))?$")
_GH_BLOB = re.compile(r"^https?://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)$")
_SSH = re.compile(r"^(?:git@|ssh://git@)github\.com[:/](?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?$")


def parse_github_url(url: str) -> Optional[RepoSpec]:
    """Parses a GitHub URL into a RepoSpec.

    Supported forms include:
        - https://github.com/owner/repo
        - https://github.com/owner/repo.git
        - https://github.com/owner/repo/tree/<ref>[/subpath]
        - https://github.com/owner/repo/blob/<ref>/<subpath>
        - git@github.com:owner/repo(.git)

    Args:
        url (str): GitHub URL to parse.

    Returns:
        RepoSpec | None: Parsed repository specification, or None if the
            URL does not match GitHub patterns.
    """
    u = url.strip()
    m = _GH_TREE.match(u)
    if m:
        sub = m.group("sub") or None
        return RepoSpec(m.group("owner"), m.group("repo"), m.group("ref"), sub)
    m = _GH_BLOB.match(u)
    if m:
        owner, repo, ref, sub = m.groups()
        return RepoSpec(owner, repo, ref, sub)
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


def github_api_get(
    path: str,
    *,
    accept: Optional[str] = None,
    timeout: int = 30,
    client: Optional[safe_http.SafeHttpClient] = None,
) -> Tuple[int, Dict[str, Any], bytes]:
    """Performs a GitHub API GET request for the given path.

    Args:
        path (str): API path starting with a leading slash.
        accept (str | None): Optional Accept header override.
        timeout (int): Request timeout in seconds.
        client (safe_http.SafeHttpClient | None): Optional HTTP client.

    Returns:
        tuple[int, Dict[str, Any], bytes]: Status code, response headers,
            and response body.

    Raises:
        urllib.error.URLError: If the request fails at the network layer.
    """
    if not path.startswith("/"):
        path = "/" + path
    url = _API_BASE + path
    req = _build_request(url, accept=accept)
    http_client = client or safe_http.get_global_http_client()
    try:
        with http_client.open_with_retries(req, timeout=timeout, retries=2) as resp:
            body = resp.read()
            headers = {k: v for k, v in resp.headers.items()}
            return resp.status, headers, body
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

def get_repo_info(spec: RepoSpec, *, client: Optional[safe_http.SafeHttpClient] = None) -> Dict[str, Any]:
    """Retrieves repository metadata and license information.

    Args:
        spec (RepoSpec): Repository specification.
        client (safe_http.SafeHttpClient | None): Optional HTTP client.

    Returns:
        Dict[str, Any]: Metadata including the default branch and any
            discovered license details.

    Raises:
        RuntimeError: If the GitHub API responds with a non-200 status.
        urllib.error.URLError: If the request fails at the network layer.
    """
    status, headers, body = github_api_get(f"/repos/{spec.owner}/{spec.repo}", client=client)
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
        client=client,
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


def get_repo_license_spdx(spec: RepoSpec, *, client: Optional[safe_http.SafeHttpClient] = None) -> Optional[str]:
    """Fetches the SPDX license identifier for a repository if available.

    Args:
        spec (RepoSpec): Repository specification.
        client (safe_http.SafeHttpClient | None): Optional HTTP client.

    Returns:
        str | None: SPDX identifier or None when it cannot be determined.
    """
    try:
        info = get_repo_info(spec, client=client)
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
    timeout: float = _DEF_ZIP_TIMEOUT,
    chunk_size: int = 1024 * 1024,                  # 1 MiB
    max_zip_bytes: int = 3 * 1024 * 1024 * 1024,    # 3 GiB hard cap
    client: Optional[safe_http.SafeHttpClient] = None,
) -> str:
    """Downloads a repository zipball to a temporary file.

    Streamed to disk in chunks to avoid loading the whole archive into
    memory.

    Args:
        spec (RepoSpec): Repository specification.
        ref (str | None): Branch, tag, or commit ref to download.
        timeout (float): Request timeout in seconds.
        chunk_size (int): Size of each streamed chunk in bytes.
        max_zip_bytes (int): Hard cap on total downloaded bytes.
        client (safe_http.SafeHttpClient | None): Optional HTTP client.

    Returns:
        str: Path to the temporary zip file on disk.

    Raises:
        RuntimeError: If the download fails or exceeds configured limits.
        urllib.error.URLError: If the request fails at the network layer.
    """
    http_client = client or safe_http.get_global_http_client()
    ref_used = ref or spec.ref
    if not ref_used:
        try:
            info = get_repo_info(spec, client=http_client)
            ref_used = info.get("default_branch") or "main"
        except Exception:
            ref_used = "main"

    url = f"{_API_BASE}/repos/{spec.owner}/{spec.repo}/zipball/{urllib.parse.quote(ref_used)}"
    req = _build_request(url, accept="application/vnd.github+json")

    try:
        with http_client.open_with_retries(req, timeout=timeout, retries=2) as resp:
            if resp.status >= 400:
                body = resp.read()
                note = _rate_limit_note({k: v for k, v in resp.headers.items()})
                raise RuntimeError(
                    f"GitHub zipball failed for {spec.full_name}@{ref_used}: "
                    f"HTTP {resp.status}{note}: {body[:256]!r}"
                )
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
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error downloading zipball: {e}")


def detect_license_for_github_repo(
    spec: RepoSpec,
    *,
    ref: Optional[str] = None,
    timeout: float = _DEF_ZIP_TIMEOUT,
    client: Optional[safe_http.SafeHttpClient] = None,
) -> Optional[str]:
    """Attempts to detect a repository license by scanning a zipball.

    Args:
        spec (RepoSpec): Repository specification.
        ref (str | None): Optional ref to download.
        timeout (float): Download timeout in seconds.
        client (safe_http.SafeHttpClient | None): Optional HTTP client.

    Returns:
        str | None: Detected license identifier (for example, "MIT") or
            None if detection fails.
    """
    zip_path: Optional[str] = None
    try:
        zip_path = download_zipball_to_temp(
            spec,
            ref=ref,
            timeout=timeout,
            client=client,
        )
        if not zip_path:
            return None
        license_id, _meta = detect_license_in_zip(zip_path, spec.subpath)
        return license_id
    except Exception as exc:
        log.debug("GitHub license detection failed for %s: %s", spec.full_name, exc)
        return None
    finally:
        if zip_path:
            try:
                os.remove(zip_path)
            except OSError as exc:
                log.debug("temp zip cleanup failed for %s: %s", zip_path, exc)


# -------------------
# Zip utilities (zip-bomb defenses)
# -------------------

def _safe_relpath(name: str) -> Optional[str]:
    """Normalizes a zip member path and rejects unsafe traversal segments.

    Args:
        name (str): Raw path from the zip archive entry.

    Returns:
        str | None: Relative path without the repository prefix, or None
            if the path is unsafe.
    """
    # Normalize and drop the leading top-level folder GitHub adds
    s = name.replace("\\", "/")
    if s.startswith("/"):
        return None
    head = s.split("/", 1)[0]
    if ":" in head:  # e.g., "C:" on Windows or scheme-like segment
        return None
    if ".." in s.split("/"):
        return None
    parts = s.split("/")
    if len(parts) <= 1:
        return None
    # Disallow any '.' or '..' traversal components
    for seg in parts:
        if seg in (".", "..") or ":" in seg:
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
) -> Iterator[Tuple[str, bytes, int]]:
    """Iterates safe members of a repository zipball.

    Applies zip-bomb defenses to limit size, member count, and compression
    ratio while skipping unsafe paths and symlinks.

    Args:
        zip_path (str): Path to the zip archive.
        max_bytes_per_file (int | None): Optional per-file size cap.
        max_total_uncompressed (int): Maximum total uncompressed bytes.
        max_members (int): Maximum number of files to process.
        max_compression_ratio (float): Maximum allowed compression ratio.

    Yields:
        tuple[str, bytes, int]: Relative path, extracted bytes, and
            original size for each file.

    Raises:
        ValueError: If provided safety limits are invalid.
        RuntimeError: If archive limits are exceeded.
    """
     # Parameter validation
    if max_compression_ratio <= 0:
        raise ValueError("max_compression_ratio must be > 0")
    if max_members <= 0:
        raise ValueError("max_members must be a positive integer")
    if max_total_uncompressed <= 0:
        raise ValueError("max_total_uncompressed must be a positive integer")
    if max_bytes_per_file is not None and max_bytes_per_file <= 0:
        raise ValueError("max_bytes_per_file must be positive when set")  

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
        for zi in sorted(zf.infolist(), key=lambda z: z.filename.casefold()):
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
                log.debug("Skipping %s: file_size=%d > per-file cap=%d",
                          rel, file_sz, max_bytes_per_file)
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

            yield rel, data, file_sz


class GitHubZipSource(Source):
    """Manages the lifecycle of a GitHub zipball for use as a source.

    Downloads on entry and cleans up the temporary file on exit to avoid
    buildup even when exceptions occur.
    """

    def __init__(
        self,
        url: str,
        *,
        config,
        context: Optional[RepoContext] = None,
        download_timeout: Optional[float] = None,
        http_client: Optional[safe_http.SafeHttpClient] = None,
    ) -> None:
        """Initializes the source with a GitHub repository URL.

        Args:
            url (str): GitHub URL pointing to the repository.
            config: Source configuration with filtering and safety limits.
            context (RepoContext | None): Optional repository context to
                update with detected metadata.
            download_timeout (float | None): Optional download timeout.
            http_client (safe_http.SafeHttpClient | None): Optional HTTP
                client to reuse.
        """
        spec = parse_github_url(url)
        if not spec:
            raise ValueError(f"Invalid GitHub URL: {url!r}")
        self.spec = spec
        self._cfg = config
        self.context = context
        self._subpath = spec.subpath.strip("/").replace("\\", "/") if spec.subpath else None
        self._zip_path: str | None = None
        self._download_timeout = download_timeout
        self._http_client = http_client
        self.include_exts = normalize_extensions(getattr(config, "include_exts", None))
        self.exclude_exts = normalize_extensions(getattr(config, "exclude_exts", None))

    def __enter__(self) -> "GitHubZipSource":
        """Downloads the zipball and performs optional license detection."""

        client = self._http_client or safe_http.get_global_http_client()
        if self._download_timeout is None:
            self._zip_path = download_zipball_to_temp(self.spec, client=client)
        else:
            self._zip_path = download_zipball_to_temp(
                self.spec,
                timeout=self._download_timeout,
                client=client,
            )
        if self._zip_path:
            license_id, meta = detect_license_in_zip(self._zip_path, self._subpath)
            if license_id and (self.context is None or not self.context.license_id):
                self.context = apply_license_to_context(self.context, license_id, meta)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Deletes the downloaded zipball when exiting the context."""

        if self._zip_path:
            try:
                os.remove(self._zip_path)
            except OSError as e:
                log.debug("temp zip cleanup failed for %s: %s", self._zip_path, e)
            self._zip_path = None

    def iter_files(self) -> Iterable[FileItem]:
        """Yields file items from the downloaded zipball with filters applied.

        Yields:
            FileItem: Archive members that satisfy extension and safety
                constraints.
        """
        assert self._zip_path, "GitHubZipSource must be entered before use"
        cfg = self._cfg
        for rel_path, data, original_size in iter_zip_members(
            self._zip_path,
            max_bytes_per_file=cfg.per_file_cap,
            max_total_uncompressed=cfg.max_total_uncompressed,
            max_members=cfg.max_members,
            max_compression_ratio=cfg.max_compression_ratio,
        ):
            rel_norm = rel_path.replace("\\", "/")
            if self._subpath:
                prefix = self._subpath
                if not (rel_norm == prefix or rel_norm.startswith(f"{prefix}/")):
                    continue
            ext = Path(rel_norm).suffix.lower()
            if self.include_exts is not None and ext not in self.include_exts:
                continue
            if self.exclude_exts is not None and ext in self.exclude_exts:
                continue
            yield FileItem(
                path=rel_norm,
                data=data,
                size=original_size or len(data),
                stream_hint="zip-member",
                streamable=False,
            )
