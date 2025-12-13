# sources_webpdf.py
# SPDX-License-Identifier: MIT
"""Sources for downloading PDFs via direct links or webpage scraping."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Dict, List, Set, Tuple
from html.parser import HTMLParser
from urllib.parse import urlparse, unquote, urljoin
import urllib.parse, urllib.request, urllib.error
import posixpath, re, io
from dataclasses import dataclass

from ..core.interfaces import Source, FileItem
from ..core import safe_http
from ..core.config import PdfSourceConfig
from ..core.log import get_logger
from ..core.concurrency import Executor, ExecutorConfig

__all__ = ["WebPdfListSource", "WebPagePdfSource"]


_USER_AGENT = "sievio/0.1 (+https://github.com/jochiraider/sievio)"
_CHUNK = 1024 * 1024  # 1 MiB
_ALLOWED_SCHEMES = {"http", "https"}


@dataclass(frozen=True)
class _PdfDownloadTask:
    index: int
    url: str

def _ensure_http_url(url: str) -> str:
    """Validates and normalizes an HTTP(S) URL for PDF ingestion.

    Args:
        url (str): URL to validate.

    Returns:
        str: Original URL if it passes validation.

    Raises:
        ValueError: If the URL scheme is unsupported or contains credentials.
    """
    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise ValueError(f"Unsupported URL scheme for PDF ingestion: {scheme or '<none>'}")
    # Reject embedded credentials to avoid accidental leaks / weird auth.
    if parsed.username or parsed.password:
        raise ValueError("Credentials in URL are not allowed")
    # Normalize path (strip dot segments) for logging/record stability.
    # note: don't recompose full URL (avoid altering server expectations).
    _ = posixpath.normpath(parsed.path or "/")    
    return url

def _sanitize_name(name: str, fallback: str = "document.pdf") -> str:
    """Generates a safe filename from a URL or header value."""
    # Keep just a safe basename; replace weird characters.
    base = posixpath.basename(name) or fallback
    base = unquote(base)
    base = base.replace("\\", "/").split("/")[-1]
    # Force a conservative character set
    base = re.sub(r"[^A-Za-z0-9._+-]", "_", base)
    # Avoid empty/hidden
    if not base or base in {".", ".."}:
        base = fallback
    return base


_CDISP_RE = re.compile(
    r"""(?ix)
    ^\s*(?:inline|attachment)\s*;
    \s*(?:filename\*=(?P<fnstar>[^;]+)|filename=(?P<fn>[^;]+))?
    """
)

def _filename_from_content_disposition(hval: Optional[str]) -> Optional[str]:
    """Parses a Content-Disposition header for filename fields.

    Args:
        hval (str | None): Header value to inspect.

    Returns:
        str | None: Filename or None if not present.
    """
    if not hval:
        return None
    m = _CDISP_RE.search(hval)
    if not m:
        return None
    val = m.group("fnstar") or m.group("fn")
    if not val:
        return None
    val = val.strip().strip('"').strip("'")
    # RFC 5987 / 6266: filename*=utf-8''encoded
    if "''" in val:
        try:
            enc, _, rest = val.split("'", 2)
            return urllib.parse.unquote(rest, encoding=(enc or "utf-8"), errors="replace")
        except Exception as exc:
            log.debug(
                "Failed to parse RFC5987 filename from Content-Disposition %r: %s",
                hval,
                exc,
            )
    return val


def _name_from_url(u: str) -> str:
    """Derives a filename from the path portion of a URL."""
    path = urlparse(u).path
    name = posixpath.basename(path) or "document.pdf"
    return name


def _looks_like_pdf(head: bytes) -> bool:
    """Checks whether a byte prefix matches PDF magic bytes."""
    # PDF files begin with "%PDF-" magic
    return head.startswith(b"%PDF-")


log = get_logger(__name__)


class WebPdfListSource(Source):
    """Downloads web-hosted PDFs and yields them as FileItem objects.

    Attributes:
        urls (Sequence[str]): Direct PDF links to fetch.
        timeout (int): Per-request timeout in seconds.
        max_pdf_bytes (int): Hard cap on downloaded bytes.
        require_pdf (bool): Whether to skip responses that do not sniff
            like a PDF.
        add_prefix (str | None): Optional folder prefix for filenames.
        retries (int): Retry count with exponential backoff.
    """

    def __init__(
        self,
        urls: Sequence[str],
        *,
        timeout: Optional[int] = None,
        max_pdf_bytes: Optional[int] = None,
        require_pdf: Optional[bool] = None,
        add_prefix: Optional[str] = None,
        retries: Optional[int] = None,
        config: Optional[PdfSourceConfig] = None,
        client: Optional[safe_http.SafeHttpClient] = None,
    ) -> None:
        """Initializes the list source with fetch configuration.

        Args:
            urls (Sequence[str]): Direct PDF URLs to download.
            timeout (int | None): Per-request timeout override.
            max_pdf_bytes (int | None): Hard cap on downloaded bytes.
            require_pdf (bool | None): Skip non-PDF responses when True.
            add_prefix (str | None): Optional prefix for emitted paths.
            retries (int | None): Number of download retries.
            config (PdfSourceConfig | None): Optional source configuration.
            client (safe_http.SafeHttpClient | None): HTTP client to use.
        """
        cfg = config or PdfSourceConfig()
        self._pdf_config = cfg
        self.urls = list(urls)
        self.timeout = int(timeout if timeout is not None else cfg.timeout)
        self.max_pdf_bytes = int(max_pdf_bytes if max_pdf_bytes is not None else cfg.max_pdf_bytes)
        self.require_pdf = cfg.require_pdf if require_pdf is None else bool(require_pdf)
        self.add_prefix = add_prefix.strip().strip("/").replace("\\", "/") if add_prefix else None
        self.retries = max(0, int(retries if retries is not None else cfg.retries))
        self._user_agent = cfg.user_agent or _USER_AGENT
        resolved_client = client or cfg.client
        if resolved_client is None:
            resolved_client = safe_http.get_global_http_client()
        if cfg.client is None:
            cfg.client = resolved_client
        self._client = resolved_client
        self._download_max_workers = cfg.download_max_workers
        self._download_submit_window = cfg.download_submit_window
        self._download_executor_kind = cfg.download_executor_kind or "thread"

    def _request(self, url: str) -> urllib.request.Request:
        """Builds a GET request for the given URL with standard headers."""
        safe_url = _ensure_http_url(url)
        return urllib.request.Request(
            safe_url,
            headers={
                "User-Agent": self._user_agent,
                "Accept": "*/*",
            },
            method="GET",
        )

    def _download(self, url: str) -> tuple[bytes, Dict[str, str], int]:
        """Streams a URL into bytes with a size cap.

        Args:
            url (str): Target URL to fetch.

        Returns:
            tuple[bytes, Dict[str, str], int]: Downloaded data, response
                headers, and original file size if declared.

        Raises:
            RuntimeError: If the download fails or exceeds configured caps.
        """
        try:
            with self._client.open_with_retries(
                self._request(url),
                timeout=self.timeout,
                retries=self.retries,
                backoff_base=1.0,
                backoff_factor=2.0,
            ) as resp:
                if resp.status >= 400:
                    raise urllib.error.HTTPError(url, resp.status, resp.reason, resp.headers, None)
                headers = {k: v for k, v in resp.headers.items()}
                # Content-Length guard if present
                cl = headers.get("Content-Length")
                if cl:
                    try:
                        if int(cl) > self.max_pdf_bytes:
                            raise RuntimeError(f"Content-Length {cl} exceeds cap {self.max_pdf_bytes}")
                    except Exception as exc:
                        log.debug(
                            "Ignoring invalid Content-Length %r for %s: %s",
                            cl,
                            url,
                            exc,
                        )
                        # If bad header, ignore and rely on streaming cap
                        pass

                # Stream with cap; also capture a small head for sniffing
                buf = io.BytesIO()
                head = b""
                first = True
                remaining = self.max_pdf_bytes
                while True:
                    to_read = min(_CHUNK, remaining)
                    if to_read <= 0:
                        break
                    chunk = resp.read(to_read)
                    if not chunk:
                        break
                    if first:
                        head = chunk[:8]
                        first = False
                    buf.write(chunk)
                    remaining -= len(chunk)
                data = buf.getvalue()
                if len(data) == 0:
                    raise RuntimeError("empty response")
                if len(data) > self.max_pdf_bytes:
                    # If we exactly hit the cap, treat as too large (likely truncated)
                    raise RuntimeError("download reached size cap (truncated)")
                # Attach a small sniff header in case the caller wants it
                headers["_X-SNIFF"] = head[:8].hex()
                declared = headers.get("Content-Length")
                original_size = len(data)
                if declared is not None:
                    try:
                        declared_size = int(declared)
                        if declared_size >= len(data):
                            original_size = declared_size
                    except ValueError:
                        pass
                return data, headers, original_size
        except Exception as exc:
            raise RuntimeError(f"failed to download {url}: {exc}") from exc

    def _normalize_download_result(
        self,
        result,
    ) -> tuple[bytes, Dict[str, str], int]:
        """Normalizes download output to a uniform tuple shape."""
        if isinstance(result, tuple):
            if len(result) == 3:
                data, headers, file_size = result
            elif len(result) == 2:
                data, headers = result
                file_size = len(data)
            else:
                raise ValueError("unexpected download result shape")
        else:
            raise TypeError("download result must be tuple")
        return data, dict(headers), int(file_size)

    def _build_file_item(
        self,
        url: str,
        data: bytes,
        headers: Dict[str, str],
        used_names: Set[str],
        file_bytes: int,
    ) -> Optional[FileItem]:
        """Builds a FileItem from downloaded PDF content."""
        cd_name = _filename_from_content_disposition(headers.get("Content-Disposition"))
        name = _sanitize_name(cd_name or _name_from_url(url))
        if not name.lower().endswith(".pdf"):
            name = f"{name}.pdf"

        sniff_head = bytes.fromhex(headers.get("_X-SNIFF", "")) if headers.get("_X-SNIFF") else data[:8]
        if self.require_pdf and not _looks_like_pdf(sniff_head):
            if not _looks_like_pdf(data[:8]):
                return None

        orig = name
        n = 1
        while name in used_names:
            stem, dot, ext = orig.rpartition(".")
            if stem:
                name = f"{stem}__{n}{dot}{ext}" if dot else f"{stem}__{n}"
            else:
                name = f"{orig}__{n}"
            n += 1
        used_names.add(name)

        if self.add_prefix:
            name = f"{self.add_prefix}/{name}"

        return FileItem(
            path=name,
            data=data,
            size=file_bytes,
            origin_path=url,
            stream_hint="http",
            streamable=False,
        )

    def iter_files(self) -> Iterable[FileItem]:
        """Yields downloaded PDFs as FileItem objects."""
        used_names: Set[str] = set()
        total_urls = len(self.urls)
        success_count = 0
        skipped_count = 0
        error_count = 0

        if total_urls <= 1:
            for _, u in enumerate(self.urls):
                try:
                    result = self._download(u)
                    data, headers, file_size = self._normalize_download_result(result)
                except Exception as exc:
                    error_count += 1
                    log.warning("Failed to download PDF %s: %s", u, exc)
                    continue
                item = self._build_file_item(u, data, headers, used_names, file_size)
                if item:
                    success_count += 1
                    yield item
                else:
                    skipped_count += 1
            log.info(
                "WebPdfListSource: processed %d URLs (success=%d, skipped=%d, errors=%d)",
                total_urls,
                success_count,
                skipped_count,
                error_count,
            )
            return

        results: Dict[int, FileItem] = {}

        def _worker(task: _PdfDownloadTask) -> Tuple[_PdfDownloadTask, List[tuple[str, bytes, Dict[str, str], int]]]:
            result = self._download(task.url)
            data, headers, file_size = self._normalize_download_result(result)
            return task, [(task.url, data, headers, file_size)]

        def _writer(task: _PdfDownloadTask, payloads: Iterable[tuple[str, bytes, Dict[str, str], int]]) -> None:
            nonlocal success_count, skipped_count
            for url, data, headers, file_size in payloads:
                item = self._build_file_item(url, data, headers, used_names, file_size)
                if item:
                    results[task.index] = item
                    success_count += 1
                else:
                    skipped_count += 1

        def _on_worker_error(exc: BaseException) -> None:
            nonlocal error_count
            error_count += 1
            log.warning("PDF download worker failed: %s", exc)

        cfg_workers = self._download_max_workers
        if cfg_workers and cfg_workers > 0:
            max_workers = min(cfg_workers, len(self.urls))
        else:
            max_workers = min(4, len(self.urls))
        window = (
            self._download_submit_window
            if self._download_submit_window is not None
            else max_workers * 4
        )
        executor_kind = (self._download_executor_kind or "thread").strip().lower()
        if executor_kind not in {"thread", "process"}:
            log.warning(
                "WebPdfListSource: unknown download_executor_kind %r; defaulting to 'thread'",
                executor_kind,
            )
            executor_kind = "thread"
        if executor_kind == "process":
            log.warning(
                "WebPdfListSource: process executor is not currently supported for downloads; "
                "falling back to thread executor."
            )
        exec_cfg = ExecutorConfig(
            max_workers=max_workers,
            window=max(window, max_workers),
            kind="thread",
        )
        executor = Executor(exec_cfg)
        executor.map_unordered(
            (_PdfDownloadTask(idx, url) for idx, url in enumerate(self.urls)),
            _worker,
            lambda result: _writer(*result),
            fail_fast=False,
            on_error=_on_worker_error,
        )

        for idx in range(total_urls):
            item = results.get(idx)
            if item:
                yield item
        log.info(
            "WebPdfListSource: processed %d URLs (success=%d, skipped=%d, errors=%d)",
            total_urls,
            success_count,
            skipped_count,
            error_count,
        )

# ----------------------------
#  WebPagePdfSource
# ----------------------------

class _PdfLinkScraper(HTMLParser):
    """Tiny link scraper to collect PDF hrefs and <base href> (if present)."""
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: List[str] = []
        self.base_href: Optional[str] = None

    def handle_starttag(self, tag: str, attrs: List[tuple[str, Optional[str]]]) -> None:
        at = dict(attrs)
        if tag.lower() == "base" and "href" in at and at["href"]:
            self.base_href = at["href"]
            return
        if tag.lower() in ("a", "area", "link"):
            href = at.get("href")
            if href:
                self.links.append(href)


class WebPagePdfSource(Source):
    """Scrapes a single HTML page for PDF links and downloads them."""

    def __init__(
        self,
        page_url: str,
        *,
        same_domain: bool = True,
        max_links: Optional[int] = None,
        match_regex: Optional[str] = None,
        include_ambiguous: Optional[bool] = None,
        timeout: Optional[int] = None,
        max_pdf_bytes: Optional[int] = None,
        require_pdf: Optional[bool] = None,
        add_prefix: Optional[str] = None,
        retries: Optional[int] = None,
        config: Optional[PdfSourceConfig] = None,
        client: Optional[safe_http.SafeHttpClient] = None,
    ) -> None:
        """Initializes the page scraper with discovery and download settings.

        Args:
            page_url (str): Page URL to scan for PDF links.
            same_domain (bool): Restrict links to the same host.
            max_links (int | None): Maximum links to fetch.
            match_regex (str | None): Regex that links must satisfy.
            include_ambiguous (bool | None): Include links without .pdf
                suffixes when True.
            timeout (int | None): Per-request timeout override.
            max_pdf_bytes (int | None): Hard cap on downloaded bytes.
            require_pdf (bool | None): Skip non-PDF responses when True.
            add_prefix (str | None): Optional path prefix for outputs.
            retries (int | None): Download retry count.
            config (PdfSourceConfig | None): Optional source config.
            client (safe_http.SafeHttpClient | None): HTTP client to use.
        """
        cfg = config or PdfSourceConfig()
        self._pdf_config = cfg
        self.page_url = page_url
        self.same_domain = bool(same_domain)
        self.max_links = max(1, int(max_links if max_links is not None else cfg.max_links))
        self.match = re.compile(match_regex) if match_regex else None
        self.include_ambiguous = cfg.include_ambiguous if include_ambiguous is None else bool(include_ambiguous)
        self.timeout = int(timeout if timeout is not None else cfg.timeout)
        self.max_pdf_bytes = int(max_pdf_bytes if max_pdf_bytes is not None else cfg.max_pdf_bytes)
        self.require_pdf = cfg.require_pdf if require_pdf is None else bool(require_pdf)
        self.add_prefix = add_prefix
        self.retries = max(0, int(retries if retries is not None else cfg.retries))
        self._user_agent = cfg.user_agent or _USER_AGENT
        resolved_client = client or cfg.client
        if resolved_client is None:
            resolved_client = safe_http.get_global_http_client()
        if cfg.client is None:
            cfg.client = resolved_client
        self._client = resolved_client

    def _req(self, url: str) -> urllib.request.Request:
        """Builds an HTML request with the configured User-Agent."""
        safe_url = _ensure_http_url(url)
        return urllib.request.Request(
            safe_url,
            headers={"User-Agent": self._user_agent, "Accept": "text/html,application/xhtml+xml,*/*;q=0.9"},
            method="GET",
        )

    def _fetch_html(self) -> str:
        """Fetches the HTML content of the configured page."""
        with self._client.open_with_retries(
            self._req(self.page_url),
            timeout=self.timeout,
            retries=self.retries,
            backoff_base=1.0,
            backoff_factor=2.0,
        ) as r:
            if r.status >= 400:
                raise urllib.error.HTTPError(self.page_url, r.status, r.reason, r.headers, None)
            raw = r.read(5 * 1024 * 1024)  # 5 MiB cap
            ct = r.headers.get_content_charset() or "utf-8"
        try:
            return raw.decode(ct, errors="replace")
        except Exception as exc:
            log.debug(
                "Failed to decode HTML for %s with declared charset %s: %s",
                self.page_url,
                ct,
                exc,
            )
            return raw.decode("utf-8", errors="replace")

    def _discover_pdf_links(self, html: str) -> List[str]:
        """Extracts candidate PDF links from HTML content.

        Args:
            html (str): HTML source to scan for links.

        Returns:
            list[str]: Resolved PDF URLs that meet filtering rules.
        """
        scraper = _PdfLinkScraper()
        scraper.feed(html)

        # Base for resolution: <base href> if present, else page URL
        base = scraper.base_href or self.page_url
        base = urljoin(self.page_url, base)  # normalize
        base_host = urlparse(self.page_url).netloc

        found: List[str] = []
        seen: Set[str] = set()
        for href in scraper.links:
            abs_url = urljoin(base, href)
            parsed = urlparse(abs_url)
            scheme = (parsed.scheme or "").lower()
            if scheme not in _ALLOWED_SCHEMES:
                continue            
            if self.same_domain and urlparse(abs_url).netloc != base_host:
                continue
            if self.match and not self.match.search(abs_url):
                continue

            # Heuristic: keep links ending with .pdf unless include_ambiguous is True
            path = urlparse(abs_url).path.lower()
            looks_pdf = path.endswith(".pdf")
            if not looks_pdf and not self.include_ambiguous:
                continue

            if abs_url not in seen:
                seen.add(abs_url)
                found.append(abs_url)
                if len(found) >= self.max_links:
                    break
        return found

    def iter_files(self) -> Iterable[FileItem]:
        """Yields FileItems for PDFs discovered on the target page."""
        try:
            html = self._fetch_html()
        except Exception as exc:
            log.warning("Failed to fetch web page %s: %s", self.page_url, exc)
            return   # no items

        urls = self._discover_pdf_links(html)
        if not urls:
            return 

        success_count = 0
        inner = WebPdfListSource(
            urls,
            timeout=self.timeout,
            max_pdf_bytes=self.max_pdf_bytes,
            require_pdf=self.require_pdf,
            add_prefix=self.add_prefix,
            retries=self.retries,
            config=self._pdf_config,
            client=self._client,
        )
        for item in inner.iter_files():
            success_count += 1
            yield item

        log.info(
            "WebPagePdfSource: processed %d discovered URLs (success=%d) from %s",
            len(urls),
            success_count,
            self.page_url,
        )
