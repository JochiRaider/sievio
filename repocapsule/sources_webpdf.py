# sources_webpdf.py
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Dict, List, Set
from html.parser import HTMLParser
from urllib.parse import urlparse, unquote, urljoin
import urllib.parse, urllib.request, urllib.error
import posixpath, re, time, io


from .interfaces import Source, FileItem
from . import safe_http
from .config import PdfSourceConfig

__all__ = ["WebPdfListSource", "WebPagePdfSource"]


_USER_AGENT = "repocapsule/0.1 (+https://github.com)"
_CHUNK = 1024 * 1024  # 1 MiB
_ALLOWED_SCHEMES = {"http", "https"}

def _ensure_http_url(url: str) -> str:
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
    """parser for Content-Disposition filename/filename*."""
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
        except Exception:
            pass
    return val


def _name_from_url(u: str) -> str:
    path = urlparse(u).path
    name = posixpath.basename(path) or "document.pdf"
    return name


def _looks_like_pdf(head: bytes) -> bool:
    # PDF files begin with "%PDF-" magic
    return head.startswith(b"%PDF-")


class WebPdfListSource(Source):
    """
    Fetch a list of web-hosted PDFs and yield them as FileItem objects.

    Parameters
    ----------
    urls : sequence of str
        Direct links to PDFs.
    timeout : int
        Per-request timeout (seconds).
    max_pdf_bytes : int
        Hard cap; downloads abort once this many bytes are read.
    require_pdf : bool
        If True, skip responses that don't *sniff* like a PDF.
    add_prefix : str | None
        Optional folder prefix to place before each emitted filename.
    retries : int
        Simple retry count with backoff (2^attempt seconds).
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
    ) -> None:
        cfg = config or PdfSourceConfig()
        self._pdf_config = cfg
        self.urls = list(urls)
        self.timeout = int(timeout if timeout is not None else cfg.timeout)
        self.max_pdf_bytes = int(max_pdf_bytes if max_pdf_bytes is not None else cfg.max_pdf_bytes)
        self.require_pdf = cfg.require_pdf if require_pdf is None else bool(require_pdf)
        self.add_prefix = add_prefix.strip().strip("/").replace("\\", "/") if add_prefix else None
        self.retries = max(0, int(retries if retries is not None else cfg.retries))
        self._user_agent = cfg.user_agent or _USER_AGENT

    def _request(self, url: str) -> urllib.request.Request:
        safe_url = _ensure_http_url(url)
        return urllib.request.Request(
            safe_url,
            headers={
                "User-Agent": self._user_agent,
                "Accept": "*/*",
            },
            method="GET",
        )

    def _download(self, url: str) -> tuple[bytes, Dict[str, str]]:
        """Stream a URL into bytes with size cap; return (data, headers)."""
        attempt = 0
        last_err: Optional[Exception] = None
        while attempt <= self.retries:
            try:
                with safe_http.SAFE_HTTP_CLIENT.open(self._request(url), timeout=self.timeout) as resp:
                    if resp.status >= 400:
                        raise urllib.error.HTTPError(url, resp.status, resp.reason, resp.headers, None)
                    headers = {k: v for k, v in resp.headers.items()}
                    # Content-Length guard if present
                    cl = headers.get("Content-Length")
                    if cl:
                        try:
                            if int(cl) > self.max_pdf_bytes:
                                raise RuntimeError(f"Content-Length {cl} exceeds cap {self.max_pdf_bytes}")
                        except Exception:
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
                    return data, headers
            except Exception as e:
                last_err = e
                if attempt >= self.retries:
                    break
                time.sleep(2 ** attempt)
                attempt += 1
        raise RuntimeError(f"failed to download {url}: {last_err}")

    def iter_files(self) -> Iterable[FileItem]:
        used_names: set[str] = set()
        # Serial by design: the pipeline owns concurrency. Avoid nested thread pools (oversubscription / deadlock hazard).
        for u in self.urls:
            try:
                data, headers = self._download(u)
            except Exception:
                continue

            # Decide on filename: Content-Disposition > URL path
            cd_name = _filename_from_content_disposition(headers.get("Content-Disposition"))
            name = _sanitize_name(cd_name or _name_from_url(u))
            # Ensure .pdf extension
            if not name.lower().endswith(".pdf"):
                name = f"{name}.pdf"

            # PDF sniff (strict if require_pdf)
            sniff_head = bytes.fromhex(headers.get("_X-SNIFF", "")) if headers.get("_X-SNIFF") else data[:8]
            if self.require_pdf and not _looks_like_pdf(sniff_head):
                # Some servers mislabel; add a lenient fallback: allow if body still looks like PDF
                if not _looks_like_pdf(data[:8]):
                    continue

            # Deduplicate filenames
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

            yield FileItem(path=name, data=data, size=len(data))

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
    """
    Scrape a single HTML page for PDF links, then download them via WebPdfListSource.

    Parameters
    ----------
    page_url : str
        The web page to scan for PDF links.
    same_domain : bool
        Keep only links on the same host as `page_url` (default True).
    max_links : int
        Cap the number of discovered links we attempt to fetch.
    match_regex : str | None
        Optional regex; only keep links whose URL matches.
    include_ambiguous : bool
        If True, include links without a .pdf suffix; the downstream
        WebPdfListSource will still sniff for real PDFs.
    timeout, max_pdf_bytes, require_pdf, add_prefix, retries :
        Passed through to WebPdfListSource.
    """

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
    ) -> None:
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

    def _req(self, url: str) -> urllib.request.Request:
        safe_url = _ensure_http_url(url)
        return urllib.request.Request(
            safe_url,
            headers={"User-Agent": self._user_agent, "Accept": "text/html,application/xhtml+xml,*/*;q=0.9"},
            method="GET",
        )

    def _fetch_html(self) -> str:
        with safe_http.SAFE_HTTP_CLIENT.open(self._req(self.page_url), timeout=self.timeout) as r:
            if r.status >= 400:
                raise urllib.error.HTTPError(self.page_url, r.status, r.reason, r.headers, None)
            raw = r.read(5 * 1024 * 1024)  # 5 MiB cap
            ct = r.headers.get_content_charset() or "utf-8"
        try:
            return raw.decode(ct, errors="replace")
        except Exception:
            return raw.decode("utf-8", errors="replace")

    def _discover_pdf_links(self, html: str) -> List[str]:
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
        try:
            html = self._fetch_html()
        except Exception:
            return   # no items

        urls = self._discover_pdf_links(html)
        if not urls:
            return 

        inner = WebPdfListSource(
            urls,
            timeout=self.timeout,
            max_pdf_bytes=self.max_pdf_bytes,
            require_pdf=self.require_pdf,
            add_prefix=self.add_prefix,
            retries=self.retries,
            config=self._pdf_config,
        )
        # Delegate actual downloads to the list source
        yield from inner.iter_files()
