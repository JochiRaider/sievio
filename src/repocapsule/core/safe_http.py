# safe_http.py
# SPDX-License-Identifier: MIT
"""Stdlib-only HTTP client with private IP and redirect safeguards."""

from __future__ import annotations

import http.client
import ipaddress
import socket
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Union

from .log import get_logger

# Note:
#   SAFE_HTTP_CLIENT and the get/set helpers below are provided as a convenience for simple scripts
#   and the CLI. Library callers and tests should prefer constructing SafeHttpClient instances
#   explicitly (e.g., via HttpConfig.build_client) and passing them into factories/sources.

log = get_logger(__name__)

RequestLike = Union[str, urllib.request.Request]


class PrivateAddressBlocked(RuntimeError):
    """Raised when DNS resolution returns a private IP address."""


class RedirectBlocked(RuntimeError):
    """Raised when a redirect targets a different host or forbidden scheme."""


class _SafeHTTPConnection(http.client.HTTPConnection):
    """HTTPConnection bound to a pre-resolved IP address."""

    def __init__(self, host: str, *, resolved_ip: str, **kwargs):
        super().__init__(host=host, **kwargs)
        self._resolved_ip = resolved_ip

    def connect(self) -> None:  
        self.sock = self._create_connection(
            (self._resolved_ip, self.port), self.timeout, self.source_address
        )


class _SafeHTTPSConnection(http.client.HTTPSConnection):
    """HTTPSConnection bound to a pre-resolved IP address with SNI."""

    def __init__(self, host: str, *, resolved_ip: str, **kwargs):
        super().__init__(host=host, **kwargs)
        self._resolved_ip = resolved_ip
        self._sni_host = host

    def connect(self) -> None:  
        self.sock = self._create_connection(
            (self._resolved_ip, self.port), self.timeout, self.source_address
        )
        if self._tunnel_host:
            self._tunnel()
        self.sock = self._context.wrap_socket(self.sock, server_hostname=self._sni_host)


@dataclass(frozen=True)
class SafeHttpResponse:
    """Minimal wrapper that mirrors urllib responses while owning the connection."""

    _response: http.client.HTTPResponse
    _connection: http.client.HTTPConnection
    url: str
    redirects: tuple[tuple[str, str, int], ...] = ()

    @property
    def status(self) -> int:
        return self._response.status

    @property
    def code(self) -> int:
        return self._response.status

    @property
    def reason(self) -> str:
        return self._response.reason

    @property
    def headers(self) -> http.client.HTTPMessage:
        return self._response.headers

    def info(self) -> http.client.HTTPMessage:
        return self._response.headers

    def getheader(self, name: str, default: Optional[str] = None) -> Optional[str]:
        return self._response.getheader(name, default)

    def read(self, amt: Optional[int] = None) -> bytes:  
        return self._response.read(amt)

    def readline(self, limit: Optional[int] = None) -> bytes:  
        return self._response.readline(limit or -1)

    def readinto(self, b) -> int:  
        return self._response.readinto(b)

    def close(self) -> None:
        try:
            self._response.close()
        finally:
            self._connection.close()

    def __enter__(self) -> "SafeHttpResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class SafeHttpClient:
    """HTTP client that blocks private IPs and enforces host-scoped redirects.

    Attributes:
        _default_timeout (float): Default request timeout in seconds.
        _max_redirects (int): Maximum redirects to follow.
        _trusted_redirect_suffixes (set[str]): Allowed redirect host suffixes.
    """

    _ALLOWED_SCHEMES = ("http", "https")
    _REDIRECT_CODES = {301, 302, 303, 307, 308}

    def __init__(
        self,
        *,
        timeout: float = 30.0,
        max_redirects: int = 5,
        allowed_redirect_suffixes: Optional[Sequence[str]] = None,
    ):
        self._default_timeout = timeout
        self._max_redirects = max_redirects
        self._trusted_redirect_suffixes = {
            suffix.lower().lstrip(".")
            for suffix in (allowed_redirect_suffixes or ())
            if suffix
        }

    # helpers
    def _resolve_ips(self, hostname: str, *, url: Optional[str] = None) -> list[str]:
        """Resolve a hostname and filter out private or disallowed addresses."""
        try:
            infos = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
        except socket.gaierror as exc:   # noqa: BLE001
            raise urllib.error.URLError(f"DNS resolution failed for {hostname}: {exc}") from exc
        ips: list[str] = []
        for family, _stype, _proto, _canon, sockaddr in infos:
            ip = sockaddr[0]
            addr = ipaddress.ip_address(ip)
            if addr.is_private or addr.is_loopback or addr.is_reserved or addr.is_multicast:
                continue
            is_link_local = getattr(addr, "is_link_local", None)
            if bool(is_link_local):
                continue
            ips.append(ip)
        if not ips:
            target = f" for {url}" if url else ""
            raise PrivateAddressBlocked(f"All resolved addresses for {hostname} are disallowed{target}")
        return ips

    def _trusted_suffix_for(self, host: Optional[str]) -> Optional[str]:
        """Return the trusted redirect suffix for a host, if any."""
        if not host:
            return None
        host_l = host.lower()
        for suffix in self._trusted_redirect_suffixes:
            if host_l == suffix or host_l.endswith("." + suffix):
                return suffix
        return None

    @staticmethod
    def _registrable_domain(host: str) -> Optional[str]:
        """Return the registrable domain portion of a host."""
        parts = host.split(".")
        if len(parts) < 2:
            return None
        return ".".join(parts[-2:])

    def _hosts_related(self, origin: Optional[str], target: Optional[str]) -> bool:
        """Determine if two hosts are related enough to allow redirects."""
        if not origin or not target:
            return False
        origin = origin.lower()
        target = target.lower()
        if target == origin:
            return True
        if target.endswith("." + origin):
            return True
        if origin.startswith("www.") and target == origin[4:]:
            return True
        if target.startswith("www.") and target[4:] == origin:
            return True
        origin_reg = self._registrable_domain(origin)
        target_reg = self._registrable_domain(target)
        if origin_reg and target_reg and origin_reg == target_reg:
            return True
        suffix = self._trusted_suffix_for(origin)
        if suffix and (target == suffix or target.endswith("." + suffix)):
            return True
        return False

    def _build_connection(
        self,
        *,
        scheme: str,
        host: str,
        ip: str,
        port: int,
        timeout: float,
    ) -> http.client.HTTPConnection:
        """Build a safe HTTP(S) connection pinned to a resolved IP."""
        common_kwargs = {"timeout": timeout}
        if scheme == "https":
            context = ssl.create_default_context()
            return _SafeHTTPSConnection(
                host,
                resolved_ip=ip,
                port=port,
                context=context,
                timeout=timeout,
            )
        return _SafeHTTPConnection(host, resolved_ip=ip, port=port, **common_kwargs)

    def _normalize_headers(
        self, headers: Optional[Mapping[str, str]], host: str, port: int, scheme: str
    ) -> Dict[str, str]:
        """Normalize headers, ensuring Host is set and not overridden."""
        host_value = host if self._is_default_port(scheme, port) else f"{host}:{port}"
        out: Dict[str, str] = {"Host": host_value}
        if headers:
            for k, v in headers.items():
                if k.lower() == "host":
                    continue
                out[k] = v
        return out

    @staticmethod
    def _is_default_port(scheme: str, port: int) -> bool:
        """Return True if the port matches the scheme's default."""
        return (scheme == "http" and port == 80) or (scheme == "https" and port == 443)

    @staticmethod
    def _build_path(parsed: urllib.parse.SplitResult) -> str:
        """Reconstruct a URL path with query string."""
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"
        return path

    # public
    def open(
        self,
        request: RequestLike,
        *,
        data: Optional[bytes] = None,
        timeout: Optional[float] = None,
        redirect_log: Optional[list[tuple[str, str, int]]] = None,
    ) -> SafeHttpResponse:
        """Open an HTTP request with redirect and IP safety checks.

        Args:
            request (RequestLike): URL string or prebuilt Request object.
            data (Optional[bytes]): Payload to send; overrides Request.data.
            timeout (Optional[float]): Request timeout; defaults to client
                default.
            redirect_log (Optional[list[tuple[str, str, int]]]): Optional log
                of redirects encountered.

        Returns:
            SafeHttpResponse: Response wrapper owning the connection.
        """
        req_obj = request
        if isinstance(request, str):
            req_obj = urllib.request.Request(request)
        method = req_obj.get_method()
        payload = req_obj.data if req_obj.data is not None else data
        headers = dict(req_obj.header_items())
        url = req_obj.full_url
        redirect_log = redirect_log if redirect_log is not None else []
        return self._request(
            url=url,
            method=method,
            headers=headers,
            body=payload,
            timeout=timeout or self._default_timeout,
            redirects_remaining=self._max_redirects,
            origin_host=urllib.parse.urlsplit(url).hostname,
            redirects_followed=0,
            redirect_log=redirect_log,
        )

    def open_with_retries(
        self,
        request: RequestLike,
        *,
        data: Optional[bytes] = None,
        timeout: Optional[float] = None,
        retries: int = 0,
        backoff_base: float = 1.0,
        backoff_factor: float = 2.0,
        only_get_like: bool = True,
        redirect_log: Optional[list[tuple[str, str, int]]] = None,
    ) -> SafeHttpResponse:
        """Retry wrapper around ``open`` with exponential backoff.

        Args:
            request (RequestLike): URL or Request to execute.
            data (Optional[bytes]): Payload to send; overrides Request.data.
            timeout (Optional[float]): Request timeout; defaults to client
                default.
            retries (int): Number of retry attempts.
            backoff_base (float): Base delay before the first retry.
            backoff_factor (float): Multiplier applied each retry.
            only_get_like (bool): Restrict retries to GET-like methods.
            redirect_log (Optional[list[tuple[str, str, int]]]): Optional log
                of redirects encountered.

        Returns:
            SafeHttpResponse: Response wrapper if the request succeeds.

        Raises:
            urllib.error.URLError: If all attempts fail.
        """
        req_obj = request
        if isinstance(request, str):
            req_obj = urllib.request.Request(request)
        method = (req_obj.get_method() or "GET").upper()
        if only_get_like and method not in {"GET", "HEAD", "OPTIONS", "TRACE"}:
            return self.open(req_obj, data=data, timeout=timeout)

        last_error: Optional[Exception] = None
        attempts = max(0, int(retries)) + 1
        for attempt in range(attempts):
            try:
                return self.open(req_obj, data=data, timeout=timeout, redirect_log=redirect_log)
            except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as exc:
                last_error = exc
                if attempt >= attempts - 1:
                    raise
                sleep_for = backoff_base * (backoff_factor ** attempt)
                if sleep_for > 0:
                    time.sleep(sleep_for)
                continue
        # This should be unreachable
        raise urllib.error.URLError(last_error or "unknown error")  # pragma: no cover

    # core
    def _request(
        self,
        *,
        url: str,
        method: str,
        headers: Optional[Mapping[str, str]],
        body: Optional[bytes],
        timeout: float,
        redirects_remaining: int,
        origin_host: Optional[str],
        redirects_followed: int,
        redirect_log: Optional[list[tuple[str, str, int]]] = None,
    ) -> SafeHttpResponse:
        """Internal request executor with redirect handling and safety checks."""
        parsed = urllib.parse.urlsplit(url)
        scheme = (parsed.scheme or "http").lower()
        if scheme not in self._ALLOWED_SCHEMES:
            raise urllib.error.URLError(f"Unsupported URL scheme: {scheme}")
        host = parsed.hostname
        if not host:
            raise urllib.error.URLError("URL missing host")
        port = parsed.port or (443 if scheme == "https" else 80)
        last_error: Optional[Exception] = None
        for ip in self._resolve_ips(host, url=url):
            conn = self._build_connection(scheme=scheme, host=host, ip=ip, port=port, timeout=timeout)
            path = self._build_path(parsed)
            all_headers = self._normalize_headers(headers, host, port, scheme)
            try:
                conn.request(method.upper(), path, body=body, headers=all_headers)
                response = conn.getresponse()
            except OSError as exc:
                last_error = exc
                conn.close()
                continue

            if response.status in self._REDIRECT_CODES:
                if redirects_remaining <= 0:
                    response.close()
                    conn.close()
                    raise RedirectBlocked("Too many redirects")
                location = response.getheader("Location")
                response.close()
                conn.close()
                if not location:
                    raise RedirectBlocked("Redirect response missing Location header")
                redirect_url = urllib.parse.urljoin(url, location)
                redirect_parts = urllib.parse.urlsplit(redirect_url)
                redirect_scheme = (redirect_parts.scheme or "http").lower()
                if redirect_scheme not in self._ALLOWED_SCHEMES:
                    raise RedirectBlocked(f"Redirect blocked: scheme {redirect_scheme!r} not permitted")
                new_host = redirect_parts.hostname
                if not self._hosts_related(origin_host, new_host):
                    raise RedirectBlocked(
                        f"Redirect blocked: cross-host redirect from {origin_host} to {new_host}"
                    )
                new_method = "GET" if response.status in (301, 302, 303) else method
                if redirect_log is not None:
                    redirect_log.append((url, redirect_url, response.status))
                return self._request(
                    url=redirect_url,
                    method=new_method,
                    headers=headers,
                    body=None if new_method == "GET" else body,
                    timeout=timeout,
                    redirects_remaining=redirects_remaining - 1,
                    origin_host=origin_host,
                    redirects_followed=redirects_followed + 1,
                    redirect_log=redirect_log,
                )
            log.debug(
                "HTTP %s %s status=%s redirects=%s",
                method.upper(),
                url,
                response.status,
                redirects_followed,
            )
            redirects = tuple(redirect_log) if redirect_log else ()
            return SafeHttpResponse(response, conn, url=url, redirects=redirects)

        raise urllib.error.URLError(f"All resolved addresses for {host} failed") from last_error


# safe_http.py – allow config overrides when instantiating the shared client
SAFE_HTTP_CLIENT: Optional[SafeHttpClient] = None


def get_global_http_client() -> SafeHttpClient:
    """Return the process-wide SafeHttpClient, creating a default if unset."""
    global SAFE_HTTP_CLIENT
    if SAFE_HTTP_CLIENT is None:
        SAFE_HTTP_CLIENT = SafeHttpClient(allowed_redirect_suffixes=("github.com",))
    return SAFE_HTTP_CLIENT


def set_global_http_client(client: Optional[SafeHttpClient]) -> None:
    """Override or clear the module-level SAFE_HTTP_CLIENT helper."""
    global SAFE_HTTP_CLIENT
    SAFE_HTTP_CLIENT = client
