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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Union

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

    def getheader(self, name: str, default: str | None = None) -> str | None:
        return self._response.getheader(name, default)

    def read(self, amt: int | None = None) -> bytes:  
        return self._response.read(amt)

    def readline(self, limit: int | None = None) -> bytes:  
        if limit is None:
            return self._response.readline()
        return self._response.readline(limit)

    def readinto(self, b) -> int:  
        return self._response.readinto(b)

    def close(self) -> None:
        try:
            self._response.close()
        finally:
            self._connection.close()

    def __enter__(self) -> SafeHttpResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


@dataclass(frozen=True)
class SafeHttpPolicy:
    """Policy hooks for SafeHttpClient decisions."""

    allow_ip: Callable[[ipaddress._BaseAddress], bool]
    allow_redirect: Callable[[str | None, str | None], bool]
    allow_redirect_scheme: Callable[[str, str], bool]
    redirect_headers: Callable[
        [Mapping[str, str] | None, str | None, str | None],
        Mapping[str, str] | None,
    ]


def _default_allow_ip(addr: ipaddress._BaseAddress) -> bool:
    """Allow only globally routable unicast addresses."""
    if not addr.is_global:
        return False
    if addr.is_multicast or addr.is_unspecified or addr.is_loopback:
        return False
    is_link_local = getattr(addr, "is_link_local", None)
    if bool(is_link_local):
        return False
    return True


def _default_redirect_headers(
    headers: Mapping[str, str] | None,
    old_host: str | None,
    new_host: str | None,
) -> Mapping[str, str] | None:
    """Drop sensitive headers when the redirect target host changes."""
    if headers is None:
        return None
    if SafeHttpClient._normalize_host(old_host) == SafeHttpClient._normalize_host(new_host):
        return dict(headers)
    sensitive = {"authorization", "cookie", "proxy-authorization"}
    return {k: v for k, v in headers.items() if k.lower() not in sensitive}


def _default_allow_redirect_scheme(old_scheme: str, new_scheme: str) -> bool:
    """Block downgrades from HTTPS to HTTP."""
    if old_scheme == "https" and new_scheme == "http":
        return False
    return True


def _build_default_allow_redirect(trusted_suffixes: set[str]) -> Callable[[str | None, str | None], bool]:
    """Build the default redirect admission function."""

    def _host_matches_suffix(host: str | None, suffix: str) -> bool:
        normalized = SafeHttpClient._normalize_host(host)
        return bool(normalized and (normalized == suffix or normalized.endswith("." + suffix)))

    def allow_redirect(origin: str | None, target: str | None) -> bool:
        origin_n = SafeHttpClient._normalize_host(origin)
        target_n = SafeHttpClient._normalize_host(target)
        if not origin_n or not target_n:
            return False
        if target_n == origin_n:
            return True
        if target_n.endswith("." + origin_n):
            return True
        if origin_n.endswith("." + target_n) and "." in target_n:
            return True
        for suffix in trusted_suffixes:
            if _host_matches_suffix(origin_n, suffix) and _host_matches_suffix(target_n, suffix):
                return True
        return False

    return allow_redirect


def _build_default_policy(trusted_suffixes: set[str]) -> SafeHttpPolicy:
    return SafeHttpPolicy(
        allow_ip=_default_allow_ip,
        allow_redirect=_build_default_allow_redirect(trusted_suffixes),
        allow_redirect_scheme=_default_allow_redirect_scheme,
        redirect_headers=_default_redirect_headers,
    )


DEFAULT_POLICY = _build_default_policy(set())


class SafeHttpClient:
    """HTTP client that blocks private IPs and enforces host-scoped redirects.

    Only globally routable unicast addresses are permitted by default, redirects
    are limited to related hosts or an explicit allowlist, and sensitive headers
    are stripped when the redirect target host changes.

    Attributes:
        _default_timeout (float): Default request timeout in seconds.
        _max_redirects (int): Maximum redirects to follow.
        _trusted_redirect_suffixes (set[str]): Allowed redirect host suffixes.
        _policy (SafeHttpPolicy): Policy hooks for IP filtering, redirects, and headers.
    """

    _ALLOWED_SCHEMES = ("http", "https")
    _REDIRECT_CODES = {301, 302, 303, 307, 308}

    def __init__(
        self,
        *,
        timeout: float = 30.0,
        max_redirects: int = 5,
        allowed_redirect_suffixes: Sequence[str] | None = None,
        policy: SafeHttpPolicy | None = None,
    ):
        self._default_timeout = timeout
        self._max_redirects = max_redirects
        self._trusted_redirect_suffixes = {
            suffix.lower().lstrip(".")
            for suffix in (allowed_redirect_suffixes or ())
            if suffix
        }
        if policy is not None:
            self._policy = policy
        elif self._trusted_redirect_suffixes:
            self._policy = _build_default_policy(self._trusted_redirect_suffixes)
        else:
            self._policy = DEFAULT_POLICY

    # helpers
    def _resolve_ips(self, hostname: str, *, url: str | None = None) -> list[str]:
        """Resolve a hostname and filter out private or disallowed addresses."""
        try:
            infos = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
        except socket.gaierror as exc:   # noqa: BLE001
            raise urllib.error.URLError(f"DNS resolution failed for {hostname}: {exc}") from exc
        ips: list[str] = []
        seen: set[str] = set()
        for family, _stype, _proto, _canon, sockaddr in infos:
            ip = sockaddr[0]
            addr = ipaddress.ip_address(ip)
            if not self._policy.allow_ip(addr):
                continue
            if ip not in seen:
                seen.add(ip)
                ips.append(ip)
        if not ips:
            target = f" for {url}" if url else ""
            raise PrivateAddressBlocked(f"All resolved addresses for {hostname} are disallowed{target}")
        return ips

    @staticmethod
    def _normalize_host(host: str | None) -> str | None:
        """Normalize a host for comparisons."""
        if not host:
            return None
        normalized = host.rstrip(".").lower()
        return normalized or None

    def _hosts_related(self, origin: str | None, target: str | None) -> bool:
        """Determine if two hosts are related enough to allow redirects."""
        return self._policy.allow_redirect(origin, target)

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
        self, headers: Mapping[str, str] | None, host: str, port: int, scheme: str
    ) -> dict[str, str]:
        """Normalize headers, ensuring Host is set and not overridden."""
        host_value = host if self._is_default_port(scheme, port) else f"{host}:{port}"
        out: dict[str, str] = {"Host": host_value}
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
        data: bytes | None = None,
        timeout: float | None = None,
        redirect_log: list[tuple[str, str, int]] | None = None,
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
        explicit_method = getattr(req_obj, "method", None)
        orig_data = getattr(req_obj, "data", None)
        payload = data if data is not None else orig_data
        method = req_obj.get_method()
        if payload is not None and explicit_method is None:
            method = "POST"
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
        data: bytes | None = None,
        timeout: float | None = None,
        retries: int = 0,
        backoff_base: float = 1.0,
        backoff_factor: float = 2.0,
        only_get_like: bool = True,
        redirect_log: list[tuple[str, str, int]] | None = None,
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
        explicit_method = getattr(req_obj, "method", None)
        orig_data = getattr(req_obj, "data", None)
        payload = data if data is not None else orig_data
        method = (req_obj.get_method() or "GET").upper()
        if payload is not None and explicit_method is None:
            method = "POST"
        if only_get_like and method not in {"GET", "HEAD", "OPTIONS", "TRACE"}:
            return self.open(req_obj, data=data, timeout=timeout)

        last_error: Exception | None = None
        attempts = max(0, int(retries)) + 1
        for attempt in range(attempts):
            try:
                return self.open(req_obj, data=data, timeout=timeout, redirect_log=redirect_log)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
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
        headers: Mapping[str, str] | None,
        body: bytes | None,
        timeout: float,
        redirects_remaining: int,
        origin_host: str | None,
        redirects_followed: int,
        redirect_log: list[tuple[str, str, int]] | None = None,
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
        last_error: Exception | None = None
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
                if not self._policy.allow_redirect_scheme(scheme, redirect_scheme):
                    raise RedirectBlocked(
                        f"Redirect blocked: scheme change from {scheme} to {redirect_scheme} not permitted"
                    )
                new_host = redirect_parts.hostname
                if not self._hosts_related(origin_host, new_host):
                    raise RedirectBlocked(
                        f"Redirect blocked: cross-host redirect from {origin_host} to {new_host}"
                    )
                new_method = "GET" if response.status in (301, 302, 303) else method
                redirected_headers = self._policy.redirect_headers(headers, host, new_host)
                if redirect_log is not None:
                    redirect_log.append((url, redirect_url, response.status))
                return self._request(
                    url=redirect_url,
                    method=new_method,
                    headers=redirected_headers,
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
SAFE_HTTP_CLIENT: SafeHttpClient | None = None


def get_global_http_client() -> SafeHttpClient:
    """Return the process-wide SafeHttpClient, creating a default if unset."""
    global SAFE_HTTP_CLIENT
    if SAFE_HTTP_CLIENT is None:
        SAFE_HTTP_CLIENT = SafeHttpClient(allowed_redirect_suffixes=("github.com",))
    return SAFE_HTTP_CLIENT


def set_global_http_client(client: SafeHttpClient | None) -> None:
    """Override or clear the module-level SAFE_HTTP_CLIENT helper."""
    global SAFE_HTTP_CLIENT
    SAFE_HTTP_CLIENT = client
