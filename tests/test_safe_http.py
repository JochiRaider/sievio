import socket
import time
import urllib.error
import urllib.request

import pytest

from sievio.core.safe_http import PrivateAddressBlocked, RedirectBlocked, SafeHttpClient


def test_safe_http_blocks_private_and_shared_ips(monkeypatch):
    def fake_getaddrinfo(host, port, *args, **kwargs):
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                6,
                "",
                ("10.0.0.1", port),
            ),
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                6,
                "",
                ("100.64.0.1", port),
            ),
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                6,
                "",
                ("127.0.0.1", port),
            )
        ]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    client = SafeHttpClient()
    with pytest.raises(PrivateAddressBlocked):
        client._resolve_ips("example.com")


def test_safe_http_allows_only_global_ips(monkeypatch):
    def fake_getaddrinfo(host, port, *args, **kwargs):
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                6,
                "",
                ("100.64.0.1", port),
            ),
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                6,
                "",
                ("93.184.216.34", port),
            ),
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                6,
                "",
                ("224.0.0.1", port),
            )
        ]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    client = SafeHttpClient()
    infos = client._resolve_ips("example.com")
    assert infos == ["93.184.216.34"]


@pytest.mark.parametrize(
    "src, dest, expected",
    [
        ("example.com", "example.com", True),
        ("example.com", "www.example.com", True),
        ("www.example.com", "example.com", True),
        ("sub.example.com", "example.com", True),
        ("example.com", "sub.example.net", False),
        ("github.com", "docs.github.com", True),
        ("example.com", "malicious.com", False),
        ("github.com", "github.com.attacker.com", False),
        ("example.com", "example.com.attacker.net", False),
        ("api.github.com", "codeload.github.com", True),
        ("a.co.uk", "b.co.uk", False),
        ("a.com.au", "b.com.au", False),
        ("example.com.", "example.com", True),
        ("example.com", "example.com.", True),
        ("sub.example.com", "example", False),
        ("example.com", None, False),
        (None, "example.com", False),
    ],
)
def test_hosts_related(src, dest, expected):
    client = SafeHttpClient(allowed_redirect_suffixes=("github.com",))
    assert client._hosts_related(src, dest) is expected


def test_open_data_overrides_request_data(monkeypatch):
    client = SafeHttpClient()
    captured = {}

    def fake_request(**kwargs):
        captured.update(kwargs)

        class DummyResponse:
            pass

        return DummyResponse()

    monkeypatch.setattr(client, "_request", fake_request)

    req = urllib.request.Request("http://example.com/path", data=b"original")
    client.open(req, data=b"override")

    assert captured["body"] == b"override"
    assert captured["method"] == "POST"


def test_open_url_with_data_defaults_to_post(monkeypatch):
    client = SafeHttpClient()
    captured = {}

    def fake_request(**kwargs):
        captured.update(kwargs)

        class DummyResponse:
            pass

        return DummyResponse()

    monkeypatch.setattr(client, "_request", fake_request)

    client.open("http://example.com/path", data=b"payload")

    assert captured["body"] == b"payload"
    assert captured["method"] == "POST"


def test_open_collects_redirect_log(monkeypatch):
    client = SafeHttpClient()
    responses = []

    class FakeResponse:
        def __init__(self, status, headers):
            self.status = status
            self.headers = headers
            self.reason = "OK"

        def getheader(self, name, default=None):
            return self.headers.get(name, default)

        def close(self):
            pass

    class FakeConnection:
        def __init__(self):
            self._response = responses.pop(0)

        def request(self, *args, **kwargs):
            return None

        def getresponse(self):
            return self._response

        def close(self):
            pass

    def fake_resolve(hostname, url=None):
        return ["8.8.8.8"]

    monkeypatch.setattr(client, "_resolve_ips", fake_resolve)
    monkeypatch.setattr(client, "_build_connection", lambda **kwargs: FakeConnection())

    responses.extend(
        [
            FakeResponse(302, {"Location": "https://example.com/next"}),
            FakeResponse(200, {}),
        ]
    )

    redirects: list[tuple[str, str, int]] = []
    resp = client.open("http://example.com/start", timeout=1, redirect_log=redirects)

    assert resp.status == 200
    assert redirects == [("http://example.com/start", "https://example.com/next", 302)]
    assert resp.redirects == tuple(redirects)


def test_cross_public_suffix_redirect_blocked(monkeypatch):
    client = SafeHttpClient()
    responses = []

    class FakeResponse:
        def __init__(self, status, headers):
            self.status = status
            self.headers = headers
            self.reason = "OK"

        def getheader(self, name, default=None):
            return self.headers.get(name, default)

        def close(self):
            pass

    class FakeConnection:
        def __init__(self):
            self._response = responses.pop(0)

        def request(self, *args, **kwargs):
            return None

        def getresponse(self):
            return self._response

        def close(self):
            pass

    monkeypatch.setattr(client, "_resolve_ips", lambda hostname, url=None: ["8.8.8.8"])
    monkeypatch.setattr(client, "_build_connection", lambda **kwargs: FakeConnection())

    responses.append(FakeResponse(302, {"Location": "https://b.co.uk/next"}))

    with pytest.raises(RedirectBlocked):
        client.open("http://a.co.uk/start", timeout=1)


def test_redirect_blocks_https_to_http(monkeypatch):
    client = SafeHttpClient()
    responses = []

    class FakeResponse:
        def __init__(self, status, headers):
            self.status = status
            self.headers = headers
            self.reason = "OK"

        def getheader(self, name, default=None):
            return self.headers.get(name, default)

        def close(self):
            pass

    class FakeConnection:
        def __init__(self):
            self._response = responses.pop(0)

        def request(self, *args, **kwargs):
            return None

        def getresponse(self):
            return self._response

        def close(self):
            pass

    monkeypatch.setattr(client, "_resolve_ips", lambda hostname, url=None: ["93.184.216.34"])
    monkeypatch.setattr(client, "_build_connection", lambda **kwargs: FakeConnection())

    responses.append(FakeResponse(302, {"Location": "http://example.com/next"}))

    with pytest.raises(RedirectBlocked):
        client.open("https://example.com/start", timeout=1)


def test_open_with_retries_data_disables_get_like(monkeypatch):
    client = SafeHttpClient()
    call_count = 0

    def fake_open(request, data=None, timeout=None, redirect_log=None):
        nonlocal call_count
        call_count += 1
        return "ok"

    monkeypatch.setattr(client, "open", fake_open)

    result = client.open_with_retries("http://example.com/path", data=b"payload", retries=3, only_get_like=True)

    assert result == "ok"
    assert call_count == 1


def test_open_with_retries_get_like_still_retries(monkeypatch):
    client = SafeHttpClient()
    call_count = 0

    def fake_open(request, data=None, timeout=None, redirect_log=None):
        nonlocal call_count
        call_count += 1
        raise urllib.error.URLError("boom")

    monkeypatch.setattr(client, "open", fake_open)
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    with pytest.raises(urllib.error.URLError):
        client.open_with_retries("http://example.com/path", retries=2, only_get_like=True)

    assert call_count == 3


def test_redirect_strips_sensitive_headers_on_host_change(monkeypatch):
    client = SafeHttpClient()
    responses = []
    seen_headers = []

    class FakeResponse:
        def __init__(self, status, headers):
            self.status = status
            self.headers = headers
            self.reason = "OK"

        def getheader(self, name, default=None):
            return self.headers.get(name, default)

        def close(self):
            pass

    class FakeConnection:
        def __init__(self):
            self._response = responses.pop(0)
            self._headers = None

        def request(self, _method, _path, body=None, headers=None):
            self._headers = headers
            seen_headers.append(headers)
            return None

        def getresponse(self):
            return self._response

        def close(self):
            pass

    def fake_resolve(hostname, url=None):
        return ["93.184.216.34"]

    monkeypatch.setattr(client, "_resolve_ips", fake_resolve)
    monkeypatch.setattr(client, "_build_connection", lambda **kwargs: FakeConnection())

    responses.extend(
        [
            FakeResponse(302, {"Location": "https://other.example.com/next"}),
            FakeResponse(200, {}),
        ]
    )

    headers = {"Authorization": "secret", "Cookie": "session", "X-Test": "1"}
    req = urllib.request.Request("http://example.com/start", headers=headers)
    resp = client.open(req, timeout=1, redirect_log=[])

    assert resp.status == 200
    assert len(seen_headers) == 2
    first = {k.lower(): v for k, v in seen_headers[0].items()}
    second = {k.lower(): v for k, v in seen_headers[1].items()}
    assert "authorization" in first  # initial request keeps header
    assert "cookie" not in second
    assert second.get("x-test") == "1"
