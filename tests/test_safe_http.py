import socket

import pytest

from sievio.core.safe_http import PrivateAddressBlocked, SafeHttpClient


def test_safe_http_blocks_private_ip(monkeypatch):
    def fake_getaddrinfo(host, port, *args, **kwargs):
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                6,
                "",
                ("10.0.0.1", port),
            )
        ]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    client = SafeHttpClient()
    with pytest.raises(PrivateAddressBlocked):
        client._resolve_ips("example.com")


def test_safe_http_allows_public_ip(monkeypatch):
    def fake_getaddrinfo(host, port, *args, **kwargs):
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                6,
                "",
                ("8.8.8.8", port),
            )
        ]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    client = SafeHttpClient()
    infos = client._resolve_ips("example.com")
    assert infos == ["8.8.8.8"]


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
    ],
)
def test_hosts_related(src, dest, expected):
    client = SafeHttpClient(allowed_redirect_suffixes=("github.com",))
    assert client._hosts_related(src, dest) is expected


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
