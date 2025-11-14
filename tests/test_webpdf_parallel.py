from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List

from repocapsule.sources_webpdf import WebPdfListSource
from repocapsule.config import PdfSourceConfig


@dataclass
class _StubResponse:
    data: bytes
    headers: Dict[str, str]


class DummyWebPdfSource(WebPdfListSource):
    def __init__(self, urls: List[str]) -> None:
        cfg = PdfSourceConfig()
        super().__init__(urls, config=cfg, require_pdf=True)
        self._payloads: Dict[str, _StubResponse] = {
            url: _StubResponse(
                data=b"%PDF-1.4\n" + url.encode("utf-8"),
                headers={"Content-Disposition": f'attachment; filename="{url.split("/")[-1]}"'},
            )
            for url in urls
        }

    def _download(self, url: str) -> Tuple[bytes, Dict[str, str]]:  # type: ignore[override]
        resp = self._payloads[url]
        return resp.data, dict(resp.headers)


def test_web_pdf_concurrency_preserves_order_and_names():
    urls = [
        "https://example.com/a.pdf",
        "https://example.com/b.pdf",
        "https://example.com/c.pdf",
    ]
    src = DummyWebPdfSource(urls)
    items = list(src.iter_files())
    assert [item.path for item in items] == ["a.pdf", "b.pdf", "c.pdf"]
    assert all(item.data.startswith(b"%PDF-") for item in items)
