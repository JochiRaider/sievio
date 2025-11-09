from __future__ import annotations

# Allow running from a source checkout without installing the package.
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))


from repocapsule.sources_web import WebPdfListSource, WebPagePdfSource
from repocapsule.pipeline import run_pipeline
from repocapsule.runner import _JSONLSink  # or copy that tiny sink locally
from repocapsule.interfaces import RepoContext
from repocapsule.chunk import ChunkPolicy

urls = [
    "https://example.org/whitepaper.pdf",
    "https://example.org/other.pdf",
]

src = WebPdfListSource(urls, max_pdf_bytes=100*1024*1024, require_pdf=True, add_prefix="webpdfs")


src = WebPagePdfSource(
    "https://example.org/resources",
    same_domain=True,
    max_links=50,
    include_ambiguous=False,  # set True if the site hides PDFs behind non-.pdf URLs
    add_prefix="webpdfs"
)

ctx = RepoContext(repo_full_name=None, repo_url=None, license_id=None)
stats = run_pipeline(
    source=src,
    sinks=[_JSONLSink("out/webpdfs.jsonl")],
    policy=ChunkPolicy(mode="doc", target_tokens=1700, overlap_tokens=40, min_tokens=400),
)
print(stats)
