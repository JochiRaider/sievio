"""
Factory helpers for building clients, sources, sinks, and derived paths.

All helpers are deterministic, side-effect free, and keep construction logic
centralized so that orchestrators (runner/config) stay thin.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Callable,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from .interfaces import RepoContext, Record, Sink
from .sinks import JSONLSink, GzipJSONLSink, PromptTextSink
from .fs import PatternFileSource

if TYPE_CHECKING:  # pragma: no cover - type-only imports
    from .chunk import ChunkPolicy
    from .config import (
        GitHubSourceConfig,
        HttpConfig,
        LocalDirSourceConfig,
        QCConfig,
        SinkConfig,
    )
    from .qc import JSONLQualityScorer
    from .safe_http import SafeHttpClient

__all__ = [
    "BytesHandler",
    "OutputPaths",
    "SinkFactoryResult",
    "Sniff",
    "UnsupportedBinary",
    "build_default_sinks",
    "make_bytes_handlers",
    "make_github_zip_source",
    "make_http_client",
    "make_local_dir_source",
    "make_output_paths_for_github",
    "make_output_paths_for_pdf",
    "make_jsonl_text_source",
    "make_qc_scorer",
    "make_repo_context_from_git",
]

Sniff = Callable[[bytes, str], bool]
BytesHandler = Callable[
    [bytes, str, Optional[RepoContext], Optional["ChunkPolicy"]],
    Optional[Iterable[Record]],
]


@dataclass(frozen=True)
class SinkFactoryResult:
    jsonl_path: str
    sinks: Sequence[Sink]
    sink_config: "SinkConfig"
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class OutputPaths:
    """
    Bundles derived output locations for downstream consumers.
    """

    jsonl: Path
    prompt: Optional[Path] = None
    artifacts: Optional[Path] = None

    def as_tuple(self) -> Tuple[str, Optional[str]]:
        return str(self.jsonl), (str(self.prompt) if self.prompt else None)


class UnsupportedBinary(Exception):
    """Raised when a recognized binary handler is unavailable in this build."""


def build_default_sinks(
    cfg: "SinkConfig",
    basename: Optional[str] = None,
    *,
    jsonl_path: Optional[str | Path] = None,
    prompt_path: Optional[str | Path] = None,
    context: Optional[RepoContext] = None,
) -> SinkFactoryResult:
    """
    Build the canonical JSONL + prompt sinks for ``cfg``.

    Exactly one of ``basename`` or ``jsonl_path`` must be provided.  When a path
    is supplied explicitly, it takes precedence over ``cfg.output_dir``.
    """
    if basename and jsonl_path:
        raise ValueError("Provide either basename or jsonl_path, not both")

    if jsonl_path is None:
        base = basename or (cfg.jsonl_basename or None)
        if not base:
            raise ValueError("A basename or jsonl_path is required")
        suffix = ".jsonl.gz" if cfg.compress_jsonl else ".jsonl"
        jsonl_path = cfg.output_dir / f"{base}{suffix}"
    jsonl_path = Path(jsonl_path)
    jsonl_str = str(jsonl_path)

    use_gzip = cfg.compress_jsonl or jsonl_str.endswith(".gz")
    sink_class = GzipJSONLSink if use_gzip else JSONLSink
    sinks: list[Sink] = [sink_class(jsonl_str)]

    prompt_target: Optional[str]
    if prompt_path is not None:
        prompt_target = str(Path(prompt_path))
    elif cfg.prompt.include_prompt_file:
        prompt_target = str(_default_prompt_path(jsonl_path))
    else:
        prompt_target = None

    if prompt_target:
        sinks.append(PromptTextSink(prompt_target, heading_fmt=cfg.prompt.heading_fmt))

    effective_context = context if context is not None else cfg.context
    sink_cfg = replace(
        cfg,
        sinks=tuple(sinks),
        context=effective_context,
        primary_jsonl_name=jsonl_str,
    )
    metadata = {"primary_jsonl": jsonl_str}
    if prompt_target:
        metadata["prompt_path"] = prompt_target
    return SinkFactoryResult(
        jsonl_path=jsonl_str,
        sinks=sink_cfg.sinks,
        sink_config=sink_cfg,
        metadata=metadata,
    )


def _default_prompt_path(jsonl_path: Path) -> Path:
    name = jsonl_path.name
    if name.endswith(".jsonl.gz"):
        base = name[:-len(".jsonl.gz")]
    else:
        base = jsonl_path.stem
    prompt_name = f"{base}.prompt.txt"
    return jsonl_path.parent / prompt_name


def make_jsonl_text_source(
    paths: Sequence[str | Path],
    *,
    context: Optional[RepoContext] = None,
    text_key: str = "text",
):
    from .jsonl_source import JSONLTextSource

    norm_paths = [Path(p) for p in paths]
    return JSONLTextSource(paths=tuple(norm_paths), context=context, text_key=text_key)


def make_pattern_file_source(
    root: str | Path,
    patterns: Sequence[str],
    *,
    config: "LocalDirSourceConfig",
    context: Optional[RepoContext] = None,
) -> PatternFileSource:
    return PatternFileSource(root, patterns, config=config, context=context)


# ---------------------------------------------------------------------------
# HTTP / QC factories
# ---------------------------------------------------------------------------

def make_http_client(http_cfg: "HttpConfig") -> "SafeHttpClient":
    """
    Build (or reuse) the SafeHttpClient described by ``http_cfg``.
    """
    if http_cfg is None:
        raise ValueError("http_cfg is required")
    if http_cfg.client is not None:
        return http_cfg.client
    from .safe_http import SafeHttpClient  # local import to avoid cycles

    return SafeHttpClient(
        timeout=http_cfg.timeout,
        max_redirects=http_cfg.max_redirects,
        allowed_redirect_suffixes=http_cfg.allowed_redirect_suffixes,
    )


def make_qc_scorer(qc_cfg: Optional["QCConfig"], *, new_instance: bool = False) -> Optional["JSONLQualityScorer"]:
    """
    Instantiate a JSONLQualityScorer when QC is enabled and extras are present.
    """
    if qc_cfg is None or not getattr(qc_cfg, "enabled", False):
        return None
    existing = getattr(qc_cfg, "scorer", None)
    if existing is not None and not new_instance:
        return existing
    try:
        from .qc import JSONLQualityScorer  # optional extra
    except Exception:
        return None
    scorer = JSONLQualityScorer()
    if not new_instance:
        qc_cfg.scorer = scorer
    return scorer


# ---------------------------------------------------------------------------
# Bytes-handler factory (PDF/EVTX)
# ---------------------------------------------------------------------------

def make_bytes_handlers() -> Sequence[Tuple[Sniff, BytesHandler]]:
    """
    Return the default sniff/handler pairs for binary formats (PDF/EVTX).
    """
    try:
        from .pdfio import sniff_pdf, handle_pdf  # type: ignore
    except Exception:
        sniff_pdf, handle_pdf = _fallback_sniff_pdf, _fallback_handle_pdf

    try:
        from .evtxio import sniff_evtx, handle_evtx  # type: ignore
    except Exception:
        sniff_evtx, handle_evtx = _fallback_sniff_evtx, _fallback_handle_evtx

    return [
        (sniff_pdf, handle_pdf),
        (sniff_evtx, handle_evtx),
    ]


def _fallback_sniff_pdf(data: bytes, rel: str) -> bool:
    return rel.lower().endswith(".pdf") or data.startswith(b"%PDF-")


def _fallback_handle_pdf(
    data: bytes,
    rel: str,
    ctx: Optional[RepoContext],
    policy: Optional["ChunkPolicy"],
) -> Optional[Iterable[Record]]:
    raise UnsupportedBinary("pdf support is not installed")


def _fallback_sniff_evtx(data: bytes, rel: str) -> bool:
    name = rel.lower()
    if name.endswith(".evtx"):
        return True
    if data.startswith(b"ElfFile"):
        return True
    if b"ElfChnk" in data[:1_048_576]:
        return True
    return False


def _fallback_handle_evtx(
    data: bytes,
    rel: str,
    ctx: Optional[RepoContext],
    policy: Optional["ChunkPolicy"],
) -> Optional[Iterable[Record]]:
    raise UnsupportedBinary("evtx support is not installed")


# ---------------------------------------------------------------------------
# Repo/source/context helpers
# ---------------------------------------------------------------------------

def make_repo_context_from_git(repo_root: Path | str) -> Optional[RepoContext]:
    """
    Infer a RepoContext from ``.git/config`` if the remote points at GitHub.
    Returns ``None`` when no usable metadata is present.
    """
    cfg_path = Path(repo_root) / ".git" / "config"
    try:
        text = cfg_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    current_remote: Optional[str] = None
    origin_url: Optional[str] = None
    fallback_url: Optional[str] = None
    remote_header = re.compile(r'\s*\[remote\s+"([^"]+)"\]')
    url_line = re.compile(r"^\s*url\s*=\s*([^\r\n]+)$")

    for line in text.splitlines():
        header = remote_header.match(line)
        if header:
            current_remote = header.group(1)
            continue
        if current_remote is None:
            continue
        m = url_line.match(line)
        if not m:
            continue
        url_value = m.group(1).strip()
        if current_remote == "origin":
            origin_url = url_value
            break
        if fallback_url is None:
            fallback_url = url_value

    remote = origin_url or fallback_url
    if not remote:
        return None
    from .githubio import parse_github_url  # local import to avoid cycles

    spec = parse_github_url(remote)
    if not spec:
        return None
    return RepoContext(
        repo_full_name=f"{spec.owner}/{spec.repo}",
        repo_url=f"https://github.com/{spec.owner}/{spec.repo}",
        license_id=None,
        extra={"source": "local"},
    )


def make_local_dir_source(
    root: Path | str,
    *,
    config: "LocalDirSourceConfig",
    context: Optional[RepoContext] = None,
):
    """
    Build a LocalDirSource for ``root`` using the supplied config/context.
    """
    if config is None:
        raise ValueError("LocalDirSourceConfig is required")
    from .fs import LocalDirSource  # local import to break cycles

    return LocalDirSource(root, config=config, context=context)


def make_github_zip_source(
    url: str,
    *,
    config: "GitHubSourceConfig",
    context: Optional[RepoContext],
    download_timeout: Optional[float],
):
    """
    Build a GitHubZipSource for ``url`` with the provided config/context.
    """
    if not url:
        raise ValueError("url is required for GitHubZipSource")
    from .githubio import GitHubZipSource  # local import to avoid cycles

    return GitHubZipSource(
        url,
        config=config,
        context=context,
        download_timeout=download_timeout,
    )


# ---------------------------------------------------------------------------
# Output path helpers
# ---------------------------------------------------------------------------

def make_output_paths_for_github(
    *,
    owner: str,
    repo: str,
    ref: Optional[str],
    license_spdx: Optional[str],
    out_dir: Path | str,
    include_prompt: bool = True,
    timestamp: Optional[str] = None,
    include_commit: Optional[str] = None,
) -> OutputPaths:
    """
    Build output paths for a GitHub dataset, optionally appending ``timestamp``.
    """
    if not owner or not repo:
        raise ValueError("owner and repo are required for GitHub output paths")
    from .naming import build_output_basename_github

    base = build_output_basename_github(
        owner=owner,
        repo=repo,
        ref=ref or "main",
        license_spdx=license_spdx,
        include_commit=include_commit,
    )
    base = _append_timestamp(base, timestamp)
    out_dir = _normalize_out_dir(out_dir)
    jsonl = out_dir / f"{base}.jsonl"
    prompt = (out_dir / f"{base}.prompt.txt") if include_prompt else None
    return OutputPaths(jsonl=jsonl, prompt=prompt)


def make_output_paths_for_pdf(
    *,
    url: str,
    title: Optional[str],
    license_spdx: Optional[str],
    out_dir: Path | str,
    include_prompt: bool = True,
    timestamp: Optional[str] = None,
) -> OutputPaths:
    """
    Build output paths for a PDF corpus using URL/title/license metadata.
    """
    if not url:
        raise ValueError("url is required for PDF output paths")
    from .naming import build_output_basename_pdf

    base = build_output_basename_pdf(url=url, title=title, license_spdx=license_spdx)
    base = _append_timestamp(base, timestamp)
    out_dir = _normalize_out_dir(out_dir)
    jsonl = out_dir / f"{base}.jsonl"
    prompt = (out_dir / f"{base}.prompt.txt") if include_prompt else None
    return OutputPaths(jsonl=jsonl, prompt=prompt)


def _normalize_out_dir(out_dir: Path | str) -> Path:
    return Path(out_dir).expanduser()


def _append_timestamp(base: str, timestamp: Optional[str]) -> str:
    if not timestamp:
        return base
    cleaned = re.sub(r"[^\w\-]+", "_", timestamp.strip())
    cleaned = re.sub(r"_{2,}", "_", cleaned).strip("_")
    if not cleaned:
        return base
    return f"{base}__{cleaned}"
