# config.py
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Tuple

from .chunk import ChunkPolicy
from .factories import make_bytes_handlers, make_http_client, make_qc_scorer
from .interfaces import Extractor, RepoContext, Source, Record
from .log import configure_logging
from .safe_http import SafeHttpClient, set_global_http_client


# Local copies of the bytes-handler type aliases to avoid circular imports at runtime.
Sniff = Callable[[bytes, str], bool]
BytesHandler = Callable[[bytes, str, Optional[RepoContext], Optional[ChunkPolicy]], Optional[Iterable[Record]]]


def _default_bytes_handlers() -> Sequence[Tuple[Sniff, BytesHandler]]:
    return tuple(make_bytes_handlers())

# ---------------------------------------------------------------------------
# Source configs
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class LocalDirSourceConfig:
    include_exts: Optional[set[str]] = None
    exclude_exts: Optional[set[str]] = None
    skip_hidden: bool = True
    follow_symlinks: bool = False
    respect_gitignore: bool = True
    max_file_bytes: Optional[int] = 200 * 1024 * 1024


@dataclass(slots=True)
class GitHubSourceConfig:
    per_file_cap: Optional[int] = 200 * 1024 * 1024
    max_total_uncompressed: int = 2 * 1024 * 1024 * 1024
    max_members: int = 200_000
    max_compression_ratio: float = 100.0
    include_exts: Optional[set[str]] = None
    exclude_exts: Optional[set[str]] = None


@dataclass(slots=True)
class PdfSourceConfig:
    timeout: int = 60
    max_pdf_bytes: int = 200 * 1024 * 1024
    max_links: int = 200
    require_pdf: bool = True
    include_ambiguous: bool = False
    retries: int = 1
    user_agent: str = "repocapsule/0.1 (+https://github.com)"


@dataclass(slots=True)
class SourceConfig:
    sources: Sequence[Source] = field(default_factory=tuple)
    local: LocalDirSourceConfig = field(default_factory=LocalDirSourceConfig)
    github: GitHubSourceConfig = field(default_factory=GitHubSourceConfig)
    pdf: PdfSourceConfig = field(default_factory=PdfSourceConfig)


# ---------------------------------------------------------------------------
# Decode / chunk / pipeline
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class DecodeConfig:
    normalize: Optional[str] = "NFC"
    strip_controls: bool = True
    fix_mojibake: bool = True
    max_bytes_per_file: Optional[int] = None


@dataclass(slots=True)
class ChunkConfig:
    policy: ChunkPolicy = field(default_factory=ChunkPolicy)
    tokenizer_name: Optional[str] = None
    attach_language_metadata: bool = True


@dataclass(slots=True)
class PipelineConfig:
    extractors: Sequence[Extractor] = field(default_factory=tuple)
    bytes_handlers: Sequence[Tuple[Sniff, BytesHandler]] = field(
        default_factory=_default_bytes_handlers
    )
    max_workers: int = 0
    submit_window: Optional[int] = None
    fail_fast: bool = False


# ---------------------------------------------------------------------------
# Sinks / naming
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PromptConfig:
    heading_fmt: str = "### {path} [chunk {chunk}]"
    include_prompt_file: bool = True


@dataclass(slots=True)
class SinkConfig:
    sinks: Sequence = field(default_factory=tuple)
    context: Optional[RepoContext] = None
    output_dir: Path = Path(".")
    primary_jsonl_name: Optional[str] = None
    prompt: PromptConfig = field(default_factory=PromptConfig)


# ---------------------------------------------------------------------------
# HTTP / PDF / Web configs
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class HttpConfig:
    timeout: float = 60.0
    max_redirects: int = 5
    allowed_redirect_suffixes: Tuple[str, ...] = ("github.com",)
    client: Optional[SafeHttpClient] = None


# ---------------------------------------------------------------------------
# QC / logging / misc
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class QCConfig:
    enabled: bool = False
    write_csv: bool = False
    csv_suffix: str = "_quality.csv"
    scorer: Optional[Any] = None  # optional extra
    fail_on_error: bool = False
    min_score: Optional[float] = 60.0
    drop_near_dups: bool = False
    mode: str = "inline"  # "inline" or "post"


@dataclass(slots=True)
class LoggingConfig:
    level: int | str = "INFO"
    propagate: bool = False
    fmt: Optional[str] = "%(asctime)s %(levelname)s %(name)s: %(message)s"

    def apply(self) -> None:
        configure_logging(level=self.level, propagate=self.propagate, fmt=self.fmt)


# ---------------------------------------------------------------------------
# Master config
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class RepocapsuleConfig:
    sources: SourceConfig = field(default_factory=SourceConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    sinks: SinkConfig = field(default_factory=SinkConfig)
    http: HttpConfig = field(default_factory=HttpConfig)
    qc: QCConfig = field(default_factory=QCConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def with_context(self, ctx: RepoContext) -> RepocapsuleConfig:
        return replace(self, sinks=replace(self.sinks, context=ctx))

    def prepare(self) -> None:
        self.logging.apply()
        client = make_http_client(self.http)
        self.http.client = client
        set_global_http_client(client)
        if not self.pipeline.bytes_handlers:
            self.pipeline.bytes_handlers = tuple(make_bytes_handlers())
        if self.qc.enabled and not self.qc.scorer:
            scorer = make_qc_scorer(self.qc)
            if scorer is None:
                raise RuntimeError("QC extras not installed; disable QC or install optional deps.")
            self.qc.scorer = scorer
        if self.qc.enabled and self.qc.scorer is None:
            raise RuntimeError("QC extras not installed; disable QC or install optional deps.")
        mode = (self.qc.mode or "inline").lower()
        if mode not in {"inline", "post"}:
            raise ValueError(f"Invalid qc.mode {self.qc.mode!r}; expected 'inline' or 'post'")
        self.qc.mode = mode


__all__ = ["RepocapsuleConfig"]
