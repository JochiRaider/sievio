# config.py
from __future__ import annotations

import json
from collections.abc import Sequence as ABCSequence
from dataclasses import dataclass, field, replace, is_dataclass, fields
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Type, TypeVar, Union, get_args, get_origin

from .chunk import ChunkPolicy
from .factories import make_bytes_handlers, make_http_client, make_qc_scorer
from .interfaces import Extractor, RepoContext, Source, Record
from .log import configure_logging, PACKAGE_LOGGER_NAME
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
    executor_kind: str = "thread"


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
    compress_jsonl: bool = False
    jsonl_basename: str = "data"


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
    mode: str = "inline"
    parallel_post: bool = False


@dataclass(slots=True)
class LoggingConfig:
    """Controls the package logger; set propagate=True/logger_name to integrate with host apps."""
    level: int | str = "INFO"
    propagate: bool = False
    fmt: Optional[str] = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logger_name: str = PACKAGE_LOGGER_NAME

    def apply(self) -> None:
        configure_logging(
            level=self.level,
            propagate=self.propagate,
            fmt=self.fmt,
            logger_name=self.logger_name or PACKAGE_LOGGER_NAME,
        )


# ---------------------------------------------------------------------------
# Master config
# ---------------------------------------------------------------------------

T = TypeVar("T")


@dataclass(slots=True)
class RunMetadata:
    primary_jsonl: Optional[str] = None
    prompt_path: Optional[str] = None
    repo_url: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        if self.primary_jsonl is not None:
            data["primary_jsonl"] = self.primary_jsonl
        if self.prompt_path is not None:
            data["prompt_path"] = self.prompt_path
        if self.repo_url is not None:
            data["repo_url"] = self.repo_url
        if self.extra:
            data["extra"] = dict(self.extra)
        return data

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "RunMetadata":
        if not data:
            return cls()
        extra = dict(data.get("extra") or {})
        known = {k: data.get(k) for k in ("primary_jsonl", "prompt_path", "repo_url")}
        return cls(extra=extra, **{k: v for k, v in known.items() if v is not None})

    def merged(self, mapping: Mapping[str, Any]) -> "RunMetadata":
        updated = RunMetadata(
            primary_jsonl=self.primary_jsonl,
            prompt_path=self.prompt_path,
            repo_url=self.repo_url,
            extra=dict(self.extra),
        )
        for key, value in mapping.items():
            if key == "primary_jsonl":
                if not updated.primary_jsonl and value:
                    updated.primary_jsonl = value  # type: ignore[assignment]
            elif key == "prompt_path":
                if not updated.prompt_path and value:
                    updated.prompt_path = value  # type: ignore[assignment]
            elif key == "repo_url":
                if not updated.repo_url and value:
                    updated.repo_url = value  # type: ignore[assignment]
            else:
                updated.extra.setdefault(key, value)
        return updated

    def get(self, key: str, default: Any = None) -> Any:
        if key == "primary_jsonl":
            return self.primary_jsonl if self.primary_jsonl is not None else default
        if key == "prompt_path":
            return self.prompt_path if self.prompt_path is not None else default
        if key == "repo_url":
            return self.repo_url if self.repo_url is not None else default
        return self.extra.get(key, default)


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
    metadata: RunMetadata = field(default_factory=RunMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, RunMetadata):
            self.metadata = RunMetadata.from_dict(dict(self.metadata or {}))

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
        allowed_modes = {"inline", "post", "off", "advisory"}
        if mode not in allowed_modes:
            raise ValueError(f"Invalid qc.mode {self.qc.mode!r}; expected one of {sorted(allowed_modes)}")
        if mode == "off":
            self.qc.enabled = False
        self.qc.mode = mode
        self.validate()

    def validate(self) -> None:
        meta = self.metadata
        p_jsonl = meta.primary_jsonl
        prompt = meta.prompt_path
        if p_jsonl and prompt:
            try:
                if Path(p_jsonl).resolve() == Path(prompt).resolve():
                    raise ValueError("primary_jsonl and prompt_path refer to the same file path.")
            except Exception:
                if p_jsonl == prompt:
                    raise ValueError("primary_jsonl and prompt_path refer to the same file path.")

        if self.qc.enabled and self.qc.mode not in {"inline", "post", "advisory"}:
            raise ValueError("QC enabled but mode is set to 'off'; disable QC or change mode.")

        if self.qc.enabled and self.qc.mode in {"inline", "advisory"} and not self.qc.scorer:
            raise ValueError("QC mode requires a scorer; ensure qc.scorer is configured or extras installed.")

        if self.pipeline.executor_kind not in {"thread", "process"}:
            raise ValueError("pipeline.executor_kind must be 'thread' or 'process'.")

    # -------------------------
    # Serialization helpers
    # -------------------------
    def to_dict(self) -> Dict[str, Any]:
        return _dataclass_to_dict(self)

    def to_json(self, path: Path | str, *, indent: int = 2) -> str:
        data = self.to_dict()
        target = Path(path)
        target.write_text(json.dumps(data, indent=indent, sort_keys=True), encoding="utf-8")
        return str(target)

    @classmethod
    def from_dict(cls: Type[T], data: Mapping[str, Any]) -> T:
        return _dataclass_from_dict(cls, data)

    @classmethod
    def from_json(cls: Type[T], path: Path | str) -> T:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)


_SKIP_FIELDS: Dict[Type[Any], set[str]] = {
    SourceConfig: {"sources"},
    SinkConfig: {"sinks"},
    PipelineConfig: {"extractors", "bytes_handlers"},
    HttpConfig: {"client"},
    QCConfig: {"scorer"},
}


def _dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    obj_type = type(obj)
    skip = _SKIP_FIELDS.get(obj_type, set())
    for f in fields(obj):
        if f.name in skip:
            continue
        value = getattr(obj, f.name)
        if value is None:
            continue
        serialized = _serialize_value(value)
        if serialized is not None:
            result[f.name] = serialized
    return result


def _serialize_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        items = [_serialize_value(v) for v in value]
        try:
            return sorted(items)
        except Exception:
            return items
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            serialized = _serialize_value(v)
            if serialized is not None:
                out[str(k)] = serialized
        return out
    if is_dataclass(value):
        return _dataclass_to_dict(value)
    return value if isinstance(value, (dict, list)) else None


def _dataclass_from_dict(cls: Type[T], data: Mapping[str, Any] | None) -> T:
    if data is None:
        return cls()  # type: ignore[call-arg]
    kwargs: Dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        kwargs[f.name] = _coerce_value(f.type, data[f.name])
    return cls(**kwargs)  # type: ignore[arg-type]


def _coerce_value(expected_type: Any, value: Any) -> Any:
    base_type, _ = _strip_optional(expected_type)
    if value is None:
        return None
    if is_dataclass_type(base_type):
        return _dataclass_from_dict(base_type, value)
    origin = get_origin(base_type)
    if origin in (list, tuple, ABCSequence):
        args = get_args(base_type)
        inner = args[0] if args else Any
        items = [_coerce_value(inner, v) for v in value]
        return tuple(items) if origin is tuple else list(items)
    if origin in (set, frozenset):
        args = get_args(base_type)
        inner = args[0] if args else Any
        return {_coerce_value(inner, v) for v in value}
    if origin is dict:
        key_type, val_type = get_args(base_type) if get_args(base_type) else (Any, Any)
        return { _coerce_value(key_type, k): _coerce_value(val_type, v) for k, v in value.items() }
    if base_type is Path:
        return Path(value)
    if base_type in {str, int, float, bool}:
        return base_type(value)
    return value


def _strip_optional(typ: Any) -> Tuple[Any, bool]:
    origin = get_origin(typ)
    if origin is Union:
        args = [arg for arg in get_args(typ) if arg is not type(None)]
        if len(args) == 1:
            inner = args[0]
            base, _ = _strip_optional(inner)
            return base, True
    return typ, False


def is_dataclass_type(typ: Any) -> bool:
    try:
        return isinstance(typ, type) and is_dataclass(typ)
    except Exception:
        return False



__all__ = ["RepocapsuleConfig", "RunMetadata"]