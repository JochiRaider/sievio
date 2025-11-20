# config.py
from __future__ import annotations

import json
try:  # pragma: no cover - optional dependency
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    try:
        import tomli as tomllib  # type: ignore[assignment]
    except Exception:  # pragma: no cover
        tomllib = None  # type: ignore[assignment]
from collections.abc import Sequence as ABCSequence
from dataclasses import dataclass, field, replace, is_dataclass, fields
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Type, TypeVar, Union, get_args, get_origin, get_type_hints, List

from .chunk import ChunkPolicy
from .interfaces import Extractor, FileExtractor, RepoContext, Source, Sink, Record, QualityScorer
from .log import configure_logging, PACKAGE_LOGGER_NAME
from .safe_http import SafeHttpClient

# Local copies of the bytes-handler type aliases to avoid circular imports at runtime.
Sniff = Callable[[bytes, str], bool]
BytesHandler = Callable[[bytes, str, Optional[RepoContext], Optional[ChunkPolicy]], Optional[Iterable[Record]]]
# ---------------------------------------------------------------------------
# QC mode helpers
# ---------------------------------------------------------------------------


class QCMode:
    """
    Supported quality-control modes.

    OFF:
        Disable QC entirely (no scoring, no annotations).
    INLINE:
        Score records during extraction and enforce gating (records may be dropped).
    ADVISORY:
        Score records inline but never drop them; annotations are for review only.
    POST:
        Run QC after the pipeline completes (no inline annotations or gating).
    """

    INLINE = "inline"
    POST = "post"
    ADVISORY = "advisory"
    OFF = "off"
    ALL = {INLINE, POST, ADVISORY}
    WITH_OFF = ALL | {OFF}

    @classmethod
    def normalize(cls, value: Optional[str]) -> str:
        mode = (value or cls.INLINE).strip().lower()
        if mode not in cls.WITH_OFF:
            raise ValueError(f"Invalid QC mode: {value!r}. Expected one of {sorted(cls.WITH_OFF)}")
        return mode

    @classmethod
    def is_inline(cls, mode: str) -> bool:
        return mode == cls.INLINE

    @classmethod
    def is_post(cls, mode: str) -> bool:
        return mode == cls.POST

    @classmethod
    def is_advisory(cls, mode: str) -> bool:
        return mode == cls.ADVISORY


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
    read_prefix_bytes: Optional[int] = None
    read_prefix_for_large_files_only: bool = True


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
    """
    Settings for web PDF fetching; download_* controls concurrency for WebPdfListSource/WebPagePdfSource only.

    download_executor_kind:
        "thread" only; "process" is currently not supported for web downloads and will
        fall back to "thread" with a warning.
    """
    timeout: int = 60
    max_pdf_bytes: int = 200 * 1024 * 1024  # Streaming cap; responses exceeding it are aborted.
    max_links: int = 200
    require_pdf: bool = True
    include_ambiguous: bool = False
    retries: int = 1
    user_agent: str = "repocapsule/0.1 (+https://github.com)"
    client: Optional[SafeHttpClient] = None
    download_max_workers: int = 4  # 0 or negative → auto based on URL count
    download_submit_window: Optional[int] = None
    download_executor_kind: str = "thread"


@dataclass(slots=True)
class SourceConfig:
    specs: List["SourceSpec"] = field(default_factory=list)
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
    max_bytes_per_file: Optional[int] = None  # Bytes passed to the decoder per file (soft cap).


@dataclass(slots=True)
class ChunkConfig:
    policy: ChunkPolicy = field(default_factory=ChunkPolicy)
    tokenizer_name: Optional[str] = None
    attach_language_metadata: bool = True


@dataclass(slots=True)
class PipelineConfig:
    """
    Controls pipeline concurrency and processing behavior.

    max_workers = 0 → auto (os.cpu_count or 1)
    submit_window = None → defaults to max_workers * 4
    executor_kind ∈ {"thread", "process", "auto"}
      - "thread": good for I/O-heavy or small text/code workloads.
      - "process": recommended for CPU-bound handlers such as PDF (pdfio) or EVTX (evtxio).
      - "auto": let the pipeline choose based on configured sources/handlers (defaults to
        threads for mostly text/code; switches to processes when PDF/EVTX-heavy).

    Where this applies:
    - main extraction pipeline (file decoding/chunking/sink writes)
    - defaults for post-QC when QCConfig.parallel_post is True and no QC overrides are set
    Does not control web PDF fetch concurrency (see PdfSourceConfig.download_*).
    """

    extractors: Sequence[Extractor] = field(default_factory=tuple)
    file_extractor: Optional[FileExtractor] = None
    bytes_handlers: Sequence[Tuple[Sniff, BytesHandler]] = field(default_factory=tuple)
    max_workers: int = 0
    submit_window: Optional[int] = None
    fail_fast: bool = False
    executor_kind: str = "auto"


@dataclass(slots=True, frozen=True)
class FileProcessingConfig:
    """
    Lightweight view used by worker processes that only need per-file settings.
    """

    decode: DecodeConfig
    chunk: ChunkConfig
    pipeline: PipelineConfig


# ---------------------------------------------------------------------------
# Sinks / naming
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PromptConfig:
    heading_fmt: str = "### {path} [chunk {chunk}]"
    include_prompt_file: bool = True


@dataclass(slots=True)
class SinkConfig:
    specs: List["SinkSpec"] = field(default_factory=list)
    sinks: Sequence[Sink] = field(default_factory=tuple)
    context: Optional[RepoContext] = None
    output_dir: Path = Path(".")
    primary_jsonl_name: Optional[str] = None
    prompt: PromptConfig = field(default_factory=PromptConfig)
    compress_jsonl: bool = False
    jsonl_basename: str = "data"


@dataclass(slots=True)
class SourceSpec:
    """Declarative source entry; factories map kind -> concrete sources."""

    kind: str
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SinkSpec:
    """Declarative sink entry; factories map kind -> concrete sinks."""

    kind: str
    options: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# HTTP / PDF / Web configs
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class HttpConfig:
    """
    HTTP client settings used by higher-level helpers and factories.

    - ``client`` can hold a pre-built SafeHttpClient instance. When set, it will be reused by
      ``build_client()`` and passed to factories.
    - ``as_global`` controls whether the builder installs this client as the
      module-wide default via ``set_global_http_client``. For CLI-style one-shot runs, leaving this
      True is convenient. For tests or long-lived processes, prefer ``as_global=False`` and pass the
      client explicitly.
    """
    timeout: float = 60.0
    max_redirects: int = 5
    allowed_redirect_suffixes: Tuple[str, ...] = ("github.com",)
    client: Optional[SafeHttpClient] = None
    as_global: bool = True

    def build_client(self) -> SafeHttpClient:
        """Construct (or reuse) the SafeHttpClient for this config without mutating globals."""
        if self.client is not None:
            return self.client
        client = SafeHttpClient(
            timeout=self.timeout,
            max_redirects=self.max_redirects,
            allowed_redirect_suffixes=self.allowed_redirect_suffixes,
        )
        self.client = client
        return client


# ---------------------------------------------------------------------------
# QC / logging / misc
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class QCHeuristics:
    """
    Tunable thresholds/weights used by quality scoring. Defaults match existing hard-coded behavior.
    """

    target_code_min: int = 2000
    target_code_max: int = 4000
    target_log_min: int = 1000
    target_log_max: int = 2000
    target_text_min: int = 1500
    target_text_max: int = 2000
    target_other_min: int = 1000
    target_other_max: int = 3000

    repetition_k: int = 16

    code_short_line_threshold: int = 60
    code_punct_weight: float = 0.5
    code_short_line_weight: float = 0.5

    # Advanced dedup tuning (Simhash + MinHash); override scorer defaults when set.
    simhash_window: int = 128
    simhash_hamm_thresh: int | None = None
    enable_minhash: bool | None = None
    minhash_perms: int | None = None
    minhash_bands: int | None = None
    minhash_shingle_k: int | None = None
    minhash_jaccard_thresh: float | None = None


@dataclass(slots=True)
class QCConfig:
    """
    Configuration for quality scoring and gating.

    Set ``enabled=True`` and pick a ``mode`` from :class:`QCMode`:

    - ``INLINE``: score records during extraction and drop those failing thresholds.
    - ``ADVISORY``: score inline but never drop; adds QC metadata for review.
    - ``POST``: skip inline scoring; run QC by re-reading the JSONL output.
    - ``OFF``: disable QC entirely.

    Semantics:
    - enabled=False → QC is off regardless of mode.
    - enabled=True with mode in {"inline", "advisory"} → requires a scorer and QC extras; validated at config prep.
    - enabled=True with mode="post" → scorer optional; if QC extras are missing, QC is skipped with a warning.
    Concurrency:
    - ``parallel_post`` enables post-QC scoring via process_items_parallel.
    - ``post_*`` knobs override pipeline concurrency for post-QC; when None, they inherit from PipelineConfig.

    heuristics:
    Advanced tuning of QC heuristics (length bands, repetition window, code complexity weights). Most
    users can ignore this unless they need to align scoring with specialized corpora.
    """

    enabled: bool = False
    write_csv: bool = False
    csv_suffix: str = "_quality.csv"
    scorer: Optional[Any] = None  # optional extra
    fail_on_error: bool = False
    min_score: Optional[float] = 60.0
    drop_near_dups: bool = False
    mode: str = QCMode.INLINE
    parallel_post: bool = False
    post_executor_kind: Optional[str] = None
    post_max_workers: Optional[int] = None
    post_submit_window: Optional[int] = None
    heuristics: "QCHeuristics" = field(default_factory=QCHeuristics)

    def normalize_mode(self) -> str:
        normalized = QCMode.normalize(self.mode)
        self.mode = normalized
        return normalized

    def validate(self) -> None:
        mode = self.normalize_mode()
        if self.enabled and mode == QCMode.OFF:
            raise ValueError("QC enabled but mode is 'off'; disable qc.enabled or choose an active mode.")
        if self.enabled and mode in {QCMode.INLINE, QCMode.ADVISORY} and self.scorer is None:
            raise ValueError("Inline/advisory QC requires a scorer; set qc.scorer or install QC extras.")
        h = self.heuristics or QCHeuristics()
        if not isinstance(h, QCHeuristics):
            try:
                h = QCHeuristics(**dict(h))  # type: ignore[arg-type]
            except Exception as exc:
                raise TypeError(f"qc.heuristics must be QCHeuristics-compatible; got {type(self.heuristics).__name__}") from exc
        self.heuristics = h
        numeric_positive = [
            ("target_code_min", h.target_code_min),
            ("target_code_max", h.target_code_max),
            ("target_log_min", h.target_log_min),
            ("target_log_max", h.target_log_max),
            ("target_text_min", h.target_text_min),
            ("target_text_max", h.target_text_max),
            ("target_other_min", h.target_other_min),
            ("target_other_max", h.target_other_max),
            ("repetition_k", h.repetition_k),
            ("code_short_line_threshold", h.code_short_line_threshold),
        ]
        for name, value in numeric_positive:
            if value <= 0:
                raise ValueError(f"qc.heuristics.{name} must be positive; got {value}")
        weight_fields = [
            ("code_punct_weight", h.code_punct_weight),
            ("code_short_line_weight", h.code_short_line_weight),
        ]
        for name, value in weight_fields:
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"qc.heuristics.{name} must be between 0 and 1; got {value}")
        if h.simhash_window is not None and h.simhash_window <= 0:
            raise ValueError(f"qc.heuristics.simhash_window must be positive; got {h.simhash_window}")


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
        if self.sources.sources:
            raise ValueError("sources.sources must be empty in declarative specs; use sources.specs instead.")
        if self.sinks.sinks:
            raise ValueError("sinks.sinks must be empty in declarative specs; use sinks.specs instead.")

    def with_context(self, ctx: RepoContext) -> RepocapsuleConfig:
        return replace(self, sinks=replace(self.sinks, context=ctx))

    def prepared(self, *, mutate: bool = False) -> "RepocapsuleConfig":
        """
        Deprecated: use ``build_pipeline_plan`` instead. Returns the prepared config for compatibility.
        """
        import warnings
        from .builder import build_pipeline_plan

        warnings.warn(
            "RepocapsuleConfig.prepared is deprecated; use build_pipeline_plan instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        plan = build_pipeline_plan(self, mutate=mutate)
        return plan.config

    def validate(self) -> None:
        self._validate_paths()
        self.qc.validate()
        allowed = {"thread", "process", "auto"}
        kind = (self.pipeline.executor_kind or "auto").strip().lower()
        if kind not in allowed:
            raise ValueError(
                f"pipeline.executor_kind must be one of {sorted(allowed)}; got {self.pipeline.executor_kind!r}."
            )
        self.pipeline.executor_kind = kind

    def _validate_paths(self) -> None:
        meta = self.metadata
        p_jsonl = meta.primary_jsonl
        prompt = meta.prompt_path
        if not (p_jsonl and prompt):
            return
        try:
            if Path(p_jsonl).resolve() == Path(prompt).resolve():
                raise ValueError("primary_jsonl and prompt_path refer to the same file path.")
        except Exception:
            if p_jsonl == prompt:
                raise ValueError("primary_jsonl and prompt_path refer to the same file path.")

    # -------------------------
    # Serialization helpers
    # -------------------------
    def to_dict(self) -> Dict[str, Any]:
        """
        Return a JSON-serializable representation of this configuration.

        Suitable for embedding in RunSummaryMeta.config or persisting via to_json();
        skips non-serializable runtime objects like sources, sinks, HTTP clients, or scorer instances.
        """
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

    @classmethod
    def from_toml(cls: Type[T], path: Path | str) -> T:
        """
        Load a RepocapsuleConfig from a TOML file.

        The TOML layout mirrors the structure of this dataclass: top-level tables like [decode],
        [chunk], [pipeline], [sinks], [http], [qc], [logging], [metadata], and so on.
        """
        if tomllib is None:
            raise RuntimeError(
                "TOML support requires Python 3.11+ (tomllib) or installing the 'tomli' package."
            )
        path_obj = Path(path)
        raw = path_obj.read_bytes()
        data = tomllib.loads(raw.decode("utf-8"))

        if not isinstance(data, Mapping):
            raise TypeError(f"Top-level TOML document must be a mapping; got {type(data).__name__}.")

        return cls.from_dict(data)


def load_config_from_path(path: str | Path) -> RepocapsuleConfig:
    """
    Convenience loader that picks a parser based on file extension (.toml or .json).
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".toml":
        return RepocapsuleConfig.from_toml(p)
    if suffix == ".json":
        return RepocapsuleConfig.from_json(p)
    raise ValueError(f"Unsupported config extension {p.suffix!r}; expected .toml or .json.")


_SKIP_FIELDS: Dict[Type[Any], set[str]] = {
    SourceConfig: {"sources"},
    SinkConfig: {"sinks"},
    PipelineConfig: {"extractors", "bytes_handlers", "file_extractor"},
    HttpConfig: {"client"},
    QCConfig: {"scorer"},
}


def _dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    """Serialize dataclasses to JSON-friendly dicts, skipping None and non-serializable fields."""
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
    """Best-effort JSON-friendly coercion; drops values that cannot be serialized."""
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
    type_hints = get_type_hints(cls)

    kwargs: Dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        field_type = type_hints.get(f.name, f.type)
        kwargs[f.name] = _coerce_value(field_type, data[f.name])
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



__all__ = ["RepocapsuleConfig", "RunMetadata", "FileProcessingConfig"]
