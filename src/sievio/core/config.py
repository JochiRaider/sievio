# config.py
# SPDX-License-Identifier: MIT
"""Configuration models and helpers for Sievio runs.

This module defines declarative dataclasses for sources, sinks, quality
control, HTTP, logging, and other pipeline settings, along with helpers
for serializing and loading configurations from JSON and TOML.
"""
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
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Type, TypeVar, Union, get_args, get_origin, get_type_hints, List, Literal

from .chunk import ChunkPolicy
from .language_id import LanguageConfig
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
    """Supported quality-control modes.

    Modes:
    * ``OFF``: Disable QC entirely (no scoring, no annotations).
    * ``INLINE``: Score records during extraction and enforce gating
    (records may be dropped).
    * ``ADVISORY``: Score records inline but never drop them; annotations
    are for review only.
    * ``POST``: Run QC after the pipeline completes (no inline
    annotations or gating).
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


DEFAULT_QC_SCORER_ID = "jsonl_default"


# ---------------------------------------------------------------------------
# Source configs
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class LocalDirSourceConfig:
    """Configuration options for local directory sources.

    Attributes:
        include_exts (set[str] | None): File extensions to include.
            When set, only files with these lowercase suffixes are
            processed.
        exclude_exts (set[str] | None): File extensions to exclude.
        skip_hidden (bool): Whether to skip dotfiles and hidden paths.
        follow_symlinks (bool): Whether to traverse symbolic links.
        respect_gitignore (bool): Whether to honor .gitignore patterns.
        max_file_bytes (int | None): Hard cap on bytes read per file.
        read_prefix_bytes (int | None): If set, only read this many
            bytes from each file.
        read_prefix_for_large_files_only (bool): When True, apply
            ``read_prefix_bytes`` only to files larger than that limit.
    """    
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
    """Configuration for reading sources from GitHub archives.

    Attributes:
        per_file_cap (int | None): Maximum uncompressed bytes to extract
            per member file.
        max_total_uncompressed (int): Maximum total uncompressed bytes
            across all members.
        max_members (int): Maximum number of archive members to inspect.
        max_compression_ratio (float): Safety limit for compressed vs.
            uncompressed size.
        include_exts (set[str] | None): Optional whitelist of file
            extensions to include.
        exclude_exts (set[str] | None): Optional blacklist of file
            extensions to exclude.
    """    
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
    user_agent: str = "sievio/0.1 (+https://github.com/jochiraider/sievio)"
    client: Optional[SafeHttpClient] = None
    download_max_workers: int = 4  # 0 or negative → auto based on URL count
    download_submit_window: Optional[int] = None
    download_executor_kind: str = "thread"


@dataclass(slots=True)
class CsvSourceConfig:
    """Default settings for reading text from CSV files.

    Attributes:
        default_text_column (str): Column name used as the text field
            when none is specified.
        default_delimiter (str | None): Delimiter override; when None
            the CSV sniffer is used.
        encoding (str): Text encoding used to decode CSV bytes.
    """    
    default_text_column: str = "text"
    default_delimiter: Optional[str] = None
    encoding: str = "utf-8"


@dataclass(slots=True)
class SQLiteSourceConfig:
    """Settings for reading text from SQLite databases.

    Attributes:
        default_text_columns (tuple[str, ...]): Column names that are
            treated as text fields by default.
        batch_size (int): Number of rows to fetch per query batch.
        download_max_bytes (int | None): Optional limit on bytes
            downloaded for remote databases.
        retries (int): Number of times to retry failed downloads.
    """    
    default_text_columns: Tuple[str, ...] = ("text",)
    batch_size: int = 1000
    download_max_bytes: Optional[int] = None
    retries: int = 2


@dataclass(slots=True)
class SourceConfig:
    """Top-level configuration for all input sources.

    Attributes:
        specs (list[SourceSpec]): Declarative source specifications used
            by factories.
        sources (Sequence[Source]): Concrete source instances, expected
            to be empty on purely declarative configs.
        local (LocalDirSourceConfig): Defaults for local directory
            sources.
        github (GitHubSourceConfig): Defaults for GitHub sources.
        pdf (PdfSourceConfig): Defaults for web PDF sources.
        csv (CsvSourceConfig): Defaults for CSV sources.
        sqlite (SQLiteSourceConfig): Defaults for SQLite sources.
        defaults (dict[str, dict[str, Any]]): Per-kind default options
            keyed by SourceSpec.kind or plugin-defined ids.
    """    
    specs: List["SourceSpec"] = field(default_factory=list)
    sources: Sequence[Source] = field(default_factory=tuple)
    local: LocalDirSourceConfig = field(default_factory=LocalDirSourceConfig)
    github: GitHubSourceConfig = field(default_factory=GitHubSourceConfig)
    pdf: PdfSourceConfig = field(default_factory=PdfSourceConfig)
    csv: CsvSourceConfig = field(default_factory=CsvSourceConfig)
    sqlite: SQLiteSourceConfig = field(default_factory=SQLiteSourceConfig)
    # Generic per-kind defaults, keyed by SourceSpec.kind or a plugin-defined id.
    defaults: Dict[str, Dict[str, Any]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Decode / chunk / pipeline
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class DecodeConfig:
    """Text decoding and normalization options.

    Attributes:
        normalize (str | None): Unicode normalization form (e.g. "NFC")
            or None to disable normalization.
        strip_controls (bool): Whether to strip control characters from
            decoded text.
        fix_mojibake (bool): Whether to attempt simple mojibake fixes.
        max_bytes_per_file (int | None): Soft cap on bytes passed to the
            decoder per file.
    """
    normalize: Optional[str] = "NFC"
    strip_controls: bool = True
    fix_mojibake: bool = True
    max_bytes_per_file: Optional[int] = None  # Bytes passed to the decoder per file (soft cap).


@dataclass(slots=True)
class ChunkConfig:
    """Configuration for text chunking behavior.

    Attributes:
        policy (ChunkPolicy): Chunking policy instance used to split
            decoded text into records.
        tokenizer_name (str | None): Optional tokenizer identifier used
            for token-based policies.
        attach_language_metadata (bool): Whether detected language
            metadata is attached to records.
    """    
    policy: ChunkPolicy = field(default_factory=ChunkPolicy)
    tokenizer_name: Optional[str] = None
    attach_language_metadata: bool = True


@dataclass(slots=True)
class LanguageIDConfig:
    """Settings for human-language detection."""

    enabled: bool = True
    backend: str = "baseline"


@dataclass(slots=True)
class CodeLanguageConfig:
    """Settings for code-language detection and hints."""

    enabled: bool = True
    backend: str = "baseline"
    hints: LanguageConfig = field(default_factory=LanguageConfig)


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
    max_error_rate: Optional[float] = None


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
    """Settings used when generating prompt-oriented outputs.

    Attributes:
        heading_fmt (str): Format string for per-chunk headings.
        include_prompt_file (bool): Whether to include a combined prompt
            file alongside JSONL outputs.
    """
    heading_fmt: str = "### {path} [chunk {chunk}]"
    include_prompt_file: bool = True


@dataclass(slots=True)
class SinkConfig:
    """Top-level configuration for output sinks.

    Attributes:
        specs (list[SinkSpec]): Declarative sink specifications used by
            factories.
        sinks (Sequence[Sink]): Concrete sink instances, expected to be
            empty on purely declarative configs.
        context (RepoContext | None): Optional repository context passed
            to sinks.
        output_dir (Path): Base directory for sink outputs.
        primary_jsonl_name (str | None): Basename of the primary JSONL
            sink, if any.
        prompt (PromptConfig): Settings controlling prompt output.
        compress_jsonl (bool): Whether to gzip-compress JSONL outputs.
        jsonl_basename (str): Basename used for JSONL files.
        defaults (dict[str, dict[str, Any]]): Per-kind default options
            keyed by SinkSpec.kind.
    """    
    specs: List["SinkSpec"] = field(default_factory=list)
    sinks: Sequence[Sink] = field(default_factory=tuple)
    context: Optional[RepoContext] = None
    output_dir: Path = Path(".")
    primary_jsonl_name: Optional[str] = None
    prompt: PromptConfig = field(default_factory=PromptConfig)
    compress_jsonl: bool = False
    jsonl_basename: str = "data"
    # Generic per-kind defaults for sinks, keyed by SinkSpec.kind.
    defaults: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # Merge with spec.options via build_config_from_defaults_and_options.


@dataclass(slots=True)
class DatasetCardConfig:
    """Configuration for generating dataset cards.

    Attributes:
        enabled (bool): Whether to emit a dataset card.
        split_name (str): Default split name associated with the
            generated dataset.
        license (str | Sequence[str] | None): License identifier or
            identifiers for the dataset.
        task_categories (str | Sequence[str] | None): High-level task
            categories (e.g. "text-generation").
        task_ids (str | Sequence[str] | None): Fine-grained task ids.
        tags (str | Sequence[str] | None): Additional tags for the
            dataset card.
    """    
    enabled: bool = True
    split_name: str = "train"
    license: Optional[Union[str, Sequence[str]]] = None
    task_categories: Optional[Union[str, Sequence[str]]] = None
    task_ids: Optional[Union[str, Sequence[str]]] = None
    tags: Optional[Union[str, Sequence[str]]] = None


@dataclass(slots=True)
class SourceSpec:
    """Declarative source entry; factories map kind -> concrete sources."""

    kind: str
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SinkSpec:
    """Declarative sink entry; factories map kind -> concrete sinks.

    Known kinds:
    - "default_jsonl_prompt": options jsonl_path, prompt_path
    - "parquet_dataset": options path (required), text_field/meta_field, partition_by,
      row_group_size, compression, overwrite
    """

    kind: str
    options: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# HTTP / PDF / Web configs
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class HttpConfig:
    """HTTP client settings used by higher-level helpers and factories.

    - ``client`` can hold a pre-built SafeHttpClient instance. When set,
    it will be reused by ``build_client()`` and passed to factories.
    - ``as_global`` controls whether the builder installs this client as
    the module-wide default via ``set_global_http_client``. For
    CLI-style one-shot runs, leaving this True is convenient. For
    tests or long-lived processes, prefer ``as_global=False`` and
    pass the client explicitly.
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
    """Tunable thresholds and weights used by quality scoring.

    Defaults match existing hard-coded behavior and can be overridden
    per-run via ``QCConfig.scorer_options["heuristics"]``.
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

    def validate(self) -> None:
        """Validate heuristic thresholds and weights."""
        numeric_positive = [
            ("target_code_min", self.target_code_min),
            ("target_code_max", self.target_code_max),
            ("target_log_min", self.target_log_min),
            ("target_log_max", self.target_log_max),
            ("target_text_min", self.target_text_min),
            ("target_text_max", self.target_text_max),
            ("target_other_min", self.target_other_min),
            ("target_other_max", self.target_other_max),
            ("repetition_k", self.repetition_k),
            ("code_short_line_threshold", self.code_short_line_threshold),
            ("simhash_window", self.simhash_window),
        ]
        for name, value in numeric_positive:
            if value is not None and value <= 0:
                raise ValueError(f"QCHeuristics.{name} must be positive; got {value!r}.")

        weight_fields = [
            ("code_punct_weight", self.code_punct_weight),
            ("code_short_line_weight", self.code_short_line_weight),
        ]
        for name, value in weight_fields:
            if value is not None and not (0.0 <= value <= 1.0):
                raise ValueError(f"QCHeuristics.{name} must be between 0 and 1; got {value!r}.")


@dataclass(slots=True)
class SafetyConfig:
    """
    Safety/PII filtering configuration layered alongside QC screening.

    Safety uses QCMode-style strings ("inline", "advisory", "post", "off")
    but its mode is independent of QCConfig.mode. QC controls quality gating;
    SafetyConfig.mode together with annotate_only control safety gating.
    """

    enabled: bool = False
    # Reuse QCMode strings for now: "inline", "advisory", "post", "off"
    mode: str = QCMode.INLINE
    scorer: Any | None = None
    scorer_id: str | None = None
    scorer_options: dict[str, Any] = field(default_factory=dict)

    # High-level toggles/thresholds (config only; scorer decides details)
    toxicity_threshold: float | None = None
    allowed_licenses: list[str] | None = None
    annotate_only: bool = False  # if True, never drop on safety
    fail_on_error: bool = False
    write_csv: bool = False
    csv_suffix: str = "_safety.csv"
    write_signals_sidecar: bool = False
    signals_suffix: str | None = None
    signals_format: Literal["csv", "parquet"] = "csv"
    parallel_post: bool = False
    post_executor_kind: str | None = None
    post_max_workers: int | None = None
    post_submit_window: int | None = None

    def normalize_mode(self) -> str:
        mode = QCMode.normalize(self.mode)
        self.mode = mode
        return mode

    def validate(self, qc_cfg: "QCConfig") -> None:
        """Validate interplay with QC settings.

        SafetyConfig.mode is independent of QCConfig.mode. This method does not
        enforce that they match.
        """

        mode = self.normalize_mode()
        if self.scorer_options is None or not isinstance(self.scorer_options, dict):
            raise TypeError("qc.safety.scorer_options must be a mapping (use {} for defaults).")
        if self.enabled and mode == QCMode.OFF:
            raise ValueError(
                "qc.safety.enabled=True but qc.safety.mode='off'; disable safety or choose 'inline'/'advisory'."
            )
        if self.signals_format not in {"csv", "parquet"}:
            raise ValueError("qc.safety.signals_format must be 'csv' or 'parquet'.")
        # Safety scorer options stay as mappings for now; when a typed default is
        # added, normalize it via build_config_from_defaults_and_options just like QC heuristics.


@dataclass(slots=True)
class QCConfig:
    """Configuration for quality scoring and gating.

    Set ``enabled=True`` and pick a ``mode`` from :class:`QCMode`:

    * ``INLINE``: Score records during extraction and drop those failing
    thresholds.
    * ``ADVISORY``: Score inline but never drop; adds QC metadata for
    review.
    * ``POST``: Do not enforce QC gates inline; QC thresholds are
    enforced only in a post-QC pass. QC may still evaluate inline when
    required (for example, to supply features to safety scoring).
    * ``OFF``: Disable QC entirely.

    Semantics:
    * ``enabled=False`` → QC is off regardless of mode.
    * ``enabled=True`` with mode in {"inline", "advisory"} → requires
    either a resolved scorer (``qc.scorer``) or a scorer id
    (``qc.scorer_id``) at config time; the builder resolves
    ``scorer_id`` later and fails if QC extras are missing.
    * ``enabled=True`` with mode="post" → QC gating happens only in the
    post-hoc pass; QC may still compute inline when needed. A scorer is
    optional; if QC extras are missing, QC is skipped with a warning.

    Concurrency:
    * ``parallel_post`` enables post-QC scoring via
    ``process_items_parallel``.
    * ``post_*`` knobs override pipeline concurrency for post-QC; when
    None, they inherit from :class:`PipelineConfig`.
    """
    enabled: bool = False
    write_csv: bool = False
    csv_suffix: str = "_quality.csv"
    write_signals_sidecar: bool = False
    signals_suffix: str | None = None
    signals_format: Literal["csv", "parquet"] = "csv"
    scorer: Optional[Any] = None  # optional extra
    scorer_id: Optional[str] = None  # None → registry default (first registered scorer)
    scorer_options: Dict[str, Any] = field(default_factory=dict)
    heuristics: Optional[QCHeuristics] = None  # Convenience mirror of scorer_options["heuristics"] when normalized.
    fail_on_error: bool = False
    min_score: Optional[float] = 60.0
    drop_near_dups: bool = False
    exact_dedup: bool = True
    mode: str = QCMode.INLINE
    parallel_post: bool = False
    post_executor_kind: Optional[str] = None
    post_max_workers: Optional[int] = None
    post_submit_window: Optional[int] = None
    safety: SafetyConfig = field(default_factory=SafetyConfig)

    def normalize_mode(self) -> str:
        """Normalize the configured QC mode and update the instance.

        Returns:
            str: Normalized QC mode string.
        """    
        normalized = QCMode.normalize(self.mode)
        self.mode = normalized
        return normalized

    def validate(self) -> None:
        """Validate the QC configuration and raise on inconsistencies.

        This checks the interaction between ``enabled``, ``mode``,
        scorer fields, heuristic options, and numeric thresholds, raising
        ValueError or TypeError when the configuration is invalid.
        """    
        mode = self.normalize_mode()
        if self.enabled and mode == QCMode.OFF:
            raise ValueError("QC enabled but mode is 'off'; disable qc.enabled or choose an active mode.")
        needs_scorer = self.enabled and mode in {QCMode.INLINE, QCMode.ADVISORY}
        has_resolved_scorer = self.scorer is not None
        has_planned_scorer = bool(self.scorer_id)

        if needs_scorer and not (has_resolved_scorer or has_planned_scorer):
            raise ValueError("Inline/advisory QC requires a scorer; set qc.scorer_id or install QC extras.")
        if self.scorer_options is None or not isinstance(self.scorer_options, dict):
            raise TypeError("qc.scorer_options must be a mapping (use {} for defaults).")
        self.heuristics = None
        if "exact_dedup" not in self.scorer_options:
            self.scorer_options["exact_dedup"] = bool(self.exact_dedup)
        if self.signals_format not in {"csv", "parquet"}:
            raise ValueError("qc.signals_format must be 'csv' or 'parquet'.")
        if hasattr(self, "safety") and isinstance(self.safety, SafetyConfig):
            self.safety.validate(self)


@dataclass(slots=True)
class LoggingConfig:
    """Controls the package logger; set propagate=True/logger_name to
    integrate with host apps.
    """
    level: int | str = "INFO"
    propagate: bool = False
    fmt: Optional[str] = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logger_name: str = PACKAGE_LOGGER_NAME

    def apply(self) -> None:
        """Apply this logging configuration to the package logger."""    
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
    """Metadata describing a single Sievio run.

    This captures paths to primary outputs plus arbitrary user-defined
    key/value pairs under ``extra``.
    """    
    primary_jsonl: Optional[str] = None
    prompt_path: Optional[str] = None
    repo_url: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the metadata to a JSON-serializable dictionary.

        Only non-None core fields are included; ``extra`` is
        shallow-copied.
        """        
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
        """Create a RunMetadata instance from a mapping.

        Args:
            data (Mapping[str, Any] | None): Optional mapping previously
                produced by :meth:`to_dict`.

        Returns:
            RunMetadata: Parsed metadata object.
        """        
        if not data:
            return cls()
        extra = dict(data.get("extra") or {})
        known = {k: data.get(k) for k in ("primary_jsonl", "prompt_path", "repo_url")}
        return cls(extra=extra, **{k: v for k, v in known.items() if v is not None})

    def merged(self, mapping: Mapping[str, Any]) -> "RunMetadata":
        """Return a new metadata object merged with values from mapping.

        Core attributes on this instance are only filled in when
        missing; keys in ``mapping`` that are not core fields are added
        to the ``extra`` dictionary if they do not already exist.

        Args:
            mapping (Mapping[str, Any]): Additional metadata values.

        Returns:
            RunMetadata: New merged metadata instance.
        """        
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
        """Return a metadata value by key with a default.

        This mirrors ``dict.get`` but understands the three core field
        names ``"primary_jsonl"``, ``"prompt_path"``, and
        ``"repo_url"``, falling back to the ``extra`` mapping for other
        keys.

        Args:
            key (str): Metadata key to retrieve.
            default (Any, optional): Default value if the key is not
                present.

        Returns:
            Any: Stored value or ``default`` if the key is missing.
        """        
        if key == "primary_jsonl":
            return self.primary_jsonl if self.primary_jsonl is not None else default
        if key == "prompt_path":
            return self.prompt_path if self.prompt_path is not None else default
        if key == "repo_url":
            return self.repo_url if self.repo_url is not None else default
        return self.extra.get(key, default)


@dataclass(slots=True)
class SievioConfig:
    """Declarative spec for a Sievio run.

    This object must remain purely declarative and serializable: it can
    hold only configuration knobs and derived scalar/string/path values.
    Anything that is callable, holds open resources, or maintains
    runtime state (HTTP clients, Source/Sink instances, file
    extractors, bytes handlers, QC scorers, plugins, executors, etc.)
    must live in PipelineRuntime or other ephemeral wiring, never on
    this spec. Derived declarative fields such as normalized modes,
    resolved primary JSONL names, or output directories are allowed.

    Safety and PII filtering is configured under ``qc.safety`` and
    follows the same execution modes as general QC.
    """
    sources: SourceConfig = field(default_factory=SourceConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    language: LanguageIDConfig = field(default_factory=LanguageIDConfig)
    code_lang: CodeLanguageConfig = field(default_factory=CodeLanguageConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    sinks: SinkConfig = field(default_factory=SinkConfig)
    http: HttpConfig = field(default_factory=HttpConfig)
    qc: QCConfig = field(default_factory=QCConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    metadata: RunMetadata = field(default_factory=RunMetadata)
    dataset_card: DatasetCardConfig = field(default_factory=DatasetCardConfig)

    def __post_init__(self) -> None:
        """Normalize nested config objects after initialization.

        This ensures metadata and dataset_card are concrete config
        instances and enforces declarative-only constraints on sources
        and sinks.
        """
        if not isinstance(self.metadata, RunMetadata):
            self.metadata = RunMetadata.from_dict(dict(self.metadata or {}))
        if not isinstance(self.dataset_card, DatasetCardConfig):
            self.dataset_card = DatasetCardConfig(**dict(self.dataset_card or {}))
        if not isinstance(self.language, LanguageIDConfig):
            self.language = LanguageIDConfig(**dict(self.language or {}))
        if not isinstance(self.code_lang, CodeLanguageConfig):
            self.code_lang = CodeLanguageConfig(**dict(self.code_lang or {}))
        if self.sources.sources:
            raise ValueError("sources.sources must be empty in declarative specs; use sources.specs instead.")
        if self.sinks.sinks:
            raise ValueError("sinks.sinks must be empty in declarative specs; use sinks.specs instead.")

    def with_context(self, ctx: RepoContext) -> SievioConfig:
        """Return a shallow copy with the given RepoContext attached.

        Args:
            ctx (RepoContext): Repository context to propagate to sinks.

        Returns:
            SievioConfig: New config sharing all other fields.
        """        
        return replace(self, sinks=replace(self.sinks, context=ctx))

    def validate(self) -> None:
        """Validate the configuration for internal consistency.

        This validates paths, QC settings, and executor kind, normalizing
        ``pipeline.executor_kind`` to one of ``{"thread", "process",
        "auto"}`` and raising ValueError when an invalid value is used.
        """        
        self._validate_paths()
        # QC validation is structural here (scorer object or scorer_id present); builder still resolves scorer_id.
        self.qc.validate()
        allowed = {"thread", "process", "auto"}
        kind = (self.pipeline.executor_kind or "auto").strip().lower()
        if kind not in allowed:
            raise ValueError(
                f"pipeline.executor_kind must be one of {sorted(allowed)}; got {self.pipeline.executor_kind!r}."
            )
        self.pipeline.executor_kind = kind
        max_error_rate = self.pipeline.max_error_rate
        if max_error_rate is not None:
            try:
                rate_val = float(max_error_rate)
            except (TypeError, ValueError):
                raise ValueError("pipeline.max_error_rate must be a float between 0.0 and 1.0 when set.")
            if rate_val < 0.0 or rate_val > 1.0:
                raise ValueError("pipeline.max_error_rate must be between 0.0 and 1.0 when set.")
            self.pipeline.max_error_rate = rate_val

    def _validate_paths(self) -> None:
        """Ensure that primary_jsonl and prompt_path, if set, differ.

        Raises:
            ValueError: If both paths resolve to the same file.
        """        
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
        """Serialize the configuration to JSON and write it to disk.

        Args:
            path (Path | str): Target file path.
            indent (int): Indentation level passed to ``json.dumps``.

        Returns:
            str: String path to the written file.
        """        
        data = self.to_dict()
        target = Path(path)
        target.write_text(json.dumps(data, indent=indent, sort_keys=True), encoding="utf-8")
        return str(target)

    @classmethod
    def from_dict(cls: Type[T], data: Mapping[str, Any]) -> T:
        """Instantiate a SievioConfig from a mapping.

        Args:
            data (Mapping[str, Any]): Mapping produced by
                :meth:`to_dict` or loaded from JSON/TOML.

        Returns:
            SievioConfig: Parsed configuration instance.
        """        
        return _dataclass_from_dict(cls, data)

    @classmethod
    def from_json(cls: Type[T], path: Path | str) -> T:
        """Load a configuration from a JSON file.

        Args:
            path (Path | str): Path to the JSON document.

        Returns:
            SievioConfig: Parsed configuration instance.
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    @classmethod
    def from_toml(cls: Type[T], path: Path | str) -> T:
        """
        Load a SievioConfig from a TOML file.

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


def load_config_from_path(path: str | Path) -> SievioConfig:
    """Load a SievioConfig from a JSON or TOML file.
    Args:
        path (Path | str): Path to a ``.toml`` or ``.json`` config
            file.
    Returns:
        SievioConfig: Parsed configuration instance.
    Raises:
        ValueError: If the file extension is not ``.toml`` or ``.json``.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".toml":
        return SievioConfig.from_toml(p)
    if suffix == ".json":
        return SievioConfig.from_json(p)
    raise ValueError(f"Unsupported config extension {p.suffix!r}; expected .toml or .json.")


_SKIP_FIELDS: Dict[Type[Any], set[str]] = {
    SourceConfig: {"sources"},
    SinkConfig: {"sinks"},
    PipelineConfig: {"extractors", "bytes_handlers", "file_extractor"},
    HttpConfig: {"client"},
    QCConfig: {"scorer", "heuristics"},
}

C = TypeVar("C")


def validate_options_for_dataclass(
    cfg_type: Type[C],
    *,
    options: Mapping[str, Any] | None,
    ignore_keys: Iterable[str] = (),
    context: str | None = None,
) -> None:
    """
    Validate that options only contain known dataclass fields or ignore_keys.

    Raises:
        ValueError: If unknown option keys are present.
    """
    if not options:
        return

    field_names = {f.name for f in fields(cfg_type)}
    allowed = field_names | set(ignore_keys)
    unknown = sorted(k for k in options.keys() if k not in allowed)

    if unknown:
        label = context or cfg_type.__name__
        allowed_list = ", ".join(sorted(allowed))
        unknown_list = ", ".join(unknown)
        raise ValueError(
            f"Unsupported options for {label}: {unknown_list}. "
            f"Allowed keys: {allowed_list}"
        )


def build_config_from_defaults_and_options(
    cfg_type: Type[C],
    *,
    defaults: Mapping[str, Any] | None,
    options: Mapping[str, Any] | None,
    ignore_keys: Iterable[str] = (),
) -> C:
    """
    Construct a config dataclass from per-kind defaults + per-spec options.

    This helper is the canonical way to layer `defaults` + `options` into typed
    config objects for sources, sinks, QC, and scorers. Keep factories lean: let
    this function perform the merge and filtering, only pulling constructor-only
    keys (paths, URLs, identifiers) directly from `spec.options`.

    - ``defaults`` typically comes from e.g. SievioConfig.sources.defaults[kind]
      or SievioConfig.sinks.defaults[kind].
    - ``options`` comes from a specific SourceSpec/SinkSpec/etc.
    - ``ignore_keys`` is for keys that belong to the factory constructor
      but not the dataclass itself (e.g., 'root_dir', 'url').

    Only keys matching dataclass field names are applied; unknown keys
    are silently ignored here (factory may log them separately).
    """
    merged: dict[str, Any] = dict(defaults or {})

    field_names = {f.name for f in fields(cfg_type)}

    if options:
        ignore = set(ignore_keys)
        for key, value in options.items():
            if key in ignore:
                continue
            if key in field_names:
                merged[key] = value

    return cfg_type(**merged)  # type: ignore[arg-type]


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
    """Instantiate a dataclass of type `cls` from a mapping.

    Args:
        cls (type[T]): Dataclass type to construct.
        data (Mapping[str, Any] | None): Source mapping, or None to use
            the type's default constructor.

    Returns:
        T: New dataclass instance.
    """    
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
    """Coerce `value` into the shape implied by `expected_type`.


    This handles nested dataclasses, container types, unions, and Paths,
    recursing into sequences and mappings when necessary.
    """    
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
    """Strip Optional from a type annotation.


    Args:
        typ (Any): Type annotation that may be a Union including ``None``.

    Returns:
        tuple[Any, bool]: A pair ``(base_type, is_optional)`` where
        ``is_optional`` is True if ``None`` was present in the union.
    """    
    origin = get_origin(typ)
    if origin is Union:
        args = [arg for arg in get_args(typ) if arg is not type(None)]
        if len(args) == 1:
            inner = args[0]
            base, _ = _strip_optional(inner)
            return base, True
    return typ, False


def is_dataclass_type(typ: Any) -> bool:
    """Return True if `typ` is a dataclass type (not an instance).


    This helper is tolerant of non-type inputs and wraps
    :func:`dataclasses.is_dataclass` in a try/except to avoid
    propagating unexpected errors.
    """    
    try:
        return isinstance(typ, type) and is_dataclass(typ)
    except Exception:
        return False



__all__ = ["SievioConfig", "RunMetadata", "FileProcessingConfig", "DEFAULT_QC_SCORER_ID"]
