# records.py
# SPDX-License-Identifier: MIT
"""Helpers for record construction, metadata normalization, and run headers."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field, fields
from typing import (
    TYPE_CHECKING,
    Any,
    NotRequired,
    TypedDict,
    cast,
)
from urllib.parse import urlparse

from .chunk import count_tokens
from .language_id import (
    DEFAULT_LANGCFG,
    LanguageConfig,
    guess_lang_from_path,
)
from .log import get_logger

__all__ = [
    "sha256_text",
    "build_record",
    "RecordMeta",
    "RunSummaryMeta",
    "RunHeaderMeta",
    "QCSummaryMeta",
    "build_run_header_record",
    "ensure_meta_dict",
    "merge_meta_defaults",
    "is_summary_record",
    "filter_qc_meta",
    "filter_safety_meta",
    "check_record_schema",
]

log = get_logger(__name__)

# -----------------------
# Hashing utilities
# -----------------------

def sha256_text(text: str) -> str:
    """Return hex sha256 of UTF-8 encoded text (no BOM)."""
    h = hashlib.sha256()
    h.update(text.encode("utf-8", "strict"))
    return h.hexdigest()


# -----------------------
# Metadata helpers
# -----------------------

# Bump when core schema fields change so downstream consumers can detect mixes.
RECORD_META_SCHEMA_VERSION = "2"
SUMMARY_META_SCHEMA_VERSION = "1"

# Canonical record meta fields used across the pipeline. Additional analyzer-specific
# metadata should be stored under ``meta["extra"]`` so downstream consumers can
# distinguish between core attributes and optional enrichments.
STANDARD_META_FIELDS: set[str] = {
    "kind",
    "source",
    "repo",
    "path",
    "url",
    "source_domain",
    "license",
    "lang",
    "lang_score",
    "perplexity",
    "ppl_bucket",
    "chunk_id",
    "n_chunks",
    "encoding",
    "had_replacement",
    "sha256",
    "approx_tokens",
    "tokens",
    "bytes",
    "file_bytes",
    "truncated_bytes",
    "nlines",
    "file_nlines",
    "schema_version",
}
# Field overview:
#   - Identity / provenance: ``source``, ``repo``, ``path``, ``url``, ``source_domain``.
#   - Content structure: ``bytes``, ``file_bytes``, ``truncated_bytes``,
#     ``nlines``, ``file_nlines``.
#   - Language & sampling: ``lang``, ``lang_score``, ``perplexity``,
#     ``ppl_bucket``.
#   - Chunking / processing: ``kind``, ``chunk_id``, ``n_chunks``,
#     ``encoding``, ``had_replacement``.
#   - Integrity & versioning: ``sha256``, ``schema_version``.
#   - Token counts: ``tokens``, ``approx_tokens``.
# Any other analyzer metadata should flow through ``RecordMeta.extra``.
# QC scorers populate a few additional meta keys; documenting them here keeps the
# schema discoverable for downstream tooling.
QC_META_FIELDS: set[str] = {
    "score",
    "qc_score",
    "qc_decision",
    "qc_drop_reason",
    "qc_reason",
    "near_dup",
    "dup_family_id",
    "dup_family_size",
    "qc_version",
}
# QC scorers may add additional metadata, but these are the only QC keys that should
# appear at the top level of ``record["meta"]``; all other QC signals belong in
# ``meta["extra"]`` (see ``filter_qc_meta``).
QC_SIGNAL_EXCLUDE_FIELDS: set[str] = {
    "path",
    "repo",
    "url",
    "doc_id",
    "chunk_id",
    "n_chunks",
    "source",
}

SAFETY_META_FIELDS: set[str] = {
    "safety_decision",
    "safety_drop_reason",
    "safety_reason",
    "pii_detected",
    "toxicity",
}


class QualitySignals(TypedDict, total=False):
    # Length/size
    len: int
    tokens: int
    len_char: int              # aliases qc_result["len"]
    len_tok: int               # aliases qc_result["tokens"]
    nlines: int
    file_nlines: NotRequired[int]

    # Formatting / noise heuristics
    ascii_ratio: float
    repetition: float
    code_complexity: float
    parse_ok: bool

    # LM and corpus-based quality
    perplexity: NotRequired[float]
    gopher_quality: NotRequired[float]
    ppl_bucket: NotRequired[str]

    # Language & domain
    lang_id: NotRequired[str]      # from meta["lang"] or language detector
    lang_score: NotRequired[float]
    source_domain: NotRequired[str]

    # Duplication signals
    near_dup: NotRequired[bool]
    near_dup_simhash: NotRequired[bool]
    near_dup_minhash: NotRequired[bool]
    minhash_jaccard: NotRequired[float]
    hamdist: NotRequired[int]
    dup_family_id: NotRequired[str]


def filter_qc_meta(qc_result: Mapping[str, Any]) -> tuple[dict[str, Any], QualitySignals]:
    """Partition a QC scorer result into canonical and extra signals.

    Canonical QC fields are allowed at the top level of ``record["meta"]``
    and include items in ``QC_META_FIELDS`` plus ``qc_score``. Everything
    else is treated as QC signals intended for
    ``meta["extra"]["qc_signals"]``.

    Args:
        qc_result (Mapping[str, Any]): Raw QC scorer output.

    Returns:
        Tuple[Dict[str, Any], QualitySignals]: Canonical QC fields and the
        remaining QC signals.
    """
    canonical: dict[str, Any] = {}
    qc_signals: dict[str, Any] = {}

    for key, value in qc_result.items():
        if key == "score":
            if value is not None:
                canonical["qc_score"] = value
            continue
        if key in QC_META_FIELDS:
            canonical[key] = value
            continue
        if key in QC_SIGNAL_EXCLUDE_FIELDS:
            continue
        qc_signals[key] = value

    # Add schema-aligned aliases
    if "len" in qc_result:
        qc_signals.setdefault("len_char", qc_result["len"])
        if "len" not in qc_signals:
            qc_signals["len"] = qc_result["len"]
    if "tokens" in qc_result:
        qc_signals.setdefault("len_tok", qc_result["tokens"])
        if "tokens" not in qc_signals:
            qc_signals["tokens"] = qc_result["tokens"]
    if "lang" in qc_result:
        qc_signals.setdefault("lang_id", qc_result["lang"])
    if "near_dup" in qc_result and "near_dup" not in qc_signals:
        qc_signals["near_dup"] = qc_result["near_dup"]

    return canonical, cast(QualitySignals, qc_signals)


def filter_safety_meta(safety_result: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Partition a safety scorer result into canonical meta fields and signals."""

    canonical: dict[str, Any] = {}
    safety_signals: dict[str, Any] = {}

    for key, value in safety_result.items():
        if key in SAFETY_META_FIELDS:
            canonical[key] = value
            continue
        safety_signals[key] = value
    return canonical, safety_signals


def check_record_schema(record: Mapping[str, Any], logger: Any | None = None) -> None:
    """Emit diagnostics when a record's schema_version differs from this library."""

    meta = record.get("meta")
    if not isinstance(meta, Mapping):
        return

    version = meta.get("schema_version")
    logger_obj = logger or log

    def _propagates_to_root(lgr: logging.Logger) -> bool:
        current = lgr
        while current:
            if current.propagate is False and current.parent is not None:
                return False
            current = current.parent
        return True

    def _emit(level: int, msg: str, *args: Any) -> None:
        logger_obj.log(level, msg, *args)
        if logger is None and not _propagates_to_root(logger_obj):
            logging.getLogger().log(level, msg, *args)

    if version is None:
        _emit(logging.DEBUG, "Record missing schema_version; treating as legacy schema.")
        return

    version_str = str(version)
    if version_str > RECORD_META_SCHEMA_VERSION:
        _emit(
            logging.WARNING,
            "Record schema_version %s is newer than library schema_version %s; "
            "ensure compatibility before scoring.",
            version_str,
            RECORD_META_SCHEMA_VERSION,
        )
    elif version_str < RECORD_META_SCHEMA_VERSION:
        _emit(
            logging.DEBUG,
            "Record schema_version %s is older than library schema_version %s; "
            "proceeding with backward compatibility.",
            version_str,
            RECORD_META_SCHEMA_VERSION,
        )


def _meta_to_dict(obj: Any) -> dict[str, Any]:
    """Flatten dataclass fields and extras into a dictionary.

    Skips ``None`` values and prefers core fields over entries duplicated in
    ``extra``.

    Args:
        obj (Any): Dataclass instance carrying metadata.

    Returns:
        Dict[str, Any]: Dictionary representation without ``None`` values.
    """
    out: dict[str, Any] = {}
    if obj is None:
        return out
    for f in fields(obj):
        name = f.name
        if name == "extra":
            continue
        value = getattr(obj, name)
        if value is not None:
            out[name] = value
    extra = getattr(obj, "extra", None)
    if isinstance(extra, dict):
        for key, value in extra.items():
            if value is None or key in out:
                continue
            out[key] = value
    return out


@dataclass(slots=True)
class RecordMeta:
    """Metadata for content records (code/docs/logs/etc).

    Attributes:
        kind (str): Coarse file type, typically "code" or "doc".
        source (Optional[str]): Source repository or dataset identifier.
        repo (Optional[str]): Repository name in owner/name form.
        path (Optional[str]): Relative path to the file or chunk.
        url (Optional[str]): Canonical URL for the source file.
        source_domain (Optional[str]): Hostname derived from the URL.
        license (Optional[str]): SPDX license identifier.
        lang (Optional[str]): Language tag for the content.
        lang_score (Optional[float]): Confidence score for language detection.
        perplexity (Optional[float]): Language model perplexity.
        ppl_bucket (Optional[str]): Qualitative perplexity bucket.
        chunk_id (int): Chunk index within the file.
        n_chunks (int): Total number of chunks for the file.
        encoding (str): Text encoding used when reading the file.
        had_replacement (bool): Whether decoding required replacement chars.
        sha256 (Optional[str]): SHA-256 hex digest of the chunk text.
        approx_tokens (Optional[int]): Estimated token count.
        tokens (Optional[int]): Exact token count when available.
        bytes (Optional[int]): Byte length of the chunk text (UTF-8).
        file_bytes (Optional[int]): Original source byte length.
        truncated_bytes (Optional[int]): Bytes omitted due to truncation limits.
        nlines (Optional[int]): Line count of the chunk text.
        file_nlines (Optional[int]): Line count of the original file.
        schema_version (str): Schema version for record metadata.
        extra (Dict[str, Any]): Arbitrary additional metadata.
    """

    kind: str = "code"
    source: str | None = None
    repo: str | None = None
    path: str | None = None
    url: str | None = None
    source_domain: str | None = None
    license: str | None = None
    lang: str | None = None
    lang_score: float | None = None
    perplexity: float | None = None
    ppl_bucket: str | None = None
    chunk_id: int = 1
    n_chunks: int = 1
    encoding: str = "utf-8"
    had_replacement: bool = False
    sha256: str | None = None
    approx_tokens: int | None = None
    tokens: int | None = None
    bytes: int | None = None
    file_bytes: int | None = None
    truncated_bytes: int | None = None
    nlines: int | None = None
    file_nlines: int | None = None
    schema_version: str = RECORD_META_SCHEMA_VERSION
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata to a dictionary, omitting ``None`` values."""
        return _meta_to_dict(self)

    @classmethod
    def from_seed(
        cls,
        seed: Mapping[str, Any] | None = None,
        *,
        kind: str | None = None,
        **overrides: Any,
    ) -> RecordMeta:
        """Construct a RecordMeta from a seed mapping and optional overrides.

        Args:
            seed (Optional[Mapping[str, Any]]): Baseline metadata values.
            kind (Optional[str]): Override for the ``kind`` field.
            **overrides: Additional fields to apply after the seed.

        Returns:
            RecordMeta: Populated metadata instance with ``extra`` merged.
        """
        seed = seed or {}
        data = dict(seed)
        if kind is not None:
            data["kind"] = kind
        data.update(overrides)
        field_names = {f.name for f in fields(cls)}
        extra: dict[str, Any] = {}
        for key in list(data.keys()):
            if key not in field_names:
                value = data.pop(key)
                if value is not None:
                    extra.setdefault(key, value)
        base_extra = data.get("extra")
        if isinstance(base_extra, dict):
            merged_extra = dict(base_extra)
            for key, value in extra.items():
                merged_extra.setdefault(key, value)
        else:
            merged_extra = extra
        data["extra"] = merged_extra
        data.setdefault("schema_version", RECORD_META_SCHEMA_VERSION)
        return cls(**data)


@dataclass(slots=True)
class RunSummaryMeta:
    """Metadata footer for runs."""

    kind: str = "run_summary"
    schema_version: str = SUMMARY_META_SCHEMA_VERSION
    config: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    qc_summary: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize run summary metadata to a dictionary."""
        return _meta_to_dict(self)


@dataclass(slots=True)
class QCSummaryMeta:
    """Metadata wrapper for QC summary entries."""

    kind: str = "qc_summary"
    schema_version: str = SUMMARY_META_SCHEMA_VERSION
    summary: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize QC summary metadata to a dictionary."""
        return _meta_to_dict(self)


@dataclass(slots=True)
class RunHeaderMeta:
    """Metadata header describing configuration at run start."""

    kind: str = "run_header"
    schema_version: str = SUMMARY_META_SCHEMA_VERSION
    config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize run header metadata to a dictionary."""
        return _meta_to_dict(self)


def ensure_meta_dict(record: MutableMapping[str, Any]) -> dict[str, Any]:
    """Return record['meta'] as a dict, creating an empty one if needed."""
    meta = record.get("meta")
    if not isinstance(meta, dict):
        meta = {}
        record["meta"] = meta
    return meta


def merge_meta_defaults(
    record: MutableMapping[str, Any],
    defaults: Mapping[str, Any],
) -> dict[str, Any]:
    """Fill in default meta values without overriding existing entries."""
    meta = ensure_meta_dict(record)
    for key, value in defaults.items():
        if key in meta or value is None:
            continue
        meta[key] = value
    return meta


def is_summary_record(record: Mapping[str, Any]) -> bool:
    """Return True if the record appears to be a summary entry."""
    meta = record.get("meta") if isinstance(record, Mapping) else None
    if not isinstance(meta, dict):
        return False
    kind = meta.get("kind")
    return kind in {"run_header","run_summary", "qc_summary"}


# -----------------------
# Record assembly
# -----------------------

def build_record(
    *,
    text: str,
    rel_path: str,
    repo_full_name: str | None = None,  # e.g., "owner/repo"
    repo_url: str | None = None,        # e.g., "https://github.com/owner/repo"
    license_id: str | None = None,      # SPDX id like 'Apache-2.0'
    url: str | None = None,              # canonical file URL when available
    source_domain: str | None = None,    # hostname derived from URL/source
    lang: str | None = None,            # language label (Title Case preferred)
    encoding: str = "utf-8",
    had_replacement: bool = False,
    chunk_id: int | None = None,
    n_chunks: int | None = None,
    extra_meta: dict[str, object] | None = None,
    langcfg: LanguageConfig | None = None,
    tokens: int | None = None,
    meta: RecordMeta | Mapping[str, Any] | None = None,
    file_bytes: int | None = None,
    truncated_bytes: int | None = None,
    file_nlines: int | None = None,
) -> dict[str, object]:
    """Create a canonical JSONL record matching the requested schema.

    Args:
        text (str): Chunk content to store.
        rel_path (str): Path relative to the repository root.
        repo_full_name (Optional[str]): Repository in ``owner/name`` form.
        repo_url (Optional[str]): Canonical repository URL.
        license_id (Optional[str]): SPDX license identifier.
        url (Optional[str]): Source file URL when available.
        source_domain (Optional[str]): Hostname derived from the URL or
            source.
        lang (Optional[str]): Language label; default derived from
            extension.
        encoding (str): Encoding used to read the file.
        had_replacement (bool): Whether decoding used replacement
            characters.
        chunk_id (Optional[int]): Chunk index within the file (1-based).
        n_chunks (Optional[int]): Total chunk count for the file.
        extra_meta (Optional[Dict[str, object]]): Extra metadata fields.
        langcfg (LanguageConfig | None): Language/extension configuration.
        tokens (Optional[int]): Exact token count if already computed.
        meta (Optional[RecordMeta | Mapping[str, Any]]): Seed metadata to
            merge.
        file_bytes (Optional[int]): Byte size of the original file.
        truncated_bytes (Optional[int]): Bytes truncated from the source.
        file_nlines (Optional[int]): Line count of the original file.

    Returns:
        Dict[str, object]: Record with ``text`` and normalized ``meta`` payload.
    """
    rp = rel_path.replace("\\", "/")
    cfg = langcfg or DEFAULT_LANGCFG

    # Derive language / estimation kind from extension when not provided
    kind, lang_hint = guess_lang_from_path(rp, cfg=cfg)
    if not lang:
        if lang_hint:
            lang = cfg.display_names.get(lang_hint, lang_hint.capitalize())
        else:
            lang = "Text"

    # Compute byte length and token estimate (approximate by default)
    bcount = len(text.encode("utf-8", "strict"))
    if tokens is not None:
        approx_tokens = tokens
        token_value = tokens
    else:
        approx_tokens = count_tokens(text, None, "code" if kind == "code" else "doc")
        token_value = approx_tokens
    nlines = 0 if text == "" else text.count("\n") + 1

    chunk_id_val = int(chunk_id) if chunk_id is not None else 1
    n_chunks_val = int(n_chunks) if n_chunks is not None else 1

    source_url = repo_url or (f"https://github.com/{repo_full_name}" if repo_full_name else None)
    domain = source_domain
    if domain is None and url:
        try:
            domain = urlparse(url).hostname
        except Exception:
            domain = None

    risky_fallback_encodings = {"latin-1", "iso-8859-1", "cp1252"}
    if encoding and encoding.lower() in risky_fallback_encodings:
        if extra_meta is None:
            extra_meta = {}
        extra_meta["decoding_fallback_used"] = True

    seed: Mapping[str, Any] | None
    if isinstance(meta, RecordMeta):
        seed = meta.to_dict()
    else:
        seed = meta

    meta_obj = RecordMeta.from_seed(
        seed,
        kind=kind,
        source=source_url,
        repo=repo_full_name,
        path=rp,
        url=url,
        source_domain=domain,
        license=license_id,
        lang=lang,
        chunk_id=chunk_id_val,
        n_chunks=n_chunks_val,
        encoding=encoding,
        had_replacement=bool(had_replacement),
        sha256=sha256_text(text),
        approx_tokens=approx_tokens,
        tokens=token_value,
        bytes=bcount,
        file_bytes=file_bytes,
        truncated_bytes=truncated_bytes,
        nlines=nlines,
        file_nlines=file_nlines,
    )

    if extra_meta:
        record_fields = {f.name for f in fields(RecordMeta)}
        for key, value in extra_meta.items():
            if key in record_fields or value is None:
                continue
            meta_obj.extra.setdefault(key, value)

    record = {
        "text": text,
        "meta": meta_obj.to_dict(),
    }
    return record
# -----------------------
# Run header helper
# -----------------------

if TYPE_CHECKING:  # pragma: no cover
    from .config import SievioConfig


def build_run_header_record(config: SievioConfig) -> dict[str, Any]:
    """Build a run_header record describing configuration at run start."""

    meta = RunHeaderMeta(
        config=config.to_dict(),
        metadata=config.metadata.to_dict(),
    )
    return {"text": "", "meta": meta.to_dict()}


def best_effort_record_path(record: Mapping[str, Any]) -> str:
    """Return a human-friendly path or identifier for a record.

    Prefers ``meta["path"]`` then ``meta["doc_id"]``/``meta["chunk_id"]``,
    then ``record["path"]`` or ``record["origin_path"]``; otherwise
    returns ``"<unknown>"``.

    Args:
        record (Mapping[str, Any]): Record dictionary containing metadata.

    Returns:
        str: Best-effort identifier for logging or QC messages.
    """
    if not isinstance(record, Mapping):
        return "<unknown>"
    meta = record.get("meta")
    if isinstance(meta, Mapping):
        path_val = meta.get("path")
        if isinstance(path_val, str) and path_val:
            return path_val
        doc_id = meta.get("doc_id") or meta.get("chunk_id")
        if doc_id:
            return str(doc_id)
    path_val = record.get("path") or record.get("origin_path")
    if isinstance(path_val, str) and path_val:
        return path_val
    return "<unknown>"
