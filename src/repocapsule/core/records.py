# records.py
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple, TYPE_CHECKING
import hashlib

from .chunk import count_tokens

__all__ = [
    "CODE_EXTS",
    "DOC_EXTS",
    "EXT_LANG",
    "LanguageConfig",
    "DEFAULT_LANGCFG",
    "guess_lang_from_path",
    "is_code_file",
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
]

# -----------------------
# Extension classifications
# -----------------------
# note: Keep these lower-cased; compare on Path.suffix.lower().
CODE_EXTS: set[str] = {
    # programming / scripting
    ".py",
    ".pyw",
    ".py3",
    ".ipynb",
    ".ps1",
    ".psm1",
    ".psd1",
    ".bat",
    ".cmd",
    ".sh",
    ".bash",
    ".zsh",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".cc",
    ".hh",
    ".cxx",
    ".hxx",
    ".cs",
    ".java",
    ".kt",
    ".kts",
    ".scala",
    ".go",
    ".rs",
    ".swift",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".rb",
    ".php",
    ".pl",
    ".pm",
    ".lua",
    ".r",
    ".jl",
    ".sql",
    ".sparql",
    # config / structured (treated as code-ish for token ratios)
    ".json",
    ".jsonc",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".xml",
    ".xslt",
    ".evtx",
    # data / rules
    ".yara",
    ".yar",
    ".sigma",
    ".ndjson",
    ".log",
}

DOC_EXTS: set[str] = {
    ".md",
    ".mdx",
    ".rst",
    ".adoc",
    ".txt",
}

# Language hints per extension (lower-case ext -> language tag)
EXT_LANG: Dict[str, str] = {
    ".py": "python",
    ".ipynb": "python",
    ".ps1": "powershell",
    ".psm1": "powershell",
    ".psd1": "powershell",
    ".bat": "batch",
    ".cmd": "batch",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "zsh",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cc": "cpp",
    ".hh": "cpp",
    ".cxx": "cpp",
    ".hxx": "cpp",
    ".cs": "csharp",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".go": "go",
    ".rs": "rust",
    ".swift": "swift",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".rb": "ruby",
    ".php": "php",
    ".pl": "perl",
    ".pm": "perl",
    ".lua": "lua",
    ".r": "r",
    ".jl": "julia",
    ".sql": "sql",
    ".sparql": "sparql",
    ".json": "json",
    ".jsonc": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
    ".xml": "xml",
    ".xslt": "xml",
    ".yara": "yara",
    ".yar": "yara",
    ".sigma": "sigma",
    ".ndjson": "ndjson",
    ".log": "log",
    ".md": "markdown",
    ".mdx": "markdown",
    ".rst": "restructuredtext",
    ".adoc": "asciidoc",
    ".txt": "text",
    ".evtx": "windows-eventlog",
}

@dataclass
class LanguageConfig:
    """Configurable file-type & language hints (defaults mirror module globals)."""
    code_exts: set[str] = field(default_factory=lambda: set(CODE_EXTS))
    doc_exts: set[str]  = field(default_factory=lambda: set(DOC_EXTS))
    ext_lang: Dict[str, str] = field(default_factory=lambda: dict(EXT_LANG))

DEFAULT_LANGCFG = LanguageConfig()


# -----------------------
# Basic classifiers / hints
# -----------------------

def guess_lang_from_path(path: str | Path, cfg: LanguageConfig | None = None) -> Tuple[str, str]:
    """Return (kind, lang) for the given path.

    kind  {"code", "doc"}
    lang is a coarse language tag from EXT_LANG, defaulting to extension or 'text'.
    """
    cfg = cfg or DEFAULT_LANGCFG
    p = Path(path)    
    ext = p.suffix.lower()
    if ext in cfg.code_exts:
        kind = "code"
    elif ext in cfg.doc_exts:
        kind = "doc"
    else:
        # Heuristic: treat unknowns as docs (safer for tokenization)
        kind = "doc"
    lang = cfg.ext_lang.get(ext, (ext[1:] if ext.startswith(".") and len(ext) > 1 else "text"))
    return kind, lang


def is_code_file(path: str | Path, cfg: LanguageConfig | None = None) -> bool:
    cfg = cfg or DEFAULT_LANGCFG
    return Path(path).suffix.lower() in cfg.code_exts


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

RECORD_META_SCHEMA_VERSION = "2"  # bump when core schema fields change so downstream consumers can detect mixes
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
#   - Content structure: ``bytes``, ``file_bytes``, ``truncated_bytes``, ``nlines``, ``file_nlines``.
#   - Language & sampling: ``lang``, ``lang_score``, ``perplexity``, ``ppl_bucket``.
#   - Chunking / processing: ``kind``, ``chunk_id``, ``n_chunks``, ``encoding``, ``had_replacement``.
#   - Integrity & versioning: ``sha256``, ``schema_version``.
#   - Token counts: ``tokens``, ``approx_tokens``.
# Any other analyzer metadata should flow through ``RecordMeta.extra``.
# QC scorers populate a few additional meta keys; documenting them here keeps the
# schema discoverable for downstream tooling.
QC_META_FIELDS: set[str] = {
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


def filter_qc_meta(qc_result: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Partition a raw QC scorer result into (canonical_qc, qc_signals).

    - canonical_qc → stable QC fields allowed at top-level ``record["meta"]`` (QC_META_FIELDS plus ``qc_score`` from ``score``).
    - qc_signals  → everything else, intended for ``meta["extra"]["qc_signals"]``.
    Tokens are skipped here so callers can handle them specially (e.g., populate approx/tokens).
    """
    canonical: Dict[str, Any] = {}
    qc_signals: Dict[str, Any] = {}

    for key, value in qc_result.items():
        if key == "tokens":
            continue
        if key == "score":
            if value is not None:
                canonical["qc_score"] = value
            continue
        if key in QC_META_FIELDS:
            canonical[key] = value
            continue
        qc_signals[key] = value
    return canonical, qc_signals


def _meta_to_dict(obj: Any) -> Dict[str, Any]:
    """
    Flatten dataclass fields (including core schema fields such as url/nlines/perplexity) plus extras,
    skipping None values and preserving the rule that core fields win over ``extra`` entries.
    """
    out: Dict[str, Any] = {}
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
    """
    Metadata for normal content records (code/docs/logs/etc).

    `kind` reflects the coarse file type: 'code' or 'doc'.
    `file_bytes` captures the original source size, while `bytes` reflects the
    processed chunk length (UTF-8). `truncated_bytes` represents bytes omitted
    from the source due to prefix/decoder caps and is repeated across chunks.
    """

    kind: str = "code"
    source: Optional[str] = None
    repo: Optional[str] = None
    path: Optional[str] = None
    url: Optional[str] = None
    source_domain: Optional[str] = None
    license: Optional[str] = None
    lang: Optional[str] = None
    lang_score: Optional[float] = None
    perplexity: Optional[float] = None
    ppl_bucket: Optional[str] = None
    chunk_id: int = 1
    n_chunks: int = 1
    encoding: str = "utf-8"
    had_replacement: bool = False
    sha256: Optional[str] = None
    approx_tokens: Optional[int] = None
    tokens: Optional[int] = None
    bytes: Optional[int] = None
    file_bytes: Optional[int] = None
    truncated_bytes: Optional[int] = None
    nlines: Optional[int] = None
    file_nlines: Optional[int] = None
    schema_version: str = RECORD_META_SCHEMA_VERSION
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _meta_to_dict(self)

    @classmethod
    def from_seed(
        cls,
        seed: Optional[Mapping[str, Any]] = None,
        *,
        kind: Optional[str] = None,
        **overrides: Any,
    ) -> "RecordMeta":
        seed = seed or {}
        data = dict(seed)
        if kind is not None:
            data["kind"] = kind
        data.update(overrides)
        field_names = {f.name for f in fields(cls)}
        extra: Dict[str, Any] = {}
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
    """Metadata footer for runs; expects config to come from RepocapsuleConfig.to_dict()."""
    kind: str = "run_summary"
    schema_version: str = SUMMARY_META_SCHEMA_VERSION
    config: Dict[str, Any] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)
    qc_summary: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _meta_to_dict(self)


@dataclass(slots=True)
class QCSummaryMeta:
    kind: str = "qc_summary"
    schema_version: str = SUMMARY_META_SCHEMA_VERSION
    summary: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _meta_to_dict(self)


@dataclass(slots=True)
class RunHeaderMeta:
    kind: str = "run_header"
    schema_version: str = SUMMARY_META_SCHEMA_VERSION
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _meta_to_dict(self)


def ensure_meta_dict(record: MutableMapping[str, Any]) -> Dict[str, Any]:
    """Return record['meta'] as a dict, creating an empty one if needed."""
    meta = record.get("meta")
    if not isinstance(meta, dict):
        meta = {}
        record["meta"] = meta
    return meta


def merge_meta_defaults(record: MutableMapping[str, Any], defaults: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Ensure a record has a meta dict and fill in default values without clobbering existing entries.
    """
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
    repo_full_name: Optional[str] = None,  # e.g., "owner/repo"
    repo_url: Optional[str] = None,        # e.g., "https://github.com/owner/repo"
    license_id: Optional[str] = None,      # SPDX id like 'Apache-2.0'
    url: Optional[str] = None,              # canonical file URL when available
    source_domain: Optional[str] = None,    # hostname derived from URL/source
    lang: Optional[str] = None,            # language label (Title Case preferred)
    encoding: str = "utf-8",
    had_replacement: bool = False,
    chunk_id: Optional[int] = None,
    n_chunks: Optional[int] = None,
    extra_meta: Optional[Dict[str, object]] = None,
    langcfg: LanguageConfig | None = None,
    tokens: Optional[int] = None,
    meta: Optional[RecordMeta | Mapping[str, Any]] = None,
    file_bytes: Optional[int] = None,
    truncated_bytes: Optional[int] = None,
    file_nlines: Optional[int] = None,
) -> Dict[str, object]:
    """Create a canonical JSONL record matching the requested schema.

    JSONL schema (one object per line):
    {
      "text": "<chunk>",
      "meta": {
        "source": "https://github.com/owner/name",
        "repo": "owner/name",
        "path": "sub/dir/file.py",
        "license": "Apache-2.0",
        "lang": "Python",
        "chunk_id": 1,
        "n_chunks": 3,
        "encoding": "utf-8",
        "had_replacement": false,
        "sha256": "....",
        "tokens": 1234,
        "bytes": 5678,
        "file_bytes": 10240,
        "truncated_bytes": 4600
      }
    }

    Standard metadata keys are cataloged in :data:`STANDARD_META_FIELDS`
    (core record fields) and :data:`QC_META_FIELDS` (quality-scoring enrichments).
    Additional analyzer-specific keys should be stored under ``meta["extra"]``.
    """
    rp = rel_path.replace("\\", "/")

    # Derive language / estimation kind from extension when not provided
    kind, lang_hint = guess_lang_from_path(rp, cfg=langcfg or DEFAULT_LANGCFG)
    if not lang:
        # Title-case common language tags for presentation
        lang = lang_hint.capitalize() if lang_hint else "text"
        # Better names for some
        overrides = {
            "ipynb": "Python",
            "ps1": "PowerShell",
            "psm1": "PowerShell",
            "psd1": "PowerShell",
            "js": "JavaScript",
            "ts": "TypeScript",
            "tsx": "TypeScript",
            "jsx": "JavaScript",
            "yml": "YAML",
            "md": "Markdown",
            "rst": "reStructuredText",
            "ndjson": "NDJSON",
            "json": "JSON",
            "xml": "XML",
            "ini": "INI",
            "toml": "TOML",
        }
        lang = overrides.get(Path(rp).suffix.lower().lstrip("."), lang)

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

    seed: Optional[Mapping[str, Any]]
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
    from .config import RepocapsuleConfig


def build_run_header_record(config: "RepocapsuleConfig") -> Dict[str, Any]:
    """
    Build a run_header record describing the configuration and metadata at start of a run.
    """

    meta = RunHeaderMeta(
        config=config.to_dict(),
        metadata=config.metadata.to_dict(),
    )
    return {"text": "", "meta": meta.to_dict()}


def best_effort_record_path(record: Mapping[str, Any]) -> str:
    """
    Return a human-friendly path/identifier for a record for logging/QC.

    Prefers meta['path'], then meta['doc_id']/meta['chunk_id'], then record['path'] or record['origin_path'];
    otherwise returns "<unknown>".
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
