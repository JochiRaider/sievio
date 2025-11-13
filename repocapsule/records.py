# records.py
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple
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
]

# -----------------------
# Extension classifications
# -----------------------
# note: Keep these lower-cased; compare on Path.suffix.lower().
CODE_EXTS: set[str] = {
    # programming / scripting
    ".py", ".pyw", ".py3", ".ipynb",
    ".ps1", ".psm1", ".psd1", ".bat", ".cmd", ".sh", ".bash", ".zsh",
    ".c", ".h", ".cpp", ".hpp", ".cc", ".hh", ".cxx", ".hxx",
    ".cs", ".java", ".kt", ".kts", ".scala", ".go", ".rs", ".swift",
    ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
    ".rb", ".php", ".pl", ".pm",
    ".lua", ".r", ".jl",
    ".sql", ".sparql",
    # config / structured (treated as code-ish for token ratios)
    ".json", ".jsonc", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".xml", ".xslt", ".evtx",
    # data / rules
    ".yara", ".yar", ".sigma", ".ndjson", ".log",
}

DOC_EXTS: set[str] = {
    ".md", ".mdx", ".rst", ".adoc", ".txt",
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
    ".evtx":"windows-eventlog",
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
# Record assembly
# -----------------------

def build_record(
    *,
    text: str,
    rel_path: str,
    repo_full_name: Optional[str] = None,  # e.g., "owner/repo"
    repo_url: Optional[str] = None,        # e.g., "https://github.com/owner/repo"
    license_id: Optional[str] = None,      # SPDX id like 'Apache-2.0'
    lang: Optional[str] = None,            # language label (Title Case preferred)
    encoding: str = "utf-8",
    had_replacement: bool = False,
    chunk_id: Optional[int] = None,
    n_chunks: Optional[int] = None,
    extra_meta: Optional[Dict[str, object]] = None,
    langcfg: LanguageConfig | None = None,
    tokens: Optional[int] = None,
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
        "bytes": 5678
      }
    }
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
    approx_tokens = tokens if tokens is not None else count_tokens(text, None, "code" if kind == "code" else "doc")

    record = {
        "text": text,
        "meta": {
            "source": repo_url or (f"https://github.com/{repo_full_name}" if repo_full_name else None),
            "repo": repo_full_name,
            "path": rp,
            "license": license_id,
            "lang": lang,
            "chunk_id": int(chunk_id or 1),
            "n_chunks": int(n_chunks or 1),
            "encoding": encoding,
            "had_replacement": bool(had_replacement),
            "sha256": sha256_text(text),
            "tokens": approx_tokens,  # retained for backward compatibility (approximate)
            "approx_tokens": approx_tokens,
            "bytes": bcount,
        },
    }

    # Drop None fields for cleanliness
    record["meta"] = {k: v for k, v in record["meta"].items() if v is not None}

    # Extra metadata (non-conflicting only)
    if extra_meta:
        for k2, v2 in extra_meta.items():
            if k2 in record["meta"]:
                continue
            record["meta"][k2] = v2

    return record
