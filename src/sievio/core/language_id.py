# language_id.py
# SPDX-License-Identifier: MIT
"""Language identification utilities and configuration.

This module centralizes extension-based hints, lightweight heuristics,
and optional backends for both human and programming language
identification. It intentionally avoids importing higher-level modules
to keep dependencies minimal.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

# ScoreKind describes how to interpret the `score` field:
# - "probability": calibrated probability in [0.0, 1.0] (higher is better).
# - "confidence": backend-defined confidence (numeric range may vary; higher is better).
# - "distance": distance/metric where lower is better (e.g., cosine distance).
# - "heuristic": arbitrary heuristic score; only comparable within the same backend.
ScoreKind = Literal["probability", "confidence", "distance", "heuristic"]


@dataclass(slots=True)
class LanguagePrediction:
    """Human-language prediction result."""

    code: str
    score: float
    reliable: bool = False
    score_kind: ScoreKind = "probability"
    backend: str | None = None
    details: dict[str, Any] | None = None


@runtime_checkable
class LanguageDetector(Protocol):
    """Interface for human-language detectors."""

    def detect(self, text: str) -> LanguagePrediction | None:
        ...

    def detect_topk(self, text: str, k: int = 3) -> Sequence[LanguagePrediction]:
        ...


@dataclass(slots=True)
class CodeLanguagePrediction:
    """Prediction for a code- or doc-language label.

    `lang` is a content-type tag derived from file hints and/or detection
    (e.g., "python", "javascript", "markdown", "restructuredtext", "json").
    It mirrors the values used in EXT_LANG and is intended to populate meta["lang"].
    """

    lang: str
    score: float
    reliable: bool = False
    score_kind: ScoreKind = "heuristic"
    backend: str | None = None
    details: dict[str, Any] | None = None


@runtime_checkable
class CodeLanguageDetector(Protocol):
    """Interface for code-language detectors."""

    def detect_code(self, text: str, *, filename: str | None = None) -> CodeLanguagePrediction | None:
        ...

    def detect_topk(self, text: str, k: int = 3, *, filename: str | None = None) -> Sequence[CodeLanguagePrediction]:
        ...


# -----------------------
# Extension classifications
# -----------------------
# Keep suffixes lowercase and include the leading dot.
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

MD_EXTS: set[str] = {".md", ".mdx", ".markdown"}
RST_EXTS: set[str] = {".rst"}
GENERIC_DOC_EXTS: set[str] = {".adoc", ".txt"}
DOC_FORMAT_BY_EXT: dict[str, str] = {
    ".md": "md",
    ".mdx": "md",
    ".markdown": "md",
    ".rst": "rst",
    ".adoc": "asciidoc",
    ".txt": "text",
}
DOC_EXTS: set[str] = MD_EXTS | RST_EXTS | GENERIC_DOC_EXTS

# NOTE: Content-type tags here populate meta["lang"] (code or doc types).
# Language hints per extension (lower-case ext -> language tag)
EXT_LANG: dict[str, str] = {
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
    ".markdown": "markdown",
    ".rst": "restructuredtext",
    ".adoc": "asciidoc",
    ".txt": "text",
    ".evtx": "windows-eventlog",
}

DEFAULT_DISPLAY_NAMES: dict[str, str] = {
    "python": "Python",
    "powershell": "PowerShell",
    "batch": "Batch",
    "bash": "Bash",
    "zsh": "Zsh",
    "c": "C",
    "cpp": "C++",
    "csharp": "C#",
    "java": "Java",
    "kotlin": "Kotlin",
    "scala": "Scala",
    "go": "Go",
    "rust": "Rust",
    "swift": "Swift",
    "typescript": "TypeScript",
    "javascript": "JavaScript",
    "ruby": "Ruby",
    "php": "PHP",
    "perl": "Perl",
    "lua": "Lua",
    "r": "R",
    "julia": "Julia",
    "sql": "SQL",
    "sparql": "SPARQL",
    "json": "JSON",
    "yaml": "YAML",
    "toml": "TOML",
    "ini": "INI",
    "xml": "XML",
    "yara": "YARA",
    "sigma": "Sigma",
    "ndjson": "NDJSON",
    "log": "Log",
    "markdown": "Markdown",
    "restructuredtext": "reStructuredText",
    "asciidoc": "AsciiDoc",
    "text": "Text",
    "windows-eventlog": "Windows Event Log",
}


@dataclass(slots=True)
class LanguageConfig:
    """Configurable file-type and language hints."""

    code_exts: set[str] = field(default_factory=lambda: set(CODE_EXTS))
    doc_exts: set[str] = field(default_factory=lambda: set(DOC_EXTS))
    ext_lang: dict[str, str] = field(default_factory=lambda: dict(EXT_LANG))
    display_names: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_DISPLAY_NAMES))


DEFAULT_LANGCFG = LanguageConfig()


# -----------------------
# Basic classifiers / hints
# -----------------------

def guess_lang_from_path(path: str | Path, cfg: LanguageConfig | None = None) -> tuple[str, str]:
    """Return (kind, lang) for the given path."""
    cfg = cfg or DEFAULT_LANGCFG
    p = Path(path)
    ext = p.suffix.lower()
    name = p.name.lower()
    if name in SPECIAL_FILENAMES:
        return "code", SPECIAL_FILENAMES[name]
    if ext in cfg.code_exts:
        kind = "code"
    elif ext in cfg.doc_exts:
        kind = "doc"
    else:
        kind = "doc"

    if ext.startswith(".") and len(ext) > 1:
        fallback = ext[1:]
    else:
        fallback = "text"
    lang = cfg.ext_lang.get(ext, fallback)
    return kind, lang


def is_code_file(path: str | Path, cfg: LanguageConfig | None = None) -> bool:
    """Return True if the path extension is recognized as code."""
    cfg = cfg or DEFAULT_LANGCFG
    return Path(path).suffix.lower() in cfg.code_exts


def classify_path_kind(rel_path: str, *, cfg: LanguageConfig | None = None) -> tuple[str, str | None]:
    """Return (kind, lang_or_format) for a path.

    kind:
        "code" for source-like files, "doc" for documentation/text-like files.
    lang_or_format:
        - for code: content-type tag such as "python", "javascript", "json".
        - for docs: doc format hint such as "md" or "rst" when known.
        - unknown extensions return ("doc", None).
    """
    cfg = cfg or DEFAULT_LANGCFG
    ext = Path(rel_path).suffix.lower()
    if ext in cfg.doc_exts:
        return "doc", DOC_FORMAT_BY_EXT.get(ext)

    kind, lang = guess_lang_from_path(rel_path, cfg)
    if kind == "code":
        return kind, lang
    return "doc", DOC_FORMAT_BY_EXT.get(ext, None)


# -----------------------
# Human-language detectors
# -----------------------

class HeuristicLanguageDetector:
    """Cheap ASCII-heavy heuristic language detector."""

    def detect(self, text: str) -> LanguagePrediction | None:
        text = text or ""
        sample = text[:2048]
        letters = [ch for ch in sample if ch.isalpha()]
        if not letters:
            return None

        if all(ord(ch) < 128 for ch in letters):
            return LanguagePrediction(
                code="en",
                score=0.6,
                reliable=False,
                score_kind="heuristic",
                backend="baseline",
            )

        return None

    def detect_topk(self, text: str, k: int = 3) -> Sequence[LanguagePrediction]:
        pred = self.detect(text)
        return [pred] if pred else []


def make_language_detector(backend: str) -> LanguageDetector | None:
    """Factory for human-language detectors."""
    backend = (backend or "none").lower()
    if backend == "none":
        return None
    if backend == "baseline":
        return HeuristicLanguageDetector()
    if backend == "lingua":
        from .extras.langid_lingua import LinguaLanguageDetector

        return LinguaLanguageDetector()
    raise ValueError(f"Unknown language detector backend: {backend}")


# -----------------------
# Code-language detectors
# -----------------------

SPECIAL_FILENAMES: dict[str, str] = {
    "makefile": "makefile",
    "cmakelists.txt": "cmake",
    "dockerfile": "docker",
    "docker-compose.yml": "yaml",
    "docker-compose.yaml": "yaml",
    "requirements.txt": "python",
    "package.json": "json",
    "package-lock.json": "json",
    "poetry.lock": "python",
    "gemfile": "ruby",
}

SHEBANG_LANG_HINTS: dict[str, str] = {
    "python": "python",
    "python3": "python",
    "python2": "python",
    "bash": "bash",
    "sh": "bash",
    "zsh": "zsh",
    "node": "javascript",
    "nodejs": "javascript",
    "deno": "javascript",
    "ruby": "ruby",
    "perl": "perl",
    "php": "php",
    "rscript": "r",
    "lua": "lua",
}


def _shebang_hint(text: str) -> str | None:
    first_line = (text.splitlines() or [""])[0].strip()
    if not first_line.startswith("#!"):
        return None
    shebang = first_line[2:].strip()
    parts = shebang.split()
    if not parts:
        return None
    interpreter = Path(parts[0]).name
    if interpreter == "env" and len(parts) > 1:
        interpreter = parts[1]
    interpreter = interpreter.lower()
    return SHEBANG_LANG_HINTS.get(interpreter)


class BaselineCodeLanguageDetector:
    """Filename and shebang driven code-language detector."""

    def __init__(self, cfg: LanguageConfig | None = None):
        self.cfg = cfg or DEFAULT_LANGCFG

    def _lang_from_filename(self, filename: str) -> str | None:
        name = Path(filename).name.lower()
        if name in SPECIAL_FILENAMES:
            return SPECIAL_FILENAMES[name]
        ext = Path(filename).suffix.lower()
        return self.cfg.ext_lang.get(ext)

    def detect_code(self, text: str, *, filename: str | None = None) -> CodeLanguagePrediction | None:
        lang = None
        backend = "baseline"
        if filename:
            lang = self._lang_from_filename(filename)
        if lang is None:
            lang = _shebang_hint(text)
        if lang is None:
            return None
        return CodeLanguagePrediction(lang=lang, score=0.6, reliable=False, score_kind="heuristic", backend=backend)

    def detect_topk(self, text: str, k: int = 3, *, filename: str | None = None) -> Sequence[CodeLanguagePrediction]:
        pred = self.detect_code(text, filename=filename)
        return [pred] if pred else []


def make_code_language_detector(backend: str, cfg: LanguageConfig | None = None) -> CodeLanguageDetector | None:
    """Factory for code-language detectors."""
    backend = (backend or "none").lower()
    if backend == "none":
        return None
    if backend == "baseline":
        return BaselineCodeLanguageDetector(cfg)
    if backend == "pygments":
        from .extras.langid_pygments import PygmentsCodeLanguageDetector

        return PygmentsCodeLanguageDetector(cfg or DEFAULT_LANGCFG)
    raise ValueError(f"Unknown code language detector backend: {backend}")


__all__ = [
    "ScoreKind",
    "LanguagePrediction",
    "LanguageDetector",
    "CodeLanguagePrediction",
    "CodeLanguageDetector",
    "CODE_EXTS",
    "DOC_EXTS",
    "EXT_LANG",
    "MD_EXTS",
    "RST_EXTS",
    "GENERIC_DOC_EXTS",
    "DOC_FORMAT_BY_EXT",
    "LanguageConfig",
    "DEFAULT_LANGCFG",
    "DEFAULT_DISPLAY_NAMES",
    "guess_lang_from_path",
    "is_code_file",
    "classify_path_kind",
    "HeuristicLanguageDetector",
    "BaselineCodeLanguageDetector",
    "make_language_detector",
    "make_code_language_detector",
]
